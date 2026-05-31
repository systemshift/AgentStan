"""
LendingMarket: the shared-state simulation loop for a DeFi lending protocol.

Unlike the ecology engine, there is no space and no per-agent neighbor search.
Each step runs a deterministic pipeline over one shared pool:

    advance price -> read oracle -> accrue interest -> agents act -> record

Agents act in a fixed stage order (borrowers, lenders, liquidators) so runs are
reproducible. The whole thing is driven by a declarative ``MarketConfig``.
"""

import random
from typing import Any, Dict, List

from .config import MarketConfig
from .protocol import LendingPool
from .oracle import Oracle, build_price_path
from .agents import Agent, MarketView, resolve_policy


# agents act in this order each step; liquidators last so they react to the
# post-accrual, post-borrower-action state.
_STAGE_ORDER = ["borrower", "lender", "liquidator"]


class LendingMarket:
    def __init__(self, config: MarketConfig):
        issues = config.validate()
        if issues:
            raise ValueError("invalid MarketConfig:\n  - " + "\n  - ".join(issues))
        self.config = config
        self.rng = random.Random(config.seed)
        self.pool = LendingPool(config.protocol)
        self.agents: List[Agent] = []
        self.history: List[Dict[str, Any]] = []
        self.step = 0
        self.oracle: Oracle = None  # built in run()
        self._build_agents()

    # --- setup -----------------------------------------------------------
    def _build_agents(self) -> None:
        next_id = 0
        for pop in self.config.populations:
            resolve_policy(pop.type, pop.policy)  # fail fast on bad policy names
            for _ in range(pop.count):
                self.agents.append(Agent(
                    id=next_id, type=pop.type, policy=pop.policy,
                    params=dict(pop.params),
                    wallet_usdc=float(pop.params.get("wallet_usdc", 0.0)),
                    capital=float(pop.params.get("capital", 0.0)),
                ))
                next_id += 1

    def _seed_positions(self, price: float) -> None:
        """Open the initial positions at t0. Lenders must supply before borrowers
        draw, so seeding runs lenders first regardless of the per-step stage order."""
        for agent in self.agents:
            if agent.type != "lender":
                continue
            supply = float(agent.params.get("supply", 0.0))
            self.pool.supply(supply)
            agent.wallet_usdc = supply  # remember what they can pull back

        for agent in self.agents:
            if agent.type != "borrower":
                continue
            p = agent.params
            collateral = float(p.get("collateral", 0.0))
            self.pool.deposit_collateral(agent.id, collateral)
            frac = float(p.get("borrow_ltv_fraction", 1.0))
            target = collateral * price * self.config.protocol.max_ltv * frac
            self.pool.borrow(agent.id, target, price)

    # --- run -------------------------------------------------------------
    def run(self, steps: int) -> Dict[str, Any]:
        prices = build_price_path(self.config.scenario, steps)
        self.oracle = Oracle(prices, self.config.oracle.lag_steps)

        self._seed_positions(self.oracle.reported_price(0))
        self._record(0)

        for t in range(1, steps + 1):
            self.step = t
            self._step(t)
            self._record(t)

        return self._results(steps)

    def _step(self, t: int) -> None:
        self.pool.reset_step_tallies()
        oracle_price = self.oracle.reported_price(t)
        true_price = self.oracle.true_price(t)

        self.pool.accrue_interest(self.config.steps_per_year)

        view = MarketView(
            step=t, oracle_price=oracle_price, true_price=true_price,
            utilization=self.pool.utilization,
            available_liquidity=self.pool.available_liquidity,
        )

        for stage in _STAGE_ORDER:
            for agent in self.agents:
                if agent.type != stage:
                    continue
                policy = resolve_policy(agent.type, agent.policy)
                for intent in policy(agent, self.pool, view):
                    self._apply(agent, intent, oracle_price, true_price)

    def _apply(self, agent: Agent, intent: Dict[str, Any],
               oracle_price: float, true_price: float) -> None:
        action = intent.get("action")
        if action == "repay":
            amount = min(intent.get("amount", 0.0), agent.wallet_usdc)
            paid = self.pool.repay(agent.id, amount)
            agent.wallet_usdc -= paid
        elif action == "withdraw":
            got = self.pool.withdraw_supply(intent.get("amount", 0.0))
            agent.wallet_usdc -= got  # cash leaves the protocol
        elif action == "liquidate":
            target = self.pool.accounts.get(intent.get("target"))
            if target is None:
                return
            spent = self.pool.liquidate(
                target, intent.get("amount", 0.0), oracle_price, true_price)
            agent.capital -= spent
        # unknown actions are ignored (validated vocabulary keeps this rare)

    # --- metrics ---------------------------------------------------------
    def _underwater_count(self, price: float) -> int:
        thr = self.config.protocol.liquidation_threshold
        return sum(
            1 for a in self.pool.accounts.values()
            if a.debt > 0 and a.health_factor(price, thr) < 1.0
        )

    def _outstanding_shortfall(self, true_price: float) -> float:
        """Latent bad debt: for every open position, how much debt exceeds the
        collateral's true value. This is the hole the protocol is carrying even
        before a liquidation realizes it — congested/un-liquidated positions
        live here."""
        return sum(
            max(0.0, a.debt - a.collateral * true_price)
            for a in self.pool.accounts.values() if a.debt > 0
        )

    def _record(self, t: int) -> None:
        pool = self.pool
        true_price = self.oracle.true_price(t) if self.oracle else self.config.scenario.initial_price
        oracle_price = self.oracle.reported_price(t) if self.oracle else true_price
        self.history.append({
            "step": t,
            "true_price": true_price,
            "oracle_price": oracle_price,
            "utilization": pool.utilization,
            "available_liquidity": pool.available_liquidity,
            "total_borrowed": pool.total_borrowed,
            "total_supplied": pool.total_supplied,
            "reserves": pool.reserves,
            "realized_bad_debt": pool.bad_debt,
            "outstanding_shortfall": self._outstanding_shortfall(true_price),
            "step_bad_debt": pool.step_bad_debt,
            "step_liquidation_volume": pool.step_liquidation_volume,
            "step_liquidations": pool.step_liquidations,
            "underwater_accounts": self._underwater_count(true_price),
        })

    def _results(self, steps: int) -> Dict[str, Any]:
        total_liq_volume = sum(h["step_liquidation_volume"] for h in self.history)
        total_liquidations = sum(h["step_liquidations"] for h in self.history)
        min_liquidity = min(h["available_liquidity"] for h in self.history)
        max_util = max(h["utilization"] for h in self.history)
        peak_underwater = max(h["underwater_accounts"] for h in self.history)
        final_true_price = self.oracle.true_price(steps)
        outstanding = self._outstanding_shortfall(final_true_price)
        total_bad_debt = self.pool.bad_debt + outstanding
        insurance_used = max(0.0, self.config.protocol.initial_reserves - self.pool.reserves)
        return {
            "config": self.config.to_dict(),
            "steps": steps,
            "history": self.history,
            "summary": {
                "bad_debt": total_bad_debt,
                "realized_bad_debt": self.pool.bad_debt,
                "outstanding_shortfall": outstanding,
                "insolvent": total_bad_debt > 1e-6,
                "reserves_remaining": self.pool.reserves,
                "insurance_fund_used": insurance_used,
                "peak_underwater_accounts": peak_underwater,
                "total_liquidation_volume": total_liq_volume,
                "total_liquidations": total_liquidations,
                "min_available_liquidity": min_liquidity,
                "max_utilization": max_util,
                "final_total_borrowed": self.pool.total_borrowed,
                "final_total_supplied": self.pool.total_supplied,
            },
        }
