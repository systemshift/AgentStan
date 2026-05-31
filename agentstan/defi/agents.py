"""
Agents and their behavior policies.

Behavior is selected by name (see ``POLICIES``) and parameterized via config —
never free code. Each policy observes the market and returns a list of
*intents* (plain dicts). The engine validates and applies them against the
shared pool, so an agent can never mutate protocol state directly.

This is also exactly the seam where M4's LLM behavioral agents plug in: an
LLM policy will return intents from the same constrained vocabulary, chosen
from a prompt instead of a rule. The deterministic mechanics never change.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .protocol import slippage_fraction


@dataclass
class Agent:
    id: int
    type: str
    policy: str
    params: Dict[str, Any] = field(default_factory=dict)
    # private off-protocol balances
    wallet_usdc: float = 0.0   # spare cash a borrower could repay with
    capital: float = 0.0       # a liquidator's deployable budget (depletes as used)


@dataclass
class MarketView:
    """Read-only snapshot handed to policies each step."""
    step: int
    oracle_price: float
    true_price: float
    utilization: float
    available_liquidity: float
    initial_price: float = 0.0   # scenario starting price, for drawdown triggers
    # debt-asset price (USD); 1.0 == debt is the numeraire
    debt_oracle_price: float = 1.0
    debt_true_price: float = 1.0
    # market depth for liquidation slippage (None == infinitely deep)
    collateral_depth_usd: Optional[float] = None
    debt_depth_usd: Optional[float] = None


# --- borrower policies ---------------------------------------------------

def _borrower_passive(agent, pool, view) -> List[Dict[str, Any]]:
    return []


def _borrower_prudent(agent, pool, view) -> List[Dict[str, Any]]:
    """Repays debt when health gets thin, if it has cash on hand."""
    acct = pool.account(agent.id)
    hf = acct.health_factor(view.oracle_price, pool.config.liquidation_threshold)
    floor = agent.params.get("repay_below_hf", 1.2)
    if hf < floor and agent.wallet_usdc > 0 and acct.debt > 0:
        amount = min(agent.wallet_usdc, acct.debt)
        return [{"action": "repay", "agent": agent.id, "amount": amount}]
    return []


# --- lender policies ------------------------------------------------------

def _lender_passive(agent, pool, view) -> List[Dict[str, Any]]:
    return []


def _lender_skittish(agent, pool, view) -> List[Dict[str, Any]]:
    """Panic-withdraws supply when utilization spikes (a bank-run impulse)."""
    trigger = agent.params.get("panic_utilization", 0.95)
    if view.utilization >= trigger:
        amount = agent.params.get("withdraw_amount", agent.wallet_usdc)
        return [{"action": "withdraw", "agent": agent.id, "amount": amount}]
    return []


def _lender_panic_on_drawdown(agent, pool, view) -> List[Dict[str, Any]]:
    """Pulls supply once the collateral price has fallen past a threshold from
    the scenario start. Powers liquidity-withdrawal and whale-panic scenarios:
    when supply leaves while funds are lent out, available liquidity collapses
    and remaining lenders can't exit (a run). ``withdraw_fraction`` of the
    lender's position is pulled (default all)."""
    if agent.wallet_usdc <= 0 or view.initial_price <= 0:
        return []
    drawdown = (view.initial_price - view.oracle_price) / view.initial_price
    trigger = agent.params.get("drawdown_trigger", 0.10)
    if drawdown >= trigger:
        frac = agent.params.get("withdraw_fraction", 1.0)
        return [{"action": "withdraw", "agent": agent.id,
                 "amount": agent.wallet_usdc * frac}]
    return []


# --- liquidator policies --------------------------------------------------

def _liquidator_greedy(agent, pool, view) -> List[Dict[str, Any]]:
    """Liquidates underwater positions, largest debt first, until its capital
    for this step is exhausted.

    Two real constraints, both of which drive bad debt:
      - Rationality: a liquidator pays in USDC and seizes collateral at the
        protocol's *oracle* price, but can only sell that collateral at the
        *true* market price. It skips a liquidation that would lose money. When
        the oracle is stale-high during a crash, liquidations are unprofitable,
        so liquidators sit out and underwater positions rot — the oracle-lag
        failure mode. Tune with ``min_profit_margin`` (fraction of repay).
      - Capital: a finite per-step budget. When underwater debt outruns it you
        get liquidation congestion, the second driver of bad debt.
    """
    if agent.capital <= 0 or view.oracle_price <= 0:
        return []
    cfg = pool.config
    margin = agent.params.get("min_profit_margin", 0.0)
    dp_oracle, dp_true = view.debt_oracle_price, view.debt_true_price
    underwater = [
        a for a in pool.accounts.values()
        if a.debt > 0 and a.health_factor(view.oracle_price, cfg.liquidation_threshold, dp_oracle) < 1.0
    ]
    underwater.sort(key=lambda a: a.debt * dp_oracle, reverse=True)  # largest USD debt first

    intents = []
    budget = agent.capital  # USD
    for acct in underwater:
        if budget <= 0:
            break
        cap_units = (budget / dp_true) if dp_true > 0 else acct.debt
        repay = min(acct.debt * cfg.close_factor, cap_units)  # debt-asset units
        if repay <= 0:
            continue
        # Collateral seized at the (possibly stale) oracle prices...
        seized = min(repay * dp_oracle * (1.0 + cfg.liquidation_bonus) / view.oracle_price,
                     acct.collateral)
        # ...but the liquidator must SOURCE the debt asset to repay (buy leg) and
        # SELL the seized collateral (sell leg), both at the true price and both
        # paying slippage into thin markets. Thin depth on the asset they must
        # trade is what makes large liquidations unprofitable (the CRV/Aave case).
        buy_size = repay * dp_true
        sell_size = seized * view.true_price
        cost = buy_size * (1.0 + slippage_fraction(buy_size, view.debt_depth_usd))
        proceeds = sell_size * (1.0 - slippage_fraction(sell_size, view.collateral_depth_usd))
        if proceeds - cost < margin * cost:
            continue  # unprofitable: rational liquidator stays out
        intents.append({"action": "liquidate", "agent": agent.id,
                        "target": acct.agent_id, "amount": repay})
        budget -= cost
    return intents


POLICIES: Dict[str, Dict[str, Callable]] = {
    "borrower": {"passive": _borrower_passive, "prudent": _borrower_prudent},
    "lender": {"passive": _lender_passive, "skittish": _lender_skittish,
               "panic_on_drawdown": _lender_panic_on_drawdown},
    "liquidator": {"greedy": _liquidator_greedy},
}


def resolve_policy(agent_type: str, policy: str) -> Callable:
    by_type = POLICIES.get(agent_type)
    if not by_type:
        raise ValueError(f"unknown agent type: {agent_type!r}")
    fn = by_type.get(policy)
    if fn is None:
        raise ValueError(
            f"unknown policy {policy!r} for {agent_type!r}; "
            f"available: {sorted(by_type)}"
        )
    return fn
