"""
Protocol state and deterministic lending mechanics.

This is the classical, non-LLM core: interest accrual, health factors, and
liquidation math. It is a single shared state machine that every agent reads
and mutates through a small set of actions — there is no space, no neighbors.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from .config import ProtocolConfig


@dataclass
class Account:
    """One borrower position: collateral (in collateral units, e.g. ETH) and
    debt (in borrow units, e.g. USDC)."""
    agent_id: int
    collateral: float = 0.0
    debt: float = 0.0

    def health_factor(self, price: float, liquidation_threshold: float) -> float:
        """HF = (collateral value x threshold) / debt. < 1 means liquidatable.
        A debt-free account is infinitely healthy."""
        if self.debt <= 0:
            return float("inf")
        return (self.collateral * price * liquidation_threshold) / self.debt


class LendingPool:
    """Shared pool state plus the deterministic mechanics that act on it."""

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.total_supplied = 0.0   # USDC supplied by lenders (grows with interest)
        self.total_borrowed = 0.0   # USDC currently borrowed (grows with interest)
        self.reserves = config.initial_reserves
        self.bad_debt = 0.0         # realized, unrecoverable debt
        self.accounts: Dict[int, Account] = {}

        # running tallies for metrics (reset each step by the engine)
        self.step_liquidation_volume = 0.0
        self.step_liquidations = 0
        self.step_bad_debt = 0.0

    # --- views -----------------------------------------------------------
    @property
    def available_liquidity(self) -> float:
        """USDC that can still be borrowed or withdrawn (cash in the pool)."""
        return max(0.0, self.total_supplied - self.total_borrowed)

    @property
    def utilization(self) -> float:
        if self.total_supplied <= 0:
            return 0.0
        return min(1.0, self.total_borrowed / self.total_supplied)

    def account(self, agent_id: int) -> Account:
        acct = self.accounts.get(agent_id)
        if acct is None:
            acct = Account(agent_id=agent_id)
            self.accounts[agent_id] = acct
        return acct

    # --- interest --------------------------------------------------------
    def accrue_interest(self, steps_per_year: int) -> None:
        """Accrue one step of interest across all debt. Interest grows what
        borrowers owe; the reserve_factor share goes to reserves and the rest
        accrues to suppliers (total_supplied)."""
        if self.total_borrowed <= 0 or steps_per_year <= 0:
            return
        annual = self.config.rate_model.borrow_rate(self.utilization)
        rate_step = annual / steps_per_year
        if rate_step <= 0:
            return
        total_interest = 0.0
        for acct in self.accounts.values():
            if acct.debt > 0:
                interest = acct.debt * rate_step
                acct.debt += interest
                total_interest += interest
        self.total_borrowed += total_interest
        reserve_cut = total_interest * self.config.reserve_factor
        self.reserves += reserve_cut
        self.total_supplied += (total_interest - reserve_cut)

    # --- actions (mutate shared state) -----------------------------------
    def supply(self, amount: float) -> None:
        if amount > 0:
            self.total_supplied += amount

    def withdraw_supply(self, amount: float) -> float:
        """Withdraw up to ``amount``, capped by available liquidity. Returns the
        amount actually withdrawn (a shortfall == liquidity crunch)."""
        actual = max(0.0, min(amount, self.available_liquidity))
        self.total_supplied -= actual
        return actual

    def deposit_collateral(self, agent_id: int, amount: float) -> None:
        if amount > 0:
            self.account(agent_id).collateral += amount

    def borrow(self, agent_id: int, amount: float, price: float) -> float:
        """Borrow against collateral, respecting max_ltv and available liquidity.
        Returns the amount actually borrowed."""
        acct = self.account(agent_id)
        max_debt = acct.collateral * price * self.config.max_ltv
        room = max(0.0, max_debt - acct.debt)
        actual = max(0.0, min(amount, room, self.available_liquidity))
        acct.debt += actual
        self.total_borrowed += actual
        return actual

    def repay(self, agent_id: int, amount: float) -> float:
        acct = self.account(agent_id)
        actual = max(0.0, min(amount, acct.debt))
        acct.debt -= actual
        self.total_borrowed -= actual
        self.total_supplied += 0.0  # principal returns to the pool's cash
        return actual

    def liquidate(self, acct: Account, repay_amount: float, oracle_price: float,
                  true_price: float) -> float:
        """Liquidator repays ``repay_amount`` of debt and seizes collateral worth
        repay x (1 + bonus) at the *oracle* price. Returns USDC the liquidator
        actually spent.

        Bad debt is recognized when seizing the collateral the discount entitles
        them to would exceed what the borrower has — i.e. the position is so
        underwater that even a full liquidation cannot make the pool whole. We
        value the realized loss at the *true* price, since that is the economic
        reality regardless of what the lagging oracle reports.
        """
        cfg = self.config
        repay_amount = max(0.0, min(repay_amount, acct.debt * cfg.close_factor, acct.debt))
        if repay_amount <= 0 or oracle_price <= 0:
            return 0.0

        seize_value = repay_amount * (1.0 + cfg.liquidation_bonus)
        seize_collateral = seize_value / oracle_price

        if seize_collateral >= acct.collateral:
            # Not enough collateral to honor the liquidation in full.
            seize_collateral = acct.collateral
            # Debt cleared is limited by what the seized collateral is truly worth.
            recoverable = (seize_collateral * true_price) / (1.0 + cfg.liquidation_bonus)
            cleared = min(acct.debt, recoverable)
            shortfall = acct.debt - cleared
            acct.collateral = 0.0
            acct.debt = 0.0
            self.total_borrowed -= (cleared + shortfall)
            self._absorb_bad_debt(shortfall)
            self.step_liquidation_volume += cleared
            self.step_liquidations += 1
            return cleared

        acct.collateral -= seize_collateral
        acct.debt -= repay_amount
        self.total_borrowed -= repay_amount
        self.step_liquidation_volume += repay_amount
        self.step_liquidations += 1
        return repay_amount

    def _absorb_bad_debt(self, shortfall: float) -> None:
        if shortfall <= 0:
            return
        self.bad_debt += shortfall
        self.step_bad_debt += shortfall
        # Reserves/insurance fund absorb what they can; the rest is a supplier loss.
        covered = min(self.reserves, shortfall)
        self.reserves -= covered
        self.total_supplied -= (shortfall - covered)

    def reset_step_tallies(self) -> None:
        self.step_liquidation_volume = 0.0
        self.step_liquidations = 0
        self.step_bad_debt = 0.0
