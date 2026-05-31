"""
Declarative configuration for the DeFi lending market engine.

Everything that defines a simulation lives here as plain data — no code.
That is deliberate: this config object is the single contract shared by the
engine, the parameter sweeper, the (future) webapp editor, and the LLM that
authors setups from natural language. It serializes cleanly to/from JSON so an
LLM can emit one and a UI can render one.

Agent behavior is selected by *name* (a built-in policy) and parameterized —
never free Python. That keeps configs safe to author, store, and run in a
multi-tenant cloud.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class RateModel:
    """Kinked ("jump-rate") interest-rate curve, Aave/Compound style.

    Rates are annualized. Below ``optimal_utilization`` the borrow rate rises
    gently along ``slope1``; above it, ``slope2`` makes borrowing expensive fast
    to defend liquidity.
    """
    base_rate: float = 0.0
    slope1: float = 0.04
    slope2: float = 0.75
    optimal_utilization: float = 0.80

    def borrow_rate(self, utilization: float) -> float:
        u_opt = self.optimal_utilization
        if utilization <= u_opt:
            return self.base_rate + self.slope1 * (utilization / u_opt if u_opt else 0.0)
        excess = (utilization - u_opt) / (1.0 - u_opt) if u_opt < 1.0 else 0.0
        return self.base_rate + self.slope1 + self.slope2 * excess


@dataclass
class ProtocolConfig:
    """The protocol parameters a team actually tunes before launch/governance."""
    max_ltv: float = 0.80                # max borrow value / collateral value at origination
    liquidation_threshold: float = 0.85  # health-factor boundary (must be > max_ltv)
    liquidation_bonus: float = 0.05      # discount the liquidator earns on seized collateral
    close_factor: float = 0.50           # max fraction of a position's debt repaid per liquidation
    reserve_factor: float = 0.10         # share of interest routed to reserves
    initial_reserves: float = 0.0        # insurance/reserve buffer (USDC) available to absorb bad debt
    rate_model: RateModel = field(default_factory=RateModel)


@dataclass
class OracleConfig:
    """Price oracle behavior. ``lag_steps`` is the single most important stress
    lever: a delayed feed makes the protocol act on stale prices during a crash."""
    lag_steps: int = 0


@dataclass
class ScenarioConfig:
    """The market shock. Either supply an explicit ``price_path`` (USDC per unit
    of collateral, one entry per step) or describe a crash to be generated."""
    name: str = "custom"
    initial_price: float = 3000.0
    price_path: Optional[List[float]] = None
    crash_pct: float = 0.0       # total fractional drop, e.g. 0.40 for -40%
    crash_start: int = 10        # step the crash begins
    crash_duration: int = 20     # steps over which it unfolds (>0 => gradual)


@dataclass
class AgentPopulation:
    """A group of identical agents. ``policy`` names a built-in behavior; ``params``
    tune it. No code is ever stored here."""
    type: str                    # "borrower" | "lender" | "liquidator"
    count: int
    policy: str = "passive"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketConfig:
    """Top-level config: the complete, serializable description of one run."""
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    populations: List[AgentPopulation] = field(default_factory=list)
    steps_per_year: int = 8760   # accrual granularity (8760 = hourly steps)
    seed: int = 0

    # --- serialization: the LLM/webapp contract ---------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketConfig":
        d = dict(d)
        protocol = dict(d.get("protocol", {}))
        if "rate_model" in protocol:
            protocol["rate_model"] = RateModel(**protocol["rate_model"])
        return cls(
            protocol=ProtocolConfig(**protocol),
            oracle=OracleConfig(**d.get("oracle", {})),
            scenario=ScenarioConfig(**d.get("scenario", {})),
            populations=[AgentPopulation(**p) for p in d.get("populations", [])],
            steps_per_year=d.get("steps_per_year", 8760),
            seed=d.get("seed", 0),
        )

    def validate(self) -> List[str]:
        """Return a list of human-readable problems (empty == valid)."""
        issues = []
        p = self.protocol
        if not (0 < p.max_ltv < 1):
            issues.append(f"max_ltv must be in (0,1), got {p.max_ltv}")
        if not (0 < p.liquidation_threshold <= 1):
            issues.append(f"liquidation_threshold must be in (0,1], got {p.liquidation_threshold}")
        if p.liquidation_threshold <= p.max_ltv:
            issues.append(
                f"liquidation_threshold ({p.liquidation_threshold}) must exceed "
                f"max_ltv ({p.max_ltv}) or positions start liquidatable"
            )
        if not (0 < p.close_factor <= 1):
            issues.append(f"close_factor must be in (0,1], got {p.close_factor}")
        if self.oracle.lag_steps < 0:
            issues.append("oracle.lag_steps cannot be negative")
        if not self.populations:
            issues.append("no agent populations defined")
        return issues
