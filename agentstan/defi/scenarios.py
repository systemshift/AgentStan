"""
Stress-scenario presets.

Each preset is a transformation of a ``MarketConfig`` — it sets the shock (price
path) and, where the scenario is behavioral, adjusts agent populations
(panicking lenders, a whale, thin liquidators). They preserve the caller's
protocol parameters so a team can run any scenario against *their* settings.

    from agentstan.defi import scenarios, LendingMarket
    cfg = scenarios.stablecoin_depeg()           # uses a sensible default market
    cfg = scenarios.eth_crash(my_config, severity=0.55)   # against my params
    results = LendingMarket(cfg).run(120)

Or by name: ``scenarios.apply("whale_panic", my_config)``.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

from .config import MarketConfig


def default_market() -> MarketConfig:
    """A reasonable baseline lending market to apply scenarios to."""
    return MarketConfig.from_dict({
        "seed": 1,
        "protocol": {
            "max_ltv": 0.75, "liquidation_threshold": 0.85,
            "liquidation_bonus": 0.08, "close_factor": 0.5,
            "reserve_factor": 0.10, "initial_reserves": 10_000.0,
        },
        "oracle": {"lag_steps": 6},
        "scenario": {"name": "baseline", "initial_price": 3000.0, "crash_pct": 0.0},
        "populations": [
            {"type": "lender", "count": 1, "policy": "passive",
             "params": {"supply": 150_000.0}},
            {"type": "borrower", "count": 40, "policy": "passive",
             "params": {"collateral": 1.0, "borrow_ltv_fraction_range": [0.85, 1.0]}},
            {"type": "liquidator", "count": 1, "policy": "greedy",
             "params": {"capital": 60_000.0}},
        ],
    })


# --- helpers -------------------------------------------------------------

def _base_dict(base: Optional[MarketConfig]) -> Dict[str, Any]:
    return (base or default_market()).to_dict()


def _for_type(d: Dict[str, Any], agent_type: str) -> List[Dict[str, Any]]:
    return [p for p in d["populations"] if p["type"] == agent_type]


def _piecewise_path(keyframes: List[Tuple[int, float]], total_steps: int) -> List[float]:
    """Linear interpolation between (step, price) keyframes; holds the ends."""
    kf = sorted(keyframes)
    path = []
    for t in range(total_steps + 1):
        if t <= kf[0][0]:
            path.append(kf[0][1])
        elif t >= kf[-1][0]:
            path.append(kf[-1][1])
        else:
            for (s0, p0), (s1, p1) in zip(kf, kf[1:]):
                if s0 <= t <= s1:
                    frac = (t - s0) / (s1 - s0) if s1 > s0 else 0.0
                    path.append(p0 + (p1 - p0) * frac)
                    break
    return path


# --- presets -------------------------------------------------------------

def eth_crash(base: Optional[MarketConfig] = None, *, severity: float = 0.40,
              start: int = 10, duration: int = 30,
              oracle_lag: Optional[int] = None) -> MarketConfig:
    """A gradual collateral-price crash of ``severity`` (e.g. 0.40 = -40%)."""
    d = _base_dict(base)
    d["scenario"] = {
        "name": f"ETH crash -{int(severity * 100)}%",
        "initial_price": d["scenario"].get("initial_price", 3000.0),
        "price_path": None, "crash_pct": severity,
        "crash_start": start, "crash_duration": duration,
    }
    if oracle_lag is not None:
        d["oracle"]["lag_steps"] = oracle_lag
    return MarketConfig.from_dict(d)


def stablecoin_depeg(base: Optional[MarketConfig] = None, *, depeg_to: float = 0.80,
                     recover_to: float = 0.97, start: int = 10, drop_steps: int = 4,
                     recover_steps: int = 50, total_steps: int = 120) -> MarketConfig:
    """Collateral is a stablecoin that breaks peg ($1 -> ``depeg_to``) sharply,
    then partially recovers. The sharp drop plus oracle lag is what bites."""
    d = _base_dict(base)
    path = _piecewise_path([
        (0, 1.0), (start, 1.0),
        (start + drop_steps, depeg_to),
        (start + drop_steps + recover_steps, recover_to),
    ], total_steps)
    d["scenario"] = {"name": f"Stablecoin depeg to ${depeg_to:.2f}",
                     "initial_price": 1.0, "price_path": path, "crash_pct": 0.0}
    return MarketConfig.from_dict(d)


def liquidity_withdrawal(base: Optional[MarketConfig] = None, *,
                         drawdown_trigger: float = 0.10, severity: float = 0.18,
                         start: int = 10, duration: int = 20) -> MarketConfig:
    """A bank run: lenders pull supply once price drops past a threshold, so
    available liquidity collapses and remaining lenders can't exit."""
    d = _base_dict(base)
    d["scenario"] = {"name": "Liquidity withdrawal", "price_path": None,
                     "initial_price": d["scenario"].get("initial_price", 3000.0),
                     "crash_pct": severity, "crash_start": start, "crash_duration": duration}
    for pop in _for_type(d, "lender"):
        pop["policy"] = "panic_on_drawdown"
        pop["params"]["drawdown_trigger"] = drawdown_trigger
        pop["params"].setdefault("withdraw_fraction", 1.0)
    return MarketConfig.from_dict(d)


def whale_panic(base: Optional[MarketConfig] = None, *, whale_supply: float = 120_000.0,
                other_supply: float = 30_000.0, drawdown_trigger: float = 0.06,
                severity: float = 0.15, start: int = 10, duration: int = 15) -> MarketConfig:
    """A single dominant lender (whale) provides most of the liquidity, then
    yanks it early on a modest dip — freezing the market for everyone else. The
    lesson is a liquidity crisis without insolvency: watch min available
    liquidity, not bad debt."""
    d = _base_dict(base)
    d["scenario"] = {"name": "Whale liquidity panic", "price_path": None,
                     "initial_price": d["scenario"].get("initial_price", 3000.0),
                     "crash_pct": severity, "crash_start": start, "crash_duration": duration}
    # Replace lenders so the whale genuinely dominates available liquidity.
    d["populations"] = [p for p in d["populations"] if p["type"] != "lender"]
    d["populations"].insert(0, {"type": "lender", "count": 1, "policy": "passive",
                                "params": {"supply": other_supply}})
    d["populations"].append({
        "type": "lender", "count": 1, "policy": "panic_on_drawdown",
        "params": {"supply": whale_supply, "drawdown_trigger": drawdown_trigger,
                   "withdraw_fraction": 1.0},
    })
    return MarketConfig.from_dict(d)


def liquidation_congestion(base: Optional[MarketConfig] = None, *,
                           severity: float = 0.40, liquidator_capital: float = 10_000.0,
                           start: int = 10, duration: int = 12) -> MarketConfig:
    """A fast, deep crash that overwhelms under-capitalized liquidators — bad
    debt from congestion rather than from oracle lag."""
    d = _base_dict(base)
    d["oracle"]["lag_steps"] = 1  # fast oracle, so congestion is the cause
    d["scenario"] = {"name": "Liquidation congestion", "price_path": None,
                     "initial_price": d["scenario"].get("initial_price", 3000.0),
                     "crash_pct": severity, "crash_start": start, "crash_duration": duration}
    for pop in _for_type(d, "liquidator"):
        pop["params"]["capital"] = liquidator_capital
    return MarketConfig.from_dict(d)


def oracle_delay(base: Optional[MarketConfig] = None, *, lag: int = 24,
                 severity: float = 0.40, start: int = 10, duration: int = 30) -> MarketConfig:
    """A crash where the protocol's price feed lags badly, so liquidations fire
    too late (or never) to protect the protocol."""
    d = _base_dict(base)
    d["oracle"]["lag_steps"] = lag
    d["scenario"] = {"name": f"Oracle delay ({lag} steps)", "price_path": None,
                     "initial_price": d["scenario"].get("initial_price", 3000.0),
                     "crash_pct": severity, "crash_start": start, "crash_duration": duration}
    return MarketConfig.from_dict(d)


SCENARIOS = {
    "eth_crash": eth_crash,
    "stablecoin_depeg": stablecoin_depeg,
    "liquidity_withdrawal": liquidity_withdrawal,
    "whale_panic": whale_panic,
    "liquidation_congestion": liquidation_congestion,
    "oracle_delay": oracle_delay,
}


def apply(name: str, base: Optional[MarketConfig] = None, **params) -> MarketConfig:
    """Build a scenario by name. ``params`` are passed to the preset builder."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; available: {sorted(SCENARIOS)}")
    return SCENARIOS[name](base, **params)
