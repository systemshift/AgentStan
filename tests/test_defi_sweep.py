"""Tests for DeFi parameter sweeps."""

import pytest

from agentstan.defi import MarketConfig, sweep, frontier, grid
from agentstan.defi.sweep import _set_nested


def _stress_spec():
    return {
        "seed": 1,
        "protocol": {
            "max_ltv": 0.70, "liquidation_threshold": 0.90,
            "liquidation_bonus": 0.08, "close_factor": 0.5,
            "reserve_factor": 0.10, "initial_reserves": 10_000.0,
        },
        "oracle": {"lag_steps": 12},
        "scenario": {"initial_price": 3000.0, "crash_pct": 0.40,
                     "crash_start": 10, "crash_duration": 30},
        "populations": [
            {"type": "lender", "count": 1, "policy": "passive",
             "params": {"supply": 150_000.0}},
            {"type": "borrower", "count": 40, "policy": "passive",
             "params": {"collateral": 1.0, "borrow_ltv_fraction": 1.0}},
            {"type": "liquidator", "count": 1, "policy": "greedy",
             "params": {"capital": 30_000.0}},
        ],
    }


def test_set_nested_handles_list_indices():
    d = {"populations": [{"params": {"capital": 1}}, {"params": {"capital": 2}}]}
    out = _set_nested(d, "populations.1.params.capital", 99)
    assert out["populations"][1]["params"]["capital"] == 99
    assert d["populations"][1]["params"]["capital"] == 2  # original untouched


def test_sweep_keys_by_value():
    base = MarketConfig.from_dict(_stress_spec())
    result = sweep(base, "protocol.max_ltv", [0.55, 0.70, 0.85], steps=120)
    assert set(result) == {0.55, 0.70, 0.85}
    assert all("bad_debt" in s for s in result.values())


def test_ltv_frontier_is_monotonic():
    """Higher LTV must never reduce bad debt under a fixed crash."""
    base = MarketConfig.from_dict(_stress_spec())
    pts = frontier(base, "protocol.max_ltv",
                   [0.50, 0.60, 0.70, 0.80, 0.85], metric="bad_debt", steps=120)
    values = [m for _, m in pts]
    assert values == sorted(values)
    assert values[0] < values[-1]  # there is a real safe->unsafe gradient


def test_oracle_lag_increases_bad_debt():
    """A faster oracle (lag 0) must not be worse than a slow one."""
    spec = _stress_spec()
    spec["populations"][2]["params"]["capital"] = 1_000_000.0  # isolate lag from congestion
    base = MarketConfig.from_dict(spec)
    pts = dict(frontier(base, "oracle.lag_steps", [0, 12], metric="bad_debt", steps=120))
    assert pts[0] < pts[12]


def test_grid_covers_cartesian_product():
    base = MarketConfig.from_dict(_stress_spec())
    results = grid(base, {
        "protocol.max_ltv": [0.6, 0.8],
        "oracle.lag_steps": [0, 12, 24],
    }, steps=80)
    assert len(results) == 2 * 3
    combos = {(r["params"]["protocol.max_ltv"], r["params"]["oracle.lag_steps"])
              for r in results}
    assert len(combos) == 6
