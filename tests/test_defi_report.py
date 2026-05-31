"""Tests for the DeFi stress-report generator."""

from agentstan.defi import LendingMarket, MarketConfig, frontier
from agentstan.defi.report import stress_report


def _spec(**proto):
    p = {"max_ltv": 0.70, "liquidation_threshold": 0.90, "liquidation_bonus": 0.08,
         "close_factor": 0.5, "reserve_factor": 0.10, "initial_reserves": 10_000.0}
    p.update(proto)
    return {
        "seed": 1,
        "protocol": p,
        "oracle": {"lag_steps": 12},
        "scenario": {"name": "ETH -40%", "initial_price": 3000.0,
                     "crash_pct": 0.40, "crash_start": 10, "crash_duration": 30},
        "populations": [
            {"type": "lender", "count": 1, "policy": "passive",
             "params": {"supply": 150_000.0}},
            {"type": "borrower", "count": 40, "policy": "passive",
             "params": {"collateral": 1.0, "borrow_ltv_fraction": 1.0}},
            {"type": "liquidator", "count": 1, "policy": "greedy",
             "params": {"capital": 30_000.0}},
        ],
    }


def test_report_has_core_sections():
    cfg = MarketConfig.from_dict(_spec())
    md = stress_report(LendingMarket(cfg).run(120), title="Test report")
    for heading in ("# Test report", "## Assumptions & limitations",
                    "## Parameters tested", "## Result", "## Recommendations"):
        assert heading in md


def test_insolvent_run_reads_as_insolvent():
    cfg = MarketConfig.from_dict(_spec(max_ltv=0.80))
    md = stress_report(LendingMarket(cfg).run(120))
    assert "INSOLVENT" in md
    assert "becomes insolvent" in md


def test_solvent_run_reads_as_solvent():
    # low LTV + fast oracle => survives the crash cleanly
    spec = _spec(max_ltv=0.50)
    spec["oracle"]["lag_steps"] = 1
    cfg = MarketConfig.from_dict(spec)
    md = stress_report(LendingMarket(cfg).run(120))
    assert "remains solvent" in md
    assert "INSOLVENT" not in md


def test_frontier_drives_ltv_recommendation():
    cfg = MarketConfig.from_dict(_spec(max_ltv=0.80))
    headline = LendingMarket(cfg).run(120)
    pts = frontier(cfg, "protocol.max_ltv",
                   [0.50, 0.55, 0.60, 0.70, 0.80], metric="bad_debt", steps=120)
    md = stress_report(headline, ltv_frontier=pts)
    assert "## Safe-LTV frontier" in md
    assert "← current" in md
    # safe frontier is well below the current 80% LTV
    assert "Lower **max LTV" in md
