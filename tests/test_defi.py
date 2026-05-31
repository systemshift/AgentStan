"""Tests for the DeFi lending-market engine."""

import pytest

from agentstan.defi import MarketConfig, LendingMarket
from agentstan.defi.config import RateModel


def _base_spec(**overrides):
    spec = {
        "seed": 1,
        "protocol": {
            "max_ltv": 0.70, "liquidation_threshold": 0.80,
            "liquidation_bonus": 0.05, "close_factor": 0.5,
            "reserve_factor": 0.10, "initial_reserves": 1000.0,
        },
        "oracle": {"lag_steps": 0},
        "scenario": {"initial_price": 3000.0, "crash_pct": 0.0},
        "populations": [
            {"type": "lender", "count": 1, "policy": "passive",
             "params": {"supply": 100_000.0}},
            {"type": "borrower", "count": 10, "policy": "passive",
             "params": {"collateral": 1.0, "borrow_ltv_fraction": 1.0}},
            {"type": "liquidator", "count": 1, "policy": "greedy",
             "params": {"capital": 500_000.0}},
        ],
    }
    spec.update(overrides)
    return spec


def test_config_roundtrip():
    cfg = MarketConfig.from_dict(_base_spec())
    again = MarketConfig.from_dict(cfg.to_dict())
    assert again.protocol.max_ltv == 0.70
    assert isinstance(again.protocol.rate_model, RateModel)
    assert again.populations[1].count == 10


def test_validation_rejects_threshold_below_ltv():
    spec = _base_spec()
    spec["protocol"]["liquidation_threshold"] = 0.65  # below max_ltv 0.70
    with pytest.raises(ValueError, match="liquidation_threshold"):
        LendingMarket(MarketConfig.from_dict(spec))


def test_quiet_market_has_no_bad_debt():
    """No crash => positions stay healthy => no bad debt."""
    results = LendingMarket(MarketConfig.from_dict(_base_spec())).run(50)
    assert results["summary"]["bad_debt"] == 0
    assert results["summary"]["insolvent"] is False
    # 10 borrowers x 1 ETH x $3000 x 0.70 LTV = $21k borrowed against $100k supply
    assert results["summary"]["max_utilization"] == pytest.approx(0.21, abs=0.01)


def test_rate_model_kink():
    rm = RateModel(base_rate=0.0, slope1=0.04, slope2=0.75, optimal_utilization=0.80)
    assert rm.borrow_rate(0.0) == pytest.approx(0.0)
    assert rm.borrow_rate(0.80) == pytest.approx(0.04)
    # above the kink the jump slope dominates
    assert rm.borrow_rate(1.0) == pytest.approx(0.04 + 0.75)
    assert rm.borrow_rate(0.90) > rm.borrow_rate(0.80)


def test_oracle_lag_and_thin_liquidators_cause_bad_debt():
    """The wedge result: a slow oracle + under-capitalized liquidators turn a
    crash into bad debt, where a fast oracle + deep liquidators would not."""
    risky = _base_spec()
    risky["protocol"]["max_ltv"] = 0.80
    risky["protocol"]["liquidation_threshold"] = 0.85
    risky["oracle"]["lag_steps"] = 24
    risky["scenario"] = {"initial_price": 3000.0, "crash_pct": 0.40,
                         "crash_start": 5, "crash_duration": 20}
    risky["populations"][2]["params"]["capital"] = 5_000.0  # thin

    safe = _base_spec()
    safe["oracle"]["lag_steps"] = 1
    safe["scenario"] = dict(risky["scenario"])
    safe["populations"][2]["params"]["capital"] = 1_000_000.0  # deep

    risky_res = LendingMarket(MarketConfig.from_dict(risky)).run(100)
    safe_res = LendingMarket(MarketConfig.from_dict(safe)).run(100)

    assert risky_res["summary"]["bad_debt"] > 0
    assert risky_res["summary"]["insolvent"] is True
    assert safe_res["summary"]["bad_debt"] < risky_res["summary"]["bad_debt"]


def test_determinism():
    spec = _base_spec()
    spec["scenario"] = {"initial_price": 3000.0, "crash_pct": 0.30,
                        "crash_start": 5, "crash_duration": 15}
    a = LendingMarket(MarketConfig.from_dict(spec)).run(60)
    b = LendingMarket(MarketConfig.from_dict(spec)).run(60)
    assert a["summary"] == b["summary"]
