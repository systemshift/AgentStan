"""Tests for the historical calibration harness."""

from agentstan.defi.calibration import (
    CASES, run_case, calibration_report, _black_thursday,
)


def test_black_thursday_uses_real_inputs():
    """Price path and risk params must come from history, untouched."""
    case = _black_thursday()
    assert case.config.scenario.initial_price == 194.0          # real ETH price
    assert abs(case.config.scenario.crash_pct - 0.43) < 1e-9    # real -43% move
    assert abs(case.config.protocol.max_ltv - 0.66) < 1e-9      # ~150% ratio
    assert abs(case.config.protocol.liquidation_bonus - 0.13) < 1e-9  # 13% penalty
    assert case.sources                                          # cited


def test_black_thursday_reproduces_mechanism_and_magnitude():
    r = run_case(_black_thursday())
    assert r["summary"]["bad_debt"] > 0
    assert r["mechanism_matches"] is True            # bad debt from failed/late liquidations
    assert r["within_order_of_magnitude"] is True    # within ~3x of documented ~6%


def test_short_freeze_avoids_disaster():
    """A fast oracle (no sustained freeze) should not produce the disaster —
    the model's loss is driven by the freeze, not the price drop alone."""
    case = _black_thursday()
    d = case.config.to_dict()
    d["oracle"]["lag_steps"] = 1
    from agentstan.defi import MarketConfig, LendingMarket
    res = LendingMarket(MarketConfig.from_dict(d)).run(48)
    exp = max(h["total_borrowed"] for h in res["history"])
    assert res["summary"]["bad_debt"] / exp < 0.02   # near-zero with working liquidations


def test_report_is_honest_about_scope():
    r = run_case(_black_thursday())
    md = calibration_report(r)
    assert "Sources" in md and "Caveats" in md and "Assumptions" in md
    assert "not validation" in md          # explicitly disclaims overreach
    assert "not fit to the outcome" in md


def test_registry_and_determinism():
    assert "black_thursday" in CASES
    a = run_case(_black_thursday())["summary"]
    b = run_case(_black_thursday())["summary"]
    assert a == b


def test_aave_crv_reproduces_mechanism_and_is_flagged_knife_edge():
    from agentstan.defi.calibration import _aave_crv
    case = _aave_crv()
    assert case.threshold_event is True        # honestly flagged as a near-miss
    assert case.sources
    r = run_case(case)
    assert r["summary"]["bad_debt"] > 0
    assert r["mechanism_matches"] is True      # volatile-debt + slippage-blocked liquidation
    assert r["within_order_of_magnitude"] is True
    md = calibration_report(r)
    assert "threshold/near-miss" in md         # report is candid that it doesn't pin magnitude
    assert "liquidator heterogeneity" in md.lower() or "heterogeneity" in md.lower()
