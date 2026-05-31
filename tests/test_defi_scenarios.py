"""Tests for scenario presets, heterogeneity, and LLM behavioral agents."""

import pytest

from agentstan.defi import LendingMarket, MarketConfig, scenarios, frontier


# --- scenario presets ----------------------------------------------------

def test_all_presets_run_and_are_distinct():
    summaries = {name: LendingMarket(scenarios.apply(name)).run(120)["summary"]
                 for name in scenarios.SCENARIOS}
    assert set(summaries) == set(scenarios.SCENARIOS)
    # at least one scenario goes insolvent and at least one freezes liquidity
    assert any(s["insolvent"] for s in summaries.values())
    assert any(s["min_available_liquidity"] <= 1.0 for s in summaries.values())


def test_apply_unknown_scenario_raises():
    with pytest.raises(ValueError, match="unknown scenario"):
        scenarios.apply("nonexistent")


def test_oracle_delay_worse_than_fast_oracle():
    fast = LendingMarket(scenarios.eth_crash(oracle_lag=0)).run(120)["summary"]
    slow = LendingMarket(scenarios.oracle_delay(lag=24)).run(120)["summary"]
    assert slow["bad_debt"] > fast["bad_debt"]


def test_whale_panic_freezes_liquidity_without_insolvency():
    s = LendingMarket(scenarios.whale_panic()).run(120)["summary"]
    assert s["min_available_liquidity"] <= 1.0   # liquidity ran dry
    assert s["insolvent"] is False               # but no bad debt


def test_scenarios_preserve_caller_protocol_params():
    base = scenarios.default_market()
    base.protocol.max_ltv = 0.55
    cfg = scenarios.eth_crash(base, severity=0.40)
    assert cfg.protocol.max_ltv == 0.55          # caller's params kept
    assert cfg.scenario.crash_pct == 0.40


# --- heterogeneity -------------------------------------------------------

def test_range_params_produce_heterogeneous_agents():
    cfg = MarketConfig.from_dict({
        "seed": 7,
        "protocol": {"max_ltv": 0.7, "liquidation_threshold": 0.85},
        "oracle": {"lag_steps": 0},
        "scenario": {"initial_price": 3000.0, "crash_pct": 0.0},
        "populations": [
            {"type": "borrower", "count": 20, "policy": "passive",
             "params": {"collateral": 1.0, "borrow_ltv_fraction_range": [0.5, 1.0]}},
        ],
    })
    market = LendingMarket(cfg)
    fracs = [a.params["borrow_ltv_fraction"] for a in market.agents]
    assert all(0.5 <= f <= 1.0 for f in fracs)
    assert len(set(round(f, 4) for f in fracs)) > 1   # genuinely varied
    assert "borrow_ltv_fraction_range" not in market.agents[0].params


def test_determinism_with_heterogeneity():
    cfg = lambda: scenarios.eth_crash()  # default market uses a _range
    a = LendingMarket(cfg()).run(80)["summary"]
    b = LendingMarket(cfg()).run(80)["summary"]
    assert a == b


# --- LLM behavioral agents ----------------------------------------------

def _llm_whale_spec():
    spec = scenarios.whale_panic().to_dict()
    for p in spec["populations"]:
        if p["type"] == "lender" and p["params"].get("supply", 0) >= 100_000:
            p["policy"] = "llm"
    return spec


def test_llm_policy_builds_without_resolve_error():
    # "llm" is not in POLICIES; the engine must accept it anyway.
    LendingMarket(MarketConfig.from_dict(_llm_whale_spec()))


def test_decider_changes_outcome():
    spec = _llm_whale_spec()

    hold = LendingMarket(MarketConfig.from_dict(spec))
    hold.attach_llm(lambda ctx: "hold")
    s_hold = hold.run(120)["summary"]

    panic = LendingMarket(MarketConfig.from_dict(spec))
    panic.attach_llm(lambda ctx: "withdraw" if ctx["price_drawdown_pct"] > 5 else "hold")
    s_panic = panic.run(120)["summary"]

    # a holding whale keeps liquidity; a panicking one drains it
    assert s_hold["min_available_liquidity"] > s_panic["min_available_liquidity"]


def test_decider_errors_fall_back_safely():
    spec = _llm_whale_spec()
    m = LendingMarket(MarketConfig.from_dict(spec))
    m.attach_llm(lambda ctx: 1 / 0)   # decider raises every call
    s = m.run(60)["summary"]          # must not crash the run
    assert "bad_debt" in s


def test_invalid_decider_choice_falls_back():
    spec = _llm_whale_spec()
    m = LendingMarket(MarketConfig.from_dict(spec))
    m.attach_llm(lambda ctx: "buy_the_dip")   # not in the menu
    s = m.run(60)["summary"]
    assert "bad_debt" in s
