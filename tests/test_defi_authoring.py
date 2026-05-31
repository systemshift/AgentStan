"""Tests for natural-language config authoring."""

import pytest

from agentstan.defi import author_config, MarketConfig, LendingMarket
from agentstan.defi.authoring import heuristic_intent, assemble


# --- intent assembly -----------------------------------------------------

def test_assemble_applies_overrides_and_scenario():
    intent = {"scenario": "oracle_delay",
              "protocol": {"max_ltv": 0.65, "liquidation_threshold": 0.80},
              "scenario_params": {"lag": 30, "severity": 0.5}}
    cfg = assemble(intent)
    assert cfg.protocol.max_ltv == 0.65
    assert cfg.protocol.liquidation_threshold == 0.80
    assert cfg.oracle.lag_steps == 30
    assert cfg.scenario.crash_pct == 0.5


def test_assemble_filters_params_to_preset_signature():
    # eth_crash takes oracle_lag (not lag); passing both must not error.
    intent = {"scenario": "eth_crash", "scenario_params": {"oracle_lag": 12, "lag": 99}}
    cfg = assemble(intent)
    assert cfg.oracle.lag_steps == 12


def test_assemble_repairs_threshold_below_ltv():
    intent = {"scenario": "eth_crash",
              "protocol": {"max_ltv": 0.80, "liquidation_threshold": 0.70}}
    cfg = assemble(intent)  # invalid as given; should be repaired, not raise
    assert cfg.protocol.liquidation_threshold > cfg.protocol.max_ltv


def test_assemble_unknown_scenario_defaults():
    cfg = assemble({"scenario": "rugpull"})
    assert cfg.scenario.crash_pct  # fell back to a real (eth_crash) scenario


# --- heuristic extraction ------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("simulate an ETH crash", "eth_crash"),
    ("stablecoin depeg risk", "stablecoin_depeg"),
    ("a whale dumps and exits", "whale_panic"),
    ("mass withdrawals / bank run", "liquidity_withdrawal"),
    ("liquidator congestion", "liquidation_congestion"),
    ("the oracle is stale", "oracle_delay"),
])
def test_heuristic_scenario_detection(text, expected):
    assert heuristic_intent(text)["scenario"] == expected


def test_heuristic_picks_percentage_nearest_anchor():
    # two percentages in one sentence must map to the right parameter
    intent = heuristic_intent("82.5% liquidation threshold with max LTV 75%")
    assert intent["protocol"]["liquidation_threshold"] == pytest.approx(0.825)
    assert intent["protocol"]["max_ltv"] == pytest.approx(0.75)


def test_author_config_heuristic_end_to_end():
    cfg = author_config("40% ETH crash, max LTV 70%, oracle lag 20 steps")
    assert cfg.protocol.max_ltv == pytest.approx(0.70)
    assert cfg.scenario.crash_pct == pytest.approx(0.40)
    assert cfg.oracle.lag_steps == 20
    assert cfg.validate() == []        # always returns a valid config
    LendingMarket(cfg).run(60)          # and it runs


# --- generator plumbing --------------------------------------------------

def test_injected_generator_is_used():
    def stub(description):
        return {"scenario": "liquidation_congestion",
                "protocol": {"max_ltv": 0.5}, "scenario_params": {"severity": 0.6}}
    cfg = author_config("anything", generator=stub)
    assert cfg.protocol.max_ltv == 0.5
    assert cfg.scenario.crash_pct == 0.6
    assert cfg.scenario.name == "Liquidation congestion"


def test_bad_generator_output_falls_back_to_heuristic():
    cfg = author_config("an ETH crash scenario", generator=lambda d: "not a dict")
    assert isinstance(cfg, MarketConfig)
    assert cfg.scenario.crash_pct  # heuristic produced a crash scenario


def test_authoring_is_deterministic():
    p = "60% max LTV, fast 50% crash with thin liquidators"
    assert author_config(p).to_dict() == author_config(p).to_dict()
