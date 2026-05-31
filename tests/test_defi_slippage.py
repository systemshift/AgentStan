"""Tests for the slippage + volatile-debt extension."""

import pytest

from agentstan.defi import MarketConfig, LendingMarket, scenarios
from agentstan.defi.protocol import slippage_fraction, Account


# --- slippage primitive --------------------------------------------------

def test_slippage_fraction_behaviour():
    assert slippage_fraction(100, None) == 0.0       # infinite depth => no slippage
    assert slippage_fraction(0, 100) == 0.0
    assert slippage_fraction(100, 100) == pytest.approx(0.5)   # size == depth => 50%
    assert 0 < slippage_fraction(50, 100) < 0.5
    # monotonic in trade size, always < 1
    assert slippage_fraction(1000, 100) > slippage_fraction(100, 100)
    assert slippage_fraction(1e12, 100) < 1.0


# --- volatile debt -------------------------------------------------------

def test_health_factor_falls_when_debt_price_rises():
    acct = Account(agent_id=0, collateral=1.0, debt=1.0)
    assert acct.health_factor(price=1.0, liquidation_threshold=1.0, debt_price=1.0) == pytest.approx(1.0)
    # a borrowed asset doubling in price halves health from the liability side
    assert acct.health_factor(price=1.0, liquidation_threshold=1.0, debt_price=2.0) == pytest.approx(0.5)


def test_debt_price_default_preserves_single_asset_behaviour():
    # debt_price defaults to 1.0 => identical to the original numeraire model
    acct = Account(agent_id=0, collateral=2.0, debt=1.0)
    assert acct.health_factor(3000.0, 0.85) == pytest.approx(2.0 * 3000.0 * 0.85 / 1.0)


# --- the illiquidity mechanism (CRV class) -------------------------------

def test_crv_squeeze_produces_latent_bad_debt():
    s = LendingMarket(scenarios.crv_short_squeeze()).run(60)["summary"]
    assert s["bad_debt"] > 0
    # the loss is the un-liquidated residual (liquidations blocked by slippage)
    assert s["outstanding_shortfall"] >= 0.5 * s["bad_debt"]


def test_thin_debt_liquidity_causes_more_bad_debt_than_deep():
    """Deep markets let liquidators source the debt asset cheaply and clear the
    position; thin markets make it unprofitable, so bad debt persists."""
    thin = LendingMarket(scenarios.crv_short_squeeze(debt_depth_usd=30_000_000.0)).run(60)["summary"]
    deep = LendingMarket(scenarios.crv_short_squeeze(debt_depth_usd=600_000_000.0)).run(60)["summary"]
    assert thin["bad_debt"] > deep["bad_debt"]


def test_no_liquidity_config_means_no_slippage():
    """A scenario without depth set behaves as the frictionless model (no
    slippage gating) — a regression guard for backward compatibility."""
    cfg = scenarios.eth_crash(oracle_lag=0)
    assert cfg.liquidity.debt_depth_usd is None
    assert cfg.liquidity.collateral_depth_usd is None
    LendingMarket(cfg).run(60)   # runs without touching slippage paths
