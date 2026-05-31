"""
Calibration harness: run the model against documented historical crises.

The point is honesty, not a victory lap. A case fixes the things we know from
history — the real price path and the protocol's real risk parameters — and
leaves the genuinely unknown things (the exact vault distribution, how long the
liquidation machinery was effectively offline) as *stated assumptions*. We then
ask: does the model reproduce the right failure MECHANISM at the right ORDER OF
MAGNITUDE? It does not, and should not claim to, reproduce an exact dollar
figure — that would be curve-fitting one data point.

One event is a sanity check, not validation. Validation needs several events
and out-of-sample checks. This module is built so cases compose into that
library over time.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .config import MarketConfig
from .engine import LendingMarket


@dataclass
class HistoricalCase:
    name: str
    description: str
    config: MarketConfig                      # real params + real price path
    documented: Dict[str, Any]                # what actually happened (sourced)
    target_bad_debt_pct: float                # documented loss / system debt (a band center)
    sources: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    threshold_event: bool = False             # near-miss; model brackets but doesn't pin the magnitude
    steps: int = 48                           # the event's natural horizon


def _black_thursday() -> HistoricalCase:
    """MakerDAO, 12 March 2020. ETH -43% in a day; gas spike + stale oracle +
    stuck keeper bots froze liquidations; ~$8.32M collateral cleared at $0 bids;
    ~5.67M DAI left uncollateralized (system debt then was < $100M).

    From-history (not fit): the -43% price path; the 150% liquidation ratio
    (~66.7% LTV) and 13% penalty. Modeling assumptions: the vault distribution
    and that liquidations were effectively offline for the worst of the move
    (here: a stale oracle + thin liquidator capital). Maker is auction/CDP-based
    and this engine is pool-based, so we calibrate the *liquidation-failure
    pathway and its magnitude*, not Maker's exact mechanics.
    """
    config = MarketConfig.from_dict({
        "seed": 1,
        "protocol": {
            # Maker ETH-A: 150% liq ratio => ~66.7% LTV-equivalent; 13% penalty.
            "max_ltv": 0.66, "liquidation_threshold": 0.667,
            "liquidation_bonus": 0.13, "close_factor": 0.5,
            "reserve_factor": 0.0, "initial_reserves": 0.0,
        },
        # Stale oracle + stuck keepers: liquidations effectively offline for the
        # worst of the crash.
        "oracle": {"lag_steps": 8},
        "scenario": {"name": "Black Thursday (ETH -43%)", "initial_price": 194.0,
                     "crash_pct": 0.43, "crash_start": 4, "crash_duration": 12},
        "populations": [
            {"type": "lender", "count": 1, "policy": "passive",
             "params": {"supply": 200_000.0}},
            # A spread of vaults; risky ones were drawn near the liquidation ratio.
            {"type": "borrower", "count": 100, "policy": "passive",
             "params": {"collateral": 1.0, "borrow_ltv_fraction_range": [0.80, 1.0]}},
            # Keepers overwhelmed/failing: badly under-capitalized liquidators.
            {"type": "liquidator", "count": 1, "policy": "greedy",
             "params": {"capital": 3_000.0}},
        ],
    })
    return HistoricalCase(
        name="black_thursday_makerdao",
        description="MakerDAO Black Thursday, 12 March 2020 — ETH -43%, "
                    "liquidations froze, ~5.67M DAI uncollateralized.",
        config=config,
        documented={
            "eth_drawdown": 0.43,
            "eth_from_to": "$194 -> $111",
            "liquidation_ratio": "150% (~66.7% LTV)",
            "liquidation_penalty": "13%",
            "bad_debt_usd": 5_670_000,
            "collateral_lost_at_zero_bids_usd": 8_320_000,
            "mechanism": "gas spike + stale oracle + stuck keeper bots -> "
                         "auctions cleared at $0 -> uncovered debt",
        },
        # ~5.67M deficit against a sub-$100M system => mid-single-digit %.
        target_bad_debt_pct=0.06,
        sources=[
            "https://medium.com/@whiterabbit_hq/black-thursday-for-makerdao-8-32-million-was-liquidated-for-0-dai-36b83cac56b6",
            "https://insights.glassnode.com/what-really-happened-to-makerdao/",
            "https://www.coindesk.com/tech/2020/07/22/mempool-manipulation-enabled-theft-of-8m-in-makerdao-collateral-on-black-thursday-report",
        ],
        assumptions=[
            "Vault distribution: 100 vaults drawn at 80-100% of the max ratio "
            "(real distribution unknown).",
            "Liquidation freeze modeled as an 8-step stale oracle + a thin "
            "liquidator (real keeper-failure duration unknown).",
        ],
        caveats=[
            "Maker is auction/CDP-based; this engine is pool-based — we match "
            "the liquidation-failure magnitude, not Maker's exact mechanics.",
            "Absolute $ depends on simulated TVL; the scale-invariant comparison "
            "is bad debt as a % of borrowings.",
            "Documented system-debt denominator (<$100M) is approximate, so the "
            "target % is a band, not a point.",
        ],
    )


def _aave_crv() -> HistoricalCase:
    """Aave V2, November 2022. Avraham Eisenberg deposited ~$63.6M USDC and
    borrowed ~92M CRV (~$40M) to short it; CRV instead *squeezed upward*
    (~$0.40 -> ~$0.62+). His position was liquidated over ~1 hour / 300+ txns by
    ~20 liquidators who had to source thin CRV to repay, leaving Aave ~2.6M CRV
    (~$1.6M) of bad debt.

    From-history (not fit): USDC collateral, CRV debt, the ~+65% CRV move, the
    ~63% effective LTV. This is the illiquidity-driven / volatile-debt class the
    base model could not represent before the slippage + debt-price extension.
    """
    from . import scenarios
    config = scenarios.crv_short_squeeze()  # USDC collateral, CRV debt, thin CRV depth
    return HistoricalCase(
        name="aave_crv_2022",
        description="Aave V2 CRV short squeeze, Nov 2022 — borrowed asset spiked, "
                    "thin liquidity left ~$1.6M bad debt.",
        config=config,
        documented={
            "collateral": "~$63.6M USDC",
            "borrowed": "~92M CRV (~$40M)",
            "crv_move": "~$0.40 -> ~$0.62+ (squeeze up)",
            "bad_debt_usd": 1_600_000,
            "bad_debt_crv": "~2.6M CRV",
            "mechanism": "volatile borrowed asset spikes + thin CRV liquidity -> "
                         "liquidating (buying CRV) is lossy/slow -> residual bad debt",
        },
        target_bad_debt_pct=0.04,   # ~$1.6M / ~$40M borrowed
        threshold_event=True,
        steps=60,                   # squeeze + multi-step liquidation winddown
        sources=[
            "https://research.kaiko.com/insights/crv-aave-liquidation",
            "https://blockworks.co/news/aave-curve-bad-debt",
            "https://thedefiant.io/news/defi/crv-trade-aave-bad-debt",
        ],
        assumptions=[
            "Single whale position at ~63% LTV (the documented shape).",
            "CRV market depth (debt_depth_usd) is an estimate — the real figure "
            "drives whether liquidation is profitable.",
        ],
        caveats=[
            "KNIFE-EDGE / near-miss: with homogeneous liquidators the model "
            "bifurcates (fully cleared vs large residual). It brackets the "
            "documented ~4% but does NOT pin the magnitude.",
            "Reproducing the real '300 partial liquidations + small residual' "
            "smoothly needs liquidator heterogeneity (varied gas/cost/speed) — "
            "the same fix that smoothed the oracle-lag cliff for borrowers.",
            "This case exercises the slippage + volatile-debt extension; it is a "
            "weaker calibration than Black Thursday by design.",
        ],
    )


CASES = {"black_thursday": _black_thursday, "aave_crv": _aave_crv}


def run_case(case: HistoricalCase, steps: int = None) -> Dict[str, Any]:
    """Run a case and return model output alongside the documented actuals."""
    results = LendingMarket(case.config).run(steps or case.steps)
    s = results["summary"]
    hist = results["history"]
    # Borrowed principal in USD = debt units x debt price at origination. (Using
    # the debt asset's price matters when the borrowed asset is itself volatile.)
    h0 = hist[0]
    exposure = h0["total_borrowed"] * h0["debt_true_price"]
    model_pct = (s["bad_debt"] / exposure) if exposure else 0.0

    target = case.target_bad_debt_pct
    within_oom = (target / 3.0) <= model_pct <= (target * 3.0)  # order-of-magnitude band
    # The documented failure was un-liquidated/under-covered debt; check the
    # model fails the same way rather than via, say, a liquidity freeze only.
    mechanism_ok = s["outstanding_shortfall"] >= 0.5 * s["bad_debt"] and s["bad_debt"] > 0

    return {
        "case": case,
        "summary": s,
        "model_bad_debt_pct": model_pct,
        "exposure": exposure,
        "within_order_of_magnitude": within_oom,
        "mechanism_matches": mechanism_ok,
    }


def calibration_report(result: Dict[str, Any]) -> str:
    case: HistoricalCase = result["case"]
    s = result["summary"]
    d = case.documented
    lines = [
        f"# Calibration: {case.name}",
        "",
        case.description,
        "",
        "## Documented (from history)",
        *[f"- {k.replace('_', ' ')}: {(f'${v:,.0f}' if isinstance(v, (int, float)) else v)}"
          for k, v in d.items()],
        f"- (documented loss ~{case.target_bad_debt_pct:.0%} of borrowings)",
        "",
        "## Model (real price path + real risk params, stated assumptions)",
        f"- Bad debt: ${s['bad_debt']:,.0f} = {result['model_bad_debt_pct']:.1%} of borrowings",
        f"  (realized ${s['realized_bad_debt']:,.0f} / latent ${s['outstanding_shortfall']:,.0f})",
        f"- Peak underwater vaults: {s['peak_underwater_accounts']}",
        f"- Liquidations executed: {s['total_liquidations']}",
        "",
        "## Verdict",
        f"- Right order of magnitude (target ~{case.target_bad_debt_pct:.0%}, "
        f"model {result['model_bad_debt_pct']:.1%}): "
        f"{'PASS' if result['within_order_of_magnitude'] else 'FAIL'}",
        f"- Right mechanism (bad debt from failed/late liquidations): "
        f"{'PASS' if result['mechanism_matches'] else 'FAIL'}",
        *(["- NOTE: threshold/near-miss event — the model brackets the outcome "
           "but does not pin its magnitude (see caveats)."] if case.threshold_event else []),
        "",
        "## Assumptions (not taken from history)",
        *[f"- {a}" for a in case.assumptions],
        "",
        "## Caveats",
        *[f"- {c}" for c in case.caveats],
        "",
        "## Sources",
        *[f"- {u}" for u in case.sources],
        "",
        "_One event is a sanity check, not validation. The price path and risk "
        "parameters are from history; they were not fit to the outcome._",
    ]
    return "\n".join(lines) + "\n"
