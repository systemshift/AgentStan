"""
Governance / audit-style report generation.

Turns a headline stress run (plus optional sweeps) into a Markdown document a
risk committee or governance delegate can read: assumptions, the parameters
tested, what happened, where the safe frontier is, and concrete recommendations
derived from the numbers — not boilerplate. Charts are embedded when matplotlib
is available, otherwise the same data is rendered as tables.

    from agentstan.defi import LendingMarket, frontier
    from agentstan.defi.report import write_report

    headline = LendingMarket(cfg).run(120)
    ltv_pts = frontier(cfg, "protocol.max_ltv", [0.5,0.6,0.7,0.8], steps=120)
    write_report("stress_report.md", headline, ltv_frontier=ltv_pts,
                 lag_grid=grid_results, chart_path="frontier.png")
"""

from typing import Any, Dict, List, Optional, Tuple


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _exposure(history: List[Dict[str, Any]]) -> float:
    """Peak USDC borrowed over the run — the capital actually at risk."""
    return max((h["total_borrowed"] for h in history), default=0.0)


# --- sections ------------------------------------------------------------

def _assumptions_section(config: Dict[str, Any]) -> str:
    s = config["scenario"]
    o = config["oracle"]
    crash = _pct(s.get("crash_pct", 0.0))
    lines = [
        "## Assumptions & limitations",
        "",
        f"- **Shock**: collateral price `{s.get('name', 'custom')}` — "
        f"starts at {_usd(s.get('initial_price', 0))}, falls {crash} over "
        f"{s.get('crash_duration', 0)} steps beginning at step {s.get('crash_start', 0)}.",
        f"- **Oracle**: reports price with a {o.get('lag_steps', 0)}-step lag.",
        "- **Deterministic price path** — this is scenario analysis, not a price forecast.",
        "- Single collateral asset and single borrow asset.",
        "- Borrower/lender behavior follows fixed rule-based policies; LLM "
        "behavioral agents (panic, coordination) are not used in this run.",
        "- Liquidators are rational (skip unprofitable liquidations at the true "
        "market price) and capital-constrained.",
    ]
    return "\n".join(lines)


def _parameters_section(config: Dict[str, Any]) -> str:
    p = config["protocol"]
    rows = [
        ("Max LTV", _pct(p["max_ltv"])),
        ("Liquidation threshold", _pct(p["liquidation_threshold"])),
        ("Liquidation bonus", _pct(p["liquidation_bonus"])),
        ("Close factor", _pct(p["close_factor"])),
        ("Reserve factor", _pct(p["reserve_factor"])),
        ("Initial reserves", _usd(p.get("initial_reserves", 0.0))),
        ("Oracle lag (steps)", str(config["oracle"].get("lag_steps", 0))),
    ]
    out = ["## Parameters tested", "", "| Parameter | Value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


def _result_section(summary: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    exposure = _exposure(history)
    bad = summary["bad_debt"]
    bad_pct = (bad / exposure) if exposure else 0.0
    verdict = "**INSOLVENT**" if summary["insolvent"] else "**Solvent**"
    rows = [
        ("Outcome", verdict),
        ("Bad debt", f"{_usd(bad)} ({_pct(bad_pct)} of peak borrowings)"),
        ("— realized via liquidation", _usd(summary["realized_bad_debt"])),
        ("— un-liquidated (latent)", _usd(summary["outstanding_shortfall"])),
        ("Peak borrowings at risk", _usd(exposure)),
        ("Liquidations executed", str(summary["total_liquidations"])),
        ("Liquidation volume", _usd(summary["total_liquidation_volume"])),
        ("Peak underwater positions", str(summary["peak_underwater_accounts"])),
        ("Insurance fund used", _usd(summary["insurance_fund_used"])),
        ("Min available liquidity", _usd(summary["min_available_liquidity"])),
        ("Peak utilization", _pct(summary["max_utilization"])),
    ]
    out = ["## Result", "", "| Metric | Value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    out += ["", _failure_narrative(summary, exposure)]
    return "\n".join(out)


def _failure_narrative(summary: Dict[str, Any], exposure: float) -> str:
    """Explain *why* — inferred from the realized/latent split, not hardcoded."""
    if not summary["insolvent"]:
        return (
            "Under this scenario the protocol remains solvent: liquidations clear "
            "underwater positions while collateral still covers the debt, and no "
            "bad debt accrues."
        )
    realized = summary["realized_bad_debt"]
    latent = summary["outstanding_shortfall"]
    parts = [
        f"Under this scenario the protocol becomes insolvent, accruing "
        f"{_usd(summary['bad_debt'])} of bad debt."
    ]
    if latent > realized:
        parts.append(
            "Most of the loss is **latent** — positions that were never "
            "liquidated. This points to liquidations failing to keep pace: either "
            "the oracle lagged so liquidations were unprofitable while prices fell, "
            "or liquidator capital was exhausted (congestion). The position count "
            f"underwater peaked at {summary['peak_underwater_accounts']}."
        )
    else:
        parts.append(
            "Most of the loss was **realized** through liquidations that completed "
            "only after collateral had fallen below the debt — i.e. liquidations "
            "fired too late to protect the protocol."
        )
    if summary["insurance_fund_used"] > 0:
        parts.append(
            f"The insurance fund absorbed {_usd(summary['insurance_fund_used'])} "
            "before suppliers took losses."
        )
    return " ".join(parts)


def _safe_ltv(ltv_frontier: List[Tuple[float, float]],
              tolerance: float) -> Optional[float]:
    """Highest tested LTV whose bad debt stays within tolerance."""
    safe = [ltv for ltv, bad in ltv_frontier if bad <= tolerance]
    return max(safe) if safe else None


def _frontier_section(ltv_frontier: List[Tuple[float, float]],
                      current_ltv: float, exposure: float,
                      chart_rel: Optional[str]) -> str:
    out = ["## Safe-LTV frontier", ""]
    if chart_rel:
        out += [f"![Bad debt vs max LTV]({chart_rel})", ""]
    out += ["| Max LTV | Bad debt |", "|---|---|"]
    for ltv, bad in ltv_frontier:
        flag = "  ← current" if abs(ltv - current_ltv) < 1e-9 else ""
        out.append(f"| {_pct(ltv)} | {_usd(bad)}{flag} |")
    return "\n".join(out)


def _recommendations_section(summary: Dict[str, Any],
                             ltv_frontier: Optional[List[Tuple[float, float]]],
                             config: Dict[str, Any], exposure: float) -> str:
    recs = []
    current_ltv = config["protocol"]["max_ltv"]
    lag = config["oracle"].get("lag_steps", 0)

    if ltv_frontier:
        tol = max(1.0, 0.001 * exposure)  # ~0.1% of exposure counts as "clean"
        safe = _safe_ltv(ltv_frontier, tol)
        if safe is None:
            worst = min(ltv_frontier, key=lambda t: t[1])
            recs.append(
                f"No tested max-LTV avoids bad debt under this shock; even "
                f"{_pct(worst[0])} produces {_usd(worst[1])}. Consider a lower LTV "
                "than tested, a higher liquidation threshold margin, or a larger "
                "reserve/insurance buffer."
            )
        elif safe < current_ltv:
            recs.append(
                f"Lower **max LTV to ≤ {_pct(safe)}** (currently {_pct(current_ltv)}) "
                f"to eliminate bad debt under this scenario at the modeled oracle "
                f"lag of {lag} steps."
            )
        else:
            recs.append(
                f"The current max LTV of {_pct(current_ltv)} is within the safe "
                f"frontier ({_pct(safe)}) for this scenario."
            )

    if summary["insolvent"] and lag > 0:
        recs.append(
            f"Bad debt is sensitive to the {lag}-step oracle lag — faster price "
            "feeds (or a liquidation grace/auction mechanism robust to stale "
            "prices) materially reduce losses."
        )
    if summary["outstanding_shortfall"] > summary["realized_bad_debt"]:
        recs.append(
            "Latent (un-liquidated) losses dominate — stress liquidator capital "
            "and incentives, not just protocol ratios."
        )
    if not recs:
        recs.append("No parameter changes indicated for this scenario.")

    return "## Recommendations\n\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(recs, 1))


def _render_chart(ltv_frontier: List[Tuple[float, float]], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    xs = [v for v, _ in ltv_frontier]
    ys = [m for _, m in ltv_frontier]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("max LTV")
    plt.ylabel("bad debt ($)")
    plt.title("Bad debt vs max LTV")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return True


# --- public API ----------------------------------------------------------

def stress_report(
    headline: Dict[str, Any],
    *,
    title: Optional[str] = None,
    ltv_frontier: Optional[List[Tuple[float, float]]] = None,
    lag_grid: Optional[List[Dict[str, Any]]] = None,
    chart_path: Optional[str] = None,
) -> str:
    """Build a governance-style Markdown stress report from a headline run.

    Args:
        headline: a results dict from ``LendingMarket.run`` (config/summary/history).
        title: report title; defaults to the scenario name.
        ltv_frontier: optional [(max_ltv, bad_debt)] from ``frontier(...)`` —
            drives the safe-frontier section and the LTV recommendation.
        lag_grid: optional ``grid(...)`` output keyed by protocol.max_ltv x
            oracle.lag_steps — rendered as a bad-debt heatmap table.
        chart_path: if set and matplotlib is installed, the frontier chart is
            written there and embedded.
    """
    config = headline["config"]
    summary = headline["summary"]
    history = headline["history"]
    exposure = _exposure(history)
    scenario_name = config["scenario"].get("name", "custom")
    title = title or f"Stress test: {scenario_name}"

    chart_rel = None
    if chart_path and ltv_frontier and _render_chart(ltv_frontier, chart_path):
        chart_rel = chart_path

    blocks = [
        f"# {title}",
        "",
        f"*Generated by AgentStan DeFi — behavioral stress testing.*",
        "",
        _assumptions_section(config),
        "",
        _parameters_section(config),
        "",
        _result_section(summary, history),
    ]
    if ltv_frontier:
        blocks += ["", _frontier_section(ltv_frontier,
                                         config["protocol"]["max_ltv"],
                                         exposure, chart_rel)]
    if lag_grid:
        blocks += ["", _lag_grid_section(lag_grid)]
    blocks += ["", _recommendations_section(summary, ltv_frontier, config, exposure)]
    return "\n".join(blocks) + "\n"


def _lag_grid_section(lag_grid: List[Dict[str, Any]]) -> str:
    ltvs = sorted({r["params"]["protocol.max_ltv"] for r in lag_grid})
    lags = sorted({r["params"]["oracle.lag_steps"] for r in lag_grid})
    lut = {(r["params"]["protocol.max_ltv"], r["params"]["oracle.lag_steps"]):
           r["summary"]["bad_debt"] for r in lag_grid}
    header = "| Max LTV \\ Oracle lag | " + " | ".join(str(l) for l in lags) + " |"
    sep = "|---|" + "|".join("---" for _ in lags) + "|"
    rows = [header, sep]
    for ltv in ltvs:
        cells = " | ".join(_usd(lut.get((ltv, l), 0.0)) for l in lags)
        rows.append(f"| {_pct(ltv)} | {cells} |")
    return "## Bad debt: max LTV vs oracle lag\n\n" + "\n".join(rows)


def write_report(path: str, headline: Dict[str, Any], **kwargs) -> str:
    """Build a report and write it to ``path``. Returns the Markdown string."""
    md = stress_report(headline, **kwargs)
    with open(path, "w") as f:
        f.write(md)
    return md
