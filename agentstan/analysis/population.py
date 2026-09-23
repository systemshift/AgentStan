"""
Population dynamics analysis.

Analyze time series from simulation results to detect patterns:
stability, oscillations, crashes, growth phases.
"""

from typing import Dict, Any, List, Optional


def analyze(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze population dynamics from simulation results.

    Args:
        results: Results dict from Simulation.run()

    Returns:
        Analysis dict with per-type and overall metrics.
    """
    history = results.get("metrics", {}).get("history", [])
    if not history:
        return {"error": "No history data"}

    summary = results.get("summary", {})
    agent_types = list(summary.get("initial_counts", {}).keys())

    report = {
        "steps": len(history),
        "agent_types": {},
        "overall": {},
    }

    # Per-type analysis
    for agent_type in agent_types:
        series = [h["agent_counts"].get(agent_type, 0) for h in history]
        report["agent_types"][agent_type] = _analyze_series(
            series, agent_type, summary
        )

    # Overall
    total_series = [h["total_agents"] for h in history]
    report["overall"] = {
        "initial": summary.get("initial_agents", 0),
        "final": summary.get("final_agents", 0),
        "peak": max(total_series) if total_series else 0,
        "trough": min(total_series) if total_series else 0,
        "stability": _classify_stability(total_series),
    }

    return report


def _analyze_series(
    series: List[int], agent_type: str, summary: Dict
) -> Dict[str, Any]:
    """Analyze a single population time series."""
    if not series:
        return {}

    initial = summary.get("initial_counts", {}).get(agent_type, 0)
    final = series[-1]
    peak = max(series)
    peak_step = series.index(peak)
    trough = min(series)
    trough_step = series.index(trough)

    # Survival
    extinct = final == 0
    extinction_step = None
    if extinct:
        for i, v in enumerate(series):
            if v == 0:
                extinction_step = i + 1
                break

    # Growth rate (average per-step change)
    if len(series) > 1:
        changes = [series[i] - series[i - 1] for i in range(1, len(series))]
        avg_change = sum(changes) / len(changes)
    else:
        avg_change = 0

    # Oscillation detection
    stability = _classify_stability(series)
    period = _detect_period(series) if stability == "oscillating" else None

    return {
        "initial": initial,
        "final": final,
        "peak": peak,
        "peak_step": peak_step,
        "trough": trough,
        "trough_step": trough_step,
        "extinct": extinct,
        "extinction_step": extinction_step,
        "avg_change_per_step": round(avg_change, 3),
        "stability": stability,
        "period": period,
    }


def _classify_stability(series: List[int]) -> str:
    """Classify a time series as stable, growing, declining, oscillating, or crashed."""
    if len(series) < 5:
        return "insufficient_data"

    if series[-1] == 0:
        return "crashed"

    # Use second half to assess trend
    half = len(series) // 2
    first_half_avg = sum(series[:half]) / half
    second_half_avg = sum(series[half:]) / (len(series) - half)

    if first_half_avg == 0:
        return "crashed"

    ratio = second_half_avg / first_half_avg

    # Check for oscillation: count direction changes
    changes = [series[i] - series[i - 1] for i in range(1, len(series))]
    sign_changes = sum(
        1 for i in range(1, len(changes))
        if (changes[i] > 0) != (changes[i - 1] > 0) and changes[i] != 0 and changes[i - 1] != 0
    )
    oscillation_rate = sign_changes / len(changes) if changes else 0

    if oscillation_rate > 0.3:
        return "oscillating"
    elif ratio > 1.2:
        return "growing"
    elif ratio < 0.8:
        return "declining"
    else:
        return "stable"


def _detect_period(series: List[int]) -> Optional[int]:
    """Detect oscillation period using autocorrelation."""
    if len(series) < 10:
        return None

    n = len(series)
    mean = sum(series) / n
    centered = [x - mean for x in series]

    # Variance
    var = sum(x * x for x in centered) / n
    if var == 0:
        return None

    # Autocorrelation for lags 2 to n//2
    best_lag = None
    best_corr = 0.3  # minimum threshold

    for lag in range(2, min(n // 2, 100)):
        corr = sum(centered[i] * centered[i + lag] for i in range(n - lag)) / (n * var)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag
