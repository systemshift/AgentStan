"""
Event stream analysis.

Break down simulation events: deaths by cause, birth rates,
interaction outcomes, energy flows.
"""

from typing import Dict, Any, List
from collections import Counter


def analyze_events(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the event stream from simulation results.

    Args:
        results: Results dict from Simulation.run()

    Returns:
        Event analysis dict.
    """
    events = results.get("events", [])

    report = {
        "total_events": len(events),
        "event_types": Counter(e["type"] for e in events),
        "deaths": _analyze_deaths(events),
        "births": _analyze_births(events),
        "interactions": _analyze_interactions(events),
    }

    return report


def _analyze_deaths(events: List[Dict]) -> Dict[str, Any]:
    """Break down agent deaths by cause and type."""
    deaths = [e for e in events if e["type"] == "agent_death"]
    if not deaths:
        return {"total": 0}

    by_cause = Counter(d.get("cause", "unknown") for d in deaths)
    by_type = Counter(d.get("agent_type", "unknown") for d in deaths)

    # Death rate over time (deaths per 10-step window)
    if deaths:
        max_step = max(d["step"] for d in deaths)
        window = 10
        death_rate = []
        for start in range(0, max_step, window):
            count = sum(1 for d in deaths if start <= d["step"] < start + window)
            death_rate.append({"step_range": f"{start}-{start + window}", "deaths": count})
    else:
        death_rate = []

    return {
        "total": len(deaths),
        "by_cause": dict(by_cause),
        "by_type": dict(by_type),
        "death_rate": death_rate,
    }


def _analyze_births(events: List[Dict]) -> Dict[str, Any]:
    """Analyze reproduction events."""
    births = [e for e in events if e["type"] == "agent_birth"]
    if not births:
        return {"total": 0}

    by_type = Counter(b.get("agent_type", "unknown") for b in births)

    return {
        "total": len(births),
        "by_type": dict(by_type),
    }


def _analyze_interactions(events: List[Dict]) -> Dict[str, Any]:
    """Analyze agent interactions."""
    interactions = [e for e in events if e["type"] == "interaction"]
    if not interactions:
        return {"total": 0}

    by_type = Counter(i.get("interaction_type", "unknown") for i in interactions)
    by_outcome = Counter(i.get("outcome", "unknown") for i in interactions)

    return {
        "total": len(interactions),
        "by_type": dict(by_type),
        "by_outcome": dict(by_outcome),
    }
