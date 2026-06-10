"""
Clean JSON serialization for API responses.

Ensures all simulation output is json.dumps() safe without default=str hacks.
"""

import copy
from typing import Dict, Any


def clean_for_json(obj: Any) -> Any:
    """Recursively make an object JSON-serializable."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    if isinstance(obj, set):
        return [clean_for_json(item) for item in sorted(obj, key=str)]
    # Fallback: stringify
    return str(obj)


def simulation_to_dict(sim) -> Dict[str, Any]:
    """
    Full simulation state as a clean JSON-serializable dict.
    Suitable for database storage or API response.
    """
    agents = []
    for agent in sim.agent_manager.agents:
        agents.append({
            "id": agent.id,
            "type": agent.type,
            "alive": agent.alive,
            "state": clean_for_json(agent.state),
        })

    return {
        "spec": clean_for_json(sim.spec),
        "step": sim.step,
        "agents": agents,
        "environment": clean_for_json(sim.environment.to_dict()),
        "metrics": clean_for_json(sim.metrics),
    }


def results_to_dict(results: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a results dict for JSON serialization."""
    return clean_for_json(results)
