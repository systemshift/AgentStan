"""
Data collection for simulations.

Attach a DataCollector to a Simulation to record time series of
model-level and agent-level metrics each step.
"""

from typing import Dict, Any, List, Callable, Optional


class DataCollector:
    """
    Collects model-level and agent-level data each step.

    Usage:
        collector = DataCollector(
            model_metrics={"total": lambda m: m.agent_manager.get_total_count()},
            agent_metrics={"energy": lambda a: a.get_attribute("energy", 0)},
        )
        collector.collect(simulation)  # call each step
        df = collector.get_model_data()  # list of dicts
    """

    def __init__(
        self,
        model_metrics: Optional[Dict[str, Callable]] = None,
        agent_metrics: Optional[Dict[str, Callable]] = None,
    ):
        self.model_metrics = model_metrics or {}
        self.agent_metrics = agent_metrics or {}
        self.model_data: List[Dict[str, Any]] = []
        self.agent_data: List[Dict[str, Any]] = []

    def collect(self, simulation) -> None:
        """Collect one snapshot of data from the simulation."""
        step = simulation.step

        # Model-level
        row = {"step": step}
        # Always collect counts
        counts = simulation.agent_manager.get_counts()
        row["total_agents"] = simulation.agent_manager.get_total_count()
        for agent_type, count in counts.items():
            row[f"count_{agent_type}"] = count

        # Custom model metrics
        for name, fn in self.model_metrics.items():
            try:
                row[name] = fn(simulation)
            except Exception:
                row[name] = None
        self.model_data.append(row)

        # Agent-level
        if self.agent_metrics:
            for agent in simulation.agent_manager.get_living_agents():
                agent_row = {
                    "step": step,
                    "agent_id": agent.id,
                    "agent_type": agent.type,
                }
                for name, fn in self.agent_metrics.items():
                    try:
                        agent_row[name] = fn(agent)
                    except Exception:
                        agent_row[name] = None
                self.agent_data.append(agent_row)

    def get_model_data(self) -> List[Dict[str, Any]]:
        """Return model-level time series as list of dicts."""
        return self.model_data

    def get_agent_data(self) -> List[Dict[str, Any]]:
        """Return agent-level data as list of dicts."""
        return self.agent_data

    def reset(self) -> None:
        """Clear collected data."""
        self.model_data = []
        self.agent_data = []
