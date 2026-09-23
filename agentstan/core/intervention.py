"""
Mid-simulation intervention engine.

Queue changes (add/remove agents, swap behaviors, modify environment)
that are safely applied between simulation steps.
"""

import threading
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Tuple

logger = logging.getLogger("agentstan")


@dataclass
class Intervention:
    """A single queued intervention."""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = "manual"


class InterventionEngine:
    """
    Queue-based intervention system. Thread-safe.

    Interventions are queued via the public API and applied
    at the start of the next simulation step.
    """

    def __init__(self, simulation):
        self.simulation = simulation
        self._queue: List[Intervention] = []
        self._lock = threading.Lock()
        self.history: List[Tuple[int, Intervention]] = []

    # --- Queue API (thread-safe) ---

    def add_agent(self, agent_type: str, state: Dict[str, Any],
                  behavior_code: str = "", position: Any = None,
                  source: str = "manual") -> None:
        with self._lock:
            self._queue.append(Intervention(
                type="add_agent",
                params={"agent_type": agent_type, "state": state,
                        "behavior_code": behavior_code, "position": position},
                source=source,
            ))

    def remove_agent(self, agent_id: int, source: str = "manual") -> None:
        with self._lock:
            self._queue.append(Intervention(
                type="remove_agent",
                params={"agent_id": agent_id},
                source=source,
            ))

    def swap_behavior(self, agent_id: int, new_behavior: Callable,
                      source: str = "manual") -> None:
        with self._lock:
            self._queue.append(Intervention(
                type="swap_behavior",
                params={"agent_id": agent_id, "new_behavior": new_behavior},
                source=source,
            ))

    def swap_type_behavior(self, agent_type: str, new_behavior: Callable,
                           source: str = "manual") -> None:
        with self._lock:
            self._queue.append(Intervention(
                type="swap_type_behavior",
                params={"agent_type": agent_type, "new_behavior": new_behavior},
                source=source,
            ))

    def modify_agent(self, agent_id: int, attribute: str,
                     value: Any = None, delta: Any = None,
                     source: str = "manual") -> None:
        with self._lock:
            self._queue.append(Intervention(
                type="modify_agent",
                params={"agent_id": agent_id, "attribute": attribute,
                        "value": value, "delta": delta},
                source=source,
            ))

    def teleport_agent(self, agent_id: int, position: Any,
                       source: str = "manual") -> None:
        with self._lock:
            self._queue.append(Intervention(
                type="teleport_agent",
                params={"agent_id": agent_id, "position": position},
                source=source,
            ))

    def modify_environment(self, property_name: str, value: Any,
                           source: str = "manual") -> None:
        with self._lock:
            self._queue.append(Intervention(
                type="modify_environment",
                params={"property_name": property_name, "value": value},
                source=source,
            ))

    # --- Application (called by simulation loop) ---

    def apply_pending(self) -> List[Intervention]:
        """Apply all queued interventions. Returns what was applied."""
        with self._lock:
            pending = list(self._queue)
            self._queue.clear()

        applied = []
        for intervention in pending:
            try:
                self._apply_one(intervention)
                self.history.append((self.simulation.step, intervention))
                applied.append(intervention)
            except Exception as e:
                logger.error(f"Intervention failed: {intervention.type} - {e}")

        return applied

    def _apply_one(self, intervention: Intervention) -> None:
        sim = self.simulation
        p = intervention.params

        if intervention.type == "add_agent":
            if p["agent_type"] in sim.spec["agent_types"]:
                # A spec type: its initial_state and behavior, plus overrides
                agent = sim._new_agent(p["agent_type"], p["state"])
            else:
                from .agent import Agent
                agent = Agent(agent_type=p["agent_type"],
                              initial_state=copy.deepcopy(p["state"]))
                if agent.state.get("position") is None:
                    agent.state["position"] = sim.environment.get_random_position()
            if p.get("behavior_code"):
                agent.behavior_function = sim._compile_behavior_function(
                    p["agent_type"], p["behavior_code"], rng=sim.rng
                )
            if p.get("position"):
                agent.state["position"] = p["position"]
            sim.agent_manager.add_agent(agent)

        elif intervention.type == "remove_agent":
            agent = sim.agent_manager.get_agent(p["agent_id"])
            if agent:
                agent.kill()

        elif intervention.type == "swap_behavior":
            agent = sim.agent_manager.get_agent(p["agent_id"])
            if agent:
                agent.behavior_function = p["new_behavior"]

        elif intervention.type == "swap_type_behavior":
            for agent in sim.agent_manager.get_agents_by_type(p["agent_type"]):
                agent.behavior_function = p["new_behavior"]

        elif intervention.type == "modify_agent":
            agent = sim.agent_manager.get_agent(p["agent_id"])
            if agent:
                if p.get("value") is not None:
                    agent.set_attribute(p["attribute"], p["value"])
                elif p.get("delta") is not None:
                    agent.modify_attribute(p["attribute"], p["delta"])

        elif intervention.type == "teleport_agent":
            agent = sim.agent_manager.get_agent(p["agent_id"])
            if agent:
                agent.state["position"] = p["position"]

        elif intervention.type == "modify_environment":
            sim.environment.set_property(p["property_name"], p["value"])
