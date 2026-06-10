"""
Core simulation engine.
"""

import copy
import time
import random as _random_module
import logging
from typing import Dict, Any, List, Optional, Callable

from .environment import Environment
from .agent import Agent, AgentManager
from .actions import ActionProcessor
from .logger import EventLogger
from .scheduler import RandomScheduler

log = logging.getLogger("agentstan")


class Simulation:
    """
    Core simulation engine that manages environment, agents, and execution.

    Args:
        specification: Simulation spec dict with environment and agent_types.
        scheduler: Agent activation scheduler (default: RandomScheduler).
        seed: RNG seed. If set, the run is deterministic: same spec + same
            seed = same results. Each simulation owns its own RNG, so
            concurrent simulations never share random state.
    """

    def __init__(self, specification: Dict[str, Any], scheduler=None,
                 seed: Optional[int] = None):
        self._validate_spec(specification)
        self.spec = specification
        self.step = 0
        self.seed = seed if seed is not None else specification.get("seed")
        self.rng = _random_module.Random(self.seed)
        self.scheduler = scheduler or RandomScheduler()
        if getattr(self.scheduler, "rng", None) is None:
            self.scheduler.rng = self.rng
        self.collectors = []
        self._behavior_cache: Dict[str, Optional[Callable]] = {}

        # Initialize core systems
        self.environment = self._create_environment()
        self.agent_manager = AgentManager()
        self.logger = EventLogger(
            enabled=True,
            log_level=specification.get("log_level", "normal"),
        )
        self.action_processor = ActionProcessor(
            self.agent_manager, self.environment, self.logger,
            behavior_resolver=self.get_behavior,
            rng=self.rng,
        )

        self._create_agents()

        # Global rules: declarative rules applied to every living agent each
        # step (after behaviors). Replaces hardcoded world laws — e.g. death
        # at zero energy is now spec data:
        #   "global_rules": [{"when": {"<=": ["$energy", 0]},
        #                     "do": [{"type": "die", "cause": "energy_depleted"}]}]
        self.global_rules = None
        if specification.get("global_rules"):
            from .rules import RuleBehavior
            self.global_rules = RuleBehavior(
                specification["global_rules"], self, agent_type="<global_rules>"
            )

        # Optional systems (attached after init)
        self.intervention_engine = None
        self.llm_engine = None

        self.metrics = {
            "initial_agents": self.agent_manager.get_total_count(),
            "initial_counts": self.agent_manager.get_counts(),
            "history": [],
        }

    def add_collector(self, collector) -> None:
        """Attach a DataCollector to this simulation."""
        self.collectors.append(collector)

    def add_observer(self, observer) -> None:
        """Attach an Observer (also registers as collector)."""
        self.collectors.append(observer)

    def attach_intervention_engine(self, engine) -> None:
        """Attach an InterventionEngine for mid-simulation modifications."""
        self.intervention_engine = engine

    def attach_llm_engine(self, engine) -> None:
        """Attach an LLMBehaviorEngine for LLM-powered agents."""
        self.llm_engine = engine

    @staticmethod
    def _validate_spec(spec: Dict[str, Any]) -> None:
        """Validate specification with clear error messages."""
        if not isinstance(spec, dict):
            raise ValueError(f"Specification must be a dict, got {type(spec).__name__}")

        if "environment" not in spec:
            raise ValueError(
                "Specification missing 'environment'. Expected: "
                '{"environment": {"type": "grid_2d", "dimensions": {"width": N, "height": N}}, "agent_types": {...}}'
            )

        env = spec["environment"]
        if "type" not in env:
            raise ValueError("environment missing 'type'. Options: 'grid_2d', 'continuous_2d', 'network'")
        if "dimensions" not in env:
            raise ValueError("environment missing 'dimensions'. Expected: {'width': N, 'height': N}")

        if "agent_types" not in spec:
            raise ValueError(
                "Specification missing 'agent_types'. Expected: "
                '{"agent_types": {"name": {"initial_count": N, "initial_state": {...}, "behavior_code": "..."}}}'
            )

        agent_types = spec["agent_types"]
        if not agent_types:
            raise ValueError("agent_types is empty — define at least one agent type")

        for name, config in agent_types.items():
            if not isinstance(config, dict):
                raise ValueError(f"agent_types['{name}'] must be a dict")
            if "initial_count" not in config:
                raise ValueError(f"agent_types['{name}'] missing 'initial_count'")
            count = config["initial_count"]
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"agent_types['{name}'].initial_count must be a non-negative integer, got {count}")

    def _create_environment(self) -> Environment:
        env_spec = self.spec.get("environment", {})
        return Environment.from_dict(env_spec, rng=self.rng)

    def _create_agents(self):
        agent_types = self.spec.get("agent_types", {})

        for agent_type, type_spec in agent_types.items():
            initial_count = type_spec.get("initial_count", 0)
            initial_state = type_spec.get("initial_state", {})
            behavior_func = self.get_behavior(agent_type)

            for _ in range(initial_count):
                agent = Agent(
                    agent_type=agent_type,
                    initial_state=copy.deepcopy(initial_state),
                    behavior_function=behavior_func,
                )
                if agent.state.get("position") is None:
                    agent.state["position"] = self.environment.get_random_position()
                self.agent_manager.add_agent(agent)

    def get_behavior(self, agent_type: str) -> Optional[Callable]:
        """Resolve a behavior function for an agent type from the spec, cached.

        Resolution order:
          1. ``behavior`` — declarative rules (preferred; pure JSON data)
          2. ``behavior_code`` — Python source string (legacy escape hatch)
        """
        if agent_type in self._behavior_cache:
            return self._behavior_cache[agent_type]
        type_spec = self.spec.get("agent_types", {}).get(agent_type, {})

        behavior_spec = type_spec.get("behavior")
        behavior_code = type_spec.get("behavior_code", "")

        if isinstance(behavior_spec, dict) and "rules" in behavior_spec:
            from .rules import RuleBehavior
            func = RuleBehavior(behavior_spec["rules"], self,
                                agent_type=agent_type)
        elif behavior_code:
            func = self._compile_behavior_function(
                agent_type, behavior_code, rng=self.rng
            )
        else:
            func = None

        self._behavior_cache[agent_type] = func
        return func

    @staticmethod
    def _compile_behavior_function(
        agent_type: str, behavior_code: str, rng=None
    ) -> Optional[Callable]:
        try:
            import random
            import math

            namespace = {
                "__builtins__": {
                    "abs": abs, "len": len, "max": max, "min": min,
                    "sum": sum, "range": range, "enumerate": enumerate,
                    "list": list, "dict": dict, "str": str, "int": int,
                    "float": float, "bool": bool, "any": any, "all": all,
                    "sorted": sorted, "reversed": reversed, "round": round,
                    "zip": zip, "map": map, "filter": filter, "tuple": tuple,
                    "set": set, "True": True, "False": False, "None": None,
                    "isinstance": isinstance, "print": print,
                },
                # A seeded Random instance is a drop-in for the module API
                "random": rng if rng is not None else random,
                "math": math,
            }

            exec(behavior_code, namespace)

            func_name = f"{agent_type}_behavior"
            if func_name in namespace:
                return namespace[func_name]

            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_") and name not in ("random", "math"):
                    return obj

            return None

        except Exception as e:
            raise ValueError(f"Error compiling behavior for {agent_type}: {e}")

    def run_step(self):
        """Execute one simulation step."""
        self.step += 1
        self.environment.update(self.step)

        # Apply queued interventions from previous cycle
        if self.intervention_engine:
            self.intervention_engine.apply_pending()

        # Rebuild spatial index for fast proximity queries
        self.agent_manager.rebuild_spatial_index()

        # Pre-compute LLM agent decisions in batch
        if self.llm_engine:
            self.llm_engine.prepare_batch(self)

        agents = self.scheduler.get_agents(self.agent_manager)
        simultaneous = getattr(self.scheduler, "simultaneous", False)

        if simultaneous:
            # Collect all actions first, then process
            all_actions = []
            for agent in agents:
                if not agent.alive:
                    continue
                actions = self._get_agent_actions(agent)
                if actions:
                    all_actions.append((agent, actions))
            for agent, actions in all_actions:
                if agent.alive:
                    self.action_processor.process_actions(agent, actions, self.step)
        else:
            for agent in agents:
                if not agent.alive:
                    continue
                actions = self._get_agent_actions(agent)
                if actions:
                    self.action_processor.process_actions(agent, actions, self.step)

        self._apply_global_rules()
        self.agent_manager.cleanup_dead_agents()
        self._record_metrics()

        # Run collectors
        for collector in self.collectors:
            collector.collect(self)

    def _get_agent_actions(self, agent: Agent) -> List[Dict[str, Any]]:
        perception_radius = agent.get_attribute("perception_radius", 5)
        agents_nearby = self.agent_manager.get_agents_near_agent(
            agent, perception_radius, self.environment
        )
        sim_state = {
            "step": self.step,
            "environment": self.environment.to_dict(),
            "agent_counts": self.agent_manager.get_counts(),
        }
        return agent.execute_behavior(sim_state, agents_nearby)

    def _apply_global_rules(self):
        """Apply spec-level global rules to every living agent.

        Global rules see the agent's own state ("$attr"), "@step", and
        global counts ({"total": type}) — not neighbors.
        """
        if not self.global_rules:
            return
        sim_state = {
            "step": self.step,
            "agent_counts": self.agent_manager.get_counts(),
        }
        for agent in self.agent_manager.get_living_agents():
            actions = self.global_rules(agent, sim_state, [])
            if actions:
                self.action_processor.process_actions(agent, actions, self.step)

    def _record_metrics(self):
        counts = self.agent_manager.get_counts()
        total = self.agent_manager.get_total_count()
        self.metrics["history"].append({
            "step": self.step,
            "agent_counts": counts,
            "total_agents": total,
        })

    def run(self, steps: int) -> Dict[str, Any]:
        """Run simulation for N steps. Returns results dict."""
        start_time = time.time()

        for _ in range(steps):
            self.run_step()
            if self.agent_manager.get_total_count() == 0:
                break

        return {
            "spec": self.spec,
            "seed": self.seed,
            "final_step": self.step,
            "duration": time.time() - start_time,
            "metrics": self.metrics,
            "summary": {
                "initial_agents": self.metrics["initial_agents"],
                "initial_counts": self.metrics["initial_counts"],
                "final_agents": self.agent_manager.get_total_count(),
                "final_counts": self.agent_manager.get_counts(),
            },
            "events": self.logger.export_json(),
            "event_summary": self.logger.get_summary(),
        }

    def run_stream(self, steps: int, delay: float = 0.1, callback: Optional[Callable] = None):
        """Run simulation yielding state each step."""
        for _ in range(steps):
            self.run_step()

            state = {
                "step": self.step,
                "agent_counts": self.agent_manager.get_counts(),
                "total_agents": self.agent_manager.get_total_count(),
                "environment": self.environment.properties,
            }

            if callback:
                callback(state)
            yield state

            if self.agent_manager.get_total_count() == 0:
                break
            if delay > 0:
                time.sleep(delay)

    def get_state(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "environment": self.environment.to_dict(),
            "agents": self.agent_manager.to_dict(),
            "metrics": self.metrics,
        }

    def save(self, path: str) -> None:
        """Save simulation state to a JSON file for later resuming."""
        import json
        checkpoint = {
            "spec": self.spec,
            "step": self.step,
            "agents": [
                {"type": a.type, "alive": a.alive, "state": a.state}
                for a in self.agent_manager.agents
            ],
            "environment_properties": self.environment.properties,
            "metrics": self.metrics,
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Simulation":
        """Load a saved simulation and resume from where it left off."""
        import json
        import copy
        from .agent import Agent

        with open(path) as f:
            checkpoint = json.load(f)

        sim = cls(checkpoint["spec"])

        # Restore step counter
        sim.step = checkpoint["step"]

        # Restore agents from checkpoint (replacing the ones __init__ created)
        sim.agent_manager.reset()

        for agent_data in checkpoint["agents"]:
            agent_type = agent_data["type"]
            agent = Agent(
                agent_type=agent_type,
                initial_state=copy.deepcopy(agent_data["state"]),
                behavior_function=sim.get_behavior(agent_type),
            )
            agent.alive = agent_data["alive"]
            sim.agent_manager.add_agent(agent)

        # Restore environment properties
        for k, v in checkpoint.get("environment_properties", {}).items():
            sim.environment.set_property(k, v)

        # Restore metrics
        sim.metrics = checkpoint["metrics"]

        return sim
