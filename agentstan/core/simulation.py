"""
Core simulation engine.
"""

import copy
import time
import warnings
import random as _random_module
import logging
from typing import Dict, Any, List, Optional, Callable

from .environment import Environment
from .agent import Agent, AgentManager
from .actions import ActionProcessor
from .logger import EventLogger
from .scheduler import RandomScheduler
from .rules import (RuleBehavior, RuleError, validate_expression,
                    check_attribute_reads, validate_initial_state,
                    compile_initial_state, compile_expr, _Context)

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
        behaviors: Python behavior functions by agent type, for local
            models the rule language can't express. Each is called as
            ``fn(agent, sim_state, agents_nearby) -> list of actions``;
            use ``sim_state["rng"]`` for randomness so runs stay
            reproducible. Overrides the spec's ``behavior`` for that type.
            Not part of the spec: a model with Python behaviors isn't
            portable data.
    """

    def __init__(self, specification: Dict[str, Any], scheduler=None,
                 seed: Optional[int] = None,
                 behaviors: Optional[Dict[str, Callable]] = None):
        self._validate_spec(specification)
        self._behaviors = dict(behaviors or {})
        unknown = set(self._behaviors) - set(specification["agent_types"])
        if unknown:
            raise ValueError(
                f"behaviors given for undefined agent types {sorted(unknown)} — "
                f"defined: {sorted(specification['agent_types'])}"
            )
        self.spec = specification
        self.step = 0
        self.seed = seed if seed is not None else specification.get("seed")
        self.rng = _random_module.Random(self.seed)
        self.scheduler = scheduler or RandomScheduler()
        if getattr(self.scheduler, "rng", None) is None:
            self.scheduler.rng = self.rng
        self.collectors = []
        self._behavior_cache: Dict[str, Optional[Callable]] = {}
        self._state_builders: Dict[str, Callable] = {}
        # Shared world state, readable in rules as "@name"
        self.globals: Dict[str, Any] = copy.deepcopy(specification.get("globals") or {})

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
            globals=self.globals,
            spawner=self._new_agent,
            type_defaults=lambda t: self._initial_state(t),
        )

        # Compile every behavior up front so a bad spec fails here, not
        # mid-run (and not only for types that happen to have agents).
        for agent_type, type_spec in specification["agent_types"].items():
            self.get_behavior(agent_type)
            validate_initial_state(
                type_spec.get("initial_state", {}),
                f"agent_types['{agent_type}'].initial_state",
                globals_=set(self.globals),
            )
        if not self._behaviors:
            check_attribute_reads(specification)

        # Global rules: declarative rules applied to every living agent each
        # step (after behaviors). Replaces hardcoded world laws — e.g. death
        # at zero energy is now spec data:
        #   "global_rules": [{"when": {"<=": ["$energy", 0]},
        #                     "do": [{"type": "die", "cause": "energy_depleted"}]}]
        self.global_rules = None
        if specification.get("global_rules"):
            self.global_rules = RuleBehavior(
                specification["global_rules"], self, path="global_rules",
                can_sense=False,
            )

        # World rules: run once per step, before agents act, with no agent —
        # prices, faucets, arrivals. Only globals and aggregates are visible.
        self.world_rules = None
        if specification.get("world_rules"):
            self.world_rules = RuleBehavior(
                specification["world_rules"], self, path="world_rules",
                world=True,
            )

        # Observables: named world-level expressions recorded every step
        self.observables: Dict[str, Any] = specification.get("observables") or {}
        for name, expr in self.observables.items():
            validate_expression(
                expr, f"observables['{name}']",
                globals_=set(self.globals),
                agent_types=set(specification["agent_types"]),
            )
        self._observable_fns = {name: compile_expr(expr)
                                for name, expr in self.observables.items()}

        self._create_agents()

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

    @classmethod
    def check(cls, spec: Dict[str, Any], smoke_steps: int = 10,
              max_agents: int = 10000, time_limit: float = 5.0) -> None:
        """Validate a spec fully: construct it, then run a few guarded steps
        so runtime rule errors (e.g. arithmetic on a missing attribute)
        surface too. Raises on any problem; hitting a resource guard is not
        an error. Does not mutate ``spec``."""
        spec = copy.deepcopy(spec)
        spec.pop("steps", None)
        sim = cls(spec)
        if smoke_steps > 0:
            sim.run(smoke_steps, max_agents=max_agents, time_limit=time_limit)

    @staticmethod
    def _validate_spec(spec: Dict[str, Any]) -> None:
        """Validate specification with clear error messages."""
        if not isinstance(spec, dict):
            raise ValueError(f"Specification must be a dict, got {type(spec).__name__}")

        if "environment" not in spec:
            raise ValueError(
                "Specification missing 'environment'. Expected e.g. "
                '{"type": "none"} (markets, economies), '
                '{"type": "grid_2d", "dimensions": {"width": N, "height": N}}, '
                'or {"type": "network", "dimensions": {"node_count": N, "topology": "random"}}'
            )

        env = spec["environment"]
        if "type" not in env:
            raise ValueError(
                "environment missing 'type'. Options: 'grid_2d', "
                "'continuous_2d', 'network', 'none' (non-spatial)"
            )
        if "dimensions" not in env and env["type"] != "none":
            raise ValueError("environment missing 'dimensions'. Expected: {'width': N, 'height': N}")

        if "agent_types" not in spec:
            raise ValueError(
                "Specification missing 'agent_types'. Expected: "
                '{"agent_types": {"name": {"initial_count": N, "initial_state": {...}, "behavior_code": "..."}}}'
            )

        known = {"environment", "agent_types", "globals", "world_rules",
                 "global_rules", "observables", "seed", "steps", "metadata",
                 "log_level", "name", "description"}
        unknown = set(spec) - known
        if unknown:
            raise ValueError(
                f"Specification has unknown top-level keys {sorted(unknown)} — "
                f"allowed: {sorted(known)}"
            )

        globals_ = spec.get("globals")
        if globals_ is not None:
            if not isinstance(globals_, dict):
                raise ValueError("'globals' must be a dict of name -> initial value")
            for name, value in globals_.items():
                if name == "step":
                    raise ValueError("'step' is reserved (read it as \"@step\")")
                if not isinstance(value, (int, float, str, bool)) or value is None:
                    raise ValueError(
                        f"globals['{name}'] must be a number, string or bool, got {value!r}"
                    )
        observables = spec.get("observables")
        if observables is not None and not isinstance(observables, dict):
            raise ValueError("'observables' must be a dict of name -> expression")

        agent_types = spec["agent_types"]
        if not agent_types:
            raise ValueError("agent_types is empty — define at least one agent type")

        type_keys = {"initial_count", "initial_state", "behavior", "behavior_code",
                     "description"}
        for name, config in agent_types.items():
            if not isinstance(config, dict):
                raise ValueError(f"agent_types['{name}'] must be a dict")
            unknown = set(config) - type_keys
            if unknown:
                raise ValueError(
                    f"agent_types['{name}'] has unknown keys {sorted(unknown)} — "
                    f"allowed: {sorted(type_keys)}"
                )
            behavior = config.get("behavior")
            if behavior is not None and (not isinstance(behavior, dict)
                                         or set(behavior) != {"rules"}):
                raise ValueError(
                    f"agent_types['{name}'].behavior must be {{\"rules\": [...]}}"
                )
            if "initial_count" not in config:
                raise ValueError(f"agent_types['{name}'] missing 'initial_count'")
            count = config["initial_count"]
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"agent_types['{name}'].initial_count must be a non-negative integer, got {count}")

    def _create_environment(self) -> Environment:
        env_spec = self.spec.get("environment", {})
        return Environment.from_dict(env_spec, rng=self.rng)

    def _create_agents(self):
        for agent_type, type_spec in self.spec.get("agent_types", {}).items():
            for _ in range(type_spec.get("initial_count", 0)):
                self.agent_manager.add_agent(self._new_agent(agent_type))

    def _new_agent(self, agent_type: str,
                   overrides: Optional[Dict[str, Any]] = None) -> Agent:
        """Build (but don't add) an agent of a spec type: initial_state plus
        overrides, at a random position unless its state places it."""
        state = self._initial_state(agent_type)
        state.update(copy.deepcopy(overrides or {}))
        agent = Agent(agent_type=agent_type, initial_state=state,
                      behavior_function=self.get_behavior(agent_type))
        if agent.state.get("position") is None:
            agent.state["position"] = self.environment.get_random_position()
        return agent

    def get_behavior(self, agent_type: str) -> Optional[Callable]:
        """Resolve a behavior function for an agent type from the spec, cached.

        Resolution order:
          1. ``behaviors`` passed to the constructor — Python functions
          2. ``behavior`` — declarative rules (preferred; pure JSON data)
          3. ``behavior_code`` — Python source string (deprecated)
        """
        if agent_type in self._behavior_cache:
            return self._behavior_cache[agent_type]
        type_spec = self.spec.get("agent_types", {}).get(agent_type, {})

        behavior_spec = type_spec.get("behavior")
        behavior_code = type_spec.get("behavior_code", "")

        if agent_type in self._behaviors:
            func = self._behaviors[agent_type]
        elif isinstance(behavior_spec, dict) and "rules" in behavior_spec:
            func = RuleBehavior(behavior_spec["rules"], self,
                                agent_type=agent_type)
        elif behavior_code:
            warnings.warn(
                f"agent_types['{agent_type}'].behavior_code is deprecated: it "
                f"runs Python source with exec and isn't portable. Use "
                f"declarative rules, or pass a function via "
                f"Simulation(spec, behaviors={{'{agent_type}': fn}}).",
                DeprecationWarning, stacklevel=3,
            )
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

        if self.world_rules:
            actions = self.world_rules.decide(None)
            if actions:
                self.action_processor.process_actions(None, actions, self.step)

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
        radius = agent.get_attribute("perception_radius", 5)
        behavior = agent.behavior_function

        def nearby():
            return self.agent_manager.get_agents_near_agent(
                agent, radius, self.environment
            )

        if isinstance(behavior, RuleBehavior):
            # Rules compute neighbors only if a rule actually looks
            return behavior.decide(agent, nearby) if agent.alive else []

        sim_state = {
            "step": self.step,
            "environment": self.environment.to_dict(),
            "agent_counts": self.agent_manager.get_counts(),
            "globals": self.globals,
            "rng": self.rng,
        }
        return agent.execute_behavior(sim_state, nearby())

    def _apply_global_rules(self):
        """Apply spec-level global rules to every living agent.

        Global rules see the agent's own state ("$attr"), "@step", and
        global counts ({"total": type}) — not neighbors.
        """
        if not self.global_rules:
            return
        for agent in self.agent_manager.get_living_agents():
            actions = self.global_rules.decide(agent, [])
            if actions:
                self.action_processor.process_actions(agent, actions, self.step)

    def _record_metrics(self):
        counts = self.agent_manager.get_counts()
        total = self.agent_manager.get_total_count()
        entry = {
            "step": self.step,
            "agent_counts": counts,
            "total_agents": total,
        }
        if self.globals:
            entry["globals"] = dict(self.globals)
        if self.observables:
            entry["observables"] = self.evaluate_observables()
        self.metrics["history"].append(entry)

    def _initial_state(self, agent_type: str) -> Dict[str, Any]:
        """A fresh initial_state for one new agent of a type (expressions
        evaluated now); {} for types the spec doesn't define."""
        if agent_type not in self._state_builders:
            state = self.spec["agent_types"].get(agent_type, {}).get("initial_state", {})
            self._state_builders[agent_type] = compile_initial_state(state)
        return self._state_builders[agent_type](self._world_context())

    def _world_context(self) -> _Context:
        """Expression context with no agent: globals, aggregates, rng."""
        return _Context(agent=None, nearby=[], rng=self.rng,
                        env=self.environment, step=self.step,
                        globals=self.globals, manager=self.agent_manager,
                        can_sense=False)

    def evaluate_observables(self) -> Dict[str, Any]:
        """Current value of every spec observable."""
        from .rules import _located
        ctx = self._world_context()
        out = {}
        for name, fn in self._observable_fns.items():
            try:
                out[name] = fn(ctx)
            except (RuleError, TypeError, ValueError, ArithmeticError) as e:
                raise _located(e, f"observables['{name}']", ctx) from e
        return out

    def run(self, steps: int, max_agents: Optional[int] = None,
            time_limit: Optional[float] = None) -> Dict[str, Any]:
        """Run simulation for N steps. Returns results dict.

        Resource guards (both optional): ``max_agents`` stops the run when
        the population exceeds it (runaway reproduction), ``time_limit``
        stops after that many wall-clock seconds. An early stop is reported
        in results["stopped"] = {"reason", "at_step"} — partial results are
        still returned in full.
        """
        start_time = time.time()
        stopped = None

        for _ in range(steps):
            self.run_step()
            total = self.agent_manager.get_total_count()
            if total == 0 and not self.world_rules:
                break
            if max_agents is not None and total > max_agents:
                stopped = {"reason": "max_agents", "at_step": self.step,
                           "agents": total}
                break
            if time_limit is not None and time.time() - start_time > time_limit:
                stopped = {"reason": "time_limit", "at_step": self.step}
                break

        return {
            "spec": self.spec,
            "seed": self.seed,
            "globals": dict(self.globals),
            "stopped": stopped,
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
                "globals": dict(self.globals),
            }

            if callback:
                callback(state)
            yield state

            if self.agent_manager.get_total_count() == 0 and not self.world_rules:
                break
            if delay > 0:
                time.sleep(delay)

    def get_state(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "environment": self.environment.to_dict(),
            "agents": self.agent_manager.to_dict(),
            "globals": dict(self.globals),
            "metrics": self.metrics,
        }

    def save(self, path: str) -> None:
        """Save simulation state to a JSON file for later resuming.

        The checkpoint includes the RNG state and agent IDs, so a loaded
        simulation continues exactly as the original would have.
        """
        import json
        version, internal, gauss = self.rng.getstate()
        checkpoint = {
            "spec": self.spec,
            "seed": self.seed,
            "step": self.step,
            "rng_state": [version, list(internal), gauss],
            "next_id": self.agent_manager._next_id,
            "agents": [
                {"id": a.id, "type": a.type, "alive": a.alive, "state": a.state}
                for a in self.agent_manager.agents
            ],
            "globals": self.globals,
            "environment_properties": self.environment.properties,
            "metrics": self.metrics,
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Simulation":
        """Load a saved simulation and resume from where it left off."""
        import json

        with open(path) as f:
            checkpoint = json.load(f)

        sim = cls(checkpoint["spec"], seed=checkpoint.get("seed"))
        sim.step = checkpoint["step"]

        # Replace the agents __init__ created with the saved ones
        sim.agent_manager.reset()
        for agent_data in checkpoint["agents"]:
            state = copy.deepcopy(agent_data["state"])
            if isinstance(state.get("position"), list):
                state["position"] = tuple(state["position"])
            agent = Agent(
                agent_type=agent_data["type"],
                initial_state=state,
                behavior_function=sim.get_behavior(agent_data["type"]),
                agent_id=agent_data.get("id"),
            )
            agent.alive = agent_data["alive"]
            sim.agent_manager.add_agent(agent)
        if "next_id" in checkpoint:
            sim.agent_manager._next_id = checkpoint["next_id"]

        sim.globals.clear()
        sim.globals.update(checkpoint.get("globals") or {})
        for k, v in checkpoint.get("environment_properties", {}).items():
            sim.environment.set_property(k, v)
        sim.metrics = checkpoint["metrics"]

        if "rng_state" in checkpoint:
            version, internal, gauss = checkpoint["rng_state"]
            sim.rng.setstate((version, tuple(internal), gauss))

        return sim
