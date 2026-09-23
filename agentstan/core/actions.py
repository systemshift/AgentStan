"""
Action processor for handling agent actions and interactions
"""

import random
import copy
from typing import Dict, Any, List, Optional, Callable
from .agent import Agent, AgentManager
from .environment import Environment
from .logger import EventLogger


class ActionProcessor:
    """
    Processes actions returned by agent behavior functions
    """

    def __init__(self, agent_manager: AgentManager, environment: Environment,
                 logger: EventLogger,
                 behavior_resolver: Optional[Callable[[str], Optional[Callable]]] = None,
                 rng=None, globals: Optional[Dict[str, Any]] = None,
                 spawner: Optional[Callable[[str, Dict[str, Any]], Agent]] = None,
                 type_defaults: Optional[Callable[[str], Dict[str, Any]]] = None):
        """
        Initialize action processor

        Args:
            agent_manager: Agent management system
            environment: Environment system
            logger: Event logger
            behavior_resolver: Optional callable agent_type -> behavior_function,
                used by the transform action to look up behavior from the spec.
            rng: random.Random instance for determinism (defaults to the
                global random module)
            globals: the simulation's shared globals dict (modify_global)
            spawner: callable (agent_type, state_overrides) -> new Agent
                built from the spec, not yet added (spawn)
            type_defaults: callable agent_type -> a copy of that type's
                initial_state (transform fills missing attributes from it)
        """
        self.agent_manager = agent_manager
        self.environment = environment
        self.logger = logger
        self.behavior_resolver = behavior_resolver
        self.rng = rng if rng is not None else random
        self.globals = globals if globals is not None else {}
        self.spawner = spawner
        self.type_defaults = type_defaults

    def process_actions(self, agent: Optional[Agent], actions: List[Dict[str, Any]],
                        step: int):
        """
        Process all actions for an agent

        Args:
            agent: The agent taking actions, or None for world actions
                (world rules may only modify_global and spawn)
            actions: List of action dictionaries
            step: Current simulation step
        """
        for action in actions:
            action_type = action.get("type")

            if action_type == "modify_global":
                self._process_modify_global(action, step)
            elif action_type == "spawn":
                self._process_spawn(agent, action, step)
            elif agent is None:
                continue
            elif action_type == "move":
                self._process_move(agent, action, step)
            elif action_type == "move_to":
                self._process_move_to(agent, action, step)
            elif action_type == "move_random":
                self._process_move_random(agent, step)
            elif action_type == "interact":
                self._process_interact(agent, action, step)
            elif action_type == "reproduce":
                self._process_reproduce(agent, action, step)
            elif action_type == "die":
                self._process_die(agent, action, step)
            elif action_type == "modify_state":
                self._process_modify_state(agent, action, step)
            elif action_type == "transform":
                self._process_transform(agent, action, step)
            elif action_type == "custom":
                self._process_custom(agent, action, step)
            else:
                # Unknown action type, log it
                self.logger.log_agent_action(
                    step=step,
                    agent_id=agent.id,
                    agent_type=agent.type,
                    action_type=action_type or "unknown",
                    details={"error": "Unknown action type"}
                )

    def _process_move(self, agent: Agent, action: Dict, step: int):
        """Process movement by direction"""
        direction = action.get("direction", [0, 0])
        current_pos = agent.state.get("position")

        if current_pos is None:
            return

        # Calculate new position
        if self.environment.env_type == "grid_2d":
            x, y = current_pos
            dx, dy = direction if isinstance(direction, (list, tuple)) else (0, 0)
            new_pos = (x + dx, y + dy)
            new_pos = self.environment.normalize_position(new_pos)

        elif self.environment.env_type == "continuous_2d":
            x, y = current_pos
            dx, dy = direction if isinstance(direction, (list, tuple)) else (0, 0)
            new_pos = (x + dx, y + dy)
            if self.environment.bounded:
                new_pos = self.environment.normalize_position(new_pos)

        else:
            # For other environment types, movement is handled differently
            return

        # Update position
        old_pos = agent.state["position"]
        agent.state["position"] = new_pos

        # Log the action
        self.logger.log_agent_action(
            step=step,
            agent_id=agent.id,
            agent_type=agent.type,
            action_type="move",
            details={"from": old_pos, "to": new_pos, "direction": direction}
        )

    def _process_move_to(self, agent: Agent, action: Dict, step: int):
        """Process movement to specific target position"""
        target = action.get("target")

        if target is None:
            return

        current_pos = agent.state.get("position")
        if current_pos is None:
            return

        # Move one step toward target
        if self.environment.env_type in ["grid_2d", "continuous_2d"]:
            x1, y1 = current_pos
            x2, y2 = target

            # Calculate direction
            dx = 1 if x2 > x1 else (-1 if x2 < x1 else 0)
            dy = 1 if y2 > y1 else (-1 if y2 < y1 else 0)

            new_pos = (x1 + dx, y1 + dy)
            new_pos = self.environment.normalize_position(new_pos)

            old_pos = agent.state["position"]
            agent.state["position"] = new_pos

            self.logger.log_agent_action(
                step=step,
                agent_id=agent.id,
                agent_type=agent.type,
                action_type="move_to",
                details={"from": old_pos, "to": new_pos, "target": target}
            )

    def _process_move_random(self, agent: Agent, step: int):
        """Process random movement"""
        current_pos = agent.state.get("position")
        if current_pos is None:
            return

        # Get random neighboring position
        neighbors = self.environment.get_neighbors(current_pos, radius=1)
        if not neighbors:
            return

        new_pos = self.rng.choice(neighbors)
        old_pos = agent.state["position"]
        agent.state["position"] = new_pos

        self.logger.log_agent_action(
            step=step,
            agent_id=agent.id,
            agent_type=agent.type,
            action_type="move_random",
            details={"from": old_pos, "to": new_pos}
        )

    # Generic effect keys understood by the interact action. Any domain
    # interaction (predation, trade, infection, ...) is a combination of
    # these — the kernel knows the mechanics, never the domain.
    _INTERACT_EFFECT_KEYS = (
        "kill_target", "self_delta", "target_delta", "transfer", "exchange",
    )

    def _process_interact(self, agent: Agent, action: Dict, step: int):
        """Process interaction with another agent.

        Effects are generic and param-driven:
          - success_rate: float — probability the interaction succeeds (default 1.0)
          - kill_target: bool — target dies (death cause = interaction_type)
          - self_delta: {attr: delta, ...} — modify own attributes
          - target_delta: {attr: delta, ...} — modify target attributes
          - transfer: {"attribute": name, "amount": N} — move amount from
            self to target
          - exchange: {"give": {attr: n, ...}, "get": {attr: n, ...}} — a
            two-sided trade: self pays ``give`` to the target and receives
            ``get`` from it

        transfer and exchange are preconditions: if either side can't cover
        its amounts, the whole interaction fails and no effect applies.

        Example — predation is just a combination of generic effects:
          {"type": "interact", "target_id": ID, "interaction_type": "predation",
           "params": {"success_rate": 0.4, "kill_target": True,
                      "self_delta": {"energy": 12}}}

        Legacy aliases (deprecated): interaction_type "predation" with
        {success_rate, energy_gain} and "transfer_energy" with {amount}
        are mapped onto the generic effects above.
        """
        target_id = action.get("target_id")
        interaction_type = action.get("interaction_type", "generic")
        params = action.get("params", {})

        target = self.agent_manager.get_agent(target_id)
        if target is None or not target.alive:
            return

        effects = self._resolve_interaction_effects(interaction_type, params)

        # Preconditions: a trade that can't be paid for doesn't happen at all.
        # Checked before the success roll so a failed trade burns no RNG.
        shortfall = self._shortfall(agent, target, effects)
        if shortfall:
            self.logger.log_interaction(
                step=step, agent_ids=[agent.id, target.id],
                interaction_type=interaction_type, outcome="failure",
                details={**params, "reason": shortfall},
            )
            return

        success = True
        if "success_rate" in effects:
            success = self.rng.random() < effects["success_rate"]

        self.logger.log_interaction(
            step=step,
            agent_ids=[agent.id, target.id],
            interaction_type=interaction_type,
            outcome="success" if success else "failure",
            details=params
        )

        if not success:
            return

        if effects.get("kill_target"):
            target.kill()
            self.logger.log_agent_death(
                step=step,
                agent_id=target.id,
                agent_type=target.type,
                cause=interaction_type,
            )

        for attr, delta in (effects.get("self_delta") or {}).items():
            old = agent.get_attribute(attr, 0)
            agent.modify_attribute(attr, delta)
            self.logger.log_state_change(
                step=step, agent_id=agent.id, attribute=attr,
                old_value=old, new_value=agent.get_attribute(attr),
                cause=interaction_type,
            )

        for attr, delta in (effects.get("target_delta") or {}).items():
            target.modify_attribute(attr, delta)

        transfer = effects.get("transfer")
        if transfer:
            attr = transfer.get("attribute", "energy")
            amount = transfer.get("amount", 0)
            agent.modify_attribute(attr, -amount)
            target.modify_attribute(attr, amount)

        exchange = effects.get("exchange")
        if exchange:
            for attr, amount in (exchange.get("give") or {}).items():
                agent.modify_attribute(attr, -amount)
                target.modify_attribute(attr, amount)
            for attr, amount in (exchange.get("get") or {}).items():
                target.modify_attribute(attr, -amount)
                agent.modify_attribute(attr, amount)

    @staticmethod
    def _shortfall(agent: Agent, target: Agent, effects: Dict) -> Optional[str]:
        """Why the agents can't cover a transfer/exchange, or None."""
        owed = []  # (payer, attribute, amount)
        transfer = effects.get("transfer")
        if transfer:
            owed.append((agent, transfer.get("attribute", "energy"),
                         transfer.get("amount", 0)))
        exchange = effects.get("exchange") or {}
        owed += [(agent, a, n) for a, n in (exchange.get("give") or {}).items()]
        owed += [(target, a, n) for a, n in (exchange.get("get") or {}).items()]
        for payer, attr, amount in owed:
            if (payer.get_attribute(attr) or 0) < amount:
                side = "self" if payer is agent else "target"
                return f"{side} lacks {amount} {attr}"
        return None

    def _resolve_interaction_effects(self, interaction_type: str,
                                     params: Dict) -> Dict:
        """Map legacy ecology aliases onto generic effects."""
        if any(k in params for k in self._INTERACT_EFFECT_KEYS):
            return params
        if interaction_type == "predation":
            return {
                "success_rate": params.get("success_rate", 0.5),
                "kill_target": True,
                "self_delta": {"energy": params.get("energy_gain", 10)},
            }
        if interaction_type == "transfer_energy":
            return {
                "transfer": {"attribute": "energy",
                             "amount": params.get("amount", 5)},
            }
        return params

    def _process_reproduce(self, agent: Agent, action: Dict, step: int):
        """Process reproduction.

        Generic form:
          {"type": "reproduce", "cost": {"attribute": "biomass", "amount": 10},
           "offspring_count": 1, "offspring_state": {...}}

        Each offspring is a clone of the parent at the parent's position.
        The cost is paid per offspring and is what that offspring starts
        with, so reproduction conserves the cost attribute. Then
        ``offspring_state`` overrides are applied. ``energy_cost`` (legacy)
        is shorthand for a cost on the "energy" attribute.
        """
        offspring_count = action.get("offspring_count", 1)

        cost = action.get("cost")
        if cost is None:
            cost = {"attribute": "energy", "amount": action.get("energy_cost", 10)}
        cost_attr = cost.get("attribute", "energy")
        cost_amount = cost.get("amount", 0)

        # Check the parent can afford every offspring
        if (agent.get_attribute(cost_attr) or 0) < cost_amount * offspring_count:
            return

        for _ in range(offspring_count):
            offspring = agent.clone()
            offspring.set_attribute(cost_attr, cost_amount)
            if "age" in offspring.state:
                offspring.set_attribute("age", 0)
            for key, value in action.get("offspring_state", {}).items():
                offspring.set_attribute(key, copy.deepcopy(value))

            # Place offspring at parent's position
            offspring.state["position"] = agent.state.get("position")

            self.agent_manager.add_agent(offspring)
            agent.modify_attribute(cost_attr, -cost_amount)

            self.logger.log_agent_birth(
                step=step,
                parent_id=agent.id,
                child_id=offspring.id,
                agent_type=offspring.type
            )

    def _process_spawn(self, agent: Optional[Agent], action: Dict, step: int):
        """Create new agents of a spec-defined type.

          {"type": "spawn", "agent_type": "player", "count": 3,
           "state": {"gold": 50}}

        New agents start from the type's initial_state, with ``state``
        overrides applied, at a random position (or their state's). Unlike
        reproduce, spawn costs nothing: it is how models express inflows —
        new players joining, arrivals, immigration.
        """
        if self.spawner is None:
            return
        count = int(action.get("count", 1))
        for _ in range(max(count, 0)):
            new_agent = self.spawner(action["agent_type"], action.get("state") or {})
            self.agent_manager.add_agent(new_agent)
            self.logger.log_agent_birth(
                step=step,
                parent_id=agent.id if agent is not None else None,
                child_id=new_agent.id,
                agent_type=new_agent.type,
            )

    def _process_modify_global(self, action: Dict, step: int):
        """Set or change a shared global: {"name": g, "value"|"delta": x}."""
        name = action.get("name")
        if name is None:
            return
        old = self.globals.get(name)
        if action.get("value") is not None:
            self.globals[name] = action["value"]
        elif action.get("delta") is not None:
            self.globals[name] = (old or 0) + action["delta"]
        else:
            return
        self.logger.log_global_change(step, name, old, self.globals[name])

    def _process_die(self, agent: Agent, action: Dict, step: int):
        """Process agent death"""
        cause = action.get("cause", "voluntary")

        agent.kill()

        self.logger.log_agent_death(
            step=step,
            agent_id=agent.id,
            agent_type=agent.type,
            cause=cause
        )

    def _process_modify_state(self, agent: Agent, action: Dict, step: int):
        """Process modification of agent state"""
        attribute = action.get("attribute")
        value = action.get("value")
        delta = action.get("delta")

        if attribute is None:
            return

        old_value = agent.get_attribute(attribute)

        if value is not None:
            agent.set_attribute(attribute, value)
            new_value = value
        elif delta is not None:
            agent.modify_attribute(attribute, delta)
            new_value = agent.get_attribute(attribute)
        else:
            return

        self.logger.log_state_change(
            step=step,
            agent_id=agent.id,
            attribute=attribute,
            old_value=old_value,
            new_value=new_value,
            cause="modify_state_action"
        )

    def _process_transform(self, agent: Agent, action: Dict, step: int):
        """Transform agent into a different type.

        Action format:
            {"type": "transform", "new_type": "infected",
             "new_state": {"days_infected": 0}}

        The new agent keeps the old agent's state (position, wealth, ...);
        attributes it lacks are filled from the new type's initial_state, and
        ``new_state`` overrides both. So an agent that becomes "infected"
        gains the infected type's defaults (e.g. recovery_time) without
        every transform having to repeat them.

        Behavior for ``new_type`` is resolved from the simulation spec via
        the ``behavior_resolver`` passed to ``ActionProcessor``. An action may
        override that with an explicit ``behavior_code`` (rare, dynamic case).
        """
        new_type = action.get("new_type")
        new_state = action.get("new_state", {})

        if not new_type:
            return

        old_type = agent.type
        old_id = agent.id

        # Snapshot state before mutation
        merged_state = self.type_defaults(new_type) if self.type_defaults else {}
        merged_state.update(copy.deepcopy(agent.state))
        merged_state.update(new_state)

        # Resolve behavior: action override > spec resolver
        behavior_func = None
        behavior_code = action.get("behavior_code", "")
        if behavior_code:
            from .simulation import Simulation
            behavior_func = Simulation._compile_behavior_function(
                new_type, behavior_code, rng=self.rng
            )
        elif self.behavior_resolver is not None:
            behavior_func = self.behavior_resolver(new_type)

        # Kill old agent (after state snapshot)
        agent.kill()
        self.logger.log_agent_death(
            step=step, agent_id=old_id,
            agent_type=old_type, cause=f"transformed_to_{new_type}",
        )

        new_agent = Agent(
            agent_type=new_type,
            initial_state=merged_state,
            behavior_function=behavior_func,
        )
        self.agent_manager.add_agent(new_agent)

        self.logger.log_agent_birth(
            step=step, parent_id=old_id,
            child_id=new_agent.id, agent_type=new_type,
        )

    def _process_custom(self, agent: Agent, action: Dict, step: int):
        """Process custom action"""
        self.logger.log_agent_action(
            step=step,
            agent_id=agent.id,
            agent_type=agent.type,
            action_type="custom",
            details=action.get("details", {})
        )
