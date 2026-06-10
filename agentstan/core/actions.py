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
                 rng=None):
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
        """
        self.agent_manager = agent_manager
        self.environment = environment
        self.logger = logger
        self.behavior_resolver = behavior_resolver
        self.rng = rng if rng is not None else random

    def process_actions(self, agent: Agent, actions: List[Dict[str, Any]], step: int):
        """
        Process all actions for an agent

        Args:
            agent: The agent taking actions
            actions: List of action dictionaries
            step: Current simulation step
        """
        for action in actions:
            action_type = action.get("type")

            if action_type == "move":
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
        "kill_target", "self_delta", "target_delta", "transfer",
    )

    def _process_interact(self, agent: Agent, action: Dict, step: int):
        """Process interaction with another agent.

        Effects are generic and param-driven:
          - success_rate: float — probability the interaction succeeds (default 1.0)
          - kill_target: bool — target dies (death cause = interaction_type)
          - self_delta: {attr: delta, ...} — modify own attributes
          - target_delta: {attr: delta, ...} — modify target attributes
          - transfer: {"attribute": name, "amount": N} — move amount from
            self to target (all-or-nothing if self lacks the amount)

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
            if agent.get_attribute(attr, 0) >= amount:
                agent.modify_attribute(attr, -amount)
                target.modify_attribute(attr, amount)

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

        The offspring is a clone of the parent at the parent's position; it
        starts with half the parent's cost attribute, then ``offspring_state``
        overrides are applied. ``energy_cost`` (legacy) is shorthand for a
        cost on the "energy" attribute.
        """
        offspring_count = action.get("offspring_count", 1)

        cost = action.get("cost")
        if cost is None:
            cost = {"attribute": "energy", "amount": action.get("energy_cost", 10)}
        cost_attr = cost.get("attribute", "energy")
        cost_amount = cost.get("amount", 0)

        # Check the parent can afford it
        if agent.get_attribute(cost_attr, 0) < cost_amount:
            return

        # Create offspring
        for _ in range(offspring_count):
            offspring = agent.clone()
            offspring.set_attribute(cost_attr, agent.get_attribute(cost_attr, 0) // 2)
            offspring.set_attribute("age", 0)
            for key, value in action.get("offspring_state", {}).items():
                offspring.set_attribute(key, copy.deepcopy(value))

            # Place offspring at parent's position
            offspring.state["position"] = agent.state.get("position")

            self.agent_manager.add_agent(offspring)

            self.logger.log_agent_birth(
                step=step,
                parent_id=agent.id,
                child_id=offspring.id,
                agent_type=offspring.type
            )

        # Deduct the cost from the parent
        agent.modify_attribute(cost_attr, -cost_amount)

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
        merged_state = copy.deepcopy(agent.state)
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
