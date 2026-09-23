"""
AI Steerer — LLM-controlled simulation tuning.

Observes the simulation periodically, sends state to an LLM,
and applies parameter-level interventions. The LLM acts like
a researcher turning knobs mid-experiment.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from ..core.observer import Observer, SimulationSnapshot
from ..core.intervention import InterventionEngine

logger = logging.getLogger("agentstan")

STEERER_SYSTEM_PROMPT = """You are controlling an agent-based simulation. Your job is to observe the state and make parameter adjustments to achieve a goal. You are a careful researcher — only intervene when needed.

Available actions (return as JSON):
- {"action": "add_agents", "agent_type": "name", "count": N} — add N new agents
- {"action": "remove_agents", "agent_type": "name", "count": N} — remove N agents
- {"action": "modify_agent_type", "agent_type": "name", "attribute": "attr", "delta": N} — adjust attribute for all agents of type
- {"action": "modify_environment", "property": "name", "value": V} — change environment property
- {"action": "none"} — do nothing (this is often the right choice)

Return a JSON object: {"interventions": [...], "reasoning": "one sentence explaining why"}

Rules:
- Maximum {max_interventions} interventions per check
- Only adjust parameters — you cannot rewrite behavior code
- Prefer small adjustments over large ones
- "none" is a valid and often correct choice
- Don't intervene just because you can — intervene because the goal requires it"""


class Steerer:
    """
    AI controller that observes a simulation and applies interventions.

    Usage:
        steerer = Steerer(goal="maintain stable coexistence of wolves and rabbits")
        steerer.attach(sim)
        results = sim.run(500)
        print(steerer.log)
    """

    def __init__(
        self,
        goal: str,
        check_every: int = 20,
        model: str = "gpt-5.5",
        api_key: Optional[str] = None,
        max_interventions_per_check: int = 3,
    ):
        self.goal = goal
        self.check_every = check_every
        self.model = model
        self.api_key = api_key
        self.max_interventions = max_interventions_per_check

        self.simulation = None
        self.observer = None
        self.intervention_engine = None
        self.log: List[Dict[str, Any]] = []
        self._prev_counts: Optional[Dict[str, int]] = None

    def attach(self, simulation) -> None:
        """Wire up observer and intervention engine to the simulation."""
        self.simulation = simulation

        self.observer = Observer(every_n_steps=self.check_every, max_history=50)
        self.observer.on_snapshot(self._on_snapshot)
        simulation.add_observer(self.observer)

        self.intervention_engine = InterventionEngine(simulation)
        simulation.attach_intervention_engine(self.intervention_engine)

    def _on_snapshot(self, snapshot: SimulationSnapshot) -> None:
        """Called by observer each check. Sends state to LLM, queues interventions."""
        try:
            prompt = self._build_prompt(snapshot)
            response = self._call_llm(prompt)
            interventions, reasoning = self._parse_response(response)
            self._apply_interventions(interventions, snapshot)

            self.log.append({
                "step": snapshot.step,
                "counts": snapshot.counts,
                "interventions": interventions,
                "reasoning": reasoning,
            })

            self._prev_counts = dict(snapshot.counts)

        except Exception as e:
            logger.error(f"Steerer error at step {snapshot.step}: {e}")
            self.log.append({
                "step": snapshot.step,
                "counts": snapshot.counts,
                "interventions": [],
                "reasoning": f"error: {e}",
            })

    def _build_prompt(self, snapshot: SimulationSnapshot) -> str:
        """Build compact state summary for the LLM."""
        parts = [f"Goal: {self.goal}"]
        parts.append(f"\nCurrent state (step {snapshot.step}):")
        parts.append(f"  Population: {', '.join(f'{t}={c}' for t, c in snapshot.counts.items())}")

        # Trend
        if self._prev_counts:
            trends = []
            for t, c in snapshot.counts.items():
                prev = self._prev_counts.get(t, c)
                delta = c - prev
                if delta > 0:
                    trends.append(f"{t} growing (+{delta})")
                elif delta < 0:
                    trends.append(f"{t} declining ({delta})")
                else:
                    trends.append(f"{t} stable")
            parts.append(f"  Trend (last {self.check_every} steps): {', '.join(trends)}")

        # Environment
        env_props = snapshot.environment.get("properties", {})
        if env_props:
            parts.append(f"  Environment: {json.dumps(env_props)}")

        # Previous interventions
        if self.log:
            parts.append("\nPrevious interventions:")
            for entry in self.log[-3:]:  # last 3 checks
                if entry["interventions"]:
                    actions = ", ".join(
                        f"{i.get('action', '?')} {i.get('agent_type', '')}"
                        for i in entry["interventions"]
                    )
                    parts.append(f"  Step {entry['step']}: {actions} ({entry['reasoning']})")
                else:
                    parts.append(f"  Step {entry['step']}: none")

        parts.append("\nWhat should be adjusted? Return JSON.")
        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI and return response text."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()

        system = STEERER_SYSTEM_PROMPT.format(max_interventions=self.max_interventions)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        return response.choices[0].message.content

    def _parse_response(self, response_text: str) -> tuple:
        """Parse LLM response into (interventions_list, reasoning_string)."""
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            return [], "failed to parse response"

        interventions = data.get("interventions", [])
        reasoning = data.get("reasoning", "")

        # Enforce limit
        if len(interventions) > self.max_interventions:
            interventions = interventions[:self.max_interventions]

        # Filter out "none" actions
        interventions = [i for i in interventions if i.get("action") != "none"]

        return interventions, reasoning

    def _apply_interventions(self, interventions: List[Dict], snapshot: SimulationSnapshot) -> None:
        """Map parsed interventions to InterventionEngine calls."""
        for intervention in interventions:
            action = intervention.get("action")

            if action == "add_agents":
                agent_type = intervention.get("agent_type")
                count = min(intervention.get("count", 1), 20)  # cap at 20
                if not agent_type:
                    continue
                if agent_type not in self.simulation.spec.get("agent_types", {}):
                    continue
                # New agents take the type's initial_state and behavior
                for _ in range(count):
                    self.intervention_engine.add_agent(agent_type, {}, source="steerer")

            elif action == "remove_agents":
                agent_type = intervention.get("agent_type")
                count = min(intervention.get("count", 1), 20)
                if not agent_type:
                    continue
                agents = self.simulation.agent_manager.get_agents_by_type(agent_type)
                to_remove = self.simulation.rng.sample(agents, min(count, len(agents)))
                for agent in to_remove:
                    self.intervention_engine.remove_agent(agent.id, source="steerer")

            elif action == "modify_agent_type":
                agent_type = intervention.get("agent_type")
                attribute = intervention.get("attribute")
                delta = intervention.get("delta")
                if not agent_type or not attribute or delta is None:
                    continue
                for agent in self.simulation.agent_manager.get_agents_by_type(agent_type):
                    self.intervention_engine.modify_agent(
                        agent.id, attribute, delta=delta, source="steerer"
                    )

            elif action == "modify_environment":
                prop = intervention.get("property")
                value = intervention.get("value")
                if prop is not None and value is not None:
                    self.intervention_engine.modify_environment(prop, value, source="steerer")
