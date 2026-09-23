"""
LLM-as-Agent: agents that think via LLM calls.

The LLMBehaviorEngine creates behavior functions backed by LLM.
Supports batched parallel execution and per-agent memory.
"""

import json
import re
from .llm import make_client, resolve_model
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from .llm_memory import AgentMemory
from .llm_budget import LLMBudget


AGENT_SYSTEM_PROMPT = """You are an agent in a simulation. Each step you observe your surroundings and choose actions.

Return a JSON array of actions. Available actions:
- {"type": "move", "direction": [dx, dy]} — move relative
- {"type": "move_to", "target": [x, y]} — move toward position
- {"type": "move_random"} — random step
- {"type": "modify_state", "attribute": "energy", "delta": 5} — change your state
- {"type": "interact", "target_id": ID, "interaction_type": "any_label", "params": {...}}
  params (all optional): "success_rate": 0..1, "kill_target": true,
  "self_delta": {"attr": change}, "target_delta": {"attr": change},
  "transfer": {"attribute": "gold", "amount": 5},
  "exchange": {"give": {"gold": 10}, "get": {"item": 1}}
- {"type": "reproduce", "cost": {"attribute": "energy", "amount": 15}}
- {"type": "die", "cause": "reason"}

Return ONLY a JSON object with an "actions" key containing the array. Example:
{"actions": [{"type": "move", "direction": [1, 0]}, {"type": "modify_state", "attribute": "energy", "delta": -1}]}"""


def build_agent_prompt(agent, sim_state, agents_nearby, memory: AgentMemory,
                       personality: str = "", goals: str = "") -> str:
    """Build the per-step decision prompt for one LLM agent."""
    parts = []

    # Identity
    parts.append(f"You are {agent.type} #{agent.id}.")
    if personality:
        parts.append(f"Personality: {personality}")
    if goals:
        parts.append(f"Goals: {goals}")

    # Current state
    state_items = []
    for k, v in agent.state.items():
        if k.startswith("_"):
            continue
        state_items.append(f"{k}={v}")
    parts.append(f"\nCurrent state: {', '.join(state_items)}")
    parts.append(f"Step: {sim_state['step']}")

    # Nearby agents
    if agents_nearby:
        nearby_lines = []
        for other in agents_nearby[:10]:  # cap at 10 to control tokens
            pos = other.get_attribute("position")
            energy = other.get_attribute("energy")
            info = f"  {other.type} #{other.id}"
            if pos:
                info += f" at {pos}"
            if energy is not None:
                info += f" (energy: {round(energy, 1)})"
            nearby_lines.append(info)
        parts.append("\nNearby agents:\n" + "\n".join(nearby_lines))
    else:
        parts.append("\nNo agents nearby.")

    # Memory
    memory_text = memory.to_prompt_text()
    if memory_text != "No memories yet.":
        parts.append(f"\nYour memory:\n{memory_text}")

    parts.append("\nWhat do you do? Return a JSON object with an 'actions' array.")

    return "\n".join(parts)


def parse_llm_actions(response_text: str) -> List[Dict[str, Any]]:
    """Parse LLM response into action dicts. Robust against formatting issues."""
    text = response_text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "actions" in data:
            return data["actions"]
        return [data]
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if "actions" in data:
                return data["actions"]
            return [data]
        except json.JSONDecodeError:
            pass

    return [{"type": "move_random"}]


def _summarize_perception(agent, agents_nearby) -> str:
    """One-line summary of what the agent perceived."""
    if not agents_nearby:
        return "no agents nearby"
    types = {}
    for a in agents_nearby:
        types[a.type] = types.get(a.type, 0) + 1
    parts = [f"{count} {t}" for t, count in types.items()]
    return f"saw {', '.join(parts)} nearby"


def _summarize_actions(actions: List[Dict]) -> str:
    """One-line summary of actions taken."""
    if not actions:
        return "did nothing"
    types = [a.get("type", "?") for a in actions]
    return ", ".join(types)


class LLMBehaviorEngine:
    """
    Manages LLM-based agent behavior with batched execution.

    Usage:
        engine = LLMBehaviorEngine(budget=LLMBudget(max_calls_total=500))
        behavior = engine.create_behavior(personality="aggressive hunter", goals="hunt prey")
        for wolf in sim.agent_manager.get_agents_by_type("wolf"):
            wolf.behavior_function = behavior
        sim.attach_llm_engine(engine)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        budget: Optional[LLMBudget] = None,
        max_concurrent: int = 10,
        client=None,
    ):
        self.model = resolve_model(model)
        self.client = client
        self.api_key = api_key
        self.base_url = base_url
        self.budget = budget or LLMBudget()
        self.max_concurrent = max_concurrent
        self._cache: Dict[int, List[Dict]] = {}
        self._llm_agents: Dict[int, Dict[str, str]] = {}  # agent_id -> {personality, goals}

    def create_behavior(
        self,
        personality: str = "",
        goals: str = "",
        memory_size: int = 20,
        fallback: Optional[Callable] = None,
    ) -> Callable:
        """
        Factory: create a behavior function backed by LLM.

        Returns a callable with the standard behavior signature that
        checks the batch cache first, falls back to sync LLM call.
        """
        engine = self

        def llm_behavior(agent, sim_state, agents_nearby):
            # Register this agent as LLM-powered
            engine._llm_agents[agent.id] = {
                "personality": personality,
                "goals": goals,
            }

            # Init memory if needed
            if "_memory" not in agent.state:
                agent.state["_memory"] = AgentMemory(max_entries=memory_size).to_dict()

            memory = AgentMemory.from_dict(agent.state["_memory"])

            # Check batch cache first (populated by prepare_batch)
            cached = engine._cache.get(agent.id)
            if cached is not None:
                actions = cached
            elif engine.budget.can_call():
                # Fallback: sync single call
                prompt = build_agent_prompt(
                    agent, sim_state, agents_nearby, memory,
                    personality=personality, goals=goals,
                )
                actions = engine._call_llm(agent.id, prompt)
            elif fallback:
                return fallback(agent, sim_state, agents_nearby)
            else:
                actions = [{"type": "move_random"}]

            # Update memory
            obs = _summarize_perception(agent, agents_nearby)
            act = _summarize_actions(actions)
            memory.add(sim_state["step"], obs, act)
            agent.state["_memory"] = memory.to_dict()

            return actions

        return llm_behavior

    def prepare_batch(self, simulation) -> None:
        """
        Pre-compute all LLM agent decisions for the current step.
        Called by Simulation.run_step() before the agent loop.
        """
        self._cache.clear()
        self.budget.reset_step()

        requests = []
        for agent in simulation.agent_manager.get_living_agents():
            if agent.id not in self._llm_agents:
                continue

            info = self._llm_agents[agent.id]
            perception_radius = agent.get_attribute("perception_radius", 5)
            nearby = simulation.agent_manager.get_agents_near_agent(
                agent, perception_radius, simulation.environment
            )
            sim_state = {
                "step": simulation.step,
                "environment": simulation.environment.to_dict(),
                "agent_counts": simulation.agent_manager.get_counts(),
            }

            memory = AgentMemory.from_dict(agent.state.get("_memory", {}))
            prompt = build_agent_prompt(
                agent, sim_state, nearby, memory,
                personality=info["personality"], goals=info["goals"],
            )
            requests.append({"agent_id": agent.id, "prompt": prompt})

        if not requests:
            return

        # Batch execute in parallel
        results = self._batch_call(requests)
        self._cache.update(results)

    def _call_llm(self, agent_id: int, prompt: str) -> List[Dict[str, Any]]:
        """Single synchronous LLM call."""
        if self.client is None:
            # one client (and connection pool) for the whole run
            self.client = make_client(self.api_key, self.base_url)
        client = self.client

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        usage = response.usage
        if usage:
            self.budget.record_call(
                usage.prompt_tokens, usage.completion_tokens,
                self.model, agent_id,
            )

        return parse_llm_actions(response.choices[0].message.content)

    def _batch_call(self, requests: List[Dict]) -> Dict[int, List[Dict]]:
        """Send multiple LLM requests in parallel."""
        results = {}

        # Filter by budget
        allowed = []
        for req in requests:
            if self.budget.can_call():
                allowed.append(req)
            else:
                results[req["agent_id"]] = [{"type": "move_random"}]

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = {
                executor.submit(self._call_llm, req["agent_id"], req["prompt"]): req["agent_id"]
                for req in allowed
            }
            for future in futures:
                agent_id = futures[future]
                try:
                    results[agent_id] = future.result(timeout=30)
                except Exception:
                    results[agent_id] = [{"type": "move_random"}]

        return results
