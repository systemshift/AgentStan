"""
Chat-to-ABM: Send natural language to an LLM, get back a running simulation.
"""

import json
import re
from typing import Dict, Any, Optional

from .prompt import get_system_prompt
from ..core.simulation import Simulation


DEFAULT_MODEL = "gpt-5.5"


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown fences and preamble."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```\w*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    # Look for first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from LLM response. Response starts with: {text[:200]}"
    )


def _spec_from_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LLM response into a valid simulation specification."""
    # The LLM response should already be close to our spec format.
    # Normalize any variations.
    spec = {}

    # Metadata
    spec["metadata"] = {
        "name": data.get("name", data.get("metadata", {}).get("name", "Simulation")),
        "description": data.get("description", data.get("metadata", {}).get("description", "")),
    }

    # Environment
    if "environment" in data:
        spec["environment"] = data["environment"]
    else:
        spec["environment"] = {
            "type": "grid_2d",
            "dimensions": {"width": 40, "height": 40, "topology": "torus"},
        }

    # Ensure environment has required fields
    env = spec["environment"]
    if "type" not in env:
        env["type"] = "grid_2d"
    if "dimensions" not in env:
        env["dimensions"] = {"width": 40, "height": 40}

    # Agent types
    if "agent_types" in data:
        spec["agent_types"] = data["agent_types"]
    elif "agents" in data:
        spec["agent_types"] = data["agents"]
    else:
        raise ValueError("LLM response missing 'agent_types' or 'agents' field")

    # Validate each agent type has required fields
    for agent_type, type_spec in spec["agent_types"].items():
        if "initial_count" not in type_spec:
            type_spec["initial_count"] = 10
        if "initial_state" not in type_spec:
            type_spec["initial_state"] = {"energy": 20}
        has_rules = isinstance(type_spec.get("behavior"), dict) and "rules" in type_spec["behavior"]
        if not has_rules and "behavior_code" not in type_spec:
            raise ValueError(
                f"Agent type '{agent_type}' missing behavior — "
                f'expected "behavior": {{"rules": [...]}}'
            )

    # Top-level fields the engine understands
    for key in ("seed", "steps"):
        if key in data:
            spec[key] = data[key]

    return spec


def _make_client(api_key: Optional[str], base_url: Optional[str]):
    from openai import OpenAI

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _chat(client, model: str, messages: list) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _generate_validated(
    client, model: str, messages: list, repair_attempts: int = 1
) -> Dict[str, Any]:
    """
    Chat until the spec parses AND constructs a Simulation, feeding
    validation errors back to the LLM for repair.

    Mutates `messages` in place (so multi-turn sessions keep the history).
    """
    last_err = None
    for _ in range(1 + repair_attempts):
        raw = _chat(client, model, messages)
        messages.append({"role": "assistant", "content": raw})
        try:
            data = _extract_json(raw)
            spec = _spec_from_response(data)
            Simulation(spec)  # full engine validation, incl. rule compilation
            return spec
        except Exception as e:
            last_err = e
            messages.append({
                "role": "user",
                "content": (
                    f"That specification failed validation with this error:\n"
                    f"{e}\n"
                    f"Return the corrected, complete JSON specification."
                ),
            })
    raise ValueError(
        f"Spec failed validation after {1 + repair_attempts} attempt(s). "
        f"Last error: {last_err}"
    )


def generate(
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client=None,
    repair_attempts: int = 1,
) -> Dict[str, Any]:
    """
    Send a natural language prompt to an LLM and get back a simulation spec.

    The spec is validated by actually constructing a Simulation (including
    rule compilation); on failure the error is sent back to the LLM for
    repair, up to `repair_attempts` times.

    Args:
        user_prompt: Natural language description of the desired simulation
        model: Model name
        api_key: API key (or set OPENAI_API_KEY env var)
        base_url: Optional API base URL for compatible endpoints
        client: Optional pre-built OpenAI-compatible client (overrides
            api_key/base_url; useful for testing and alternative providers)
        repair_attempts: How many validation-error round-trips to allow

    Returns:
        Validated simulation specification dict (pure JSON data)
    """
    client = client or _make_client(api_key, base_url)
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]
    return _generate_validated(client, model, messages, repair_attempts)


def run_chat(
    user_prompt: str,
    steps: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end: natural language -> simulation spec -> run simulation -> results.

    Args:
        user_prompt: What to simulate
        steps: Override step count (default: use LLM's suggestion or 200)
        model: OpenAI model name
        api_key: OpenAI API key
        base_url: Optional API base URL
        verbose: Print progress info

    Returns:
        Simulation results dict
    """
    if verbose:
        print(f"Generating simulation from: \"{user_prompt}\"")
        print(f"Using model: {model}")

    spec = generate(
        user_prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    if verbose:
        name = spec.get("metadata", {}).get("name", "Unnamed")
        agent_types = list(spec.get("agent_types", {}).keys())
        total = sum(
            t.get("initial_count", 0)
            for t in spec.get("agent_types", {}).values()
        )
        print(f"Created: {name}")
        print(f"Agents: {agent_types} ({total} total)")

    # Determine steps
    run_steps = steps or spec.get("metadata", {}).get("steps", 200)
    # Also check top-level steps field from LLM
    if steps is None and "steps" in spec:
        run_steps = spec.pop("steps")

    sim = Simulation(spec)

    if verbose:
        print(f"Running {run_steps} steps...")

    results = sim.run(run_steps)

    if verbose:
        summary = results.get("summary", {})
        print(f"Done. Final population: {summary.get('final_counts', {})}")

    return results


class ChatSession:
    """
    Multi-turn chat session for iterating on simulations.

    Keeps conversation history so the LLM can refine simulations
    based on feedback.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client=None,
        repair_attempts: int = 1,
    ):
        self.model = model
        self.client = client or _make_client(api_key, base_url)
        self.repair_attempts = repair_attempts
        self.messages = [
            {"role": "system", "content": get_system_prompt()}
        ]
        self.last_spec = None
        self.last_results = None

    def send(self, message: str, run: bool = True, steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Send a message and optionally run the resulting simulation.

        Args:
            message: User message (description or refinement)
            run: Whether to run the simulation (default True)
            steps: Override step count

        Returns:
            If run=True: simulation results
            If run=False: parsed specification
        """
        self.messages.append({"role": "user", "content": message})

        self.last_spec = _generate_validated(
            self.client, self.model, self.messages, self.repair_attempts
        )

        if not run:
            return self.last_spec

        run_steps = steps or self.last_spec.get("steps") \
            or self.last_spec.get("metadata", {}).get("steps", 200)

        sim = Simulation(self.last_spec)
        self.last_results = sim.run(run_steps)

        # Add results context for next turn
        summary = self.last_results.get("summary", {})
        self.messages.append({
            "role": "user",
            "content": f"[System: Simulation completed. Results: {json.dumps(summary)}. "
                       f"If I ask for changes, update the full spec JSON.]"
        })

        return self.last_results

    def reset(self):
        """Reset conversation history."""
        self.messages = [
            {"role": "system", "content": get_system_prompt()}
        ]
        self.last_spec = None
        self.last_results = None
