"""Tests for ai/generate.py — no network: an OpenAI-compatible stub client."""
import json
from types import SimpleNamespace

import pytest

from agentstan.ai.generate import generate, ChatSession, _spec_from_response


class StubClient:
    """Minimal OpenAI-compatible client returning canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.responses.pop(0)
        message = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


GOOD_SPEC = {
    "name": "Foxes",
    "description": "Foxes wander",
    "environment": {"type": "grid_2d", "dimensions": {"width": 10, "height": 10}},
    "agent_types": {
        "fox": {
            "initial_count": 5,
            "initial_state": {"energy": 20, "perception_radius": 3},
            "behavior": {"rules": [
                {"do": [{"type": "move", "direction": [{"choice": [-1, 0, 1]},
                                                        {"choice": [-1, 0, 1]}]}]},
                {"do": [{"type": "modify_state", "attribute": "energy", "delta": -0.5}]},
                {"when": {"<=": ["$energy", 0]}, "do": [{"type": "die", "cause": "starvation"}]},
            ]},
        }
    },
    "steps": 50,
}

# Same spec but with an invalid action type — engine validation must reject it
BAD_SPEC = json.loads(json.dumps(GOOD_SPEC))
BAD_SPEC["agent_types"]["fox"]["behavior"]["rules"][0]["do"][0]["type"] = "teleport"


def test_generate_returns_validated_rules_spec():
    client = StubClient([json.dumps(GOOD_SPEC)])
    spec = generate("simulate foxes", client=client)
    assert "fox" in spec["agent_types"]
    assert "rules" in spec["agent_types"]["fox"]["behavior"]
    assert spec["steps"] == 50
    assert len(client.calls) == 1


def test_generate_repairs_invalid_spec():
    client = StubClient([json.dumps(BAD_SPEC), json.dumps(GOOD_SPEC)])
    spec = generate("simulate foxes", client=client)
    assert len(client.calls) == 2
    # The repair turn must contain the engine's validation error
    repair_messages = client.calls[1]["messages"]
    assert any(
        "failed validation" in m["content"] and "teleport" in m["content"]
        for m in repair_messages if m["role"] == "user"
    )
    assert spec["agent_types"]["fox"]["behavior"]["rules"][0]["do"][0]["type"] == "move"


def test_generate_gives_up_after_repair_attempts():
    client = StubClient([json.dumps(BAD_SPEC), json.dumps(BAD_SPEC)])
    with pytest.raises(ValueError, match="failed validation"):
        generate("simulate foxes", client=client, repair_attempts=1)
    assert len(client.calls) == 2


def test_generate_keeps_passive_types_and_shared_state():
    """Types without behavior (a shop, a resource) are legitimate, and the
    normalizer must pass globals/world_rules/observables through."""
    spec_in = {
        "name": "shop world",
        "environment": {"type": "none"},
        "globals": {"tax": 1},
        "world_rules": [{"do": [{"type": "modify_global", "name": "tax", "delta": 1}]}],
        "observables": {"tax": "@tax"},
        "agent_types": {"shop": {"initial_count": 1, "initial_state": {"stock": 3}}},
    }
    spec = generate("a shop", client=StubClient([json.dumps(spec_in)]))
    assert spec["metadata"]["name"] == "shop world"
    for key in ("globals", "world_rules", "observables"):
        assert spec[key] == spec_in[key]


def test_generate_does_not_invent_defaults():
    """A missing environment is an error for the LLM to repair, not a grid."""
    no_env = {"agent_types": {"a": {"initial_count": 1, "initial_state": {}}}}
    fixed = dict(no_env, environment={"type": "none"})
    client = StubClient([json.dumps(no_env), json.dumps(fixed)])
    spec = generate("x", client=client)
    assert spec["environment"] == {"type": "none"}
    repair_prompts = [m["content"] for m in client.calls[1]["messages"] if m["role"] == "user"]
    assert any("missing 'environment'" in text for text in repair_prompts)


def test_spec_from_response_accepts_legacy_behavior_code():
    data = {
        "environment": {"type": "grid_2d", "dimensions": {"width": 5, "height": 5}},
        "agent_types": {
            "fox": {
                "initial_count": 2,
                "behavior_code": "def fox_behavior(agent, model, nearby):\n    return []",
            }
        },
    }
    spec = _spec_from_response(data)
    assert "behavior_code" in spec["agent_types"]["fox"]


def test_chat_session_runs_and_keeps_history():
    client = StubClient([json.dumps(GOOD_SPEC)])
    session = ChatSession(client=client)
    results = session.send("simulate foxes", steps=10)
    assert results["final_step"] >= 1
    assert session.last_spec is not None
    roles = [m["role"] for m in session.messages]
    assert roles[0] == "system" and "assistant" in roles


def test_every_full_spec_in_the_system_prompt_runs():
    """The prompt is the LLM's spec reference — its examples must be valid."""
    import json as _json
    import re as _re
    from agentstan import Simulation
    from agentstan.ai.prompt import get_system_prompt

    blocks = _re.findall(r"```json\n(.*?)```", get_system_prompt(), _re.S)
    specs = [_json.loads(b) for b in blocks if '"agent_types"' in b and '"environment"' in b
             and "..." not in b]
    assert len(specs) >= 2
    for spec in specs:
        Simulation.check(spec, smoke_steps=30)


def test_interpret_and_validate_use_injected_client(monkeypatch):
    from agentstan import Simulation
    from agentstan.ai import interpret, validate

    monkeypatch.setenv("AGENTSTAN_MODEL", "some-local-model")
    spec = {"environment": {"type": "none"}, "globals": {"price": 3},
            "observables": {"price": "@price"},
            "agent_types": {"a": {"initial_count": 2, "initial_state": {}}}}
    results = Simulation(spec, seed=0).run(3)

    client = StubClient(["Prices stayed flat.",
                         json.dumps({"valid": True, "issues": [], "suggestions": ["x"]})])
    assert interpret(results, client=client) == "Prices stayed flat."
    assert validate(spec, "a flat market", client=client)["valid"] is True
    assert [c["model"] for c in client.calls] == ["some-local-model"] * 2
    prompt = client.calls[0]["messages"][0]["content"]
    assert '"observables"' in prompt and '"final_globals"' in prompt
