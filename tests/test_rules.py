"""Tests for the declarative rule language (core/rules.py)."""
import json
import random

import pytest

from agentstan import Simulation, Agent, RuleError
from agentstan.core.environment import Environment
from agentstan.core.rules import evaluate, validate_rules, _Context


def make_ctx(agent_state=None, nearby=None, step=3, counts=None, seed=0):
    env = Environment("grid_2d", {"width": 20, "height": 20, "topology": "torus"})
    agent = Agent("fox", agent_state or {"energy": 10, "position": (5, 5)})
    return _Context(
        agent=agent,
        nearby=nearby or [],
        rng=random.Random(seed),
        env=env,
        step=step,
        counts=counts or {"fox": 4},
    )


# --- Expressions ---

def test_literals_and_agent_attrs():
    ctx = make_ctx({"energy": 12, "position": (5, 5)})
    assert evaluate(7, ctx) == 7
    assert evaluate("$energy", ctx) == 12
    assert evaluate("@step", ctx) == 3
    assert evaluate("hello", ctx) == "hello"


def test_arithmetic_and_comparison():
    ctx = make_ctx({"energy": 10, "position": (5, 5)})
    assert evaluate({"+": ["$energy", 5]}, ctx) == 15
    assert evaluate({"-": [20, "$energy"]}, ctx) == 10
    assert evaluate({"*": [2, 3, 4]}, ctx) == 24
    assert evaluate({"/": [10, 4]}, ctx) == 2.5
    assert evaluate({"<": ["$energy", 20]}, ctx) is True
    assert evaluate({">=": ["$energy", 10]}, ctx) is True
    assert evaluate({"!=": ["$energy", 10]}, ctx) is False


def test_logic_ops():
    ctx = make_ctx({"energy": 10, "position": (5, 5)})
    assert evaluate({"and": [{">": ["$energy", 5]}, {"<": ["$energy", 20]}]}, ctx) is True
    assert evaluate({"or": [{">": ["$energy", 50]}, True]}, ctx) is True
    assert evaluate({"not": {">": ["$energy", 50]}}, ctx) is True


def test_count_and_nearest_distance():
    wolf = Agent("wolf", {"position": (8, 5)})
    far_wolf = Agent("wolf", {"position": (5, 9)})
    sheep = Agent("sheep", {"position": (6, 5)})
    ctx = make_ctx({"position": (5, 5)}, nearby=[wolf, far_wolf, sheep])

    assert evaluate({"count": {"type": "wolf"}}, ctx) == 2
    assert evaluate({"count": {}}, ctx) == 3
    assert evaluate({"nearest_distance": {"type": "wolf"}}, ctx) == 3.0
    assert evaluate({"nearest_distance": {"type": "bear"}}, ctx) == float("inf")
    assert evaluate({"total": "fox"}, ctx) == 4


def test_random_ops_are_seeded():
    a = evaluate({"uniform": [0, 100]}, make_ctx(seed=42))
    b = evaluate({"uniform": [0, 100]}, make_ctx(seed=42))
    assert a == b


# --- Validation ---

def test_validate_rejects_unknown_action():
    with pytest.raises(RuleError, match="unknown action type"):
        validate_rules([{"do": [{"type": "teleport_home"}]}], "fox")


def test_validate_rejects_missing_do():
    with pytest.raises(RuleError, match="missing 'do'"):
        validate_rules([{"when": {">": [1, 0]}}], "fox")


def test_validate_rejects_unknown_operator():
    with pytest.raises(RuleError, match="unknown operator"):
        validate_rules([{"when": {"frobnicate": [1, 2]}, "do": []}], "fox")


def test_bad_rules_fail_at_simulation_construction():
    spec = {
        "environment": {"type": "grid_2d", "dimensions": {"width": 5, "height": 5}},
        "agent_types": {
            "fox": {
                "initial_count": 1,
                "behavior": {"rules": [{"do": [{"type": "nonsense"}]}]},
            }
        },
    }
    with pytest.raises(RuleError, match="agent_types\\['fox'\\]"):
        Simulation(spec)


# --- End-to-end ---

RULES_SPEC = {
    "environment": {
        "type": "grid_2d",
        "dimensions": {"width": 20, "height": 20, "topology": "torus"},
    },
    "agent_types": {
        "rabbit": {
            "initial_count": 30,
            "initial_state": {"energy": 25, "perception_radius": 5},
            "behavior": {
                "rules": [
                    {"when": {"<": ["$energy", 20]},
                     "do": [{"type": "modify_state", "attribute": "energy", "delta": 2}]},
                    {"when": {">": [{"count": {"type": "wolf"}}, 0]},
                     "do": [{"type": "move_away", "from": {"nearest": {"type": "wolf"}}}]},
                    {"when": {"==": [{"count": {"type": "wolf"}}, 0]},
                     "do": [{"type": "move",
                             "direction": [{"choice": [-1, 0, 1]}, {"choice": [-1, 0, 1]}]}]},
                    {"when": {">": ["$energy", 30]}, "prob": 0.08,
                     "do": [{"type": "reproduce", "energy_cost": 15}]},
                    {"do": [{"type": "modify_state", "attribute": "energy", "delta": -0.8}]},
                    {"when": {"<=": ["$energy", 0]},
                     "do": [{"type": "die", "cause": "starvation"}]},
                ]
            },
        },
        "wolf": {
            "initial_count": 5,
            "initial_state": {"energy": 40, "perception_radius": 7},
            "behavior": {
                "rules": [
                    {"when": {"<=": [{"nearest_distance": {"type": "rabbit"}}, 0]},
                     "do": [{"type": "interact",
                             "target": {"nearest": {"type": "rabbit"}},
                             "interaction_type": "predation",
                             "params": {"success_rate": 0.4, "energy_gain": 12}}]},
                    {"when": {"and": [
                        {">": [{"nearest_distance": {"type": "rabbit"}}, 0]},
                        {"<": [{"nearest_distance": {"type": "rabbit"}}, 999]},
                    ]},
                     "do": [{"type": "move_toward", "target": {"nearest": {"type": "rabbit"}}}]},
                    {"when": {"==": [{"count": {"type": "rabbit"}}, 0]},
                     "do": [{"type": "move",
                             "direction": [{"choice": [-1, 0, 1]}, {"choice": [-1, 0, 1]}]}]},
                    {"do": [{"type": "modify_state", "attribute": "energy", "delta": -1.0}]},
                    {"when": {"<=": ["$energy", 0]},
                     "do": [{"type": "die", "cause": "starvation"}]},
                ]
            },
        },
    },
}


def test_rules_spec_is_pure_json():
    """The whole spec must survive a JSON round-trip — no code anywhere."""
    restored = json.loads(json.dumps(RULES_SPEC))
    sim = Simulation(restored, seed=1)
    results = sim.run(20)
    assert results["final_step"] >= 1


def test_rules_simulation_runs_and_agents_act():
    sim = Simulation(RULES_SPEC, seed=5)
    results = sim.run(30)
    events = results["event_summary"]["event_types"]
    assert events.get("agent_action", 0) > 0
    assert events.get("state_change", 0) > 0


def test_move_toward_closes_distance():
    spec = {
        "environment": {"type": "grid_2d",
                        "dimensions": {"width": 30, "height": 30, "topology": "bounded"}},
        "agent_types": {
            "hunter": {
                "initial_count": 1,
                "initial_state": {"perception_radius": 25, "position": [2, 2]},
                "behavior": {"rules": [
                    {"do": [{"type": "move_toward", "target": {"nearest": {"type": "prey"}}}]},
                ]},
            },
            "prey": {
                "initial_count": 1,
                "initial_state": {"perception_radius": 1, "position": [12, 12]},
            },
        },
    }
    sim = Simulation(spec, seed=0)
    hunter = sim.agent_manager.get_agents_by_type("hunter")[0]
    prey = sim.agent_manager.get_agents_by_type("prey")[0]
    d0 = sim.environment.distance(hunter["position"], prey["position"])
    sim.run(5)
    d1 = sim.environment.distance(hunter["position"], prey["position"])
    assert d1 < d0


def test_same_seed_same_results():
    a = Simulation(RULES_SPEC, seed=42).run(30)
    b = Simulation(RULES_SPEC, seed=42).run(30)
    assert a["metrics"]["history"] == b["metrics"]["history"]
    assert a["summary"]["final_counts"] == b["summary"]["final_counts"]


def test_seed_in_spec_is_used():
    spec = dict(RULES_SPEC, seed=42)
    a = Simulation(spec).run(20)
    b = Simulation(spec).run(20)
    assert a["seed"] == 42
    assert a["metrics"]["history"] == b["metrics"]["history"]


def test_behavior_code_is_also_seeded():
    from tests.test_core import SPEC as CODE_SPEC
    a = Simulation(CODE_SPEC, seed=9).run(25)
    b = Simulation(CODE_SPEC, seed=9).run(25)
    assert a["metrics"]["history"] == b["metrics"]["history"]
