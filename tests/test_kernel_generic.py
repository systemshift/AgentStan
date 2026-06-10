"""Tests for the domain-blind kernel: generic interactions, generic
reproduction cost, global_rules, and the absence of hardcoded ecology."""
from agentstan import Simulation


def _base_env(width=10):
    return {"type": "grid_2d",
            "dimensions": {"width": width, "height": width, "topology": "torus"}}


def test_no_hardcoded_energy_death():
    """Agents with energy <= 0 survive unless the spec says otherwise."""
    spec = {
        "environment": _base_env(),
        "agent_types": {
            "ghost": {"initial_count": 5, "initial_state": {"energy": -10}},
        },
    }
    sim = Simulation(spec, seed=0)
    results = sim.run(10)
    assert results["summary"]["final_counts"]["ghost"] == 5


def test_global_rules_replace_energy_death():
    spec = {
        "environment": _base_env(),
        "global_rules": [
            {"when": {"<=": ["$energy", 0]},
             "do": [{"type": "die", "cause": "energy_depleted"}]},
        ],
        "agent_types": {
            "mortal": {"initial_count": 5, "initial_state": {"energy": -10}},
            "immortal": {"initial_count": 3, "initial_state": {}},
        },
    }
    sim = Simulation(spec, seed=0)
    results = sim.run(5)
    assert results["summary"]["final_counts"]["mortal"] == 0
    assert results["summary"]["final_counts"]["immortal"] == 3


def test_generic_transfer_moves_any_attribute():
    spec = {
        "environment": _base_env(),
        "agent_types": {
            "giver": {
                "initial_count": 1,
                "initial_state": {"gold": 10, "perception_radius": 20, "position": [1, 1]},
                "behavior": {"rules": [
                    {"do": [{"type": "interact",
                             "target": {"nearest": {"type": "taker"}},
                             "interaction_type": "gift",
                             "params": {"transfer": {"attribute": "gold", "amount": 2}}}]},
                ]},
            },
            "taker": {
                "initial_count": 1,
                "initial_state": {"gold": 0, "position": [2, 2]},
            },
        },
    }
    sim = Simulation(spec, seed=0)
    sim.run(3)
    giver = sim.agent_manager.get_agents_by_type("giver")[0]
    taker = sim.agent_manager.get_agents_by_type("taker")[0]
    assert giver["gold"] == 4
    assert taker["gold"] == 6


def test_generic_kill_and_deltas():
    spec = {
        "environment": _base_env(),
        "agent_types": {
            "assassin": {
                "initial_count": 1,
                "initial_state": {"contracts": 0, "perception_radius": 20, "position": [1, 1]},
                "behavior": {"rules": [
                    {"do": [{"type": "interact",
                             "target": {"nearest": {"type": "victim"}},
                             "interaction_type": "assassination",
                             "params": {"kill_target": True,
                                        "self_delta": {"contracts": 1}}}]},
                ]},
            },
            "victim": {
                "initial_count": 1,
                "initial_state": {"position": [2, 2]},
            },
        },
    }
    sim = Simulation(spec, seed=0)
    results = sim.run(2)
    assert results["summary"]["final_counts"]["victim"] == 0
    assassin = sim.agent_manager.get_agents_by_type("assassin")[0]
    assert assassin["contracts"] == 1
    # Death cause is the interaction type, not a hardcoded ecology word
    deaths = [e for e in results["events"] if e["type"] == "agent_death"]
    assert deaths and deaths[0]["cause"] == "assassination"


def test_legacy_predation_alias_still_works():
    spec = {
        "environment": _base_env(),
        "agent_types": {
            "wolf": {
                "initial_count": 1,
                "initial_state": {"energy": 10, "perception_radius": 20, "position": [1, 1]},
                "behavior": {"rules": [
                    {"do": [{"type": "interact",
                             "target": {"nearest": {"type": "rabbit"}},
                             "interaction_type": "predation",
                             "params": {"success_rate": 1.0, "energy_gain": 5}}]},
                ]},
            },
            "rabbit": {"initial_count": 1, "initial_state": {"position": [2, 2]}},
        },
    }
    sim = Simulation(spec, seed=0)
    results = sim.run(2)
    assert results["summary"]["final_counts"]["rabbit"] == 0
    wolf = sim.agent_manager.get_agents_by_type("wolf")[0]
    assert wolf["energy"] == 15


def test_reproduce_with_generic_cost():
    spec = {
        "environment": _base_env(),
        "agent_types": {
            "cell": {
                "initial_count": 1,
                "initial_state": {"biomass": 100},
                "behavior": {"rules": [
                    {"when": {">=": ["$biomass", 50]},
                     "do": [{"type": "reproduce",
                             "cost": {"attribute": "biomass", "amount": 40},
                             "offspring_state": {"generation": 1}}]},
                ]},
            },
        },
    }
    sim = Simulation(spec, seed=0)
    sim.run(1)
    cells = sim.agent_manager.get_agents_by_type("cell")
    assert len(cells) == 2
    parent = next(c for c in cells if c.get_attribute("generation") is None)
    child = next(c for c in cells if c.get_attribute("generation") == 1)
    assert parent["biomass"] == 60   # 100 - 40 cost
    assert child["biomass"] == 50    # half of parent's 100 at clone time
