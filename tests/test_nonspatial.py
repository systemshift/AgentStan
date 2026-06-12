"""Tests for the non-spatial ("none") environment."""
import json

from agentstan import FrameRecorder, Simulation


ECONOMY = {
    "environment": {"type": "none"},
    "agent_types": {
        "farmer": {
            "initial_count": 5,
            "initial_state": {"gold": 0, "wheat": 0},
            "behavior": {"rules": [
                {"do": [{"type": "modify_state", "attribute": "wheat", "delta": 1}]},
                {"when": {">": ["$wheat", 0]},
                 "do": [{"type": "interact", "target": {"random": {"type": "baker"}},
                         "interaction_type": "sell_wheat",
                         "params": {"transfer": {"attribute": "wheat", "amount": 1},
                                    "self_delta": {"gold": 2}}}]},
            ]},
        },
        "baker": {"initial_count": 3, "initial_state": {"wheat": 0}},
    },
}


def test_everyone_perceives_everyone():
    sim = Simulation(ECONOMY, seed=1)
    farmer = sim.agent_manager.get_agents_by_type("farmer")[0]
    nearby = sim.agent_manager.get_agents_near_agent(farmer, 1, sim.environment)
    assert len(nearby) == 7  # all other agents, no radius, no positions


def test_economy_trades_without_geography():
    sim = Simulation(ECONOMY, seed=1)
    sim.run(20)
    total_baker_wheat = sum(a["wheat"] for a in sim.agent_manager.get_agents_by_type("baker"))
    total_farmer_gold = sum(a["gold"] for a in sim.agent_manager.get_agents_by_type("farmer"))
    assert total_baker_wheat > 0          # wheat actually moved
    assert total_farmer_gold == 2 * total_baker_wheat  # paid for every transfer


def test_frames_for_nonspatial_are_type_only():
    recorder = FrameRecorder()
    sim = Simulation(ECONOMY, seed=2)
    sim.add_collector(recorder)
    sim.run(5)
    data = recorder.to_dict()
    assert data["environment"]["type"] == "none"
    assert len(data["frames"]) == 5
    entry = data["frames"][0]["agents"][0]
    assert len(entry) == 2  # [id, type_index] — no fake coordinates
    json.dumps(data)


def test_spec_without_dimensions_validates():
    sim = Simulation(ECONOMY, seed=3)
    assert sim.environment.env_type == "none"
