"""Tests for the transform action — agents changing type mid-simulation."""
from agentstan import Simulation


SPEC = {
    "environment": {"type": "grid_2d", "dimensions": {"width": 10, "height": 10}},
    "agent_types": {
        "susceptible": {
            "initial_count": 5,
            "initial_state": {"energy": 20},
            "behavior_code": (
                "def susceptible_behavior(agent, model, agents_nearby):\n"
                "    if agent.get_attribute('should_transform'):\n"
                "        return [{'type': 'transform', 'new_type': 'infected',\n"
                "                 'new_state': {'days_infected': 0}}]\n"
                "    return []\n"
            ),
        },
        "infected": {
            "initial_count": 0,
            "initial_state": {"days_infected": 0},
            "behavior_code": (
                "def infected_behavior(agent, model, agents_nearby):\n"
                "    return [{'type': 'modify_state', 'attribute': 'days_infected', 'delta': 1}]\n"
            ),
        },
    },
}


def test_transform_changes_type():
    sim = Simulation(SPEC)
    target = sim.agent_manager.get_living_agents()[0]
    target.set_attribute("should_transform", True)

    sim.run_step()

    counts = sim.agent_manager.get_counts()
    # One susceptible should have become infected
    assert counts.get("infected", 0) == 1
    assert counts.get("susceptible", 0) == 4


def test_transformed_agent_has_working_behavior():
    """Regression: transformed agents must use the spec's behavior_code,
    not a no-op."""
    sim = Simulation(SPEC)
    target = sim.agent_manager.get_living_agents()[0]
    target.set_attribute("should_transform", True)

    # Step 1: susceptible → infected with days_infected=0
    sim.run_step()
    infected = sim.agent_manager.get_agents_by_type("infected")
    assert len(infected) == 1
    assert infected[0].get_attribute("days_infected") == 0

    # Step 2: infected behavior runs, days_infected increments to 1
    sim.run_step()
    infected = sim.agent_manager.get_agents_by_type("infected")
    assert len(infected) == 1
    assert infected[0].get_attribute("days_infected") == 1


def test_transform_preserves_position():
    sim = Simulation(SPEC)
    target = sim.agent_manager.get_living_agents()[0]
    original_pos = target.get_attribute("position")
    target.set_attribute("should_transform", True)

    sim.run_step()

    infected = sim.agent_manager.get_agents_by_type("infected")[0]
    assert infected.get_attribute("position") == original_pos


def test_transform_merges_state():
    sim = Simulation(SPEC)
    target = sim.agent_manager.get_living_agents()[0]
    target.set_attribute("should_transform", True)

    sim.run_step()

    infected = sim.agent_manager.get_agents_by_type("infected")[0]
    # Old state preserved
    assert infected.get_attribute("energy") == 20
    # New state applied
    assert infected.get_attribute("days_infected") == 0


def test_transform_with_inline_behavior_code():
    """Action override path: explicit behavior_code in the action."""
    spec = {
        "environment": {"type": "grid_2d", "dimensions": {"width": 10, "height": 10}},
        "agent_types": {
            "starter": {
                "initial_count": 1,
                "initial_state": {"energy": 10},
                "behavior_code": (
                    "def starter_behavior(agent, model, agents_nearby):\n"
                    "    return [{'type': 'transform', 'new_type': 'overrider',\n"
                    "             'new_state': {'tag': 'override'},\n"
                    "             'behavior_code': 'def overrider_behavior(a,m,n):\\n"
                    "    return [{\"type\": \"modify_state\", \"attribute\": \"tag\", \"value\": \"ran\"}]\\n'}]\n"
                ),
            },
        },
    }
    sim = Simulation(spec)
    sim.run_step()  # transform fires
    sim.run_step()  # overrider behavior runs

    overriders = sim.agent_manager.get_agents_by_type("overrider")
    assert len(overriders) == 1
    assert overriders[0].get_attribute("tag") == "ran"


def test_transform_fills_missing_attributes_from_new_type():
    spec = {
        "environment": {"type": "none"},
        "agent_types": {
            "healthy": {"initial_count": 1, "initial_state": {"gold": 5},
                        "behavior": {"rules": [
                            {"do": [{"type": "transform", "new_type": "sick",
                                     "new_state": {"days": 1}}]}]}},
            "sick": {"initial_count": 0,
                     "initial_state": {"days": 0, "recovery_time": 14, "gold": 0}},
        },
    }
    sim = Simulation(spec, seed=0)
    sim.run(1)
    sick = sim.agent_manager.get_agents_by_type("sick")[0]
    assert sick["recovery_time"] == 14   # filled from the new type
    assert sick["gold"] == 5             # the agent's own state wins
    assert sick["days"] == 1             # new_state wins over both
