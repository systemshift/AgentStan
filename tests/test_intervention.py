"""Tests for intervention system."""
from agentstan import Simulation
from agentstan.core.intervention import InterventionEngine

SPEC = {
    "environment": {"type": "grid_2d", "dimensions": {"width": 10, "height": 10}},
    "agent_types": {
        "dot": {
            "initial_count": 5,
            "initial_state": {"energy": 20},
            "behavior_code": "def dot_behavior(a,m,n):\n    return [{'type':'move_random'}]",
        }
    },
}


def test_add_agent_intervention():
    sim = Simulation(SPEC)
    engine = InterventionEngine(sim)
    sim.attach_intervention_engine(engine)

    assert sim.agent_manager.get_total_count() == 5

    engine.add_agent("dot", {"energy": 50})
    sim.run_step()

    assert sim.agent_manager.get_total_count() == 6


def test_remove_agent_intervention():
    sim = Simulation(SPEC)
    engine = InterventionEngine(sim)
    sim.attach_intervention_engine(engine)

    first_id = sim.agent_manager.get_living_agents()[0].id
    engine.remove_agent(first_id)
    sim.run_step()

    # After cleanup, should be 4
    assert sim.agent_manager.get_total_count() == 4


def test_modify_agent_intervention():
    sim = Simulation(SPEC)
    engine = InterventionEngine(sim)
    sim.attach_intervention_engine(engine)

    agent = sim.agent_manager.get_living_agents()[0]
    agent_id = agent.id
    engine.modify_agent(agent_id, "energy", value=999)
    sim.run_step()

    agent = sim.agent_manager.get_agent(agent_id)
    assert agent.get_attribute("energy") == 999


def test_modify_environment_intervention():
    sim = Simulation(SPEC)
    engine = InterventionEngine(sim)
    sim.attach_intervention_engine(engine)

    engine.modify_environment("food_level", 100)
    sim.run_step()

    assert sim.environment.get_property("food_level") == 100


def test_teleport_agent():
    # Use agents with no movement behavior so teleport sticks
    spec = {
        "environment": {"type": "grid_2d", "dimensions": {"width": 10, "height": 10}},
        "agent_types": {
            "dot": {"initial_count": 3, "initial_state": {"energy": 20},
                    "behavior_code": "def dot_behavior(a,m,n):\n    return []"}
        },
    }
    sim = Simulation(spec)
    engine = InterventionEngine(sim)
    sim.attach_intervention_engine(engine)

    agent = sim.agent_manager.get_living_agents()[0]
    engine.teleport_agent(agent.id, (0, 0))
    sim.run_step()

    assert agent.get_attribute("position") == (0, 0)


def test_intervention_history():
    sim = Simulation(SPEC)
    engine = InterventionEngine(sim)
    sim.attach_intervention_engine(engine)

    engine.add_agent("dot", {"energy": 10})
    engine.modify_environment("test", True)
    sim.run_step()

    assert len(engine.history) == 2


def test_swap_behavior():
    sim = Simulation(SPEC)
    engine = InterventionEngine(sim)
    sim.attach_intervention_engine(engine)

    def new_behavior(agent, sim_state, nearby):
        return [{"type": "modify_state", "attribute": "energy", "delta": 100}]

    agent = sim.agent_manager.get_living_agents()[0]
    original_energy = agent.get_attribute("energy")
    engine.swap_behavior(agent.id, new_behavior)
    sim.run_step()

    # Energy should have increased by ~100 (minus any global decay)
    assert agent.get_attribute("energy") > original_energy + 50


def test_added_agents_of_a_rules_type_get_its_behavior_and_defaults():
    spec = {
        "environment": {"type": "none"},
        "agent_types": {
            "worker": {"initial_count": 0, "initial_state": {"gold": 0, "rate": 2},
                       "behavior": {"rules": [
                           {"do": [{"type": "modify_state", "attribute": "gold",
                                    "delta": "$rate"}]}]}},
        },
    }
    sim = Simulation(spec, seed=0)
    engine = InterventionEngine(sim)
    sim.attach_intervention_engine(engine)
    engine.add_agent("worker", {"rate": 5})
    sim.run(3)
    worker = sim.agent_manager.get_agents_by_type("worker")[0]
    assert worker["gold"] == 15   # acted every step with the override rate
