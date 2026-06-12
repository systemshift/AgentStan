"""Tests for the core simulation engine."""
from agentstan import Simulation, Agent, DataCollector
from agentstan.core.scheduler import RandomScheduler, StagedScheduler, SimultaneousScheduler

SPEC = {
    "environment": {"type": "grid_2d", "dimensions": {"width": 20, "height": 20, "topology": "torus"}},
    "agent_types": {
        "rabbit": {
            "initial_count": 30,
            "initial_state": {"energy": 25, "perception_radius": 5},
            "behavior_code": (
                "def rabbit_behavior(agent, model, agents_nearby):\n"
                "    actions = []\n"
                "    energy = agent['energy']\n"
                "    if energy < 20:\n"
                "        actions.append({'type': 'modify_state', 'attribute': 'energy', 'delta': 2})\n"
                "    actions.append({'type': 'move', 'direction': [random.choice([-1,0,1]), random.choice([-1,0,1])]})\n"
                "    actions.append({'type': 'modify_state', 'attribute': 'energy', 'delta': -0.8})\n"
                "    if energy <= 0:\n"
                "        actions.append({'type': 'die', 'cause': 'starvation'})\n"
                "    return actions\n"
            ),
        },
        "wolf": {
            "initial_count": 5,
            "initial_state": {"energy": 40, "perception_radius": 7},
            "behavior_code": (
                "def wolf_behavior(agent, model, agents_nearby):\n"
                "    actions = []\n"
                "    energy = agent['energy']\n"
                "    position = agent['position']\n"
                "    rabbits = [a for a in agents_nearby if a.type == 'rabbit' and a.alive]\n"
                "    if rabbits:\n"
                "        r = rabbits[0]\n"
                "        rpos = r['position']\n"
                "        if position == rpos:\n"
                "            actions.append({'type': 'interact', 'target_id': r.id, "
                "'interaction_type': 'predation', 'params': {'success_rate': 0.4, 'energy_gain': 12}})\n"
                "        else:\n"
                "            actions.append({'type': 'move_to', 'target': (rpos[0], rpos[1])})\n"
                "    else:\n"
                "        actions.append({'type': 'move', 'direction': [random.choice([-1,0,1]), random.choice([-1,0,1])]})\n"
                "    actions.append({'type': 'modify_state', 'attribute': 'energy', 'delta': -1.0})\n"
                "    if energy <= 0:\n"
                "        actions.append({'type': 'die', 'cause': 'starvation'})\n"
                "    return actions\n"
            ),
        },
    },
}


def test_simulation_runs():
    sim = Simulation(SPEC)
    results = sim.run(50)
    assert results["final_step"] >= 1
    assert "summary" in results
    assert "rabbit" in results["summary"]["initial_counts"]


def test_agent_dict_access():
    agent = Agent("test", {"energy": 25, "position": (5, 5)})
    assert agent["energy"] == 25
    agent["energy"] = 30
    assert agent["energy"] == 30
    assert "energy" in agent
    assert agent.get_attribute("energy") == 30


def test_schedulers():
    sim_random = Simulation(SPEC, scheduler=RandomScheduler())
    sim_random.run(5)

    sim_staged = Simulation(SPEC, scheduler=StagedScheduler(["rabbit", "wolf"]))
    sim_staged.run(5)

    sim_simul = Simulation(SPEC, scheduler=SimultaneousScheduler())
    sim_simul.run(5)


def test_data_collector():
    collector = DataCollector(
        model_metrics={"avg_energy": lambda sim: sum(
            a.get_attribute("energy", 0) for a in sim.agent_manager.get_living_agents()
        ) / max(sim.agent_manager.get_total_count(), 1)},
    )
    sim = Simulation(SPEC)
    sim.add_collector(collector)
    sim.run(20)

    data = collector.get_model_data()
    assert len(data) == 20 or len(data) > 0  # might stop early if all die
    assert "total_agents" in data[0]
    assert "avg_energy" in data[0]
    assert "count_rabbit" in data[0]


EXPLODER = {
    "environment": {"type": "grid_2d", "dimensions": {"width": 10, "height": 10}},
    "agent_types": {
        "amoeba": {
            "initial_count": 4,
            "initial_state": {"energy": 100},
            "behavior": {"rules": [
                # Unbounded doubling: the population guard must catch this
                {"do": [{"type": "reproduce", "cost": {"attribute": "energy", "amount": 0}}]},
            ]},
        },
    },
}


def test_max_agents_guard_stops_runaway_growth():
    sim = Simulation(EXPLODER, seed=1)
    results = sim.run(100, max_agents=200)
    assert results["stopped"]["reason"] == "max_agents"
    assert results["stopped"]["at_step"] < 100
    assert results["summary"]["final_agents"] > 200  # partial results intact


def test_time_limit_guard():
    sim = Simulation(EXPLODER, seed=1)
    results = sim.run(10_000, time_limit=0.05)
    assert results["stopped"]["reason"] == "time_limit"


def test_normal_run_reports_no_stop():
    sim = Simulation(SPEC, seed=1)
    results = sim.run(10, max_agents=100_000, time_limit=60)
    assert results["stopped"] is None
