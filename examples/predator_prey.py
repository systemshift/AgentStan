"""
Predator-Prey (Lotka-Volterra) Model

Wolves hunt rabbits. Rabbits graze for energy. Both reproduce when
energy is high. Classic oscillating population dynamics emerge:
rabbits boom -> wolves boom -> rabbits decline -> wolves decline.

Try varying wolf count, predation success rate, or energy decay
to find stable vs unstable parameter regions.

The behaviors are Python functions passed via Simulation(spec,
behaviors=...); examples/predator_prey_rules.py is the same model as
pure-JSON rules.
"""

from agentstan import Simulation, DataCollector
from agentstan.analysis import analyze_population
from agentstan.experiment import sweep

def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def rabbit_behavior(agent, sim_state, agents_nearby):
    """Python behavior: plain function, randomness from sim_state["rng"]."""
    rng = sim_state["rng"]
    actions = []
    energy = agent["energy"]
    position = agent["position"]

    # Graze for energy
    if energy < 22:
        actions.append({"type": "modify_state", "attribute": "energy", "delta": 2.5})

    # Flee from the nearest wolf, otherwise wander
    wolves = [a for a in agents_nearby if a.type == "wolf" and a.alive]
    if wolves:
        w = min(wolves, key=lambda w: _dist(position, w["position"]))["position"]
        actions.append({"type": "move", "direction": [1 if position[0] > w[0] else -1,
                                                      1 if position[1] > w[1] else -1]})
    else:
        actions.append({"type": "move", "direction": [rng.choice([-1, 0, 1]),
                                                      rng.choice([-1, 0, 1])]})

    if energy > 24 and rng.random() < 0.08:
        actions.append({"type": "reproduce", "cost": {"attribute": "energy", "amount": 10}})

    actions.append({"type": "modify_state", "attribute": "energy", "delta": -0.6})
    if energy <= 0:
        actions.append({"type": "die", "cause": "starvation"})
    return actions


def wolf_behavior(agent, sim_state, agents_nearby):
    rng = sim_state["rng"]
    actions = []
    energy = agent["energy"]
    position = agent["position"]

    # Hunt: attack if adjacent, and chase
    rabbits = [a for a in agents_nearby if a.type == "rabbit" and a.alive]
    if rabbits:
        prey = min(rabbits, key=lambda r: _dist(position, r["position"]))
        if _dist(position, prey["position"]) <= 1:
            actions.append({"type": "interact", "target_id": prey.id,
                            "interaction_type": "predation",
                            "params": {"success_rate": 0.4, "kill_target": True,
                                       "self_delta": {"energy": 20}}})
        actions.append({"type": "move_to", "target": tuple(prey["position"])})
    else:
        actions.append({"type": "move", "direction": [rng.choice([-1, 0, 1]),
                                                      rng.choice([-1, 0, 1])]})

    if energy > 45 and rng.random() < 0.04:
        actions.append({"type": "reproduce", "cost": {"attribute": "energy", "amount": 20}})

    actions.append({"type": "modify_state", "attribute": "energy", "delta": -0.8})
    if energy <= 0:
        actions.append({"type": "die", "cause": "starvation"})
    return actions


BEHAVIORS = {"rabbit": rabbit_behavior, "wolf": wolf_behavior}

spec = {
    "seed": 11,
    "environment": {
        "type": "grid_2d",
        "dimensions": {"width": 30, "height": 30, "topology": "torus"},
    },
    "agent_types": {
        "rabbit": {
            "initial_count": 80,
            "initial_state": {"energy": 25, "perception_radius": 4},
        },
        "wolf": {
            "initial_count": 15,
            "initial_state": {"energy": 50, "perception_radius": 6},
        },
    },
}


def run_single():
    print("=== Predator-Prey Model ===")
    print("80 rabbits, 15 wolves, 30x30 grid")
    print()

    sim = Simulation(spec, behaviors=BEHAVIORS)
    collector = DataCollector()
    sim.add_collector(collector)
    results = sim.run(200)

    # Population analysis
    report = analyze_population(results)
    summary = results["summary"]

    print(f"Steps: {results['final_step']}")
    print(f"Initial: rabbit={summary['initial_counts'].get('rabbit', 0)}, wolf={summary['initial_counts'].get('wolf', 0)}")
    print(f"Final:   rabbit={summary['final_counts'].get('rabbit', 0)}, wolf={summary['final_counts'].get('wolf', 0)}")
    print()

    for agent_type in ["rabbit", "wolf"]:
        info = report["agent_types"].get(agent_type, {})
        print(f"  {agent_type}:")
        print(f"    stability: {info.get('stability', '?')}")
        print(f"    peak: {info.get('peak', '?')} (step {info.get('peak_step', '?')})")
        if info.get("extinct"):
            print(f"    EXTINCT at step {info.get('extinction_step')}")
        if info.get("period"):
            print(f"    oscillation period: ~{info['period']} steps")
        print()

    # Print population timeline
    data = collector.get_model_data()
    print("Population over time:")
    for i in range(0, len(data), 20):
        row = data[i]
        r = row.get("count_rabbit", 0)
        w = row.get("count_wolf", 0)
        r_bar = "R" * min(r // 2, 40)
        w_bar = "W" * min(w, 20)
        print(f"  Step {row['step']:3d}: {r_bar} ({r}) | {w_bar} ({w})")


def run_wolf_sweep():
    print("\n=== Wolf Count Sweep ===")
    print("How many wolves are sustainable?\n")

    results = sweep(
        spec,
        param="agent_types.wolf.initial_count",
        values=[5, 10, 15, 20, 30],
        steps=200,
        n_runs=3,
        behaviors=BEHAVIORS,
    )

    for val in sorted(results.keys()):
        runs = results[val]
        avg_rabbits = sum(r["summary"]["final_counts"].get("rabbit", 0) for r in runs) / len(runs)
        avg_wolves = sum(r["summary"]["final_counts"].get("wolf", 0) for r in runs) / len(runs)
        extinct = sum(1 for r in runs if r["summary"]["final_counts"].get("wolf", 0) == 0)
        print(f"  wolves={val:2d}: avg final rabbit={avg_rabbits:5.1f}, wolf={avg_wolves:4.1f}, wolf extinct {extinct}/{len(runs)}")


if __name__ == "__main__":
    run_single()
    run_wolf_sweep()
