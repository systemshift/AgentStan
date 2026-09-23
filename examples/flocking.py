"""
Flocking (Boids) Model

Birds follow three simple rules:
1. Separation — avoid crowding nearby birds
2. Alignment — steer toward average heading of neighbors
3. Cohesion — steer toward average position of neighbors

These three rules produce realistic emergent flocking behavior.
Uses continuous-like movement on a grid. The behavior is a Python function
passed via Simulation(spec, behaviors=...) — the escape hatch for logic the
rule language doesn't express (the spec itself stays pure JSON).
"""

from agentstan import Simulation, DataCollector

def bird_behavior(agent, sim_state, agents_nearby):
    """Python behavior: plain function, randomness from sim_state["rng"]."""
    rng = sim_state["rng"]
    wander = {"type": "move", "direction": [rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])]}
    flock = [a for a in agents_nearby if a.alive]
    if not flock:
        return [wander]

    x, y = agent["position"]
    # Cohesion target: average position of neighbors
    avg_x = sum(a["position"][0] for a in flock) / len(flock)
    avg_y = sum(a["position"][1] for a in flock) / len(flock)

    # Separation: step away from the nearest bird if it's too close
    nearest = min(flock, key=lambda a: abs(a["position"][0] - x) + abs(a["position"][1] - y))
    nx, ny = nearest["position"]
    if abs(nx - x) + abs(ny - y) <= 1:
        dx = (x > nx) - (x < nx)
        dy = (y > ny) - (y < ny)
    else:
        dx = (avg_x > x) - (avg_x < x)
        dy = (avg_y > y) - (avg_y < y)

    # Noise
    if rng.random() < 0.2:
        return [wander]
    return [{"type": "move", "direction": [dx, dy]}]


spec = {
    "seed": 3,
    "environment": {
        "type": "grid_2d",
        "dimensions": {"width": 40, "height": 40, "topology": "torus"},
    },
    "agent_types": {
        "bird": {
            "initial_count": 50,
            "initial_state": {"perception_radius": 5},
        },
    },
}


def compute_avg_cluster_size(sim):
    """Average number of neighbors per bird (proxy for flocking)."""
    total_neighbors = 0
    n = 0
    for agent in sim.agent_manager.get_living_agents():
        nearby = sim.agent_manager.get_agents_near_agent(
            agent, agent.get_attribute("perception_radius", 5), sim.environment
        )
        total_neighbors += len(nearby)
        n += 1
    return total_neighbors / n if n > 0 else 0


def run_single():
    print("=== Flocking (Boids) Model ===")
    print("50 birds on a 40x40 torus grid")
    print("Rules: separation, cohesion, random noise")
    print()

    sim = Simulation(spec, behaviors={"bird": bird_behavior})
    collector = DataCollector(
        model_metrics={"avg_neighbors": compute_avg_cluster_size},
    )
    sim.add_collector(collector)
    results = sim.run(100)

    data = collector.get_model_data()

    print(f"Steps: {results['final_step']}")
    print(f"Initial avg neighbors: {data[0]['avg_neighbors']:.1f}")
    print(f"Final avg neighbors:   {data[-1]['avg_neighbors']:.1f}")
    print()

    print("Flocking density over time:")
    for i in range(0, len(data), 10):
        bar = "#" * int(data[i]["avg_neighbors"] * 3)
        print(f"  Step {data[i]['step']:3d}: {data[i]['avg_neighbors']:.1f} neighbors {bar}")


if __name__ == "__main__":
    run_single()
