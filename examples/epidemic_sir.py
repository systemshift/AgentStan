"""
SIR Epidemic Model

Agents are Susceptible, Infected, or Recovered. Infected agents spread
the disease to nearby susceptible agents with some probability. After
an infection period, agents recover and become immune.

Shows classic epidemic curves: exponential growth, peak, decline as
herd immunity builds. Pure rules: state changes are `transform` actions.
Try varying the infection chance or recovery_time.
"""

from agentstan import Simulation

WANDER = {"type": "move", "direction": [{"choice": [-1, 0, 1]}, {"choice": [-1, 0, 1]}]}

spec = {
    "seed": 7,
    "environment": {
        "type": "grid_2d",
        "dimensions": {"width": 30, "height": 30, "topology": "torus"},
    },
    "agent_types": {
        "susceptible": {
            "initial_count": 290,
            "initial_state": {"perception_radius": 2},
            "behavior": {"rules": [
                # 15% chance per infected neighbor (capped at certainty)
                {"when": {">": [{"count": {"type": "infected"}}, 0]},
                 "prob": {"min": [1, {"*": [0.15, {"count": {"type": "infected"}}]}]},
                 "do": [{"type": "transform", "new_type": "infected",
                         "new_state": {"days_infected": 0}}]},
                {"do": [WANDER]},
            ]},
        },
        "infected": {
            "initial_count": 10,
            "initial_state": {"perception_radius": 2, "days_infected": 0,
                              "recovery_time": 14},
            "behavior": {"rules": [
                {"do": [{"type": "modify_state", "attribute": "days_infected", "delta": 1}]},
                {"when": {">=": ["$days_infected", "$recovery_time"]},
                 "do": [{"type": "transform", "new_type": "recovered"}]},
                {"prob": 0.5, "do": [WANDER]},  # infected agents move less
            ]},
        },
        "recovered": {
            "initial_count": 0,
            "initial_state": {"perception_radius": 2},
            "behavior": {"rules": [{"do": [WANDER]}]},
        },
    },
}


def run_single():
    print("=== SIR Epidemic Model ===")
    print("300 agents (290 susceptible, 10 infected)")
    print("Infection: 15% per infected neighbor per step, recovery: 14 steps")
    print()

    results = Simulation(spec).run(100)
    data = [{"step": h["step"],
             "S": h["agent_counts"].get("susceptible", 0),
             "I": h["agent_counts"].get("infected", 0),
             "R": h["agent_counts"].get("recovered", 0)}
            for h in results["metrics"]["history"]]

    peak = max(data, key=lambda d: d["I"])
    print(f"Peak infection: {peak['I']} at step {peak['step']}")
    print(f"Final: S={data[-1]['S']}, I={data[-1]['I']}, R={data[-1]['R']}")
    print(f"Attack rate: {(data[-1]['R'] + data[-1]['I']) / 300:.0%}")
    print()

    print("SIR Curve:")
    for d in data[::5]:
        bars = "S" * (d["S"] // 10) + "|" + "I" * (d["I"] // 5) + "|" + "R" * (d["R"] // 10)
        print(f"  Step {d['step']:3d}: {bars}  (S={d['S']} I={d['I']} R={d['R']})")


if __name__ == "__main__":
    run_single()
