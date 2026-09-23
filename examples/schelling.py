"""
Schelling Segregation Model

Two types of agents on a grid. Each checks if enough neighbors are the
same type. If not, they move to a random cell. Simple rules produce
dramatic segregation from a well-mixed initial state.

Pure rules: each agent records whether it is happy, and the "segregation"
observable is the share of happy agents. Try varying tolerance (0.1 = very
tolerant, 0.8 = very intolerant).
"""

from agentstan import Simulation
from agentstan.experiment import sweep, summarize

TOLERANCE = 0.3


def rules_for(color):
    same = {"count": {"type": color}}
    anyone = {"count": {}}
    happy = {"or": [{"==": [anyone, 0]},
                    {">=": [{"/": [same, {"max": [anyone, 1]}]}, "$tolerance"]}]}
    return [
        {"do": [{"type": "modify_state", "attribute": "happy", "value": happy}]},
        {"when": {"not": happy}, "do": [{"type": "move_random"}]},
    ]


spec = {
    "seed": 1,
    "environment": {
        "type": "grid_2d",
        "dimensions": {"width": 30, "height": 30, "topology": "torus"},
    },
    "observables": {"segregation": {"mean": {"attr": "happy"}}},
    "agent_types": {
        color: {
            "initial_count": 200,
            "initial_state": {"tolerance": TOLERANCE, "perception_radius": 2,
                              "happy": False},
            "behavior": {"rules": rules_for(color)},
        }
        for color in ("blue", "red")
    },
}


def run_single():
    print("=== Schelling Segregation Model ===")
    print(f"Grid: 30x30, 400 agents (200 blue, 200 red), tolerance {TOLERANCE}")
    print()

    results = Simulation(spec).run(50)
    data = [(h["step"], h["observables"]["segregation"])
            for h in results["metrics"]["history"]]

    print(f"Happy after step 1: {data[0][1]:.1%}")
    print(f"Happy at the end:   {data[-1][1]:.1%}")
    print()
    for step, value in data[::10]:
        print(f"  Step {step:3d}: {value:.1%} {'#' * int(value * 40)}")


def run_tolerance_sweep():
    print("\n=== Tolerance Sweep ===")
    print("How does tolerance affect how many agents end up happy?\n")

    grouped = sweep(spec, param="agent_types.blue.initial_state.tolerance",
                    values=[0.1, 0.3, 0.5, 0.7], steps=50, n_runs=5)
    for tolerance, runs in sorted(grouped.items()):
        stats = summarize(runs, ["segregation"])["metrics"]["segregation"]
        print(f"  blue tolerance={tolerance:.1f}: {stats['mean']:.1%} happy "
              f"(p5 {stats['p5']:.1%}, p95 {stats['p95']:.1%})")


if __name__ == "__main__":
    run_single()
    run_tolerance_sweep()
