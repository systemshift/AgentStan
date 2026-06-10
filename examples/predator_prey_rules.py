"""
Predator-prey expressed as declarative rules: the whole model is pure JSON.

No behavior_code, no Python in the spec. This is the same wolves-and-rabbits
model as examples/predator_prey.py, rewritten in the rule language — the spec
below could be stored in a database, emitted by an LLM, or fed to a non-Python
engine implementing the same rule semantics.

Run: python examples/predator_prey_rules.py
"""

import json
from agentstan import Simulation

SPEC = {
    "seed": 42,
    "environment": {
        "type": "grid_2d",
        "dimensions": {"width": 40, "height": 40, "topology": "torus"},
    },
    "agent_types": {
        "rabbit": {
            "initial_count": 80,
            "initial_state": {"energy": 25, "perception_radius": 5},
            "behavior": {
                "rules": [
                    # Graze when hungry
                    {"when": {"<": ["$energy", 20]},
                     "do": [{"type": "modify_state", "attribute": "energy", "delta": 2}]},
                    # Flee the nearest wolf
                    {"when": {">": [{"count": {"type": "wolf"}}, 0]},
                     "do": [{"type": "move_away", "from": {"nearest": {"type": "wolf"}}}]},
                    # Otherwise wander
                    {"when": {"==": [{"count": {"type": "wolf"}}, 0]},
                     "do": [{"type": "move",
                             "direction": [{"choice": [-1, 0, 1]}, {"choice": [-1, 0, 1]}]}]},
                    # Reproduce when well-fed
                    {"when": {">": ["$energy", 30]}, "prob": 0.08,
                     "do": [{"type": "reproduce", "energy_cost": 15, "offspring_count": 1}]},
                    # Metabolism
                    {"do": [{"type": "modify_state", "attribute": "energy", "delta": -0.8}]},
                    # Starvation
                    {"when": {"<=": ["$energy", 0]},
                     "do": [{"type": "die", "cause": "starvation"}]},
                ]
            },
        },
        "wolf": {
            "initial_count": 15,
            "initial_state": {"energy": 40, "perception_radius": 7},
            "behavior": {
                "rules": [
                    # Pounce when sharing a cell with a rabbit
                    {"when": {"<=": [{"nearest_distance": {"type": "rabbit"}}, 0]},
                     "do": [{"type": "interact",
                             "target": {"nearest": {"type": "rabbit"}},
                             "interaction_type": "predation",
                             "params": {"success_rate": 0.4, "energy_gain": 12}}]},
                    # Chase the nearest rabbit
                    {"when": {"and": [
                        {">": [{"nearest_distance": {"type": "rabbit"}}, 0]},
                        {"<": [{"nearest_distance": {"type": "rabbit"}}, 999]},
                    ]},
                     "do": [{"type": "move_toward", "target": {"nearest": {"type": "rabbit"}}}]},
                    # No rabbits in sight: wander
                    {"when": {"==": [{"count": {"type": "rabbit"}}, 0]},
                     "do": [{"type": "move",
                             "direction": [{"choice": [-1, 0, 1]}, {"choice": [-1, 0, 1]}]}]},
                    # Reproduce when strong
                    {"when": {">": ["$energy", 50]}, "prob": 0.04,
                     "do": [{"type": "reproduce", "energy_cost": 20, "offspring_count": 1}]},
                    # Metabolism
                    {"do": [{"type": "modify_state", "attribute": "energy", "delta": -1.0}]},
                    # Starvation
                    {"when": {"<=": ["$energy", 0]},
                     "do": [{"type": "die", "cause": "starvation"}]},
                ]
            },
        },
    },
}


def main():
    # The spec is pure data — prove it round-trips through JSON.
    spec = json.loads(json.dumps(SPEC))

    sim = Simulation(spec)
    results = sim.run(200)

    print(f"Ran {results['final_step']} steps (seed={results['seed']})")
    print(f"Initial: {results['summary']['initial_counts']}")
    print(f"Final:   {results['summary']['final_counts']}")

    events = results["event_summary"].get("event_types", {})
    if events:
        print(f"Events:  {events}")


if __name__ == "__main__":
    main()
