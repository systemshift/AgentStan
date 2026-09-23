"""Tests for the experiment module."""
from agentstan.experiment import batch_run, sweep

SPEC = {
    "environment": {"type": "grid_2d", "dimensions": {"width": 10, "height": 10}},
    "agent_types": {
        "dot": {
            "initial_count": 10,
            "initial_state": {"energy": 15},
            "behavior_code": (
                "def dot_behavior(agent, model, agents_nearby):\n"
                "    actions = []\n"
                "    actions.append({'type': 'move_random'})\n"
                "    actions.append({'type': 'modify_state', 'attribute': 'energy', 'delta': -0.5})\n"
                "    if agent['energy'] <= 0:\n"
                "        actions.append({'type': 'die', 'cause': 'starvation'})\n"
                "    return actions\n"
            ),
        }
    },
}


def test_batch_run_basic():
    results = batch_run(SPEC, n_runs=3, steps=10, max_workers=2)
    assert len(results) == 3
    assert all("summary" in r for r in results)
    assert all("run_id" in r for r in results)


def test_batch_run_with_vary():
    results = batch_run(
        SPEC,
        n_runs=2,
        steps=10,
        vary={"agent_types.dot.initial_count": [5, 10]},
        max_workers=2,
    )
    # 2 values x 2 runs = 4 total
    assert len(results) == 4


def test_sweep():
    results = sweep(
        SPEC,
        param="agent_types.dot.initial_count",
        values=[5, 10, 15],
        steps=10,
        n_runs=2,
        max_workers=2,
    )
    assert set(results.keys()) == {5, 10, 15}
    assert all(len(runs) == 2 for runs in results.values())


SEEDED = {
    "seed": 7,
    "environment": {"type": "none"},
    "observables": {"total_gold": {"sum": {"attr": "gold"}}},
    "agent_types": {
        "miner": {
            "initial_count": 5,
            "initial_state": {"gold": 0},
            "behavior": {"rules": [
                {"do": [{"type": "modify_state", "attribute": "gold",
                         "delta": {"randint": [0, 10]}}]},
            ]},
        }
    },
}


def test_spec_seed_gives_distinct_but_reproducible_runs():
    a = batch_run(SEEDED, n_runs=4, steps=5, max_workers=1)
    b = batch_run(SEEDED, n_runs=4, steps=5, max_workers=2)  # processes
    golds = [r["observables"]["total_gold"] for r in a]
    assert len(set(golds)) > 1                 # not the same run 4 times
    assert [r["seed"] for r in a] == [7, 8, 9, 10]
    assert golds == [r["observables"]["total_gold"] for r in b]


def test_summarize_distributions():
    from agentstan.experiment import summarize
    runs = batch_run(SEEDED, n_runs=10, steps=5, max_workers=1)
    report = summarize(runs)
    assert report["runs"] == 10 and report["stopped_early"] == 0
    gold = report["metrics"]["total_gold"]
    assert gold["min"] <= gold["p5"] <= gold["median"] <= gold["p95"] <= gold["max"]
    assert report["metrics"]["miner"]["mean"] == 5


def test_sweep_over_a_global():
    spec = dict(SEEDED, globals={"bonus": 0})
    spec["agent_types"] = {"miner": {
        "initial_count": 2, "initial_state": {"gold": 0},
        "behavior": {"rules": [{"do": [{"type": "modify_state", "attribute": "gold",
                                        "delta": "@bonus"}]}]}}}
    grouped = sweep(spec, "globals.bonus", [1, 3], steps=4, n_runs=2, max_workers=1)
    assert [r["observables"]["total_gold"] for r in grouped[3]] == [24, 24]
