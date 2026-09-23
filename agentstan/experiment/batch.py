"""
Batch runner: run the same model many times with parameter variations.
"""

import copy
import json
import os
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..core.simulation import Simulation


def _set_nested(d: dict, path: str, value: Any) -> dict:
    """Set a nested dict value by dot-path. e.g. 'agent_types.wolf.initial_count'."""
    out = copy.deepcopy(d)
    keys = path.split(".")
    target = out
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    return out


def _run_one(spec: dict, steps: int, run_id: int, params: dict,
             seed: Optional[int] = None, max_agents: Optional[int] = None,
             time_limit: Optional[float] = None) -> Dict[str, Any]:
    """Run a single simulation and return results with metadata."""
    sim = Simulation(spec, seed=seed)
    results = sim.run(steps, max_agents=max_agents, time_limit=time_limit)
    history = results["metrics"]["history"]
    return {
        "run_id": run_id,
        "seed": seed,
        "params": params,
        "summary": results["summary"],
        "observables": history[-1].get("observables", {}) if history else {},
        "globals": results["globals"],
        "stopped": results["stopped"],
        "final_step": results["final_step"],
        "duration": results["duration"],
        "history": history,
    }


def batch_run(
    spec: Dict[str, Any],
    n_runs: int = 10,
    steps: int = 200,
    vary: Optional[Dict[str, list]] = None,
    max_workers: Optional[int] = None,
    output_file: Optional[str] = None,
    seed: Optional[int] = None,
    max_agents: Optional[int] = None,
    time_limit: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Run a model many times, optionally varying parameters.

    Args:
        spec: Base simulation specification.
        n_runs: Number of runs per parameter combination.
        steps: Steps per run.
        vary: Dict mapping dot-path params to lists of values.
              e.g. {"agent_types.wolf.initial_count": [5, 10, 15, 20]}
              If None, runs the same spec n_runs times.
        max_workers: Parallel worker processes (default: CPU count).
            1 runs everything in this process.
        seed: Base seed (defaults to the spec's "seed"). Run i uses
            seed + i, so the batch is reproducible while runs stay
            statistically independent. With no seed anywhere, runs are
            unseeded (different every time).
        max_agents, time_limit: per-run resource guards (Simulation.run).

    Returns:
        List of result dicts sorted by run_id, each with run_id, seed,
        params, summary, observables (final values), globals, stopped,
        history.

    Example:
        results = batch_run(spec, n_runs=20, steps=200,
                            vary={"agent_types.wolf.initial_count": [5, 10, 20]})
    """
    if seed is None:
        seed = spec.get("seed")

    # Build list of (spec, params) to run
    jobs = []

    if vary:
        # Generate all parameter combinations
        param_combos = [{}]
        for path, values in vary.items():
            new_combos = []
            for combo in param_combos:
                for val in values:
                    new_combo = dict(combo)
                    new_combo[path] = val
                    new_combos.append(new_combo)
            param_combos = new_combos

        run_id = 0
        for combo in param_combos:
            modified_spec = copy.deepcopy(spec)
            for path, val in combo.items():
                modified_spec = _set_nested(modified_spec, path, val)
            for _ in range(n_runs):
                jobs.append((modified_spec, run_id, combo))
                run_id += 1
    else:
        for run_id in range(n_runs):
            jobs.append((spec, run_id, {}))

    def args_for(job):
        job_spec, rid, params = job
        run_seed = seed + rid if seed is not None else None
        return (job_spec, steps, rid, params, run_seed, max_agents, time_limit)

    workers = max_workers or os.cpu_count() or 1
    workers = min(workers, len(jobs)) or 1

    out_file = open(output_file, "w") if output_file else None
    results = []

    def collect(result):
        if out_file:
            # Stream to JSONL; don't hold history in memory
            compact = {k: v for k, v in result.items() if k != "history"}
            out_file.write(json.dumps(compact, default=str) + "\n")
            result.pop("history", None)
        results.append(result)

    try:
        if workers == 1:
            for job in jobs:
                collect(_run_one(*args_for(job)))
        else:
            # Processes, not threads: the engine is pure Python, so threads
            # would serialize on the GIL. Specs are plain JSON, so they
            # pickle cheaply.
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_run_one, *args_for(job)) for job in jobs]
                for future in as_completed(futures):
                    collect(future.result())
    finally:
        if out_file:
            out_file.close()

    results.sort(key=lambda r: r["run_id"])
    return results


def _metric(run: Dict[str, Any], name: str):
    if name in run.get("observables", {}):
        return run["observables"][name]
    if name in run.get("globals", {}):
        return run["globals"][name]
    return run["summary"]["final_counts"].get(name, 0)


def summarize(runs: List[Dict[str, Any]],
              metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Distribution of final values across runs.

    ``metrics`` names observables, globals, or agent types (final count);
    by default every observable plus every agent type. Returns::

        {"runs": n,
         "stopped_early": fraction of runs cut short by a resource guard,
         "metrics": {name: {mean, std, min, p5, median, p95, max}}}

    Example:
        grouped = sweep(spec, "globals.quest_reward", [5, 10, 20], n_runs=30)
        for reward, runs in grouped.items():
            print(reward, summarize(runs)["metrics"]["gold_supply"]["p95"])
    """
    if not runs:
        return {"runs": 0, "stopped_early": 0.0, "metrics": {}}
    if metrics is None:
        names = set()
        for run in runs:
            names.update(run.get("observables", {}))
            names.update(run["summary"]["final_counts"])
        metrics = sorted(names)

    out = {}
    for name in metrics:
        values = sorted(v for v in (_metric(r, name) for r in runs)
                        if isinstance(v, (int, float)) and not isinstance(v, bool))
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5

        def q(p):
            return values[min(n - 1, int(round(p * (n - 1))))]

        out[name] = {"mean": mean, "std": std, "min": values[0], "p5": q(0.05),
                     "median": q(0.5), "p95": q(0.95), "max": values[-1]}
    return {
        "runs": len(runs),
        "stopped_early": sum(1 for r in runs if r.get("stopped")) / len(runs),
        "metrics": out,
    }
