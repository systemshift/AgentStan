"""
Batch runner: run the same model many times with parameter variations.
"""

import copy
import json
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
             seed: Optional[int] = None) -> Dict[str, Any]:
    """Run a single simulation and return results with metadata."""
    sim = Simulation(spec, seed=seed)
    results = sim.run(steps)
    return {
        "run_id": run_id,
        "seed": seed,
        "params": params,
        "summary": results["summary"],
        "final_step": results["final_step"],
        "duration": results["duration"],
        "history": results["metrics"]["history"],
    }


def batch_run(
    spec: Dict[str, Any],
    n_runs: int = 10,
    steps: int = 200,
    vary: Optional[Dict[str, list]] = None,
    max_workers: int = 4,
    output_file: Optional[str] = None,
    seed: Optional[int] = None,
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
        max_workers: Parallel threads.
        seed: Base seed. Run i uses seed + i, making the whole batch
              reproducible while keeping runs statistically independent.
              If None, runs are unseeded (different every time).

    Returns:
        List of result dicts, each with run_id, params, summary, history.

    Example:
        results = batch_run(spec, n_runs=20, steps=200,
                            vary={"agent_types.wolf.initial_count": [5, 10, 20]})
    """
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
                jobs.append((modified_spec, steps, run_id, combo))
                run_id += 1
    else:
        for run_id in range(n_runs):
            jobs.append((copy.deepcopy(spec), steps, run_id, {}))

    # Execute
    results = []
    out_file = None
    if output_file:
        out_file = open(output_file, "w")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, s, st, rid, p,
                            seed + rid if seed is not None else None): rid
            for s, st, rid, p in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            if out_file:
                # Stream to JSONL — don't hold history in memory
                import json
                compact = {
                    "run_id": result["run_id"],
                    "params": result["params"],
                    "summary": result["summary"],
                    "final_step": result["final_step"],
                    "duration": result["duration"],
                }
                out_file.write(json.dumps(compact, default=str) + "\n")
                # Strip history from in-memory result to save RAM
                result.pop("history", None)
            results.append(result)

    if out_file:
        out_file.close()

    results.sort(key=lambda r: r["run_id"])
    return results
