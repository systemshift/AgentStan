"""
Parameter sweep: vary one parameter, run multiple times per value.
"""

from typing import Dict, Any, List, Optional
from .batch import batch_run


def sweep(
    spec: Dict[str, Any],
    param: str,
    values: list,
    steps: int = 200,
    n_runs: int = 10,
    max_workers: Optional[int] = None,
    seed: Optional[int] = None,
    max_agents: Optional[int] = None,
    time_limit: Optional[float] = None,
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Sweep a single parameter across values with replications.

    Args:
        spec: Base simulation specification.
        param: Dot-path to the parameter (e.g. "agent_types.wolf.initial_count"
            or "globals.tax_rate").
        values: List of values to try.
        steps: Steps per run.
        n_runs: Replications per value.
        max_workers, seed, max_agents, time_limit: as for batch_run.

    Returns:
        Dict mapping each value to a list of run results (see batch_run);
        pass a list to experiment.summarize for distributions.

    Example:
        results = sweep(spec, "agent_types.wolf.initial_count",
                        values=[5, 10, 15, 20], n_runs=10)
        for val, runs in results.items():
            print(val, summarize(runs)["metrics"]["wolf"]["mean"])
    """
    all_results = batch_run(
        spec,
        n_runs=n_runs,
        steps=steps,
        vary={param: list(values)},
        max_workers=max_workers,
        seed=seed,
        max_agents=max_agents,
        time_limit=time_limit,
    )

    # Group by parameter value
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for result in all_results:
        grouped.setdefault(result["params"].get(param), []).append(result)

    return grouped
