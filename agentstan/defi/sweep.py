"""
Parameter sweeps for the lending market.

The ecology engine's ``batch_run`` is welded to the spatial ``Simulation`` and
its result schema, so DeFi gets its own thin sweeper. It operates on the
declarative ``MarketConfig`` by dot-path, so the same machinery drives a CLI, a
webapp's "vary this slider", or an LLM proposing a parameter grid.

    from agentstan.defi import MarketConfig
    from agentstan.defi.sweep import sweep, frontier, grid

    base = MarketConfig.from_dict(spec)
    pts = frontier(base, "protocol.max_ltv", [0.5, 0.6, 0.7, 0.8], metric="bad_debt")
"""

import copy
from typing import Any, Dict, List, Tuple, Union

from .config import MarketConfig
from .engine import LendingMarket

ConfigLike = Union[MarketConfig, Dict[str, Any]]


def _as_dict(config: ConfigLike) -> Dict[str, Any]:
    return config.to_dict() if isinstance(config, MarketConfig) else copy.deepcopy(config)


def _set_nested(d: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """Set a value by dot-path, e.g. 'protocol.max_ltv' or
    'populations.2.params.capital'. List segments use integer indices."""
    out = copy.deepcopy(d)
    keys = path.split(".")
    target = out
    for key in keys[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    last = keys[-1]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value
    return out


def run_config(config: ConfigLike, steps: int = 120) -> Dict[str, Any]:
    """Run a single config (dict or MarketConfig) and return the full results."""
    cfg = config if isinstance(config, MarketConfig) else MarketConfig.from_dict(config)
    return LendingMarket(cfg).run(steps)


def sweep(base: ConfigLike, param: str, values: List[Any],
          steps: int = 120) -> Dict[Any, Dict[str, Any]]:
    """Vary one parameter across ``values``; return {value: summary}."""
    base_dict = _as_dict(base)
    out: Dict[Any, Dict[str, Any]] = {}
    for v in values:
        spec = _set_nested(base_dict, param, v)
        out[v] = LendingMarket(MarketConfig.from_dict(spec)).run(steps)["summary"]
    return out


def frontier(base: ConfigLike, param: str, values: List[Any],
             metric: str = "bad_debt", steps: int = 120) -> List[Tuple[Any, float]]:
    """Sweep one parameter and extract a single metric: [(value, metric), ...]."""
    swept = sweep(base, param, values, steps)
    return [(v, swept[v][metric]) for v in values]


def grid(base: ConfigLike, params: Dict[str, List[Any]],
         steps: int = 120) -> List[Dict[str, Any]]:
    """Sweep the cartesian product of several parameters.

    Returns a list of {"params": {path: value, ...}, "summary": {...}}, one per
    combination — the shape a 2-D heatmap (e.g. LTV x oracle-lag) renders from.
    """
    base_dict = _as_dict(base)
    combos: List[Dict[str, Any]] = [{}]
    for path, values in params.items():
        combos = [{**combo, path: v} for combo in combos for v in values]

    results = []
    for combo in combos:
        spec = base_dict
        for path, value in combo.items():
            spec = _set_nested(spec, path, value)
        summary = LendingMarket(MarketConfig.from_dict(spec)).run(steps)["summary"]
        results.append({"params": combo, "summary": summary})
    return results
