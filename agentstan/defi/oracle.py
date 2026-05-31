"""
Price paths and the lagged oracle.

The *true* price is deterministic given a scenario — we do not predict markets,
we replay a chosen shock. The oracle reports a possibly-delayed view of that
true price; the gap between the two during a crash is what produces bad debt.
"""

from typing import List, Optional
from .config import ScenarioConfig


def build_price_path(scenario: ScenarioConfig, steps: int) -> List[float]:
    """Materialize the true price for each step [0, steps].

    If ``price_path`` is given explicitly it is used (padded/truncated to length).
    Otherwise a gradual crash is generated: flat until ``crash_start``, a linear
    decline of ``crash_pct`` over ``crash_duration`` steps, then flat at the floor.
    A gradual (not instantaneous) crash is what makes oracle lag and liquidator
    throughput matter — an instant drop would defeat any protocol.
    """
    n = steps + 1
    if scenario.price_path:
        path = list(scenario.price_path)
        if len(path) < n:
            path += [path[-1]] * (n - len(path))
        return path[:n]

    p0 = scenario.initial_price
    floor = p0 * (1.0 - scenario.crash_pct)
    start = scenario.crash_start
    duration = max(1, scenario.crash_duration)

    path = []
    for t in range(n):
        if t < start:
            path.append(p0)
        elif t < start + duration:
            frac = (t - start + 1) / duration
            path.append(p0 + (floor - p0) * frac)
        else:
            path.append(floor)
    return path


def build_debt_price_path(scenario: ScenarioConfig, steps: int) -> List[float]:
    """True price (USD) of the borrowed asset for each step.

    Default is a constant ``debt_initial_price`` (1.0 == debt is the numeraire).
    ``debt_spike_pct`` generates a rise over the crash window — the liability-side
    shock (a borrowed token squeezing upward)."""
    n = steps + 1
    if scenario.debt_price_path:
        path = list(scenario.debt_price_path)
        if len(path) < n:
            path += [path[-1]] * (n - len(path))
        return path[:n]

    p0 = scenario.debt_initial_price
    if not scenario.debt_spike_pct:
        return [p0] * n
    peak = p0 * (1.0 + scenario.debt_spike_pct)
    start = scenario.crash_start
    duration = max(1, scenario.crash_duration)
    path = []
    for t in range(n):
        if t < start:
            path.append(p0)
        elif t < start + duration:
            frac = (t - start + 1) / duration
            path.append(p0 + (peak - p0) * frac)
        else:
            path.append(peak)
    return path


class Oracle:
    """Reports the true collateral price delayed by ``lag_steps``. Before enough
    history exists it reports the initial price. Optionally also tracks the debt
    asset's price (same lag); by default the debt price is a constant 1.0."""

    def __init__(self, true_prices: List[float], lag_steps: int,
                 debt_true_prices: Optional[List[float]] = None):
        self.true_prices = true_prices
        self.lag_steps = max(0, lag_steps)
        self.debt_true_prices = debt_true_prices or [1.0] * len(true_prices)

    def _at(self, series: List[float], step: int) -> float:
        return series[min(step, len(series) - 1)]

    def true_price(self, step: int) -> float:
        return self._at(self.true_prices, step)

    def reported_price(self, step: int) -> float:
        return self._at(self.true_prices, max(0, step - self.lag_steps))

    def debt_true_price(self, step: int) -> float:
        return self._at(self.debt_true_prices, step)

    def debt_reported_price(self, step: int) -> float:
        return self._at(self.debt_true_prices, max(0, step - self.lag_steps))
