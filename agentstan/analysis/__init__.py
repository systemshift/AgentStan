"""Analysis tools: population dynamics, event breakdowns."""
from .population import analyze as analyze_population
from .events import analyze_events

__all__ = ["analyze_population", "analyze_events"]
