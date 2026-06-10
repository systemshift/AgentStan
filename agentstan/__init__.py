"""
AgentStan: AI-native agent-based modeling framework.

STAN = Simulate, Test, Analyze, Narrate.

    from agentstan import Simulation, Observer, InterventionEngine
    from agentstan.experiment import batch_run, sweep
    from agentstan.analysis import analyze_population
    from agentstan.ai import generate, interpret
    from agentstan.ai.llm_behavior import LLMBehaviorEngine
"""

from .core.simulation import Simulation
from .core.agent import Agent, AgentManager
from .core.environment import Environment
from .core.collectors import DataCollector
from .core.scheduler import RandomScheduler, StagedScheduler, SimultaneousScheduler
from .core.observer import Observer
from .core.intervention import InterventionEngine
from .core.rules import RuleBehavior, RuleError

__version__ = "0.1.0"

__all__ = [
    "Simulation",
    "RuleBehavior",
    "RuleError",
    "Agent",
    "AgentManager",
    "Environment",
    "DataCollector",
    "RandomScheduler",
    "StagedScheduler",
    "SimultaneousScheduler",
    "Observer",
    "InterventionEngine",
]
