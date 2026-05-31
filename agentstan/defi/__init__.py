"""
AgentStan DeFi: behavioral stress-testing for lending protocols.

Open-source lending-market simulation engine. Deterministic protocol mechanics
(interest, oracle, liquidation) plus constrained-action agents, all driven by a
declarative, serializable config that an LLM or webapp can author.

    from agentstan.defi import LendingMarket, MarketConfig

    config = MarketConfig.from_dict(spec)
    results = LendingMarket(config).run(steps=120)
    print(results["summary"]["bad_debt"])
"""

from .config import (
    MarketConfig, ProtocolConfig, OracleConfig, ScenarioConfig,
    AgentPopulation, RateModel,
)
from .engine import LendingMarket
from .sweep import sweep, frontier, grid, run_config

__all__ = [
    "LendingMarket",
    "MarketConfig",
    "ProtocolConfig",
    "OracleConfig",
    "ScenarioConfig",
    "AgentPopulation",
    "RateModel",
    "sweep",
    "frontier",
    "grid",
    "run_config",
]
