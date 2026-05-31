# AgentStan

**AI-native agent-based modeling framework.**

**S**imulate. **T**est. **A**nalyze. **N**arrate.

```bash
pip install agentstan
```

## Quick Start

```python
from agentstan import Simulation

spec = {
    "environment": {"type": "grid_2d", "dimensions": {"width": 40, "height": 40, "topology": "torus"}},
    "agent_types": {
        "rabbit": {
            "initial_count": 80,
            "initial_state": {"energy": 25, "perception_radius": 5},
            "behavior_code": """
def rabbit_behavior(agent, model, agents_nearby):
    actions = []
    energy = agent['energy']

    if energy < 20:
        actions.append({'type': 'modify_state', 'attribute': 'energy', 'delta': 2})

    wolves = [a for a in agents_nearby if a.type == 'wolf' and a.alive]
    if wolves:
        w = wolves[0]['position']
        pos = agent['position']
        dx = 1 if pos[0] > w[0] else -1
        dy = 1 if pos[1] > w[1] else -1
        actions.append({'type': 'move', 'direction': [dx, dy]})
    else:
        actions.append({'type': 'move', 'direction': [random.choice([-1,0,1]), random.choice([-1,0,1])]})

    if energy > 30 and random.random() < 0.08:
        actions.append({'type': 'reproduce', 'energy_cost': 15, 'offspring_count': 1})

    actions.append({'type': 'modify_state', 'attribute': 'energy', 'delta': -0.8})
    if energy <= 0:
        actions.append({'type': 'die', 'cause': 'starvation'})
    return actions
"""
        }
    }
}

sim = Simulation(spec)
results = sim.run(200)
print(results["summary"]["final_counts"])
```

## The STAN Framework

### Simulate — Run agent-based models

```python
from agentstan import Simulation, StagedScheduler, DataCollector

sim = Simulation(spec, scheduler=StagedScheduler(["prey", "predator"]))

collector = DataCollector(
    model_metrics={"avg_energy": lambda s: sum(a["energy"] for a in s.agent_manager.get_living_agents()) / max(s.agent_manager.get_total_count(), 1)},
)
sim.add_collector(collector)

results = sim.run(200)
time_series = collector.get_model_data()
```

### Test — Batch runs and parameter sweeps

```python
from agentstan.experiment import batch_run, sweep

# Run 50 times to get statistical confidence
results = batch_run(spec, n_runs=50, steps=200)

# Sweep a parameter
results = sweep(spec, param="agent_types.wolf.initial_count", values=range(5, 50, 5), n_runs=10)
for val, runs in results.items():
    avg = sum(r["summary"]["final_counts"].get("wolf", 0) for r in runs) / len(runs)
    print(f"wolves={val}: avg final = {avg:.1f}")
```

### Analyze — Understand what happened

```python
from agentstan.analysis import analyze_population, analyze_events

pop_report = analyze_population(results)
# {'agent_types': {'rabbit': {'stability': 'oscillating', 'period': 34, ...}}}

event_report = analyze_events(results)
# {'deaths': {'by_cause': {'starvation': 42, 'predation': 18}, ...}}
```

### Narrate — AI explains the results

```python
# pip install agentstan[ai]
from agentstan.ai import generate, interpret, validate

# Generate a model from natural language
spec = generate("simulate wolves hunting rabbits in a forest")

# Run it
sim = Simulation(spec)
results = sim.run(200)

# AI explains what happened
explanation = interpret(results)
# "Rabbits peaked at step 34 then crashed due to overgrazing..."

# Validate the model matches the description
issues = validate(spec, "wolves should hunt rabbits")
```

## CLI

```bash
# Run from spec file
agentstan --from-spec ecosystem.json --steps 200

# Batch run
agentstan --from-spec ecosystem.json --batch 50 --analyze

# Generate from natural language (requires agentstan[ai])
agentstan "simulate ants foraging for food" --steps 300
```

## Architecture

```
agentstan/
  core/           # Simulation engine, agents, environments, schedulers
  experiment/     # Batch runs, parameter sweeps
  analysis/       # Population dynamics, event analysis
  ai/             # LLM-powered generation, interpretation, validation
  defi/           # DeFi lending stress-testing (see below)
```

## DeFi stress testing

`agentstan.defi` is a self-serve simulation lab for DeFi lending protocols —
*stress-test your protocol before it breaks*. Describe a market in plain
language, run crash / depeg / whale-panic scenarios against it, sweep parameters
to find the safe frontier, and export a governance-ready report.

```python
from agentstan.defi import author_config, LendingMarket

cfg = author_config("USDC market, 75% max LTV, stress a 40% ETH crash with a slow oracle")
print(LendingMarket(cfg).run(120)["summary"]["bad_debt"])
```

Deterministic protocol mechanics (interest, oracle, liquidation, slippage) +
constrained LLM behavioral agents, validated against real events (MakerDAO Black
Thursday, the Aave CRV squeeze). See **[`agentstan/defi/README.md`](agentstan/defi/README.md)**.

## License

BSD 3-Clause
