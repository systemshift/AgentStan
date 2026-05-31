# AgentStan DeFi

**Behavioral stress-testing for DeFi lending protocols.**

> Stress-test your protocol before it breaks.

A self-serve simulation lab for lending markets: describe your protocol, run a
crash or a depeg or a whale panic against it, sweep your parameters to find the
safe frontier, and export a governance-ready report. Run Gauntlet-style stress
tests in-house.

**What this is:** scenario simulation, parameter design, and economic
failure-mode discovery. **What this is not:** a price predictor. We don't
forecast markets — we replay chosen shocks and show how your *parameters* behave
under them.

---

## Quick start

```python
from agentstan.defi import author_config, LendingMarket

# Describe the test in plain language (LLM-authored; offline keyword fallback)
cfg = author_config(
    "USDC lending market, 82.5% liquidation threshold, max LTV 75%, "
    "stress a 40% ETH crash with a 20-step oracle lag"
)
result = LendingMarket(cfg).run(120)
print(result["summary"]["bad_debt"], result["summary"]["insolvent"])
```

Or build it explicitly with a scenario preset:

```python
from agentstan.defi import scenarios, LendingMarket

cfg = scenarios.eth_crash(severity=0.40, oracle_lag=20)
summary = LendingMarket(cfg).run(120)["summary"]
```

## The pitch, in one table

Same protocol, same 40% ETH crash — two parameter sets:

| | Risky (80% LTV, slow oracle, thin liquidators) | Safe (60% LTV, fast oracle, deep liquidators) |
|---|---|---|
| **Bad debt** | **$24,233 — insolvent** | **$0 — solvent** |

*"These parameters break; these don't"* — that contrast, with the numbers behind
it, is the product. (`examples/defi_eth_crash.py`)

## How it works

- **Deterministic protocol mechanics.** Interest (kinked rate model), health
  factors, oracle with configurable lag, liquidation with bonus/close-factor,
  and bad-debt accounting are all classical, deterministic code.
- **Rational, capital-constrained liquidators.** They value seized collateral at
  the true market price (with slippage) and skip unprofitable liquidations — so
  a lagging oracle or a thin market produces bad debt the way it does in reality.
- **Constrained LLM behavioral agents.** Panic-withdrawals, whale strategy, and
  governance choices are made by an LLM choosing *one action from a fixed menu* —
  never free-form, never touching the math. (Pluggable; runs deterministically
  with no API key.)
- **Declarative config.** A run is fully described by serializable data (no
  code), so the same config drives the engine, sweeps, the report, and the
  natural-language front door.

Two distinct failure modes are measured separately: **insolvency** (bad debt)
and **liquidity crisis** (available liquidity → $0). Most simple models conflate
them.

## Scenarios

```python
from agentstan.defi import scenarios
cfg = scenarios.apply("stablecoin_depeg", my_config)   # or call the preset directly
```

| Preset | Failure mode |
|---|---|
| `eth_crash`, `oracle_delay` | bad debt from a lagging oracle (insolvent) |
| `liquidation_congestion` | bad debt from under-capitalized liquidators (insolvent) |
| `whale_panic`, `liquidity_withdrawal` | liquidity freeze (min liquidity → $0) without insolvency |
| `stablecoin_depeg` | sharp collateral depeg + recovery |
| `crv_short_squeeze` | illiquidity-driven loss on a *volatile borrowed asset* |

Presets are transformations that preserve *your* protocol parameters, so any
scenario can be run against your own settings. (`examples/defi_scenarios.py`)

## Parameter sweeps & the safe frontier

```python
from agentstan.defi import frontier, grid

pts = frontier(cfg, "protocol.max_ltv", [0.5, 0.6, 0.7, 0.8], metric="bad_debt")
heat = grid(cfg, {"protocol.max_ltv": [0.6, 0.7, 0.8],
                  "oracle.lag_steps": [0, 6, 12, 24]})
```

Answers the governance question directly: *how high can max LTV go before a
crash creates bad debt, and how much oracle latency can we tolerate?*
(`examples/defi_param_sweep.py`)

## Governance reports

```python
from agentstan.defi import write_report, frontier

headline = LendingMarket(cfg).run(120)
ltv_pts = frontier(cfg, "protocol.max_ltv", [0.5, 0.6, 0.7, 0.8])
write_report("stress_report.md", headline, ltv_frontier=ltv_pts)
```

Produces an audit-style Markdown report: assumptions, parameters, results, a
data-driven failure narrative (realized vs. latent bad debt), the safe frontier,
recommendations, and an embedded chart (if matplotlib is installed). This is the
deliverable. (`examples/defi_report.py`)

## Credibility: calibration against real events

The model is checked against documented historical crises — does it reproduce
the right *mechanism* at the right *order of magnitude*? Price paths and risk
parameters come from history; they are **not** fit to the outcome.

```python
from agentstan.defi import calibration
r = calibration.run_case(calibration.CASES["black_thursday"]())
print(calibration.calibration_report(r))
```

| Event | Documented | Model | Verdict |
|---|---|---|---|
| **MakerDAO Black Thursday** (2020, ETH −43%, oracle/keeper freeze) | ~6% of debt | **5.5%**, right mechanism; robust ~3–9% across assumptions | strong match |
| **Aave CRV squeeze** (2022, illiquidity) | ~4% (~$1.6M) | **10.7%**, right mechanism; **knife-edge near-miss** | brackets, doesn't pin |

Honest scope: one event is a sanity check, not validation. The CRV case is a
deliberately weaker calibration — the model identifies it as a recover/fail
boundary event but can't pin the magnitude without liquidator heterogeneity.
(`examples/defi_calibration.py`)

## Known limitations

Read these before trusting a number:

- **Calibration is ongoing** — two events so far, not a validated model.
- **Simplified market structure** — single collateral / single borrow asset;
  liquidators don't yet recycle capital; liquidator behavior is homogeneous
  (which is why illiquidity-class magnitudes like CRV aren't pinned).
- **Scenario analysis, not prediction** — every result is conditional on a chosen
  shock and stated assumptions, which the report makes explicit.

## API surface

```python
from agentstan.defi import (
    MarketConfig, LendingMarket,        # config + engine
    scenarios,                          # stress presets + apply()
    sweep, frontier, grid,              # parameter sweeps
    stress_report, write_report,        # governance reports
    author_config,                      # natural-language authoring
    calibration,                        # historical calibration harness
    OpenAIDecider, OpenAIConfigAuthor,  # production LLM backends (agentstan[ai])
)
```

Examples live in [`examples/`](../../examples): `defi_eth_crash.py`,
`defi_scenarios.py`, `defi_param_sweep.py`, `defi_report.py`,
`defi_authoring.py`, `defi_calibration.py`.
