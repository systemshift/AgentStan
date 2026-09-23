# Changelog

## 0.2.0

The rule language can now express markets and economies, specs fail loudly
instead of silently, and batch experiments are real experiments. Several
changes alter results for existing specs — see **Behavior changes**.

### Added
- **Shared state.** `globals` (read as `"@name"`, written with
  `modify_global`), `world_rules` (run once per step before agents act),
  `observables` (named expressions recorded in `metrics.history`).
- **Targeted rules.** A rule-level `target` selector binds one agent;
  `"&attr"` reads it, and `interact` / `move_toward` / `move_away` default
  to it. Queries take `where` filters (`"&attr"` is the candidate).
- Selectors `lowest` / `highest` with `by`; aggregates `sum` / `mean`;
  `total` accepts a filtered query.
- `exchange`: an atomic two-sided trade. `spawn`: create agents from a
  type's `initial_state` (inflows).
- `choose`: a rule runs exactly one weighted branch.
- `initial_state` values may be expressions, evaluated per agent at
  creation (`choice`, `uniform`, `randint`, `@globals`).
- `Simulation(spec, behaviors={type: fn})`: Python behavior functions,
  seeded through `sim_state["rng"]`.
- `Simulation.check(spec)`: construct and smoke-run a spec. Used by
  `ai.generate` (repair loop) and `Pack.validate(deep=True)`.
- `experiment.summarize(runs)`: per-metric distributions (mean, std, p5,
  median, p95) over batch or sweep results.
- CLI subcommands: `run`, `validate`, `batch` (with `--vary`), `generate`;
  all accept a spec or a `.pack.json`.
- Checkpoints (`save` / `load`) restore RNG state, agent IDs and globals, so
  a resumed run continues exactly as the original.

### Behavior changes
- **Strict validation.** Unknown action fields, `interact` params, agent
  types, globals, and wrong operator arity are errors. So is reading
  `"$attr"` that no agent of that type can ever have.
- **Rule errors raise** `RuleError` with the rule path, agent and step,
  instead of being logged and leaving the agent inert.
- **Nested fields are evaluated.** Expressions inside `new_state`,
  `params`, `offspring_state`, etc. used to be stored literally (e.g. the
  string `"$xp"`).
- **`transfer` and `exchange` are preconditions.** If either side can't
  cover its amounts, the whole interaction fails and no effect applies
  (previously the other effects still happened, creating value).
- **`reproduce` conserves its cost.** Each offspring starts with the cost
  amount; the cost is paid per offspring. Previously the child got half
  the parent's value and the cost was charged once.
- **`transform`** fills attributes the agent lacks from the new type's
  `initial_state` (the agent's own state and `new_state` still win).
- **`nearest`** in a world without positions picks at random among matches
  instead of always the first.
- Torus neighbor queries now see across the edge.
- Events no longer carry wall-clock `timestamp`s; traces are deterministic.
- `batch_run` gives run *i* the seed `base + i`, where the base defaults
  to the spec's seed (previously every run reused the spec seed, so a
  seeded batch was one run repeated). Runs execute in worker processes.
- CLI: the old `agentstan "prompt"` / `--from-spec` form is replaced by
  subcommands.
- Default `log_level` ("normal") no longer records every agent action and
  state change; use `"detailed"` for those. `"minimal"` now also drops
  interactions.
- Unknown top-level spec keys and agent-type keys are errors.
- An action that repeats its rule's `target` selector acts on the bound
  target instead of selecting again.
- `ai.generate` passes the LLM's spec through without inventing defaults
  (a missing environment is an error to repair, not a grid).

### Deprecated
- `behavior_code` (Python source strings run with `exec`). Use rules, or
  `behaviors=` functions.

### Removed
- `agentstan.defi` (the separate DeFi lending engine).

### Performance
- Expressions compile to closures once; rules compute neighbors lazily;
  non-spatial worlds look up agents by type. A 2,000-agent grid model runs
  at ~48 ms/step (was ~180), a 2,000-agent market at ~13 ms/step (was
  ~340).
