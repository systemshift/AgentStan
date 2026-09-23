"""
Command line interface.

    agentstan run model.json                    # a spec or a .pack.json
    agentstan run economy.pack.json gold-rush   # a pack scenario
    agentstan validate economy.pack.json        # full engine validation
    agentstan batch model.json --runs 50 --vary globals.tax=0.05,0.1,0.2
    agentstan generate "a F2P economy with a gold sink" -o economy.pack.json
"""

import argparse
import json
import sys


def _load(path):
    """(pack, None) for a pack file, (None, spec) for a bare spec."""
    from .pack import Pack

    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("format") == "agentstan-pack":
        return Pack(data), None
    return None, data


def _resolve(args):
    """The runnable spec for a file + optional target, and its step count."""
    pack, spec = _load(args.file)
    if pack is not None:
        spec = pack.spec(args.target)
    elif args.target:
        sys.exit("error: a target (model/scenario name) needs a pack file")
    steps = args.steps or spec.pop("steps", None) or 200
    spec.pop("steps", None)
    return spec, steps


def _parse_vary(items):
    """["a.b=1,2,3", ...] -> {"a.b": [1, 2, 3]} (values parsed as JSON)."""
    vary = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"error: --vary expects path=v1,v2,... got {item!r}")
        path, raw = item.split("=", 1)
        values = []
        for token in raw.split(","):
            try:
                values.append(json.loads(token))
            except json.JSONDecodeError:
                values.append(token)
        vary[path] = values
    return vary


def _fmt(v):
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 1e4 else f"{v:,.0f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _print_final(results):
    summary = results["summary"]
    print(f"steps: {results['final_step']}   seed: {results['seed']}")
    if results.get("stopped"):
        stop = results["stopped"]
        print(f"stopped early at step {stop['at_step']}: {stop['reason']}")
    print("agents: " + ", ".join(f"{t}={n}" for t, n in summary["final_counts"].items()))
    history = results["metrics"]["history"]
    final = history[-1] if history else {}
    for label, row in (("observables", final.get("observables")),
                       ("globals", results.get("globals"))):
        if row:
            print(f"{label}: " + ", ".join(f"{k}={_fmt(v)}" for k, v in row.items()))


def cmd_run(args):
    from .core.simulation import Simulation

    spec, steps = _resolve(args)
    sim = Simulation(spec, seed=args.seed)
    results = sim.run(steps, max_agents=args.max_agents, time_limit=args.time_limit)
    _print_final(results)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"results written to {args.output}")


def cmd_validate(args):
    from .core.simulation import Simulation

    pack, spec = _load(args.file)
    try:
        if pack is not None:
            pack.validate(deep=True, smoke_steps=args.smoke_steps)
            names = pack.models + pack.scenarios
        else:
            Simulation.check(spec, smoke_steps=args.smoke_steps)
            names = ["spec"]
    except Exception as e:
        print(f"invalid: {e}")
        sys.exit(1)
    print(f"ok: {', '.join(names)}")


def cmd_batch(args):
    from .experiment import batch_run, summarize

    spec, steps = _resolve(args)
    vary = _parse_vary(args.vary)
    runs = batch_run(spec, n_runs=args.runs, steps=steps, vary=vary or None,
                     seed=args.seed, max_workers=args.workers,
                     max_agents=args.max_agents, time_limit=args.time_limit)

    groups = {}
    for run in runs:
        key = tuple(sorted(run["params"].items()))
        groups.setdefault(key, []).append(run)

    for key, group in groups.items():
        if key:
            print(", ".join(f"{p}={_fmt(v)}" for p, v in key))
        report = summarize(group)
        print(f"  {report['runs']} runs"
              + (f", {report['stopped_early']:.0%} stopped early" if report["stopped_early"] else ""))
        for name, stats in report["metrics"].items():
            print(f"  {name:<24} mean {_fmt(stats['mean']):>12}   "
                  f"p5 {_fmt(stats['p5']):>12}   p95 {_fmt(stats['p95']):>12}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(runs, f, indent=2, default=str)
        print(f"results written to {args.output}")


def cmd_generate(args):
    from .ai.generate import generate
    from .pack import Pack

    spec = generate(args.prompt, model=args.model, base_url=args.base_url)
    name = spec.get("metadata", {}).get("name") or "model"
    pack = Pack.new(name, spec, description=spec.get("metadata", {}).get("description", ""))
    text = pack.to_json()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"{name}: written to {args.output} "
              f"(agents: {', '.join(spec['agent_types'])})")
    else:
        print(text)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="agentstan",
        description="AgentStan: declarative agent-based modeling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_run_args(p):
        p.add_argument("file", help="spec JSON or .pack.json")
        p.add_argument("target", nargs="?", help="model or scenario name (packs)")
        p.add_argument("--steps", type=int, help="steps (default: the spec's, else 200)")
        p.add_argument("--seed", type=int, help="seed (default: the spec's)")
        p.add_argument("--max-agents", type=int, help="stop if the population exceeds this")
        p.add_argument("--time-limit", type=float, help="stop after this many seconds")
        p.add_argument("--output", "-o", help="write full results JSON here")

    p = sub.add_parser("run", help="run a model once")
    add_run_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("validate", help="validate a spec or pack (constructs and smoke-runs)")
    p.add_argument("file")
    p.add_argument("--smoke-steps", type=int, default=10)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("batch", help="run many times and summarize distributions")
    add_run_args(p)
    p.add_argument("--runs", type=int, default=20, help="runs per parameter combination")
    p.add_argument("--vary", action="append", metavar="PATH=V1,V2",
                   help="dot-path parameter values (repeatable)")
    p.add_argument("--workers", type=int, help="worker processes (default: CPUs)")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("generate", help="generate a model from a description (needs agentstan[ai])")
    p.add_argument("prompt")
    p.add_argument("--model", help="LLM model (default: $AGENTSTAN_MODEL, else gpt-5.5)")
    p.add_argument("--base-url", help="OpenAI-compatible API base URL")
    p.add_argument("--output", "-o", help="write the pack here (default: stdout)")
    p.set_defaults(func=cmd_generate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
