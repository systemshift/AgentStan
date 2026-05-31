"""
Parameter sweeps: where does the protocol cross from safe to insolvent?

Two views of the same 40% crash:
  1. Bad-debt-vs-LTV frontier — the single most asked governance question:
     "how high can we set max LTV before a crash creates bad debt?"
  2. A max-LTV x oracle-lag grid — showing the two stress levers interact.

    python examples/defi_param_sweep.py

If matplotlib is installed, PNG charts are written alongside the ASCII output.
"""

from agentstan.defi import MarketConfig, frontier, grid


def base_config():
    return MarketConfig.from_dict({
        "seed": 1,
        "protocol": {
            "max_ltv": 0.70,
            "liquidation_threshold": 0.90,   # fixed; we isolate the LTV effect
            "liquidation_bonus": 0.08,
            "close_factor": 0.5,
            "reserve_factor": 0.10,
            "initial_reserves": 10_000.0,
        },
        "oracle": {"lag_steps": 12},
        "scenario": {"name": "ETH -40%", "initial_price": 3000.0,
                     "crash_pct": 0.40, "crash_start": 10, "crash_duration": 30},
        "populations": [
            {"type": "lender", "count": 1, "policy": "passive",
             "params": {"supply": 150_000.0}},
            {"type": "borrower", "count": 40, "policy": "passive",
             "params": {"collateral": 1.0, "borrow_ltv_fraction": 1.0}},
            {"type": "liquidator", "count": 1, "policy": "greedy",
             "params": {"capital": 30_000.0}},
        ],
    })


def ascii_bars(points, width=40):
    hi = max((v for _, v in points), default=0) or 1
    for value, metric in points:
        bar = "#" * int(round(width * metric / hi))
        print(f"  max_ltv={value:.2f} | {bar:<{width}} ${metric:,.0f}")


def print_grid(rows, row_key, col_key, results):
    """ASCII heatmap of bad debt: rows = row_key values, cols = col_key values."""
    cols = sorted({r["params"][col_key] for r in results})
    lut = {(r["params"][row_key], r["params"][col_key]): r["summary"]["bad_debt"]
           for r in results}
    header = "  " + f"{row_key.split('.')[-1]:>10} | " + " ".join(f"{c:>10}" for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rv in rows:
        cells = " ".join(f"{lut.get((rv, c), 0):>10,.0f}" for c in cols)
        print(f"  {rv:>10.2f} | {cells}")
    print(f"  (rows: {row_key}, cols: {col_key}; cells = bad debt $)")


def maybe_plot(ltv_points, grid_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to also get PNG charts)")
        return

    xs = [v for v, _ in ltv_points]
    ys = [m for _, m in ltv_points]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("max LTV")
    plt.ylabel("bad debt ($)")
    plt.title("Bad debt vs max LTV — ETH -40%, oracle lag 12")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("defi_ltv_frontier.png", dpi=120)
    print("\nwrote defi_ltv_frontier.png")


if __name__ == "__main__":
    base = base_config()

    print("\n=== Bad debt vs max LTV (ETH -40%, oracle lag 12) ===")
    ltv_points = frontier(
        base, "protocol.max_ltv",
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
        metric="bad_debt", steps=120,
    )
    ascii_bars(ltv_points)

    print("\n=== Bad debt: max LTV x oracle lag ===")
    ltv_values = [0.60, 0.70, 0.80]
    grid_results = grid(base, {
        "protocol.max_ltv": ltv_values,
        "oracle.lag_steps": [0, 6, 12, 24],
    }, steps=120)
    print_grid(ltv_values, "protocol.max_ltv", "oracle.lag_steps", grid_results)

    maybe_plot(ltv_points, grid_results)
