"""
Generate a governance-style stress-test report.

Runs the headline scenario, computes the safe-LTV frontier and an
LTV x oracle-lag grid, then writes a Markdown report a risk committee can read.
This is the artifact you hand to a protocol team or DAO.

    python examples/defi_report.py        # writes defi_stress_report.md
"""

from agentstan.defi import LendingMarket, MarketConfig, frontier, grid
from agentstan.defi.report import write_report


def headline_config():
    return MarketConfig.from_dict({
        "seed": 1,
        "protocol": {
            "max_ltv": 0.80, "liquidation_threshold": 0.90,
            "liquidation_bonus": 0.08, "close_factor": 0.5,
            "reserve_factor": 0.10, "initial_reserves": 10_000.0,
        },
        "oracle": {"lag_steps": 12},
        "scenario": {"name": "ETH -40% with oracle lag", "initial_price": 3000.0,
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


if __name__ == "__main__":
    cfg = headline_config()
    steps = 120

    headline = LendingMarket(cfg).run(steps)
    ltv_pts = frontier(cfg, "protocol.max_ltv",
                       [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
                       metric="bad_debt", steps=steps)
    lag_grid = grid(cfg, {
        "protocol.max_ltv": [0.60, 0.70, 0.80],
        "oracle.lag_steps": [0, 6, 12, 24],
    }, steps=steps)

    md = write_report(
        "defi_stress_report.md", headline,
        title="Lending market stress test — ETH -40% scenario",
        ltv_frontier=ltv_pts, lag_grid=lag_grid,
        chart_path="defi_stress_frontier.png",
    )
    print(md)
    print("\n--- wrote defi_stress_report.md ---")
