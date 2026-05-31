"""
The wedge demo: an Aave-like lending market under a 40% ETH crash.

Same shock, two parameter sets. The risky one — high LTV, a slow oracle, and
under-capitalized liquidators — produces bad debt. The safe one survives the
exact same crash with little or none. That contrast is the whole pitch:
"these parameters break; these don't."

    python examples/defi_eth_crash.py
"""

from agentstan.defi import MarketConfig


def build_config(name, *, max_ltv, liq_threshold, oracle_lag, liquidator_capital,
                 liquidator_count, n_borrowers=40, supply=120_000.0):
    return MarketConfig.from_dict({
        "seed": 1,
        "steps_per_year": 8760,
        "protocol": {
            "max_ltv": max_ltv,
            "liquidation_threshold": liq_threshold,
            "liquidation_bonus": 0.08,
            "close_factor": 0.5,
            "reserve_factor": 0.10,
            "initial_reserves": 5_000.0,
            "rate_model": {"base_rate": 0.0, "slope1": 0.04, "slope2": 0.75,
                           "optimal_utilization": 0.80},
        },
        "oracle": {"lag_steps": oracle_lag},
        "scenario": {
            "name": name,
            "initial_price": 3000.0,
            "crash_pct": 0.40,      # ETH -40%
            "crash_start": 10,
            "crash_duration": 30,   # gradual, over 30 hourly steps
        },
        "populations": [
            {"type": "lender", "count": 1, "policy": "passive",
             "params": {"supply": supply}},
            {"type": "borrower", "count": n_borrowers, "policy": "passive",
             "params": {"collateral": 1.0, "borrow_ltv_fraction": 1.0}},
            {"type": "liquidator", "count": liquidator_count, "policy": "greedy",
             "params": {"capital": liquidator_capital}},
        ],
    })


def report(label, results):
    s = results["summary"]
    print(f"\n=== {label} ===")
    print(f"  bad debt:               ${s['bad_debt']:,.0f}")
    print(f"  insolvent:              {s['insolvent']}")
    print(f"  reserves remaining:     ${s['reserves_remaining']:,.0f}")
    print(f"  liquidations:           {s['total_liquidations']}")
    print(f"  liquidation volume:     ${s['total_liquidation_volume']:,.0f}")
    print(f"  peak utilization:       {s['max_utilization']:.0%}")
    print(f"  min available liquidity:${s['min_available_liquidity']:,.0f}")


if __name__ == "__main__":
    from agentstan.defi import LendingMarket

    steps = 120

    risky = build_config(
        "ETH crash 40% (risky params)",
        max_ltv=0.80, liq_threshold=0.85,
        oracle_lag=24,                 # ~24 steps of stale prices
        liquidator_capital=15_000.0,   # under-capitalized
        liquidator_count=1,
    )
    safe = build_config(
        "ETH crash 40% (safe params)",
        max_ltv=0.60, liq_threshold=0.75,
        oracle_lag=1,                  # near-real-time feed
        liquidator_capital=500_000.0,  # deep liquidator capital
        liquidator_count=3,
    )

    report("RISKY", LendingMarket(risky).run(steps))
    report("SAFE", LendingMarket(safe).run(steps))
