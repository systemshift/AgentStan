"""
The scenario library: one market, six distinct failure modes.

Runs every built-in stress preset against the same baseline protocol and prints
how each breaks — some via bad debt (insolvency), some via a liquidity freeze
(min available liquidity hits zero) without any bad debt at all. Different
shocks fail differently, and the metrics show which.

Also demonstrates an LLM-driven whale: its panic decision is delegated to a
pluggable "decider" (a stub here; swap in agentstan.defi.OpenAIDecider with an
API key for a real model).

    python examples/defi_scenarios.py
"""

from agentstan.defi import LendingMarket, MarketConfig, scenarios


def run_all():
    print(f"{'scenario':<24} {'outcome':<10} {'bad debt':>12} {'min liquidity':>14} {'liquidations':>13}")
    print("-" * 76)
    for name in scenarios.SCENARIOS:
        s = LendingMarket(scenarios.apply(name)).run(120)["summary"]
        outcome = "INSOLVENT" if s["insolvent"] else "solvent"
        print(f"{name:<24} {outcome:<10} ${s['bad_debt']:>11,.0f} "
              f"${s['min_available_liquidity']:>13,.0f} {s['total_liquidations']:>13}")
    print("\nNote: whale_panic and liquidity_withdrawal stay solvent but freeze "
          "liquidity (min liquidity -> $0) — a different failure mode than bad debt.")


def llm_whale_demo():
    """Same whale-panic market, but the whale's decision is made by a decider."""
    spec = scenarios.whale_panic().to_dict()
    for p in spec["populations"]:
        if p["type"] == "lender" and p["params"].get("supply", 0) >= 100_000:
            p["policy"] = "llm"

    print("\n=== LLM-driven whale (decider chooses hold/withdraw each step) ===")
    deciders = {
        "always holds": lambda ctx: "hold",
        "panics at >5% drawdown": lambda ctx: "withdraw" if ctx["price_drawdown_pct"] > 5 else "hold",
    }
    for label, decider in deciders.items():
        m = LendingMarket(MarketConfig.from_dict(spec))
        m.attach_llm(decider)
        s = m.run(120)["summary"]
        print(f"  whale {label:<26} -> min liquidity ${s['min_available_liquidity']:,.0f}")
    print("  (swap in agentstan.defi.OpenAIDecider for a real LLM with OPENAI_API_KEY)")


if __name__ == "__main__":
    run_all()
    llm_whale_demo()
