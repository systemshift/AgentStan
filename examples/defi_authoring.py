"""
Natural-language authoring: describe a stress test in English, get a config.

This is the self-serve front door — the OSS function the cloud chat will wrap.
Runs here with the no-API keyword heuristic so it works offline; for real
authoring quality, pass a generator backed by an LLM.

    python examples/defi_authoring.py
"""

from agentstan.defi import author_config, LendingMarket


PROMPTS = [
    "USDC market, 82.5% liquidation threshold, max LTV 75%, stress a 40% ETH "
    "crash with an oracle lag of 20 steps",
    "what happens if a whale pulls its liquidity during a 15% dip",
    "stablecoin collateral depeg scenario",
    "conservative protocol, 60% max LTV, hit it with a fast 50% crash and thin "
    "liquidators",
]


def main():
    for prompt in PROMPTS:
        cfg = author_config(prompt)  # no generator -> heuristic (offline)
        s = LendingMarket(cfg).run(120)["summary"]
        outcome = "INSOLVENT" if s["insolvent"] else "solvent"
        print(f"> {prompt}")
        print(f"    scenario={cfg.scenario.name!r}  max_ltv={cfg.protocol.max_ltv:.2f}  "
              f"liq_threshold={cfg.protocol.liquidation_threshold:.3f}  oracle_lag={cfg.oracle.lag_steps}")
        print(f"    => {outcome}: bad debt ${s['bad_debt']:,.0f}, "
              f"min liquidity ${s['min_available_liquidity']:,.0f}\n")

    print("For production authoring, pass an LLM generator:")
    print("    from agentstan.defi import OpenAIConfigAuthor, author_config")
    print("    cfg = author_config(prompt, generator=OpenAIConfigAuthor())")


if __name__ == "__main__":
    main()
