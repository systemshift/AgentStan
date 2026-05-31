"""
Constrained-action LLM behavioral agents.

This is the behavioral layer: an LLM agent at a human decision point (panic,
coordination, whale strategy) chooses *one action from a fixed menu* — never
free-form, never touching the protocol math. Deterministic mechanics
(liquidation, interest, oracle) are untouched; the LLM only decides intent.

A "decider" is any callable ``ctx -> action_str``. ``OpenAIDecider`` is the
default; tests inject a stub. With no decider attached, "llm" agents fall back
to a safe rule, so a model runs (deterministically) even without an API key.

LLM agents are meant to be *few* — a handful of whales or a governance delegate,
not every retail borrower — because each is a model call. Keep populations of
"llm"-policy agents small.
"""

from typing import Any, Callable, Dict, List


# The constrained menus. An LLM may only return one of these per agent type.
ACTION_MENUS: Dict[str, List[str]] = {
    "borrower": ["hold", "repay"],          # repay reduces debt using cash on hand
    "lender": ["hold", "withdraw"],         # withdraw pulls supply (run risk)
    "liquidator": ["hold"],                 # LLM liquidators unsupported for now
}


def _menu(agent_type: str) -> List[str]:
    return ACTION_MENUS.get(agent_type, ["hold"])


def _context(agent, pool, view) -> Dict[str, Any]:
    """The decision context handed to the decider — plain JSON-able data."""
    acct = pool.account(agent.id)
    hf = acct.health_factor(view.oracle_price, pool.config.liquidation_threshold,
                            view.debt_oracle_price)
    drawdown = ((view.initial_price - view.oracle_price) / view.initial_price
                if view.initial_price else 0.0)
    return {
        "agent_type": agent.type,
        "step": view.step,
        "price_drawdown_pct": round(drawdown * 100, 1),
        "utilization_pct": round(view.utilization * 100, 1),
        "your_collateral": round(acct.collateral, 4),
        "your_debt": round(acct.debt, 2),
        "your_health_factor": (round(hf, 3) if hf != float("inf") else None),
        "your_cash": round(agent.wallet_usdc, 2),
        "actions": _menu(agent.type),
    }


def _to_intent(agent, pool, choice: str) -> List[Dict[str, Any]]:
    if choice == "repay":
        amount = min(agent.wallet_usdc, pool.account(agent.id).debt)
        if amount <= 0:
            return []
        return [{"action": "repay", "agent": agent.id, "amount": amount}]
    if choice == "withdraw":
        if agent.wallet_usdc <= 0:
            return []
        return [{"action": "withdraw", "agent": agent.id, "amount": agent.wallet_usdc}]
    return []  # "hold" or anything unrecognized


def rule_fallback(agent, pool, view) -> List[Dict[str, Any]]:
    """Deterministic stand-in approximating a cautious human: lenders run on a
    deep drawdown, borrowers de-lever when health gets thin and they have cash."""
    if agent.type == "lender":
        drawdown = ((view.initial_price - view.oracle_price) / view.initial_price
                    if view.initial_price else 0.0)
        trigger = agent.params.get("drawdown_trigger", 0.15)
        return _to_intent(agent, pool, "withdraw") if drawdown >= trigger else []
    if agent.type == "borrower":
        acct = pool.account(agent.id)
        hf = acct.health_factor(view.oracle_price, pool.config.liquidation_threshold,
                                view.debt_oracle_price)
        if hf < agent.params.get("repay_below_hf", 1.1) and agent.wallet_usdc > 0:
            return _to_intent(agent, pool, "repay")
    return []


def llm_decide(agent, pool, view, decider: Callable[[Dict[str, Any]], str] = None
               ) -> List[Dict[str, Any]]:
    """Resolve one LLM agent's intents. Falls back to a rule when there is no
    decider, the decider errors, or it returns an action outside the menu."""
    if decider is None:
        return rule_fallback(agent, pool, view)
    ctx = _context(agent, pool, view)
    try:
        choice = decider(ctx)
    except Exception:
        return rule_fallback(agent, pool, view)
    if choice not in _menu(agent.type):
        return rule_fallback(agent, pool, view)
    return _to_intent(agent, pool, choice)


_SYSTEM_PROMPT = (
    "You are an agent in a DeFi lending market under stress. Given your position "
    "and current market conditions, choose the single best action for your own "
    "interest. You MUST reply with a JSON object of the form {\"action\": \"<one "
    "of the allowed actions>\"} and nothing else."
)


class OpenAIDecider:
    """Default decider backed by the OpenAI API. Lazily imports ``openai`` so the
    library has no hard AI dependency (install ``agentstan[ai]``)."""

    def __init__(self, model: str = "gpt-5.5", api_key: str = None,
                 base_url: str = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def __call__(self, ctx: Dict[str, Any]) -> str:
        import json
        from openai import OpenAI

        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)

        prompt = (
            f"Your situation:\n{json.dumps(ctx, indent=2)}\n\n"
            f"Allowed actions: {ctx['actions']}. Reply with one."
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("action", "hold")
