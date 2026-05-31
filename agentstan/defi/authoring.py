"""
Natural-language authoring: English -> a validated MarketConfig.

This is the self-serve front door. A protocol team describes their market and
the shock they fear in plain language; this returns a runnable, validated
config. It is OSS — the cloud chat experience that wraps it (conversation,
iterative tweaking, saved projects, paying the LLM bill) is the paid layer.

Design: the LLM does NOT emit a full nested config (error-prone). It emits a
compact *intent* — which built-in scenario, plus a few numeric overrides — and
deterministic code assembles and validates the MarketConfig from the scenario
library. A keyword heuristic provides the same intent with no API key, so the
function always returns something usable and is deterministic under test.

    from agentstan.defi import author_config, LendingMarket

    cfg = author_config("USDC market, 82.5% liquidation threshold, "
                        "stress a 40% ETH crash with a slow oracle")
    results = LendingMarket(cfg).run(120)
"""

import inspect
import re
from typing import Any, Callable, Dict, Optional

from .config import MarketConfig
from . import scenarios


# An intent is: {"scenario": <name>, "protocol": {overrides}, "scenario_params": {...}}
Intent = Dict[str, Any]


# --- assembling a config from an intent ----------------------------------

def _filter_kwargs(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs the chosen preset accepts (presets differ, e.g. eth_crash
    takes ``oracle_lag`` while oracle_delay takes ``lag``)."""
    fn = scenarios.SCENARIOS[name]
    accepted = set(inspect.signature(fn).parameters)
    return {k: v for k, v in params.items() if k in accepted}


def assemble(intent: Intent, base: Optional[MarketConfig] = None) -> MarketConfig:
    """Turn an intent into a validated MarketConfig via the scenario library."""
    scenario = intent.get("scenario", "eth_crash")
    if scenario not in scenarios.SCENARIOS:
        scenario = "eth_crash"

    base_cfg = base or scenarios.default_market()
    d = base_cfg.to_dict()
    for key, value in (intent.get("protocol") or {}).items():
        if key in d["protocol"]:
            d["protocol"][key] = value
    base_cfg = MarketConfig.from_dict(d)

    params = _filter_kwargs(scenario, intent.get("scenario_params") or {})
    cfg = scenarios.apply(scenario, base_cfg, **params)

    issues = cfg.validate()
    if issues:
        # Light repair of the common violation: threshold must exceed max LTV.
        d = cfg.to_dict()
        if d["protocol"]["liquidation_threshold"] <= d["protocol"]["max_ltv"]:
            d["protocol"]["liquidation_threshold"] = min(0.99, d["protocol"]["max_ltv"] + 0.05)
        cfg = MarketConfig.from_dict(d)
        issues = cfg.validate()
        if issues:
            raise ValueError("could not build a valid config: " + "; ".join(issues))
    return cfg


# --- keyword heuristic (no-API fallback) ---------------------------------

_SCENARIO_KEYWORDS = [
    ("stablecoin_depeg", ("depeg", "de-peg", "peg", "stablecoin")),
    ("whale_panic", ("whale",)),
    ("liquidity_withdrawal", ("withdraw", "bank run", "run on", "outflow")),
    ("liquidation_congestion", ("congestion", "congested", "overwhelm")),
    ("oracle_delay", ("oracle", "stale", "lag", "delay")),
    ("eth_crash", ("crash", "drop", "fall", "decline", "down")),
]


def _nearest(text: str, anchor: str, token_re: str):
    """Return the numeric token (matched by ``token_re``) whose position is
    closest to any occurrence of ``anchor`` — robust to several numbers being in
    the same sentence (e.g. an LTV and a threshold)."""
    anchors = [m.start() for m in re.finditer(anchor, text)]
    if not anchors:
        return None
    best, best_dist = None, None
    for tok in re.finditer(token_re, text):
        dist = min(abs(tok.start() - a) for a in anchors)
        if best_dist is None or dist < best_dist:
            best, best_dist = tok.group(1), dist
    return best


def _find_pct(text: str, anchor: str) -> Optional[float]:
    """Percentage closest to an anchor word, returned as a fraction."""
    val = _nearest(text, anchor, r"(\d+(?:\.\d+)?)\s*%")
    return float(val) / 100.0 if val is not None else None


def _find_int(text: str, anchor: str) -> Optional[int]:
    val = _nearest(text, anchor, r"(\d+)\s*(?:step|block|min|hour|second)?s?")
    return int(val) if val is not None else None


def heuristic_intent(description: str) -> Intent:
    """Crude keyword/number extraction. A fallback, not the product — the LLM
    path is the real authoring experience."""
    text = description.lower()

    scenario = "eth_crash"
    for name, keywords in _SCENARIO_KEYWORDS:
        if any(k in text for k in keywords):
            scenario = name
            break

    protocol: Dict[str, Any] = {}
    ltv = _find_pct(text, "ltv")
    if ltv is not None:
        protocol["max_ltv"] = ltv
    threshold = _find_pct(text, "threshold")
    if threshold is not None:
        protocol["liquidation_threshold"] = threshold

    params: Dict[str, Any] = {}
    severity = (_find_pct(text, "crash") or _find_pct(text, "drop")
                or _find_pct(text, "fall") or _find_pct(text, "depeg"))
    if severity is not None:
        params["severity"] = severity
    lag = _find_int(text, "lag") or _find_int(text, "delay")
    if lag is not None:
        # eth_crash names it oracle_lag; oracle_delay names it lag — set both,
        # assemble() filters to whichever the chosen preset accepts.
        params["oracle_lag"] = lag
        params["lag"] = lag

    return {"scenario": scenario, "protocol": protocol, "scenario_params": params}


# --- LLM-backed author ----------------------------------------------------

_SYSTEM_PROMPT = (
    "You translate a DeFi lending stress-test request into a compact JSON intent. "
    "Reply ONLY with a JSON object of this shape:\n"
    '{"scenario": <one of: %s>,\n'
    ' "protocol": {optional overrides: max_ltv, liquidation_threshold, '
    "liquidation_bonus, close_factor, reserve_factor (all fractions 0-1)},\n"
    ' "scenario_params": {optional: severity (fraction), oracle_lag or lag (int steps)}}\n'
    "Pick the single best scenario. liquidation_threshold must exceed max_ltv."
) % ", ".join(sorted(scenarios.SCENARIOS))


class OpenAIConfigAuthor:
    """Default LLM author. Lazily imports ``openai`` (install ``agentstan[ai]``).
    Callable: ``author(description) -> intent dict``."""

    def __init__(self, model: str = "gpt-5.5", api_key: str = None, base_url: str = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def __call__(self, description: str) -> Intent:
        import json
        from openai import OpenAI

        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)


def author_config(description: str, *,
                  generator: Optional[Callable[[str], Intent]] = None,
                  base: Optional[MarketConfig] = None) -> MarketConfig:
    """Build a validated MarketConfig from a natural-language description.

    Args:
        description: plain-language market + shock description.
        generator: callable ``description -> intent dict``. Defaults to an
            OpenAI-backed author; if ``openai`` is unavailable or the call fails,
            falls back to a keyword heuristic so the function always returns a
            usable config. Tests inject a stub generator.
        base: optional base market to apply overrides/scenario onto.
    """
    intent: Optional[Intent] = None
    if generator is not None:
        intent = generator(description)
    else:
        try:
            intent = OpenAIConfigAuthor()(description)
        except Exception:
            intent = heuristic_intent(description)
    if not isinstance(intent, dict):
        intent = heuristic_intent(description)
    return assemble(intent, base=base)
