"""
Declarative behavior rules: agent behaviors as pure JSON data, no code.

This is the preferred way to define behaviors. A rules behavior is fully
serializable, safe to accept from an LLM or an untrusted user, and portable
to any engine that implements this spec (the Python engine is the reference
implementation).

Spec shape (inside an agent type):

    "behavior": {
        "rules": [
            {"when": {"<": ["$energy", 20]},
             "do": [{"type": "modify_state", "attribute": "energy", "delta": 2}]},

            {"when": {">": [{"count": {"type": "wolf"}}, 0]},
             "do": [{"type": "move_away", "from": {"nearest": {"type": "wolf"}}}]},

            {"when": {">": ["$energy", 30]}, "prob": 0.08,
             "do": [{"type": "reproduce", "energy_cost": 15}]},

            {"do": [{"type": "modify_state", "attribute": "energy", "delta": -0.8}]},

            {"when": {"<=": ["$energy", 0]},
             "do": [{"type": "die", "cause": "starvation"}]}
        ]
    }

Semantics: every rule whose ``when`` is true (and whose ``prob`` passes)
fires, in order. A rule with no ``when`` always fires. Actions are the
engine's standard action dicts; any field may be an expression.

Expressions
-----------
- Numbers, booleans: literals.
- ``"$name"``: the agent's own attribute (``"$energy"``, ``"$position"``).
- ``"@step"``: current simulation step.
- Strings not starting with ``$``/``@``: literal strings.
- Single-key dicts are operators:
    {"+": [a, b, ...]}  {"-": [a, b]}  {"*": [a, b, ...]}  {"/": [a, b]}
    {"<": [a, b]} {"<=": [a, b]} {">": [a, b]} {">=": [a, b]}
    {"==": [a, b]} {"!=": [a, b]}
    {"and": [...]} {"or": [...]} {"not": x}
    {"min": [...]} {"max": [...]} {"abs": x}
    {"random": []}            -> float in [0, 1)
    {"uniform": [a, b]}       -> float in [a, b]
    {"randint": [a, b]}       -> int in [a, b]
    {"choice": [v1, v2, ...]} -> pick one (literal values)
    {"count": QUERY}          -> number of nearby agents matching QUERY
    {"nearest_distance": QUERY} -> distance to nearest match (inf if none)
    {"total": "type"}         -> global living count of an agent type

Queries select among *nearby* agents (within perception_radius):
    {}                  -> any nearby agent
    {"type": "wolf"}    -> nearby agents of that type

Selectors resolve a query to one agent (used in action targets):
    {"nearest": QUERY}  -> closest match, or None
    {"random": QUERY}   -> uniformly random match, or None

Extra actions (compiled down to engine actions):
    {"type": "move_toward", "target": SELECTOR}   -> move_to its position
    {"type": "move_away", "from": SELECTOR}       -> step away from it
    {"type": "interact", "target": SELECTOR, ...} -> interact with target_id
"""

import math
from typing import Any, Dict, List, Optional

_INF = float("inf")

_BINARY_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

_KNOWN_OPS = set(_BINARY_OPS) | {
    "+", "-", "*", "/", "and", "or", "not", "min", "max", "abs",
    "random", "uniform", "randint", "choice",
    "count", "nearest_distance", "total",
}

_ACTION_TYPES = {
    "move", "move_to", "move_random", "move_toward", "move_away",
    "interact", "reproduce", "die", "modify_state", "transform", "custom",
}


class RuleError(ValueError):
    """A rules spec is malformed. Message says which rule and why."""


def validate_rules(rules: Any, agent_type: str = "?") -> None:
    """Validate a rules list, raising RuleError with a precise message."""
    if not isinstance(rules, list):
        raise RuleError(
            f"agent_types['{agent_type}'].behavior.rules must be a list, "
            f"got {type(rules).__name__}"
        )
    for i, rule in enumerate(rules):
        where = f"agent_types['{agent_type}'].behavior.rules[{i}]"
        if not isinstance(rule, dict):
            raise RuleError(f"{where} must be a dict, got {type(rule).__name__}")
        unknown = set(rule) - {"when", "prob", "do"}
        if unknown:
            raise RuleError(
                f"{where} has unknown keys {sorted(unknown)} — "
                f"allowed: 'when', 'prob', 'do'"
            )
        if "do" not in rule:
            raise RuleError(f"{where} missing 'do' (list of actions)")
        if not isinstance(rule["do"], list):
            raise RuleError(f"{where}.do must be a list of action dicts")
        for j, action in enumerate(rule["do"]):
            if not isinstance(action, dict) or "type" not in action:
                raise RuleError(
                    f"{where}.do[{j}] must be a dict with a 'type' key"
                )
            if action["type"] not in _ACTION_TYPES:
                raise RuleError(
                    f"{where}.do[{j}] unknown action type "
                    f"'{action['type']}' — options: {sorted(_ACTION_TYPES)}"
                )
        if "when" in rule:
            _validate_expr(rule["when"], f"{where}.when")


def _validate_expr(expr: Any, where: str) -> None:
    """Shallow structural validation of an expression tree."""
    if isinstance(expr, dict):
        if len(expr) != 1:
            raise RuleError(
                f"{where}: operator dicts must have exactly one key, "
                f"got {sorted(expr)}"
            )
        op = next(iter(expr))
        if op not in _KNOWN_OPS:
            raise RuleError(
                f"{where}: unknown operator '{op}' — options: {sorted(_KNOWN_OPS)}"
            )
        args = expr[op]
        if isinstance(args, list) and op not in (
            "choice", "count", "nearest_distance", "total"
        ):
            for k, sub in enumerate(args):
                _validate_expr(sub, f"{where}.{op}[{k}]")


class _Context:
    """Everything an expression can see during one agent's decision."""

    __slots__ = ("agent", "nearby", "rng", "env", "step", "counts")

    def __init__(self, agent, nearby, rng, env, step, counts):
        self.agent = agent
        self.nearby = nearby
        self.rng = rng
        self.env = env
        self.step = step
        self.counts = counts


def _query(query: Dict, ctx: _Context) -> List:
    """Filter nearby living agents by query."""
    if not isinstance(query, dict):
        raise RuleError(f"query must be a dict, got {query!r}")
    wanted = query.get("type")
    out = []
    for other in ctx.nearby:
        if not other.alive:
            continue
        if wanted is not None and other.type != wanted:
            continue
        out.append(other)
    return out


def _select(selector: Any, ctx: _Context):
    """Resolve a selector ({"nearest": q} / {"random": q}) to an agent or None."""
    if not isinstance(selector, dict) or len(selector) != 1:
        raise RuleError(
            f"selector must be {{'nearest': query}} or {{'random': query}}, "
            f"got {selector!r}"
        )
    kind = next(iter(selector))
    matches = _query(selector[kind], ctx)
    if not matches:
        return None
    if kind == "random":
        return ctx.rng.choice(matches)
    if kind == "nearest":
        my_pos = ctx.agent.state.get("position")
        if my_pos is None:
            return matches[0]
        return min(
            matches,
            key=lambda a: ctx.env.distance(my_pos, a.state.get("position"))
            if a.state.get("position") is not None else _INF,
        )
    raise RuleError(f"unknown selector '{kind}' — options: 'nearest', 'random'")


def evaluate(expr: Any, ctx: _Context) -> Any:
    """Evaluate an expression tree against a context."""
    if isinstance(expr, (int, float, bool)) or expr is None:
        return expr

    if isinstance(expr, str):
        if expr.startswith("$"):
            return ctx.agent.state.get(expr[1:])
        if expr == "@step":
            return ctx.step
        return expr

    if isinstance(expr, list):
        return [evaluate(e, ctx) for e in expr]

    if isinstance(expr, dict):
        if len(expr) != 1:
            raise RuleError(
                f"operator dicts must have exactly one key, got {sorted(expr)}"
            )
        op = next(iter(expr))
        args = expr[op]

        if op in _BINARY_OPS:
            a, b = (evaluate(x, ctx) for x in args)
            return _BINARY_OPS[op](a, b)
        if op == "+":
            return sum(evaluate(x, ctx) for x in args)
        if op == "-":
            a, b = (evaluate(x, ctx) for x in args)
            return a - b
        if op == "*":
            out = 1
            for x in args:
                out *= evaluate(x, ctx)
            return out
        if op == "/":
            a, b = (evaluate(x, ctx) for x in args)
            return a / b
        if op == "and":
            return all(evaluate(x, ctx) for x in args)
        if op == "or":
            return any(evaluate(x, ctx) for x in args)
        if op == "not":
            return not evaluate(args, ctx)
        if op == "min":
            return min(evaluate(x, ctx) for x in args)
        if op == "max":
            return max(evaluate(x, ctx) for x in args)
        if op == "abs":
            return abs(evaluate(args, ctx))
        if op == "random":
            return ctx.rng.random()
        if op == "uniform":
            a, b = (evaluate(x, ctx) for x in args)
            return ctx.rng.uniform(a, b)
        if op == "randint":
            a, b = (evaluate(x, ctx) for x in args)
            return ctx.rng.randint(a, b)
        if op == "choice":
            return ctx.rng.choice(args)
        if op == "count":
            return len(_query(args, ctx))
        if op == "nearest_distance":
            target = _select({"nearest": args}, ctx)
            if target is None:
                return _INF
            my_pos = ctx.agent.state.get("position")
            other_pos = target.state.get("position")
            if my_pos is None or other_pos is None:
                return _INF
            return ctx.env.distance(my_pos, other_pos)
        if op == "total":
            return ctx.counts.get(args, 0)
        raise RuleError(f"unknown operator '{op}'")

    raise RuleError(f"cannot evaluate expression: {expr!r}")


def _sign_away(mine: float, other: float, rng) -> int:
    """Direction component pointing away from `other`."""
    if mine > other:
        return 1
    if mine < other:
        return -1
    return rng.choice([-1, 1])


def _compile_action(action: Dict, ctx: _Context) -> Optional[Dict]:
    """Turn one action template into a concrete engine action (or None)."""
    a_type = action["type"]

    if a_type == "move_toward":
        target = action.get("target")
        if isinstance(target, dict):
            other = _select(target, ctx)
            if other is None:
                return None
            target = other.state.get("position")
        if target is None:
            return None
        return {"type": "move_to", "target": tuple(target)}

    if a_type == "move_away":
        other = _select(action.get("from"), ctx)
        if other is None:
            return None
        my_pos = ctx.agent.state.get("position")
        other_pos = other.state.get("position")
        if my_pos is None or other_pos is None:
            return None
        dx = _sign_away(my_pos[0], other_pos[0], ctx.rng)
        dy = _sign_away(my_pos[1], other_pos[1], ctx.rng)
        return {"type": "move", "direction": [dx, dy]}

    if a_type == "interact" and isinstance(action.get("target"), dict):
        other = _select(action["target"], ctx)
        if other is None:
            return None
        out = {k: v for k, v in action.items() if k != "target"}
        out["target_id"] = other.id
        return _evaluate_fields(out, ctx)

    return _evaluate_fields(action, ctx)


def _evaluate_fields(action: Dict, ctx: _Context) -> Dict:
    """Evaluate expression-valued fields in a plain action dict."""
    out = {}
    for key, value in action.items():
        if key == "type":
            out[key] = value
        elif isinstance(value, dict) and len(value) == 1 and next(iter(value)) in _KNOWN_OPS:
            out[key] = evaluate(value, ctx)
        elif isinstance(value, str) and (value.startswith("$") or value == "@step"):
            out[key] = evaluate(value, ctx)
        elif isinstance(value, list):
            out[key] = [
                evaluate(v, ctx)
                if isinstance(v, (dict, str)) and not isinstance(v, bool)
                else v
                for v in value
            ]
        else:
            out[key] = value
    return out


class RuleBehavior:
    """
    A behavior function compiled from declarative rules.

    Callable with the standard behavior signature
    ``(agent, sim_state, agents_nearby) -> actions`` so the engine treats it
    exactly like a Python behavior function.
    """

    def __init__(self, rules: List[Dict], simulation, agent_type: str = "?"):
        validate_rules(rules, agent_type)
        self.rules = rules
        self.simulation = simulation
        self.agent_type = agent_type

    def __call__(self, agent, sim_state, agents_nearby) -> List[Dict]:
        ctx = _Context(
            agent=agent,
            nearby=agents_nearby,
            rng=self.simulation.rng,
            env=self.simulation.environment,
            step=sim_state.get("step", 0),
            counts=sim_state.get("agent_counts", {}),
        )

        actions = []
        for rule in self.rules:
            when = rule.get("when")
            if when is not None and not evaluate(when, ctx):
                continue
            prob = rule.get("prob")
            if prob is not None and ctx.rng.random() >= evaluate(prob, ctx):
                continue
            for action in rule["do"]:
                compiled = _compile_action(action, ctx)
                if compiled is not None:
                    actions.append(compiled)
        return actions
