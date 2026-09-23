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

            {"target": {"lowest": {"type": "shop", "by": "&price",
                                   "where": {">": ["&stock", 0]}}},
             "when": {">=": ["$gold", "&price"]},
             "do": [{"type": "interact", "interaction_type": "buy",
                     "params": {"exchange": {"give": {"gold": "&price"},
                                             "get": {"stock": 1}}}}]}
        ]
    }

Top-level spec keys that use the same language:

    "globals":     {"price": 10, "treasury": 0}   shared world state
    "world_rules": [...]   rules run ONCE per step, before agents act; they
                           see globals and aggregates, never "$" or neighbors;
                           actions: modify_global, spawn
    "global_rules": [...]  rules applied to EVERY living agent each step,
                           after behaviors (world laws such as death at 0)
    "observables": {"gold_supply": {"sum": {"type": "player", "attr": "gold"}}}
                           expressions recorded in history every step

Semantics
---------
Every rule whose ``when`` is true (and whose ``prob`` passes) fires, in
order. A rule with no ``when`` always fires.

An agent's rules are all evaluated first, against the world as it is at
that moment, and the resulting actions are then applied in order. So a
rule that reads ``$energy`` sees the value from before this step's
``modify_state`` actions, not after.

Globals, counts and aggregates are live: they reflect every action applied
so far this step (agents act one after another, in scheduler order).

A rule with ``target`` first selects one agent; if none matches, the rule
does not fire. Inside that rule, ``"&attr"`` reads the target's attributes,
and ``interact`` / ``move_toward`` / ``move_away`` act on it by default.

Expressions
-----------
- Numbers, booleans: literals.
- ``"$name"``: the agent's own attribute (``"$energy"``, ``"$position"``).
- ``"&name"``: the other agent's attribute — the rule's target, or the
  candidate being tested inside a query's ``where`` / ``by``.
- ``"@step"``: current simulation step. ``"@name"``: a declared global.
- Strings not starting with ``$``/``&``/``@``: literal strings.
- Single-key dicts are operators:
    {"+": [a, b, ...]}  {"-": [a, b]}  {"*": [a, b, ...]}  {"/": [a, b]}  {"%": [a, b]}
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
    {"total": "type" | QUERY} -> global number of living matches
    {"sum": AGG}  {"mean": AGG} -> global sum / mean of an attribute
        AGG = {"attr": "gold", "type": "player", "where": EXPR}
        (type and where optional; agents lacking the attribute are
        skipped; the mean of nothing is 0)

Queries select among *nearby* agents (within perception_radius; in a
non-spatial world every other agent is nearby):
    {}                                     -> any nearby agent
    {"type": "wolf"}                       -> nearby agents of that type
    {"type": "shop", "where": {">": ["&stock", 0]}}  -> filtered

Selectors resolve a query to one agent:
    {"nearest": QUERY}  -> closest match (random among matches when the
                           world has no positions)
    {"random": QUERY}   -> uniformly random match
    {"lowest": {..QUERY.., "by": EXPR}}   -> match minimizing EXPR
    {"highest": {..QUERY.., "by": EXPR}}  -> match maximizing EXPR

Actions (any field may be an expression, at any depth)
-------
    {"type": "move", "direction": [dx, dy]}
    {"type": "move_to", "target": [x, y]}
    {"type": "move_random"}
    {"type": "move_toward", "target": SELECTOR | position}
    {"type": "move_away", "from": SELECTOR}
    {"type": "modify_state", "attribute": a, "value": v | "delta": d}
    {"type": "modify_global", "name": g, "value": v | "delta": d}
    {"type": "interact", "target": SELECTOR, "interaction_type": name,
     "params": {success_rate, kill_target, self_delta, target_delta,
                transfer, exchange}}
    {"type": "reproduce", "cost": {"attribute": a, "amount": n},
     "offspring_count": k, "offspring_state": {...}}
    {"type": "spawn", "agent_type": t, "count": n, "state": {...}}
    {"type": "transform", "new_type": t, "new_state": {...}}
    {"type": "die", "cause": c}
    {"type": "custom", "details": {...}}
"""

from typing import Any, Dict, List, Optional

_INF = float("inf")


def _cmp(op):
    """Comparison that treats a missing attribute (None) as 'no match'.

    A rule like {"<=": ["$energy", 0]} simply doesn't fire for agents that
    have no energy attribute, instead of crashing the run.
    """
    def compare(a, b):
        if op in ("==", "!="):
            return (a == b) if op == "==" else (a != b)
        if a is None or b is None:
            return False
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        return a >= b
    return compare


_BINARY_OPS = {op: _cmp(op) for op in ("<", "<=", ">", ">=", "==", "!=")}

# operator -> arity: an int (exact list length), "n" (non-empty list),
# "1" (single operand, not a list), or a special shape
_OP_SHAPES = {
    **{op: 2 for op in _BINARY_OPS},
    "+": "n", "-": 2, "*": "n", "/": 2, "%": 2,
    "and": "n", "or": "n", "not": "1",
    "min": "n", "max": "n", "abs": "1",
    "random": 0, "uniform": 2, "randint": 2, "choice": "literal",
    "count": "query", "nearest_distance": "query", "total": "total",
    "sum": "agg", "mean": "agg",
}
_KNOWN_OPS = set(_OP_SHAPES)

_SELECTORS = ("nearest", "random", "lowest", "highest")

# Fields whose value is a {attribute: expression} map rather than an
# expression, so an attribute named like an operator ("max") is not
# mistaken for one.
_MAPPING_FIELDS = {
    "self_delta", "target_delta", "new_state", "offspring_state", "state",
    "give", "get", "details",
}

# action type -> allowed fields (besides "type")
_ACTION_FIELDS = {
    "move": {"direction"},
    "move_to": {"target"},
    "move_random": set(),
    "move_toward": {"target"},
    "move_away": {"from"},
    "interact": {"target", "target_id", "interaction_type", "params"},
    "reproduce": {"cost", "energy_cost", "offspring_count", "offspring_state"},
    "spawn": {"agent_type", "count", "state"},
    "die": {"cause"},
    "modify_state": {"attribute", "value", "delta"},
    "modify_global": {"name", "value", "delta"},
    "transform": {"new_type", "new_state"},
    "custom": {"details"},
}
_ACTION_TYPES = set(_ACTION_FIELDS)
_WORLD_ACTIONS = {"modify_global", "spawn"}

_INTERACT_PARAMS = {
    "success_rate", "kill_target", "self_delta", "target_delta",
    "transfer", "exchange",
    "energy_gain", "amount",  # legacy aliases (predation / transfer_energy)
}

_RULE_KEYS = {"when", "prob", "do", "choose", "target"}


class RuleError(ValueError):
    """A rules spec is malformed, or failed while running. The message
    says which rule and why."""


# --- Validation ------------------------------------------------------------

class _Scope:
    """What an expression may reference where it appears."""

    def __init__(self, globals_=None, types=None, has_self=True,
                 has_other=False, can_sense=True):
        self.globals = globals_
        self.types = types
        self.has_self = has_self
        self.has_other = has_other
        self.can_sense = can_sense

    def with_other(self):
        return _Scope(self.globals, self.types, self.has_self, True,
                      self.can_sense)


def validate_rules(rules: Any, agent_type: str = "?", *,
                   path: Optional[str] = None,
                   globals_: Optional[set] = None,
                   agent_types: Optional[set] = None,
                   world: bool = False) -> None:
    """Validate a rules list, raising RuleError with a precise message.

    ``globals_`` / ``agent_types``, when given, are the declared names that
    ``@name`` and type references are checked against. ``world=True``
    validates world_rules: no agent, so no ``$``, neighbors or agent actions.
    """
    path = path or f"agent_types['{agent_type}'].behavior.rules"
    if not isinstance(rules, list):
        raise RuleError(f"{path} must be a list, got {type(rules).__name__}")
    for i, rule in enumerate(rules):
        where = f"{path}[{i}]"
        if not isinstance(rule, dict):
            raise RuleError(f"{where} must be a dict, got {type(rule).__name__}")
        unknown = set(rule) - _RULE_KEYS
        if unknown:
            raise RuleError(
                f"{where} has unknown keys {sorted(unknown)} — "
                f"allowed: {sorted(_RULE_KEYS)}"
            )
        if ("do" in rule) == ("choose" in rule):
            raise RuleError(
                f"{where} needs exactly one of 'do' (list of actions) or "
                f"'choose' (weighted branches)"
            )
        if "do" in rule and not isinstance(rule["do"], list):
            raise RuleError(f"{where}.do must be a list of action dicts")
        if "choose" in rule:
            branches = rule["choose"]
            if not isinstance(branches, list) or not branches:
                raise RuleError(
                    f"{where}.choose must be a non-empty list of "
                    f"{{'weight': w, 'do': [...]}} branches"
                )
            for k, branch in enumerate(branches):
                if not isinstance(branch, dict) or set(branch) != {"weight", "do"} \
                        or not isinstance(branch["do"], list):
                    raise RuleError(
                        f"{where}.choose[{k}] must be {{'weight': w, 'do': [...]}}"
                    )

        scope = _Scope(globals_, agent_types, has_self=not world,
                       can_sense=not world)
        if "target" in rule:
            if world:
                raise RuleError(f"{where}: world rules have no agent, so no 'target'")
            _validate_selector(rule["target"], f"{where}.target", scope)
            scope = scope.with_other()
        if "when" in rule:
            _validate_expr(rule["when"], f"{where}.when", scope)
        if "prob" in rule:
            _validate_expr(rule["prob"], f"{where}.prob", scope)
        for label, actions in _action_lists(rule):
            for j, action in enumerate(actions):
                _validate_action(action, f"{where}.{label}[{j}]", scope,
                                 has_rule_target="target" in rule, world=world)
        for k, branch in enumerate(rule.get("choose") or []):
            _validate_expr(branch["weight"], f"{where}.choose[{k}].weight", scope)


def _action_lists(rule: Dict):
    """(label, actions) for a rule's 'do' list or each 'choose' branch."""
    if "do" in rule:
        return [("do", rule["do"])]
    return [(f"choose[{k}].do", branch["do"])
            for k, branch in enumerate(rule.get("choose") or [])]


def is_expression(value: Any) -> bool:
    """True for values the engine evaluates: operator dicts and @globals."""
    if isinstance(value, dict):
        return len(value) == 1 and next(iter(value)) in _KNOWN_OPS
    return isinstance(value, str) and value.startswith("@")


def validate_initial_state(state: Any, where: str, *,
                           globals_: Optional[set] = None) -> None:
    """initial_state values may be expressions evaluated per agent at
    creation (randomness, globals) — no agent or neighbors exist yet."""
    if not isinstance(state, dict):
        raise RuleError(f"{where} must be a dict of attribute -> value")
    for attr, value in state.items():
        if is_expression(value):
            _validate_expr(value, f"{where}.{attr}",
                           _Scope(globals_, None, has_self=False, can_sense=False))


def compile_initial_state(state: Dict[str, Any]):
    """fn(ctx) -> a fresh state dict with expressions evaluated. Literal
    values are deep-copied per call so agents never share them."""
    import copy
    exprs = {attr: compile_expr(value) for attr, value in state.items()
             if is_expression(value)}

    def build(ctx):
        out = {}
        for attr, value in state.items():
            out[attr] = exprs[attr](ctx) if attr in exprs else copy.deepcopy(value)
        return out
    return build


def validate_expression(expr: Any, where: str, *,
                        globals_: Optional[set] = None,
                        agent_types: Optional[set] = None) -> None:
    """Validate a world-level expression (e.g. an observable)."""
    _validate_expr(expr, where, _Scope(globals_, agent_types, has_self=False,
                                       can_sense=False))


def _check_type(name: Any, where: str, scope: _Scope) -> None:
    if not isinstance(name, str):
        raise RuleError(f"{where}: agent type must be a string, got {name!r}")
    if scope.types is not None and name not in scope.types:
        raise RuleError(
            f"{where}: unknown agent type '{name}' — defined: {sorted(scope.types)}"
        )


def _validate_expr(expr: Any, where: str, scope: _Scope) -> None:
    """Structural validation of an expression tree."""
    if isinstance(expr, str):
        if expr.startswith("$") and not scope.has_self:
            raise RuleError(
                f"{where}: '{expr}' reads an agent attribute, but there is no "
                f"agent here (world rules and observables see only globals "
                f"and aggregates)"
            )
        if expr.startswith("&") and not scope.has_other:
            raise RuleError(
                f"{where}: '{expr}' reads the other agent, but nothing is "
                f"selected here — give the rule a 'target', or use it inside "
                f"a query's 'where' / 'by'"
            )
        if expr.startswith("@") and expr != "@step" and scope.globals is not None:
            if expr[1:] not in scope.globals:
                raise RuleError(
                    f"{where}: unknown global '{expr}' — declared globals: "
                    f"{sorted(scope.globals)} (declare it in the spec's 'globals')"
                )
        return
    if isinstance(expr, list):
        for k, sub in enumerate(expr):
            _validate_expr(sub, f"{where}[{k}]", scope)
        return
    if not isinstance(expr, dict):
        return

    if len(expr) != 1:
        raise RuleError(
            f"{where}: operator dicts must have exactly one key, got {sorted(expr)}"
        )
    op = next(iter(expr))
    if op not in _KNOWN_OPS:
        raise RuleError(
            f"{where}: unknown operator '{op}' — options: {sorted(_KNOWN_OPS)}"
        )
    args = expr[op]
    shape = _OP_SHAPES[op]
    here = f"{where}.{op}"

    if isinstance(shape, int) or shape == "n":
        if not isinstance(args, list):
            raise RuleError(f"{here} takes a list of operands, got {args!r}")
        if shape == "n" and not args:
            raise RuleError(f"{here} needs at least one operand")
        if isinstance(shape, int) and len(args) != shape:
            raise RuleError(f"{here} takes exactly {shape} operand(s), got {len(args)}")
        for k, sub in enumerate(args):
            _validate_expr(sub, f"{here}[{k}]", scope)
    elif shape == "1":
        _validate_expr(args, here, scope)
    elif shape == "literal":
        if not isinstance(args, list) or not args:
            raise RuleError(f"{here} takes a non-empty list of values")
    elif shape == "query":
        if not scope.can_sense:
            raise RuleError(
                f"{here}: neighbor queries need an agent — use 'total', 'sum' "
                f"or 'mean' for world-level counts"
            )
        _validate_query(args, here, scope)
    elif shape == "total":
        if isinstance(args, str):
            _check_type(args, here, scope)
        else:
            _validate_query(args, here, scope)
    elif shape == "agg":
        if not isinstance(args, dict) or "attr" not in args:
            raise RuleError(
                f"{here} takes {{'attr': name, 'type'?: t, 'where'?: expr}}, got {args!r}"
            )
        if not isinstance(args["attr"], str):
            raise RuleError(f"{here}.attr must be an attribute name")
        _validate_query({k: v for k, v in args.items() if k != "attr"},
                        here, scope)


def _validate_query(query: Any, where: str, scope: _Scope,
                    extra_keys=()) -> None:
    if not isinstance(query, dict):
        raise RuleError(f"{where}: query must be a dict like {{'type': 'wolf'}}, got {query!r}")
    unknown = set(query) - {"type", "where"} - set(extra_keys)
    if unknown:
        raise RuleError(
            f"{where}: unknown query keys {sorted(unknown)} — allowed: "
            f"{sorted({'type', 'where', *extra_keys})}"
        )
    if "type" in query:
        _check_type(query["type"], f"{where}.type", scope)
    if "where" in query:
        _validate_expr(query["where"], f"{where}.where", scope.with_other())


def _validate_selector(selector: Any, where: str, scope: _Scope) -> None:
    if not scope.can_sense:
        raise RuleError(f"{where}: selectors need an agent")
    if not isinstance(selector, dict) or len(selector) != 1 \
            or next(iter(selector)) not in _SELECTORS:
        raise RuleError(
            f"{where}: selector must be one of "
            f"{{'nearest'|'random'|'lowest'|'highest': query}}, got {selector!r}"
        )
    kind = next(iter(selector))
    query = selector[kind]
    ranked = kind in ("lowest", "highest")
    _validate_query(query, f"{where}.{kind}", scope,
                    extra_keys=("by",) if ranked else ())
    if ranked:
        if "by" not in query:
            raise RuleError(f"{where}.{kind} needs 'by' — e.g. \"by\": \"&price\"")
        _validate_expr(query["by"], f"{where}.{kind}.by", scope.with_other())


def _validate_value(key: str, value: Any, where: str, scope: _Scope) -> None:
    """Validate an action field value, which may nest expressions."""
    if key in _MAPPING_FIELDS:
        if not isinstance(value, dict):
            raise RuleError(f"{where} must be a {{attribute: value}} map")
        for attr, sub in value.items():
            _validate_value("", sub, f"{where}.{attr}", scope)
    elif isinstance(value, dict) and len(value) == 1 and next(iter(value)) in _KNOWN_OPS:
        _validate_expr(value, where, scope)
    elif isinstance(value, dict):
        for k, sub in value.items():
            _validate_value(k, sub, f"{where}.{k}", scope)
    else:
        _validate_expr(value, where, scope)


def _validate_action(action: Any, where: str, scope: _Scope,
                     has_rule_target: bool, world: bool) -> None:
    if not isinstance(action, dict) or "type" not in action:
        raise RuleError(f"{where} must be a dict with a 'type' key")
    a_type = action["type"]
    if a_type not in _ACTION_TYPES:
        raise RuleError(
            f"{where} unknown action type '{a_type}' — options: {sorted(_ACTION_TYPES)}"
        )
    if world and a_type not in _WORLD_ACTIONS:
        raise RuleError(
            f"{where}: world rules have no agent, so '{a_type}' is not allowed "
            f"— options: {sorted(_WORLD_ACTIONS)}"
        )
    unknown = set(action) - _ACTION_FIELDS[a_type] - {"type"}
    if unknown:
        raise RuleError(
            f"{where} ({a_type}) has unknown fields {sorted(unknown)} — "
            f"allowed: {sorted(_ACTION_FIELDS[a_type])}"
        )

    # selector-valued fields
    for field in ("target", "from"):
        if field in action and isinstance(action[field], dict) and \
                a_type in ("interact", "move_toward", "move_away"):
            _validate_selector(action[field], f"{where}.{field}", scope)
    if a_type in ("interact", "move_away"):
        field = "from" if a_type == "move_away" else "target"
        if field not in action and "target_id" not in action and not has_rule_target:
            raise RuleError(
                f"{where} ({a_type}) needs '{field}': a selector like "
                f"{{'random': {{'type': ...}}}}, or a rule-level 'target'"
            )
    if a_type == "move_toward" and "target" not in action and not has_rule_target:
        raise RuleError(f"{where} (move_toward) needs 'target' or a rule-level 'target'")

    if a_type == "modify_global":
        name = action.get("name")
        if not isinstance(name, str):
            raise RuleError(f"{where} (modify_global) needs 'name': a declared global")
        if scope.globals is not None and name not in scope.globals:
            raise RuleError(
                f"{where}: unknown global '{name}' — declared globals: "
                f"{sorted(scope.globals)}"
            )
    if a_type in ("modify_state", "modify_global") and \
            "value" not in action and "delta" not in action:
        raise RuleError(f"{where} ({a_type}) needs 'value' or 'delta'")
    if a_type == "modify_state" and "attribute" not in action:
        raise RuleError(f"{where} (modify_state) needs 'attribute'")
    if a_type == "spawn":
        if "agent_type" not in action:
            raise RuleError(f"{where} (spawn) needs 'agent_type'")
        _check_type(action["agent_type"], f"{where}.agent_type", scope)
    if a_type == "transform":
        if "new_type" not in action:
            raise RuleError(f"{where} (transform) needs 'new_type'")
        _check_type(action["new_type"], f"{where}.new_type", scope)

    if a_type == "interact":
        params = action.get("params", {})
        if not isinstance(params, dict):
            raise RuleError(f"{where}.params must be a dict")
        bad = set(params) - _INTERACT_PARAMS
        if bad:
            raise RuleError(
                f"{where}.params has unknown keys {sorted(bad)} — allowed: "
                f"{sorted(_INTERACT_PARAMS - {'energy_gain', 'amount'})}. "
                f"To require something of the target, filter the selector "
                f"with 'where'; for a two-sided trade use 'exchange'"
            )
        if "exchange" in params:
            ex = params["exchange"]
            if not isinstance(ex, dict) or not set(ex) <= {"give", "get"} or not ex:
                raise RuleError(
                    f"{where}.params.exchange must be "
                    f"{{'give': {{attr: amount}}, 'get': {{attr: amount}}}}"
                )

    for key, value in action.items():
        if key in ("type", "target", "from") and not (
                key == "target" and a_type == "move_to"):
            continue
        _validate_value(key, value, f"{where}.{key}", scope)


# --- Evaluation ------------------------------------------------------------

class _Context:
    """Everything an expression can see during one decision.

    ``nearby`` may be a list or a zero-argument callable that computes it
    on first use (most rules never look at neighbors). ``manager`` enables
    global queries (total with a filter, sum, mean) and non-spatial
    neighbor lookups by type. ``other`` is the agent ``&attr`` reads.
    """

    __slots__ = ("agent", "_nearby", "rng", "env", "step", "counts",
                 "globals", "manager", "other", "can_sense")

    def __init__(self, agent, nearby, rng, env, step, counts=None,
                 globals=None, manager=None, other=None, can_sense=True):
        self.agent = agent
        self._nearby = nearby
        self.rng = rng
        self.env = env
        self.step = step
        self.counts = counts
        self.globals = globals if globals is not None else {}
        self.manager = manager
        self.other = other
        self.can_sense = can_sense

    @property
    def nearby(self) -> List:
        if callable(self._nearby):
            self._nearby = self._nearby()
        return self._nearby or []


def _candidates(wanted: Optional[str], ctx: _Context) -> List:
    """Nearby living agents, optionally of one type."""
    if not ctx.can_sense:
        return []
    if ctx.manager is not None and ctx.env.env_type == "none":
        # Non-spatial: everyone is nearby, so read the type index directly
        # instead of materializing the whole population per decision.
        pool = (ctx.manager.agents_by_type.get(wanted, ()) if wanted is not None
                else ctx.manager.agents)
        me = ctx.agent
        return [a for a in pool if a.alive and a is not me]
    return [a for a in ctx.nearby
            if a.alive and (wanted is None or a.type == wanted)]


def _global_pool(wanted: Optional[str], ctx: _Context) -> List:
    if ctx.manager is None:
        raise RuleError("global queries need a running simulation")
    pool = (ctx.manager.agents_by_type.get(wanted, ()) if wanted is not None
            else ctx.manager.agents)
    return [a for a in pool if a.alive]


# Expressions are compiled once into closures ``fn(ctx) -> value``; the
# JSON tree is walked at load time, not on every evaluation. Evaluation
# order (and so RNG call order) matches the tree left to right.

def _const(value):
    return lambda ctx: value


def _compile_filter(where):
    """fn(agents, ctx) -> agents for which ``where`` holds ("&" = candidate)."""
    if where is None:
        return lambda agents, ctx: agents
    test = compile_expr(where)

    def apply(agents, ctx):
        saved = ctx.other
        out = []
        try:
            for a in agents:
                ctx.other = a
                if test(ctx):
                    out.append(a)
        finally:
            ctx.other = saved
        return out
    return apply


def _compile_query(query, global_=False):
    """fn(ctx) -> matching agents (nearby, or world-wide if global_)."""
    if not isinstance(query, dict):
        raise RuleError(f"query must be a dict, got {query!r}")
    wanted = query.get("type")
    keep = _compile_filter(query.get("where"))
    pool = _global_pool if global_ else _candidates
    return lambda ctx: keep(pool(wanted, ctx), ctx)


def _compile_selector(selector):
    """fn(ctx) -> one agent or None."""
    if not isinstance(selector, dict) or len(selector) != 1:
        raise RuleError(
            f"selector must be {{'nearest'|'random'|'lowest'|'highest': query}}, "
            f"got {selector!r}"
        )
    kind = next(iter(selector))
    if kind not in _SELECTORS:
        raise RuleError(f"unknown selector '{kind}' — options: {list(_SELECTORS)}")
    query = selector[kind]
    matches_of = _compile_query(query)

    if kind == "random":
        def select(ctx):
            matches = matches_of(ctx)
            return ctx.rng.choice(matches) if matches else None
        return select

    if kind == "nearest":
        def select(ctx):
            matches = matches_of(ctx)
            if not matches:
                return None
            my_pos = ctx.agent.state.get("position")
            if my_pos is None:
                # No geometry: every match is equally near.
                return ctx.rng.choice(matches)
            dist = ctx.env.distance
            return min(matches, key=lambda a: dist(my_pos, a.state["position"])
                       if a.state.get("position") is not None else _INF)
        return select

    by = compile_expr(query["by"])
    pick = min if kind == "lowest" else max

    def select(ctx):
        matches = matches_of(ctx)
        if not matches:
            return None
        saved = ctx.other
        ranked = []
        try:
            for a in matches:
                ctx.other = a
                value = by(ctx)
                if value is not None:
                    ranked.append((value, a))
        finally:
            ctx.other = saved
        return pick(ranked, key=lambda pair: pair[0])[1] if ranked else None
    return select


def _compile_all(args):
    return [compile_expr(a) for a in args]


def _pair(args):
    a, b = _compile_all(args)
    return a, b


def compile_expr(expr: Any):
    """Compile an expression tree into ``fn(ctx) -> value``."""
    if isinstance(expr, (int, float, bool)) or expr is None:
        return _const(expr)

    if isinstance(expr, str):
        if expr.startswith("$"):
            name = expr[1:]

            def read_self(ctx):
                if ctx.agent is None:
                    raise RuleError(f"'{expr}' used where there is no agent")
                return ctx.agent.state.get(name)
            return read_self
        if expr.startswith("&"):
            name = expr[1:]

            def read_other(ctx):
                if ctx.other is None:
                    raise RuleError(f"'&{name}' used with no selected agent")
                return ctx.other.state.get(name)
            return read_other
        if expr == "@step":
            return lambda ctx: ctx.step
        if expr.startswith("@"):
            name = expr[1:]

            def read_global(ctx):
                try:
                    return ctx.globals[name]
                except KeyError:
                    raise RuleError(f"unknown global '{expr}'") from None
            return read_global
        return _const(expr)

    if isinstance(expr, list):
        parts = _compile_all(expr)
        return lambda ctx: [p(ctx) for p in parts]

    if not isinstance(expr, dict):
        raise RuleError(f"cannot evaluate expression: {expr!r}")
    if len(expr) != 1:
        raise RuleError(f"operator dicts must have exactly one key, got {sorted(expr)}")
    op = next(iter(expr))
    args = expr[op]

    if op in _BINARY_OPS:
        a, b = _pair(args)
        cmp = _BINARY_OPS[op]
        return lambda ctx: cmp(a(ctx), b(ctx))
    if op == "+":
        parts = _compile_all(args)
        return lambda ctx: sum(p(ctx) for p in parts)
    if op == "-":
        a, b = _pair(args)
        return lambda ctx: a(ctx) - b(ctx)
    if op == "*":
        parts = _compile_all(args)

        def mul(ctx):
            out = 1
            for p in parts:
                out *= p(ctx)
            return out
        return mul
    if op == "/":
        a, b = _pair(args)
        return lambda ctx: a(ctx) / b(ctx)
    if op == "%":
        a, b = _pair(args)
        return lambda ctx: a(ctx) % b(ctx)
    if op == "and":
        parts = _compile_all(args)
        return lambda ctx: all(p(ctx) for p in parts)
    if op == "or":
        parts = _compile_all(args)
        return lambda ctx: any(p(ctx) for p in parts)
    if op == "not":
        x = compile_expr(args)
        return lambda ctx: not x(ctx)
    if op == "min":
        parts = _compile_all(args)
        return lambda ctx: min(p(ctx) for p in parts)
    if op == "max":
        parts = _compile_all(args)
        return lambda ctx: max(p(ctx) for p in parts)
    if op == "abs":
        x = compile_expr(args)
        return lambda ctx: abs(x(ctx))
    if op == "random":
        return lambda ctx: ctx.rng.random()
    if op == "uniform":
        a, b = _pair(args)
        return lambda ctx: ctx.rng.uniform(a(ctx), b(ctx))
    if op == "randint":
        a, b = _pair(args)
        return lambda ctx: ctx.rng.randint(a(ctx), b(ctx))
    if op == "choice":
        options = list(args)
        return lambda ctx: ctx.rng.choice(options)
    if op == "count":
        matches = _compile_query(args)
        return lambda ctx: len(matches(ctx))
    if op == "nearest_distance":
        nearest = _compile_selector({"nearest": args})

        def distance(ctx):
            target = nearest(ctx)
            if target is None:
                return _INF
            my_pos = ctx.agent.state.get("position")
            other_pos = target.state.get("position")
            if my_pos is None or other_pos is None:
                return _INF
            return ctx.env.distance(my_pos, other_pos)
        return distance
    if op == "total":
        if isinstance(args, str):
            def total(ctx):
                if ctx.manager is not None:
                    return len(ctx.manager.get_agents_by_type(args))
                return (ctx.counts or {}).get(args, 0)
            return total
        matches = _compile_query(args, global_=True)
        return lambda ctx: len(matches(ctx))
    if op in ("sum", "mean"):
        attr = args["attr"]
        matches = _compile_query({k: v for k, v in args.items() if k != "attr"},
                                 global_=True)

        def aggregate(ctx):
            values = [v for v in (a.state.get(attr) for a in matches(ctx))
                      if v is not None]
            if op == "sum":
                return sum(values)
            return sum(values) / len(values) if values else 0
        return aggregate
    raise RuleError(f"unknown operator '{op}'")


def evaluate(expr: Any, ctx: _Context) -> Any:
    """Evaluate an expression tree against a context (compiles it first;
    hot paths compile once and reuse the closure)."""
    return compile_expr(expr)(ctx)


def _compile_value(key: str, value: Any):
    """Compile an action field; nested maps and lists rebuild on every
    call, so no two agents ever share a mutable value."""
    if key in _MAPPING_FIELDS and isinstance(value, dict):
        items = [(k, _compile_value("", v)) for k, v in value.items()]
        return lambda ctx: {k: f(ctx) for k, f in items}
    if isinstance(value, dict):
        if len(value) == 1 and next(iter(value)) in _KNOWN_OPS:
            return compile_expr(value)
        items = [(k, _compile_value(k, v)) for k, v in value.items()]
        return lambda ctx: {k: f(ctx) for k, f in items}
    if isinstance(value, list):
        parts = [_compile_value("", v) for v in value]
        return lambda ctx: [p(ctx) for p in parts]
    if isinstance(value, str) and value[:1] in ("$", "&", "@"):
        return compile_expr(value)
    return _const(value)


def _compile_fields(action: Dict, skip=()):
    """fn(ctx) -> dict with every field evaluated ("type" kept verbatim)."""
    fields = [(k, _const(v) if k == "type" else _compile_value(k, v))
              for k, v in action.items() if k not in skip]
    return lambda ctx: {k: f(ctx) for k, f in fields}


def _sign_away(mine: float, other: float, rng) -> int:
    """Direction component pointing away from `other`."""
    if mine > other:
        return 1
    if mine < other:
        return -1
    return rng.choice([-1, 1])


def _compile_target(action: Dict, field: str):
    """fn(ctx) -> the agent an action is aimed at: its own selector, or
    the rule's bound target."""
    spec = action.get(field)
    if isinstance(spec, dict):
        return _compile_selector(spec)
    return lambda ctx: ctx.other


def _compile_action(action: Dict):
    """Compile an action template into ``fn(ctx) -> engine action or None``."""
    a_type = action["type"]

    if a_type == "move_toward":
        target = action.get("target")
        if isinstance(target, dict) or target is None:
            other_of = _compile_target(action, "target")

            def move_toward(ctx):
                other = other_of(ctx)
                pos = other.state.get("position") if other is not None else None
                return {"type": "move_to", "target": tuple(pos)} if pos is not None else None
            return move_toward
        position = _compile_value("target", target)

        def move_to_position(ctx):
            pos = position(ctx)
            return {"type": "move_to", "target": tuple(pos)} if pos is not None else None
        return move_to_position

    if a_type == "move_away":
        other_of = _compile_target(action, "from")

        def move_away(ctx):
            other = other_of(ctx)
            if other is None:
                return None
            my_pos = ctx.agent.state.get("position")
            other_pos = other.state.get("position")
            if my_pos is None or other_pos is None:
                return None
            dx = _sign_away(my_pos[0], other_pos[0], ctx.rng)
            dy = _sign_away(my_pos[1], other_pos[1], ctx.rng)
            return {"type": "move", "direction": [dx, dy]}
        return move_away

    if a_type == "interact" and "target_id" not in action:
        other_of = _compile_target(action, "target")
        fields = _compile_fields(action, skip=("target",))

        def interact(ctx):
            other = other_of(ctx)
            if other is None:
                return None
            saved = ctx.other
            ctx.other = other  # &attr in params reads the interaction partner
            try:
                compiled = fields(ctx)
            finally:
                ctx.other = saved
            compiled["target_id"] = other.id
            return compiled
        return interact

    return _compile_fields(action)


def _located(err: Exception, where: str, ctx: _Context) -> RuleError:
    """Wrap a runtime failure with the rule it came from."""
    who = f"agent {ctx.agent.id} ({ctx.agent.type})" if ctx.agent is not None else "world"
    msg = str(err)
    hint = ""
    if "NoneType" in msg:
        hint = (" — an '$attr' or '&attr' read an attribute that agent doesn't "
                "have; give it a value in initial_state")
    return RuleError(f"{where}: {msg}{hint} [{who}, step {ctx.step}]")


def _choose(branches: List, ctx: _Context) -> List:
    """Pick one compiled branch's actions with probability proportional to
    weight. branches: [(weight_fn, action_fns), ...]."""
    weights = [max(0.0, float(weight(ctx) or 0)) for weight, _ in branches]
    total = sum(weights)
    if total <= 0:
        return []
    roll = ctx.rng.random() * total
    for (_, actions), weight in zip(branches, weights):
        roll -= weight
        if roll < 0:
            return actions
    return branches[-1][1]


def _collapse_repeated_target(rule: Dict) -> Dict:
    """An action that repeats its rule's target selector means the rule's
    target. Re-selecting would pick a different agent for a 'random'
    selector than the one the condition checked."""
    if "target" not in rule:
        return rule
    field_of = {"interact": "target", "move_toward": "target", "move_away": "from"}

    def collapse(actions):
        out = []
        for action in actions:
            field = field_of.get(action.get("type")) if isinstance(action, dict) else None
            if field and action.get(field) == rule["target"]:
                action = {k: v for k, v in action.items() if k != field}
            out.append(action)
        return out

    rule = dict(rule)
    if "do" in rule:
        rule["do"] = collapse(rule["do"])
    else:
        rule["choose"] = [dict(b, do=collapse(b["do"])) for b in rule["choose"]]
    return rule


class RuleBehavior:
    """
    A behavior function compiled from declarative rules.

    Callable with the standard behavior signature
    ``(agent, sim_state, agents_nearby) -> actions`` so the engine treats it
    exactly like a Python behavior function; the simulation calls
    ``decide`` directly, which computes neighbors only if a rule needs them.
    """

    def __init__(self, rules: List[Dict], simulation, agent_type: str = "?",
                 path: Optional[str] = None, world: bool = False,
                 can_sense: bool = True):
        spec = getattr(simulation, "spec", None) or {}
        self.path = path or f"agent_types['{agent_type}'].behavior.rules"
        validate_rules(
            rules, agent_type, path=self.path,
            globals_=set(spec.get("globals") or {}),
            agent_types=set(spec.get("agent_types") or {}) or None,
            world=world,
        )
        self.rules = [_collapse_repeated_target(rule) for rule in rules]
        self._compiled = [self._compile_rule(rule) for rule in self.rules]
        self.simulation = simulation
        self.agent_type = agent_type
        self.world = world
        self.can_sense = can_sense and not world

    @staticmethod
    def _compile_rule(rule: Dict):
        """(target_fn, when_fn, prob_fn, actions) with actions either a list
        of action fns or ("choose", [(weight_fn, action fns), ...])."""
        target = _compile_selector(rule["target"]) if "target" in rule else None
        when = compile_expr(rule["when"]) if "when" in rule else None
        prob = compile_expr(rule["prob"]) if "prob" in rule else None
        if "do" in rule:
            actions = [_compile_action(a) for a in rule["do"]]
        else:
            actions = ("choose", [(compile_expr(b["weight"]),
                                   [_compile_action(a) for a in b["do"]])
                                  for b in rule["choose"]])
        return target, when, prob, actions

    def _context(self, agent, nearby, step, counts=None) -> _Context:
        sim = self.simulation
        return _Context(
            agent=agent, nearby=nearby, rng=sim.rng, env=sim.environment,
            step=step, counts=counts,
            globals=getattr(sim, "globals", None),
            manager=getattr(sim, "agent_manager", None),
            can_sense=self.can_sense,
        )

    def __call__(self, agent, sim_state, agents_nearby) -> List[Dict]:
        ctx = self._context(agent, agents_nearby, sim_state.get("step", 0),
                            sim_state.get("agent_counts", {}))
        return self._run(ctx)

    def decide(self, agent=None, nearby=None) -> List[Dict]:
        """Evaluate the rules for one agent (or for the world, if agent is
        None). ``nearby`` may be a list or a lazy zero-argument callable."""
        return self._run(self._context(agent, nearby, self.simulation.step))

    def _run(self, ctx: _Context) -> List[Dict]:
        actions = []
        for i, (target, when, prob, rule_actions) in enumerate(self._compiled):
            try:
                ctx.other = None
                if target is not None:
                    ctx.other = target(ctx)
                    if ctx.other is None:
                        continue
                if when is not None and not when(ctx):
                    continue
                # draw first, then evaluate prob (keeps RNG order stable)
                if prob is not None and ctx.rng.random() >= prob(ctx):
                    continue
                if isinstance(rule_actions, tuple):
                    rule_actions = _choose(rule_actions[1], ctx)
                for make in rule_actions:
                    compiled = make(ctx)
                    if compiled is not None:
                        actions.append(compiled)
            except RuleError as e:
                if str(e).startswith(self.path):
                    raise
                raise _located(e, f"{self.path}[{i}]", ctx) from e
            except (TypeError, ValueError, ArithmeticError, KeyError,
                    IndexError, AttributeError) as e:
                raise _located(e, f"{self.path}[{i}]", ctx) from e
        ctx.other = None
        return actions


# --- Whole-spec checks -----------------------------------------------------

def _dollar_reads(node: Any, out: set) -> None:
    if isinstance(node, str):
        if node.startswith("$"):
            out.add(node[1:])
    elif isinstance(node, dict):
        for value in node.values():
            _dollar_reads(value, out)
    elif isinstance(node, list):
        for value in node:
            _dollar_reads(value, out)


def _literal_keys(mapping: Any) -> set:
    return set(mapping) if isinstance(mapping, dict) else set()


def check_attribute_reads(spec: Dict[str, Any]) -> None:
    """Reject rules that read an attribute their agents can never have.

    A comparison on a missing attribute is simply false, so a typo like
    "$enrgy", or an attribute only another type defines, would otherwise
    make a rule silently never fire. An attribute counts as available to a
    type if it is in the type's initial_state, or any rule could write it
    onto agents of that type (its own modify_state/self_delta/exchange,
    another agent's target_delta/exchange/transfer, global_rules, spawn
    state, or a transform that carries the old agent's state over).

    Specs with Python behaviors are skipped: code can write anything.
    """
    types = spec.get("agent_types") or {}
    rules_of = {}
    for name, type_spec in types.items():
        if type_spec.get("behavior_code"):
            return
        behavior = type_spec.get("behavior")
        if isinstance(behavior, dict) and isinstance(behavior.get("rules"), list):
            rules_of[name] = behavior["rules"]

    provided = {name: set((t.get("initial_state") or {})) | {"position"}
                for name, t in types.items()}
    for_everyone = set()
    carries = []  # (from_type, to_type) via transform

    def scan(actions, owner):
        """Record what these actions write. owner=None: applies to anyone."""
        mine = provided[owner] if owner is not None else for_everyone
        for action in actions:
            if not isinstance(action, dict):
                continue
            a_type = action.get("type")
            if a_type == "modify_state":
                attr = action.get("attribute")
                if isinstance(attr, str):
                    mine.add(attr)
            elif a_type == "interact":
                params = action.get("params") or {}
                mine.update(_literal_keys(params.get("self_delta")))
                for_everyone.update(_literal_keys(params.get("target_delta")))
                exchange = params.get("exchange") or {}
                for side in ("give", "get"):
                    keys = _literal_keys(exchange.get(side))
                    mine.update(keys)
                    for_everyone.update(keys)
                transfer = params.get("transfer") or {}
                if isinstance(transfer.get("attribute"), str):
                    mine.add(transfer["attribute"])
                    for_everyone.add(transfer["attribute"])
                if params.get("energy_gain") is not None:
                    mine.add("energy")
            elif a_type == "reproduce":
                cost = action.get("cost") or {"attribute": "energy"}
                if isinstance(cost.get("attribute"), str):
                    mine.add(cost["attribute"])
                mine.update(_literal_keys(action.get("offspring_state")))
            elif a_type == "spawn" and action.get("agent_type") in provided:
                provided[action["agent_type"]].update(_literal_keys(action.get("state")))
            elif a_type == "transform" and action.get("new_type") in provided:
                target = action["new_type"]
                provided[target].update(_literal_keys(action.get("new_state")))
                if owner is not None:
                    carries.append((owner, target))

    def actions_of(rule):
        if not isinstance(rule, dict):
            return []
        out = list(rule.get("do") or [])
        for branch in rule.get("choose") or []:
            if isinstance(branch, dict):
                out += branch.get("do") or []
        return out

    for name, rules in rules_of.items():
        for rule in rules:
            scan(actions_of(rule), name)
    for key in ("global_rules", "world_rules"):
        for rule in spec.get(key) or []:
            scan(actions_of(rule), None)

    # Transforms carry the old agent's attributes to the new type
    changed = True
    while changed:
        changed = False
        for source, target in carries:
            before = len(provided[target])
            provided[target] |= provided[source]
            changed |= len(provided[target]) != before

    for name, rules in rules_of.items():
        reads = set()
        _dollar_reads(rules, reads)
        missing = sorted(reads - provided[name] - for_everyone)
        if missing:
            listed = ", ".join(f"'${m}'" for m in missing)
            raise RuleError(
                f"agent_types['{name}'].behavior.rules read {listed}, but no "
                f"{name} agent ever has {'that attribute' if len(missing) == 1 else 'those attributes'} "
                f"— give {'it' if len(missing) == 1 else 'them'} a value in "
                f"agent_types['{name}'].initial_state"
            )
