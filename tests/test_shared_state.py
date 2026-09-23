"""Tests for shared state in the rule language: globals, world rules,
observables, targeted rules (&attr), filtered queries, aggregates,
exchange, spawn — and loud failures for broken specs."""
import json

import pytest

from agentstan import Simulation, RuleError, Pack

NONE_ENV = {"type": "none"}


def _sim(spec, seed=0):
    return Simulation(json.loads(json.dumps(spec)), seed=seed)


# --- Globals and world rules ---

def test_globals_readable_and_writable():
    spec = {
        "environment": NONE_ENV,
        "globals": {"treasury": 0},
        "agent_types": {
            "citizen": {
                "initial_count": 4,
                "initial_state": {"gold": 10},
                "behavior": {"rules": [
                    {"when": {"<": ["@treasury", 6]},
                     "do": [{"type": "modify_state", "attribute": "gold", "delta": -1},
                            {"type": "modify_global", "name": "treasury", "delta": 1}]},
                ]},
            },
        },
    }
    sim = _sim(spec)
    results = sim.run(3)
    # Globals are live: once the treasury reaches 6, later agents stop paying
    assert results["globals"]["treasury"] == 6
    gold = sum(a["gold"] for a in sim.agent_manager.get_living_agents())
    assert gold + results["globals"]["treasury"] == 40  # conserved
    assert results["metrics"]["history"][0]["globals"] == {"treasury": 4}


def test_world_rules_run_once_per_step_with_aggregates():
    spec = {
        "environment": NONE_ENV,
        "globals": {"price": 10.0},
        "world_rules": [
            # price follows demand: +1 per buyer holding enough gold
            {"do": [{"type": "modify_global", "name": "price",
                     "value": {"+": [10, {"total": {"type": "buyer",
                                                    "where": {">=": ["&gold", 5]}}}]}}]},
        ],
        "agent_types": {
            "buyer": {"initial_count": 3, "initial_state": {"gold": 5}},
            "poor": {"initial_count": 2, "initial_state": {"gold": 1}},
        },
    }
    sim = _sim(spec)
    sim.run(2)
    assert sim.globals["price"] == 13


def test_world_rules_spawn_inflows_and_keep_empty_world_alive():
    spec = {
        "environment": NONE_ENV,
        "world_rules": [
            {"when": {"==": [{"%": ["@step", 2]}, 0]},
             "do": [{"type": "spawn", "agent_type": "player", "count": 3,
                     "state": {"gold": 7}}]},
        ],
        "agent_types": {
            "player": {"initial_count": 0, "initial_state": {"gold": 0, "level": 1}},
        },
    }
    sim = _sim(spec)
    results = sim.run(4)
    assert results["final_step"] == 4  # an empty world with inflows runs on
    players = sim.agent_manager.get_agents_by_type("player")
    assert len(players) == 6
    assert all(p["gold"] == 7 and p["level"] == 1 for p in players)


def test_observables_recorded_each_step():
    spec = {
        "environment": NONE_ENV,
        "observables": {
            "gold_supply": {"sum": {"type": "player", "attr": "gold"}},
            "avg_gold": {"mean": {"attr": "gold"}},
            "rich": {"total": {"type": "player", "where": {">": ["&gold", 10]}}},
        },
        "agent_types": {
            "player": {
                "initial_count": 4,
                "initial_state": {"gold": 10},
                "behavior": {"rules": [
                    {"do": [{"type": "modify_state", "attribute": "gold", "delta": 1}]},
                ]},
            },
        },
    }
    history = _sim(spec).run(2)["metrics"]["history"]
    assert history[0]["observables"] == {"gold_supply": 44, "avg_gold": 11, "rich": 4}
    assert history[1]["observables"]["gold_supply"] == 48


# --- Targets, filters, selectors ---

MARKET = {
    "environment": NONE_ENV,
    "agent_types": {
        "buyer": {
            "initial_count": 1,
            "initial_state": {"gold": 100, "bread": 0},
            "behavior": {"rules": [
                {"target": {"lowest": {"type": "baker", "by": "&price",
                                       "where": {">": ["&stock", 0]}}},
                 "when": {">=": ["$gold", "&price"]},
                 "do": [{"type": "interact", "interaction_type": "buy_bread",
                         "params": {"exchange": {"give": {"gold": "&price"},
                                                 "get": {"stock": 1}}}},
                        {"type": "modify_state", "attribute": "bread", "delta": 1}]},
            ]},
        },
        "baker": {"initial_count": 0, "initial_state": {"gold": 0, "stock": 0, "price": 0}},
    },
}


def _market(bakers):
    sim = _sim(MARKET)
    for state in bakers:
        sim.agent_manager.add_agent(sim._new_agent("baker", state))
    return sim


def test_targeted_rule_buys_from_cheapest_with_stock():
    sim = _market([
        {"stock": 5, "price": 8},
        {"stock": 5, "price": 3},
        {"stock": 0, "price": 1},   # cheapest, but sold out
    ])
    sim.run(1)
    buyer = sim.agent_manager.get_agents_by_type("buyer")[0]
    bakers = sim.agent_manager.get_agents_by_type("baker")
    assert buyer["gold"] == 97 and buyer["stock"] == 1
    assert [b["stock"] for b in bakers] == [5, 4, 0]
    assert bakers[1]["gold"] == 3


def test_targeted_rule_skips_when_nothing_matches():
    sim = _market([{"stock": 0, "price": 1}])
    sim.run(3)
    buyer = sim.agent_manager.get_agents_by_type("buyer")[0]
    assert buyer["gold"] == 100 and buyer["bread"] == 0


def test_exchange_is_atomic():
    """If the target can't deliver, nothing changes hands."""
    spec = {
        "environment": NONE_ENV,
        "agent_types": {
            "buyer": {
                "initial_count": 1,
                "initial_state": {"gold": 10, "wheat": 0},
                "behavior": {"rules": [
                    {"do": [{"type": "interact", "target": {"random": {"type": "farm"}},
                             "interaction_type": "buy",
                             "params": {"exchange": {"give": {"gold": 2},
                                                     "get": {"wheat": 1}}}}]},
                ]},
            },
            "farm": {"initial_count": 1, "initial_state": {"gold": 0, "wheat": 2}},
        },
    }
    sim = _sim(spec)
    results = sim.run(4)
    buyer = sim.agent_manager.get_agents_by_type("buyer")[0]
    farm = sim.agent_manager.get_agents_by_type("farm")[0]
    assert (buyer["gold"], buyer["wheat"]) == (6, 2)
    assert (farm["gold"], farm["wheat"]) == (4, 0)
    outcomes = [e["outcome"] for e in results["events"] if e["type"] == "interaction"]
    assert outcomes == ["success", "success", "failure", "failure"]


def test_transfer_is_a_precondition_for_other_effects():
    spec = {
        "environment": NONE_ENV,
        "agent_types": {
            "seller": {
                "initial_count": 1,
                "initial_state": {"wheat": 1, "gold": 0},
                "behavior": {"rules": [
                    {"do": [{"type": "interact", "target": {"random": {"type": "baker"}},
                             "interaction_type": "sell",
                             "params": {"transfer": {"attribute": "wheat", "amount": 1},
                                        "self_delta": {"gold": 2},
                                        "target_delta": {"gold": -2}}}]},
                ]},
            },
            "baker": {"initial_count": 1, "initial_state": {"gold": 10, "wheat": 0}},
        },
    }
    sim = _sim(spec)
    sim.run(3)
    seller = sim.agent_manager.get_agents_by_type("seller")[0]
    # sold once; later attempts had no wheat, so no gold was conjured
    assert (seller["wheat"], seller["gold"]) == (0, 2)


def test_nested_fields_are_evaluated():
    """Expressions inside maps (new_state, params, ...) are evaluated —
    previously they were stored as literal strings like "$xp"."""
    spec = {
        "environment": NONE_ENV,
        "agent_types": {
            "novice": {
                "initial_count": 1,
                "initial_state": {"xp": 7},
                "behavior": {"rules": [
                    {"do": [{"type": "transform", "new_type": "veteran",
                             "new_state": {"xp": {"*": ["$xp", 2]}, "rank": "$xp"}}]},
                ]},
            },
            "veteran": {"initial_count": 0, "initial_state": {}},
        },
    }
    sim = _sim(spec)
    sim.run(1)
    vet = sim.agent_manager.get_agents_by_type("veteran")[0]
    assert vet["xp"] == 14 and vet["rank"] == 7


def test_nearest_without_positions_is_not_always_the_first_agent():
    spec = {
        "environment": NONE_ENV,
        "agent_types": {
            "customer": {
                "initial_count": 40,
                "initial_state": {},
                "behavior": {"rules": [
                    {"do": [{"type": "interact", "target": {"nearest": {"type": "shop"}},
                             "interaction_type": "visit",
                             "params": {"target_delta": {"visits": 1}}}]},
                ]},
            },
            "shop": {"initial_count": 3, "initial_state": {"visits": 0}},
        },
    }
    sim = _sim(spec, seed=1)
    sim.run(1)
    visits = [s["visits"] for s in sim.agent_manager.get_agents_by_type("shop")]
    assert sum(visits) == 40 and all(v > 0 for v in visits)


# --- Loud failures ---

def _one_rule(rule, **extra):
    return {"environment": NONE_ENV, **extra,
            "agent_types": {"a": {"initial_count": 1, "initial_state": {"gold": 1},
                                  "behavior": {"rules": [rule]}}}}


@pytest.mark.parametrize("rule,match", [
    ({"do": [{"type": "interact", "target": {"random": {"type": "a"}},
              "params": {"target_min_state": {"gold": 1}}}]},
     "unknown keys \\['target_min_state'\\]"),
    ({"do": [{"type": "modify_state", "attribute": "gold", "detla": 1}]},
     "unknown fields \\['detla'\\]"),
    ({"when": {">": [{"total": "wolves"}, 0]}, "do": []}, "unknown agent type 'wolves'"),
    ({"when": {">": ["@price", 0]}, "do": []}, "unknown global '@price'"),
    ({"when": {">": ["&gold", 0]}, "do": []}, "nothing is selected"),
    ({"when": {"<": ["$gold"]}, "do": []}, "exactly 2 operand"),
    ({"do": [{"type": "interact", "params": {}}]}, "needs 'target'"),
])
def test_bad_specs_fail_at_construction(rule, match):
    with pytest.raises(RuleError, match=match):
        _sim(_one_rule(rule))


def test_world_rules_cannot_touch_agents():
    with pytest.raises(RuleError, match="no agent"):
        _sim(_one_rule({"do": []}, world_rules=[
            {"when": {">": ["$gold", 0]}, "do": []}]))
    with pytest.raises(RuleError, match="not allowed"):
        _sim(_one_rule({"do": []}, world_rules=[
            {"do": [{"type": "die"}]}]))


def test_runtime_errors_surface_with_location():
    spec = _one_rule({"do": [{"type": "modify_state", "attribute": "gold",
                              "value": {"/": ["$gold", {"-": ["$gold", 1]}]}}]})
    with pytest.raises(RuleError, match=r"rules\[0\].*division by zero.*agent 1 \(a\), step 1"):
        _sim(spec).run(3)


def test_reading_an_attribute_no_agent_has_fails_at_construction():
    spec = _one_rule({"when": {"<": ["$enrgy", 5]}, "do": []})
    with pytest.raises(RuleError, match=r"read '\$enrgy'.*initial_state"):
        _sim(spec)


def test_attributes_written_by_rules_or_carried_by_transform_are_known():
    spec = {
        "environment": NONE_ENV,
        "agent_types": {
            "larva": {
                "initial_count": 1, "initial_state": {"food": 0},
                "behavior": {"rules": [
                    {"do": [{"type": "modify_state", "attribute": "age", "value": 1}]},
                    {"when": {">": ["$age", 0]},
                     "do": [{"type": "transform", "new_type": "adult"}]}]},
            },
            "adult": {
                "initial_count": 0, "initial_state": {"wings": 2},
                "behavior": {"rules": [
                    # food and age come from the larva it used to be
                    {"when": {">": ["$age", "$food"]},
                     "do": [{"type": "modify_state", "attribute": "flights", "delta": "$wings"}]}]},
            },
        },
    }
    sim = _sim(spec)
    sim.run(3)
    adult = sim.agent_manager.get_agents_by_type("adult")[0]
    assert adult["wings"] == 2 and adult["flights"] == 2


# --- Checkpoints ---

def test_save_load_resumes_identically(tmp_path):
    spec = {
        "environment": {"type": "grid_2d", "dimensions": {"width": 15, "height": 15}},
        "globals": {"ticks": 0},
        "world_rules": [{"do": [{"type": "modify_global", "name": "ticks", "delta": 1}]}],
        "agent_types": {
            "walker": {
                "initial_count": 10,
                "initial_state": {"energy": 5},
                "behavior": {"rules": [
                    {"do": [{"type": "move", "direction": [{"choice": [-1, 0, 1]},
                                                           {"choice": [-1, 0, 1]}]}]},
                    {"prob": 0.2, "do": [{"type": "reproduce", "energy_cost": 1}]},
                ]},
            },
        },
    }
    straight = _sim(spec, seed=4)
    straight.run(10)

    resumed = _sim(spec, seed=4)
    resumed.run(5)
    resumed.save(tmp_path / "ck.json")
    resumed = Simulation.load(tmp_path / "ck.json")
    resumed.run(5)

    def snapshot(sim):
        return sorted((a.id, a.state["position"], a.state["energy"])
                      for a in sim.agent_manager.get_living_agents())

    assert snapshot(resumed) == snapshot(straight)
    assert resumed.globals == straight.globals == {"ticks": 10}


def test_pack_with_shared_state_round_trips():
    spec = dict(MARKET, globals={"tax": 0})
    pack = Pack.new("market", spec)
    pack.validate(deep=True)
    assert json.loads(pack.to_json())["models"]["base"]["globals"] == {"tax": 0}


# --- Variation, weighted branches, repeated targets ---

def test_initial_state_expressions_vary_per_agent():
    spec = {
        "environment": NONE_ENV,
        "globals": {"base_gold": 100},
        "agent_types": {
            "member": {
                "initial_count": 60,
                "initial_state": {"guild": {"choice": ["red", "blue", "green"]},
                                  "skill": {"uniform": [0, 1]},
                                  "gold": "@base_gold",
                                  "inventory": {"sword": 1}},
            },
        },
    }
    members = _sim(spec, seed=2).agent_manager.get_agents_by_type("member")
    assert {m["guild"] for m in members} == {"red", "blue", "green"}
    assert len({m["skill"] for m in members}) == 60
    assert all(m["gold"] == 100 and m["inventory"] == {"sword": 1} for m in members)
    again = _sim(spec, seed=2).agent_manager.get_agents_by_type("member")
    assert [m["skill"] for m in again] == [m["skill"] for m in members]


def test_initial_state_cannot_read_an_agent():
    spec = {"environment": NONE_ENV, "agent_types": {"a": {
        "initial_count": 1, "initial_state": {"x": {"+": ["$y", 1]}}}}}
    with pytest.raises(RuleError, match="initial_state.x.*no agent"):
        _sim(spec)


def test_choose_picks_exactly_one_weighted_branch():
    spec = {
        "environment": NONE_ENV,
        "agent_types": {
            "roller": {
                "initial_count": 1,
                "initial_state": {"common": 0, "rare": 0, "pulls": 0},
                "behavior": {"rules": [
                    {"do": [{"type": "modify_state", "attribute": "pulls", "delta": 1}]},
                    {"choose": [
                        {"weight": 90, "do": [{"type": "modify_state", "attribute": "common", "delta": 1}]},
                        {"weight": 10, "do": [{"type": "modify_state", "attribute": "rare", "delta": 1}]},
                    ]},
                ]},
            },
        },
    }
    sim = _sim(spec, seed=4)
    sim.run(2000)
    roller = sim.agent_manager.get_agents_by_type("roller")[0]
    assert roller["common"] + roller["rare"] == roller["pulls"] == 2000
    assert 140 < roller["rare"] < 260  # ~10%


def test_action_repeating_the_rule_target_acts_on_the_same_agent():
    """With a random selector, re-selecting in the action would trade with a
    different shop than the one the condition checked."""
    pick = {"random": {"type": "shop"}}
    spec = {
        "environment": NONE_ENV,
        "agent_types": {
            "buyer": {
                "initial_count": 1,
                "initial_state": {"gold": 1000},
                "behavior": {"rules": [
                    {"target": pick,
                     "when": {">": ["&stock", 0]},
                     "do": [{"type": "interact", "target": pick, "interaction_type": "buy",
                             "params": {"exchange": {"give": {"gold": 1},
                                                     "get": {"stock": 1}}}}]},
                ]},
            },
            "shop": {"initial_count": 5, "initial_state": {"stock": 1, "gold": 0}},
        },
    }
    results = _sim(spec, seed=0).run(30)
    outcomes = [e["outcome"] for e in results["events"] if e["type"] == "interaction"]
    assert outcomes == ["success"] * 5  # never bought from an empty shop


def test_filtered_random_selector_is_uniform_over_matches():
    spec = {
        "environment": NONE_ENV,
        "agent_types": {
            "buyer": {"initial_count": 1, "initial_state": {},
                      "behavior": {"rules": [
                          {"target": {"random": {"type": "shop",
                                                 "where": {"==": ["&open", 1]}}},
                           "do": [{"type": "interact", "interaction_type": "visit",
                                   "params": {"target_delta": {"visits": 1}}}]}]}},
            # 3 of 60 shops are open: rare matches exercise the fallback too
            "shop": {"initial_count": 60, "initial_state": {"open": 0, "visits": 0}},
        },
    }
    sim = _sim(spec, seed=9)
    shops = sim.agent_manager.get_agents_by_type("shop")
    for shop in shops[:3]:
        shop.state["open"] = 1
    sim.run(3000)
    visits = [s["visits"] for s in shops]
    assert sum(visits) == 3000 and sum(visits[3:]) == 0
    assert all(850 < v < 1150 for v in visits[:3])  # ~1000 each
