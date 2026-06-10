"""Tests for the pack format (agentstan/pack.py)."""
import json

import pytest

from agentstan import Pack, PackError
from agentstan.pack import load, save, FORMAT_NAME, SCHEMA_VERSION


def base_model():
    return {
        "environment": {"type": "grid_2d",
                        "dimensions": {"width": 10, "height": 10, "topology": "torus"}},
        "agent_types": {
            "fox": {
                "initial_count": 10,
                "initial_state": {"energy": 20, "perception_radius": 3},
                "behavior": {"rules": [
                    {"do": [{"type": "move", "direction": [{"choice": [-1, 0, 1]},
                                                            {"choice": [-1, 0, 1]}]}]},
                    {"do": [{"type": "modify_state", "attribute": "energy", "delta": -0.5}]},
                    {"when": {"<=": ["$energy", 0]},
                     "do": [{"type": "die", "cause": "starvation"}]},
                ]},
            },
        },
    }


def base_pack_data():
    return {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "name": "fox-world",
        "description": "Foxes wandering",
        "metadata": {"author": "test"},
        "models": {"base": base_model()},
        "scenarios": {
            "crowded": {
                "model": "base",
                "description": "Lots of foxes",
                "overrides": {"agent_types.fox.initial_count": 50},
                "steps": 10,
                "seed": 42,
            },
        },
        "default_model": "base",
    }


def test_pack_roundtrip(tmp_path):
    path = str(tmp_path / "fox.pack.json")
    save(Pack(base_pack_data()), path)
    pack = load(path)
    assert pack.name == "fox-world"
    assert pack.models == ["base"]
    assert pack.scenarios == ["crowded"]
    assert pack.to_dict() == base_pack_data()


def test_scenario_resolution_applies_overrides():
    pack = Pack(base_pack_data())
    spec = pack.spec("crowded")
    assert spec["agent_types"]["fox"]["initial_count"] == 50
    assert spec["seed"] == 42
    assert spec["steps"] == 10
    # The underlying model is untouched
    assert pack.spec("base")["agent_types"]["fox"]["initial_count"] == 10


def test_run_scenario_and_default():
    pack = Pack(base_pack_data())
    results = pack.run("crowded")
    assert results["summary"]["initial_counts"]["fox"] == 50
    assert results["final_step"] == 10
    assert results["seed"] == 42

    default = pack.run(steps=5)
    assert default["summary"]["initial_counts"]["fox"] == 10
    assert default["final_step"] == 5


def test_scenario_runs_are_reproducible():
    pack = Pack(base_pack_data())
    a = pack.run("crowded")
    b = pack.run("crowded")
    assert a["metrics"]["history"] == b["metrics"]["history"]


def test_pack_new_wraps_a_spec():
    pack = Pack.new("quick", base_model(), description="from chat")
    assert pack.default_model == "base"
    results = pack.run(steps=3)
    assert results["final_step"] >= 1


def test_rejects_wrong_format():
    data = base_pack_data()
    data["format"] = "something-else"
    with pytest.raises(PackError, match="format"):
        Pack(data)


def test_rejects_newer_schema_version():
    data = base_pack_data()
    data["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(PackError, match="newer than this library"):
        Pack(data)


def test_rejects_unknown_scenario_model_ref():
    data = base_pack_data()
    data["scenarios"]["crowded"]["model"] = "missing"
    with pytest.raises(PackError, match="references model 'missing'"):
        Pack(data)


def test_rejects_invalid_model_spec():
    data = base_pack_data()
    del data["models"]["base"]["environment"]
    with pytest.raises(PackError, match="models\\['base'\\]"):
        Pack(data)


def test_unknown_name_lists_options():
    pack = Pack(base_pack_data())
    with pytest.raises(PackError, match="models.*scenarios"):
        pack.spec("nope")


def test_bad_override_path_raises():
    data = base_pack_data()
    data["scenarios"]["crowded"]["overrides"] = {"agent_types.wolf.initial_count": 5}
    pack = Pack(data)  # structural validation passes
    with pytest.raises(PackError, match="override path"):
        pack.spec("crowded")


def test_deep_validation_catches_bad_rules():
    data = base_pack_data()
    data["models"]["base"]["agent_types"]["fox"]["behavior"]["rules"][0]["do"][0]["type"] = "fly"
    pack = Pack(data)  # structural validation doesn't compile rules
    with pytest.raises(PackError, match="fails engine validation"):
        pack.validate(deep=True)


def test_pack_file_is_pure_json(tmp_path):
    path = str(tmp_path / "fox.pack.json")
    save(Pack(base_pack_data()), path)
    with open(path) as f:
        text = f.read()
    assert "def " not in text and "lambda" not in text
    json.loads(text)
