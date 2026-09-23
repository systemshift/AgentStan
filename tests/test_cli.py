"""Tests for the command line interface."""
import json

import pytest

from agentstan.__main__ import main
from agentstan import Pack

SPEC = {
    "seed": 3,
    "steps": 12,
    "environment": {"type": "none"},
    "globals": {"minted": 0},
    "observables": {"gold": {"sum": {"attr": "gold"}}},
    "agent_types": {
        "miner": {
            "initial_count": 3,
            "initial_state": {"gold": 0},
            "behavior": {"rules": [
                {"do": [{"type": "modify_state", "attribute": "gold", "delta": "@minted"},
                        {"type": "modify_global", "name": "minted", "delta": 1}]},
            ]},
        }
    },
}


@pytest.fixture
def files(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(SPEC))
    pack = Pack.new("miners", SPEC).to_dict()
    pack["scenarios"] = {"short": {"model": "base", "steps": 2}}
    pack_path = tmp_path / "miners.pack.json"
    pack_path.write_text(json.dumps(pack))
    return spec_path, pack_path


def test_run_uses_spec_steps_and_prints_observables(files, capsys):
    main(["run", str(files[0])])
    out = capsys.readouterr().out
    assert "steps: 12" in out and "observables: gold=" in out


def test_run_pack_scenario(files, capsys, tmp_path):
    main(["run", str(files[1]), "short", "-o", str(tmp_path / "r.json")])
    assert "steps: 2" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json").read_text())["final_step"] == 2


def test_validate_ok_and_failure(files, capsys, tmp_path):
    main(["validate", str(files[1])])
    assert "ok: base, short" in capsys.readouterr().out
    bad = dict(SPEC, globals={})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit) as exit_info:
        main(["validate", str(path)])
    assert exit_info.value.code == 1
    assert "unknown global '@minted'" in capsys.readouterr().out


def test_batch_with_vary(files, capsys):
    main(["batch", str(files[0]), "--runs", "2", "--workers", "1",
          "--vary", "agent_types.miner.initial_count=1,4"])
    out = capsys.readouterr().out
    assert "agent_types.miner.initial_count=1" in out
    assert "agent_types.miner.initial_count=4" in out
    assert "gold" in out
