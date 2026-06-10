"""Tests for FrameRecorder (core/frames.py)."""
import json

from agentstan import FrameRecorder, Pack, Simulation


SPEC = {
    "environment": {"type": "grid_2d",
                    "dimensions": {"width": 12, "height": 12, "topology": "torus"}},
    "agent_types": {
        "fox": {
            "initial_count": 6,
            "initial_state": {"perception_radius": 2},
            "behavior": {"rules": [
                {"do": [{"type": "move", "direction": [{"choice": [-1, 0, 1]},
                                                        {"choice": [-1, 0, 1]}]}]},
            ]},
        },
        "hen": {"initial_count": 4, "initial_state": {}},
    },
}


def test_records_frames_with_positions_and_types():
    recorder = FrameRecorder()
    sim = Simulation(SPEC, seed=1)
    sim.add_collector(recorder)
    sim.run(10)

    data = recorder.to_dict()
    assert data["environment"]["type"] == "grid_2d"
    assert set(data["types"]) == {"fox", "hen"}
    assert len(data["frames"]) == 10
    first = data["frames"][0]
    assert first["step"] == 1
    assert len(first["agents"]) == 10
    agent_id, type_idx, x, y = first["agents"][0]
    assert data["types"][type_idx] in ("fox", "hen")
    assert 0 <= x < 12 and 0 <= y < 12
    # Pure JSON
    json.dumps(data)


def test_every_and_max_frames():
    recorder = FrameRecorder(every=5, max_frames=3)
    sim = Simulation(SPEC, seed=1)
    sim.add_collector(recorder)
    sim.run(40)
    steps = [f["step"] for f in recorder.frames]
    assert steps == [5, 10, 15]


def test_network_frames_include_edges():
    spec = {
        "environment": {"type": "network",
                        "dimensions": {"node_count": 6, "topology": "ring"}},
        "agent_types": {"walker": {"initial_count": 3, "initial_state": {}}},
    }
    recorder = FrameRecorder()
    sim = Simulation(spec, seed=2)
    sim.add_collector(recorder)
    sim.run(3)

    data = recorder.to_dict()
    assert data["environment"]["type"] == "network"
    assert len(data["environment"]["edges"]) == 6  # ring of 6
    aid, ti, node = data["frames"][0]["agents"][0]
    assert node in data["environment"]["nodes"]


def test_pack_run_accepts_collectors():
    pack = Pack.new("foxes", dict(SPEC))
    recorder = FrameRecorder()
    results = pack.run(steps=5, seed=3, collectors=[recorder])
    assert results["final_step"] == 5
    assert len(recorder.frames) == 5
