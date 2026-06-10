"""
Frame recording for visualization and replay.

A FrameRecorder is a collector that captures a compact spatial snapshot of
the simulation each step: which agents exist, where they are, what type
they are. The output is pure JSON — a renderer (web canvas, notebook,
video export) replays it without touching the engine.

Frame format (compact arrays to keep payloads small):

    {
      "environment": {"type": "grid_2d", "dimensions": {...}},   # header
      "types": ["rabbit", "wolf"],                                # index -> name
      "frames": [
        {"step": 1, "agents": [[id, type_index, x, y], ...]},
        ...
      ]
    }

For network environments the agent entry is [id, type_index, node_id] and
the header includes the edge list, so a renderer can lay out the graph.
"""

from typing import Any, Dict, List, Optional


class FrameRecorder:
    """
    Collector that records agent positions each step for replay.

    Args:
        every: Record every N-th step (1 = every step).
        max_frames: Stop recording after this many frames (size guard).

    Usage:
        recorder = FrameRecorder()
        sim.add_collector(recorder)
        sim.run(200)
        replay = recorder.to_dict()   # JSON-safe
    """

    def __init__(self, every: int = 1, max_frames: int = 1000):
        self.every = max(1, int(every))
        self.max_frames = max_frames
        self.frames: List[Dict[str, Any]] = []
        self.environment: Optional[Dict[str, Any]] = None
        self._types: List[str] = []
        self._type_index: Dict[str, int] = {}

    def _type_idx(self, agent_type: str) -> int:
        idx = self._type_index.get(agent_type)
        if idx is None:
            idx = len(self._types)
            self._types.append(agent_type)
            self._type_index[agent_type] = idx
        return idx

    def _capture_environment(self, simulation) -> None:
        env = simulation.environment
        header = {"type": env.env_type, "dimensions": dict(env.dimensions)}
        if env.env_type == "network":
            header["nodes"] = list(env.nodes.keys())
            header["edges"] = [list(e) for e in env.edges]
        self.environment = header

    def collect(self, simulation) -> None:
        """Record one frame. Called by the simulation each step."""
        if simulation.step % self.every != 0:
            return
        if len(self.frames) >= self.max_frames:
            return
        if self.environment is None:
            self._capture_environment(simulation)

        is_network = simulation.environment.env_type == "network"
        agents = []
        for agent in simulation.agent_manager.get_living_agents():
            pos = agent.state.get("position")
            if pos is None:
                continue
            ti = self._type_idx(agent.type)
            if is_network:
                agents.append([agent.id, ti, pos])
            else:
                x, y = pos
                agents.append([agent.id, ti, round(float(x), 2), round(float(y), 2)])

        self.frames.append({"step": simulation.step, "agents": agents})

    def to_dict(self) -> Dict[str, Any]:
        """Full replay data, JSON-safe."""
        return {
            "environment": self.environment,
            "types": list(self._types),
            "frames": self.frames,
        }

    def reset(self) -> None:
        self.frames = []
        self.environment = None
        self._types = []
        self._type_index = {}
