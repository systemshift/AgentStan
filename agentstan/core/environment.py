"""
Flexible environment system supporting multiple spatial types and custom properties
"""

import copy
import random
from typing import Dict, Any, List, Tuple, Optional, Callable


class Environment:
    """
    Base environment supporting grids, continuous space, networks, or custom types
    """

    def __init__(self, env_type: str, dimensions: Dict[str, Any],
                 properties: Optional[Dict[str, Any]] = None, rng=None):
        """
        Initialize environment

        Args:
            env_type: "grid_2d", "continuous_2d", "network", or "custom"
            dimensions: Type-specific dimensions
            properties: Custom environment properties
            rng: random.Random instance for determinism (defaults to the
                global random module)
        """
        self.env_type = env_type
        self.dimensions = dimensions
        self.properties = properties or {}
        self.step = 0
        self._rng = rng if rng is not None else random

        # Initialize spatial structure based on type
        if env_type == "grid_2d":
            self._init_grid_2d()
        elif env_type == "continuous_2d":
            self._init_continuous_2d()
        elif env_type == "network":
            self._init_network()
        elif env_type == "none":
            # Non-spatial: no positions, no movement; every agent can
            # perceive and interact with every other. The natural fit for
            # markets, economies, and games where geography is irrelevant.
            pass
        elif env_type == "custom":
            self._init_custom()
        else:
            raise ValueError(f"Unknown environment type: {env_type}")

    def _init_grid_2d(self):
        """Initialize 2D grid environment"""
        self.width = self.dimensions.get("width", 50)
        self.height = self.dimensions.get("height", 50)
        self.topology = self.dimensions.get("topology", "torus")  # torus or bounded

        # Initialize grid cells
        self.cells = {}
        for x in range(self.width):
            for y in range(self.height):
                self.cells[(x, y)] = {
                    "position": (x, y),
                    "agents": [],
                    "properties": {}
                }

    def _init_continuous_2d(self):
        """Initialize continuous 2D space"""
        self.width = self.dimensions.get("width", 100.0)
        self.height = self.dimensions.get("height", 100.0)
        self.bounded = self.dimensions.get("bounded", True)

    def _init_network(self):
        """Initialize network/graph environment.

        Supported dimensions:
          - node_count: number of nodes (default 50)
          - edges: list of [a, b] pairs for explicit edges
          - topology: "complete" | "ring" | "lattice" | "random"
          - edge_probability: float (used by "random" topology, default 0.1)
          - lattice_dims: [rows, cols] (used by "lattice" topology)
        """
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.edges: List[Tuple[int, int]] = []
        self._adjacency: Dict[int, set] = {}

        node_count = self.dimensions.get("node_count", 50)
        for i in range(node_count):
            self.nodes[i] = {"id": i, "agents": [], "properties": {}}
            self._adjacency[i] = set()

        for pair in self.dimensions.get("edges", []) or []:
            if len(pair) == 2:
                self.add_edge(pair[0], pair[1])

        topology = self.dimensions.get("topology")
        if topology == "complete":
            for i in range(node_count):
                for j in range(i + 1, node_count):
                    self.add_edge(i, j)
        elif topology == "ring":
            for i in range(node_count):
                self.add_edge(i, (i + 1) % node_count)
        elif topology == "lattice":
            rows, cols = self.dimensions.get("lattice_dims", [0, 0])
            if rows * cols == node_count:
                for r in range(rows):
                    for c in range(cols):
                        node = r * cols + c
                        if c + 1 < cols:
                            self.add_edge(node, r * cols + c + 1)
                        if r + 1 < rows:
                            self.add_edge(node, (r + 1) * cols + c)
        elif topology == "random":
            edge_prob = self.dimensions.get("edge_probability", 0.1)
            for i in range(node_count):
                for j in range(i + 1, node_count):
                    if self._rng.random() < edge_prob:
                        self.add_edge(i, j)

    def add_edge(self, a: int, b: int) -> None:
        """Add an undirected edge between two nodes."""
        if self.env_type != "network":
            raise ValueError("add_edge only valid for network environments")
        if a not in self.nodes or b not in self.nodes:
            raise ValueError(f"Unknown node(s): {a}, {b}")
        if a == b or b in self._adjacency[a]:
            return
        self.edges.append((a, b))
        self._adjacency[a].add(b)
        self._adjacency[b].add(a)

    def _init_custom(self):
        """Initialize custom environment type"""
        # Custom environments are fully defined by properties
        pass

    def get_random_position(self) -> Any:
        """Get a random valid position in the environment"""
        if self.env_type == "grid_2d":
            return (self._rng.randint(0, self.width - 1),
                   self._rng.randint(0, self.height - 1))
        elif self.env_type == "continuous_2d":
            return (self._rng.uniform(0, self.width),
                   self._rng.uniform(0, self.height))
        elif self.env_type == "network":
            return self._rng.choice(list(self.nodes.keys()))
        else:
            return None

    def get_neighbors(self, position: Any, radius: float = 1) -> List[Any]:
        """
        Get neighboring positions within radius

        Args:
            position: Current position
            radius: Radius to search

        Returns:
            List of neighboring positions
        """
        if self.env_type == "grid_2d":
            return self._get_grid_neighbors(position, int(radius))
        elif self.env_type == "continuous_2d":
            # For continuous space, would need all positions to check distance
            # This is handled at agent level instead
            return []
        elif self.env_type == "network":
            return self._get_network_neighbors(position)
        else:
            return []

    def _get_grid_neighbors(self, position: Tuple[int, int], radius: int) -> List[Tuple[int, int]]:
        """Get grid neighbors within radius"""
        x, y = position
        neighbors = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                # Handle topology
                if self.topology == "torus":
                    nx = nx % self.width
                    ny = ny % self.height
                elif self.topology == "bounded":
                    if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                        continue

                neighbors.append((nx, ny))

        return neighbors

    def _get_network_neighbors(self, node_id: int) -> List[int]:
        """Get directly connected nodes in network (1-hop)."""
        return list(self._adjacency.get(node_id, ()))

    def get_nodes_within_hops(self, node_id: int, hops: int) -> List[int]:
        """BFS: nodes reachable within `hops` edges (excludes the node itself)."""
        if node_id not in self._adjacency or hops < 1:
            return []
        visited = {node_id}
        frontier = {node_id}
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                next_frontier.update(self._adjacency.get(n, ()))
            next_frontier -= visited
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        visited.discard(node_id)
        return list(visited)

    def distance(self, pos1: Any, pos2: Any) -> float:
        """
        Calculate distance between two positions

        Args:
            pos1: First position
            pos2: Second position

        Returns:
            Distance between positions
        """
        if self.env_type == "grid_2d":
            x1, y1 = pos1
            x2, y2 = pos2

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # Handle torus topology
            if self.topology == "torus":
                dx = min(dx, self.width - dx)
                dy = min(dy, self.height - dy)

            return (dx ** 2 + dy ** 2) ** 0.5

        elif self.env_type == "continuous_2d":
            x1, y1 = pos1
            x2, y2 = pos2
            return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        elif self.env_type == "network":
            # Graph distance via BFS, capped at node count.
            if pos1 == pos2:
                return 0
            if pos1 not in self._adjacency:
                return float("inf")
            visited = {pos1}
            frontier = {pos1}
            depth = 0
            while frontier:
                depth += 1
                next_frontier = set()
                for n in frontier:
                    for m in self._adjacency.get(n, ()):
                        if m == pos2:
                            return depth
                        if m not in visited:
                            visited.add(m)
                            next_frontier.add(m)
                frontier = next_frontier
            return float("inf")

        return 0

    def is_valid_position(self, position: Any) -> bool:
        """Check if position is valid in this environment"""
        if self.env_type == "grid_2d":
            x, y = position
            if self.topology == "torus":
                return True
            return 0 <= x < self.width and 0 <= y < self.height

        elif self.env_type == "continuous_2d":
            x, y = position
            if self.bounded:
                return 0 <= x < self.width and 0 <= y < self.height
            return True

        elif self.env_type == "network":
            return position in self.nodes

        return True

    def normalize_position(self, position: Any) -> Any:
        """Normalize position to valid coordinates"""
        if self.env_type == "grid_2d" and self.topology == "torus":
            x, y = position
            return (x % self.width, y % self.height)

        elif self.env_type == "continuous_2d" and self.bounded:
            x, y = position
            return (max(0, min(x, self.width)), max(0, min(y, self.height)))

        return position

    def get_property(self, name: str) -> Any:
        """Get environment property value"""
        return self.properties.get(name)

    def set_property(self, name: str, value: Any):
        """Set environment property value"""
        self.properties[name] = value

    def update(self, step: int):
        """
        Update environment state for new step

        Can be overridden by custom update functions
        """
        self.step = step

        # Update dynamic properties
        for prop_name, prop_config in list(self.properties.items()):
            if isinstance(prop_config, dict) and "type" in prop_config:
                if prop_config["type"] == "cyclic":
                    # Cyclic properties (e.g., seasons)
                    period = prop_config.get("period", 100)
                    values = prop_config.get("values", [])
                    if values:
                        index = (step % period) // (period // len(values))
                        self.properties[prop_name] = values[index % len(values)]

    def to_dict(self) -> Dict[str, Any]:
        """Export environment state as dictionary"""
        return {
            "type": self.env_type,
            "dimensions": self.dimensions,
            "properties": self.properties,
            "step": self.step
        }

    @staticmethod
    def from_dict(env_dict: Dict[str, Any], rng=None) -> 'Environment':
        """Create environment from dictionary specification"""
        return Environment(
            env_type=env_dict["type"],
            dimensions=env_dict.get("dimensions", {}),
            properties=env_dict.get("properties", {}),
            rng=rng,
        )
