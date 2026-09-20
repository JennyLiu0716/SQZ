# zxqec - space-time ZX-diagrams for dynamic quantum error correction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware topology: an undirected graph of qubit sites with coordinates.

A :class:`Topology` records where the physical qubits sit (1D, 2D or 3D
coordinates) and which pairs are connected. It is used by
:class:`~zxqec.qprogram.QProgram` to validate permutation layers (only swaps
along edges are allowed when ``check_topology=True``) and by the 2+1D drawing
code to place the qubit wires in space.

This is a port of the thesis ``Topology`` class with plain names
(``neighbors`` instead of ``get_neighbors``, ``distance`` instead of
``get_edge_distance``, ``coordinates`` instead of ``get_node_coordinates``,
``draw`` instead of ``visualize``) and constructors for the common lattices.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

__all__ = ['Topology']

Coord = tuple[float, ...]


class Topology:
    """Undirected graph whose nodes are qubit sites with coordinates.

    Args:
        dimension: number of spatial coordinates per node (1, 2 or 3).
    """

    def __init__(self, dimension: int = 2) -> None:
        if dimension not in (1, 2, 3):
            raise ValueError("dimension must be 1, 2 or 3, got {}".format(dimension))
        self.dimension = dimension
        self.nodes: dict[int, Coord] = {}
        self.edges: set[frozenset[int]] = set()

    # -- construction -------------------------------------------------------

    @classmethod
    def line(cls, n: int) -> 'Topology':
        """A chain of ``n`` sites at integer positions ``0..n-1`` (1D)."""
        t = cls(1)
        for i in range(n):
            t.add_node(i, (float(i),))
        for i in range(n - 1):
            t.add_edge(i, i + 1)
        return t

    @classmethod
    def grid(cls, rows: int, cols: int) -> 'Topology':
        """A ``rows x cols`` square lattice (2D) with nearest-neighbour edges.

        Node ``r*cols + c`` sits at coordinates ``(c, r)``.
        """
        t = cls(2)
        for r in range(rows):
            for c in range(cols):
                t.add_node(r * cols + c, (float(c), float(r)))
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if c + 1 < cols:
                    t.add_edge(i, i + 1)
                if r + 1 < rows:
                    t.add_edge(i, i + cols)
        return t

    def add_node(self, node: int, coordinates: Iterable[float]) -> None:
        """Register site ``node`` at ``coordinates`` (length must equal the dimension)."""
        coords = tuple(float(c) for c in coordinates)
        if len(coords) != self.dimension:
            raise ValueError("expected {} coordinates for node {}, got {}".format(
                self.dimension, node, len(coords)))
        if node in self.nodes:
            raise ValueError("node {} already exists".format(node))
        self.nodes[node] = coords

    def add_edge(self, a: int, b: int) -> None:
        """Connect sites ``a`` and ``b`` (both must exist; no self-loops)."""
        if a not in self.nodes or b not in self.nodes:
            raise ValueError("both nodes must exist before adding an edge")
        if a == b:
            raise ValueError("self-loops are not allowed")
        self.edges.add(frozenset((a, b)))

    def remove_edge(self, a: int, b: int) -> None:
        self.edges.discard(frozenset((a, b)))

    def remove_node(self, node: int) -> None:
        """Remove ``node`` and every edge at it (no-op if absent)."""
        if node not in self.nodes:
            return
        self.edges = {e for e in self.edges if node not in e}
        del self.nodes[node]

    # -- queries ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node: object) -> bool:
        return node in self.nodes

    def __iter__(self) -> Iterator[int]:
        return iter(self.nodes)

    def coordinates(self, node: int) -> Coord:
        return self.nodes[node]

    def neighbors(self, node: int) -> list[int]:
        """Sites adjacent to ``node``, in increasing order."""
        out = []
        for e in self.edges:
            if node in e:
                (other,) = e - {node}
                out.append(other)
        return sorted(out)

    def are_adjacent(self, a: int, b: int) -> bool:
        return frozenset((a, b)) in self.edges

    def degree(self, node: int) -> int:
        return len(self.neighbors(node))

    def distance(self, a: int, b: int) -> float:
        """Euclidean distance between two sites (they need not be adjacent)."""
        return math.dist(self.nodes[a], self.nodes[b])

    def edge_list(self) -> list[tuple[int, int]]:
        """All edges as sorted ``(a, b)`` pairs, in a deterministic order."""
        return sorted(tuple(sorted(e)) for e in self.edges)  # type: ignore[misc]

    def is_connected(self) -> bool:
        """Whether every site is reachable from every other (empty topology counts as connected)."""
        if not self.nodes:
            return True
        start = next(iter(self.nodes))
        seen = {start}
        stack = [start]
        while stack:
            v = stack.pop()
            for w in self.neighbors(v):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == len(self.nodes)

    # -- (de)serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            'dimension': self.dimension,
            'nodes': {str(n): list(c) for n, c in self.nodes.items()},
            'edges': self.edge_list(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'Topology':
        t = cls(int(data['dimension']))
        for n, c in data['nodes'].items():
            t.add_node(int(n), c)
        for a, b in data['edges']:
            t.add_edge(int(a), int(b))
        return t

    def __repr__(self) -> str:
        return "Topology({}D, {} nodes, {} edges)".format(
            self.dimension, len(self.nodes), len(self.edges))

    # -- drawing ------------------------------------------------------------

    def draw(self, ax: Any = None, labels: bool = True, node_size: float = 200,
             node_color: str = 'skyblue', edge_color: str = 'gray') -> Any:
        """Draw the topology with matplotlib; returns the figure.

        1D and 2D topologies are drawn in the plane, 3D ones with a 3D axis.
        ``ax`` may be an existing axes (of the right projection) to draw into.
        """
        import matplotlib.pyplot as plt  # lazy: matplotlib is optional
        if self.dimension == 3:
            if ax is None:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection='3d')
            for a, b in self.edge_list():
                (x1, y1, z1), (x2, y2, z2) = self.nodes[a], self.nodes[b]
                ax.plot([x1, x2], [y1, y2], [z1, z2], color=edge_color, linewidth=1)
            xs, ys, zs = zip(*self.nodes.values()) if self.nodes else ((), (), ())
            ax.scatter(xs, ys, zs, s=node_size, c=node_color, edgecolors='black')
            if labels:
                for n, (x, y, z) in self.nodes.items():
                    ax.text(x, y, z, str(n), fontsize=9)
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 4))
            pos = {n: (c[0], c[1] if self.dimension == 2 else 0.0) for n, c in self.nodes.items()}
            for a, b in self.edge_list():
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        color=edge_color, linewidth=1, zorder=1)
            if pos:
                ax.scatter(*zip(*pos.values()), s=node_size, c=node_color,
                           edgecolors='black', zorder=2)
            if labels:
                for n, (x, y) in pos.items():
                    ax.annotate(str(n), (x, y), xytext=(0, 8), textcoords='offset points',
                                ha='center', fontsize=9)
            ax.set_aspect('equal')
            if self.dimension == 1:
                ax.set_yticks([])
        return ax.figure
