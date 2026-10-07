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

"""Relaxed layouts: a diagram laid out by the lowest energy of a physical model, with no coordinates given.

A diagram is treated like a molecule: only its edges are given, and the positions
are where an energy is lowest. Two models from graph drawing, both in
``networkx`` (an optional dependency, ``pip install zxqec[layout]``):

* ``'kamada-kawai'`` (default): a spring between every pair of vertices, whose rest
  length is their distance in the graph (the number of edges between them), and
  the total spring energy minimised (:func:`networkx.kamada_kawai_layout`, with
  scipy's L-BFGS). Vertices close in the diagram end up close in space.
* ``'spring'``: the Fruchterman-Reingold force simulation, edges pulling and all
  vertices pushing each other apart (:func:`networkx.spring_layout`).

The start is random but seeded, so a layout is reproducible. Each connected part
of the diagram is relaxed on its own, and the parts are set side by side along the
first axis; lengths are scaled so that an edge is about one unit long. The
positions mean nothing beyond the connectivity: they are neither space nor time.

:func:`draw_relaxed` draws a diagram in 2D with PyZX's own ``draw_matplotlib`` at
the relaxed positions; :func:`draw_relaxed_3d` in 3D with
:func:`~zxqec.spacetime.draw_graph_3d`. Both can ``shade`` the neighbours of chosen
spiders, one colour each: for the diagram of a code (one spider per check, joined
to its qubits) these are the checks' faces or cells, so a code's geometry shows
(the plaquettes of the surface code, the faces of the Steane code, the cells of
the tetrahedral ``[[15,1,3]]`` code).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pyzx.graph.base import BaseGraph
from pyzx.utils import VertexType

__all__ = ['RELAX_METHODS', 'SHADE_COLORS', 'relax_layout', 'draw_relaxed', 'draw_relaxed_3d']

RELAX_METHODS = ('kamada-kawai', 'spring')

#: the colours of shaded checks, in turn (red, blue, lime, orange, violet, teal, pink, yellow)
SHADE_COLORS = ('#ff6b6b', '#4dabf7', '#a9e34b', '#ffa94d', '#b197fc', '#38d9a9', '#f783ac', '#ffd43b')


def relax_layout(g: BaseGraph, dim: int = 2, method: str = 'kamada-kawai', seed: int = 0,
                 gap: float = 1.5, boundaries: str = 'relax') -> dict[Any, tuple[float, ...]]:
    """Positions in ``dim`` dimensions for every vertex of ``g``, where the energy of ``method`` is lowest.

    See the module docstring. ``seed`` fixes the random start; ``gap`` is the space
    between two connected parts. ``boundaries='outward'`` leaves the inputs and
    outputs that hang off a spider out of the energy, and then puts each one unit
    from its spider, pointing away from the middle of its part: the shape is that
    of the spiders alone (for a code, its checks and qubits), with the wires
    sticking out. With ``'relax'`` (default) they are relaxed like any vertex.
    """
    if method not in RELAX_METHODS:
        raise ValueError("method must be one of {}, got {!r}".format(RELAX_METHODS, method))
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    if boundaries not in ('relax', 'outward'):
        raise ValueError("boundaries must be 'relax' or 'outward', got {!r}".format(boundaries))
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("relax_layout needs networkx (pip install zxqec[layout])") from e

    leaves: dict[Any, list[Any]] = {}                    # spider -> the boundaries hanging off it
    if boundaries == 'outward':
        for b in sorted(g.vertices()):
            if g.type(b) == VertexType.BOUNDARY:
                (nb, *rest) = list(g.neighbors(b)) or [None]
                if nb is not None and not rest and g.type(nb) != VertexType.BOUNDARY:
                    leaves.setdefault(nb, []).append(b)
    hanging = {b for bs in leaves.values() for b in bs}
    graph = nx.Graph()
    graph.add_nodes_from(v for v in g.vertices() if v not in hanging)
    graph.add_edges_from(e for e in (g.edge_st(x) for x in g.edges()) if not (set(e) & hanging))
    parts = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: (-len(c), c[0]))
    rng = np.random.default_rng(seed)
    out: dict[Any, tuple[float, ...]] = {}
    offset = 0.0
    for part in parts:
        if len(part) == 1:
            arr = np.zeros((1, dim))
        else:
            sub = graph.subgraph(part)
            start = {v: rng.uniform(-1, 1, dim) for v in part}
            if method == 'kamada-kawai':
                pos = nx.kamada_kawai_layout(sub, pos=start, dim=dim)
            else:
                pos = nx.spring_layout(sub, pos=start, dim=dim, seed=int(rng.integers(2 ** 31)), iterations=300)
            arr = np.array([pos[v] for v in part], dtype=float)
            index = {v: i for i, v in enumerate(part)}
            lengths = [np.linalg.norm(arr[index[a]] - arr[index[b]]) for a, b in sub.edges()]
            mean = float(np.mean(lengths)) if lengths else 0.0
            if mean > 0:
                arr = arr / mean                         # an edge is about one unit long
        arr = arr - arr.mean(axis=0)
        if leaves:
            part, arr = _hang(part, arr, leaves, dim)
        arr[:, 0] += offset - arr[:, 0].min()
        offset = float(arr[:, 0].max()) + gap
        for v, p in zip(part, arr):
            out[v] = tuple(float(c) for c in p)
    return out


def _hang(part: list[Any], arr: np.ndarray, leaves: dict[Any, list[Any]], dim: int) -> tuple[list[Any], np.ndarray]:
    """``part`` and its positions (centred), with the boundaries hanging off its spiders one unit outward.

    Outward is away from the middle of the part; for a spider in the middle, it is
    the direction furthest (in angle) from the spiders within two units of it.
    """
    radius = float(np.linalg.norm(arr, axis=1).mean()) if len(arr) else 0.0
    if dim == 2:
        turns = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        candidates = np.stack([np.cos(turns), np.sin(turns)], axis=1)
    else:                                                # a Fibonacci sphere
        k = np.arange(200) + 0.5
        polar, around = np.arccos(1 - 2 * k / 200), np.pi * (1 + 5 ** 0.5) * k
        candidates = np.stack([np.cos(around) * np.sin(polar), np.sin(around) * np.sin(polar), np.cos(polar)], axis=1)
    extra_v, extra_p = [], []
    for v, p in zip(part, arr):
        if not leaves.get(v):
            continue
        if np.linalg.norm(p) > 0.25 * radius:
            out = p / np.linalg.norm(p)
        else:
            near = [q - p for q in arr if 1e-9 < np.linalg.norm(q - p) < 2.0]
            dirs = [d / np.linalg.norm(d) for d in near]
            out = candidates[int(np.argmin([max((c @ d for d in dirs), default=-1.0) for c in candidates]))]
        others = [w for w in np.eye(dim) if abs(float(w @ out)) < 0.9]
        for k_, b in enumerate(leaves[v]):
            d = out if k_ == 0 else -out if k_ == 1 else others[(k_ - 2) % len(others)]
            extra_v.append(b)
            extra_p.append(p + d)
    if not extra_v:
        return part, arr
    return part + extra_v, np.vstack([arr, np.array(extra_p)])


def _shaded(g: BaseGraph, shade: Any) -> list[tuple[Any, list[Any]]]:
    """``(spider, its neighbours that are not boundaries)`` for each spider to shade.

    ``shade`` is ``'Z'`` or ``'X'`` (every spider of that colour with no boundary
    neighbour, such as the check spiders of a code's diagram), or the spiders
    themselves.
    """
    if shade is None:
        return []
    if isinstance(shade, str):
        kinds = {'Z': VertexType.Z, 'X': VertexType.X}
        if shade not in kinds:
            raise ValueError("shade must be 'Z', 'X' or a list of spiders, got {!r}".format(shade))
        spiders = [v for v in sorted(g.vertices()) if g.type(v) == kinds[shade]
                   and not any(g.type(w) == VertexType.BOUNDARY for w in g.neighbors(v))]
    else:
        spiders = list(shade)
    return [(v, sorted(w for w in g.neighbors(v) if g.type(w) != VertexType.BOUNDARY)) for v in spiders]


def _polygon(points: np.ndarray) -> np.ndarray | None:
    """The corners of the convex hull of planar ``points`` (any dimension), in order around it; None if too few."""
    centre = points.mean(axis=0)
    _, sv, basis = np.linalg.svd(points - centre)
    if len(points) < 3 or sv[1] < 1e-9 * max(sv[0], 1e-12):
        return None
    flat = (points - centre) @ basis[:2].T
    from scipy.spatial import ConvexHull
    return points[ConvexHull(flat).vertices]


def draw_relaxed(g: BaseGraph, method: str = 'kamada-kawai', seed: int = 0, spacing: float = 1.2,
                 figsize: tuple[float, float] = (7, 5), shade: Any = None,
                 shade_colors: Sequence[str] = SHADE_COLORS, boundaries: str = 'relax', **kwargs: Any) -> Any:
    """PyZX's 2D drawing of ``g`` (``draw_matplotlib``) with every vertex at its place in :func:`relax_layout`.

    ``spacing`` is the length of an edge in PyZX's units (a spider has radius 0.2).
    ``shade`` (``'Z'``, ``'X'`` or a list of spiders) shades the convex hull of
    each such spider's neighbours, in the colours ``shade_colors`` in turn (a
    spider with two neighbours is a corner of its own patch). Other keyword
    arguments go to ``draw_matplotlib``; ``boundaries`` to :func:`relax_layout`.
    Returns the figure.
    """
    from matplotlib.patches import Polygon
    from pyzx.drawing import draw_matplotlib
    from pyzx.utils import get_mode, settings

    from .drawing import _finish

    h = g.clone()                                       # clone keeps the vertex numbers
    pos = relax_layout(g, 2, method, seed, boundaries=boundaries)
    for v, (x, y) in pos.items():
        h.set_row(v, spacing * x)
        h.set_qubit(v, spacing * y)
    mode = get_mode()
    settings.mode = "notebook"                          # PyZX would call plt.show() itself otherwise
    try:
        fig = draw_matplotlib(h, figsize=figsize, **kwargs)
    finally:
        settings.mode = mode
    at = lambda v: (spacing * pos[v][0], -spacing * pos[v][1])     # where draw_matplotlib puts v
    for i, (v, around) in enumerate(_shaded(g, shade)):
        corners = _polygon(np.array([at(w) for w in around] + ([at(v)] if len(around) < 3 else []), dtype=float))
        if corners is not None:
            fig.axes[0].add_patch(Polygon(corners, closed=True, facecolor=shade_colors[i % len(shade_colors)],
                                          alpha=0.35, edgecolor='none', zorder=0))
    return _finish(fig)


def draw_relaxed_3d(g: BaseGraph, method: str = 'kamada-kawai', seed: int = 0, shade: Any = None,
                    shade_colors: Sequence[str] = SHADE_COLORS, box: bool = True, boundaries: str = 'relax',
                    views: Sequence[Any] | None = None, titles: Sequence[str] | None = None,
                    **kwargs: Any) -> Any:
    """:func:`~zxqec.spacetime.draw_graph_3d` of ``g`` with every vertex at its place in :func:`relax_layout`.

    The axes carry no labels or ticks, since the positions are neither space nor
    time; turn the picture with the mouse in a window. ``shade`` as for
    :func:`draw_relaxed`: the neighbours of each such spider are shaded as a
    translucent solid (their convex hull), or as a polygon when they lie nearly
    in a plane. ``box=False`` leaves out the axes box and scales the three
    directions equally, so that a shape (a cube, a tetrahedron) is not
    stretched. ``views`` (a list of ``view`` values) draws the same layout from
    each of them, side by side in one figure, with ``titles`` above them.
    ``boundaries`` goes to :func:`relax_layout`; other keyword arguments to
    ``draw_graph_3d`` (e.g. ``ax``, ``figsize``, ``show_phases``, ``view``).
    Returns the figure.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from .drawing import _finish, draw_graph_3d

    if views is not None:
        import matplotlib.pyplot as plt
        if 'ax' in kwargs or 'view' in kwargs:
            raise ValueError("give either views or ax / view")
        fig = plt.figure(figsize=kwargs.pop('figsize', (4.2 * len(views), 4.2)))
        kwargs.pop('finish', None)
        for i, view in enumerate(views):
            ax = fig.add_subplot(1, len(views), i + 1, projection='3d')
            draw_relaxed_3d(g, method, seed, shade, shade_colors, box, boundaries, view=view, ax=ax, finish=False,
                            **kwargs)
            if titles:
                ax.set_title(titles[i], fontsize=10)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92 if titles else 1, wspace=0)
        return _finish(fig)

    kwargs.setdefault('show_topology', False)
    kwargs.setdefault('layer_lines', False)
    finish = kwargs.pop('finish', True)
    pos = relax_layout(g, 3, method, seed, boundaries=boundaries)
    fig = draw_graph_3d(g, None, positions=pos, finish=False, **kwargs)
    ax = kwargs.get('ax') or fig.axes[0]
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for i, (v, around) in enumerate(_shaded(g, shade)):
        color = shade_colors[i % len(shade_colors)]
        pts = np.array([pos[w] for w in around] + ([pos[v]] if len(around) < 3 else []), dtype=float)
        if len(pts) < 3:
            continue
        sv = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
        if len(pts) == 3 or sv[2] < 0.15 * sv[1]:              # (nearly) flat: one polygon
            corners = _polygon(pts)
            faces = [corners] if corners is not None else []
        else:
            from scipy.spatial import ConvexHull
            faces = [pts[t] for t in ConvexHull(pts).simplices]
        if faces:
            ax.add_collection3d(Poly3DCollection(faces, facecolor=color, alpha=0.22, edgecolor='none'))
    if not box:
        coords = np.array(list(pos.values()), dtype=float)
        mid = (coords.max(axis=0) + coords.min(axis=0)) / 2
        half = float(np.linalg.norm(coords - mid, axis=1).max())    # a sphere: the drawing fits from every side
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_box_aspect((1, 1, 1), zoom=1.6)
        ax.set_axis_off()
        if 'ax' not in kwargs:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return _finish(fig) if finish else fig
