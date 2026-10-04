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
:func:`~zxqec.spacetime.draw_graph_3d`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyzx.graph.base import BaseGraph

__all__ = ['RELAX_METHODS', 'relax_layout', 'draw_relaxed', 'draw_relaxed_3d']

RELAX_METHODS = ('kamada-kawai', 'spring')


def relax_layout(g: BaseGraph, dim: int = 2, method: str = 'kamada-kawai', seed: int = 0,
                 gap: float = 1.5) -> dict[Any, tuple[float, ...]]:
    """Positions in ``dim`` dimensions for every vertex of ``g``, where the energy of ``method`` is lowest.

    See the module docstring. ``seed`` fixes the random start; ``gap`` is the space
    between two connected parts.
    """
    if method not in RELAX_METHODS:
        raise ValueError("method must be one of {}, got {!r}".format(RELAX_METHODS, method))
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("relax_layout needs networkx (pip install zxqec[layout])") from e

    graph = nx.Graph()
    graph.add_nodes_from(g.vertices())
    graph.add_edges_from(g.edge_st(e) for e in g.edges())
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
        arr[:, 0] += offset - arr[:, 0].min()
        offset = float(arr[:, 0].max()) + gap
        for v, p in zip(part, arr):
            out[v] = tuple(float(c) for c in p)
    return out


def draw_relaxed(g: BaseGraph, method: str = 'kamada-kawai', seed: int = 0, spacing: float = 1.2,
                 figsize: tuple[float, float] = (7, 5), **kwargs: Any) -> Any:
    """PyZX's 2D drawing of ``g`` (``draw_matplotlib``) with every vertex at its place in :func:`relax_layout`.

    ``spacing`` is the length of an edge in PyZX's units (a spider has radius 0.2);
    other keyword arguments go to ``draw_matplotlib``. Returns the figure.
    """
    from pyzx.drawing import draw_matplotlib
    from pyzx.utils import get_mode, settings

    from .drawing import _finish

    h = g.clone()                                       # clone keeps the vertex numbers
    for v, (x, y) in relax_layout(g, 2, method, seed).items():
        h.set_row(v, spacing * x)
        h.set_qubit(v, spacing * y)
    mode = get_mode()
    settings.mode = "notebook"                          # PyZX would call plt.show() itself otherwise
    try:
        fig = draw_matplotlib(h, figsize=figsize, **kwargs)
    finally:
        settings.mode = mode
    return _finish(fig)


def draw_relaxed_3d(g: BaseGraph, method: str = 'kamada-kawai', seed: int = 0, **kwargs: Any) -> Any:
    """:func:`~zxqec.spacetime.draw_graph_3d` of ``g`` with every vertex at its place in :func:`relax_layout`.

    The axes carry no labels or ticks, since the positions are neither space nor
    time; turn the picture with the mouse in a window. Keyword arguments go to
    ``draw_graph_3d`` (e.g. ``ax``, ``figsize``, ``show_phases``, ``view``). Returns the figure.
    """
    from .drawing import draw_graph_3d

    kwargs.setdefault('show_topology', False)
    kwargs.setdefault('layer_lines', False)
    fig = draw_graph_3d(g, None, positions=relax_layout(g, 3, method, seed), **kwargs)
    ax = kwargs.get('ax') or fig.axes[0]
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return fig
