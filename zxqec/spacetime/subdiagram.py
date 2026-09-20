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

"""Per-spider program data and induced sub-diagrams.

On top of the ``timestep`` / ``delay`` vertex data of :mod:`pyzx.spacetime`,
a :class:`~zxqec.qprogram.QProgram` tags every spider of its master graph with

``stage``
    index of the program stage whose gate created the spider;
``logical_qubit``
    the circuit qubit the spider acts on (its ``qubit`` coordinate is the
    *physical* wire, which permutations may change from stage to stage).

:func:`induced_subdiagram` cuts an arbitrary subset of spiders out of such a
graph as a valid ZX-diagram, which is how the sub-diagram of a stage or of a
levelled-up block is obtained.
"""

from __future__ import annotations

from collections.abc import Callable

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.spacetime import get_timestep, set_timestep
from pyzx.utils import EdgeType, VertexType

__all__ = [
    'STAGE_KEY', 'LOGICAL_QUBIT_KEY',
    'get_stage', 'set_stage', 'get_logical_qubit', 'set_logical_qubit',
    'shift_timesteps', 'induced_subdiagram',
]

STAGE_KEY = 'stage'
LOGICAL_QUBIT_KEY = 'logical_qubit'


def get_stage(g: BaseGraph[VT, ET], v: VT, default: int | None = None) -> int | None:
    """The ``stage`` vertex data of ``v``, or ``default`` if unset."""
    return g.vdata(v, STAGE_KEY, default)


def set_stage(g: BaseGraph[VT, ET], v: VT, stage: int) -> None:
    g.set_vdata(v, STAGE_KEY, int(stage))


def get_logical_qubit(g: BaseGraph[VT, ET], v: VT, default: int | None = None) -> int | None:
    """The ``logical_qubit`` vertex data of ``v``, or ``default`` if unset."""
    return g.vdata(v, LOGICAL_QUBIT_KEY, default)


def set_logical_qubit(g: BaseGraph[VT, ET], v: VT, q: int) -> None:
    g.set_vdata(v, LOGICAL_QUBIT_KEY, int(q))


def shift_timesteps(g: BaseGraph[VT, ET], offset: int) -> None:
    """Add ``offset`` to the ``timestep`` of every vertex that has one (in place)."""
    for v in g.vertices():
        t = get_timestep(g, v)
        if t is not None:
            set_timestep(g, v, t + offset)


def _boundary(h: BaseGraph[VT, ET], g: BaseGraph[VT, ET], at: VT, wire_of: VT) -> VT:
    """A fresh boundary in ``h`` at the position of ``at``, on the logical wire of
    ``at`` if it has one, else that of ``wire_of``."""
    b = h.add_vertex(VertexType.BOUNDARY, g.qubit(at), g.row(at))
    k = get_logical_qubit(g, at, get_logical_qubit(g, wire_of))
    if k is not None:
        set_logical_qubit(h, b, k)
    return b


def induced_subdiagram(g: BaseGraph[VT, ET],
                       keep: Callable[[VT], bool],
                       key: Callable[[VT], float]) -> BaseGraph[VT, ET]:
    """The sub-diagram of ``g`` on the vertices selected by ``keep``.

    ``key`` orders the vertices in time (e.g. their stage index or timestep);
    it must be defined for every vertex, including boundaries, and every kept
    vertex must have a key. The result is a valid ZX-diagram with the same
    backend as ``g``:

    * an edge between two kept vertices is copied;
    * an edge from a kept vertex to a dropped one is closed with a fresh
      boundary, registered as an input if the dropped vertex comes earlier
      (``key`` smaller) and as an output otherwise;
    * an edge between two dropped vertices that lie on opposite sides of the
      kept set (one before its first key, one after its last) is a wire that
      passes through the selection untouched and becomes an identity wire,
      i.e. an input boundary connected to an output boundary -- so a block
      acting on a subset of the qubits still comes out as a map on all of them.

    Edge types follow the rule that an edge is created by the gate of its
    *later* endpoint: a Hadamard edge is kept only when the kept vertex is the
    later one, otherwise the Hadamard belongs to a gate outside the selection
    and the new boundary edge is plain. Fresh boundaries inherit the
    ``logical_qubit`` of their wire, and inputs and outputs are ordered by
    logical qubit (falling back to the qubit coordinate), then row, so the
    result has the boundary order of a circuit on those qubits. Vertex data is
    copied; the scalar is not.
    """
    kept = [v for v in g.vertices() if keep(v)]
    if not kept:
        raise ValueError("induced_subdiagram: no vertex selected")
    kept_set = set(kept)
    first = min(key(v) for v in kept)
    last = max(key(v) for v in kept)

    h = g.__class__()
    vmap: dict[VT, VT] = {}
    for v in kept:
        w = h.add_vertex(g.type(v), g.qubit(v), g.row(v), g.phase(v), ground=g.is_ground(v))
        for k in g.vdata_keys(v):
            h.set_vdata(w, k, g.vdata(v, k))
        vmap[v] = w

    inputs: list[VT] = []
    outputs: list[VT] = []
    for e in g.edges():
        s, t = g.edge_st(e)
        et = g.edge_type(e)
        s_in, t_in = s in kept_set, t in kept_set
        if s_in and t_in:
            h.add_edge((vmap[s], vmap[t]), et)
        elif s_in or t_in:
            inside, outside = (s, t) if s_in else (t, s)
            before = key(outside) < key(inside)
            b = _boundary(h, g, outside, inside)
            h.add_edge((vmap[inside], b), et if before else EdgeType.SIMPLE)
            (inputs if before else outputs).append(b)
        else:
            a, c = (s, t) if key(s) <= key(t) else (t, s)
            if key(a) < first and key(c) > last:
                bi = _boundary(h, g, a, c)
                bo = _boundary(h, g, c, a)
                h.add_edge((bi, bo), EdgeType.SIMPLE)
                inputs.append(bi)
                outputs.append(bo)

    def order(b: VT) -> tuple[float, float]:
        k = get_logical_qubit(h, b)
        return (float(h.qubit(b)) if k is None else float(k), float(h.row(b)))

    h.set_inputs(tuple(sorted(inputs, key=order)))
    h.set_outputs(tuple(sorted(outputs, key=order)))
    return h
