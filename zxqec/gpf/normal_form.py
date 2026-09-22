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

"""Reduction of a ZX-diagram to generalized parity form (GPF) with phases.

A diagram in GPF is bipartite: a layer of Z-spiders, each attached to at most
one input, followed by a layer of X-spiders, each attached to at most one
output, connected by plain edges (the phase-free normal form of Yeh et al.,
Algorithm 1 of the thesis). This module produces that shape for diagrams that
also carry phases:

* **Pauli phases** -- concrete ``pi`` or symbolic Boolean parameters such as
  measurement outcomes -- stay on the spiders. Fusion adds them, the Boolean
  bialgebra rule copies them to the new spiders (with a ``(-1)^{ab}`` global
  scalar), so the skeleton reduction goes through unchanged. In the result
  every layer spider may carry a Pauli phase.
* **Non-Pauli phases** (``T``, ``S``, arbitrary rotations) are isolated first
  into *phase gadgets*: the phase is unfused onto a degree-one *leaf* of the
  same colour, attached to the original spider through a phase-free *stem* of
  the opposite colour. Gadgets take no part in the skeleton reduction and are
  treated as fixed external targets by the matrix reduction.

``isolate_paulis=True`` isolates *every* non-zero phase, which reproduces the
thesis's Algorithm 2 (a phase-free skeleton plus gadgets).

Hadamard edges are pushed to the boundary wires beforehand (see
:mod:`zxqec.gpf.hadamard`), so the input may be any diagram in the
Clifford+phases fragment whose Hadamards can be absorbed by recolouring.
"""

from __future__ import annotations

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.rewrite_rules import bialgebra, check_bialgebra
from pyzx.simplify import hopf_simp, spider_simp
from pyzx.utils import EdgeType, VertexType, is_pauli, toggle_vertex

from .hadamard import push_hadamards_to_boundary

__all__ = ['GPF_ROLE_KEY', 'LEAF', 'to_gpf', 'is_leaf', 'is_stem', 'is_gpf', 'layout_gpf']

GPF_ROLE_KEY = 'gpf'
LEAF = 'leaf'


# -- roles ------------------------------------------------------------------

def is_leaf(g: BaseGraph[VT, ET], v: VT) -> bool:
    """A phase gadget's leaf: the degree-one spider that carries an isolated phase."""
    return g.vdata(v, GPF_ROLE_KEY, None) == LEAF


def is_stem(g: BaseGraph[VT, ET], v: VT) -> bool:
    """A spider adjacent to a leaf (the phase-free spider a gadget hangs from)."""
    return g.type(v) != VertexType.BOUNDARY and any(is_leaf(g, w) for w in g.neighbors(v))


def _touches(g: BaseGraph[VT, ET], v: VT, boundaries: set[VT]) -> bool:
    return any(w in boundaries for w in g.neighbors(v))


# -- the algorithm ----------------------------------------------------------

def to_gpf(g: BaseGraph[VT, ET], isolate_paulis: bool = False,
           max_steps: int | None = None) -> BaseGraph[VT, ET]:
    """Return a copy of ``g`` in generalized parity form with phases.

    Steps (Algorithm 2 of the thesis, generalised):

    1. push Hadamard edges to the boundary wires; fuse spiders;
    2. isolate phases into gadgets: all non-zero phases if ``isolate_paulis``,
       otherwise only non-Pauli ones;
    3. skeleton reduction: repeatedly apply the (Boolean) bialgebra rule to an
       X-spider not touching an output and a Z-neighbour not touching an input,
       neither being part of a gadget, with fusion and Hopf clean-up;
    4. pad every input with a Z-spider and every output with an X-spider where
       needed, and give every spider at most one boundary.

    ``max_steps`` bounds the number of bialgebra applications (default
    ``10 * (|V| + 1)^2``); exceeding it raises ``RuntimeError``.
    """
    h = g.copy()
    for v in h.vertices():
        if h.type(v) not in (VertexType.Z, VertexType.X, VertexType.BOUNDARY):
            raise ValueError("to_gpf: vertex {} has unsupported type {}".format(v, h.type(v)))
    push_hadamards_to_boundary(h)
    spider_simp(h)
    h.remove_isolated_vertices()

    _isolate_phases(h, isolate_paulis)
    _reduce_skeleton(h, max_steps if max_steps is not None else 10 * (h.num_vertices() + 1) ** 2)
    _pad_boundaries(h)
    _split_multi_boundary_spiders(h)
    layout_gpf(h)
    return h


def _isolate_phases(g: BaseGraph[VT, ET], isolate_paulis: bool) -> None:
    for v in list(g.vertices()):
        if g.type(v) == VertexType.BOUNDARY:
            continue
        p = g.phase(v)
        if p == 0 or (is_pauli(p) and not isolate_paulis):
            continue
        if g.vertex_degree(v) == 0:
            continue                          # a scalar; removed by the callers' clean-up
        if g.vertex_degree(v) == 1:
            g.set_vdata(v, GPF_ROLE_KEY, LEAF)  # already a leaf hanging off its neighbour
            continue
        g.set_phase(v, 0)
        stem = g.add_vertex(toggle_vertex(g.type(v)), g.qubit(v), g.row(v) + 0.5)
        leaf = g.add_vertex(g.type(v), g.qubit(v), g.row(v) + 1.0, p)
        g.set_vdata(leaf, GPF_ROLE_KEY, LEAF)
        g.add_edge((v, stem))
        g.add_edge((stem, leaf))


def _reduce_skeleton(g: BaseGraph[VT, ET], max_steps: int) -> None:
    steps = 0
    while True:
        pair = _find_bialgebra_pair(g)
        if pair is None:
            return
        x, z = pair
        bialgebra(g, x, z)
        spider_simp(g)
        hopf_simp(g)
        g.remove_isolated_vertices()
        steps += 1
        if steps > max_steps:
            raise RuntimeError("to_gpf: skeleton reduction did not terminate within {} "
                               "bialgebra steps".format(max_steps))


def _find_bialgebra_pair(g: BaseGraph[VT, ET]) -> tuple[VT, VT] | None:
    ins, outs = set(g.inputs()), set(g.outputs())
    for x in sorted(g.vertices()):
        if g.type(x) != VertexType.X or is_leaf(g, x) or is_stem(g, x) or _touches(g, x, outs):
            continue
        for z in sorted(g.neighbors(x)):
            if g.type(z) != VertexType.Z or is_leaf(g, z) or is_stem(g, z) or _touches(g, z, ins):
                continue
            if check_bialgebra(g, x, z):
                return x, z
    return None


def _pad_boundaries(g: BaseGraph[VT, ET]) -> None:
    """Every input gets a Z-spider neighbour and every output an X-spider neighbour."""
    for b in list(g.inputs()):
        _pad(g, b, VertexType.Z)
    for b in list(g.outputs()):
        _pad(g, b, VertexType.X)


def _pad(g: BaseGraph[VT, ET], b: VT, ty: VertexType) -> None:
    (n,) = list(g.neighbors(b))
    if g.type(n) == ty and not is_leaf(g, n):
        return
    e = g.edge(b, n)
    et = g.edge_type(e)
    g.remove_edge(e)
    pad = g.add_vertex(ty, g.qubit(b), (g.row(b) + g.row(n)) / 2)
    g.add_edge((b, pad), EdgeType.SIMPLE)
    g.add_edge((pad, n), et)


def _split_multi_boundary_spiders(g: BaseGraph[VT, ET]) -> None:
    """A spider attached to several boundaries becomes one copy per boundary.

    The first copy keeps the interior connections (and the phase); the others
    are chained to it through phase-free spiders of the opposite colour, which
    is the reverse of fusion followed by bialgebra unfusion (thesis Stage 4).
    """
    ins, outs = set(g.inputs()), set(g.outputs())
    for v in list(g.vertices()):
        ty = g.type(v)
        if ty not in (VertexType.Z, VertexType.X):
            continue
        mine = ins if ty == VertexType.Z else outs
        bs = [w for w in g.neighbors(v) if w in mine]
        if len(bs) <= 1:
            continue
        others = [w for w in g.neighbors(v) if w not in bs]
        prev = None
        for k, b in enumerate(bs):
            copy = g.add_vertex(ty, g.qubit(b), g.row(v), g.phase(v) if k == 0 else 0)
            g.add_edge((b, copy), EdgeType.SIMPLE)
            if k == 0:
                for w in others:
                    g.add_edge((copy, w), g.edge_type(g.edge(v, w)))
            else:
                link = g.add_vertex(toggle_vertex(ty), g.qubit(b), g.row(v) + 0.5)
                g.add_edge((prev, link))
                g.add_edge((link, copy))
            prev = copy
        g.remove_vertex(v)


# -- shape check and layout ---------------------------------------------------

def is_gpf(g: BaseGraph[VT, ET]) -> bool:
    """Whether ``g`` has the GPF shape (phases and gadgets allowed).

    Plain edges only; no two adjacent spiders of the same colour; every input
    adjacent to a Z-spider and every output to an X-spider; every Z-spider on
    at most one input and every X-spider on at most one output.
    """
    ins, outs = set(g.inputs()), set(g.outputs())
    for e in g.edges():
        if g.edge_type(e) != EdgeType.SIMPLE:
            return False
        s, t = g.edge_st(e)
        if g.type(s) == g.type(t) and g.type(s) != VertexType.BOUNDARY:
            return False
    for b in ins:
        if any(g.type(n) != VertexType.Z for n in g.neighbors(b)):
            return False
    for b in outs:
        if any(g.type(n) != VertexType.X for n in g.neighbors(b)):
            return False
    for v in g.vertices():
        if g.type(v) == VertexType.Z and sum(1 for n in g.neighbors(v) if n in ins) > 1:
            return False
        if g.type(v) == VertexType.X and sum(1 for n in g.neighbors(v) if n in outs) > 1:
            return False
    return True


def layout_gpf(g: BaseGraph[VT, ET]) -> None:
    """Rows: inputs 0, Z layer 1, X layer 2, outputs 3; gadgets beside their stem."""
    ins, outs = set(g.inputs()), set(g.outputs())
    zq = xq = 0
    for v in sorted(g.vertices()):
        ty = g.type(v)
        if v in ins:
            g.set_row(v, 0)
        elif v in outs:
            g.set_row(v, 3)
        elif is_leaf(g, v):
            continue
        elif ty == VertexType.Z:
            g.set_row(v, 1)
            g.set_qubit(v, zq)
            zq += 1
        elif ty == VertexType.X:
            g.set_row(v, 2)
            g.set_qubit(v, xq)
            xq += 1
    for b in ins | outs:
        (n,) = list(g.neighbors(b))
        g.set_qubit(b, g.qubit(n))
    for v in g.vertices():
        if is_leaf(g, v):
            (stem,) = list(g.neighbors(v))
            g.set_qubit(v, g.qubit(stem) + 0.5)
            g.set_row(v, g.row(stem) + (-0.7 if g.type(stem) == VertexType.Z else 0.7))
