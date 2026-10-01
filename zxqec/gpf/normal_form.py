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
  the opposite colour. Later rewrites add edges to that stem, so a gadget's
  connector is a *hub* of any degree, not necessarily degree two; nothing in
  the pipeline depends on its degree. Gadgets take no part in the skeleton reduction and are
  treated as fixed external targets by the matrix reduction.

``isolate_paulis=True`` isolates *every* non-zero phase, which reproduces the
thesis's Algorithm 2 (a phase-free skeleton plus gadgets).

Hadamard edges are pushed to the boundary wires beforehand (see
:mod:`zxqec.gpf.hadamard`). Those that cannot be absorbed by recolouring, because
they close a cycle with an odd number of Hadamards, become ``CZ`` s written as
``pi/2`` phase gadgets, so the input may be any diagram of Z- and X-spiders.
"""

from __future__ import annotations

from typing import Any

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.rewrite_rules import bialgebra, check_bialgebra
from pyzx.simplify import hopf_simp, id_simp, spider_simp
from pyzx.utils import EdgeType, VertexType, is_pauli, toggle_vertex

from .hadamard import push_hadamards_to_boundary

__all__ = ['GPF_ROLE_KEY', 'LEAF', 'to_gpf', 'is_leaf', 'is_stem', 'is_gpf', 'layout_gpf', 'skeleton_count']

GPF_ROLE_KEY = 'gpf'
LEAF = 'leaf'


# -- roles ------------------------------------------------------------------

def is_leaf(g: BaseGraph[VT, ET], v: VT) -> bool:
    """A phase gadget's leaf: the degree-one spider that carries an isolated phase."""
    return g.vdata(v, GPF_ROLE_KEY, None) == LEAF


def is_stem(g: BaseGraph[VT, ET], v: VT) -> bool:
    """A spider adjacent to a leaf: the hub a gadget hangs from, of any degree.

    Unfusing creates a degree-two connector, but the bialgebra step and the row
    reduction of the opposite layer both add edges to it, after which the gadget
    is the usual phase gadget on the hub's whole neighbourhood.
    """
    return g.type(v) != VertexType.BOUNDARY and any(is_leaf(g, w) for w in g.neighbors(v))


def _touches(g: BaseGraph[VT, ET], v: VT, boundaries: set[VT]) -> bool:
    return any(w in boundaries for w in g.neighbors(v))


# -- the algorithm ----------------------------------------------------------

def to_gpf(g: BaseGraph[VT, ET], isolate_paulis: bool = False,
           max_steps: int | None = None, cz_gadgets: bool = True) -> BaseGraph[VT, ET]:
    """Return a copy of ``g`` in generalized parity form with phases.

    Steps (Algorithm 2 of the thesis, generalised):

    1. push Hadamard edges to the boundary wires, writing those that close an
       odd-Hadamard cycle as ``CZ`` phase gadgets (``cz_gadgets=False`` raises
       ``ValueError`` instead, the behaviour for CSS-only diagrams); fuse spiders;
    2. isolate phases into gadgets: all non-zero phases if ``isolate_paulis``,
       otherwise only non-Pauli ones;
    3. skeleton reduction: repeatedly apply the (Boolean) bialgebra rule to an
       X-spider not touching an output and a Z-neighbour not touching an input,
       neither being part of a gadget, with fusion and Hopf clean-up;
    4. pad every input with a Z-spider and every output with an X-spider where
       needed, and give every spider at most one boundary.

    ``max_steps`` bounds the number of bialgebra applications. By default it is
    the bound of the termination lemma (appendix A of the paper, after Yeh et
    al., Lemma B.1): the number of Z-spiders that are neither leaves nor stems
    and touch no input when step 3 starts (:func:`skeleton_count`), which every
    application lowers. Exceeding it raises ``RuntimeError``, which with the
    default bound means a bug.
    """
    h = g.copy()
    for v in h.vertices():
        if h.type(v) not in (VertexType.Z, VertexType.X, VertexType.BOUNDARY):
            raise ValueError("to_gpf: vertex {} has unsupported type {}".format(v, h.type(v)))
    _presimplify(h)                       # identities and same-colour pairs first: they can
    push_hadamards_to_boundary(h, cz_gadgets)   # only shrink the cycle space Algorithm 1 solves
    spider_simp(h)
    h.remove_isolated_vertices()

    _isolate_phases(h, isolate_paulis)
    _reduce_skeleton(h, max_steps if max_steps is not None else skeleton_count(h))
    _pad_boundaries(h)
    _split_multi_boundary_spiders(h)
    layout_gpf(h)
    if not is_gpf(h):
        raise RuntimeError("to_gpf: the result is not in generalized parity form; "
                           "this is a bug, please report the input diagram")
    return h


def _presimplify(g: BaseGraph[VT, ET]) -> None:
    """Remove identity spiders and fuse same-colour neighbours, to a fixpoint.

    Run before :func:`push_hadamards_to_boundary`, because both are equalities
    that can only shrink the cycle space: a diagram whose only odd-Hadamard
    cycle runs through an identity spider is accepted after this, although the
    diagram as drawn would be rejected.
    """
    while True:
        changed = bool(id_simp(g))
        changed |= bool(spider_simp(g))
        if not changed:
            return


def _isolate_phases(g: BaseGraph[VT, ET], isolate_paulis: bool) -> None:
    for v in list(g.vertices()):
        if g.type(v) == VertexType.BOUNDARY:
            continue
        p = g.phase(v)
        if p == 0 or (is_pauli(p) and not isolate_paulis):
            continue
        if g.vertex_degree(v) == 0:
            continue                          # a scalar; removed by the callers' clean-up
        if g.vertex_degree(v) == 1 and not _on_boundary(g, v):
            (hub,) = list(g.neighbors(v))
            if not _on_boundary(g, hub) and not any(is_leaf(g, w) for w in g.neighbors(hub)):
                g.set_vdata(v, GPF_ROLE_KEY, LEAF)  # already a leaf hanging off another spider
                continue
            # otherwise it is not a phase gadget: next to a boundary the leaf's bit is summed
            # over (a map on that wire, not a phase), and with a second leaf the hub fixes only
            # the parity of the two leaf bits. The spider stays a bit and gets its own gadget.
        # a degree-one spider on a boundary wire is a state or effect, not a gadget leaf: its
        # phase is unfused onto a gadget below, so that the wire's spider stays phase-free
        keep = 0 if isolate_paulis else _pauli_part(p)
        g.set_phase(v, keep)
        p = p - keep if keep else p
        stem = g.add_vertex(toggle_vertex(g.type(v)), g.qubit(v), g.row(v) + 0.5)
        leaf = g.add_vertex(g.type(v), g.qubit(v), g.row(v) + 1.0, p)
        g.set_vdata(leaf, GPF_ROLE_KEY, LEAF)
        g.add_edge((v, stem))
        g.add_edge((stem, leaf))


def _pauli_part(p: Any) -> Any:
    """The parity of outcomes in a phase ``a + c_1 + ... + c_m`` (units of pi), or 0.

    A phase such as ``pi/2 + c`` -- a ``CZ`` end on a spider that also carries an
    outcome -- keeps its Pauli part ``c`` on the spider and isolates only ``pi/2``.
    Phases of any other shape are isolated whole.
    """
    if not hasattr(p, 'terms'):
        return 0
    from .gadgets import _split_phase
    try:
        const, _names = _split_phase(p)
    except ValueError:
        return 0
    return p - const if const else 0


def _on_boundary(g: BaseGraph[VT, ET], v: VT) -> bool:
    """Whether ``v`` is adjacent to an input or output."""
    return any(g.type(w) == VertexType.BOUNDARY for w in g.neighbors(v))


def skeleton_count(g: BaseGraph[VT, ET]) -> int:
    """The Z-spiders that are neither leaves nor stems and touch no input.

    Every step of the skeleton reduction removes one of them (its Z-spider) and
    creates none: a new Z-spider sits on another leg of the X-spider, which is a
    Z-spider it fuses into or an input. So the reduction takes at most this many
    steps.
    """
    ins = set(g.inputs())
    return sum(1 for v in g.vertices() if g.type(v) == VertexType.Z and not is_leaf(g, v)
               and not is_stem(g, v) and not _touches(g, v, ins))


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
            raise RuntimeError("to_gpf: the skeleton reduction took more than {} bialgebra steps, which "
                               "the termination bound rules out; this is a bug, please report the input "
                               "diagram".format(max_steps))


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
    at most one input and every X-spider on at most one output; and every
    non-Pauli phase on a gadget leaf, so that the rows of the split matrices
    carry Pauli phases only.
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
        if g.type(v) != VertexType.BOUNDARY and not is_leaf(g, v) and not is_pauli(g.phase(v)):
            return False                      # (G3)/(G4): non-Pauli phases live on leaves
        if is_leaf(g, v):
            (hub,) = list(g.neighbors(v))
            if _on_boundary(g, hub) or sum(1 for w in g.neighbors(hub) if is_leaf(g, w)) > 1:
                return False                  # a phase gadget: one leaf, and a hub away from the boundary
    return True


def layout_gpf(g: BaseGraph[VT, ET]) -> None:
    """Lay out a GPF diagram for PyZX's drawing functions (``zx.draw``, ``zx.draw_matplotlib``).

    Qubits (top to bottom): the X-spider of output ``j`` on line ``j``, then the
    other X-spiders and the gadget hubs one per line; the Z-spiders spread evenly
    over the same height, those on inputs first in input order and the others in
    the order of the mean line of the X-spiders they are wired to, which keeps
    edges short. Rows (left to right): inputs 0, Z-layer 1, X-layer ``1 + gap``,
    outputs ``2 + gap``, with ``gap = max(2, height / 3)`` so that the edges
    between the layers stay readable in a tall diagram (PyZX draws rows and
    lines on the same scale). A boundary is on the line of its spider, and a
    leaf beside its hub.
    """
    ins, outs = list(g.inputs()), list(g.outputs())
    bnd = set(ins) | set(outs)
    leaves = [v for v in g.vertices() if is_leaf(g, v)]
    hubs = {next(iter(g.neighbors(v))) for v in leaves}
    line: dict[VT, float] = {}
    for j, b in enumerate(outs):
        for w in g.neighbors(b):
            if w not in bnd and g.type(w) == VertexType.X and w not in line:
                line[w] = j
    rest = [v for v in sorted(g.vertices())
            if v not in bnd and v not in line and not is_leaf(g, v) and g.type(v) != VertexType.Z]
    rest.sort(key=lambda v: (v in hubs, v))
    for i, v in enumerate(rest):
        line[v] = len(outs) + 0.5 + i
    on_input = [next(iter(g.neighbors(b))) for b in ins]
    zs = [v for v in sorted(g.vertices()) if g.type(v) == VertexType.Z and v not in bnd and not is_leaf(g, v)]

    def mean_line(v: VT) -> float:
        ls = [line[w] for w in g.neighbors(v) if w in line]
        return sum(ls) / len(ls) if ls else 0.0

    first = [v for v in on_input if v in zs]
    zs = list(dict.fromkeys(first)) + sorted((v for v in zs if v not in first), key=mean_line)
    height = max(len(outs) + len(rest) - 1, len(zs) - 1, 0)
    x_row = 1 + max(2.0, height / 3)
    for v in line:
        g.set_row(v, x_row)
    for i, v in enumerate(zs):
        g.set_row(v, 1)
        g.set_qubit(v, i * height / (len(zs) - 1) if len(zs) > 1 else mean_line(v))
    for v, q in line.items():
        g.set_qubit(v, q)
    for b in ins + outs:
        g.set_row(b, 0 if b in ins else x_row + 1)
        (n,) = list(g.neighbors(b))
        g.set_qubit(b, g.qubit(n))
    for v in leaves:
        (stem,) = list(g.neighbors(v))
        g.set_qubit(v, g.qubit(stem))
        g.set_row(v, g.row(stem) + (-0.7 if g.type(stem) == VertexType.Z else 0.7))
