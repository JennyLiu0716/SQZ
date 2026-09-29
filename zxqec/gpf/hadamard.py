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

"""Moving Hadamard edges out of the interior of a diagram.

The generalized parity form is a normal form for the phase-free fragment, which
has no Hadamards. A diagram with Hadamard edges can still be brought into the
form ``H^{S_in} ; D ; H^{S_out}`` -- Hadamards on some boundary wires around a
diagram ``D`` without internal Hadamard edges -- by colour-changing spiders
(``H`` conjugation toggles a spider's colour and the type of all its edges).
Choosing which spiders to flip is a linear system over GF(2) with one equation
per internal edge; it is solvable exactly when every cycle of the diagram
crosses an even number of Hadamard edges. That covers, for instance, a
transversal layer of Hadamards between two phase-free rounds. Success is a
property of the diagram *as drawn*: a diagram with an odd-Hadamard cycle may
still denote a Pauli, which is why :func:`~zxqec.gpf.to_gpf` removes identity
spiders and fuses same-colour neighbours first. What remains after that and
still fails is a map whose tableau needs a ``Y``: a code that is not CSS in any
Hadamard frame, such as the ``[[5,1,3]]`` code.

With ``cz_gadgets=True`` such a diagram is not rejected. A spanning forest is
coloured as before, and every edge that closes an odd cycle is left as a
Hadamard edge and then rewritten as a ``CZ`` in phase gadgets. On the bits of two
Z-spiders ``u`` and ``v`` a Hadamard edge is the phase ``(-1)^(y_u y_v)``, and

    y_u y_v = (y_u + y_v - (y_u xor y_v)) / 2 ,

so it equals a ``pi/2`` phase on each spider and a gadget with leaf ``-pi/2`` on
their parity. An X-spider at either end first gets a phase-free Z-spider on the
edge next to it. The result has no internal Hadamard edges, and the ``CZ`` s sit
in the gadget layer next to any ``T`` gates, where the rest of the pipeline
already handles them.

The boundary Hadamards are recorded as the vertex data ``hadamard=True`` on the
boundary vertex, so that ``D`` itself only has plain edges;
:func:`restore_boundary_hadamards` puts them back as Hadamard edges.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.rewrite_rules.color_change_rule import unsafe_color_change
from pyzx.utils import EdgeType, VertexType, is_pauli

__all__ = ['BOUNDARY_HADAMARD_KEY', 'push_hadamards_to_boundary',
           'restore_boundary_hadamards', 'has_boundary_hadamard']

BOUNDARY_HADAMARD_KEY = 'hadamard'


def has_boundary_hadamard(g: BaseGraph[VT, ET], b: VT) -> bool:
    """Whether boundary ``b`` carries a Hadamard on its wire (see module docstring)."""
    return bool(g.vdata(b, BOUNDARY_HADAMARD_KEY, False))


def push_hadamards_to_boundary(g: BaseGraph[VT, ET], cz_gadgets: bool = False) -> int:
    """Colour-change spiders so that no edge between two spiders is a Hadamard edge (in place).

    Hadamards that cannot be absorbed end up on boundary wires and are recorded
    as ``hadamard=True`` vertex data on the boundary vertex (toggling an
    existing flag), leaving the edge plain. If the diagram has a cycle with an
    odd number of Hadamard edges no colouring works: this raises ``ValueError``,
    or with ``cz_gadgets`` rewrites the edges that close such cycles as ``CZ``
    phase gadgets (see the module docstring). Returns the number of edges so
    rewritten.
    """
    spiders = [v for v in g.vertices() if g.type(v) != VertexType.BOUNDARY]
    for v in spiders:
        if g.type(v) not in (VertexType.Z, VertexType.X):
            raise ValueError("push_hadamards_to_boundary: only Z and X spiders are supported, "
                             "vertex {} has type {}".format(v, g.type(v)))
    flip, odd = _colouring(g, spiders, {}, cz_gadgets)
    _orient(g, flip, set())
    if cz_gadgets:
        # every non-Pauli phase should end on a Z-spider, where the gadget layer can read it; the
        # plain colouring is kept whenever it already does that, so CSS diagrams are unaffected
        pins = {v: int(g.type(v) == VertexType.X) for v in spiders if not is_pauli(g.phase(v))}
        if any(flip[v] != want for v, want in pins.items()):
            flip, odd = _colouring(g, spiders, pins, cz_gadgets)
            _orient(g, flip, set(pins))
    for v, f in flip.items():
        if f:
            unsafe_color_change(g, v)
    for s, t in sorted(odd):
        _hadamard_edge_as_cz(g, s, t)
    # whatever Hadamards remain sit on boundary wires: move them into the flag
    for b in g.vertices():
        if g.type(b) != VertexType.BOUNDARY:
            continue
        for e in g.incident_edges(b):
            if g.edge_type(e) == EdgeType.HADAMARD:
                s, t = g.edge_st(e)
                other = t if s == b else s
                if g.type(other) == VertexType.BOUNDARY and other < b:
                    continue        # boundary-boundary wire: handled from the smaller id
                g.set_edge_type(e, EdgeType.SIMPLE)
                g.set_vdata(b, BOUNDARY_HADAMARD_KEY, not has_boundary_hadamard(g, b))
    return len(odd)


def _colouring(g: BaseGraph[VT, ET], spiders: list[VT], pins: dict[VT, int],
               cz_gadgets: bool) -> tuple[dict[VT, int], set[tuple[VT, VT]]]:
    """Which spiders to colour-change: a breadth-first colouring of each component.

    ``pins`` fixes the choice for some spiders (they are coloured first). An edge
    whose two ends disagree closes a cycle with an odd number of Hadamard edges;
    it raises ``ValueError``, or with ``cz_gadgets`` is returned in the second set.
    """
    flip: dict[VT, int] = dict(pins)
    odd: set[tuple[VT, VT]] = set()
    done: set[VT] = set()
    for root in [v for v in spiders if v in pins] + [v for v in spiders if v not in pins]:
        if root in done:
            continue
        flip.setdefault(root, 0)
        done.add(root)
        queue = deque([root])
        while queue:
            v = queue.popleft()
            for e in g.incident_edges(v):
                s, t = g.edge_st(e)
                w = t if s == v else s
                if g.type(w) == VertexType.BOUNDARY or w == v:
                    continue
                want = flip[v] ^ (1 if g.edge_type(e) == EdgeType.HADAMARD else 0)
                if w in flip:
                    if flip[w] != want:
                        if not cz_gadgets:
                            raise ValueError("diagram has a cycle with an odd number of Hadamard "
                                             "edges; it is not phase-free up to boundary Hadamards")
                        odd.add((min(v, w), max(v, w)))
                else:
                    flip[w] = want
                if w not in done:
                    done.add(w)
                    queue.append(w)
    return flip, odd


def _hadamard_edge_as_cz(g: BaseGraph[VT, ET], s: VT, t: VT) -> None:
    """Replace the Hadamard edge between spiders ``s`` and ``t`` by a ``CZ`` in phase gadgets.

    ``(-1)^(y_u y_v) = exp(i pi/2 (y_u + y_v - (y_u xor y_v)))`` on the bits of two
    Z-spiders, up to a non-zero scalar. An X-spider end gets a phase-free
    Z-spider on the edge next to it.
    """
    e = g.edge(s, t)
    if g.edge_type(e) != EdgeType.HADAMARD:
        raise RuntimeError("push_hadamards_to_boundary: edge {}-{} should be a Hadamard edge "
                           "after recolouring; this is a bug".format(s, t))
    g.remove_edge(e)
    ends = []
    for v, other in ((s, t), (t, s)):
        if g.type(v) == VertexType.Z:
            ends.append(v)
            continue
        z = g.add_vertex(VertexType.Z, (2 * g.qubit(v) + g.qubit(other)) / 3,
                         (2 * g.row(v) + g.row(other)) / 3)
        g.add_edge((v, z))
        ends.append(z)
    u, w = ends
    g.add_to_phase(u, Fraction(1, 2))
    g.add_to_phase(w, Fraction(1, 2))
    q, r = (g.qubit(u) + g.qubit(w)) / 2, (g.row(u) + g.row(w)) / 2
    hub = g.add_vertex(VertexType.X, q, r + 0.3)
    leaf = g.add_vertex(VertexType.Z, q, r + 0.6, Fraction(-1, 2))
    g.add_edge((hub, u))
    g.add_edge((hub, w))
    g.add_edge((hub, leaf))


def _orient(g: BaseGraph[VT, ET], flip: dict[VT, int], pinned: set[VT]) -> None:
    """Choose each component's overall parity so that boundary Hadamards land on inputs.

    Both parities of a connected component solve the colouring equally well --
    flipping every spider of a component toggles only which wires keep a
    Hadamard. We prefer the parity that leaves fewer Hadamards on outputs, so
    that the reduced diagram reads "Hadamard first, then the outcome-dependent
    Paulis": that is the order in which a decoder applies its corrections.
    Components containing a ``pinned`` spider keep their parity.
    """
    from collections import deque

    seen: set[VT] = set()
    for root in list(flip):
        if root in seen:
            continue
        component, queue = [], deque([root])
        seen.add(root)
        while queue:
            v = queue.popleft()
            component.append(v)
            for w in g.neighbors(v):
                if w in flip and w not in seen:
                    seen.add(w)
                    queue.append(w)
        outs = set(g.outputs())
        cost = 0                              # Hadamards left on outputs with the current parity
        for v in component:
            for e in g.incident_edges(v):
                a, b = g.edge_st(e)
                other = b if a == v else a
                if other in outs:
                    had = 1 if g.edge_type(e) == EdgeType.HADAMARD else 0
                    cost += had ^ flip[v]
        flipped_cost = sum(1 for v in component for e in g.incident_edges(v)
                           for a, b in [g.edge_st(e)]
                           if (b if a == v else a) in outs) - cost
        if flipped_cost < cost and not pinned.intersection(component):
            for v in component:
                flip[v] ^= 1


def restore_boundary_hadamards(g: BaseGraph[VT, ET]) -> BaseGraph[VT, ET]:
    """A copy of ``g`` in which every flagged boundary wire is a Hadamard edge again."""
    h = g.copy()
    for b in h.vertices():
        if h.type(b) == VertexType.BOUNDARY and has_boundary_hadamard(h, b):
            for e in h.incident_edges(b):
                h.set_edge_type(e, EdgeType.HADAMARD if h.edge_type(e) == EdgeType.SIMPLE
                                else EdgeType.SIMPLE)
            h.set_vdata(b, BOUNDARY_HADAMARD_KEY, False)
    return h
