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
still fails is a map whose tableau needs a ``Y``.

The boundary Hadamards are recorded as the vertex data ``hadamard=True`` on the
boundary vertex, so that ``D`` itself only has plain edges;
:func:`restore_boundary_hadamards` puts them back as Hadamard edges.
"""

from __future__ import annotations

from collections import deque

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.rewrite_rules.color_change_rule import unsafe_color_change
from pyzx.utils import EdgeType, VertexType

__all__ = ['BOUNDARY_HADAMARD_KEY', 'push_hadamards_to_boundary',
           'restore_boundary_hadamards', 'has_boundary_hadamard']

BOUNDARY_HADAMARD_KEY = 'hadamard'


def has_boundary_hadamard(g: BaseGraph[VT, ET], b: VT) -> bool:
    """Whether boundary ``b`` carries a Hadamard on its wire (see module docstring)."""
    return bool(g.vdata(b, BOUNDARY_HADAMARD_KEY, False))


def push_hadamards_to_boundary(g: BaseGraph[VT, ET]) -> None:
    """Colour-change spiders so that no edge between two spiders is a Hadamard edge (in place).

    Hadamards that cannot be absorbed end up on boundary wires and are recorded
    as ``hadamard=True`` vertex data on the boundary vertex (toggling an
    existing flag), leaving the edge plain. Raises ``ValueError`` if the
    diagram has a cycle with an odd number of Hadamard edges, in which case no
    colouring works.
    """
    spiders = [v for v in g.vertices() if g.type(v) != VertexType.BOUNDARY]
    for v in spiders:
        if g.type(v) not in (VertexType.Z, VertexType.X):
            raise ValueError("push_hadamards_to_boundary: only Z and X spiders are supported, "
                             "vertex {} has type {}".format(v, g.type(v)))
    flip: dict[VT, int] = {}
    for root in spiders:
        if root in flip:
            continue
        flip[root] = 0
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
                        raise ValueError("diagram has a cycle with an odd number of Hadamard "
                                         "edges; it is not phase-free up to boundary Hadamards")
                else:
                    flip[w] = want
                    queue.append(w)
    _orient(g, flip)
    for v, f in flip.items():
        if f:
            unsafe_color_change(g, v)
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


def _orient(g: BaseGraph[VT, ET], flip: dict[VT, int]) -> None:
    """Choose each component's overall parity so that boundary Hadamards land on inputs.

    Both parities of a connected component solve the colouring equally well --
    flipping every spider of a component toggles only which wires keep a
    Hadamard. We prefer the parity that leaves fewer Hadamards on outputs, so
    that the reduced diagram reads "Hadamard first, then the outcome-dependent
    Paulis": that is the order in which a decoder applies its corrections.
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
        if flipped_cost < cost:
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
