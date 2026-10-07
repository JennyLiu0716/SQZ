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

"""Rewrite rules for diagrams whose phases hold measurement results, as in a derivation by hand.

PyZX's rules (``pyzx.rewrite_rules``) can be applied one at a time to chosen
spiders. Fusion and identity removal work with Boolean phases as they are:
``c[0] + c[0]`` is 0. The copy rule does not: PyZX copies a spider with one leg
only when its phase is 0 or pi. :func:`copy_leaf` is that rule for a phase such
as ``c[0]`` (pi times a measurement result), which holds because it holds for
each value of the result.

Three rewrites of a whole diagram, made of these rules, are the steps of a
derivation such as Eq. (6.42) of L. Yeh's thesis (gate teleportation); they are
level-up rewrites by the same names (:mod:`zxqec.qprogram.rewrites`):

* :func:`thesis_notation` -- PyZX's diagram of a circuit in the notation of the
  thesis: identity spiders removed (a swap becomes crossing wires), a spider with
  one leg on a Hadamard edge drawn in the other colour (``|0>`` then ``H`` is
  ``|+>``), each measurement result copied off its wire (a measurement effect, and
  the state it leaves), and a closed piece left by a reset made a number;
* :func:`copy_and_fuse` -- states (``|0>``, ``|1>``, ``|+>``, ``|->``) copied through
  spiders of the other colour, spiders with one leg fused into their neighbour of
  the same colour, and identities on a straight piece of wire removed; an identity
  where a wire turns is kept, to show the turn;
* :func:`remove_identities` -- every identity spider removed, and neighbouring
  spiders of the same colour whose phases cancel (``c[0]`` and ``c[0]``) fused.

Each keeps the linear map, the scalar included (a closed piece becomes part of
the scalar, which may depend on the results).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from pyzx.graph.base import BaseGraph
from pyzx.symbolic import Poly
from pyzx.utils import vertex_is_zx

__all__ = ['copy_leaf', 'thesis_notation', 'copy_and_fuse', 'remove_identities']


def _is_pauli(phase: Any) -> bool:
    """0, pi, or an integer constant plus a parity of Boolean variables (units of pi)."""
    if not isinstance(phase, Poly):
        return Fraction(phase).denominator == 1
    for coeff, term in phase.terms:
        if Fraction(coeff).denominator != 1:
            return False
        if term.vars and (len(term.vars) != 1 or term.vars[0][1] != 1 or not term.vars[0][0].is_bool):
            return False
    return True


def copy_leaf(g: BaseGraph, v: Any) -> dict[Any, Any]:
    """The copy rule for a spider with one leg whose phase is 0, pi or a measurement result times pi.

    ``v`` is copied through its only neighbour ``w``, a spider of the other colour
    joined by a plain edge (or of the same colour joined by a Hadamard edge), as in
    PyZX's :func:`~pyzx.rewrite_rules.copy_rule.copy`: ``v`` and ``w`` are removed,
    and every other neighbour ``n`` of ``w`` gets a new spider with ``v``'s phase on
    the edge it had to ``w`` (drawn on ``n``'s wire just before ``n`` when ``n`` is along a
    wire, as a new state is). Returns ``{n: new spider}``, so that each copy can be
    fused into its neighbour. PyZX's own rule accepts only the phases 0 and pi;
    the rule holds for each value of a result, so for ``c[0]`` too, provided ``w``
    has phase 0 (else the number it leaves behind depends on the result).
    Raises ``ValueError`` if the rule does not apply.
    """
    from pyzx.rewrite_rules.copy_rule import check_copy_zx, unsafe_copy

    if v not in g.vertices() or g.vertex_degree(v) != 1:
        raise ValueError("copy_leaf: {} is not a spider with one leg".format(v))
    (w,) = list(g.neighbors(v))
    if not (vertex_is_zx(g.type(v)) and vertex_is_zx(g.type(w))) or check_copy_zx(g, v, w) is None:
        raise ValueError("copy_leaf: {} cannot be copied through {} (it must be of the other colour, "
                         "or of the same colour across a Hadamard edge)".format(v, w))
    if not _is_pauli(g.phase(v)):
        raise ValueError("copy_leaf: the phase {} of {} is not 0, pi or a parity of results".format(g.phase(v), v))
    if isinstance(g.phase(v), Poly) and g.phase(w) != 0:
        raise ValueError("copy_leaf: {} has phase {}; a result can only be copied through a phase-0 spider"
                         .format(w, g.phase(w)))
    others = [n for n in g.neighbors(w) if n != v]
    if len(others) != len(list(g.incident_edges(w))) - 1:
        raise ValueError("copy_leaf: {} has two edges to one neighbour".format(w))
    row = g.row(w)
    before = set(g.vertices())
    unsafe_copy(g, v)
    new = {next(iter(g.neighbors(u))): u for u in set(g.vertices()) - before}
    for n, u in new.items():           # a copy along a wire: drawn on that wire, just before its neighbour
        if abs(g.row(n) - row) > 1e-9:
            g.set_qubit(u, g.qubit(n))
            g.set_row(u, g.row(n) - 0.5 if g.row(n) > row else g.row(n) + 0.5)
    return new


# -- three rewrites of a whole diagram -------------------------------------------------------------

def _is_identity(g: BaseGraph, v: Any) -> bool:
    return (vertex_is_zx(g.type(v)) and g.phase(v) == 0 and g.vertex_degree(v) == 2
            and v not in g.neighbors(v))


def _leaves(g: BaseGraph) -> list[tuple[Any, Any]]:
    """(spider with one leg, its neighbour) for every Z- or X-spider with one leg."""
    out = []
    for v in list(g.vertices()):
        if v in g.vertices() and vertex_is_zx(g.type(v)) and g.vertex_degree(v) == 1:
            out.append((v, next(iter(g.neighbors(v)))))
    return out


def _fuse_leaf(g: BaseGraph, leaf: Any, w: Any) -> bool:
    """Fuse ``leaf`` into ``w`` if they are spiders of one colour joined by a plain edge; ``w`` stays."""
    from pyzx.rewrite_rules.fuse_rule import fuse
    from pyzx.utils import EdgeType
    if (vertex_is_zx(g.type(w)) and g.type(w) == g.type(leaf)
            and g.edge_type(g.edge(leaf, w)) == EdgeType.SIMPLE):
        return fuse(g, w, leaf)
    return False


def thesis_notation(g: BaseGraph) -> BaseGraph:
    """PyZX's diagram of a circuit in the notation of L. Yeh's thesis (see the module docstring); ``g`` in place.

    Identity spiders are removed; a spider with one leg on a Hadamard edge is
    colour changed; every measurement result (a spider PyZX tags
    ``outcome_type='measurement'``) is copied through the spider that measured it,
    if that has phase 0, leaving the measurement effect on the wire before and the
    state it leaves on the wire after; two spiders of one colour with one leg each,
    joined only to each other (what a reset's thrown-away result becomes), are fused
    and moved into the scalar.
    """
    from pyzx.rewrite_rules.color_change_rule import color_change
    from pyzx.rewrite_rules.remove_id_rule import remove_id
    from pyzx.utils import EdgeType, VertexType
    changed = True
    while changed:
        changed = False
        for v in list(g.vertices()):
            if v in g.vertices() and _is_identity(g, v):
                changed |= remove_id(g, v)
        for v, w in _leaves(g):
            if g.type(w) != VertexType.BOUNDARY and g.edge_type(g.edge(v, w)) == EdgeType.HADAMARD:
                changed |= color_change(g, v)
        for v, w in _leaves(g):
            if v in g.vertices() and g.vdata(v, 'outcome_type', None) == 'measurement' and g.phase(w) == 0:
                try:
                    copy_leaf(g, v)
                    changed = True
                except ValueError:
                    pass
        for v, w in _leaves(g):
            if v in g.vertices() and w in g.vertices() and g.vertex_degree(w) == 1:
                changed |= _fuse_leaf(g, v, w)
        g.remove_isolated_vertices()
    return g


def copy_and_fuse(g: BaseGraph) -> BaseGraph:
    """States copied, spiders with one leg fused, identities on straight wires removed (module docstring); in place.

    A state is a spider with one leg and phase 0 or pi; it is copied (:func:`copy_leaf`)
    through a neighbour of the other colour whose phase is not a measurement result.
    A spider with one leg of any phase is fused into its neighbour of the same colour.
    An identity spider is removed when both its neighbours sit on its own wire (the
    same ``qubit`` coordinate); one where the wire turns stays. Repeated until none applies.
    """
    from pyzx.rewrite_rules.remove_id_rule import remove_id
    from pyzx.utils import VertexType
    changed = True
    while changed:
        changed = False
        for v, w in _leaves(g):
            if (v in g.vertices() and w in g.vertices() and not isinstance(g.phase(v), Poly)
                    and g.type(w) != VertexType.BOUNDARY and not isinstance(g.phase(w), Poly)):
                try:
                    copy_leaf(g, v)
                    changed = True
                except ValueError:
                    pass
        for v, w in _leaves(g):
            if v in g.vertices() and w in g.vertices():
                changed |= _fuse_leaf(g, v, w)
        for v in list(g.vertices()):
            if v in g.vertices() and _is_identity(g, v) and all(
                    abs(g.qubit(n) - g.qubit(v)) < 1e-9 for n in g.neighbors(v)):
                changed |= remove_id(g, v)
        g.remove_isolated_vertices()
    return g


def remove_identities(g: BaseGraph) -> BaseGraph:
    """Every identity spider removed, and neighbours of one colour whose phases cancel fused; ``g`` in place."""
    from pyzx.rewrite_rules.fuse_rule import fuse
    from pyzx.rewrite_rules.remove_id_rule import remove_id
    from pyzx.utils import EdgeType
    changed = True
    while changed:
        changed = False
        for v in list(g.vertices()):
            if v in g.vertices() and _is_identity(g, v):
                changed |= remove_id(g, v)
        for e in list(g.edges()):
            v, w = g.edge_st(e)
            if (v in g.vertices() and w in g.vertices() and v != w and vertex_is_zx(g.type(v))
                    and g.type(v) == g.type(w) and g.edge_type(e) == EdgeType.SIMPLE
                    and _cancel(g.phase(v), g.phase(w))):
                changed |= fuse(g, v, w)
        g.remove_isolated_vertices()
    return g


def _cancel(a: Any, b: Any) -> bool:
    total = a + b
    if isinstance(total, Poly):
        return total == 0
    return Fraction(total) % 2 == 0
