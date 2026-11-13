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
derivation such as gate teleportation; they are
level-up rewrites by the same names (:mod:`zxqec.qprogram.rewrites`):

* :func:`standard_notation` -- PyZX's diagram of a circuit in the notation used
  in hand derivations: identity spiders removed (a swap becomes crossing wires), a spider with
  one leg on a Hadamard edge drawn in the other colour (``|0>`` then ``H`` is
  ``|+>``), each measurement result copied off its wire (a measurement effect, and
  the state it leaves), and a closed piece left by a reset made a number;
* :func:`copy_and_fuse` -- states (``|0>``, ``|1>``, ``|+>``, ``|->``) copied through
  spiders of the other colour, spiders with one leg fused into their neighbour of
  the same colour, and identities on a straight piece of wire removed; an identity
  where a wire turns is kept, to show the turn; with ``results=True`` (``'copy
  results'``) also states such as ``|a>``, whose phase ``a pi`` is a Boolean variable;
* :func:`remove_identities` -- every identity spider removed, and neighbouring
  spiders of the same colour whose phases cancel (``c[0]`` and ``c[0]``) fused.

and two more for derivations with phase gadgets, such as the transversal
CCZ of the ``[[8,3,2]]`` code, level-up rewrites ``'phase
gadgets'`` and ``'spider nest'``:

* :func:`phase_gadgets` -- the diagram reduced to GPF with its non-Pauli phases on
  phase gadgets (a ``T`` after a ladder of CNOTs and before its inverse becomes a
  gadget on the parity the ladder computes), written back along the wires: each
  wire one Z-spider, a gadget on one wire a phase on it, the other gadgets beside
  the wires they touch; with ``simplify=True`` the gadget layer is simplified
  first (:func:`~zxqec.gpf.simplify_gadgets`: the spider-nest identities);
* :func:`unfuse_outcomes` -- each phase that depends on measurement outcomes split
  off into a spider of its own (level-up rewrite ``'unfuse outcomes'``), the first
  step of correction finding: the outcome phases ``j pi`` made visible;
* :func:`x_z_normal_form` -- the normal form with the colours swapped (level-up
  rewrite ``'x-z gpf'``): an X-spider for each input and a Z-spider for each
  output, as the adjoint of an encoder's Z-X normal form is, e.g. the decoder in
  magic-state distillation.

Each keeps the linear map, the scalar included (a closed piece becomes part of
the scalar, which may depend on the results).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from pyzx.graph.base import BaseGraph
from pyzx.symbolic import Poly
from pyzx.utils import vertex_is_zx

__all__ = ['copy_leaf', 'standard_notation', 'copy_and_fuse', 'remove_identities', 'phase_gadgets',
           'x_z_normal_form', 'unfuse_outcomes']


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


def standard_notation(g: BaseGraph) -> BaseGraph:
    """PyZX's diagram of a circuit in the notation used in hand derivations (see the module docstring); ``g`` in place.

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
            if g.edge_type(g.edge(v, w)) == EdgeType.HADAMARD:
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


def copy_and_fuse(g: BaseGraph, results: bool = False) -> BaseGraph:
    """States copied, spiders with one leg fused, identities on straight wires removed (module docstring); in place.

    A state is a spider with one leg and phase 0 or pi; it is copied (:func:`copy_leaf`)
    through a neighbour of the other colour whose phase is not a measurement result.
    With ``results=True`` (level-up rewrite ``'copy results'``) a state whose phase is
    a parity of Boolean variables, such as ``a pi`` for a basis state ``|a>`` or a
    measurement result, is copied too, through a neighbour of phase 0. A spider with
    one leg of any phase is fused into its neighbour of the same colour; a spider left
    with no leg becomes part of the scalar. An identity spider is removed when both its
    neighbours sit on its own wire (the same ``qubit`` coordinate); one where the wire
    turns stays. Repeated until none applies.
    """
    from pyzx.rewrite_rules.remove_id_rule import remove_id
    from pyzx.utils import VertexType
    changed = True
    while changed:
        changed = False
        for v, w in _leaves(g):
            if (v in g.vertices() and w in g.vertices() and (results or not isinstance(g.phase(v), Poly))
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


def phase_gadgets(g: BaseGraph, simplify: bool = False) -> tuple[BaseGraph, list[Any]]:
    """``g`` with its non-Pauli phases on phase gadgets, drawn along its wires; returns ``(diagram, detectors)``.

    ``g`` is reduced to GPF (:func:`~zxqec.gpf.reduce_gpf`; the gadget layer is
    simplified as a whole only with ``simplify``, which applies the spider-nest
    identities) and written along the wires of ``g``: an output's X-spider joined
    to one Z-spider only is an identity and is removed, so the Z-spider of each
    input runs from the input to its output; a gadget on one Z-spider becomes a
    phase on it. Each Z-spider is drawn on the wire of its input (or output), one
    with neither -- a variable inside the diagram -- below the wires, all in one
    column; each gadget just below the lowest wire it touches, its phase beside it,
    the gadgets spread over two rows each, starting at the first row of ``g``. The
    detectors are those of the reduction.
    """
    from pyzx.rewrite_rules.fuse_rule import fuse
    from pyzx.rewrite_rules.remove_id_rule import remove_id
    from pyzx.utils import EdgeType, VertexType

    from ..spacetime.subdiagram import get_logical_qubit
    from .normal_form import GPF_ROLE_KEY
    from .reduce import reduce_gpf

    wire = {}                              # each boundary's wire: where its neighbour sits
    for b in list(g.inputs()) + list(g.outputs()):
        (n,) = list(g.neighbors(b))
        wire[(b in g.inputs(), get_logical_qubit(g, b))] = (
            g.qubit(n) if g.type(n) != VertexType.BOUNDARY else g.qubit(b))
    rows = [g.row(v) for v in g.vertices() if g.type(v) != VertexType.BOUNDARY] or [0]
    first = min(rows)
    for v in g.vertices():                 # markings of an earlier reduction no longer hold
        if g.vdata(v, GPF_ROLE_KEY, None) is not None:
            g.set_vdata(v, GPF_ROLE_KEY, None)
    res = reduce_gpf(g, gadgets=simplify)
    h = res.to_graph()
    for v in h.vertices():
        if h.vdata(v, GPF_ROLE_KEY, None) is not None:
            h.set_vdata(v, GPF_ROLE_KEY, None)
    ins, outs = list(h.inputs()), list(h.outputs())
    for o in outs:
        (x,) = list(h.neighbors(o))
        if h.type(x) == VertexType.X and _is_identity(h, x):
            remove_id(h, x)
    for leaf in [v for v in h.vertices() if h.type(v) == VertexType.Z and h.vertex_degree(v) == 1]:
        (hub,) = list(h.neighbors(leaf))
        if h.type(hub) == VertexType.X and _is_identity(h, hub):
            (z,) = [w for w in h.neighbors(hub) if w != leaf]
            if h.type(z) == VertexType.Z and h.edge_type(h.edge(hub, z)) == EdgeType.SIMPLE:
                remove_id(h, hub)
                fuse(h, z, leaf)
    on: dict[Any, float] = {}
    for is_input, boundaries in ((True, ins), (False, outs)):
        for b in boundaries:
            h.set_qubit(b, wire.get((is_input, get_logical_qubit(h, b)), h.qubit(b)))
            (z,) = list(h.neighbors(b))
            if z not in on and h.type(z) != VertexType.BOUNDARY:
                on[z] = h.qubit(b)
    hubs = [v for v in h.vertices() if h.type(v) == VertexType.X
            and any(h.type(w) == VertexType.Z and h.vertex_degree(w) == 1 for w in h.neighbors(v))]
    inside = [v for v in h.vertices() if h.type(v) == VertexType.Z and h.vertex_degree(v) > 1 and v not in on]
    lowest = max(list(on.values()) + [0])
    for k, z in enumerate(inside):
        on[z] = lowest + 1 + k
    width = max(4, 2 * len(hubs) + 2)
    mid = first + width / 2
    for z, q in on.items():
        h.set_qubit(z, q)
        h.set_row(z, mid)
    hubs.sort(key=lambda v: sorted(on.get(w, lowest + 1) for w in h.neighbors(v)))
    below: dict[float, int] = {}          # gadgets already drawn below each wire
    for k, hub in enumerate(hubs):
        q = max([on[w] for w in h.neighbors(hub) if w in on] + [0])
        j = below.get(q, 0)
        below[q] = j + 1
        r = first + 1 + k * (width - 2) / max(1, len(hubs) - 1)
        if abs(r - mid) < 0.5:             # the middle row is the wires' spiders'
            r = mid + 0.75
        h.set_row(hub, r)
        h.set_qubit(hub, q + 0.4 + 0.25 * (j % 3))
        for leaf in [w for w in h.neighbors(hub) if h.vertex_degree(w) == 1]:
            h.set_row(leaf, r - 0.6)
            h.set_qubit(leaf, h.qubit(hub))
    for b in ins:
        h.set_row(b, first - 1)
    for b in outs:
        h.set_row(b, first + width + 1)
    return h, list(res.detectors)


def x_z_normal_form(g: BaseGraph, checks: bool = False) -> tuple[BaseGraph, list[Any]]:
    """``g`` in X-Z normal form: the Z-X normal form (:func:`~zxqec.gpf.reduce_gpf`) with the colours swapped.

    Swapping the colour of every spider, edges untouched, gives ``g`` with a
    Hadamard on each boundary wire; that is reduced to the Z-X normal form, and the
    colours are swapped back, which takes the Hadamards off again. The result: an
    X-spider on each input, a Z-spider on each output, phase gadgets with hubs of
    the other colour. With ``checks``, the checks are written one per spider
    (:func:`~zxqec.gpf.one_spider_per_outcome`, before the colours are swapped
    back). Returns ``(diagram, detectors)``.
    """
    from pyzx.utils import toggle_vertex

    from .matrix import one_spider_per_outcome
    from .normal_form import GPF_ROLE_KEY
    from .reduce import reduce_gpf

    def swap(h: BaseGraph) -> None:
        for v in h.vertices():
            if vertex_is_zx(h.type(v)):
                h.set_type(v, toggle_vertex(h.type(v)))
            if h.vdata(v, GPF_ROLE_KEY, None) is not None:
                h.set_vdata(v, GPF_ROLE_KEY, None)
    h = g.copy()
    swap(h)
    res = reduce_gpf(h)
    out = res.to_graph()
    detectors = list(res.detectors)
    if checks:
        detectors += [c for c in one_spider_per_outcome(out) if hasattr(c, 'free_vars') and c.free_vars()]
    swap(out)
    return out, detectors


def unfuse_outcomes(g: BaseGraph) -> BaseGraph:
    """Each Z- or X-spider whose phase depends on outcomes, with that part split off; ``g`` in place.

    A phase ``alpha + (c[i] + c[j] + ...) pi`` becomes ``alpha`` on the spider and
    ``(c[i] + c[j] + ...) pi`` on a new spider of the same colour with one leg, joined
    to it by a plain edge (spider fusion read backwards) and drawn below it, a little to
    the left, so that its phase does not cover the next qubit's spider. A
    spider with one leg is left as it is: its phase is already on its own.
    """
    from ..codes.phasepoly import affine_parts
    for v in list(g.vertices()):
        if not vertex_is_zx(g.type(v)) or g.vertex_degree(v) <= 1 or not isinstance(g.phase(v), Poly):
            continue
        try:
            names, _const = affine_parts(g.phase(v))
        except ValueError:
            continue
        if not names:
            continue
        phase = g.phase(v)
        g.set_phase(v, sum((Fraction(c) for c, t in phase.terms if not t.vars), Fraction(0)))
        leaf = g.add_vertex(g.type(v), g.qubit(v) + 0.25, g.row(v) - 0.4, Poly([(c, t) for c, t in phase.terms if t.vars]))
        g.add_edge((v, leaf))
    return g
