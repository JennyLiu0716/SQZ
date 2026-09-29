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

"""Simplifying the layer of phase gadgets as a whole.

After the row reduction every non-Pauli phase sits on a gadget. The gadgets of
Z colour (a Z-leaf on an X-hub, the hub reading spiders of the Z-layer)
together multiply each term of the diagram by ``exp(i pi P(y, c))``, where the
``y`` are the bits of the Z-layer spiders and the ``c`` the measurement
outcomes that appear on hubs and leaves. Writing each parity as an integer
polynomial,

    y_1 xor ... xor y_n = sum over non-empty T of (-2)^(|T|-1) prod_{i in T} y_i ,

makes ``P`` a multilinear polynomial whose coefficients are taken modulo 2
(units of pi). That representation is unique: two gadget layers act the same
exactly when they have the same ``P``. So in one pass:

* gadgets that cancel disappear -- a ``T`` and a ``T^dag`` on the same parity,
  or a whole layer whose parity matrix is triorthogonal (the spider-nest
  identities; for ``pi/4`` gadgets the coefficients of degree 1, 2 and 3 are
  exactly the weight, pair-overlap and triple-overlap counts mod 8, 4 and 2);
* terms ``pi * y_i`` and ``pi * c_j * y_i`` are Pauli, and move onto the
  spider of ``y_i`` as a (Boolean) phase;
* what remains is written back as one canonical set of gadgets, using
  ``prod_{i in T} y_i = 2^(1-|T|) sum over non-empty S in T of (-1)^(|S|+1) xor(S)``;
* terms in the outcomes alone are global phases and are dropped (the reduction
  is up to a non-zero scalar).

Rows of the Z-layer that only gadgets were holding become free, and the next
round of row reduction removes them; this is how the reduction finds
detectors whose determinism depends on non-Clifford gates. Gadgets of X colour
are left untouched.

A Z-spider held by gadgets alone, whose bit ``y`` enters the phase only as
``y * (a + F)`` with ``a`` a multiple of ``pi/2`` and ``F`` a parity of other bits
and outcomes (the Clifford case, e.g. an ``S`` gate just before a measurement),
is summed out in closed form:

* ``a = 0`` or ``pi``: the sum is ``2`` if ``a + F = 0`` and ``0`` otherwise. With
  no other bit in ``F`` this is a detector (the spider is left isolated with
  phase ``a + F`` and becomes one); otherwise the spider is replaced by an
  X-spider with that phase on the bits of ``F``, a parity check that the next
  round of row reduction takes up;
* ``a = +-pi/2``: ``1 + e^{i a} (-1)^F`` is a non-zero multiple of ``e^{-i a F}``,
  so the spider is removed and one gadget with leaf ``-a`` on the parity ``F``
  takes its place (expanded, single bits of ``F`` get ``-a`` and pairs get ``pi``:
  the ``S`` and ``CZ`` gates of a local complementation).

``F`` must be a parity of single bits and single outcomes, so that every phase
stays affine in the outcomes; otherwise the spider is left alone.

Without this, the zero branches such a spider causes would not be reported
as detectors.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Any

from pyzx.graph.base import BaseGraph
from pyzx.symbolic import new_var
from pyzx.utils import VertexType

from .matrix import phase_mod2
from .normal_form import GPF_ROLE_KEY, LEAF, is_leaf

__all__ = ['simplify_gadgets', 'phase_polynomial']

Poly = dict          # frozenset of variables -> Fraction in [0, 2)


def _add(p: Poly, mono: frozenset, coeff: Fraction) -> None:
    c = (p.get(mono, Fraction(0)) + coeff) % 2
    if c:
        p[mono] = c
    else:
        p.pop(mono, None)


def _split_phase(phase: Any) -> tuple[Fraction, list[str]]:
    """A phase ``a + (c_1 + ... + c_m)`` (units of pi) as its constant and its Boolean variables.

    Raises ``ValueError`` for anything else (a non-dyadic constant, a product of
    variables, a non-Boolean variable, or a non-integer coefficient on a variable).
    """
    if not hasattr(phase, 'terms'):
        return Fraction(phase) % 2, []
    const = Fraction(0)
    names: list[str] = []
    for coeff, term in phase.terms:
        c = Fraction(coeff) % 2
        if not term.vars:
            const += c
        elif len(term.vars) == 1 and term.vars[0][1] == 1 and term.vars[0][0].is_bool and c in (0, 1):
            if c:
                names.append(term.vars[0][0].name)
        else:
            raise ValueError("phase {} is not a constant plus a parity of outcomes".format(phase))
    return const % 2, names


def _xor_terms(variables: list[Any], weight: Fraction) -> Poly:
    """``weight * xor(variables)`` as a multilinear polynomial mod 2, truncated where terms vanish."""
    out: Poly = {}
    for size in range(1, len(variables) + 1):
        coeff = weight * Fraction(-2) ** (size - 1)
        if coeff % 2 == 0:
            break                      # every larger size vanishes too
        for t in combinations(variables, size):
            _add(out, frozenset(t), coeff)
    return out


def _times_var(p: Poly, var: Any) -> Poly:
    out: Poly = {}
    for mono, c in p.items():
        _add(out, mono | {var}, c)
    return out


def phase_polynomial(g: BaseGraph) -> tuple[Poly, list[tuple[Any, list[Any]]]]:
    """The phase polynomial of the Z-gadgets of ``g`` and the gadgets themselves (hub, leaves).

    Variables are ``('z', v)`` for the bit of Z-spider ``v`` and outcome names
    (strings) for measurement outcomes.
    """
    total: Poly = {}
    gadgets = []
    for hub in sorted(g.vertices()):
        if g.type(hub) != VertexType.X or is_leaf(g, hub):
            continue
        leaves = [w for w in g.neighbors(hub) if is_leaf(g, w)]
        if not leaves or any(g.type(w) != VertexType.Z for w in leaves):
            continue
        reads = [w for w in g.neighbors(hub) if not is_leaf(g, w)]
        if any(g.type(w) != VertexType.Z for w in reads):
            continue                   # not a gadget on Z-layer bits; leave it alone
        hub_const, hub_vars = _split_phase(g.phase(hub))
        variables: list[Any] = [('z', w) for w in sorted(reads)] + sorted(hub_vars)
        k = int(hub_const) % 2            # the leaf's bit is xor(variables) + k
        if hub_const % 1:
            raise ValueError("hub phase {} is not Pauli".format(g.phase(hub)))
        for leaf in leaves:
            alpha, leaf_vars = _split_phase(g.phase(leaf))
            # (alpha + sum of leaf_vars) * (k + (-1)^k xor(variables)), term by term
            sign = Fraction(-1) if k else Fraction(1)
            if k:
                _add(total, frozenset(), alpha)
                for c in leaf_vars:
                    _add(total, frozenset([c]), Fraction(1))
            for mono, c in _xor_terms(variables, sign * alpha).items():
                _add(total, mono, c)
            for var in leaf_vars:
                for mono, c in _times_var(_xor_terms(variables, sign), var).items():
                    _add(total, mono, c)
        gadgets.append((hub, leaves))
    return total, gadgets


def _synthesise(p: Poly, search: int = 10) -> dict[frozenset, Fraction]:
    """Parities and angles whose gadgets have phase polynomial ``p`` (up to terms in outcomes only).

    A coefficient of a monomial of degree ``k`` is only defined modulo 2, but it
    is divided by ``2^(k-1)`` when written as parities, so the lift matters:
    ``y1 y2 y3`` with coefficient 1 is ``pi/4`` on three parities of size 1 and
    3 and ``-pi/4`` on three of size 2, while with coefficient -1 it gives the
    opposite signs, which can cancel against other monomials. The lift of each
    monomial of degree 2 or more is chosen to give the fewest gadgets: by trying
    all choices when there are at most ``search`` such monomials, and otherwise
    by taking coefficients in ``(-1, 1]``. Either way the result has phase
    polynomial ``p``.
    """
    monos = sorted(p.items(), key=lambda kv: (-len(kv[0]), sorted(map(repr, kv[0]))))

    def angles_for(lifts: list[Fraction]) -> dict[frozenset, Fraction]:
        angles: dict[frozenset, Fraction] = {}
        for (mono, _a), a in zip(monos, lifts):
            k = len(mono)
            for size in range(1, k + 1):
                for sub in combinations(sorted(mono, key=repr), size):
                    key = frozenset(sub)
                    angles[key] = (angles.get(key, Fraction(0)) + a * (-1) ** (size + 1) / 2 ** (k - 1)) % 2
        return {s: a for s, a in angles.items() if a and any(isinstance(v, tuple) for v in s)}

    base = [a if a <= 1 else a - 2 for _m, a in monos]              # coefficients in (-1, 1]
    flexible = [i for i, (m, _a) in enumerate(monos) if len(m) >= 2]
    best = angles_for(base)
    if len(flexible) <= search:
        for choice in range(1, 1 << len(flexible)):
            lifts = list(base)
            for bit, i in enumerate(flexible):
                if choice >> bit & 1:
                    lifts[i] = lifts[i] - 2 if lifts[i] > 0 else lifts[i] + 2
            cand = angles_for(lifts)
            if len(cand) < len(best):
                best = cand
    return best


def _outcomes_off_leaves(g: BaseGraph, gadgets: list[tuple[Any, list[Any]]]) -> int:
    """Move the outcome part of leaf phases onto the spiders the gadget reads (in place).

    A leaf ``a + c`` on a gadget that reads the bits ``R`` (hub parity ``s``)
    contributes ``(a + c) * xor(R, s)``. Its ``c`` part is ``pi c`` on each bit of
    ``R``, a Pauli, plus ``pi c s``, which involves outcomes only and is a
    global phase. So the leaf keeps ``a`` and each read spider gains ``c``,
    without rewriting the rest of the layer. Returns how many leaves changed.
    """
    reg = g.var_registry
    changed = 0
    for hub, leaves in gadgets:
        reads = [w for w in g.neighbors(hub) if not is_leaf(g, w)]
        for leaf in leaves:
            try:
                a, names = _split_phase(g.phase(leaf))
            except ValueError:
                continue
            if not names:
                continue
            g.set_phase(leaf, a)
            for r in reads:
                ph = g.phase(r)
                for c in names:
                    ph = ph + new_var(c, True, reg)
                g.set_phase(r, phase_mod2(ph))
            changed += 1
    return changed


def _sum_out_free_bits(g: BaseGraph, p: Poly, gadgets: list[tuple[Any, list[Any]]]) -> int:
    """Sum out the Z-spiders that only Clifford gadgets hold (see the module docstring).

    Works on ``g`` locally and keeps ``p`` up to date; returns how many spiders
    were summed out. With ``P = y (a + F) + P_0`` and ``P_0`` free of ``y``,
    disconnecting ``y`` from its hubs leaves gadgets whose polynomial is exactly
    ``P_0`` (it is ``P`` at ``y = 0``), so only the ``y (a + F)`` part needs
    replacing: by a detector, a parity check, or one gadget for ``-a F``.
    """
    hubs = {hub for hub, _leaves in gadgets}
    reg = g.var_registry
    done = 0
    for y in sorted(g.vertices()):
        if y not in g.vertices() or g.type(y) != VertexType.Z or is_leaf(g, y):
            continue
        around = list(g.neighbors(y))
        if not around or not hubs.issuperset(around):
            continue
        try:
            a, own = _split_phase(g.phase(y))
        except ValueError:
            continue
        var = ('z', y)
        mine = {mono: c for mono, c in p.items() if var in mono}
        alpha = (mine.get(frozenset([var]), Fraction(0)) + a) % 2
        if alpha % Fraction(1, 2):
            continue
        parity: set[frozenset] = set()         # F as a set of monomials, each with coefficient 1
        clifford = True
        for mono, c in list(mine.items()) + [(frozenset([var, n]), Fraction(1)) for n in own]:
            rest = mono - {var}
            if not rest:
                continue
            if c != 1 or len(rest) > 1:
                clifford = False               # a controlled phase on y, or not affine in the outcomes
                break
            parity ^= {rest}
        if not clifford:
            continue
        variables = [next(iter(m)) for m in parity]
        bits = sorted((v for v in variables if isinstance(v, tuple)), key=repr)
        outcomes = sorted(v for v in variables if not isinstance(v, tuple))
        # detach y: the gadgets now contribute P at y = 0; drop gadgets left reading nothing
        for hub in around:
            g.remove_edge(g.edge(y, hub))
            if not any(not is_leaf(g, w) for w in g.neighbors(hub)):
                g.remove_vertices([w for w in g.neighbors(hub)] + [hub])
                hubs.discard(hub)
        for mono in mine:
            p.pop(mono, None)
        parity_phase: Any = 0
        for c in outcomes:
            parity_phase = parity_phase + new_var(c, True, reg)
        q = sum(g.qubit(v[1]) for v in bits) / len(bits) if bits else g.qubit(y)
        if alpha in (0, 1):
            if not bits:
                g.set_phase(y, parity_phase + int(alpha))    # isolated: a detector (or the scalar 2)
            else:
                x = g.add_vertex(VertexType.X, q, g.row(y), parity_phase + int(alpha))
                for v in bits:
                    g.add_edge((x, v[1]))
                g.remove_vertex(y)
        else:
            g.remove_vertex(y)
            if bits:                           # one gadget: -alpha * xor(bits, outcomes)
                hub = g.add_vertex(VertexType.X, q + 0.5, 1.5, parity_phase)
                leaf = g.add_vertex(VertexType.Z, q + 0.5, 0.8, (-alpha) % 2)
                g.set_vdata(leaf, GPF_ROLE_KEY, LEAF)
                g.add_edge((hub, leaf))
                for v in bits:
                    g.add_edge((hub, v[1]))
                hubs.add(hub)
                for mono, c in _xor_terms(bits + outcomes, -alpha).items():
                    _add(p, mono, c)
        done += 1
    return done


def simplify_gadgets(g: BaseGraph) -> bool:
    """Simplify the layer of Z-gadgets of ``g`` (in place); returns whether the diagram changed.

    One step per call, the first that applies: move the outcome part of leaf
    phases onto the spiders the gadgets read; sum out the Z-spiders that only
    Clifford gadgets hold (see the module docstring); or replace the gadgets by
    the canonical set with the same phase polynomial, with Pauli terms moved
    onto the spiders -- only if that does not make the layer larger. Leaves the
    diagram alone if a phase is not of the supported form (a dyadic constant
    plus a parity of outcomes).
    """
    try:
        p, gadgets = phase_polynomial(g)
    except ValueError:
        return False
    if not gadgets:
        return False
    if _outcomes_off_leaves(g, gadgets):
        return True                    # the next round simplifies what is left
    if _sum_out_free_bits(g, p, gadgets):
        return True
    # Pauli terms: pi * y_i, or pi * c_j * y_i, become phases on the spiders
    paulis: dict[Any, list[Any]] = {}
    rest: Poly = {}
    for mono, c in p.items():
        ys = [v for v in mono if isinstance(v, tuple)]
        cs = [v for v in mono if not isinstance(v, tuple)]
        if c == 1 and len(ys) == 1 and len(cs) <= 1:
            paulis.setdefault(ys[0][1], []).append(cs[0] if cs else 1)
        elif ys:
            rest[mono] = c
        # terms in the outcomes alone are global phases
    new = _synthesise(rest)
    old = set()
    for hub, leaves in gadgets:
        reads = frozenset(('z', w) for w in g.neighbors(hub) if not is_leaf(g, w))
        old.add((reads, str(g.phase(hub)), tuple(sorted(str(g.phase(w)) for w in leaves))))
    proposed = set()
    for s, a in new.items():
        reads = frozenset(v for v in s if isinstance(v, tuple))
        hub_text = " + ".join(sorted(v for v in s if not isinstance(v, tuple))) or "0"
        proposed.add((reads, hub_text, (str(a),)))
    if not paulis and proposed == old:
        return False
    if len(new) > len(gadgets):
        return False                   # never make the layer larger, even to move Pauli terms
    reg = g.var_registry
    for hub, leaves in gadgets:
        g.remove_vertices(leaves + [hub])
    for v, parts in paulis.items():
        phase = g.phase(v)
        for part in parts:
            phase = phase + (1 if part == 1 else new_var(part, True, reg))
        g.set_phase(v, phase_mod2(phase))
    for s, a in sorted(new.items(), key=lambda kv: sorted(map(repr, kv[0]))):
        ys = sorted(v[1] for v in s if isinstance(v, tuple))
        cs = sorted(v for v in s if not isinstance(v, tuple))
        hub_phase: Any = 0
        for c in cs:
            hub_phase = hub_phase + new_var(c, True, reg)
        q = sum(g.qubit(v) for v in ys) / len(ys)
        hub = g.add_vertex(VertexType.X, q + 0.5, 1.5, hub_phase)
        leaf = g.add_vertex(VertexType.Z, q + 0.5, 0.8, a)
        g.set_vdata(leaf, GPF_ROLE_KEY, LEAF)
        g.add_edge((hub, leaf))
        for v in ys:
            g.add_edge((hub, v))
    return True
