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

"""A Clifford logical gate without a matrix, and the Clifford part of a non-Clifford one.

**The tableau** (:func:`clifford_tableau`). A Clifford isometry ``L`` is fixed, up
to a global phase, by where it sends each logical Pauli: ``L X_i L^dag`` and
``L Z_i L^dag``, each a Pauli string with a sign, together with the stabilizers
of its image. These are read off the Choi state of ``L`` by
:func:`~zxqec.codes.code_of`, which is Gaussian elimination for a Clifford
diagram, so the cost is polynomial in the number of logical qubits ``k``; the
matrix of ``L`` has ``4^k`` entries. Signs may be Boolean polynomials in
measurement outcomes: they are the Pauli frame, for all runs at once.

**The name** (:meth:`CliffordTableau.name`). The tableau without its signs fixes
``L`` up to a Pauli. It is split into connected parts (logical qubits that the
images join), and each part of one or two qubits is looked up among the named
Clifford gates of :mod:`~zxqec.codes.logical` (``H``, ``S``, ``CNOT``, ``CZ``,
``SWAP``, ...), whose tableaux are computed once from their small matrices. The
Pauli correction ``P`` with ``L = P . G`` then follows from the signs by one
linear solve over GF(2), with the signs as right-hand side, so it is a formula
in the outcomes. A part with no name is reported as a Clifford on its qubits;
its tableau describes it completely.

**The Clifford part** (:func:`clifford_plus_phases`). Any diagram, Clifford or
not, is a Clifford diagram ``U`` with extra outputs, followed by one phase effect
on each extra output (the paper's proposition on Clifford isometries followed by
phases): :func:`~zxqec.gpf.to_gpf` puts every non-Clifford phase on the leaf of a
gadget, and :func:`~zxqec.gpf.open_leaves` takes those leaves off. ``U`` is
Clifford, so its isometry AP normal form (:func:`~zxqec.codes.iap_normal_form`)
and its tableau are read as above.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
from pyzx.graph.base import BaseGraph

from .gf2 import rref
from .pauli import PauliString
from .pauli import phase_sum as _sum

__all__ = ['CliffordTableau', 'clifford_tableau', 'CliffordPlusPhases', 'clifford_plus_phases',
           'NotCliffordError']


class NotCliffordError(ValueError):
    """The diagram has a phase that is not a multiple of ``pi/2``."""


# -- the tableau --------------------------------------------------------------------------------

@dataclass
class CliffordTableau:
    """Where a Clifford isometry sends each logical Pauli; see the module docstring.

    ``x_images[i]`` is ``L X_i L^dag`` and ``z_images[i]`` is ``L Z_i L^dag``, on the
    ``k_out`` outputs; ``stabilizers`` generate the stabilizers of the image (none
    for a unitary; the prepared state's for ``k_in = 0``). Signs may be Boolean
    polynomials in outcomes.
    """
    k_in: int
    k_out: int
    x_images: list[PauliString]
    z_images: list[PauliString]
    stabilizers: list[PauliString] = field(default_factory=list)

    def describe(self) -> str:
        lines = ["Clifford tableau: {} -> {} logical qubit(s)".format(self.k_in, self.k_out)]
        for i, (x, z) in enumerate(zip(self.x_images, self.z_images)):
            lines.append("  X{} -> {}    Z{} -> {}".format(i, x, i, z))
        for s in self.stabilizers:
            lines.append("  stabilizer {}".format(s))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()

    def is_deterministic(self) -> bool:
        """Always ``True``: the tableau exists only when the gate does not depend on the run beyond its signs."""
        return True

    def name(self) -> tuple[str | None, str]:
        """``(operation, correction)``: the gate up to a Pauli, and the Pauli as a formula in the outcomes.

        ``correction`` is ``''`` for none, a string of letters on the outputs when it
        is the same in every run (as :meth:`LogicalAction.branches` prints it), or
        one ``X^(p) Z^(q) on j`` factor per output whose Pauli depends on the run.
        ``operation`` is ``None`` when the tableau is not of a gate or a one-qubit
        state that this module names (for example a split or a measurement).
        """
        if self.k_in == self.k_out:
            return self._name_unitary()
        if self.k_in == 0 and self.k_out == 1 and len(self.stabilizers) == 1:
            return self._name_state()
        return None, ''

    # -- naming -----------------------------------------------------------------------------

    def _rows(self) -> list[PauliString]:
        return list(self.x_images) + list(self.z_images)

    def _name_unitary(self) -> tuple[str | None, str]:
        k = self.k_in
        if k == 0:
            return 'a non-zero scalar', ''
        rows = self._rows()
        if k <= 2:
            best = None
            for name, ref in _library(k).items():
                if [r.letters for r in rows] != [p for p, _s in ref]:
                    continue
                corr = _correction(rows, [s for _p, s in ref], k)
                key = _weight_at_zero(corr)
                if best is None or key < best[0]:
                    best = (key, name, corr)
            if best is not None:
                return best[1], _render(best[2], k)
        # several parts: name each part of one or two logical qubits
        parts = _parts(rows, k)
        names, ref_signs = [], [None] * (2 * k)
        for qs in parts:
            sub = _restrict(rows, qs, k)
            named = None
            for name, ref in _library(len(qs)).items():
                if [r.letters for r in sub] == [p for p, _s in ref]:
                    named = (name, [s for _p, s in ref])
                    break
            if named is None:
                names.append("a Clifford on logical qubit{} {}".format('s' if len(qs) > 1 else '',
                                                                          ",".join(map(str, qs))))
                signs = [_value_at_zero(r.sign) for r in sub]
            else:
                if named[0] != 'I':
                    names.append("{} on logical qubit{} {}".format(named[0], 's' if len(qs) > 1 else '',
                                                                   ",".join(map(str, qs))))
                signs = named[1]
            m = len(qs)
            for j, q in enumerate(qs):
                ref_signs[q] = signs[j]
                ref_signs[k + q] = signs[m + j]
        corr = _correction(rows, ref_signs, k)
        return (" (x) ".join(names) if names else 'I'), _render(corr, k)

    def _name_state(self) -> tuple[str | None, str]:
        (s,) = self.stabilizers
        letter = s.letters
        state = {'Z': '|0>', 'X': '|+>', 'Y': '|+i>'}[letter]
        flip = {'Z': 'X', 'X': 'Y', 'Y': 'X'}[letter]        # the lightest Pauli that flips it, as _identify picks
        sign = s.sign
        if isinstance(sign, (int, np.integer)):
            return 'prepares ' + state, (flip if sign % 2 else '')
        return 'prepares ' + state, '{}^({})'.format(flip, sign)


def clifford_tableau(g: BaseGraph) -> CliffordTableau:
    """The Clifford tableau of the isometry ``g`` (inputs to outputs), read off its Choi state.

    Raises :class:`NotCliffordError` if a phase of ``g`` is not a multiple of
    ``pi/2`` (Boolean outcome terms aside), and the errors of
    :func:`~zxqec.codes.code_of` if ``g`` is not an isometry or its tableau
    depends on the run beyond its signs.
    """
    _check_clifford(g)
    n, k, stabs, lx, lz = _read_choi(g)
    as_string = lambda p: PauliString.from_arrays(p.x, p.z, p.sign())
    return CliffordTableau(k, n, [as_string(p) for p in lx], [as_string(p) for p in lz],
                           [as_string(p) for p in stabs])


def _choi_generators(g: BaseGraph) -> list[PauliString]:
    """Generators of the stabilizers of the Choi state of ``g`` (outputs first, then the bent inputs).

    Read by :func:`~zxqec.codes.code_of` with the echelon generators, which is
    Gaussian elimination for a Clifford diagram. Raises ``NotACodeError`` for the
    zero map.
    """
    from .code import code_of
    choi = g.copy()
    choi.set_outputs(tuple(choi.outputs()) + tuple(choi.inputs()))
    choi.set_inputs(())
    state = code_of(choi, generators='echelon')
    if state.k:
        raise ValueError("the diagram is not Clifford here: its Choi state is not a stabilizer state")
    return list(state.stabilizers)


def _read_choi(g: BaseGraph) -> tuple[int, int, list[Any], list[Any], list[Any]]:
    """As :func:`~zxqec.codes.iapnf._from_diagram`, with the echelon generators (Gaussian elimination only).

    The Choi state ``sum_x |x> (x) L|x>`` (outputs first, then inputs) is stabilized
    by ``s (x) I`` for the stabilizers ``s`` of the image and by ``Xbar_i (x) X_i``,
    ``Zbar_i (x) Z_i``; each image is the product of generators with that input part.
    """
    from .gf2 import nullspace
    from .iapnf import _Pauli, _product, _solve_left
    n, k = len(g.outputs()), len(g.inputs())
    if n + k == 0:
        return 0, 0, [], [], []                    # a scalar: no qubits, nothing to read
    gens = [_Pauli.of(p) for p in _choi_generators(g)]
    sym = np.array([np.concatenate([p.x[n:], p.z[n:]]) for p in gens], np.uint8).reshape(len(gens), 2 * k)
    out = lambda p: _Pauli(p.x[:n], p.z[:n], p.e, p.s)

    def image(target: np.ndarray) -> Any:
        c = _solve_left(sym, target)
        if c is None:
            raise ValueError("the diagram is not an isometry")
        return out(_product(gens, c, n + k))
    stabs = [out(_product(gens, c, n + k)) for c in nullspace(sym.T)] if len(gens) else []
    lx, lz = [], []
    for i in range(k):
        t = np.zeros(2 * k, np.uint8)
        t[i] = 1
        lx.append(image(t))
        t = np.zeros(2 * k, np.uint8)
        t[k + i] = 1
        lz.append(image(t))
    return n, k, stabs, lx, lz


# -- the Clifford part ---------------------------------------------------------------------------

@dataclass
class CliffordPlusPhases:
    """A diagram as a Clifford diagram ``clifford`` followed by phase effects on its extra outputs.

    ``phases[j]`` is ``(angle, kind)`` for the extra output ``k_out + j`` of
    ``clifford``: the effect ``<0| + e^(i pi angle) <1|`` for kind ``'Z'``
    (``<+| + e^(i pi angle) <-|`` for ``'X'``). ``normal_form`` and ``tableau`` are
    those of ``clifford``, or ``None`` with ``why`` when it is not an isometry.
    """
    clifford: BaseGraph
    k_in: int
    k_out: int
    phases: list[tuple[Fraction, str]]
    normal_form: Any = None
    tableau: CliffordTableau | None = None
    why: str = ''

    @property
    def is_isometry(self) -> bool:
        return self.tableau is not None

    def to_graph(self) -> BaseGraph:
        """``clifford`` with each phase effect plugged back onto its extra output: the diagram it came from."""
        from pyzx.utils import VertexType
        u = self.clifford.copy()
        outs = list(u.outputs())
        for b, (a, kind) in zip(outs[self.k_out:], self.phases):
            u.set_type(b, VertexType.Z if kind == 'Z' else VertexType.X)
            u.set_phase(b, a)
        u.set_outputs(tuple(outs[:self.k_out]))
        return u

    def describe(self) -> str:
        lines = ["Clifford part U: {} -> {} + {} outputs, followed by phase effects".format(
            self.k_in, self.k_out, len(self.phases))]
        for j, (a, kind) in enumerate(self.phases):
            lines.append("  output {}: {} phase {} pi".format(self.k_out + j, kind, a))
        if self.tableau is None:
            lines.append("  U is not an isometry here: " + self.why)
        else:
            lines.append("  U is a Clifford isometry; " + "\n  ".join(self.tableau.describe().splitlines()))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def clifford_plus_phases(g: BaseGraph) -> CliffordPlusPhases:
    """``g`` as a Clifford diagram followed by phase effects; see the module docstring.

    Only the gadget leaves whose angle is not a multiple of ``pi/2`` are taken off;
    Clifford gadgets stay in ``U``. ``U``'s normal form and tableau are read when
    ``U`` is an isometry.
    """
    from ..gpf import is_leaf, open_leaves, to_gpf
    from ..gpf.hadamard import restore_boundary_hadamards
    from .iapnf import iap_normal_form
    h = restore_boundary_hadamards(to_gpf(g.copy()))
    leaves = sorted(v for v in h.vertices() if is_leaf(h, v) and not _is_clifford_angle(h.phase(v)))
    phases = []
    from pyzx.utils import VertexType
    for v in leaves:
        a = h.phase(v)
        if hasattr(a, 'free_vars') and a.free_vars():
            raise NotImplementedError("clifford_plus_phases: a gadget leaf with an outcome in its angle")
        const = (sum((Fraction(c) for c, t in a.terms if not t.vars), Fraction(0)) if hasattr(a, 'terms')
                 else Fraction(a))
        phases.append((const % 2, 'Z' if h.type(v) == VertexType.Z else 'X'))
    u, _new = open_leaves(h, leaves)
    k_in, k_out = len(g.inputs()), len(g.outputs())
    out = CliffordPlusPhases(u, k_in, k_out, phases)
    try:
        out.tableau = clifford_tableau(u)
        out.normal_form = iap_normal_form(u, generators='echelon')
    except (ValueError, NotImplementedError) as e:      # NotACodeError is a ValueError
        out.tableau, out.normal_form, out.why = None, None, str(e)
    return out


# -- helpers --------------------------------------------------------------------------------------

def _is_clifford_angle(a: Any) -> bool:
    if hasattr(a, 'terms'):
        const = sum((Fraction(c) for c, t in a.terms if not t.vars), Fraction(0))
        others = [Fraction(c) for c, t in a.terms if t.vars]
        return const % Fraction(1, 2) == 0 and all(c % 1 == 0 for c in others)
    return Fraction(a) % Fraction(1, 2) == 0


def _check_clifford(g: BaseGraph) -> None:
    for v in g.vertices():
        if not _is_clifford_angle(g.phase(v)):
            raise NotCliffordError(
                "phase {} on vertex {} is not a multiple of pi/2, or a Clifford phase that depends on an "
                "outcome: there is no single Clifford tableau; name it by its matrix, or split it with "
                "clifford_plus_phases".format(g.phase(v), v))


def _value_at_zero(s: Any) -> int:
    """A sign with every outcome set to 0 (the reference run, as :meth:`LogicalAction.branches` takes it)."""
    if isinstance(s, (int, np.integer)):
        return int(s) % 2
    subs = {v: Fraction(0) for v in s.free_vars()}
    c = s.substitute(subs) if subs else s
    total = sum((Fraction(cc) for cc, t in c.terms if not t.vars), Fraction(0)) if hasattr(c, 'terms') else c
    return int(total) % 2


_LIBRARY: dict[int, dict[str, list[tuple[str, int]]]] = {}


def _library(k: int) -> dict[str, list[tuple[str, int]]]:
    """The named Clifford gates on ``k`` qubits with their tableaux: rows ``X_0..X_k-1, Z_0..Z_k-1`` as (letters, sign)."""
    if k not in _LIBRARY:
        from . import logical as L
        if k == 1:
            mats = {n: m for n, m in L._ONE.items() if n not in ('T', 'T^dag')}
        elif k == 2:
            mats = dict(L._TWO)
            one = {n: m for n, m in L._ONE.items() if n not in ('T', 'T^dag')}
            for (a, ma), (b, mb) in itertools.product(one.items(), repeat=2):
                mats['{} (x) {}'.format(a, b) if (a, b) != ('I', 'I') else 'I'] = np.kron(ma, mb)
        elif k == 3:
            mats = {'I': np.eye(8)}
        else:
            mats = {}
        _LIBRARY[k] = {n: _tableau_of_matrix(m, k) for n, m in mats.items()}
    return _LIBRARY[k]


def _tableau_of_matrix(u: np.ndarray, k: int) -> list[tuple[str, int]]:
    from .logical import _paulis
    paulis = _paulis(k)
    out = []
    for kind in 'XZ':
        for i in range(k):
            letters = ['I'] * k
            letters[i] = kind
            p = paulis[''.join(letters)]
            q = u @ p @ u.conj().T
            for name, r in paulis.items():
                if np.allclose(q, r, atol=1e-9):
                    out.append((name, 0))
                    break
                if np.allclose(q, -r, atol=1e-9):
                    out.append((name, 1))
                    break
            else:
                raise ValueError("not a Clifford matrix")
    return out


def _parts(rows: list[PauliString], k: int) -> list[list[int]]:
    """The logical qubits joined by the images (input i to every output its X_i or Z_i image touches)."""
    parent = list(range(k))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(k):
        for r in (rows[i], rows[k + i]):
            for j in r.support:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for q in range(k):
        groups.setdefault(find(q), []).append(q)
    return sorted(groups.values())


def _restrict(rows: list[PauliString], qs: list[int], k: int) -> list[PauliString]:
    return [rows[q].restricted(qs) for q in qs] + [rows[k + q].restricted(qs) for q in qs]


def _correction(rows: list[PauliString], ref_signs: list[Any], k: int) -> list[tuple[Any, Any]]:
    """The Pauli ``P`` on the outputs with ``L = P . G``, per output ``(x part, z part)``, as signs (maybe symbolic).

    ``P`` flips the sign of a row ``R`` exactly when it anticommutes with ``R``:
    ``omega(P, R) = sign(L) + sign(G)``. The rows form a symplectic basis, so the
    system has one solution.
    """
    a = np.array([np.concatenate([np.array(r.z, np.uint8), np.array(r.x, np.uint8)]) for r in rows], np.uint8)
    aug = np.concatenate([a, np.eye(2 * k, dtype=np.uint8)], axis=1)
    r, pivots, _ = rref(aug)
    if pivots[:2 * k] != list(range(2 * k)):
        raise ValueError("the images are not a symplectic basis: not a Clifford unitary")
    inv = r[:, 2 * k:]                                         # a^-1
    rhs = [_sum([row.sign, s]) for row, s in zip(rows, ref_signs)]
    p = [_sum([rhs[i] for i in range(2 * k) if inv[j, i]]) for j in range(2 * k)]
    return [(p[j], p[k + j]) for j in range(k)]


def _weight_at_zero(corr: list[tuple[Any, Any]]) -> tuple[int, str]:
    letters = ''.join({(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}[(_value_at_zero(x), _value_at_zero(z))]
                      for x, z in corr)
    order = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    return sum(c != 'I' for c in letters), ''.join(str(order[c]) for c in letters)


def _render(corr: list[tuple[Any, Any]], k: int) -> str:
    constant = all(isinstance(x, (int, np.integer)) and isinstance(z, (int, np.integer)) for x, z in corr)
    if constant:
        letters = ''.join({(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}[(int(x) % 2, int(z) % 2)]
                          for x, z in corr)
        return '' if set(letters) <= {'I'} else letters
    factors = []
    for j, (x, z) in enumerate(corr):
        bits = []
        for letter, s in (('X', x), ('Z', z)):
            if isinstance(s, (int, np.integer)):
                if s % 2:
                    bits.append(letter)
            else:
                bits.append('{}^({})'.format(letter, s))
        if bits:
            factors.append('{} on {}'.format(' '.join(bits), j))
    return ', '.join(factors)
