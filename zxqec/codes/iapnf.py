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

"""The isometry affine-with-phases normal form (iAPNF) of a Clifford isometry.

Yeh, Huang, Kissinger, Li and van de Wetering (*A three-way normal form for
stabiliser codes*, Theorem 4.1, in preparation) write every Clifford isometry
``E`` from ``k`` logical qubits to ``n`` physical qubits, such as the ideal
encoder of a stabilizer code, as three layers (Figure 4.12 of the thesis):

``E = U_O . C . U_I``

* ``U_I``, a Clifford on the ``k`` inputs, in three layers: ``S`` and ``CZ``
  gates, then Hadamards, then ``X``-``CZ`` gates (two X-spiders joined by a
  Hadamard edge);
* ``C``, a phase-free CSS core in X-Z generalized parity form: one X-spider on
  each input and one X-spider per Z-check, each with a Pauli phase ``b pi``,
  joined by the matrix ``B`` to one Z-spider on each output;
* ``U_O``, a diagonal Clifford on the outputs: phases in multiples of
  ``pi/2`` (``S``, ``Z``) and ``CZ`` gates.

The manuscript is not public; the conventions here are derived as follows,
and make the form unique (the tests check this on thousands of presentations
of the same maps).

* **The code part.** The stabilizer group of the image is ``U_O`` applied to
  a CSS code (Yeh et al.'s factorisation of a tableau into a CSS core and a
  diagonal Clifford, Theorem 6.1 and Proposition 6.2). The Z-checks of that
  CSS code are the rows of ``B`` below the logical rows, in reduced row
  echelon form, with their signs as the phases ``b``.
* **U_I.** Let ``Lambda`` be the logical Paulis that ``E`` maps to Z-type
  Paulis (up to stabilizers). Its X parts, in reduced row echelon form, have
  pivots ``A``: the Hadamards act on ``A``; the ``S`` and ``CZ`` gates act
  only within ``A`` and are read off ``Lambda``; the ``X``-``CZ`` gates join
  each ``a`` in ``A`` to the qubits outside ``A`` where row ``a`` of the
  echelon form has a 1. This parametrisation has exactly one choice for each
  coset of the Cliffords that the core absorbs (``CNOT`` circuits, logical
  ``S`` and ``CZ``, Paulis); the tests enumerate it for up to three inputs. ``S`` is never ``S^dagger`` here: a ``Z`` on an input
  is moved through to the outputs.
* **The logical rows of B** are what ``E . U_I^{-1}`` maps the logical ``Z_i``
  to, reduced modulo the Z-checks (zero on their pivot columns); their signs
  are the phases ``b`` of the logical X-spiders.
* **U_O** is *reduced*: it acts only on the outputs that are not pivot columns
  of the Z-checks. On the outputs the core can reach, the other coordinates are
  affine functions of these, so every diagonal Clifford has exactly one
  reduced form there (the reduced affine-with-phases form of a state).

Signs may be Boolean polynomials in measurement outcomes, as in the codes
:func:`~zxqec.codes.code_of` reads; they appear as phases ``c pi`` on the
X-spiders of the core and on the output Z-spiders.

:func:`iap_normal_form` takes a :class:`~zxqec.codes.StabilizerCode` (its
ideal encoder, the isometry with ``E X_i = Xbar_i E`` and ``E Z_i = Zbar_i E``
for its canonical logical operators) or the ZX-diagram of a Clifford isometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Sequence

import numpy as np
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph
from pyzx.utils import EdgeType, VertexType

from .gf2 import as_matrix, nullspace, rref
from .pauli import PauliString
from .pauli import phase_sum as _sum

__all__ = ['IAPNormalForm', 'iap_normal_form']


# -- signed Pauli strings, closed under products and Clifford conjugation -----------------------------

class _Pauli:
    """``(-1)^s i^e Z^z X^x`` with ``e`` in Z_4 and ``s`` a 0/1 or a Boolean polynomial."""
    __slots__ = ('x', 'z', 'e', 's')

    def __init__(self, x: Any, z: Any, e: int = 0, s: Any = 0) -> None:
        self.x = np.asarray(x, dtype=np.uint8) % 2
        self.z = np.asarray(z, dtype=np.uint8) % 2
        self.e = int(e) % 4
        self.s = _sum([s])

    @classmethod
    def hermitian(cls, x: Any, z: Any, sign: Any = 0) -> '_Pauli':
        """``(-1)^sign L(x, z)`` with ``L(x, z) = i^(-x.z) Z^z X^x``: letters X, Y, Z as in :class:`PauliString`."""
        x, z = np.asarray(x, dtype=np.uint8) % 2, np.asarray(z, dtype=np.uint8) % 2
        return cls(x, z, -int((x & z).sum()), sign)

    @classmethod
    def of(cls, p: PauliString) -> '_Pauli':
        return cls.hermitian(p.x, p.z, p.sign)

    @classmethod
    def single(cls, n: int, q: int, kind: str) -> '_Pauli':
        v, zero = np.zeros(n, np.uint8), np.zeros(n, np.uint8)
        v[q] = 1
        return cls(v, zero) if kind == 'X' else cls(zero, v)

    def __mul__(self, o: '_Pauli') -> '_Pauli':
        return _Pauli(self.x ^ o.x, self.z ^ o.z, self.e + o.e + 2 * int((self.x & o.z).sum()), _sum([self.s, o.s]))

    def sign(self) -> Any:
        """The sign of this Hermitian string as ``(-1)^sign L(x, z)``."""
        t = (self.e + int((self.x & self.z).sum())) % 4
        if t % 2:
            raise ValueError("not a Hermitian Pauli string")
        return _sum([self.s, t // 2])


def _image(gate: tuple, n: int, q: int, kind: str) -> _Pauli:
    """``U P U^dagger`` for ``P = X_q`` or ``Z_q`` and the gate ``U``."""
    name, qs = gate[0], gate[1:]
    one = _Pauli.single
    if name == 'Z':
        qs = qs[:1]
    if q not in qs:
        return one(n, q, kind)
    if name == 'H':
        return one(n, q, 'Z' if kind == 'X' else 'X')
    if name in ('S', 'Sdg'):
        if kind == 'Z':
            return one(n, q, 'Z')
        y = one(n, q, 'Z') * one(n, q, 'X')
        return _Pauli(y.x, y.z, 3 if name == 'S' else 1)        # S X S^dag = Y = i^-1 Z X; S^dag X S = -Y
    if name == 'Z':                                               # ('Z', q, c): Z^c with c a 0/1 or a polynomial
        p = one(n, q, kind)
        return _Pauli(p.x, p.z, 0, gate[2]) if kind == 'X' else p
    a, b = qs
    other = b if q == a else a
    if name == 'CZ':
        return one(n, other, 'Z') * one(n, q, 'X') if kind == 'X' else one(n, q, 'Z')
    if name == 'XCZ':
        return one(n, q, 'X') if kind == 'X' else one(n, q, 'Z') * one(n, other, 'X')
    raise ValueError("unknown gate {!r}".format(name))


def _conj(p: _Pauli, gates: Sequence[tuple]) -> _Pauli:
    """``U p U^dagger`` for the circuit ``gates``, applied first to last."""
    n = len(p.x)
    for g in gates:
        out = _Pauli(np.zeros(n), np.zeros(n), p.e, p.s)
        for q in np.flatnonzero(p.z):
            out = out * _image(g, n, int(q), 'Z')
        for q in np.flatnonzero(p.x):
            out = out * _image(g, n, int(q), 'X')
        p = out
    return p


def _inverse(gates: Sequence[tuple]) -> list[tuple]:
    swap = {'S': 'Sdg', 'Sdg': 'S'}
    return [(swap.get(g[0], g[0]),) + tuple(g[1:]) for g in reversed(gates)]


def _product(paulis: Sequence[_Pauli], coeffs: Any, n: int) -> _Pauli:
    out = _Pauli(np.zeros(n), np.zeros(n))
    for c, p in zip(coeffs, paulis):
        if c:
            out = out * p
    return out


def _diagonal(gamma: np.ndarray) -> list[tuple]:
    """``S`` on the diagonal of a symmetric GF(2) matrix and ``CZ`` on its other entries."""
    n = len(gamma)
    return ([('S', q) for q in range(n) if gamma[q, q]]
            + [('CZ', q, p) for q in range(n) for p in range(q + 1, n) if gamma[q, p]])


def _inverse_gf2(a: np.ndarray) -> np.ndarray:
    k = len(a)
    if not k:
        return np.zeros((0, 0), np.uint8)
    r, piv, t = rref(a, track=True)
    if len(piv) != k:
        raise ValueError("not invertible")
    return t % 2


def _times(a: np.ndarray, symbols: Sequence[Any]) -> list[Any]:
    """``a @ symbols`` over GF(2), for a 0/1 matrix and a vector of 0/1s or Boolean polynomials."""
    return [_sum([s for c, s in zip(row, symbols) if c]) for row in a]


# -- the normal form ------------------------------------------------------------------------------

@dataclass
class IAPNormalForm:
    """A Clifford isometry ``E = U_O . C . U_I`` in isometry AP normal form; see the module docstring.

    Inputs and outputs are numbered from 0; ``qubits`` labels the outputs. Phases
    are in units of pi; a sign may be a Boolean polynomial in outcomes.
    """
    k: int
    n: int
    qubits: tuple[Any, ...]
    hadamards: tuple[int, ...]                 #: ``A``: the inputs with a Hadamard in ``U_I``
    s_gates: tuple[int, ...]                   #: inputs with an ``S`` in ``U_I`` (all in ``A``)
    cz_in: tuple[tuple[int, int], ...]         #: ``CZ`` gates of ``U_I`` (within ``A``)
    xcz_in: tuple[tuple[int, int], ...]        #: ``X``-``CZ`` gates of ``U_I``: ``(a in A, j not in A)``
    logical_rows: np.ndarray                   #: ``k x n``: the outputs each logical X-spider of the core reaches
    logical_signs: tuple[Any, ...]             #: their phases ``b``
    check_rows: np.ndarray                     #: the Z-checks of the core, in reduced row echelon form
    check_signs: tuple[Any, ...]               #: their phases ``b``
    out_phases: dict[int, Any] = field(default_factory=dict)   #: output -> phase of ``U_O`` (units of pi)
    cz_out: tuple[tuple[int, int], ...] = ()   #: ``CZ`` gates of ``U_O``

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IAPNormalForm):
            return NotImplemented
        same = lambda a, b: a.shape == b.shape and bool((a == b).all())
        return (self.k == other.k and self.n == other.n and self.hadamards == other.hadamards
                and self.s_gates == other.s_gates and self.cz_in == other.cz_in and self.xcz_in == other.xcz_in
                and same(self.logical_rows, other.logical_rows) and same(self.check_rows, other.check_rows)
                and [str(s) for s in self.logical_signs] == [str(s) for s in other.logical_signs]
                and [str(s) for s in self.check_signs] == [str(s) for s in other.check_signs]
                and {q: str(v) for q, v in self.out_phases.items()} == {q: str(v) for q, v in other.out_phases.items()}
                and self.cz_out == other.cz_out)

    # -- reading it ---------------------------------------------------------------------------------

    def describe(self) -> str:
        """The three layers in words."""
        lab = lambda q: str(self.qubits[q])
        pr = lambda pairs, f=str: ', '.join('{}-{}'.format(f(a), f(b)) for a, b in pairs) or 'none'
        rows = lambda m: ['{' + ', '.join(lab(int(j)) for j in np.flatnonzero(r)) + '}' for r in m]
        sgn = lambda s: '' if str(s) == '0' else ' (phase {} pi)'.format(s)
        lines = ['isometry AP normal form: {} input(s) -> {} output(s)'.format(self.k, self.n),
                 '  U_I   S on inputs {}; CZ {}; H on inputs {}; X-CZ {}'.format(
                     list(self.s_gates) or 'none', pr(self.cz_in), list(self.hadamards) or 'none', pr(self.xcz_in)),
                 '  core  input X-spiders: ' + (', '.join(
                     'input {} -> outputs {}{}'.format(i, r, sgn(s)) for i, (r, s) in
                     enumerate(zip(rows(self.logical_rows), self.logical_signs))) or 'none'),
                 '        Z-check X-spiders: ' + (', '.join('outputs {}{}'.format(r, sgn(s)) for r, s in
                                                          zip(rows(self.check_rows), self.check_signs)) or 'none'),
                 '  U_O   phases {}; CZ {}'.format(
                     ', '.join('{}: {}'.format(lab(q), _phase_str(v)) for q, v in sorted(self.out_phases.items()))
                     or 'none', pr(self.cz_out, lab))]
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.describe()

    # -- the diagram --------------------------------------------------------------------------------

    def to_graph(self) -> BaseGraph:
        """The ZX-diagram of the three layers, laid out as in Figure 4.12 (inputs left, outputs right).

        Columns: ``S``/``CZ`` spiders (row 1), Hadamard edges, ``X``-``CZ`` spiders
        (row 3), the core's X-spiders (row 5), its output Z-spiders (row 7),
        ``U_O``'s spiders (row 9), the outputs (row 10).
        """
        g = Graph()
        reg = g.var_registry
        for s in list(self.logical_signs) + list(self.check_signs) + list(self.out_phases.values()):
            for v in getattr(s, 'free_vars', lambda: set())():
                reg.set_type(v.name, True)
        k, n = self.k, self.n
        ins = [g.add_vertex(VertexType.BOUNDARY, i, 0) for i in range(k)]
        outs = [g.add_vertex(VertexType.BOUNDARY, j, 10) for j in range(n)]
        last = list(ins)
        pending_h = set(self.hadamards)        # a Hadamard is the edge into the next spider
        zs: dict[int, Any] = {}
        for a in sorted(set(self.s_gates) | {q for pr in self.cz_in for q in pr}):
            v = g.add_vertex(VertexType.Z, a, 1, Fraction(1, 2) if a in self.s_gates else 0)
            g.add_edge((last[a], v))
            last[a] = v
            zs[a] = v
        for a, b in self.cz_in:
            g.add_edge((zs[a], zs[b]), EdgeType.HADAMARD)
        xs: dict[int, Any] = {}
        for a, j in self.xcz_in:
            for q in (a, j):
                if q not in xs:
                    v = g.add_vertex(VertexType.X, q, 3)
                    g.add_edge((last[q], v), EdgeType.HADAMARD if q in pending_h else EdgeType.SIMPLE)
                    pending_h.discard(q)
                    last[q] = v
                    xs[q] = v
            g.add_edge((xs[a], xs[j]), EdgeType.HADAMARD)
        zout = [g.add_vertex(VertexType.Z, j, 7) for j in range(n)]
        for i in range(k):
            v = g.add_vertex(VertexType.X, i, 5, self.logical_signs[i])
            g.add_edge((last[i], v), EdgeType.HADAMARD if i in pending_h else EdgeType.SIMPLE)
            for col in np.flatnonzero(self.logical_rows[i]):
                g.add_edge((v, zout[int(col)]))
        for h, (row, b) in enumerate(zip(self.check_rows, self.check_signs)):
            v = g.add_vertex(VertexType.X, k + h, 5, b)
            for col in np.flatnonzero(row):
                g.add_edge((v, zout[int(col)]))
        touched = {q for pr in self.cz_out for q in pr}
        uo: dict[int, Any] = {}
        for j in range(n):
            if j in self.out_phases or j in touched:
                v = g.add_vertex(VertexType.Z, j, 9, self.out_phases.get(j, 0))
                g.add_edge((zout[j], v))
                g.add_edge((v, outs[j]))
                uo[j] = v
            else:
                g.add_edge((zout[j], outs[j]))
        for q, p in self.cz_out:
            g.add_edge((uo[q], uo[p]), EdgeType.HADAMARD)
        g.set_inputs(tuple(ins))
        g.set_outputs(tuple(outs))
        g.rebind_variables_to_registry()
        return g

    def draw(self, figsize: tuple[float, float] | None = None, labels: bool = False, **kwargs: Any) -> Any:
        """PyZX's drawing of :meth:`to_graph`, with the three layers shaded and named as in Figure 4.12."""
        from pyzx.drawing import draw_matplotlib
        from pyzx.utils import get_mode, settings

        from ..spacetime.drawing import _finish

        rows = max(self.k + len(self.check_rows), self.n, 1)
        figsize = figsize or (7, 0.55 * rows + 1.2)
        mode = get_mode()
        settings.mode = "notebook"
        try:
            fig = draw_matplotlib(self.to_graph(), figsize=figsize, labels=labels, **kwargs)
        finally:
            settings.mode = mode
        ax = fig.axes[0]
        top = ax.get_ylim()[1]
        for (lo, hi), name, color in (((0.5, 4.0), 'U_I', '#8c8cd9'), ((4.4, 7.6), 'X-Z GPF', '#9a9a9a'),
                                      ((8.4, 9.6), 'U_O', '#d98c8c')):
            ax.axvspan(lo, hi, color=color, alpha=0.12, zorder=-1)
            ax.text((lo + hi) / 2, top, name, ha='center', va='bottom', fontsize=9, color=color)
        return _finish(fig)


def _phase_str(v: Any) -> str:
    try:
        f = Fraction(v)
    except (TypeError, ValueError):
        return '({}) pi'.format(v)
    return {Fraction(0): '0', Fraction(1, 2): 'pi/2', Fraction(1): 'pi', Fraction(3, 2): '3pi/2'}.get(f, str(f))


# -- computing it ---------------------------------------------------------------------------------

def iap_normal_form(e: Any) -> IAPNormalForm:
    """The isometry AP normal form of a Clifford isometry (see the module docstring).

    ``e`` is a :class:`~zxqec.codes.StabilizerCode` (its ideal encoder) or a
    ZX-diagram of a Clifford isometry, from its inputs to its outputs, whose
    code :func:`~zxqec.codes.code_of` can read (Boolean phases allowed).
    """
    from .code import StabilizerCode
    if isinstance(e, StabilizerCode):
        n, k = e.n, e.k
        stabs = [_Pauli.of(p) for p in e.stabilizers]
        lx = [_Pauli.of(p) for p in e.logical_x]
        lz = [_Pauli.of(p) for p in e.logical_z]
        return _normal_form(n, k, stabs, lx, lz, tuple(e.qubits))
    if isinstance(e, BaseGraph):
        n, k, stabs, lx, lz = _from_diagram(e)
        return _normal_form(n, k, stabs, lx, lz, tuple(range(n)))
    raise TypeError("iap_normal_form takes a StabilizerCode or a ZX-diagram, not {}".format(type(e).__name__))


def _from_diagram(g: BaseGraph) -> tuple[int, int, list[_Pauli], list[_Pauli], list[_Pauli]]:
    """The code and logical operators of the isometry ``g``, read off its Choi state.

    The Choi state ``sum_x |x> (x) E|x>`` is stabilized by ``I (x) s`` for the
    stabilizers ``s`` of the image and by ``X_i (x) Xbar_i``, ``Z_i (x) Zbar_i``.
    """
    from .code import code_of
    n, k = len(g.outputs()), len(g.inputs())
    choi = g.copy()
    choi.set_outputs(tuple(choi.outputs()) + tuple(choi.inputs()))
    choi.set_inputs(())
    state = code_of(choi)
    if state.k:
        raise ValueError("the diagram is not an isometry: its Choi state is not a stabilizer state")
    gens = [_Pauli.of(p) for p in state.stabilizers]
    sym = np.array([np.concatenate([p.x[n:], p.z[n:]]) for p in gens], np.uint8).reshape(len(gens), 2 * k)
    out = lambda p: _Pauli(p.x[:n], p.z[:n], p.e, p.s)

    def with_input_part(target: np.ndarray) -> _Pauli:
        # the product of generators whose input part is the given (x | z), restricted to the outputs
        c = _solve_left(sym, target)
        if c is None:
            raise ValueError("the diagram is not an isometry")
        p = _product(gens, c, n + k)
        return out(p)
    stabs = [out(_product(gens, c, n + k)) for c in nullspace(sym.T)] if len(gens) else []
    lx, lz = [], []
    for i in range(k):
        t = np.zeros(2 * k, np.uint8)
        t[i] = 1
        lx.append(with_input_part(t))
        t = np.zeros(2 * k, np.uint8)
        t[k + i] = 1
        lz.append(with_input_part(t))
    return n, k, stabs, lx, lz


def _solve_left(m: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """``c`` with ``c @ m = target`` over GF(2)."""
    from .gf2 import solve
    if not len(m):
        return np.zeros(0, np.uint8) if not target.any() else None
    return solve(m.T.copy(), target)


def _normal_form(n: int, k: int, stabs: list[_Pauli], lx: list[_Pauli], lz: list[_Pauli],
                 qubits: tuple[Any, ...]) -> IAPNormalForm:
    # 1. generators with independent X parts, in RREF; the pure Z ones, in RREF; Z parts reduced
    xm = as_matrix([p.x for p in stabs], n)
    if len(stabs):
        rx, xpiv, tx = rref(xm, track=True)
        xgens = [_product(stabs, c, n) for c in tx]
        zgens0 = [_product(stabs, c, n) for c in nullspace(xm.T)]
    else:
        rx, xpiv, xgens, zgens0 = np.zeros((0, n), np.uint8), [], [], []
    zm = as_matrix([p.z for p in zgens0], n)
    if len(zgens0):
        hz, zpiv, tz = rref(zm, track=True)
        zgens = [_product(zgens0, c, n) for c in tz]
    else:
        hz, zpiv, zgens = np.zeros((0, n), np.uint8), [], []
    free = [c for c in range(n) if c not in zpiv]

    def reduce_z(p: _Pauli) -> _Pauli:
        for row, c in zip(zgens, zpiv):
            if p.z[c]:
                p = p * row
        return p
    xgens = [reduce_z(p) for p in xgens]
    hx = as_matrix([p.x for p in xgens], n)
    w = as_matrix([p.z for p in xgens], n)
    r = len(xgens)
    # 2. U_O's code part: Gamma with Hx Gamma = W (Yeh et al., Prop. 6.2, R picking the pivot columns),
    #    then Z^t so that the X-checks of the CSS core have sign +
    gamma = np.zeros((n, n), np.int64)
    if r:
        rinv = np.zeros((n, r), np.int64)
        for i, c in enumerate(xpiv):
            rinv[c, i] = 1
        h64, w64 = hx.astype(np.int64), w.astype(np.int64)
        gamma = (rinv @ w64 + w64.T @ rinv.T + rinv @ h64 @ w64.T @ rinv.T) % 2
        assert ((h64 @ gamma) % 2 == w64).all() and (gamma == gamma.T).all()
    d1 = _diagonal(gamma)
    rho = [_conj(p, _inverse(d1)).sign() for p in xgens]
    t = _solve_rows(hx, rho, n)
    w1 = [('Z', q, t[q]) for q in range(n) if str(t[q]) != '0'] + d1      # Z^t, then D
    xchecks = [_Pauli.hermitian(v, np.zeros(n, np.uint8)) for v in hx]
    for p, v in zip(xgens, hx):
        q = _conj(p, _inverse(w1))
        assert (q.x == v).all() and not q.z.any() and str(q.sign()) == '0', "not a CSS core"

    def reduce_x(p: _Pauli) -> _Pauli:
        for row, c in zip(xchecks, xpiv):
            if p.x[c]:
                p = p * row
        return p
    # 3. the logical operators, pulled back through U_O's code part
    LX = [_conj(p, _inverse(w1)) for p in lx]
    LZ = [_conj(p, _inverse(w1)) for p in lz]
    # 4. Lambda: the logical Paulis mapped to Z-type ones (up to X-checks)
    xparts = as_matrix([reduce_x(p).x for p in LX + LZ], n)
    lam = nullspace(xparts.T) if k else np.zeros((0, 0), np.uint8)
    if len(lam) != k:
        raise ValueError("the logical operators do not form a symplectic basis")
    # 5. U_I from Lambda
    xl = lam[:, :k] if k else np.zeros((0, 0), np.uint8)
    ra, a_piv, ta = rref(xl, track=True) if k else (np.zeros((0, 0)), [], np.zeros((0, 0)))
    lam_rows = [(ta[i] @ lam) % 2 for i in range(len(a_piv))]
    zero_x = nullspace(xl.T) if k else np.zeros((0, 0), np.uint8)       # combinations with no x part
    pure_z = [(c @ lam) % 2 for c in zero_x]
    hadamards = tuple(a_piv)
    abar = [q for q in range(k) if q not in a_piv]
    xcz = tuple((a, j) for i, a in enumerate(a_piv) for j in abar if ra[i, j])
    gam_in = np.zeros((k, k), np.uint8)
    for i, a in enumerate(a_piv):
        vec = lam_rows[i].copy()
        if abar:
            zb = as_matrix([row[k:][abar] for row in pure_z], len(abar))
            comb = _solve_left(zb, vec[k:][abar])
            assert comb is not None
            for cc, row in zip(comb, pure_z):
                if cc:
                    vec ^= row
        assert not vec[k:][abar].any()
        for b in a_piv:
            gam_in[a, b] = vec[k + b]
    assert (gam_in == gam_in.T).all()
    s_gates = tuple(a for a in a_piv if gam_in[a, a])
    cz_in = tuple((a, b) for a in a_piv for b in a_piv if a < b and gam_in[a, b])
    ui = ([('S', a) for a in s_gates] + [('CZ', a, b) for a, b in cz_in] + [('H', a) for a in hadamards]
          + [('XCZ', a, j) for a, j in xcz])

    # 6. the logical operators of E . U_I^-1
    def through_e(pk: _Pauli) -> _Pauli:
        out = _Pauli(np.zeros(n), np.zeros(n), pk.e, pk.s)
        for i in np.flatnonzero(pk.z):
            out = out * LZ[i]
        for i in np.flatnonzero(pk.x):
            out = out * LX[i]
        return out
    e2x = [reduce_x(through_e(_conj(_Pauli.single(k, i, 'X'), _inverse(ui)))) for i in range(k)]
    e2z = [reduce_x(through_e(_conj(_Pauli.single(k, i, 'Z'), _inverse(ui)))) for i in range(k)]
    assert all(not p.x.any() for p in e2z), "U_I did not make the logical Z operators Z-type"
    e2z = [reduce_z(p) for p in e2z]
    logical_rows = as_matrix([p.z for p in e2z], n)
    logical_signs = tuple(p.sign() for p in e2z)
    check_signs = tuple(p.sign() for p in zgens)
    e2x = [reduce_z(p) for p in e2x]
    # 7. the diagonal that turns the core's logical X (pure X) into E . U_I^-1's, on the free columns
    kmat = as_matrix([list(v) for v in hx] + [list(p.x) for p in e2x], n)
    tmat = as_matrix([[0] * n for _ in range(r)] + [list(p.z) for p in e2x], n)
    beta = (kmat.astype(np.int64) @ tmat.T.astype(np.int64)) % 2
    assert (beta == beta.T).all()
    kf = kmat[:, free]
    kfi = _inverse_gf2(kf).astype(np.int64)
    g2f = (kfi @ beta @ kfi.T) % 2
    gamma2 = np.zeros((n, n), np.int64)
    for i, fi in enumerate(free):
        for j, fj in enumerate(free):
            gamma2[fi, fj] = g2f[i, j]
    d2 = _diagonal(gamma2)
    rhs = []
    for row, target in zip(kmat, xchecks + e2x):
        img = reduce_z(_conj(_Pauli.hermitian(row, np.zeros(n, np.uint8)), d2))
        assert (img.x == target.x).all() and (img.z == target.z).all()
        rhs.append(_sum([img.sign(), target.sign()]))
    t2f = _times(kfi % 2, rhs)
    uo_gates = w1 + [('Z', fq, t2f[i]) for i, fq in enumerate(free) if str(t2f[i]) != '0'] + d2
    # 8. U_O reduced to the free columns of the Z-checks
    out_phases, cz_out = _reduce_diagonal(uo_gates, n, hz, zpiv, free, check_signs)
    return IAPNormalForm(k=k, n=n, qubits=qubits, hadamards=hadamards, s_gates=s_gates, cz_in=cz_in,
                         xcz_in=xcz, logical_rows=logical_rows, logical_signs=logical_signs, check_rows=hz,
                         check_signs=check_signs, out_phases=out_phases, cz_out=cz_out)


def _solve_rows(a: np.ndarray, rhs: Sequence[Any], n: int) -> list[Any]:
    """``t`` with ``a @ t = rhs`` (``a`` with independent rows), the right-hand side possibly symbolic."""
    if not len(a):
        return [0] * n
    r, piv, tr = rref(a, track=True)
    t: list[Any] = [0] * n
    for p, tv in zip(piv, tr):
        t[p] = _sum([s for c, s in zip(tv, rhs) if c])
    return t


def _reduce_diagonal(gates: Sequence[tuple], n: int, hz: np.ndarray, zpiv: list[int], free: list[int],
                     offsets: Sequence[Any]) -> tuple[dict[int, Any], tuple[tuple[int, int], ...]]:
    """The diagonal Clifford ``gates`` on ``{z : Hz z = offsets}``, as phases and ``CZ`` s on the free columns.

    On that set each pivot coordinate is ``offset + (its row of Hz on the free
    columns) . y`` with ``y`` the free coordinates, so the phase ``i^E(z)`` is a
    function of ``y``: ``i^(a.y + 2 sum c_fg y_f y_g) (-1)^(tau . y)`` up to a
    constant. ``a`` and ``c`` are read off values at ``0``, ``e_f`` and
    ``e_f + e_g`` (constant offsets); a symbolic offset or ``Z`` exponent only
    adds to ``tau``.
    """
    a = np.zeros(n, np.int64)
    cz = np.zeros((n, n), np.int64)
    tau: list[Any] = [0] * n
    for g in gates:
        if g[0] == 'S':
            a[g[1]] += 1
        elif g[0] == 'Sdg':
            a[g[1]] -= 1
        elif g[0] == 'Z':
            c = g[2]
            if isinstance(c, (int, np.integer)):
                a[g[1]] += 2 * int(c)
            else:
                tau[g[1]] = _sum([tau[g[1]], c])
        elif g[0] == 'CZ':
            cz[g[1], g[2]] ^= 1
            cz[g[2], g[1]] ^= 1
    pivot_row = {c: i for i, c in enumerate(zpiv)}
    const = [int(s) if isinstance(s, (int, np.integer)) else 0 for s in offsets]
    sym = [None if isinstance(s, (int, np.integer)) else s for s in offsets]
    fidx = {f: i for i, f in enumerate(free)}

    def linear(q: int) -> np.ndarray:
        """Coordinate ``q`` as a GF(2) combination of the free coordinates (without its offset)."""
        v = np.zeros(len(free), np.int64)
        if q in fidx:
            v[fidx[q]] = 1
        else:
            v[:] = hz[pivot_row[q], free]
        return v

    def point(y: Sequence[int]) -> np.ndarray:
        z = np.zeros(n, np.int64)
        for f, val in zip(free, y):
            z[f] = val
        for c, i in pivot_row.items():
            z[c] = (const[i] + int(hz[i, free] @ np.array(y, np.int64))) % 2 if free else const[i]
        return z

    def expo(z: np.ndarray) -> int:
        return int(a @ z + 2 * (z @ np.triu(cz, 1) @ z)) % 4

    m = len(free)
    e0 = expo(point([0] * m))
    single = []
    for i in range(m):
        y = [0] * m
        y[i] = 1
        single.append(expo(point(y)))
    out_a = {free[i]: (single[i] - e0) % 4 for i in range(m)}
    cz_out = []
    for i in range(m):
        for j in range(i + 1, m):
            y = [0] * m
            y[i] = y[j] = 1
            d = (expo(point(y)) - single[i] - single[j] + e0) % 4
            assert d in (0, 2)
            if d:
                cz_out.append((free[i], free[j]))
    # symbolic parts: (-1)^(tau'.y)
    taup: list[Any] = [0] * m

    def add(vec: np.ndarray, s: Any) -> None:
        for i in np.flatnonzero(vec % 2):
            taup[i] = _sum([taup[i], s])
    for q in range(n):
        if str(tau[q]) != '0':
            add(linear(q), tau[q])
        if q in pivot_row and sym[pivot_row[q]] is not None:
            s = sym[pivot_row[q]]
            if a[q] % 2:
                add(linear(q), s)                      # i^(a z) with z = s + L: (-1)^(a s L)
            for p in np.flatnonzero(cz[q]):
                add(linear(int(p)), s)                 # CZ: (s + L_q)(L_p) contributes s L_p
    phases: dict[int, Any] = {}
    for i, f in enumerate(free):
        ph: Any = Fraction(int(out_a[f]), 2)
        if str(taup[i]) != '0':
            ph = taup[i] + ph if ph else taup[i]
        if str(ph) != '0':
            phases[f] = ph
    return phases, tuple(cz_out)
