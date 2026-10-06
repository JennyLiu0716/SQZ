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
stabiliser codes*, Section 4.4, Algorithm 3 and Theorem 4.2, in preparation)
write every Clifford isometry ``E`` from ``k`` logical qubits to ``n``
physical qubits, such as the ideal encoder of a stabilizer code, as three
layers:

``E = U_O . C . U_I``

* ``U_I``, a Clifford on the ``k`` inputs, in three layers: Z-phases (multiples
  of ``pi/2``) and ``CZ`` gates, then Hadamards, then ``X``-``CZ`` gates (two
  X-spiders joined by a Hadamard edge);
* ``C``, a phase-free CSS core in X-Z generalized parity form: one X-spider on
  each input and one X-spider per Z-check, each with a Pauli phase ``b pi``,
  joined by the matrix ``B`` to one Z-spider on each output;
* ``U_O``, a diagonal Clifford on the outputs: phases in multiples of
  ``pi/2`` and ``CZ`` gates.

:func:`iap_normal_form` follows their algorithm.

1. **The Choi state in reduced AP form** (:func:`reduced_ap_form`). The Choi state ``sum_x |x> E|x>``,
   with the inputs ordered before the outputs, is a uniform superposition
   over ``{a : B a = b}`` with the phase ``i^(sum_f A_ff a_f) (-1)^(sum_(f<g)
   A_fg a_f a_g)``. *Reduced*: ``(B, b)`` in reduced row echelon form (so an
   input is a pivot whenever it can be), and ``A`` only on the free variables
   (the columns that are not pivots).
2. **Cutting off U_I and U_O.** ``A`` between free inputs gives ``U_I``'s
   phases and ``CZ`` gates; ``A`` between free outputs gives ``U_O``. What is
   left is ``B`` and the ``CZ`` edges between a free input and a free output.
3. **Inputs, top to bottom.** An input that is a pivot is the X-spider of its
   row of ``B`` (Case 1). An input ``j`` that is not a pivot (Cases 2 and 3)
   gets a Hadamard, and becomes the pivot of a new X-spider joined to the
   outputs it had ``CZ`` edges to; for each row of ``B`` it was in, whose
   pivot is an input ``i < j``, it leaves that row and gets an ``X``-``CZ``
   gate with ``i``.

So the Hadamards act on the inputs that are not pivots, the phases and ``CZ``
gates only between them, and each ``X``-``CZ`` gate joins one of them to an
earlier pivot input. The rows of ``B`` with an output pivot are the Z-checks of
the CSS core, in reduced row echelon form, and ``U_O`` acts only on the outputs
that are not their pivots. The reduced AP form of a state is unique and the
steps are deterministic, so the normal form is unique.

Signs may be Boolean polynomials in measurement outcomes, as in the codes
:func:`~zxqec.codes.code_of` reads; they appear as phases ``c pi`` on the
X-spiders of the core and on the Z-spiders of ``U_I`` and ``U_O``.

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

__all__ = ['APForm', 'IAPNormalForm', 'iap_normal_form', 'reduced_ap_form']


# -- signed Pauli strings, closed under products ------------------------------------------------------

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

    def __mul__(self, o: '_Pauli') -> '_Pauli':
        return _Pauli(self.x ^ o.x, self.z ^ o.z, self.e + o.e + 2 * int((self.x & o.z).sum()), _sum([self.s, o.s]))

    def sign(self) -> Any:
        """The sign of this Hermitian string as ``(-1)^sign L(x, z)``."""
        t = (self.e + int((self.x & self.z).sum())) % 4
        if t % 2:
            raise ValueError("not a Hermitian Pauli string")
        return _sum([self.s, t // 2])


def _product(paulis: Sequence[_Pauli], coeffs: Any, n: int) -> _Pauli:
    out = _Pauli(np.zeros(n), np.zeros(n))
    for c, p in zip(coeffs, paulis):
        if c:
            out = out * p
    return out


# -- the reduced AP form of a stabilizer state ----------------------------------------------------

@dataclass
class APForm:
    """A stabilizer state in reduced affine-with-phases form (Yeh et al., Definition 4.1), up to a scalar:

    ``sum_(a : B a = b) i^(sum_f A_ff a_f) (-1)^(sum_(f<g) A_fg a_f a_g) |a>``.

    ``B`` (:attr:`rows`) is in reduced row echelon form, and ``A`` is zero
    unless both variables are free (not :attr:`pivots`). In ZX: a Z-spider on
    each variable, an X-spider with phase ``b pi`` for each row, joined to the
    variables of the row, a phase ``A_ff pi/2`` on each free variable, and a
    Hadamard edge (``CZ``) for each ``A_fg = 1``. For an isometry
    (:func:`reduced_ap_form`), the variables are the inputs, then the outputs,
    of its Choi state.
    """
    labels: tuple[str, ...]                    #: the variables, in order
    rows: np.ndarray                           #: ``B``, in reduced row echelon form
    signs: tuple[Any, ...]                     #: ``b``: the phase ``b pi`` of each row's X-spider
    pivots: tuple[int, ...]                    #: the pivot column of each row
    phases: dict[int, Any]                     #: free variable -> ``A_ff pi/2`` (units of pi)
    cz: tuple[tuple[int, int], ...]            #: the pairs of free variables with ``A_fg = 1``

    @property
    def free(self) -> tuple[int, ...]:
        """The variables that are not pivots."""
        return tuple(c for c in range(len(self.labels)) if c not in self.pivots)

    def describe(self) -> str:
        """The rows, the free variables and the phases, in words."""
        lab = self.labels
        sgn = lambda s: '' if str(s) == '0' else ' (phase {} pi)'.format(s)
        lines = ['reduced AP form on {} variables: {}'.format(len(lab), ', '.join(lab)),
                 '  rows of B (X-spiders): ' + ('; '.join(
                     '{{{}}}, pivot {}{}'.format(', '.join(lab[int(j)] for j in np.flatnonzero(r)), lab[p], sgn(s))
                     for r, p, s in zip(self.rows, self.pivots, self.signs)) or 'none'),
                 '  free variables: ' + (', '.join(lab[f] for f in self.free) or 'none'),
                 '  phases: ' + (', '.join('{}: {}'.format(lab[f], _phase_str(v))
                                           for f, v in sorted(self.phases.items())) or 'none'),
                 '  CZ: ' + (', '.join('{}-{}'.format(lab[a], lab[b]) for a, b in self.cz) or 'none')]
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.describe()

    def to_graph(self, inputs: int = 0) -> BaseGraph:
        """The ZX-diagram of the form; the first ``inputs`` variables are drawn as inputs, on the left.

        With ``inputs = k`` the diagram of the Choi state of an isometry from
        ``k`` qubits is the isometry itself (Yeh et al., Eq. 20): input
        variables in column 1, the X-spiders of the rows in column 2.5 (a
        quarter row down, off the straight lines between the two columns), the
        output variables in column 4.
        """
        g = Graph()
        reg = g.var_registry
        for v in list(self.signs) + list(self.phases.values()):
            for name in getattr(v, 'free_vars', lambda: set())():
                reg.set_type(name.name, True)
        m = len(self.labels)
        spiders = []
        for q in range(m):
            left = q < inputs
            v = g.add_vertex(VertexType.Z, q if left else q - inputs, 1 if left else 4, self.phases.get(q, 0))
            b = g.add_vertex(VertexType.BOUNDARY, q if left else q - inputs, 0 if left else 5)
            g.add_edge((b, v))
            spiders.append(v)
        for r, (row, sign) in enumerate(zip(self.rows, self.signs)):
            x = g.add_vertex(VertexType.X, r + 0.25, 2.5, sign)
            for q in np.flatnonzero(row):
                g.add_edge((x, spiders[int(q)]))
        for a, b in self.cz:
            g.add_edge((spiders[a], spiders[b]), EdgeType.HADAMARD)
        bounds = [list(g.neighbors(v)) for v in spiders]
        boundary = [next(w for w in nb if g.type(w) == VertexType.BOUNDARY) for nb in bounds]
        g.set_inputs(tuple(boundary[:inputs]))
        g.set_outputs(tuple(boundary[inputs:]))
        g.rebind_variables_to_registry()
        return g


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
    hadamards: tuple[int, ...]                 #: ``A``: the inputs with a Hadamard in ``U_I`` (the non-pivots)
    in_phases: dict[int, Any]                  #: input -> phase of ``U_I`` (units of pi; inputs in ``A``)
    cz_in: tuple[tuple[int, int], ...]         #: ``CZ`` gates of ``U_I`` (within ``A``)
    xcz_in: tuple[tuple[int, int], ...]        #: ``X``-``CZ`` gates of ``U_I``: ``(i not in A, j in A)``, ``i < j``
    logical_rows: np.ndarray                   #: ``k x n``: the outputs each logical X-spider of the core reaches
    logical_signs: tuple[Any, ...]             #: their phases ``b`` (0 on the inputs in ``A``)
    check_rows: np.ndarray                     #: the Z-checks of the core, in reduced row echelon form
    check_signs: tuple[Any, ...]               #: their phases ``b``
    out_phases: dict[int, Any] = field(default_factory=dict)   #: output -> phase of ``U_O`` (units of pi)
    cz_out: tuple[tuple[int, int], ...] = ()   #: ``CZ`` gates of ``U_O``

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IAPNormalForm):
            return NotImplemented
        same = lambda a, b: a.shape == b.shape and bool((a == b).all())
        phases = lambda d: {q: str(v) for q, v in d.items()}
        return (self.k == other.k and self.n == other.n and self.hadamards == other.hadamards
                and phases(self.in_phases) == phases(other.in_phases) and self.cz_in == other.cz_in
                and self.xcz_in == other.xcz_in
                and same(self.logical_rows, other.logical_rows) and same(self.check_rows, other.check_rows)
                and [str(s) for s in self.logical_signs] == [str(s) for s in other.logical_signs]
                and [str(s) for s in self.check_signs] == [str(s) for s in other.check_signs]
                and phases(self.out_phases) == phases(other.out_phases) and self.cz_out == other.cz_out)

    # -- reading it ---------------------------------------------------------------------------------

    def describe(self) -> str:
        """The three layers in words."""
        lab = lambda q: str(self.qubits[q])
        pr = lambda pairs, f=str: ', '.join('{}-{}'.format(f(a), f(b)) for a, b in pairs) or 'none'
        ph = lambda d, f=str: ', '.join('{}: {}'.format(f(q), _phase_str(v)) for q, v in sorted(d.items())) or 'none'
        rows = lambda m: ['{' + ', '.join(lab(int(j)) for j in np.flatnonzero(r)) + '}' for r in m]
        sgn = lambda s: '' if str(s) == '0' else ' (phase {} pi)'.format(s)
        lines = ['isometry AP normal form: {} input(s) -> {} output(s)'.format(self.k, self.n),
                 '  U_I   phases {}; CZ {}; H on inputs {}; X-CZ {}'.format(
                     ph(self.in_phases), pr(self.cz_in), list(self.hadamards) or 'none', pr(self.xcz_in)),
                 '  core  input X-spiders: ' + (', '.join(
                     'input {} -> outputs {}{}'.format(i, r, sgn(s)) for i, (r, s) in
                     enumerate(zip(rows(self.logical_rows), self.logical_signs))) or 'none'),
                 '        Z-check X-spiders: ' + (', '.join('outputs {}{}'.format(r, sgn(s)) for r, s in
                                                          zip(rows(self.check_rows), self.check_signs)) or 'none'),
                 '  U_O   phases {}; CZ {}'.format(ph(self.out_phases, lab), pr(self.cz_out, lab))]
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.describe()

    # -- the diagram --------------------------------------------------------------------------------

    def to_graph(self, u_i: bool = True, u_o: bool = True, labelled: bool = False) -> BaseGraph:
        """The ZX-diagram of the three layers, laid out as in Theorem 4.2 (inputs left, outputs right).

        Columns: ``U_I``'s phase and ``CZ`` spiders (row 1), Hadamard edges,
        ``X``-``CZ`` spiders (row 3), the core's X-spiders (row 5), its output
        Z-spiders (row 7), ``U_O``'s spiders (row 9), the outputs (row 10).
        ``u_i=False`` or ``u_o=False`` leaves that layer out (the core alone is
        ``u_i=u_o=False``); ``labelled`` names each output by its qubit in
        :attr:`qubits` (the ``logical_qubit`` vertex data
        :func:`~zxqec.codes.glue` plugs by).
        """
        g = Graph()
        reg = g.var_registry
        for s in (list(self.logical_signs) + list(self.check_signs) + list(self.in_phases.values())
                  + list(self.out_phases.values())):
            for v in getattr(s, 'free_vars', lambda: set())():
                reg.set_type(v.name, True)
        k, n = self.k, self.n
        ins = [g.add_vertex(VertexType.BOUNDARY, i, 0) for i in range(k)]
        outs = [g.add_vertex(VertexType.BOUNDARY, j, 10) for j in range(n)]
        last = list(ins)
        pending_h = set(self.hadamards) if u_i else set()     # a Hadamard is the edge into the next spider
        zs: dict[int, Any] = {}
        for a in sorted(set(self.in_phases) | {q for pr in self.cz_in for q in pr}) if u_i else []:
            v = g.add_vertex(VertexType.Z, a, 1, self.in_phases.get(a, 0))
            g.add_edge((last[a], v))
            last[a] = v
            zs[a] = v
        for a, b in self.cz_in if u_i else ():
            g.add_edge((zs[a], zs[b]), EdgeType.HADAMARD)
        xs: dict[int, Any] = {}
        for i, j in self.xcz_in if u_i else ():
            for q in (i, j):
                if q not in xs:
                    v = g.add_vertex(VertexType.X, q, 3)
                    g.add_edge((last[q], v), EdgeType.HADAMARD if q in pending_h else EdgeType.SIMPLE)
                    pending_h.discard(q)
                    last[q] = v
                    xs[q] = v
            g.add_edge((xs[i], xs[j]), EdgeType.HADAMARD)
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
        touched = {q for pr in self.cz_out for q in pr} if u_o else set()
        uo: dict[int, Any] = {}
        for j in range(n):
            if u_o and (j in self.out_phases or j in touched):
                v = g.add_vertex(VertexType.Z, j, 9, self.out_phases.get(j, 0))
                g.add_edge((zout[j], v))
                g.add_edge((v, outs[j]))
                uo[j] = v
            else:
                g.add_edge((zout[j], outs[j]))
        for q, p in self.cz_out if u_o else ():
            g.add_edge((uo[q], uo[p]), EdgeType.HADAMARD)
        g.set_inputs(tuple(ins))
        g.set_outputs(tuple(outs))
        if labelled:
            from ..spacetime.subdiagram import set_logical_qubit
            for b, q in zip(outs, self.qubits):
                set_logical_qubit(g, b, q)
        g.rebind_variables_to_registry()
        return g

    def draw(self, figsize: tuple[float, float] | None = None, labels: bool = False, **kwargs: Any) -> Any:
        """PyZX's drawing of :meth:`to_graph`, with the three layers shaded and named as in Theorem 4.2."""
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
        for (lo, hi), name, color in (((0.5, 4.0), '$U_I$', '#8c8cd9'), ((4.4, 7.6), 'X-Z GPF', '#9a9a9a'),
                                      ((8.4, 9.6), '$U_O$', '#d98c8c')):
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
    code :func:`~zxqec.codes.code_of` can read (Boolean phases allowed). A
    diagram is read through its Choi state, so :func:`~zxqec.codes.code_of`'s
    limits apply (for instance, at most 14 independent parities read by
    ``pi/2`` gadgets); a code has no such limit.
    """
    n, k, stabs, lx, lz, qubits = _isometry_of(e)
    labels = tuple('in{}'.format(i) for i in range(k)) + tuple(str(q) for q in qubits)
    return _from_ap_form(_reduced_ap_form(_choi(n, k, stabs, lx, lz), labels), n, k, qubits)


def reduced_ap_form(e: Any) -> APForm:
    """The reduced AP form of the Choi state ``sum_x |x> (x) E|x>`` of a Clifford isometry, inputs first.

    ``e`` is a :class:`~zxqec.codes.StabilizerCode` (its ideal encoder) or a
    ZX-diagram of a Clifford isometry, as for :func:`iap_normal_form`, which
    starts from this form (step 1 of Yeh et al.'s Algorithm 3). The variables
    are named ``in0, in1, ...`` and then by the qubits of the outputs.
    """
    n, k, stabs, lx, lz, qubits = _isometry_of(e)
    labels = tuple('in{}'.format(i) for i in range(k)) + tuple(str(q) for q in qubits)
    return _reduced_ap_form(_choi(n, k, stabs, lx, lz), labels)


def _isometry_of(e: Any) -> tuple[int, int, list[_Pauli], list[_Pauli], list[_Pauli], tuple[Any, ...]]:
    from .code import StabilizerCode
    if isinstance(e, StabilizerCode):
        return (e.n, e.k, [_Pauli.of(p) for p in e.stabilizers], [_Pauli.of(p) for p in e.logical_x],
                [_Pauli.of(p) for p in e.logical_z], tuple(e.qubits))
    if isinstance(e, BaseGraph):
        n, k, stabs, lx, lz = _from_diagram(e)
        return n, k, stabs, lx, lz, tuple(range(n))
    raise TypeError("expected a StabilizerCode or a ZX-diagram, not {}".format(type(e).__name__))


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


def _phase(p: _Pauli) -> Any:
    """``A_ff`` in units of pi, from the stabilizer ``p = (-1)^s i^e Z^z X^(d_f)`` of the free variable ``f``.

    Moving ``f`` from 0 to 1 multiplies the amplitude by ``i^(A_ff)`` and by
    ``(-1)^(A_fg a_g)`` for the other free ``g``; ``p`` does the same with
    ``(-1)^s i^e (-1)^(z . a)``. So ``A_fg = z_g``, ``A_ff = z_f`` mod 2, and
    ``i^(-A_ff) = (-1)^s i^e``, which is ``A_ff = -(e + 2 s)`` mod 4.
    """
    const = Fraction((-p.e) % 4, 2)
    if isinstance(p.s, (int, np.integer)):
        return (const + int(p.s)) % 2
    return p.s + const if const else p.s


def _choi(n: int, k: int, stabs: list[_Pauli], lx: list[_Pauli], lz: list[_Pauli]) -> list[_Pauli]:
    """Stabilizers of the Choi state, inputs first: ``I (x) s``, ``X_i (x) Xbar_i`` and ``Z_i (x) Zbar_i``."""
    zk = np.zeros(k, np.uint8)

    def unit(i: int) -> np.ndarray:
        v = np.zeros(k, np.uint8)
        v[i] = 1
        return v
    gens = [_Pauli(np.concatenate([zk, p.x]), np.concatenate([zk, p.z]), p.e, p.s) for p in stabs]
    for i in range(k):
        gens.append(_Pauli(np.concatenate([unit(i), lx[i].x]), np.concatenate([zk, lx[i].z]), lx[i].e, lx[i].s))
        gens.append(_Pauli(np.concatenate([zk, lz[i].x]), np.concatenate([unit(i), lz[i].z]), lz[i].e, lz[i].s))
    if len(gens) != n + k:
        raise ValueError("expected n - k stabilizers and k pairs of logical operators")
    return gens


def _reduced_ap_form(gens: list[_Pauli], labels: tuple[str, ...]) -> APForm:
    """The reduced AP form of the state with the independent stabilizers ``gens``."""
    m = len(labels)
    # (B, b) from the Z-type stabilizers Z^c (-1)^b, i.e. the constraints c.a = b, in RREF
    xm = as_matrix([p.x for p in gens], m)
    zgens = [_product(gens, c, m) for c in nullspace(xm.T)]
    if zgens:
        _, piv, t = rref(as_matrix([p.z for p in zgens], m), track=True)
        rows = [_product(zgens, c, m) for c in t]
    else:
        piv, rows = [], []
    bmat = as_matrix([p.z for p in rows], m)
    free = [c for c in range(m) if c not in piv]
    # A on the free variables: for each free f, the stabilizer with X part d_f (1 on f, 0 on the other free
    # variables, and on each pivot what B then fixes), its Z part cleared on the pivots by the rows of B
    amat = np.zeros((m, m), np.uint8)
    phases: dict[int, Any] = {}
    for f in free:
        d = np.zeros(m, np.uint8)
        d[f] = 1
        for r, p in enumerate(piv):
            d[p] = bmat[r, f]
        c = _solve_left(xm, d)
        if c is None:
            raise ValueError("the operators given do not stabilize a state")
        q = _product(gens, c, m)
        for r, p in enumerate(piv):
            if q.z[p]:
                q = q * rows[r]
        amat[f] = q.z
        if str(_phase(q)) != '0':
            phases[f] = _phase(q)
    amat[np.arange(m), np.arange(m)] = 0
    assert (amat == amat.T).all() and not amat[piv].any()
    return APForm(labels=labels, rows=bmat, signs=tuple(p.sign() for p in rows), pivots=tuple(piv), phases=phases,
                  cz=tuple((f, g) for f in free for g in free if f < g and amat[f, g]))


def _from_ap_form(ap: APForm, n: int, k: int, qubits: tuple[Any, ...]) -> IAPNormalForm:
    """Yeh et al.'s Algorithm 3 on the reduced AP form of a Choi state (inputs ``0..k-1``, then the outputs)."""
    free = ap.free
    edges = {frozenset(e) for e in ap.cz}
    joined = lambda f: np.array([int(frozenset((f, k + j)) in edges) for j in range(n)], np.uint8)
    # cut off U_I (A between free inputs) and U_O (A between free outputs)
    fin = [f for f in free if f < k]
    in_phases = {f: v for f, v in ap.phases.items() if f < k}
    cz_in = tuple((a, b) for a, b in ap.cz if b < k)
    out_phases = {f - k: v for f, v in ap.phases.items() if f >= k}
    cz_out = tuple((a - k, b - k) for a, b in ap.cz if a >= k)
    # the inputs, top to bottom
    row_of = {p: r for r, p in enumerate(ap.pivots)}
    logical_rows = np.zeros((k, n), np.uint8)
    logical_signs: list[Any] = []
    xcz = []
    for i in range(k):
        if i in row_of:                                    # Case 1: the pivot of its row
            logical_rows[i] = ap.rows[row_of[i], k:]
            logical_signs.append(ap.signs[row_of[i]])
            continue
        if not joined(i).any():                            # Lemma 4.2
            raise ValueError("not an isometry: input {} has no X-spider and no CZ edge to an output".format(i))
        # Case 2 (in no row) or 3 (in the rows of earlier pivot inputs): H, and an X-CZ with each of those pivots
        xcz += [(p, i) for r, p in enumerate(ap.pivots) if p < k and ap.rows[r, i]]
        logical_rows[i] = joined(i)
        logical_signs.append(0)
    checks = [r for r, p in enumerate(ap.pivots) if p >= k]
    check_rows = as_matrix([ap.rows[r, k:] for r in checks], n)
    if k and len(rref(np.concatenate([logical_rows, check_rows], 0))[1]) != k + len(checks):
        raise ValueError("not an isometry: the logical X-spiders of the core are not independent")
    return IAPNormalForm(k=k, n=n, qubits=qubits, hadamards=tuple(fin), in_phases=in_phases, cz_in=cz_in,
                         xcz_in=tuple(sorted(xcz)), logical_rows=logical_rows, logical_signs=tuple(logical_signs),
                         check_rows=check_rows, check_signs=tuple(ap.signs[r] for r in checks),
                         out_phases=out_phases, cz_out=cz_out)
