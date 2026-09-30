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

"""Which code a diagram encodes into: reading a stabilizer code off the GPF.

Take a diagram whose inputs are logical qubits and whose outputs are physical
qubits -- an encoder, or a program cut after some stage, with its fresh qubits
prepared inside it. If it is a Clifford isometry, its image is a stabilizer
code. After :func:`~zxqec.gpf.reduce_gpf` the code can be read off the two
layers (Kissinger, "Phase-free ZX diagrams are CSS codes", 2022):

* each *internal* Z-spider (one without an input) is the X-type stabilizer
  ``X`` on the outputs it is wired to; its Boolean phase is the stabilizer's
  sign, i.e. the outcome of the measurement that projected onto it;
* each *input* Z-spider is a logical ``X`` operator, on the outputs it is wired to;
* the Z-type stabilizers are the ``Z`` strings orthogonal to every Z-spider's
  row; the phases of the output X-spiders give their signs;
* a Hadamard recorded on an output wire swaps ``X`` and ``Z`` on that qubit.

Codes that are not CSS in any Hadamard frame (the ``[[5,1,3]]`` code, or the
Steane code after one ``S``) have ``pi/2`` phase gadgets in the reduced diagram.
Flipping the bits of a combination ``a`` of internal Z-spiders then changes the
gadgets' phase by ``c + sum_i d_i y_i`` (a constant, and a Pauli term in the bits
``y`` of the Z-layer). With ``w`` the Z-string on the outputs for which
``w . (y A) = d . y`` (``A`` the rows of the Z-layer), the operator
``Z^w X^{aM}`` fixes the image, so the stabilizer has an X part and a Z part:
this is the ``[H_X | H_X G]`` block of the tableau normal form of a stabilizer
code. Its sign accounts for the ``i`` of each ``Y``.

A non-Clifford phase gadget hanging from internal Z-spiders means the
corresponding X-strings are *not* stabilizers of the image (flipping the bit
under the gadget changes its phase by something that is not a Pauli); they are
counted as logical instead. This is how the prefix of a magic-state injection
comes out as "the Steane code, holding a non-stabilizer logical state".

The logical operators are chosen canonically from the stabilizer group alone
(pure ``X`` and pure ``Z`` representatives for a CSS code), so two cuts of a
program that carry the same code also carry the same logical basis. That is
what makes "the logical operation of a block" well defined (see
:mod:`zxqec.codes.logical`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph
from pyzx.utils import EdgeType, VertexType

from ..gpf import has_boundary_hadamard, is_leaf, is_stem, reduce_gpf
from .pauli import phase_sum as _sum
from .gf2 import as_matrix, complement_basis, in_rowspace, nullspace, reduce_mod, rref, solve
from .pauli import PauliString

__all__ = ['StabilizerCode', 'NotACodeError', 'OutcomeDependentCodeError', 'code_of']


class NotACodeError(ValueError):
    """The diagram is not a Clifford isometry, so its image is not a stabilizer code."""


class OutcomeDependentCodeError(NotImplementedError):
    """The stabilizers themselves, not only their signs, depend on measurement outcomes.

    A Clifford phase controlled by an outcome (an ``S`` applied in some runs
    only) gives an X stabilizer a ``Z`` part in those runs only, so different
    runs are in different codes. Reading the code run by run is not
    implemented; this is raised instead of reporting a wrong code.
    """


# -- the code ------------------------------------------------------------------

@dataclass
class StabilizerCode:
    """A stabilizer code on labelled qubits, with signs and canonical logical operators.

    The generators are stored by their type in the frame in which the diagram was
    read (``hadamard[j]`` records a Hadamard on qubit ``j``): ``sx`` rows are X-type
    and ``sz`` rows Z-type in that frame. For a code that is not CSS in that frame,
    ``sx_z`` gives the Z part of each X-type generator, which is then the Pauli
    string with letters ``X``, ``Y``, ``Z`` where ``(sx, sx_z)`` is ``(1,0)``,
    ``(1,1)``, ``(0,1)`` and sign ``(-1)^sx_signs``. :attr:`stabilizers` gives the
    generators as physical Pauli strings.
    """
    qubits: tuple[Any, ...]
    hadamard: tuple[int, ...]
    sx: np.ndarray
    sx_signs: list[Any]
    sz: np.ndarray
    sz_signs: list[Any]
    logical_x: list[PauliString]
    logical_z: list[PauliString]
    non_clifford: bool = False
    positions: tuple[Any, ...] | None = None
    nonclifford_columns: frozenset = frozenset()   #: qubits reached by a non-Clifford phase gadget
    sx_z: np.ndarray | None = None                 #: Z part of the X-type generators (non-CSS codes)
    _distance: Any = field(default=None, repr=False)

    # -- basic data -----------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.qubits)

    @property
    def k(self) -> int:
        return len(self.logical_x)

    @property
    def stabilizers(self) -> list[PauliString]:
        """Independent generators as physical Pauli strings, X-type ones first."""
        zs: list[Any] = list(self.sx_z) if self.sx_z is not None and self._mixed else [None] * len(self.sx)
        return ([self._physical(row, s, 'X', zr) for row, s, zr in zip(self.sx, self.sx_signs, zs)]
                + [self._physical(row, s, 'Z') for row, s in zip(self.sz, self.sz_signs)])

    @property
    def _mixed(self) -> bool:
        """Whether some X-type generator has a Z part in the frame (the code is not CSS there)."""
        return self.sx_z is not None and bool(self.sx_z.size) and bool(self.sx_z.any())

    def _physical(self, row: np.ndarray, sign: Any, kind: str, zrow: np.ndarray | None = None) -> PauliString:
        h = np.array(self.hadamard, dtype=np.uint8)
        x = row if kind == 'X' else np.zeros_like(row)
        z = (zrow if zrow is not None else np.zeros_like(row)) if kind == 'X' else row
        flips = int((x & z & h).sum())                 # a Hadamard swaps X and Z, and H Y H = -Y
        return PauliString.from_arrays((x & (1 - h)) | (z & h), (z & (1 - h)) | (x & h),
                                       _sum([sign, flips]) if flips else sign)

    def symplectic(self) -> np.ndarray:
        """The stabilizer generators as rows ``(x | z)`` over GF(2), signs dropped."""
        return as_matrix([p.symplectic for p in self.stabilizers], 2 * self.n)

    @property
    def is_css(self) -> bool:
        """Every stabilizer and chosen logical operator is a pure ``X`` or pure ``Z`` string."""
        def pure(p: PauliString) -> bool:
            return not (any(p.x) and any(p.z))
        return all(pure(p) for p in self.stabilizers + self.logical_x + self.logical_z)

    # -- structure --------------------------------------------------------------

    def fixed_qubits(self) -> dict[Any, str]:
        """Qubits held in a known single-qubit state, outside every logical operator.

        Typically ancillas that are prepared but not yet used. Values are
        ``'0'``, ``'1'``, ``'+'``, ``'-'``, or ``'|c>'`` / ``'|(-1)^c>'`` style
        strings when the state depends on an outcome; ``'+i'`` and ``'-i'`` for the
        eigenstates of ``Y`` (codes that are not CSS).
        """
        if self._mixed:
            return {self.qubits[j]: _state_label(letter, sign)
                    for j, (letter, sign) in self._single_qubit_stabilizers().items()}
        out: dict[Any, str] = {}
        logical_support = {j for p in self.logical_x + self.logical_z for j in p.support}
        for j in range(self.n):
            if j in logical_support:
                continue
            for kind, gens, signs in (('X', self.sx, self.sx_signs), ('Z', self.sz, self.sz_signs)):
                e = np.zeros(self.n, dtype=np.uint8)
                e[j] = 1
                coeffs = _combination(gens, e)
                if coeffs is None:
                    continue
                sign = _sum([s for c, s in zip(coeffs, signs) if c])
                physical = kind if not self.hadamard[j] else ('Z' if kind == 'X' else 'X')
                out[self.qubits[j]] = _state_label(physical, sign)
                break
        return out

    def data(self) -> 'StabilizerCode':
        """The code with its fixed qubits (see :meth:`fixed_qubits`) removed."""
        fixed = set(self.fixed_qubits())
        keep = [j for j, q in enumerate(self.qubits) if q not in fixed]
        return self.restricted(keep) if len(keep) < self.n else self

    def restricted(self, columns: list[int]) -> 'StabilizerCode':
        """The code on ``columns``, assuming every other qubit is fixed (see :meth:`fixed_qubits`).

        Generators are first cleared on the dropped columns by multiplying with the
        single-qubit stabilizers there (keeping track of signs), then made independent.
        """
        if self._mixed:
            return self._restricted_mixed(list(columns))
        drop = [j for j in range(self.n) if j not in columns]
        sx, sxs = _clear_columns(self.sx, self.sx_signs, drop)
        sz, szs = _clear_columns(self.sz, self.sz_signs, drop)
        cols = list(columns)
        # generators living entirely on the dropped columns (another block) go
        sx, sxs = _nonzero_on(sx, sxs, cols)
        sz, szs = _nonzero_on(sz, szs, cols)
        keep = set(cols)
        pairs = [(x, z) for x, z in zip(self.logical_x, self.logical_z)
                 if set(x.support) & keep or set(z.support) & keep]
        nc = frozenset(cols.index(j) for j in self.nonclifford_columns if j in keep)
        return StabilizerCode(
            qubits=tuple(self.qubits[j] for j in cols),
            hadamard=tuple(self.hadamard[j] for j in cols),
            sx=sx[:, cols] if sx.size else np.zeros((0, len(cols)), np.uint8), sx_signs=sxs,
            sz=sz[:, cols] if sz.size else np.zeros((0, len(cols)), np.uint8), sz_signs=szs,
            logical_x=[x.restricted(cols) for x, _ in pairs],
            logical_z=[z.restricted(cols) for _, z in pairs],
            non_clifford=bool(nc),
            positions=tuple(self.positions[j] for j in cols) if self.positions else None,
            nonclifford_columns=nc)

    def _single_qubit_stabilizers(self) -> dict[int, tuple[str, Any]]:
        """Qubits outside every logical operator whose single-qubit Pauli is a stabilizer.

        Values are the physical letter and its sign. For codes with mixed generators.
        """
        n = self.n
        sxz = self.sx_z if self.sx_z is not None else np.zeros_like(self.sx)
        logical_support = {j for p in self.logical_x + self.logical_z for j in p.support}
        out: dict[int, tuple[str, Any]] = {}
        for j in range(n):
            if j in logical_support:
                continue
            found = _frame_single(j, self.sx, sxz, self.sx_signs, self.sz, self.sz_signs, n)
            if found is None:
                continue
            letter, sign = found
            if self.hadamard[j]:
                letter, sign = {'X': ('Z', sign), 'Z': ('X', sign), 'Y': ('Y', _sum([sign, 1]))}[letter]
            out[j] = (letter, sign)
        return out

    def _restricted_mixed(self, columns: list[int]) -> 'StabilizerCode':
        """:meth:`restricted` for a code with mixed generators (see :attr:`sx_z`)."""
        n = self.n
        sxz = self.sx_z if self.sx_z is not None else np.zeros_like(self.sx)
        zero = np.zeros(n, dtype=np.uint8)
        gens = [(x.copy(), z.copy(), sg) for x, z, sg in zip(self.sx, sxz, self.sx_signs)]
        gens += [(zero.copy(), z.copy(), sg) for z, sg in zip(self.sz, self.sz_signs)]
        drop = [j for j in range(n) if j not in columns]
        singles: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, Any]] = {}
        for j in drop:
            e = np.zeros(n, dtype=np.uint8)
            e[j] = 1
            c = _combination(self.sx, e)
            if c is not None:
                x, z, sg = _combine(self.sx, sxz, self.sx_signs, c, n)
                single = _reduce_z_part(x, z, sg, self.sz, self.sz_signs, n, target=e)
                if single is not None:
                    singles[(j, 'x')] = single
            c = _combination(self.sz, e)
            if c is not None:
                singles[(j, 'z')] = (zero.copy(), e, _sum([sg for ci, sg in zip(c, self.sz_signs) if ci]))
        cleared = []
        for x, z, sg in gens:
            for j in drop:
                if x[j] and (j, 'x') in singles:
                    x, z, sg = _times(x, z, sg, *singles[(j, 'x')])
                if z[j] and (j, 'z') in singles:
                    x, z, sg = _times(x, z, sg, *singles[(j, 'z')])
            cleared.append((x[columns], z[columns], sg))
        m = len(columns)
        xt = [(x, z, sg) for x, z, sg in cleared if x.any()]
        zt = [(z, sg) for x, z, sg in cleared if not x.any() and z.any()]
        sx, sxz2, sxs, extra_z = _independent_mixed(as_matrix([t[0] for t in xt], m), as_matrix([t[1] for t in xt], m),
                                                    [t[2] for t in xt], m)
        zt += extra_z
        sz, szs = _min_weight_basis(*_independent(as_matrix([t[0] for t in zt], m), [t[1] for t in zt], m), m)
        sx, sxz2, sxs = _min_weight_basis_mixed(sx, sxz2, sxs, m, sz, szs)
        keep = set(columns)
        pairs = [(x, z) for x, z in zip(self.logical_x, self.logical_z)
                 if set(x.support) & keep or set(z.support) & keep]
        nc = frozenset(columns.index(j) for j in self.nonclifford_columns if j in keep)
        return StabilizerCode(
            qubits=tuple(self.qubits[j] for j in columns),
            hadamard=tuple(self.hadamard[j] for j in columns),
            sx=sx, sx_signs=sxs, sz=sz, sz_signs=szs,
            logical_x=[x.restricted(columns) for x, _ in pairs],
            logical_z=[z.restricted(columns) for _, z in pairs],
            non_clifford=bool(nc),
            positions=tuple(self.positions[j] for j in columns) if self.positions else None,
            nonclifford_columns=nc, sx_z=sxz2)

    def blocks(self) -> list['StabilizerCode']:
        """The data code split into independent blocks (connected components of the qubits)."""
        d = self.data()
        parent = list(range(d.n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for p in d.stabilizers + d.logical_x + d.logical_z:
            s = p.support
            for a in s[1:]:
                parent[find(a)] = find(s[0])
        groups: dict[int, list[int]] = {}
        for j in range(d.n):
            groups.setdefault(find(j), []).append(j)
        if len(groups) == 1:
            return [d]
        return [d.restricted(cols) for cols in sorted(groups.values())]

    # -- parameters -------------------------------------------------------------

    @property
    def distance(self) -> int | None:
        """The minimum weight of a logical operator not in the stabilizer group.

        Found by enumerating the normalizer, so only for small codes (``None`` if
        that would take more than about 2^22 elements, or if ``k = 0``).
        """
        if self._distance is None:
            self._distance = _distance(self.data())
        return self._distance if self._distance != -1 else None

    @property
    def parameters(self) -> tuple[int, int, int | None]:
        d = self.data()
        return d.n, d.k, self.distance

    def contains(self, pauli: 'PauliString | str') -> bool:
        """Whether a Pauli string (signs ignored) is in the stabilizer group."""
        p = PauliString.from_str(pauli) if isinstance(pauli, str) else pauli
        if p.n != self.n:
            raise ValueError("a {}-qubit string for a {}-qubit code".format(p.n, self.n))
        stab = self.symplectic()
        if not stab.size:
            return not any(p.x) and not any(p.z)
        r, piv, _ = rref(stab)
        return in_rowspace(p.symplectic, r, piv)

    def same_code_as(self, other: 'StabilizerCode', by: str = 'label') -> bool:
        """Same data qubits and the same stabilizer group, ignoring signs.

        ``by='label'`` identifies qubits by their label (the program's qubit
        index); ``by='position'`` by where they sit on the hardware, which is the
        right comparison across a permutation stage that moves the qubits.
        """
        a, b = self.data(), other.data()
        if by == 'position':
            if a.positions is None or b.positions is None:
                raise ValueError("positions are not known for both codes")
            ka, kb = list(a.positions), list(b.positions)
        elif by == 'label':
            ka, kb = list(a.qubits), list(b.qubits)
        else:
            raise ValueError("by must be 'label' or 'position'")
        if set(ka) != set(kb):
            return False
        order = [kb.index(q) for q in ka]
        bs = b.symplectic()
        bs = bs[:, order + [b.n + j for j in order]] if bs.size else bs
        ra, rb = rref(a.symplectic())[0], rref(bs)[0]
        return ra.shape == rb.shape and bool((ra == rb).all())

    # -- the ideal encoder ----------------------------------------------------------

    def encoder(self) -> BaseGraph:
        """The ideal encoder of this code: a ZX-diagram from ``k`` logical inputs to the qubits.

        It is the isometry ``E`` with ``E X_i = Xbar_i E`` and ``E Z_i = Zbar_i E``
        for the canonical logical operators, whose image is the code space with the
        stabilizer signs of this code (which may be Boolean functions of outcomes).
        Its adjoint is the ideal decoder. Up to a non-zero scalar.

        For a CSS code it is the two-layer diagram below; for any other stabilizer
        code it is read off its Choi state (:func:`_choi_encoder`).
        """
        if not self.is_css:
            return _choi_encoder(self)
        stabs = self.stabilizers
        xs = [(p, p.sign) for p in stabs if any(p.x)]
        zs = [(p, p.sign) for p in stabs if any(p.z)]
        n = self.n
        # output X-spider phases t: Z-stabilizer signs are t.z, logical Z acts without sign
        rows = [np.array(p.z, np.uint8) for p, _ in zs] + [np.array(p.z, np.uint8) for p in self.logical_z]
        rhs: list[Any] = [s for _, s in zs] + [0] * self.k
        t = _solve_symbolic(as_matrix(rows, n), rhs, n)
        g = Graph()
        reg = g.var_registry
        for s in rhs + [s for _, s in xs]:
            for v in getattr(s, 'free_vars', lambda: set())():
                reg.set_type(v.name, True)
        outs = []
        xsp = []
        for j in range(n):
            w = g.add_vertex(VertexType.X, j, 2, t[j])
            o = g.add_vertex(VertexType.BOUNDARY, j, 3)
            g.add_edge((w, o))
            xsp.append(w)
            outs.append(o)
        ins = []
        for i, p in enumerate(self.logical_x):
            b = g.add_vertex(VertexType.BOUNDARY, i, 0)
            z = g.add_vertex(VertexType.Z, i, 1)
            g.add_edge((b, z))
            ins.append(b)
            for j in p.support:
                g.add_edge((z, xsp[j]))
        for i, (p, s) in enumerate(xs):
            z = g.add_vertex(VertexType.Z, self.k + i, 1, s)
            for j in p.support:
                g.add_edge((z, xsp[j]))
        g.set_inputs(tuple(ins))
        g.set_outputs(tuple(outs))
        g.rebind_variables_to_registry()
        return g

    # -- display ----------------------------------------------------------------------

    def describe(self) -> str:
        lines = []
        fixed = self.fixed_qubits()
        for b in self.blocks():
            n, k, d = b.parameters
            kind = "CSS code" if b.is_css else "stabilizer code"
            where = ", ".join(str(q) for q in b.qubits)
            if b.positions is not None and tuple(b.positions) != tuple(b.qubits):
                where += " (at positions {})".format(", ".join(_fmt_pos(x) for x in b.positions))
            extra = " (holding a non-Clifford logical state)" if b.non_clifford else ""
            lines.append("[[{}, {}, {}]] {} on qubits {}{}".format(
                n, k, d if d is not None else '?', kind, where, extra))
            for p in b.stabilizers:
                lines.append("    stabilizer {}".format(p))
            for x, z in zip(b.logical_x, b.logical_z):
                lines.append("    logical X {}   logical Z {}".format(x.letters, z.letters))
        if fixed:
            lines.append("fixed qubits: " + ", ".join("{} in |{}>".format(q, s) for q, s in fixed.items()))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


# -- reading a code off a diagram ---------------------------------------------------

def code_of(g: BaseGraph, labels: Any = None, positions: Any = None,
            common_to_all_runs: bool = False) -> StabilizerCode:
    """The stabilizer code that ``g`` encodes into (question 1 of the paper).

    ``g`` maps its inputs (logical qubits) to its outputs (physical qubits);
    ``labels[j]`` names output ``j`` (default: its index). Symbolic Boolean phases
    (measurement outcomes) are allowed and become stabilizer signs. The code
    need not be CSS. Raises :class:`NotACodeError` if ``g`` is zero or not an
    isometry, :class:`OutcomeDependentCodeError` if the stabilizers themselves
    (not only their signs) depend on the outcomes, and ``NotImplementedError``
    for phase gadgets that the reduction leaves in the X basis.

    With ``common_to_all_runs`` the stabilizers that depend on the outcomes are
    left out instead of raising, so the result holds the Pauli strings that
    are stabilizers (up to sign) in every run; its logical operators and ``k``
    are then those of that smaller group. Used to check that a block stays in
    a code (:meth:`~zxqec.codes.LogicalAction.stays_in_code`).
    """
    res = reduce_gpf(g)
    if res.is_zero:
        raise NotACodeError("the diagram is the zero map (its detectors are inconsistent)")
    h = res.graph
    outs = list(h.outputs())
    n = len(outs)
    labels = tuple(labels) if labels is not None else tuple(range(n))
    if len(labels) != n:
        raise ValueError("{} labels for {} outputs".format(len(labels), n))
    out_col: dict[Any, int] = {}
    for j, b in enumerate(outs):
        (w,) = list(h.neighbors(b))
        out_col[w] = j
    in_spiders = {next(iter(h.neighbors(b))) for b in h.inputs()}

    rows_l: list[np.ndarray] = []
    rows_m: list[np.ndarray] = []
    signs_m: list[Any] = []
    zlayer: list[Any] = []          # Z-spiders of the layer, L rows first then M rows
    zl: list[tuple[Any, np.ndarray]] = []
    zm: list[tuple[Any, np.ndarray]] = []
    gadget_hubs = []
    for v in sorted(h.vertices()):
        ty = h.type(v)
        if ty == VertexType.BOUNDARY or is_leaf(h, v):
            continue
        if is_stem(h, v):
            if ty == VertexType.Z:
                raise NotImplementedError("code_of: a non-Clifford phase in the X basis "
                                          "(an X-leaf gadget) is not supported yet")
            gadget_hubs.append(v)
            continue
        if ty == VertexType.X:
            if v not in out_col:
                raise NotACodeError("the diagram is not an isometry: an X-spider without output "
                                    "(a post-selection) remains after reduction")
            continue
        row = np.zeros(n, dtype=np.uint8)
        for w in h.neighbors(v):
            if w in out_col:
                row[out_col[w]] = 1
        (zl if v in in_spiders else zm).append((v, row))
    for v, row in zl:
        rows_l.append(row)
        zlayer.append(v)
    for v, row in zm:
        rows_m.append(row)
        signs_m.append(h.phase(v))
        zlayer.append(v)
    m = len(rows_m)
    t = [h.phase(w) for w in sorted(out_col, key=lambda w: out_col[w])]
    hflags = tuple(int(has_boundary_hadamard(h, b)) for b in outs)

    all_rows = as_matrix(rows_l + rows_m, n)
    zstab = nullspace(all_rows)
    sz_signs = [_sum([ti for ti, zi in zip(t, z) if zi]) for z in zstab]
    sz, sz_signs = _min_weight_basis(*_independent(zstab, sz_signs, n), n)

    # which X-strings of the internal rows survive the phase gadgets (see .phasepoly), and the
    # Z part a Clifford gadget gives them
    mm = as_matrix(rows_m, n)
    combos, extra, dvecs, nonclifford, touched = _gadget_layer(h, zl, zm, zlayer, gadget_hubs, labels,
                                                               drop_run_dependent=common_to_all_runs)
    sx_rows = (combos @ mm) % 2 if m else np.zeros((0, n), np.uint8)
    sx_z, sx_signs = [], []
    for a, v, e, w in zip(combos, sx_rows, extra, dvecs):
        sx_z.append(w)
        sx_signs.append(_x_type_sign(a, v, w, e, signs_m, t, h.var_registry))
    if any(w.any() for w in sx_z):
        sx, sxz, sx_signs, more_z = _independent_mixed(sx_rows, as_matrix(sx_z, n), sx_signs, n)
        if more_z:
            zt = [(row, sg) for row, sg in zip(sz, sz_signs)] + more_z
            sz, sz_signs = _min_weight_basis(*_independent(as_matrix([r for r, _ in zt], n),
                                                           [sg for _, sg in zt], n), n)
        sx, sxz, sx_signs = _min_weight_basis_mixed(sx, sxz, sx_signs, n, sz, sz_signs)
    else:
        sx, sx_signs = _min_weight_basis(*_independent(sx_rows, sx_signs, n), n)
        sxz = None

    code = StabilizerCode(qubits=labels, hadamard=hflags, sx=sx, sx_signs=sx_signs, sz=sz,
                          sz_signs=sz_signs, logical_x=[], logical_z=[],
                          non_clifford=nonclifford,
                          positions=tuple(positions) if positions is not None else None,
                          nonclifford_columns=frozenset(touched), sx_z=sxz)
    stab = code.symplectic()
    lx, lz = _canonical_logicals(stab, n)
    code.logical_x = [PauliString.from_arrays(*_lightest(v, stab, n)) for v in lx]
    code.logical_z = [PauliString.from_arrays(*_lightest(v, stab, n)) for v in lz]
    return code


# -- helpers ------------------------------------------------------------------------------

def _gadget_layer(h: BaseGraph, zl: list, zm: list, zlayer: list, hubs: list, labels: Any,
                  cap: int = 14, drop_run_dependent: bool = False
                  ) -> tuple[np.ndarray, list[dict], list[np.ndarray], bool, set[int]]:
    """How the phase gadgets of a reduced diagram affect its code.

    Returns ``(combos, changes, zparts, non_clifford, columns)``:

    * ``combos``: a basis of the combinations ``a`` of internal Z-spiders whose
      X-string is part of a stabilizer; for each, the part of the phase change
      that does not depend on the bits (a constant and outcome terms, as a
      phase-polynomial dictionary), and the stabilizer's Z part ``w`` on the
      outputs;
    * ``non_clifford``: whether the gadgets add non-Clifford content, i.e.
      flipping some Z-spider bits changes their total phase by something that is
      not a Pauli (the "surviving stabilizers" proposition of the paper);
    * ``columns``: the qubits reached by such non-Clifford gadgets.

    Flipping the bits ``a`` changes the gadgets' total phase by
    :func:`~zxqec.codes.phasepoly.delta`, which depends on ``a`` only through
    which gadgets see ``a`` an odd number of times. So the combinations no gadget
    sees oddly survive with no extra sign, and only one representative per
    pattern of gadgets has to be tried: at most ``2^r`` for ``r`` independent
    gadget parities. Each representative falls in one of three cases:

    * the change does not depend on any bit: a stabilizer, with the change as sign;
    * it depends on the bits only through a Pauli ``Z`` string (every term has
      one bit and coefficient ``pi``), ``d . y`` say: if some ``w`` has
      ``w . (y A) = d . y`` for all bits, the X-string is part of the stabilizer
      ``Z^w X^{aM}``, which mixes ``X`` and ``Z`` (the code is not CSS). If no such
      ``w`` exists, the change depends on bits the outputs cannot see, e.g. a
      logical input, and the X-string is a (Clifford) logical operator. If a
      term also contains an outcome, the mixing happens in some runs only and
      :class:`OutcomeDependentCodeError` is raised;
    * anything else: the X-string is broken by non-Clifford phases, and the
      code holds non-Clifford logical content.

    Raises ``NotImplementedError`` if there are more than ``cap`` independent
    gadget parities to try.
    """
    from .gf2 import complement_basis
    from .phasepoly import Gadget, affine_parts, delta, to_pyzx
    k, m = len(zl), len(zm)
    n = len(labels)
    if not hubs:
        return np.eye(m, dtype=np.uint8), [{}] * m, [np.zeros(n, np.uint8)] * m, False, set()
    layer_rows = as_matrix([row for _v, row in zl] + [row for _v, row in zm], n)
    index = {v: i for i, v in enumerate(zlayer)}          # L rows first, then M rows
    gadgets = []
    for hub in hubs:
        leaves = [w for w in h.neighbors(hub) if is_leaf(h, w)]
        angle = sum((h.phase(w) for w in leaves), Fraction(0))
        if hasattr(angle, 'free_vars'):
            raise NotImplementedError("code_of: a gadget with a symbolic angle")
        sign_vars, const = affine_parts(h.phase(hub))
        bits = [('z', index[w]) for w in h.neighbors(hub) if w in index]
        gadgets.append(Gadget(Fraction(angle), bits, sign_vars, const))
    incidence = as_matrix([[1 if ('z', i) in g.bits else 0 for i in range(k + m)] for g in gadgets], k + m)

    def flipped(vec: np.ndarray, offset: int) -> set:
        return {('z', offset + i) for i, b in enumerate(vec) if b}

    def z_terms(d: dict) -> list:
        return [mono for mono in d if any(isinstance(v, tuple) for v in mono)]

    def pauli_only(terms: list, d: dict) -> bool:
        return all(sum(isinstance(v, tuple) for v in mono) == 1 and d[mono] == 1 for mono in terms)

    def representatives(g_part: np.ndarray, dim: int) -> tuple[np.ndarray, list[np.ndarray]]:
        kernel = nullspace(g_part) if g_part.size else np.eye(dim, dtype=np.uint8)
        comp = complement_basis(kernel, np.eye(dim, dtype=np.uint8))
        if comp.shape[0] > cap:
            raise NotImplementedError(
                "code_of: {} independent gadget parities; deciding which stabilizers survive "
                "them is limited to {}".format(comp.shape[0], cap))
        reps = []
        for mask in range(1, 1 << comp.shape[0]):
            v = np.zeros(dim, dtype=np.uint8)
            for i in range(comp.shape[0]):
                if mask >> i & 1:
                    v ^= comp[i]
            reps.append(v)
        return kernel, reps

    # which X-strings of the internal rows survive
    nonclifford = False
    if m:
        kernel, reps = representatives(incidence[:, k:], m)
        passing, changes, zparts = [], [], []
        for a in reps:
            d = delta(gadgets, flipped(a, k))
            terms = z_terms(d)
            if terms and not pauli_only(terms, d):
                nonclifford = True
                continue
            run_dependent = (any(not isinstance(v, tuple) for mono in terms for v in mono)
                             or any(c % 1 for mono, c in d.items() if mono and mono not in terms))
            if run_dependent and drop_run_dependent:
                continue                           # a stabilizer in some runs only: not common to all
            if run_dependent:
                support = sorted({int(j) for i, b in enumerate(a) if b for j in np.flatnonzero(zm[i][1])})
                raise OutcomeDependentCodeError(
                    "a Clifford phase controlled by measurement outcomes gives the X stabilizer on qubits "
                    "{} a Z part (or a factor i) in some runs only, so the code differs between runs; "
                    "reading it run by run is not implemented".format(", ".join(str(labels[j]) for j in support)))
            dv = np.zeros(k + m, dtype=np.uint8)
            for mono in terms:
                (bit,) = mono
                dv[bit[1]] ^= 1
            w = solve(layer_rows.copy(), dv) if dv.any() else np.zeros(n, np.uint8)
            if w is None:
                continue                           # the change reaches bits the outputs cannot see
            passing.append(a)
            changes.append({mono: c for mono, c in d.items() if mono not in terms})
            zparts.append(w.astype(np.uint8))
        combos = [row for row in kernel]
        extra: list[dict] = [{} for _ in combos]
        dvecs: list[np.ndarray] = [np.zeros(n, np.uint8) for _ in combos]
        if passing:
            basis = rref(as_matrix(passing, m))[0]     # the passing set is a subspace
            for row in basis:
                idx = next(i for i, a in enumerate(passing) if (a == row).all())
                combos.append(passing[idx])
                extra.append(changes[idx])
                dvecs.append(zparts[idx])
        combo_matrix = as_matrix(combos, m)
    else:
        combo_matrix = np.zeros((0, 0), np.uint8)
        extra, dvecs = [], []

    # non-Clifford logical content: some flip, of any rows, changes the phase by a non-Pauli
    if not nonclifford:
        try:
            _kernel, reps = representatives(incidence, k + m)
            for b in reps:
                d = delta(gadgets, flipped(b, 0))
                terms = z_terms(d)
                if terms and not pauli_only(terms, d):
                    nonclifford = True
                    break
        except NotImplementedError:
            nonclifford = any(g.angle % Fraction(1, 2) != 0 for g in gadgets)

    columns: set[int] = set()
    if nonclifford:
        rows = [row for _v, row in zl] + [row for _v, row in zm]
        for g in gadgets:
            if g.angle % Fraction(1, 2) != 0:
                for _z, i in g.bits:
                    columns |= {int(j) for j in np.flatnonzero(rows[i])}
    return combo_matrix, extra, dvecs, nonclifford, columns


def _x_type_sign(a: np.ndarray, v: np.ndarray, w: np.ndarray, change: dict, signs_m: list[Any],
                 t: list[Any], registry: Any) -> Any:
    """The sign of the stabilizer ``(-1)^sign L(v, w)`` read for the combination ``a``.

    Flipping the bits ``a`` multiplies the image by ``exp(i pi (s.a + c + d.y))``,
    where ``s`` are the Z-spider phases, ``c`` the bit-free part of the gadgets'
    change and ``d.y = w . (output bits)``. On outputs ``u = yA + t`` that is
    ``X^v E = exp(i pi (s.a + c + w.t)) Z^w E``, so ``Z^w X^v`` fixes ``E`` up to that
    phase, and ``L(v, w) = i^(-w.v) Z^w X^v`` gives the sign
    ``w.v/2 - c + s.a + w.t`` (mod 2). The first two terms together are an integer.
    """
    const = change.get(frozenset(), Fraction(0))
    whole = Fraction(int((v & w).sum()), 2) - const
    if whole.denominator != 1:
        raise RuntimeError("code_of: a stabilizer with a phase that is not +-1; this is a bug")
    parts: list[Any] = [int(whole) % 2]
    parts += [sg for c, sg in zip(a, signs_m) if c]
    parts += [tj for tj, wj in zip(t, w) if wj]
    rest = {mono: c for mono, c in change.items() if mono}
    if rest:
        if any(c % 1 for c in rest.values()):
            raise OutcomeDependentCodeError("a stabilizer's phase depends on the outcomes by a factor of i; "
                                            "the code differs between runs")
        from .phasepoly import to_pyzx
        parts.append(to_pyzx(rest, registry))
    return _sum(parts)


def _independent(rows: np.ndarray, signs: list[Any], n: int) -> tuple[np.ndarray, list[Any]]:
    """RREF of ``rows`` with the signs carried along (products of same-type strings)."""
    if rows.size == 0:
        return np.zeros((0, n), np.uint8), []
    r, _, tr = rref(rows, track=True)
    return r, [_sum([s for c, s in zip(tv, signs) if c]) for tv in tr]


def _min_weight_basis(rows: np.ndarray, signs: list[Any], n: int,
                      cap: int = 16) -> tuple[np.ndarray, list[Any]]:
    """A basis of the row space made of lightest elements, chosen greedily.

    The elements of the space are sorted by weight, then by the number of
    outcomes their sign depends on (a stabilizer the program measured has a
    single outcome as its sign), then by support, and taken whenever they are
    independent of those already taken. This gives a basis of minimum total
    weight (the greedy algorithm is exact for linear matroids) that depends only
    on the space and its signs. Signs follow the combinations. Spaces of dimension above ``cap`` keep their RREF basis.
    """
    r = rows.shape[0]
    if r == 0 or r > cap:
        return rows, signs
    elements = []
    for bits in range(1, 1 << r):
        c = [(bits >> i) & 1 for i in range(r)]
        v = np.zeros(n, dtype=np.uint8)
        for ci, row in zip(c, rows):
            if ci:
                v ^= row
        sg = _sum([sg for ci, sg in zip(c, signs) if ci])
        nvars = len(sg.free_vars()) if hasattr(sg, 'free_vars') else 0
        elements.append((int(v.sum()), nvars, tuple(int(j) for j in np.flatnonzero(v)), v, sg))
    elements.sort(key=lambda e: (e[0], e[1], e[2]))
    picked: list[np.ndarray] = []
    picked_signs: list[Any] = []
    for _w, _nv, _sup, v, sg in elements:
        trial = as_matrix(picked + [v], n)
        if rref(trial)[0].shape[0] > len(picked):
            picked.append(v)
            picked_signs.append(sg)
            if len(picked) == r:
                break
    return as_matrix(picked, n), picked_signs


def _lightest(v: np.ndarray, stab: np.ndarray, n: int, cap: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """The lightest element of the coset ``v + <stabilizers of the same pure type>``.

    Returns ``(x, z)``. For a logical operator this is a canonical, readable
    representative (e.g. a boundary-to-boundary string of a surface code). Mixed
    operators, or too many stabilizers to enumerate, are returned unchanged.
    """
    x, z = v[:n].copy(), v[n:].copy()
    if x.any() and z.any():
        return x, z
    kind = 'X' if x.any() else 'Z'
    base = x if kind == 'X' else z
    pure = _pure(stab, n, kind)
    if pure.shape[0] > cap:
        return x, z
    best = (int(base.sum()), tuple(int(j) for j in np.flatnonzero(base)), base)
    for bits in range(1, 1 << pure.shape[0]):
        w = base.copy()
        for i in range(pure.shape[0]):
            if bits >> i & 1:
                w ^= pure[i]
        key = (int(w.sum()), tuple(int(j) for j in np.flatnonzero(w)))
        if key < best[:2]:
            best = (key[0], key[1], w)
    zero = np.zeros(n, dtype=np.uint8)
    return (best[2], zero) if kind == 'X' else (zero, best[2])


# -- Pauli products, for generators that mix X and Z ------------------------------------------

def _times(x1: np.ndarray, z1: np.ndarray, s1: Any,
           x2: np.ndarray, z2: np.ndarray, s2: Any) -> tuple[np.ndarray, np.ndarray, Any]:
    """The product of two commuting Hermitian Pauli strings ``(-1)^s L(x, z)``.

    ``L(x, z)`` has letter ``X``, ``Y``, ``Z`` where ``(x_j, z_j)`` is ``(1,0)``,
    ``(1,1)``, ``(0,1)``; since ``Y = -i Z X``, ``L(x, z) = i^(-x.z) Z^z X^x``, and
    ``L1 L2 = i^e L3`` with ``e = x3.z3 - x1.z1 - x2.z2 + 2 x1.z2``, which is even
    for commuting strings.
    """
    x3, z3 = x1 ^ x2, z1 ^ z2
    e = (int((x3 & z3).sum()) - int((x1 & z1).sum()) - int((x2 & z2).sum())
         + 2 * int((x1 & z2).sum())) % 4
    if e % 2:
        raise ValueError("the Pauli strings anticommute")
    return x3, z3, _sum([s1, s2, e // 2])


def _combine(xs: np.ndarray, zs: np.ndarray, signs: list[Any], coeffs: Any,
             n: int) -> tuple[np.ndarray, np.ndarray, Any]:
    """The ordered product of the generators ``(xs[i], zs[i], signs[i])`` with ``coeffs[i] = 1``."""
    x, z, sg = np.zeros(n, np.uint8), np.zeros(n, np.uint8), 0
    for c, xr, zr, si in zip(coeffs, xs, zs, signs):
        if c:
            x, z, sg = _times(x, z, sg, xr.astype(np.uint8), zr.astype(np.uint8), si)
    return x, z, sg


def _reduce_z_part(x: np.ndarray, z: np.ndarray, sign: Any, sz: np.ndarray, sz_signs: list[Any], n: int,
                   target: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, Any] | None:
    """Multiply ``(x, z, sign)`` by Z-type stabilizers.

    With ``target``: make the Z part ``0`` or equal to ``target``, returning ``None``
    if neither is possible. Without: make it as light as possible (fewest qubits
    touched, then fewest ``Z`` s, then first support), enumerating up to ``2^12``
    combinations and otherwise only reducing it modulo their span.
    """
    zero = np.zeros(n, dtype=np.uint8)
    if target is not None:
        for goal in (zero, target):
            c = _combination(sz, z ^ goal) if (z ^ goal).any() else np.zeros(len(sz), np.uint8)
            if c is not None:
                u = (c @ sz) % 2 if len(sz) else zero
                return _times(x, z, sign, zero, u.astype(np.uint8), _sum([t for ci, t in zip(c, sz_signs) if ci]))
        return None
    if not z.any() or not len(sz):
        return x, z, sign
    if len(sz) > 12:
        r, piv, tr = rref(sz, track=True)
        zr = reduce_mod(z, r, piv)
        c = _combination(sz, z ^ zr)
        if c is None:
            return x, z, sign
        return _times(x, z, sign, zero, (z ^ zr).astype(np.uint8), _sum([t for ci, t in zip(c, sz_signs) if ci]))
    best: tuple[Any, list[int], np.ndarray] | None = None
    for bits in range(1 << len(sz)):
        pick = [(bits >> i) & 1 for i in range(len(sz))]
        u = np.zeros(n, dtype=np.uint8)
        for ci, row in zip(pick, sz):
            if ci:
                u ^= row
        z2 = z ^ u
        key = (int((x | z2).sum()), int(z2.sum()), tuple(int(j) for j in np.flatnonzero(x | z2)))
        if best is None or key < best[0]:
            best = (key, pick, u)
    assert best is not None
    _key, pick, u = best
    return _times(x, z, sign, zero, u, _sum([t for ci, t in zip(pick, sz_signs) if ci]))


def _frame_single(j: int, sx: np.ndarray, sxz: np.ndarray, sx_signs: list[Any], sz: np.ndarray,
                  sz_signs: list[Any], n: int) -> tuple[str, Any] | None:
    """The single-qubit stabilizer on qubit ``j`` in the frame (letter and sign), if there is one."""
    e = np.zeros(n, dtype=np.uint8)
    e[j] = 1
    c = _combination(sx, e)
    if c is not None:
        x, z, sg = _combine(sx, sxz, sx_signs, c, n)
        single = _reduce_z_part(x, z, sg, sz, sz_signs, n, target=e)
        if single is not None:
            return ('Y' if single[1].any() else 'X'), single[2]
    c = _combination(sz, e)
    if c is not None:
        return 'Z', _sum([sg for ci, sg in zip(c, sz_signs) if ci])
    return None


def _independent_mixed(xs: np.ndarray, zs: np.ndarray, signs: list[Any],
                       n: int) -> tuple[np.ndarray, np.ndarray, list[Any], list[tuple[np.ndarray, Any]]]:
    """Generators with independent X parts, as products of the given ones.

    A product whose X part cancels is a Z-type stabilizer; those are returned
    separately (Z part and sign) instead of being lost.
    """
    if xs.size == 0:
        return np.zeros((0, n), np.uint8), np.zeros((0, n), np.uint8), [], []
    r, _, tr = rref(xs, track=True)
    out_x, out_z, out_s = [], [], []
    for tv in tr:
        x, z, sg = _combine(xs, zs, signs, tv, n)
        out_x.append(x)
        out_z.append(z)
        out_s.append(sg)
    # the combinations that cancel the X part: the null space of the X parts
    extra = []
    for c in nullspace(xs.T.copy()) if xs.shape[0] > r.shape[0] else []:
        x, z, sg = _combine(xs, zs, signs, c, n)
        if z.any():
            extra.append((z, sg))
    return as_matrix(out_x, n), as_matrix(out_z, n), out_s, extra


def _choi_encoder(code: 'StabilizerCode') -> BaseGraph:
    """The ideal encoder of any stabilizer code, read off its Choi state.

    ``sum_b |b> (x) E|b>`` is the stabilizer state on ``k + n`` qubits with the
    generators ``I (x) g`` for the stabilizers ``g`` of the code (with their
    signs) and ``X_i (x) Xbar_i``, ``Z_i (x) Zbar_i`` for its canonical logical
    operators: ``X_i (x) Xbar_i`` fixes it exactly when ``Xbar_i E = E X_i``, and
    likewise for ``Z``. These ``n + k`` generators are independent and commute,
    so they fix the state up to a scalar. :func:`_stabilizer_state` draws it,
    and its first ``k`` outputs become the inputs, which reads the same tensor
    as a map: the route of Yeh et al. (Section 4.1, "un-Choi"). Up to a
    non-zero scalar.
    """
    n, k = code.n, code.k
    m = n + k
    xs: list[list[int]] = []
    zs: list[list[int]] = []
    signs: list[Any] = []
    none = [0] * k
    for p in code.stabilizers:
        xs.append(none + list(p.x))
        zs.append(none + list(p.z))
        signs.append(p.sign)
    for i, (lx, lz) in enumerate(zip(code.logical_x, code.logical_z)):
        e = [int(j == i) for j in range(k)]
        xs.append(e + list(lx.x))
        zs.append(none + list(lx.z))
        signs.append(lx.sign)
        xs.append(none + list(lz.x))
        zs.append(e + list(lz.z))
        signs.append(lz.sign)
    g = _stabilizer_state(as_matrix(xs, m), as_matrix(zs, m), signs, m)
    bounds = list(g.outputs())
    for i, b in enumerate(bounds[:k]):
        g.set_qubit(b, i)
    for j, b in enumerate(bounds[k:]):
        g.set_qubit(b, j)
    g.set_inputs(tuple(bounds[:k]))
    g.set_outputs(tuple(bounds[k:]))
    return g


def _stabilizer_state(xs: np.ndarray, zs: np.ndarray, signs: list[Any], m: int) -> BaseGraph:
    """A ZX-diagram, up to a scalar, of the state fixed by ``m`` independent commuting generators.

    Generator ``j`` is ``(-1)^signs[j] L(xs[j], zs[j])`` as in :func:`_times`; the
    signs may be Boolean polynomials. The state is
    ``D sum_y (-1)^(rho . y) |y R + t>``, where the rows ``v_j`` of ``R`` are the
    independent X parts of generators ``(-1)^s_j L(v_j, w_j)``, ``t`` gives the
    Z-type generators ``(-1)^s Z^w`` their signs (``w . t = s``), and ``D`` is the
    diagonal Clifford ``|x> -> i^(a . x) (-1)^(sum_{q<p} B_qp x_q x_p) |x>`` (``S``
    gates and ``CZ`` s) with ``Gamma = diag(a) + B`` symmetric and ``R Gamma = W``:
    the factorisation of a stabilizer tableau into an inner CSS code and a
    diagonal Clifford of Yeh, Huang, Kissinger, Li and van de Wetering (*A
    three-way normal form for stabiliser codes*, Theorem 6.1 and Proposition
    6.2), with the signs added. Generator ``j`` shifts ``y_j`` by one and
    multiplies the term of ``x`` by ``i^(a.v_j - v_j.w_j) (-1)^(Q(v_j) + rho_j)``
    over its own phase (``Gamma v_j = w_j`` cancels the parts that depend on
    ``x``), and ``rho_j`` makes the factor 1. This is not their reduced or
    isometry normal form, which is unique; the logical action does not need one.
    """
    vx, vz, vs, pure_z = _independent_mixed(xs, zs, signs, m)
    r = vx.shape[0]
    if r + len(pure_z) != m:
        raise ValueError("not a full set of independent stabilizer generators")
    t = _solve_symbolic(as_matrix([z for z, _ in pure_z], m), [s for _, s in pure_z], m)
    gamma = np.zeros((m, m), dtype=np.int64)
    if r:
        rinv = np.zeros((m, r), dtype=np.int64)          # R rinv = I
        for col in range(r):
            unit = np.zeros(r, dtype=np.uint8)
            unit[col] = 1
            u = solve(vx, unit)
            if u is None:
                raise ValueError("the X parts are not independent")
            rinv[:, col] = u
        rv, rw = vx.astype(np.int64), vz.astype(np.int64)
        if not ((rv @ rw.T) % 2 == (rw @ rv.T) % 2).all():
            raise ValueError("the generators do not commute")
        # Yeh et al., Proposition 6.2: G = R N + N^T R^T + R H N^T R^T is symmetric and H G = N,
        # for H the X parts, N the Z parts and R a right inverse of H (the tableau normal form
        # [H | H G ; 0 | H_Z] of their Theorem 6.1)
        gamma = (rinv @ rw + rw.T @ rinv.T + rinv @ rv @ rw.T @ rinv.T) % 2
        assert ((rv @ gamma) % 2 == rw).all() and (gamma == gamma.T).all()
    a = np.diag(gamma).copy()
    b = np.triu(gamma, 1)
    rho = []
    for v, w, s in zip(vx.astype(np.int64), vz.astype(np.int64), vs):
        av, vw = int(a @ v), int(v @ w)
        assert (av - vw) % 2 == 0
        rho.append(_sum([s, ((av - vw) % 4) // 2, int(v @ b @ v) % 2]))
    g = Graph()
    reg = g.var_registry
    for ph in list(t) + rho:
        for var in getattr(ph, 'free_vars', lambda: set())():
            reg.set_type(var.name, True)
    xsp, last, outs = [], [], []
    for q in range(m):
        x = g.add_vertex(VertexType.X, q, 2, t[q])
        end = x
        if a[q] or (gamma[q].sum() - gamma[q, q]):
            end = g.add_vertex(VertexType.Z, q, 3, Fraction(int(a[q]), 2))
            g.add_edge((x, end))
        o = g.add_vertex(VertexType.BOUNDARY, q, 4)
        g.add_edge((end, o))
        xsp.append(x)
        last.append(end)
        outs.append(o)
    for q in range(m):
        for p in range(q + 1, m):
            if b[q, p]:
                g.add_edge((last[q], last[p]), EdgeType.HADAMARD)
    for j, (v, ph) in enumerate(zip(vx, rho)):
        z = g.add_vertex(VertexType.Z, j, 1, ph)
        for q in np.flatnonzero(v):
            g.add_edge((z, xsp[int(q)]))
    g.set_outputs(tuple(outs))
    g.rebind_variables_to_registry()
    return g


def _min_weight_basis_mixed(xs: np.ndarray, zs: np.ndarray, signs: list[Any], n: int, sz: np.ndarray,
                            sz_signs: list[Any], cap: int = 16) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """:func:`_min_weight_basis` for X-type generators with Z parts.

    Each element of the span is taken with its lightest Z part (modulo the
    Z-type stabilizers ``sz``); weight counts the qubits an element touches.
    Ties go to fewer outcomes in the sign, then fewer ``Y`` letters (so the
    ``[[5,1,3]]`` code shows ``XZZXI`` and its cyclic shifts), then support.
    Elements are picked greedily while their X parts stay independent.
    """
    r = xs.shape[0]
    if r == 0:
        return xs, zs, signs
    if r > cap:
        out = [_reduce_z_part(x, z, sg, sz, sz_signs, n) for x, z, sg in zip(xs, zs, signs)]
        return (as_matrix([o[0] for o in out if o], n), as_matrix([o[1] for o in out if o], n),
                [o[2] for o in out if o])
    elements = []
    for bits in range(1, 1 << r):
        c = [(bits >> i) & 1 for i in range(r)]
        prod = _reduce_z_part(*_combine(xs, zs, signs, c, n), sz, sz_signs, n)
        assert prod is not None
        x, z, sg = prod
        nvars = len(sg.free_vars()) if hasattr(sg, 'free_vars') else 0
        elements.append((int((x | z).sum()), nvars, int((x & z).sum()),
                         tuple(int(j) for j in np.flatnonzero(x | z)), int(z.sum()), x, z, sg))
    elements.sort(key=lambda e: e[:5])
    px: list[np.ndarray] = []
    pz: list[np.ndarray] = []
    ps: list[Any] = []
    for _w, _nv, _ny, _sup, _nz, x, z, sg in elements:
        if rref(as_matrix(px + [x], n))[0].shape[0] > len(px):
            px.append(x)
            pz.append(z)
            ps.append(sg)
            if len(px) == r:
                break
    return as_matrix(px, n), as_matrix(pz, n), ps


def _nonzero_on(gens: np.ndarray, signs: list[Any], cols: list[int]) -> tuple[np.ndarray, list[Any]]:
    if gens.size == 0:
        return gens, signs
    keep = [i for i in range(gens.shape[0]) if gens[i, cols].any()]
    return gens[keep], [signs[i] for i in keep]


def _combination(gens: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Coefficients ``c`` with ``c @ gens = target``, or ``None``."""
    if gens.size == 0:
        return None
    return solve(gens.T.copy(), target)


def _clear_columns(gens: np.ndarray, signs: list[Any], drop: list[int]) -> tuple[np.ndarray, list[Any]]:
    """Clear ``drop`` columns by multiplying with single-column generators; drop those."""
    if gens.size == 0:
        return gens, signs
    n = gens.shape[1]
    r, _, tr = rref(gens, track=True)
    s = [_sum([x for c, x in zip(tv, signs) if c]) for tv in tr]
    singles = {}
    for j in drop:
        e = np.zeros(n, dtype=np.uint8)
        e[j] = 1
        c = _combination(r, e)
        if c is not None:
            singles[j] = (e, _sum([x for cc, x in zip(c, s) if cc]))
    rows, out_signs = [], []
    for row, sg in zip(r, s):
        row = row.copy()
        for j, (e, es) in singles.items():
            if row[j]:
                row ^= e
                sg = _sum([sg, es])
        if row.any():
            rows.append(row)
            out_signs.append(sg)
    return _min_weight_basis(*_independent(as_matrix(rows, n), out_signs, n), n)


def _canonical_logicals(stab: np.ndarray, n: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Canonical logical operators (rows ``(x|z)``) for the stabilizer group ``stab``.

    Pure ``X`` representatives are used when the code has them (always, for a CSS
    code), reduced modulo the pure-``X`` stabilizers; the ``Z`` partners are then
    the dual basis among pure ``Z`` strings, reduced modulo the pure-``Z``
    stabilizers. Otherwise a symplectic Gram--Schmidt on a canonical basis of the
    normalizer modulo the stabilizers. Either way the result depends only on the
    group.
    """
    sx_part, sz_part = stab[:, :n], stab[:, n:]
    normalizer = nullspace(np.concatenate([sz_part, sx_part], axis=1)) if stab.size else np.eye(2 * n, dtype=np.uint8)
    k2 = normalizer.shape[0] - (rref(stab)[0].shape[0] if stab.size else 0)
    k = k2 // 2
    if k == 0:
        return [], []
    # pure X: x with sz_part @ x = 0; pure-X stabilizers: combinations with zero z-part
    wx = nullspace(sz_part) if stab.size else np.eye(n, dtype=np.uint8)
    sxp = _pure(stab, n, 'X')
    lx = complement_basis(sxp, wx) if wx.size else np.zeros((0, n), np.uint8)
    if lx.shape[0] == k:
        wz = nullspace(sx_part) if stab.size else np.eye(n, dtype=np.uint8)
        szp = _pure(stab, n, 'Z')
        rz, pz, _ = rref(szp) if szp.size else (np.zeros((0, n), np.uint8), [], None)
        a = (lx @ wz.T) % 2
        lz = []
        for i in range(k):
            e = np.zeros(k, dtype=np.uint8)
            e[i] = 1
            y = solve(a, e)
            if y is None:
                break
            z = reduce_mod((y @ wz) % 2, rz, pz)
            lz.append(z)
        if len(lz) == k:
            zero = np.zeros(n, dtype=np.uint8)
            return ([np.concatenate([x, zero]) for x in lx], [np.concatenate([zero, z]) for z in lz])
    # general case: symplectic Gram--Schmidt
    basis = [v for v in complement_basis(stab, normalizer)]
    xs, zs = [], []
    while basis:
        v = basis.pop(0)
        at = next((i for i, w in enumerate(basis) if _omega(v, w, n)), None)
        if at is None:
            continue
        partner = basis.pop(at)
        basis = [(u + _omega(u, partner, n) * v + _omega(u, v, n) * partner) % 2 for u in basis]
        xs.append(v)
        zs.append(partner)
    return xs, zs


def _pure(stab: np.ndarray, n: int, kind: str) -> np.ndarray:
    """The stabilizers (as n-bit rows) that are pure ``kind`` strings."""
    if not stab.size:
        return np.zeros((0, n), np.uint8)
    other = stab[:, n:] if kind == 'X' else stab[:, :n]
    mine = stab[:, :n] if kind == 'X' else stab[:, n:]
    c = nullspace(other.T.copy())
    return rref((c @ mine) % 2)[0] if c.size else np.zeros((0, n), np.uint8)


def _omega(u: np.ndarray, v: np.ndarray, n: int) -> int:
    return int((u[:n] @ v[n:] + u[n:] @ v[:n]) % 2)


def _solve_symbolic(a: np.ndarray, rhs: list[Any], n: int) -> list[Any]:
    """A vector ``t`` of phases with ``a @ t = rhs``, where the right-hand side may be symbolic."""
    if a.size == 0:
        return [0] * n
    r, pivots, tr = rref(a, track=True)
    t: list[Any] = [0] * n
    for row, p, tv in zip(r, pivots, tr):
        t[p] = _sum([s for c, s in zip(tv, rhs) if c])
    return t


def _fmt_pos(x: Any) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _state_label(pauli: str, sign: Any) -> str:
    const = sign if isinstance(sign, int) else None
    if const is None and hasattr(sign, 'free_vars') and not sign.free_vars():
        const = int(sum(c for c, _ in sign.terms)) % 2 if sign.terms else 0
    if const is not None:
        return {'Z': ('0', '1'), 'X': ('+', '-'), 'Y': ('+i', '-i')}[pauli][const % 2]
    if pauli == 'Y':
        return '(-1)^({}) i'.format(sign)
    return '{}' .format(sign) if pauli == 'Z' else '(-1)^({})'.format(sign)


def _distance(code: StabilizerCode, cap: int = 22) -> int:
    """Brute-force distance; -1 when too large to enumerate or ``k = 0``."""
    if code.k == 0:
        return -1
    n = code.n
    stab = code.symplectic()
    rs, ps, _ = rref(stab) if stab.size else (np.zeros((0, 2 * n), np.uint8), [], None)
    if code.is_css:
        best = n + 1
        for kind in ('X', 'Z'):
            other = stab[:, n:] if kind == 'X' else stab[:, :n]
            w = nullspace(other) if stab.size else np.eye(n, dtype=np.uint8)
            if w.shape[0] > cap:
                return -1
            for bits in range(1, 1 << w.shape[0]):
                v = np.zeros(n, dtype=np.uint8)
                for i in range(w.shape[0]):
                    if bits >> i & 1:
                        v ^= w[i]
                full = np.concatenate([v, np.zeros(n, np.uint8)]) if kind == 'X' else np.concatenate([np.zeros(n, np.uint8), v])
                if in_rowspace(full, rs, ps):
                    continue
                best = min(best, int(v.sum()))
        return best
    norm = nullspace(np.concatenate([stab[:, n:], stab[:, :n]], axis=1))
    if norm.shape[0] > cap:
        return -1
    best = n + 1
    for bits in range(1, 1 << norm.shape[0]):
        v = np.zeros(2 * n, dtype=np.uint8)
        for i in range(norm.shape[0]):
            if bits >> i & 1:
                v ^= norm[i]
        if in_rowspace(v, rs, ps):
            continue
        best = min(best, int(((v[:n] | v[n:]) > 0).sum()))
    return best
