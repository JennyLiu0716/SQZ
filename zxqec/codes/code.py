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

:func:`code_layers` returns what is read: the reduced diagram and its rows
(the input Z-spiders' rows ``L``, the internal ones' rows ``M``), with names for
the spiders. :func:`stabilizes` checks a Pauli string on a diagram itself, by
comparing tensors.

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

from ..gpf import GPFResult, has_boundary_hadamard, is_leaf, is_stem, reduce_gpf
from .pauli import phase_sum as _sum
from .gf2 import as_matrix, complement_basis, in_rowspace, nullspace, reduce_mod, rref, solve
from .pauli import PauliString

__all__ = ['StabilizerCode', 'NotACodeError', 'OutcomeDependentCodeError', 'code_of', 'CodeLayers', 'code_layers',
           'stabilizes']


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

    def separated(self, parts: list[list[Any]]) -> 'StabilizerCode':
        """The code read as separate codes on ``parts`` (lists of qubit labels covering every qubit).

        Only the stabilizers that act within a single part are kept, with their
        signs; the logical operators are chosen again from what is left. After a
        lattice-surgery split, the two patches still share a stabilizer (the
        product of their logical ``Z``), so the code read off the program has one
        logical qubit; treating the patches as separate codes from then on, as
        the logical description of the split does, is this code. CSS codes only.
        """
        if self._mixed:
            raise NotImplementedError("separated: a code that is not CSS")
        index = {q: j for j, q in enumerate(self.qubits)}
        part_of: dict[int, int] = {}
        for i, part in enumerate(parts):
            for q in part:
                if q not in index or index[q] in part_of:
                    raise ValueError("qubit {!r} is not in the code, or in two parts".format(q))
                part_of[index[q]] = i
        n = self.n
        if len(part_of) != n:
            raise ValueError("the parts leave out qubits {}".format(
                [q for j, q in enumerate(self.qubits) if j not in part_of]))

        def within(rows: np.ndarray, signs: list[Any]) -> tuple[np.ndarray, list[Any]]:
            kept, kept_signs = [], []
            for i in range(len(parts)):
                outside = [j for j in range(n) if part_of[j] != i]
                if not rows.size:
                    continue
                if not outside:                                # one part: every stabilizer is within it
                    kept.extend(rows)
                    kept_signs.extend(signs)
                    continue
                for c in nullspace(rows[:, outside].T):     # combinations that vanish outside part i
                    kept.append((c @ rows) % 2)
                    kept_signs.append(_sum([s for ci, s in zip(c, signs) if ci]))
            return _min_weight_basis(*_independent(as_matrix(kept, n), kept_signs, n), n)

        sx, sx_signs = within(self.sx, self.sx_signs)
        sz, sz_signs = within(self.sz, self.sz_signs)
        code = StabilizerCode(qubits=self.qubits, hadamard=self.hadamard, sx=sx, sx_signs=sx_signs, sz=sz,
                              sz_signs=sz_signs, logical_x=[], logical_z=[], non_clifford=self.non_clifford,
                              positions=self.positions, nonclifford_columns=self.nonclifford_columns)
        stab = code.symplectic()
        lx, lz = _canonical_logicals(stab, n)
        code.logical_x = [PauliString.from_arrays(*_lightest(v, stab, n)) for v in lx]
        code.logical_z = [PauliString.from_arrays(*_lightest(v, stab, n)) for v in lz]
        return code

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
        matched = self._matched(other, by)
        if matched is None:
            return False
        a, b, columns = matched
        bs = b.symplectic()
        bs = bs[:, columns] if bs.size else bs
        ra, rb = rref(a.symplectic())[0], rref(bs)[0]
        return ra.shape == rb.shape and bool((ra == rb).all())

    def same_logicals_as(self, other: 'StabilizerCode', by: str = 'label') -> bool | None:
        """Whether the two codes name the same logical operators: each logical ``X`` and ``Z``
        of one is that of the other, up to stabilizers (signs ignored).

        ``None`` when they are not the same code (:meth:`same_code_as`, with
        qubits matched the same way). A block's logical action is named after the
        logical operators of the code after it; when that is the code before it,
        this says whether the names agree, so that the action is a gate on that one
        code. Matching qubits by position can map one logical qubit onto another.
        """
        if not self.same_code_as(other, by):
            return None
        a, b, columns = self._matched(other, by)
        if a.k != b.k:
            return False
        r, pivots, _ = rref(a.symplectic())
        return all(in_rowspace((p.symplectic + q.symplectic[columns]) % 2, r, pivots)
                   for ps, qs in ((a.logical_x, b.logical_x), (a.logical_z, b.logical_z)) for p, q in zip(ps, qs))

    def _matched(self, other: 'StabilizerCode', by: str) -> 'tuple[StabilizerCode, StabilizerCode, list[int]] | None':
        """Both codes on their data qubits, and the columns of ``other``'s ``(x | z)`` rows in ``self``'s
        qubit order; ``None`` when the qubits (labels or positions) differ."""
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
            return None
        order = [kb.index(q) for q in ka]
        return a, b, order + [b.n + j for j in order]

    def signs_modulo(self, constraints: Any, latest: bool = False) -> 'StabilizerCode':
        """The same code with every stabilizer sign rewritten modulo the linear ``constraints`` on the outcomes.

        Each sign is rewritten by :func:`~zxqec.gpf.reduce_modulo`: in the earliest
        outcomes it can be written in, or with ``latest`` in the latest. With the
        detectors of the program up to a cut as ``constraints``, the sign of a check
        that several rounds measured is written in the result of the last of them.
        The two codes are the same in every run the constraints allow.
        """
        from dataclasses import replace

        from ..gpf import reduce_modulo
        constraints = list(constraints)

        def rewrite(signs: list[Any]) -> list[Any]:
            return [reduce_modulo(sg, constraints, latest=latest) for sg in signs]
        return replace(self, sx_signs=rewrite(self.sx_signs), sz_signs=rewrite(self.sz_signs), _distance=None)

    # -- the ideal encoder ----------------------------------------------------------

    def encoder(self, checks: str = 'X', stabilizers: Any = None) -> BaseGraph:
        """The ideal encoder of this code: a ZX-diagram from ``k`` logical inputs to the qubits.

        It is the isometry ``E`` with ``E X_i = Xbar_i E`` and ``E Z_i = Zbar_i E``
        for the canonical logical operators, whose image is the code space with the
        stabilizer signs of this code (which may be Boolean functions of outcomes).
        Its adjoint is the ideal decoder. Up to a non-zero scalar.

        For a CSS code it is the two-layer diagram below: an X-spider on each qubit, a
        Z-spider for each logical input wired along its logical ``X``, and a Z-spider for
        each X stabilizer with its sign as its phase; the signs of the Z stabilizers are
        the phases of the qubits' X-spiders. ``checks='Z'`` gives the same map in the
        other colours: a Z-spider on each qubit, an X-spider for each logical input wired
        along its logical ``Z``, and an X-spider for each Z stabilizer with its sign as its
        phase. ``stabilizers`` builds it on other generators of the same stabilizer group,
        with their signs (:class:`PauliString`), instead of :attr:`stabilizers`: the same
        map, with a spider for each of those. For any other stabilizer code it is read off
        its Choi state (:func:`_choi_encoder`).
        """
        if checks not in ('X', 'Z'):
            raise ValueError("checks is 'X' or 'Z', not {!r}".format(checks))
        if not self.is_css:
            if checks == 'Z' or stabilizers is not None:
                raise NotImplementedError("encoder(checks='Z' or stabilizers=...): a code that is not CSS")
            return _choi_encoder(self)
        stabs = self.stabilizers if stabilizers is None else list(stabilizers)
        xs = [(p, p.sign) for p in stabs if any(p.x)]
        zs = [(p, p.sign) for p in stabs if any(p.z)]
        n = self.n
        on_qubit, on_check = VertexType.X, VertexType.Z
        logical_in, logical_fixed = self.logical_x, self.logical_z

        def part(p: PauliString) -> Any:
            return p.z

        if checks == 'Z':                               # the same construction with X and Z swapped
            xs, zs = zs, xs
            on_qubit, on_check = VertexType.Z, VertexType.X
            logical_in, logical_fixed = self.logical_z, self.logical_x

            def part(p: PauliString) -> Any:
                return p.x
        # output X-spider phases t: Z-stabilizer signs are t.z, logical Z acts without sign
        rows = [np.array(part(p), np.uint8) for p, _ in zs] + [np.array(part(p), np.uint8) for p in logical_fixed]
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
            w = g.add_vertex(on_qubit, j, 2, t[j])
            o = g.add_vertex(VertexType.BOUNDARY, j, 3)
            g.add_edge((w, o))
            xsp.append(w)
            outs.append(o)
        ins = []
        for i, p in enumerate(logical_in):
            b = g.add_vertex(VertexType.BOUNDARY, i, 0)
            z = g.add_vertex(on_check, i, 1)
            g.add_edge((b, z))
            ins.append(b)
            for j in p.support:
                g.add_edge((z, xsp[j]))
        for i, (p, s) in enumerate(xs):
            z = g.add_vertex(on_check, self.k + i, 1, s)
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

    def layout_checks(self, coords: dict[Any, Any]) -> dict[str, list[tuple[Any, ...]]]:
        """A generating set of the X-type and of the Z-type stabilizers, chosen to be drawn on a layout.

        ``coords`` gives the place ``(x, y)`` of every position (or label, if the code
        has no positions). The group is listed (:func:`~zxqec.codes.gf2.span`) and its
        elements taken greedily: fewest qubits first, then the smallest area of their
        bounding box on the layout, then by their qubits; an element is kept if it is not
        a product of those kept already. On a surface code these are its faces. Returns
        ``{'X': [...], 'Z': [...]}``, each check as the tuple of the positions it acts on.
        Signs are not kept. CSS codes only (``NotImplementedError`` otherwise).
        """
        from .gf2 import rank, span

        d = self.data()
        if not d.is_css:
            raise NotImplementedError("layout_checks: a code that is not CSS")
        where = list(d.positions) if d.positions is not None else list(d.qubits)
        missing = [w for w in where if w not in coords]
        if missing:
            raise ValueError("no coordinates for positions {}".format(missing))
        out: dict[str, list[tuple[Any, ...]]] = {}
        for kind in ('X', 'Z'):
            rows = [np.array(p.x if kind == 'X' else p.z, np.uint8) for p in d.stabilizers
                    if any(p.x if kind == 'X' else p.z)]
            if not rows:
                out[kind] = []
                continue
            group = [v for v in span(as_matrix(rows, d.n)) if v.any()]

            def key(v: np.ndarray) -> tuple[Any, ...]:
                pts = [coords[where[j]] for j in np.flatnonzero(v)]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                return (int(v.sum()), (max(xs) - min(xs)) * (max(ys) - min(ys)), tuple(np.flatnonzero(v)))

            kept: list[np.ndarray] = []
            target = rank(as_matrix(rows, d.n))
            for v in sorted(group, key=key):
                if rank(as_matrix(kept + [v], d.n)) > len(kept):
                    kept.append(v)
                    if len(kept) == target:
                        break
            out[kind] = [tuple(where[j] for j in np.flatnonzero(v)) for v in kept]
        return out

    def draw_layout(self, coords: dict[Any, Any], figsize: tuple[float, float] | None = None,
                    logicals: bool = True, y_down: bool = True, title: str | None = None) -> Any:
        """The code drawn on the hardware layout, the way surface codes are usually drawn.

        Each data qubit is a dot at ``coords[position]`` (the code's positions, or its
        labels if it has none), named by its label, so a code read after a permutation
        shows where each qubit has moved. Each check of :meth:`layout_checks` is a patch
        over the qubits it acts on, red for ``X`` and green for ``Z`` (PyZX's colours): a
        face for three or more qubits, a half-disc on the outer side for two, a ring for
        one. With ``logicals`` each logical operator is a dashed line through its qubits,
        named ``X`` or ``Z`` (with its index if the code has several). Signs are not shown.
        ``y_down`` puts larger ``y`` lower, as rows are numbered. CSS codes only. Returns
        the figure.
        """
        import matplotlib.pyplot as plt

        from ..spacetime.drawing import _VERTEX_COLORS, _finish

        d = self.data()
        checks = d.layout_checks(coords)
        where = list(d.positions) if d.positions is not None else list(d.qubits)
        pts = {w: tuple(float(c) for c in coords[w][:2]) for w in where}
        fill = {'X': _VERTEX_COLORS[VertexType.X], 'Z': _VERTEX_COLORS[VertexType.Z]}
        edge = {'X': '#b03030', 'Z': '#2e8b2e'}
        fig, ax = plt.subplots(figsize=figsize or (3.2, 3.2))
        _draw_check_faces(ax, checks, pts, {kind: dict(facecolor=fill[kind], edgecolor=edge[kind], alpha=0.75,
                                                       lw=1, zorder=1) for kind in ('X', 'Z')})
        if logicals:
            for kind, ops in (('X', d.logical_x), ('Z', d.logical_z)):
                for i, op in enumerate(ops):
                    line = sorted((pts[where[j]] for j in op.support), key=lambda p: (p[0], p[1]))
                    off = 0.08 if kind == 'X' else -0.08
                    ax.plot([p[0] + off for p in line], [p[1] + off for p in line], ls='--', lw=1.6,
                            color=edge[kind], zorder=3)
                    name = kind + (str(i) if len(ops) > 1 else '')
                    ax.text(line[-1][0] + off + 0.18, line[-1][1] + off + 0.18, name, color=edge[kind],
                            fontsize=11, fontweight='bold', ha='center', va='center', zorder=4)
        for w, q in zip(where, d.qubits):
            ax.plot(*pts[w], 'o', color='black', ms=5, zorder=5)
            ax.text(pts[w][0] - 0.12, pts[w][1] - 0.12, str(q), fontsize=9, ha='right', va='bottom', zorder=6)
        xs, ys = [p[0] for p in pts.values()], [p[1] for p in pts.values()]
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
        if y_down:
            ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        return _finish(fig)

    def draw_encoder(self, coords: dict[Any, Any], adjoint: bool = False, checks: str = 'auto',
                     figsize: tuple[float, float] | None = None, y_down: bool = True, spacing: float = 1.6,
                     label_size: float | None = None, face_colors: dict[str, str] | None = None,
                     face_alpha: float = 0.35, wire_ends: str = 'dot', title: str | None = None) -> Any:
        """The ideal encoder (:meth:`encoder`) drawn on the hardware layout, as encoders of surface codes are drawn by hand.

        The ZX-diagram of :meth:`encoder` (of the data qubits, :meth:`data`), each spider at
        a place on the layout: the spider of each qubit at ``coords[position]``, as in
        :meth:`draw_layout`, with its output wire a short stub below it; the spider of each
        check in the middle of the qubits it is wired to, or, for two qubits, on the outer
        side, where :meth:`draw_layout` puts the half-disc; the spider of each logical input
        beyond its logical operator, on the side away from the middle of the code (of its
        block, :meth:`blocks`; beside the line of its qubits if they are on one line), with
        its input wire pointing further away; of the places a step or more out, the nearest
        that is half a step clear of every qubit and of the other logical spiders and, on the
        side the stubs point to, off their lines (moving aside first, then further out).
        ``checks`` picks which stabilizers get a spider (:meth:`encoder`): ``'X'``, ``'Z'``,
        or ``'auto'``, the type with more signs that are not ``+`` (``'X'`` on a tie). The
        encoder is built on the checks of :meth:`layout_checks` (``encoder(stabilizers=...)``),
        the faces drawn behind it as in :meth:`draw_layout`, so that each check spider sits
        on its own face (moved a little where that place is taken by a qubit or another
        spider, as in a symmetric or one-dimensional layout); the sign of each face is worked out from those of
        :attr:`stabilizers` (the sum of the signs of the generators whose product it is). It
        is the phase of the face's spider; the signs of the other type are sums on the
        qubits' spiders. The faces are in ``face_colors`` (``{'X': ..., 'Z': ...}``, by
        default PyZX's colours) at opacity ``face_alpha``. ``label_size`` sets the font size of the phase labels and puts
        each beside its spider, as :func:`~zxqec.spacetime.draw_graph` does (without the
        white box behind it); for a check on the edge, outside it, clear of the stubs. ``adjoint`` draws the ideal decoder, the adjoint diagram, at
        the same places with the stubs above the qubits. ``wire_ends='tick'`` draws the
        ends of the wires bare, as by hand, instead of PyZX's dots. PyZX draws the diagram,
        its spiders a fixed size, so ``spacing`` (the drawn length of one unit of
        ``coords``) sets how far apart they are. Without ``figsize`` the figure is 0.3 inch
        per drawn unit, so that pictures drawn with the same ``spacing`` have the same
        scale; labels beyond the drawing are kept by saving with ``bbox_inches='tight'``. A
        long sign (a sum of outcomes) beside a qubit's spider can run over the next qubit, and
        two checks on facing edges of two patches put their half-discs and labels close.
        ``y_down`` as in :meth:`draw_layout`. CSS codes only. Returns the figure.
        """
        d = self.data()
        if not d.is_css:
            raise NotImplementedError("draw_encoder: a code that is not CSS")
        faces, on_faces = _face_generators(d, coords)
        if checks == 'auto':
            def signed(kind: str) -> int:
                return sum(1 for p in on_faces if any(p.x if kind == 'X' else p.z)
                           and (getattr(p.sign, 'free_vars', lambda: set())() or
                                (isinstance(p.sign, (int, Fraction)) and p.sign % 2 != 0)))
            checks = 'Z' if signed('Z') > signed('X') else 'X'
        g = d.encoder(checks, stabilizers=on_faces)
        if adjoint:
            g = g.adjoint()
        qubit_ends, logical_ends = (g.inputs(), g.outputs()) if adjoint else (g.outputs(), g.inputs())
        stub = 0.45 if adjoint else -0.45              # the decoder's inputs above the qubits, the encoder's outputs below
        ends = [[(b, stub)] for b in qubit_ends]
        return _draw_code_layout(d, coords, faces, g, ends, list(logical_ends), figsize, y_down, spacing, label_size,
                                 face_colors, face_alpha, wire_ends, title)

    def measurement(self, kind: str, stabilizers: Any = None) -> BaseGraph:
        """The ideal measurement of all checks of one ``kind`` (``'X'`` or ``'Z'``), with their signs: a ZX-diagram.

        The projector onto the states in which each stabilizer of that kind has its
        sign (which may be a Boolean function of outcomes), from the qubits to the
        qubits, up to a non-zero scalar: for X checks an X-spider on each qubit's wire
        and, for each check, a Z-spider wired to the X-spiders of its qubits with its
        sign as its phase; for Z checks the same in the other colours (the measurement of
        a Z parity: a Z-spider on each qubit, joined to an X-spider with the result as its
        phase). ``stabilizers``, as in :meth:`encoder`, uses other generators of the same
        group, with their signs. CSS codes only.
        """
        if kind not in ('X', 'Z'):
            raise ValueError("kind is 'X' or 'Z', not {!r}".format(kind))
        if not self.is_css:
            raise NotImplementedError("measurement: a code that is not CSS")
        stabs = self.stabilizers if stabilizers is None else list(stabilizers)
        checked = [p for p in stabs if any(p.x if kind == 'X' else p.z)]
        g = Graph()
        reg = g.var_registry
        for p in checked:
            for v in getattr(p.sign, 'free_vars', lambda: set())():
                reg.set_type(v.name, True)
        on_qubit, on_check = (VertexType.X, VertexType.Z) if kind == 'X' else (VertexType.Z, VertexType.X)
        ins, outs, spiders = [], [], []
        for j in range(self.n):
            i = g.add_vertex(VertexType.BOUNDARY, j, 0)
            w = g.add_vertex(on_qubit, j, 1)
            o = g.add_vertex(VertexType.BOUNDARY, j, 2)
            g.add_edge((i, w))
            g.add_edge((w, o))
            ins.append(i)
            outs.append(o)
            spiders.append(w)
        for p in checked:
            c = g.add_vertex(on_check, self.n, 1, p.sign)
            for j in p.support:
                g.add_edge((c, spiders[j]))
        g.set_inputs(tuple(ins))
        g.set_outputs(tuple(outs))
        g.rebind_variables_to_registry()
        return g

    def draw_measurement(self, coords: dict[Any, Any], kind: str, results: Any = None,
                         figsize: tuple[float, float] | None = None, y_down: bool = True, spacing: float = 1.6,
                         label_size: float | None = None, face_colors: dict[str, str] | None = None,
                         face_alpha: float = 0.35, wire_ends: str = 'dot', title: str | None = None) -> Any:
        """The ideal measurement of the checks of one ``kind`` (``'X'`` or ``'Z'``) drawn on the hardware layout.

        The diagram of :meth:`measurement` (a round of checks of one kind, as it is drawn
        by hand): the spider of each qubit at ``coords[position]``, with its input wire a
        short stub above it and its output wire a short stub below; a spider for each check
        of that kind of :meth:`layout_checks`, placed and labelled as by
        :meth:`draw_encoder`, with the check's sign as its phase (worked out as there).
        There is no logical wire: on the code space the measurement is the identity.
        ``results`` (``{name: operator}``, as :meth:`~zxqec.qprogram.QProgram.measured_operators`
        gives for the stage that measures the checks) puts on each check the result that
        measures it: its phase is that result (plus the operator's sign), so the picture
        shows what the stage reads, also where the result is always the same. Without it, or
        for a check no result measures, the phase is the check's sign. The signs of the code after a round of
        checks, in that round's results (``QProgram.code_at(..., outcomes='latest')``),
        show the round as it measured them. The faces of both kinds are drawn behind, and
        the other arguments are those of :meth:`draw_encoder`. CSS codes only. Returns the
        figure.
        """
        if kind not in ('X', 'Z'):
            raise ValueError("kind is 'X' or 'Z', not {!r}".format(kind))
        d = self.data()
        if not d.is_css:
            raise NotImplementedError("draw_measurement: a code that is not CSS")
        faces, on_faces = _face_generators(d, coords)
        if results:
            on_faces = [_read_by(d, p, kind, results) for p in on_faces]
        g = d.measurement(kind, stabilizers=on_faces)
        ins, outs = g.inputs(), g.outputs()
        ends = [[(i, 0.45), (o, -0.45)] for i, o in zip(ins, outs)]   # input above, output below
        return _draw_code_layout(d, coords, faces, g, ends, [], figsize, y_down, spacing, label_size,
                                 face_colors, face_alpha, wire_ends, title)


def _read_by(d: 'StabilizerCode', p: Any, kind: str, results: Any) -> Any:
    """The check ``p`` of the code ``d``, with the result that measures it as its sign, if one of ``results`` does.

    ``results`` maps names to operators on the program's qubits (``None`` for a result
    with no fixed operator); a result measures ``p`` if its operator is ``kind`` on
    exactly the qubits (labels) of ``p``. Its sign is then the result plus the
    operator's sign; otherwise ``p`` is returned unchanged.
    """
    from pyzx.symbolic import new_var

    if not any(p.x if kind == 'X' else p.z):
        return p
    labels = {d.qubits[j] for j in p.support}
    for name, op in results.items():
        if op is None:
            continue
        letters = op.x if kind == 'X' else op.z
        other = op.z if kind == 'X' else op.x
        if set(op.support) == labels and all(letters[q] for q in labels) and not any(other):
            sign = new_var(name, True)
            return PauliString(p.x, p.z, sign + 1 if op.sign else sign)
    return p


def _face_generators(d: 'StabilizerCode', coords: dict[Any, Any]) -> tuple[dict[str, list[tuple[Any, ...]]], list[Any]]:
    """The checks of :meth:`StabilizerCode.layout_checks` of the CSS code ``d``, and the same checks as generators with
    their signs: each face's sign is the sum of the signs of the stored generators whose product it is."""
    faces = d.layout_checks(coords)
    where = list(d.positions) if d.positions is not None else list(d.qubits)
    index = {w: j for j, w in enumerate(where)}
    on_faces = []
    for kind in ('X', 'Z'):
        gens = [p for p in d.stabilizers if any(p.x if kind == 'X' else p.z)]
        rows = as_matrix([np.array(p.x if kind == 'X' else p.z, np.uint8) for p in gens], d.n)
        for face in faces[kind]:
            v = np.zeros(d.n, np.uint8)
            v[[index[w] for w in face]] = 1
            combo = solve(rows.T, v)                    # the face as a product of the stored generators
            if combo is None:
                raise ValueError("the check on {} is not a stabilizer of the code".format(face))
            sign = _sum(p.sign for p, bit in zip(gens, combo) if bit)
            zero = np.zeros(d.n, np.uint8)
            on_faces.append(PauliString.from_arrays(v, zero, sign) if kind == 'X'
                            else PauliString.from_arrays(zero, v, sign))
    return faces, on_faces


def _draw_code_layout(d: 'StabilizerCode', coords: dict[Any, Any], faces: dict[str, list[tuple[Any, ...]]],
                      g: BaseGraph, ends: list[list[tuple[Any, float]]], logical_ends: list[Any],
                      figsize: tuple[float, float] | None, y_down: bool, spacing: float, label_size: float | None,
                      face_colors: dict[str, str] | None, face_alpha: float, wire_ends: str, title: str | None) -> Any:
    """Draw ``g``, a diagram on the qubits of the code ``d``, on the layout ``coords``.

    ``ends[j]`` lists the wire ends of qubit ``j`` (in the order of the code's positions)
    with the drawn offset of each stub (positive: above); the spider next to them is the
    qubit's. ``logical_ends`` are the wire ends of the logical spiders, and every other
    spider is a check. The placement, labels, faces and window are those described in
    :meth:`StabilizerCode.draw_encoder`.
    """
    from pyzx.drawing import draw_matplotlib
    from pyzx.utils import get_mode, settings

    from ..spacetime.drawing import _VERTEX_COLORS, _finish, _fit_layout, _set_labels, _set_wire_ends

    where = list(d.positions) if d.positions is not None else list(d.qubits)
    flip = -spacing if y_down else spacing          # places are in the plot's (x, y): PyZX draws at (row, -qubit)
    pts = {w: (spacing * float(coords[w][0]), flip * float(coords[w][1])) for w in where}
    middle: dict[Any, tuple[float, float]] = {}     # the middle of each position's block
    for b in d.blocks():
        ps = [pts[where[d.qubits.index(q)]] for q in b.qubits]
        for q in b.qubits:
            middle[where[d.qubits.index(q)]] = (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))
    place: dict[Any, tuple[float, float]] = {}
    centre: dict[Any, tuple[float, float]] = {}
    for w, wire in zip(where, ends):
        (v,) = list(g.neighbors(wire[0][0]))
        place[v], centre[v] = pts[w], middle[w]
        for b, dy in wire:
            place[b] = (pts[w][0], pts[w][1] + dy)
    stubs = sorted({1 if dy > 0 else -1 for wire in ends for _b, dy in wire})   # the directions the stubs point to
    placed: list[tuple[float, float]] = []          # the logical spiders placed so far
    for b in logical_ends:
        (z,) = list(g.neighbors(b))
        qs = [v for v in g.neighbors(z) if v != b]
        mx, my = sum(place[v][0] for v in qs) / len(qs), sum(place[v][1] for v in qs) / len(qs)
        cx, cy = centre[qs[0]]
        ux, uy = _away(mx - cx, my - cy)
        lx, ly = max(((place[v][0] - mx, place[v][1] - my) for v in qs), key=lambda t: t[0] ** 2 + t[1] ** 2)
        norm = (lx * lx + ly * ly) ** 0.5
        if norm > 1e-9 and all(abs((place[v][0] - mx) * ly - (place[v][1] - my) * lx) / norm < 0.2 * spacing
                               for v in qs):   # its qubits on one line (within a fifth of a step): beside it
            nx, ny = _away(-ly, lx)
            if nx * (mx - cx) + ny * (my - cy) < 0:
                nx, ny = -nx, -ny
            ux, uy = nx, ny
        block = [p for w, p in pts.items() if middle[w] == (cx, cy)]
        reach = max((p[0] - mx) * ux + (p[1] - my) * uy for p in block) + spacing   # beyond the block
        ax_, ay_ = -uy, ux                          # along the block's side, pointing right (or up)
        if ax_ < -1e-9 or (abs(ax_) <= 1e-9 and ay_ < 0):
            ax_, ay_ = -ax_, -ay_
        on_stubs = any(uy * sg > 0.7 for sg in stubs)   # beyond the qubits on the side their stubs point to

        def clear(x: float, y: float, wires: bool) -> bool:
            """Half a step from every qubit and logical spider, off the line of its qubits' stubs, and (with
            ``wires``) with no wire to its qubits passing over another qubit's spider."""
            others = list(pts.values()) + placed
            if any((x - p[0]) ** 2 + (y - p[1]) ** 2 < (0.5 * spacing) ** 2 for p in others):
                return False
            if on_stubs and any(abs(x - place[v][0]) < 0.25 * spacing for v in qs):
                return False
            return not wires or not any(_near_segment(p, (x, y), place[v], 0.15 * spacing)
                                        for v in qs for p in others if p != place[v])

        spot = None
        for wires in (True, False):                 # the nearest clear place: aside first, then further out
            for out in (0.0, 0.5, 1.0, 1.5, 2.0):
                for side in (0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5):
                    x = mx + (reach + out * spacing) * ux + side * spacing * ax_
                    y = my + (reach + out * spacing) * uy + side * spacing * ay_
                    if clear(x, y, wires):
                        spot = (x, y)
                        break
                if spot is not None:
                    break
            if spot is not None:
                break
        place[z] = spot if spot is not None else (mx + reach * ux, my + reach * uy)
        placed.append(place[z])
        place[b] = (place[z][0] + 0.45 * ux, place[z][1] + 0.45 * uy)
    outward: dict[tuple[float, float], tuple[float, float]] = {}   # checks on the edge: where their label goes
    checks_placed: list[tuple[float, float]] = []
    for v in g.vertices():
        if v not in place:                          # the spider of a check
            qs = list(g.neighbors(v))
            base = _check_place([place[w] for w in qs], *centre[qs[0]], 0.25 * spacing)

            def free(x: float, y: float) -> bool:
                """Half a step from every qubit and spider placed so far, and not beyond one of its qubits
                on the side of that qubit's stub, where its wire would run along the stub."""
                if any((x - p[0]) ** 2 + (y - p[1]) ** 2 < (0.5 * spacing) ** 2
                       for p in list(pts.values()) + placed + checks_placed):
                    return False
                return not any(abs(x - place[w][0]) < 0.25 * spacing and (y - place[w][1]) * sg > 0
                               for w in qs for sg in stubs)

            place[v] = base
            if not free(*base):                     # (a symmetric or collinear layout) the nearest free place
                for dx, dy in ((0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3), (0.3, 0.3), (-0.3, 0.3), (0.3, -0.3),
                               (-0.3, -0.3), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)):
                    if free(base[0] + dx * spacing, base[1] + dy * spacing):
                        place[v] = (base[0] + dx * spacing, base[1] + dy * spacing)
                        break
            checks_placed.append(place[v])
            if len(qs) <= 2:
                mx = sum(place[w][0] for w in qs) / len(qs)
                my = sum(place[w][1] for w in qs) / len(qs)
                outward[(round(place[v][0], 6), round(place[v][1], 6))] = _away(place[v][0] - mx, place[v][1] - my)
    for v, (x, y) in place.items():
        g.set_row(v, x)
        g.set_qubit(v, -y)
    mode = get_mode()
    settings.mode = "notebook"                      # PyZX would call plt.show() itself otherwise
    try:                                            # (no Hadamard edges here: 'box' keeps PyZX from bending the stubs)
        fig = draw_matplotlib(g, figsize=figsize or (3.2, 3.2), h_edge_draw='box', show_time=False)
    finally:
        settings.mode = mode
    ax = fig.axes[0]
    colors = face_colors or {'X': _VERTEX_COLORS[VertexType.X], 'Z': _VERTEX_COLORS[VertexType.Z]}
    _draw_check_faces(ax, faces, pts, {kind: dict(facecolor=colors[kind], edgecolor='white', alpha=face_alpha,
                                                  lw=1.5, zorder=-1) for kind in ('X', 'Z')},
                      middle=lambda check: middle[check[0]])
    if label_size is not None:
        _set_labels(ax, label_size, 'right')
        for t in ax.texts:
            t.set_bbox(None)                        # its white box would hide the wire ends next to it
            x, y = t.get_position()                 # (_set_labels put it 0.28 right of its spider)
            key = (round(x - 0.28, 6), round(y, 6))
            if key in outward:                      # a check on the edge: its label outside, clear of the stubs
                ux, uy = outward[key]
                t.set_position((key[0] + 0.42 * ux, key[1] + 0.42 * uy))
                if abs(uy) >= abs(ux):
                    t.set_horizontalalignment('center')
                    t.set_verticalalignment('bottom' if uy > 0 else 'top')
                else:
                    t.set_horizontalalignment('left' if ux > 0 else 'right')
                    t.set_verticalalignment('center')
    _set_wire_ends(ax, wire_ends)
    _fit_layout(fig, ax, list(place.values()), list(pts.values()), 0.55 * spacing, figsize is None, title)
    return _finish(fig)


def _draw_check_faces(ax: Any, checks: dict[str, list[tuple[Any, ...]]], pts: dict[Any, tuple[float, float]],
                      style: dict[str, dict[str, Any]], middle: Any = None) -> None:
    """Each check of ``checks`` (as :meth:`StabilizerCode.layout_checks` gives them) as a patch over its qubits.

    ``pts`` gives each position's place; ``style[kind]`` the keyword arguments of the
    patches of the ``'X'`` and the ``'Z'`` checks. A face for three or more qubits, a
    half-disc on the outer side for two, a ring for one. The outer side is away from
    ``middle(check)``, by default the middle of all the places.
    """
    from matplotlib.patches import Circle, Polygon, Wedge

    cx = sum(p[0] for p in pts.values()) / len(pts)
    cy = sum(p[1] for p in pts.values()) / len(pts)
    for kind in ('X', 'Z'):
        for check in checks[kind]:
            ps = [pts[w] for w in check]
            if len(ps) == 1:
                ax.add_patch(Circle(ps[0], 0.3, **style[kind]))
            elif len(ps) == 2:
                if middle is not None:
                    cx, cy = middle(check)
                (x1, y1), (x2, y2) = ps
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                r = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 2
                base = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                out = np.degrees(np.arctan2(my - cy, mx - cx))
                start = base if np.cos(np.radians(out - base - 90)) > 0 else base + 180
                ax.add_patch(Wedge((mx, my), r, start, start + 180, **style[kind]))
            else:
                mx, my = sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps)
                ps = sorted(ps, key=lambda p: np.arctan2(p[1] - my, p[0] - mx))
                ax.add_patch(Polygon(ps, closed=True, **style[kind]))


def _near_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float], r: float) -> bool:
    """Whether ``p`` is within ``r`` of the segment from ``a`` to ``b``, away from its ends."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        return False
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2
    return 0.02 < t < 0.98 and abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / length2 ** 0.5 < r


def _away(dx: float, dy: float) -> tuple[float, float]:
    """The unit vector along ``(dx, dy)``; to the left if that is (almost) zero."""
    norm = (dx * dx + dy * dy) ** 0.5
    return (dx / norm, dy / norm) if norm > 1e-9 else (-1.0, 0.0)


def _check_place(ps: list[tuple[float, float]], cx: float, cy: float, reach: float) -> tuple[float, float]:
    """Where :meth:`StabilizerCode.draw_encoder` puts the spider of a check on the qubits at ``ps``.

    The middle of the qubits; for two, out on the side away from ``(cx, cy)`` (the middle
    of the code), where the half-disc of :func:`_draw_check_faces` bulges; for one,
    ``reach`` away from its qubit, away from the middle of the code.
    """
    mx, my = sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps)
    if len(ps) >= 3:
        return (mx, my)
    if len(ps) == 2:
        (x1, y1), (x2, y2) = ps
        r = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 2
        nx, ny = -(y2 - y1) / (2 * r), (x2 - x1) / (2 * r)
        if nx * (mx - cx) + ny * (my - cy) < 0:
            nx, ny = -nx, -ny
        return (mx + 0.8 * r * nx, my + 0.8 * r * ny)
    ux, uy = _away(mx - cx, my - cy)
    return (mx + reach * ux, my + reach * uy)


# -- reading a code off a diagram ---------------------------------------------------

# -- the two layers, as code_of reads them ---------------------------------------------------

@dataclass
class CodeLayers:
    """The reduced diagram of an isometry as :func:`code_of` reads it, before any choice of basis.

    Each Z-spider of the reduced diagram that is not a gadget hub gives a row, with
    one column per output: 1 where the spider is wired to that output's X-spider.
    The rows of the Z-spiders on inputs form the matrix ``L`` (:attr:`input_rows`),
    the rows of the others the matrix ``M`` (:attr:`check_rows`), each in the order
    in which :meth:`draw` shows the spiders, top to bottom. Without phase
    gadgets, each row of ``M`` is an X stabilizer with the spider's phase as its
    sign, and the Z stabilizers are the Z strings that meet every row of ``L`` and
    ``M`` an even number of times (:meth:`z_rows`), each with the sum of the phases
    of the output X-spiders it touches as its sign (Kissinger, *Phase-free ZX
    diagrams are CSS codes*). :func:`code_of` takes these and chooses the
    generators and logical operators it prints.
    """
    gpf: GPFResult                  #: the reduced diagram
    labels: tuple[Any, ...]         #: the label of each output, i.e. of each column
    input_spiders: list[Any]        #: the Z-spider of each row of ``L``
    check_spiders: list[Any]        #: the Z-spider of each row of ``M``
    output_spiders: list[Any]       #: the X-spider of each output (column)
    hubs: list[Any]                 #: the hubs of the phase gadgets
    input_rows: np.ndarray          #: ``L``
    check_rows: np.ndarray          #: ``M``

    @property
    def check_phases(self) -> list[Any]:
        """The phase of each check spider (in units of pi): the sign of its X stabilizer."""
        return [self.gpf.graph.phase(v) for v in self.check_spiders]

    @property
    def output_phases(self) -> list[Any]:
        """The phase of each output's X-spider (in units of pi), the signs of the Z stabilizers."""
        return [self.gpf.graph.phase(v) for v in self.output_spiders]

    @property
    def names(self) -> dict[Any, str]:
        """``in i`` for the Z-spider on input ``i``, ``c1, c2, ...`` for the rows of ``M``,
        ``out q`` for the X-spider of the output labelled ``q``, ``h1, h2, ...`` for gadget hubs."""
        h = self.gpf.graph
        name = {next(iter(h.neighbors(b))): 'in {}'.format(i) for i, b in enumerate(h.inputs())}
        name.update({v: 'c{}'.format(i + 1) for i, v in enumerate(self.check_spiders)})
        name.update({v: 'out {}'.format(q) for v, q in zip(self.output_spiders, self.labels)})
        name.update({v: 'h{}'.format(i + 1) for i, v in enumerate(self.hubs)})
        return name

    def z_rows(self) -> np.ndarray:
        """A basis (as rows) of the Z strings that meet every row of ``L`` and ``M`` an even
        number of times: the null space of ``L`` stacked on ``M`` over GF(2)."""
        n = len(self.labels)
        return nullspace(as_matrix(list(self.input_rows) + list(self.check_rows), n))

    def describe(self) -> str:
        """The matrices ``L`` and ``M`` as a table, with the phases."""
        names = self.names
        w = max([len(str(q)) for q in self.labels] + [1])
        tag = max([len(names[v]) for v in self.input_spiders + self.check_spiders] + [1])
        lines = ["{} input(s), {} check spider(s), {} output(s), {} phase gadget(s)".format(
            len(self.input_spiders), len(self.check_spiders), len(self.labels), len(self.hubs)),
            "rows: Z-spiders; columns: outputs; 1 = wired to that output's X-spider",
            "     {}   {}  phase".format(' ' * tag, ' '.join(str(q).rjust(w) for q in self.labels))]
        for mark, spiders, rows in (('L', self.input_spiders, self.input_rows),
                                    ('M', self.check_spiders, self.check_rows)):
            for i, (v, row) in enumerate(zip(spiders, rows)):
                lines.append("  {}  {}   {}  {}".format(mark if i == 0 else ' ', names[v].ljust(tag),
                                                       ' '.join(str(int(b)).rjust(w) for b in row),
                                                       self.gpf.graph.phase(v)))
        lines.append("phases of the output X-spiders: {}".format(' '.join(str(p) for p in self.output_phases)))
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.describe()

    def draw(self, figsize: tuple[float, float] | None = None, **kwargs: Any) -> Any:
        """PyZX's drawing of the reduced diagram as :func:`~zxqec.gpf.layout_gpf` lays it out, each
        Z-spider named as in :meth:`describe` and each output by its label. Keyword arguments go to
        ``draw_matplotlib``. Returns the figure."""
        from pyzx.drawing import draw_matplotlib
        from pyzx.utils import get_mode, settings

        from ..spacetime.drawing import _finish

        h = self.gpf.graph
        lines = max([h.qubit(v) for v in h.vertices()] + [0]) + 1
        mode = get_mode()
        settings.mode = "notebook"                # PyZX would call plt.show() itself otherwise
        try:
            fig = draw_matplotlib(self.gpf.to_graph(), figsize=figsize or (3.5, 0.5 * lines + 0.6), **kwargs)
        finally:
            settings.mode = mode
        ax = fig.axes[0]
        names = self.names
        # beside the spiders, where PyZX writes nothing (it writes phases below them): a spider on an input
        # is named left of the input's wire end, a check spider left of itself
        inputs = set(h.inputs())
        for v in self.input_spiders + self.check_spiders:
            end = next((b for b in h.neighbors(v) if b in inputs), v)
            ax.text(h.row(end) - 0.25, -h.qubit(end), names[v], ha='right', va='center', fontsize=9)
        for b, q in zip(h.outputs(), self.labels):
            ax.text(h.row(b) + 0.25, -h.qubit(b), str(q), ha='left', va='center', fontsize=9)
        return _finish(fig)


def code_layers(g: BaseGraph, labels: Any = None) -> CodeLayers:
    """The two layers of ``g``'s reduced diagram, as :func:`code_of` reads them (:class:`CodeLayers`).

    ``labels`` name the outputs as in :func:`code_of`. Raises :class:`NotACodeError`
    if ``g`` is zero or not an isometry, and ``NotImplementedError`` for phase
    gadgets in the X basis.
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

    zl: list[tuple[Any, np.ndarray]] = []
    zm: list[tuple[Any, np.ndarray]] = []
    hubs = []
    for v in sorted(h.vertices()):
        ty = h.type(v)
        if ty == VertexType.BOUNDARY or is_leaf(h, v):
            continue
        if is_stem(h, v):
            if ty == VertexType.Z:
                raise NotImplementedError("code_of: a non-Clifford phase in the X basis "
                                          "(an X-leaf gadget) is not supported yet")
            hubs.append(v)
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
    # rows in the order layout_gpf draws their spiders, top to bottom (what code_of reads does not depend on it)
    zl.sort(key=lambda vr: (h.qubit(vr[0]), vr[0]))
    zm.sort(key=lambda vr: (h.qubit(vr[0]), vr[0]))
    return CodeLayers(gpf=res, labels=labels, input_spiders=[v for v, _ in zl], check_spiders=[v for v, _ in zm],
                      output_spiders=sorted(out_col, key=lambda w: out_col[w]), hubs=hubs,
                      input_rows=as_matrix([r for _, r in zl], n), check_rows=as_matrix([r for _, r in zm], n))


def stabilizes(g: BaseGraph, pauli: 'PauliString | str') -> bool:
    """Whether the Pauli string ``pauli`` on the outputs of ``g`` leaves ``g`` unchanged.

    Letter ``j`` acts on output ``j``, and the sign counts: ``-XX`` stabilizes a
    map that ``XX`` negates. Compared as tensors with their scalars
    (:func:`~zxqec.gpf.same_map`), so only for small diagrams. A check of
    :func:`code_of` from outside: :func:`code_of` reads the stabilizers off the
    reduced diagram, this applies them to ``g`` itself.
    """
    from ..gpf import same_map
    from .pauli import _constant

    p = PauliString.from_str(pauli) if isinstance(pauli, str) else pauli
    outs = list(g.outputs())
    if p.n != len(outs):
        raise ValueError("a {}-qubit string for a diagram with {} outputs".format(p.n, len(outs)))
    if hasattr(p.sign, 'free_vars') and p.sign.free_vars():
        raise ValueError("the sign {} depends on measurement outcomes".format(p.sign))
    after = g.copy()                                   # a copy numbers its vertices anew: use its own outputs
    for b, x, z in zip(after.outputs(), p.x, p.z):
        if not (x or z):
            continue
        (w,) = list(after.neighbors(b))
        e = after.edge(w, b)
        et = after.edge_type(e)
        after.remove_edge(e)
        last, row = w, after.row(b)
        # Y = i X Z: first Z, then X, and a scalar i
        for ty, on in ((VertexType.Z, z), (VertexType.X, x)):
            if on:
                v = after.add_vertex(ty, qubit=after.qubit(b), row=row, phase=1)
                after.add_edge((last, v), et if last == w else EdgeType.SIMPLE)
                last = v
        after.add_edge((last, b), EdgeType.SIMPLE)
        if x and z:
            after.scalar.add_phase(Fraction(1, 2))
    sign = p.sign if isinstance(p.sign, int) else _constant(p.sign)
    if sign % 2:
        after.scalar.add_phase(1)
    return same_map(g, after, preserve_scalar=True)


def code_of(g: BaseGraph, labels: Any = None, positions: Any = None,
            common_to_all_runs: bool = False, generators: str = 'lightest') -> StabilizerCode:
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

    ``generators`` chooses which generators of the stabilizer group are kept; the
    group, ``k``, the logical operators and the distance do not depend on it.
    ``'lightest'`` (default): the lightest elements of the group, then those with
    the fewest outcomes in their signs, then by support, found by listing the
    group (up to ``2^16`` elements; a larger group keeps the echelon basis).
    ``'echelon'``: the group's reduced row echelon basis in the order of the
    outputs, by Gaussian elimination only. For a code that is not CSS this is
    the echelon form of the X parts, each with its Z part reduced modulo the
    Z-type stabilizers, in the frame the code is read in (``hadamard``), so the
    printed strings need not look like an echelon form.
    """
    if generators not in ('lightest', 'echelon'):
        raise ValueError("generators must be 'lightest' or 'echelon', not {!r}".format(generators))
    cap = 16 if generators == 'lightest' else 0      # a group above the cap keeps its echelon basis
    layers = code_layers(g, labels)
    h = layers.gpf.graph
    outs = list(h.outputs())
    labels = layers.labels
    n = len(labels)
    zl = list(zip(layers.input_spiders, layers.input_rows))
    zm = list(zip(layers.check_spiders, layers.check_rows))
    rows_m = list(layers.check_rows)
    signs_m = layers.check_phases
    zlayer = layers.input_spiders + layers.check_spiders     # Z-spiders of the layer, L rows first then M rows
    gadget_hubs = layers.hubs
    m = len(rows_m)
    t = layers.output_phases
    hflags = tuple(int(has_boundary_hadamard(h, b)) for b in outs)

    zstab = layers.z_rows()
    sz_signs = [_sum([ti for ti, zi in zip(t, z) if zi]) for z in zstab]
    sz, sz_signs = _min_weight_basis(*_independent(zstab, sz_signs, n), n, cap)

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
                                                           [sg for _, sg in zt], n), n, cap)
        sx, sxz, sx_signs = _min_weight_basis_mixed(sx, sxz, sx_signs, n, sz, sz_signs, cap)
    else:
        sx, sx_signs = _min_weight_basis(*_independent(sx_rows, sx_signs, n), n, cap)
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
            if angle.free_vars():
                raise NotImplementedError("code_of: a gadget with a symbolic angle")
            angle = sum((Fraction(c) for c, t in angle.terms if not t.vars), Fraction(0))   # a constant
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
