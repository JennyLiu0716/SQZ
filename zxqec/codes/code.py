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

A non-Clifford phase gadget hanging from internal Z-spiders means the
corresponding X-strings are *not* stabilizers of the image (flipping the bit
under the gadget changes its phase); they are counted as logical instead. This
is how the prefix of a magic-state injection comes out as "the Steane code,
holding a non-stabilizer logical state".

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
from pyzx.utils import VertexType

from ..gpf import has_boundary_hadamard, is_leaf, is_stem, reduce_gpf
from .pauli import phase_sum as _sum
from .gf2 import as_matrix, complement_basis, in_rowspace, nullspace, reduce_mod, rref, solve
from .pauli import PauliString

__all__ = ['StabilizerCode', 'NotACodeError', 'code_of']


class NotACodeError(ValueError):
    """The diagram is not a Clifford isometry, so its image is not a stabilizer code."""


# -- the code ------------------------------------------------------------------

@dataclass
class StabilizerCode:
    """A stabilizer code on labelled qubits, with signs and canonical logical operators.

    The generators are stored by their type in the frame in which the diagram was
    read (``hadamard[j]`` records a Hadamard on qubit ``j``), so that products of
    generators never pick up factors of ``i``: ``sx`` rows are X-type and ``sz``
    rows Z-type in that frame. :attr:`stabilizers` gives them as physical Pauli
    strings.
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
        return ([self._physical(row, s, 'X') for row, s in zip(self.sx, self.sx_signs)]
                + [self._physical(row, s, 'Z') for row, s in zip(self.sz, self.sz_signs)])

    def _physical(self, row: np.ndarray, sign: Any, kind: str) -> PauliString:
        h = np.array(self.hadamard, dtype=np.uint8)
        a = row & (1 - h)
        b = row & h
        return PauliString.from_arrays(a, b, sign) if kind == 'X' else PauliString.from_arrays(b, a, sign)

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
        strings when the state depends on an outcome.
        """
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
        Its adjoint is the ideal decoder. Up to a non-zero scalar. CSS codes only.
        """
        if not self.is_css:
            raise NotImplementedError("the ideal encoder is only built for CSS codes")
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

def code_of(g: BaseGraph, labels: Any = None, positions: Any = None) -> StabilizerCode:
    """The stabilizer code that ``g`` encodes into (question 1 of the paper).

    ``g`` maps its inputs (logical qubits) to its outputs (physical qubits);
    ``labels[j]`` names output ``j`` (default: its index). Symbolic Boolean phases
    (measurement outcomes) are allowed and become stabilizer signs. Raises
    :class:`NotACodeError` if ``g`` is zero or not an isometry, and
    ``NotImplementedError`` for non-Clifford phases in the X basis.
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

    # which X-strings of the internal rows survive the phase gadgets (see .phasepoly)
    mm = as_matrix(rows_m, n)
    combos, extra = _surviving_combinations(h, zm, zlayer, gadget_hubs)
    sx_rows = (combos @ mm) % 2 if m else np.zeros((0, n), np.uint8)
    sx_signs = [_sum([s for c, s in zip(a, signs_m) if c] + [e]) for a, e in zip(combos, extra)] if m else []
    sx, sx_signs = _min_weight_basis(*_independent(sx_rows, sx_signs, n), n)
    touched: set[int] = set()
    for hub in gadget_hubs:
        for v in h.neighbors(hub):
            for w, row in zl + zm:
                if w == v:
                    touched |= {int(j) for j in np.flatnonzero(row)}

    all_rows = as_matrix(rows_l + rows_m, n)
    zstab = nullspace(all_rows)
    sz_signs = [_sum([ti for ti, zi in zip(t, z) if zi]) for z in zstab]
    sz, sz_signs = _min_weight_basis(*_independent(zstab, sz_signs, n), n)

    code = StabilizerCode(qubits=labels, hadamard=hflags, sx=sx, sx_signs=sx_signs, sz=sz,
                          sz_signs=sz_signs, logical_x=[], logical_z=[],
                          non_clifford=bool(gadget_hubs),
                          positions=tuple(positions) if positions is not None else None,
                          nonclifford_columns=frozenset(touched))
    stab = code.symplectic()
    lx, lz = _canonical_logicals(stab, n)
    code.logical_x = [PauliString.from_arrays(*_lightest(v, stab, n)) for v in lx]
    code.logical_z = [PauliString.from_arrays(*_lightest(v, stab, n)) for v in lz]
    return code


# -- helpers ------------------------------------------------------------------------------

def _surviving_combinations(h: BaseGraph, zm: list, zlayer: list, hubs: list,
                            cap: int = 12) -> tuple[np.ndarray, list[Any]]:
    """The combinations ``a`` of internal Z-spiders whose X-string is still a stabilizer.

    Without gadgets that is every combination. With gadgets, ``a`` survives when
    flipping its bits changes the gadgets' total phase by an amount that does
    not depend on any bit (:func:`~zxqec.codes.phasepoly.delta`); that amount,
    a function of the outcomes, is returned as an extra sign. Up to ``cap``
    internal spiders are searched exhaustively; beyond that, any gadget that
    sees a combination oddly breaks it (sound, but it can call a code
    non-Clifford too eagerly).
    """
    from .phasepoly import Gadget, affine_parts, delta, to_pyzx
    m = len(zm)
    if not hubs or m == 0:
        return np.eye(m, dtype=np.uint8), [0] * m
    name = {v: 'z{}'.format(i) for i, v in enumerate(zlayer)}
    gadgets = []
    for hub in hubs:
        leaves = [w for w in h.neighbors(hub) if is_leaf(h, w)]
        angle = sum((h.phase(w) for w in leaves), Fraction(0))
        if hasattr(angle, 'free_vars'):
            raise NotImplementedError("code_of: a gadget with a symbolic angle")
        sign_vars, const = affine_parts(h.phase(hub))
        bits = [name[w] for w in h.neighbors(hub) if w in name]
        gadgets.append(Gadget(Fraction(angle), bits, sign_vars, const))
    mnames = [name[v] for v, _ in zm]
    if m > cap:
        inc = [[1 if nm in g.bits else 0 for nm in mnames] for g in gadgets]
        return nullspace(as_matrix(inc, m)), [0] * nullspace(as_matrix(inc, m)).shape[0]
    passing, signs = [], []
    for mask in range(1, 1 << m):
        a = np.array([(mask >> i) & 1 for i in range(m)], dtype=np.uint8)
        d = delta(gadgets, {nm for nm, ai in zip(mnames, a) if ai})
        if any(any(v.startswith('z') and v[1:].isdigit() for v in mono) for mono in d):
            continue
        passing.append(a)
        signs.append(to_pyzx(d, h.var_registry))
    if not passing:
        return np.zeros((0, m), np.uint8), []
    # the passing combinations form a subspace, so the rows of its RREF are among them
    r = rref(as_matrix(passing, m))[0]
    out, out_signs = [], []
    for row in r:
        idx = next(i for i, a in enumerate(passing) if (a == row).all())
        out.append(passing[idx])
        out_signs.append(signs[idx])
    return as_matrix(out, m), out_signs


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
        partner = next((w for w in basis if _omega(v, w, n)), None)
        if partner is None:
            continue
        basis.remove(partner)
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
        return ('0', '1')[const % 2] if pauli == 'Z' else ('+', '-')[const % 2]
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
