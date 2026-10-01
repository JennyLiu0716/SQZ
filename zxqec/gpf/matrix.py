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

"""Split biadjacency matrices of a GPF diagram and their reduction over GF(2).

One layer of a GPF diagram (say the Z-spiders) is encoded as a binary matrix
with one row per spider and one column per thing it can be connected to:

* **external** columns -- input boundaries, gadget leaves and gadget hubs of
  the other colour: fixed targets whose identity must be preserved. Anchoring
  the rows that merely *carry* a gadget of the opposite colour is sufficient
  but not necessary: such a row could be a source if its leaf were kept as a
  variable of its own. We anchor it so that each gadget stays on one row;
* **internal** columns -- the remaining spiders of the other colour;
* a **phase** per row: its Pauli phase, a Boolean polynomial (0 or 1 for a
  concrete phase).

Adding row ``i`` to row ``j`` is the rewrite that replaces ``j``'s neighbourhood
by the symmetric difference with ``i``'s *and adds the phases*. It is sound
exactly when row ``i`` is **free**: not connected to any external column (its
bit is a free summation variable, not fixed by a boundary or a gadget) and
Pauli-phased. Row ``j`` may be anything. :func:`reduce_rows` is Gauss-Jordan
elimination with that restriction: pivots are free rows only, chosen column by
column over the internal columns; anchored rows are never used as a source and
therefore never contaminate each other (the thesis's boundary-isolation
argument), but they do absorb free rows. A free row that becomes all-zero is a
spider that lost every edge, i.e. the scalar ``1 + e^{i pi p}`` for its phase
``p``: ``2`` if ``p = 0``, ``0`` if ``p = 1``, and for a symbolic ``p`` the
constraint ``p = 0`` -- a detector of the measurement outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.symbolic import Poly
from pyzx.utils import VertexType, is_pauli

from .normal_form import is_leaf, is_stem, layout_gpf

__all__ = ['SplitMatrix', 'split_matrix', 'reduce_rows', 'apply_matrix', 'change_variables', 'phase_mod2',
           'is_zero_phase']


def phase_mod2(p: Any) -> Any:
    """Normalise a Pauli phase: ``0``/``1`` for constants, a ``Poly`` without zero terms otherwise.

    Only Pauli phases reach this: a row of a split matrix carries one by (G3),
    and a non-Pauli constant would be silently truncated (``int(3/2) % 2 == 1``),
    so it is rejected instead.
    """
    if not isinstance(p, Poly) and p != int(p):
        raise ValueError("phase_mod2: {} is not a Pauli phase".format(p))
    if isinstance(p, Poly):
        q = p % 2
        terms = [(c, t) for c, t in q.terms if c != 0]
        if not terms:
            return 0
        if len(terms) == 1 and not terms[0][1].vars:
            return int(terms[0][0]) % 2
        return Poly(terms)
    return int(p) % 2


def is_zero_phase(p: Any) -> bool:
    return not isinstance(p, Poly) and int(p) % 2 == 0


@dataclass
class SplitMatrix:
    """One layer of a GPF diagram as a matrix (see the module docstring)."""
    side: VertexType                       # the colour of the rows
    rows: list[Any]                        # row spiders
    external: list[Any]                    # external column targets
    internal: list[Any]                    # internal column targets
    entries: list[list[int]]               # rows x (external + internal), over GF(2)
    phases: list[Any]                      # Pauli phase of each row (int or Poly)
    anchored: list[bool] = field(default_factory=list)   # row may not be a source

    @property
    def num_external(self) -> int:
        return len(self.external)

    def free_rows(self) -> list[int]:
        return [i for i, a in enumerate(self.anchored) if not a]

    def zero_rows(self) -> list[int]:
        return [i for i, r in enumerate(self.entries) if not any(r)]

    def dense(self) -> list[list[int]]:
        """Entries with the phase's constant part appended (for display/tests)."""
        return [r + [0 if isinstance(p, Poly) else int(p)] for r, p in zip(self.entries, self.phases)]


def split_matrix(g: BaseGraph[VT, ET], side: VertexType) -> SplitMatrix:
    """The split matrix of the ``side``-coloured layer of a GPF diagram ``g``."""
    if side not in (VertexType.Z, VertexType.X):
        raise ValueError("side must be VertexType.Z or VertexType.X")
    other = VertexType.X if side == VertexType.Z else VertexType.Z

    rows = sorted(v for v in g.vertices() if g.type(v) == side and not is_leaf(g, v))
    bnd = [b for b in (g.inputs() if side == VertexType.Z else g.outputs())]
    leaves = sorted(v for v in g.vertices() if g.type(v) == other and is_leaf(g, v))
    stems = sorted(v for v in g.vertices() if g.type(v) == other and is_stem(g, v) and not is_leaf(g, v))
    external = list(bnd) + leaves + stems
    ext_set = set(external)
    internal = sorted(v for v in g.vertices()
                      if g.type(v) == other and v not in ext_set and not is_leaf(g, v))
    cols = external + internal

    entries = [[1 if g.connected(r, c) else 0 for c in cols] for r in rows]
    phases = [phase_mod2(g.phase(r)) for r in rows]
    anchored = [any(row[:len(external)]) or not is_pauli(g.phase(r))
                for row, r in zip(entries, rows)]
    return SplitMatrix(side, rows, external, internal, entries, phases, anchored)


def reduce_rows(m: SplitMatrix) -> list[int]:
    """Gauss-Jordan elimination with free rows as the only sources (in place).

    Returns the pivot columns used. Afterwards the free rows are in reduced row
    echelon form over the internal columns, every other row has a ``0`` in each
    pivot column, and phases have been added along with the rows.
    """
    n_ext = m.num_external
    n_cols = len(m.entries[0]) if m.entries else 0
    used: set[int] = set()
    pivots: list[int] = []
    for col in range(n_ext, n_cols):
        pivot = next((i for i in m.free_rows() if i not in used and m.entries[i][col]), None)
        if pivot is None:
            continue
        used.add(pivot)
        pivots.append(col)
        src = m.entries[pivot]
        for i in range(len(m.entries)):
            if i != pivot and m.entries[i][col]:
                m.entries[i] = [a ^ b for a, b in zip(m.entries[i], src)]
                m.phases[i] = phase_mod2(m.phases[i] + m.phases[pivot])
    return pivots


def apply_matrix(g: BaseGraph[VT, ET], m: SplitMatrix) -> list[Any]:
    """Rewrite the ``m.side`` layer of ``g`` to match ``m`` (in place).

    Row spiders keep their identity; their edges to the column targets and
    their phases are replaced. A row without any edge left is removed and its
    phase returned in the list of constraints (a non-zero one means the diagram
    is the scalar ``1 + e^{i pi p}``: zero for ``p = 1``, the condition ``p = 0``
    for a symbolic ``p``). Rows that are zero and phase-free contribute the
    scalar ``2`` each, which is dropped.
    """
    cols = m.external + m.internal
    constraints: list[Any] = []
    for i, r in enumerate(m.rows):
        row = m.entries[i]
        if not any(row):
            if not is_zero_phase(m.phases[i]):
                constraints.append(m.phases[i])
            g.remove_vertex(r)
            continue
        current = {c for c in g.neighbors(r) if c in cols}
        wanted = {c for c, bit in zip(cols, row) if bit}
        for c in current - wanted:
            g.remove_edge(g.edge(r, c))
        for c in wanted - current:
            g.add_edge((r, c))
        g.set_phase(r, m.phases[i])
    return constraints


def change_variables(g: BaseGraph[VT, ET], rows: Any, fuse: bool = True) -> BaseGraph[VT, ET]:
    """A clone of the state ``g`` in new summation variables, so that its Z-layer rows are ``rows``.

    ``g`` is a state (no inputs) in generalized parity form whose Z-spiders carry
    Pauli phases; ``rows`` are 0/1 vectors over the outputs spanning the same
    space as the Z-layer rows (over the outputs). With ``A`` those rows and
    ``T A = rows``, the old bits are ``y = y' T``: this is the row operation of
    this module (:func:`split_matrix`, :func:`apply_matrix`) with the matrix
    ``T``, applied to every column, so the Pauli phases become ``T s`` and a
    gadget that read the parity ``l . y`` reads ``(T l) . y'``. The map is the
    same; only the order of summation changes. If the Z-layer has more spiders
    than ``rows``, the remaining new variables are *hidden*: their rows are zero
    on the outputs (a basis of the combinations ``c`` with ``c A = 0``).

    The Z-spiders keep their vertices: the ``i``-th of them in increasing order
    gets ``rows[i]``, the hidden ones come last. With ``fuse``, every gadget left
    reading a single spider, with a hub of phase 0, is fused into that spider by
    PyZX's ``remove_id`` (the hub) and ``fuse`` (the leaf). The clone is laid out
    by :func:`~zxqec.gpf.normal_form.layout_gpf`.
    """
    from pyzx.rewrite_rules import fuse as fuse_spiders, remove_id
    from ..codes.gf2 import as_matrix, nullspace, rank, solve

    if list(g.inputs()):
        raise ValueError("change_variables takes a state (a diagram without inputs)")
    h = g.clone()
    m = split_matrix(h, VertexType.Z)
    if not all(is_pauli(h.phase(r)) for r in m.rows):
        raise ValueError("a Z-spider has a phase that is not Pauli")
    cols = m.external + m.internal
    out_cols = [cols.index(next(iter(h.neighbors(b)))) for b in h.outputs()]
    k = len(m.rows)
    old = as_matrix([[e[c] for c in out_cols] for e in m.entries], len(out_cols))
    t = []
    for r in as_matrix([list(r) for r in rows], len(out_cols)):
        c = solve(old.T.copy(), r)
        if c is None:
            raise ValueError("row {} is not in the span of the Z-layer".format(list(r)))
        t.append(list(c))
    t += [list(c) for c in nullspace(old.T.copy())] if k else []
    if len(t) != k or rank(as_matrix(t, k)) < k:
        raise ValueError("the rows are not a basis of the span of the Z-layer")
    entries = [[sum(t[i][j] * m.entries[j][c] for j in range(k)) % 2 for c in range(len(cols))] for i in range(k)]
    phases = []
    for i in range(k):
        p: Any = 0
        for j in range(k):
            if t[i][j]:
                p = p + m.phases[j]
        phases.append(phase_mod2(p))
    m.entries, m.phases = entries, phases
    apply_matrix(h, m)
    if fuse:
        for leaf in [v for v in h.vertices() if is_leaf(h, v)]:
            (hub,) = list(h.neighbors(leaf))
            reads = [z for z in h.neighbors(hub) if z != leaf]
            if len(reads) == 1 and h.type(reads[0]) == VertexType.Z and is_zero_phase(h.phase(hub)):
                remove_id(h, hub)
                fuse_spiders(h, reads[0], leaf)
    h.rebind_variables_to_registry()
    layout_gpf(h)
    return h
