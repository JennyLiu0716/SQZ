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

"""The GPF reduction pipeline (Algorithm 3 of the thesis, with phases).

:func:`reduce_gpf` brings a diagram into generalized parity form
(:func:`~zxqec.gpf.normal_form.to_gpf`), then alternates the split-matrix row
reduction of the Z-layer and of the X-layer (:mod:`zxqec.gpf.matrix`) until
nothing changes, and finally removes isolated Z-X pairs. The result is a
:class:`GPFResult` holding the reduced diagram, the *constraints* collected on
the way -- Boolean polynomials in the measurement outcomes that must vanish,
i.e. the detectors of the circuit -- and whether the diagram is identically
zero (a constraint that is the constant ``1``).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.symbolic import Poly
from pyzx.utils import VertexType, is_pauli

from .hadamard import restore_boundary_hadamards
from .matrix import apply_matrix, is_zero_phase, phase_mod2, reduce_rows, split_matrix
from .normal_form import is_leaf, layout_gpf, to_gpf

__all__ = ['GPFResult', 'reduce_gpf', 'equal_for_all_assignments', 'boolean_variables', 'evaluate_phase',
           'sample_consistent_assignment']


@dataclass
class GPFResult:
    """Outcome of :func:`reduce_gpf`."""
    graph: BaseGraph                 #: the reduced diagram (GPF with phases and gadgets)
    constraints: list[Any] = field(default_factory=list)  #: Boolean polynomials that must be 0
    is_zero: bool = False            #: the diagram is the zero map (a constraint is identically 1)

    @property
    def detectors(self) -> list[Poly]:
        """The symbolic constraints: parities of outcomes that are always 0."""
        return [c for c in self.constraints if isinstance(c, Poly)]

    def to_graph(self) -> BaseGraph:
        """The reduced diagram with its boundary Hadamards put back as Hadamard edges."""
        return restore_boundary_hadamards(self.graph)


def reduce_gpf(g: BaseGraph[VT, ET], isolate_paulis: bool = False,
               max_rounds: int = 20) -> GPFResult:
    """Reduce ``g`` to a minimal GPF with phases; see the module docstring.

    ``isolate_paulis=True`` isolates every non-zero phase into a gadget (the
    thesis's behaviour); by default Pauli phases stay on the spiders and are
    carried through the row reduction.
    """
    h = to_gpf(g, isolate_paulis=isolate_paulis)
    constraints = _scalar_constraints(h)
    for _ in range(max_rounds):
        changed = False
        for side in (VertexType.Z, VertexType.X):
            m = split_matrix(h, side)
            before = [row[:] for row in m.entries], [p for p in m.phases]
            reduce_rows(m)
            if m.entries != before[0] or [str(p) for p in m.phases] != [str(p) for p in before[1]]:
                changed = True
            constraints += apply_matrix(h, m)
        constraints += _remove_isolated_pairs(h)
        if not changed:
            break
    layout_gpf(h)
    constraints = [phase_mod2(c) for c in constraints]
    constraints = [c for c in constraints if not is_zero_phase(c)]
    is_zero = any(not isinstance(c, Poly) for c in constraints)
    return GPFResult(h, constraints, is_zero)


def _scalar_constraints(g: BaseGraph[VT, ET]) -> list[Any]:
    """Constraints recorded by PyZX when it removed isolated spiders during ``to_gpf``.

    An isolated Z/X spider with Pauli phase ``p`` is the scalar ``1 + e^{i pi p}``.
    Non-Pauli isolated spiders are ordinary non-zero scalars and are ignored.
    """
    out = []
    for p in g.scalar.phasenodes:
        if is_pauli(p) and not is_zero_phase(phase_mod2(p)):
            out.append(phase_mod2(p))
    g.scalar.phasenodes = []
    return out


def _remove_isolated_pairs(g: BaseGraph[VT, ET]) -> list[Any]:
    """Drop Z-X pairs connected only to each other (a non-zero scalar) and isolated spiders."""
    constraints: list[Any] = []
    for v in list(g.vertices()):
        if v not in g.vertices() or g.type(v) == VertexType.BOUNDARY or is_leaf(g, v):
            continue
        d = g.vertex_degree(v)
        if d == 0:
            p = phase_mod2(g.phase(v)) if is_pauli(g.phase(v)) else 0
            if not is_zero_phase(p):
                constraints.append(p)
            g.remove_vertex(v)
        elif d == 1:
            (w,) = list(g.neighbors(v))
            if (g.type(w) != VertexType.BOUNDARY and not is_leaf(g, w)
                    and g.vertex_degree(w) == 1 and g.type(w) != g.type(v)):
                g.remove_vertices([v, w])          # <X(q)|Z(p)> is never zero
    return constraints


# -- verification helpers ------------------------------------------------------

def boolean_variables(*graphs: BaseGraph[VT, ET]) -> list[str]:
    """Names of the Boolean variables appearing in the phases of the given graphs."""
    names: set[str] = set()
    for g in graphs:
        for v in g.vertices():
            p = g.phase(v)
            if isinstance(p, Poly):
                names |= {var.name for var in p.free_vars() if var.is_bool}
    return sorted(names)


def equal_for_all_assignments(g: BaseGraph[VT, ET], h: BaseGraph[VT, ET],
                              variables: Sequence[str] | None = None,
                              constraints: Iterable[Any] = (),
                              preserve_scalar: bool = False) -> bool:
    """Whether ``g`` and ``h`` are equal as linear maps for every Boolean assignment.

    Assignments that violate one of ``constraints`` (polynomials that must be
    0) are skipped, since the reduced diagram is only defined on outcomes that
    can occur. ``variables`` defaults to all Boolean variables of both graphs.
    """
    import pyzx

    names = list(variables) if variables is not None else boolean_variables(g, h)
    for values in itertools.product([0, 1], repeat=len(names)):
        subs = dict(zip(names, values))
        if any(evaluate_phase(c, subs) for c in constraints):
            continue
        a = g.substitute_variables(subs) if names else g
        b = h.substitute_variables(subs) if names else h
        if not pyzx.compare_tensors(a, b, preserve_scalar=preserve_scalar):
            return False
    return True


def evaluate_phase(p: Any, assignment: dict[str, int]) -> int:
    """Value (0 or 1) of a Pauli phase polynomial under a Boolean ``assignment`` (unset variables are 0)."""
    subs = assignment
    if isinstance(p, Poly):
        total = 0
        for c, t in p.terms:
            term = 1
            for var, exp in t.vars:
                term *= subs.get(var.name, 0) ** exp
            total += int(c) * term
        return total % 2
    return int(p) % 2


def sample_consistent_assignment(constraints: Iterable[Any], variables: Sequence[str],
                                 rng: Any = None) -> dict[str, int]:
    """A uniformly random assignment of ``variables`` satisfying the *linear* ``constraints``.

    The constraints (Boolean polynomials that must be 0, e.g. the detectors of
    :func:`reduce_gpf`) are solved over GF(2): free variables are drawn at
    random and the pivot variables computed from them. Raises ``ValueError``
    if the system has no solution or a constraint is not linear.
    """
    import random as _random
    rng = rng if rng is not None else _random.Random()
    names = list(variables)
    index = {n: i for i, n in enumerate(names)}
    rows: list[list[int]] = []
    for c in constraints:
        row = [0] * (len(names) + 1)
        if isinstance(c, Poly):
            for coeff, term in c.terms:
                if int(coeff) % 2 == 0:
                    continue
                if not term.vars:
                    row[-1] ^= 1
                elif len(term.vars) == 1 and term.vars[0][1] == 1:
                    row[index[term.vars[0][0].name]] ^= 1
                else:
                    raise ValueError("constraint {} is not linear".format(c))
        elif int(c) % 2:
            row[-1] = 1
        rows.append(row)
    # Gauss-Jordan over GF(2) on [A | b]
    pivots: list[int] = []
    r = 0
    for col in range(len(names)):
        piv = next((i for i in range(r, len(rows)) if rows[i][col]), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][col]:
                rows[i] = [a ^ b for a, b in zip(rows[i], rows[r])]
        pivots.append(col)
        r += 1
    if any(row[-1] and not any(row[:-1]) for row in rows):
        raise ValueError("the constraints have no solution")
    values = [rng.randint(0, 1) for _ in names]
    for k, col in enumerate(pivots):
        row = rows[k]
        values[col] = row[-1] ^ sum(values[j] for j in range(len(names)) if j != col and row[j]) % 2
    return dict(zip(names, values))
