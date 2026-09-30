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

"""Checking the detectors a circuit declares against the ones the tool derives.

Circuit files often declare their own detectors, such as Stim's ``DETECTOR``
lines: parities of measurement results that the author expects to take the
same value in every noiseless run. The tool derives the detectors of a
program itself (:func:`~zxqec.gpf.reduce_gpf`). :func:`check_detectors`
compares the two. Each declared detector gets one of three answers:

* ``deterministic`` -- the parity is a sum of derived detectors, so it takes
  the same value in every run the program allows; the value (0 or 1) is
  reported;
* ``not deterministic`` -- two runs that satisfy every derived detector, and
  in which the reduced diagram is not zero, give the parity different values;
  the two runs are reported;
* ``unknown`` -- the parity is not a sum of derived detectors, but no such
  pair of runs was found. With non-Clifford gates left in the diagram the
  derived detectors need not be complete, so the tool does not call such a
  detector non-deterministic (cultivation without simplifying its gadget
  layer is an example).

The derived detectors that are not sums of declared ones are listed as well:
checks the program has that the file does not declare.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ['DeclaredDetector', 'DetectorCheck', 'check_detectors']

_RESET = re.compile(r'(s\d+_)?_?r\d+')           #: PyZX's variables for the discarded value of a reset


@dataclass
class DeclaredDetector:
    """One declared detector and what the tool finds for it."""
    index: int
    results: list[str]
    status: str                                   #: 'deterministic', 'not deterministic' or 'unknown'
    value: int | None = None                      #: its value in every run, when deterministic
    witness: tuple[dict[str, int], dict[str, int]] | None = None   #: two runs with different values

    def __str__(self) -> str:
        what = " + ".join(self.results) if self.results else "(empty)"
        if self.status == 'deterministic':
            return "detector {}: {} = {} in every run".format(self.index, what, self.value)
        if self.status == 'not deterministic':
            return "detector {}: {} is NOT deterministic: it is 0 in one allowed run and 1 in another".format(
                self.index, what)
        return ("detector {}: {} is not a sum of the derived detectors, and no two allowed runs with different "
                "values were found: unknown".format(self.index, what))


@dataclass
class DetectorCheck:
    """The result of :func:`check_detectors`."""
    declared: list[DeclaredDetector]
    derived: int                                  #: independent derived detectors on the results
    undeclared: list[str] = field(default_factory=list)   #: derived detectors not sums of declared ones

    @property
    def ok(self) -> bool:
        """Every declared detector is deterministic."""
        return all(d.status == 'deterministic' for d in self.declared)

    def count(self, status: str) -> int:
        return sum(1 for d in self.declared if d.status == status)

    def describe(self) -> str:
        values = sorted({d.value for d in self.declared if d.value is not None})
        lines = ["{} declared detector(s): {} deterministic{}, {} not deterministic, {} unknown".format(
            len(self.declared), self.count('deterministic'),
            " (value {})".format(" or ".join(map(str, values))) if values else "",
            self.count('not deterministic'), self.count('unknown'))]
        lines += ["  " + str(d) for d in self.declared if d.status != 'deterministic']
        if self.undeclared:
            lines.append("the program has {} independent detectors on its results; {} of them {} not sums of "
                         "declared ones, e.g.:".format(self.derived, len(self.undeclared),
                                                       "is" if len(self.undeclared) == 1 else "are"))
            lines += ["  " + u for u in self.undeclared[:5]]
        else:
            lines.append("the program has {} independent detectors on its results, all sums of declared "
                         "ones".format(self.derived))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def check_detectors(program: Any, declared: Sequence[Sequence[Any]], gadgets: bool = True,
                    tries: int = 32, seed: int = 0, max_boundary: int = 20) -> DetectorCheck:
    """Compare ``declared`` detectors with the detectors of ``program`` (see the module docstring).

    ``declared`` is a list of detectors, each a list of measurement results given
    by name (``'c[5]'``) or by index (``5`` for ``'c[5]'``), as in
    :attr:`StimCircuit.detectors <zxqec.qprogram.StimCircuit>`. ``gadgets`` is
    passed to :func:`~zxqec.gpf.reduce_gpf`. For a parity that is not a sum of
    derived detectors, up to ``tries`` random runs with each value are drawn
    from the solutions of the derived detectors, and the reduced diagram is
    contracted in them to find one that is not zero; this is skipped (and the
    answer is ``unknown``) when the reduced diagram has more than
    ``max_boundary`` open wires.
    """
    from ..gpf import boolean_variables, reduce_gpf
    g = program.quantum_graph()
    res = reduce_gpf(g, gadgets=gadgets)
    if res.is_zero:
        raise ValueError("the program is the zero map: its derived detectors contradict each other")
    names = list(boolean_variables(g))
    for c in res.detectors:
        for v in c.free_vars():
            if v.name not in names:
                names.append(v.name)
    resets = [n for n in names if _RESET.fullmatch(n)]
    results = [n for n in names if not _RESET.fullmatch(n)]
    order = resets + results                      # resets first, so they can be eliminated
    col = {n: i for i, n in enumerate(order)}
    rows = np.array([_row(c, col) for c in res.constraints], dtype=np.uint8).reshape(-1, len(order) + 1)
    red, pivots = _rref(rows)
    # derived detectors on the results alone: echelon rows whose pivot is a result
    on_results = [(r, p) for r, p in zip(red, pivots) if p >= len(resets)]

    out: list[DeclaredDetector] = []
    rng = random.Random(seed)
    graph = res.to_graph()
    boundary = len(graph.inputs()) + len(graph.outputs())
    for i, det in enumerate(declared):
        members = [_name(x) for x in det]
        for m in members:
            if m not in col or _RESET.fullmatch(m):
                raise ValueError("{} is not a measurement result of the program".format(m))
        d = np.zeros(len(order), dtype=np.uint8)
        for m in members:
            d[col[m]] ^= 1
        value = _in_span(red, pivots, d)
        if value is not None:
            out.append(DeclaredDetector(i, members, 'deterministic', value))
            continue
        witness = None
        if boundary <= max_boundary:
            runs = []
            for target in (0, 1):
                ext = np.vstack([rows, np.concatenate([d, [target]])]).astype(np.uint8)
                found = None
                for _ in range(tries):
                    x = _random_solution(ext, rng)
                    if x is None:
                        break
                    a = {n: int(x[col[n]]) for n in order}
                    if _non_zero(graph, a):
                        found = a
                        break
                runs.append(found)
            if runs[0] is not None and runs[1] is not None:
                witness = (runs[0], runs[1])
        out.append(DeclaredDetector(i, members, 'not deterministic' if witness else 'unknown', None, witness))

    # derived detectors that are not sums of declared ones
    dec = np.zeros((0, len(order) + 1), dtype=np.uint8)
    for det_out in out:
        v = np.zeros(len(order) + 1, dtype=np.uint8)
        for m in det_out.results:
            v[col[m]] ^= 1
        dec = np.vstack([dec, v])
    undeclared = []
    for r, _p in on_results:
        lin = r.copy()
        lin[-1] = 0
        drd, dpv = _rref(dec)
        if _in_span(drd, dpv, lin[:-1]) is None:
            undeclared.append("{} = {}".format(" + ".join(order[j] for j in np.flatnonzero(r[:-1])), int(r[-1])))
            dec = np.vstack([dec, lin])
    return DetectorCheck(out, len(on_results), undeclared)


# -- GF(2) helpers -------------------------------------------------------------------------------

def _name(x: Any) -> str:
    return 'c[{}]'.format(x) if isinstance(x, (int, np.integer)) else str(x)


def _row(c: Any, col: dict[str, int]) -> list[int]:
    """A constraint ``lin(x) + const = 0`` as the row ``[lin | const]``."""
    row = [0] * (len(col) + 1)
    if not hasattr(c, 'terms'):
        row[-1] = int(c) % 2
        return row
    for coeff, term in c.terms:
        if int(coeff) % 2 == 0:
            continue
        if not term.vars:
            row[-1] ^= 1
        elif len(term.vars) == 1 and term.vars[0][1] == 1:
            row[col[term.vars[0][0].name]] ^= 1
        else:
            raise ValueError("constraint {} is not linear".format(c))
    return row


def _rref(rows: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form of ``[A | b]`` (pivots only among the columns of ``A``)."""
    m = rows.copy() % 2
    pivots: list[int] = []
    r = 0
    for c in range(m.shape[1] - 1):
        hit = next((i for i in range(r, m.shape[0]) if m[i, c]), None)
        if hit is None:
            continue
        m[[r, hit]] = m[[hit, r]]
        for i in range(m.shape[0]):
            if i != r and m[i, c]:
                m[i] ^= m[r]
        pivots.append(c)
        r += 1
    return m[:r], pivots


def _in_span(red: np.ndarray, pivots: list[int], d: np.ndarray) -> int | None:
    """If ``d`` is a sum of rows of ``A``: the sum of their constants (the parity's value). Else ``None``."""
    v = d.copy() % 2
    const = 0
    for row, p in zip(red, pivots):
        if v[p]:
            v ^= row[:-1]
            const ^= int(row[-1])
    return None if v.any() else const


def _random_solution(aug: np.ndarray, rng: random.Random) -> np.ndarray | None:
    """A uniformly random solution of ``A x = b`` for the rows ``[A | b]``, or ``None``."""
    red, pivots = _rref(aug)
    if any(not row[:-1].any() and row[-1] for row in red):
        return None
    n = aug.shape[1] - 1
    x = np.zeros(n, dtype=np.uint8)
    free = [c for c in range(n) if c not in pivots]
    for c in free:
        x[c] = rng.randint(0, 1)
    for row, p in zip(red, pivots):
        x[p] = (int(row[-1]) + int(row[:-1] @ x) - int(x[p])) % 2
    return x


def _non_zero(graph: Any, values: dict[str, int]) -> bool:
    """Whether the reduced diagram is not the zero map in the run ``values``."""
    h = graph.substitute_variables(values) if values else graph.copy()
    return bool(np.abs(h.to_tensor(preserve_scalar=True)).max() > 1e-9)
