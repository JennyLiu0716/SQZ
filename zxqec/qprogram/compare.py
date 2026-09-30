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

"""Comparing two programs block by block: a weaker, logical equivalence.

Two programs, for example a circuit as written and the same circuit after a
compiler, are compared at a level of each: the ``i``-th block of one against the
``i``-th block of the other, and then the two programs as a whole. Two blocks
agree when

* the code before them and the code after them are the same: the same
  stabilizer group, signs ignored, with qubits matched by program label or by
  physical position; and
* they perform the same logical operation up to a Pauli frame: every class of
  runs of one equals a class of runs of the other up to a Pauli on the outputs
  (or a flipped outcome of a measured part), and the other way round. The
  logical basis is read from the codes, so this is compared only when the
  codes around the block agree by program label.

The detectors of the two whole programs are compared too: the number of
independent ones, and, when the two programs name their measurement results
alike, the space they span. To show where checks differ, each pair of blocks
also lists how many of its results some detector of the whole program mentions
(so that a flip of that result alone would fire a detector), in each program;
that column does not enter the verdict.

This is weaker than equality of the two circuits: schedules, helper qubits,
gauge outcomes and the names of the results may differ, and the signs of the
stabilizers are not compared (they belong to the Pauli frame). What it asks is
what a user of a compiled program needs: the same codes, the same logical
operations, and no fewer checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ['BlockComparison', 'ProgramComparison', 'compare_programs']


@dataclass
class BlockComparison:
    """The comparison of one pair of blocks (or of the two whole programs)."""
    index: int | None                          #: ``None`` for the whole programs
    first: str
    second: str
    same_code_before: bool | None               #: ``None``: a code could not be identified
    same_code_after: bool | None
    same_action: bool | None                    #: up to a Pauli frame; ``None``: not compared
    notes: list[str] = field(default_factory=list)
    checked: tuple[int, int] | None = None      #: results of the block some detector mentions, in each

    @property
    def ok(self) -> bool:
        return all(x is True for x in (self.same_code_before, self.same_code_after, self.same_action))

    @property
    def differs(self) -> bool:
        """Something was compared and found different (``ok`` is also false when something was not compared)."""
        return any(x is False for x in (self.same_code_before, self.same_code_after, self.same_action))

    def describe(self) -> str:
        head = "whole programs" if self.index is None else "block {}: '{}' / '{}'".format(
            self.index, self.first, self.second)
        lines = [head + ("  -- same" if self.ok else "  -- DIFFERENT" if self.differs else "  -- not decided")]
        for text, value in (("code before", self.same_code_before), ("code after", self.same_code_after),
                            ("logical action", self.same_action)):
            lines.append("  {:15} {}".format(text, _word(value)))
        lines += ["  " + n for n in self.notes]
        return "\n".join(lines)


@dataclass
class ProgramComparison:
    """The result of :func:`compare_programs`."""
    blocks: list[BlockComparison]
    whole: BlockComparison
    detectors: tuple[int, int]                   #: independent detectors of each program
    same_detectors: bool | None                  #: same span; ``None`` if the results are named differently
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Logically equivalent: everything was compared and found the same."""
        return (all(b.ok for b in self.blocks) and self.whole.ok
                and self.detectors[0] == self.detectors[1] and self.same_detectors is not False)

    @property
    def differs(self) -> bool:
        """Not logically equivalent: something was found different (not just left uncompared)."""
        return (any(b.differs for b in self.blocks) or self.whole.differs
                or self.detectors[0] != self.detectors[1] or self.same_detectors is False)

    def verdict(self) -> str:
        if self.ok:
            return "logically equivalent"
        if self.differs:
            return "NOT logically equivalent"
        return "not decided: no difference found, but not everything could be compared"

    def table(self) -> str:
        rows = [("block", "code before", "code after", "logical action", "results checked")]
        for b in self.blocks + [self.whole]:
            name = "whole program" if b.index is None else "{} {}".format(b.index, b.first)
            checked = "" if b.checked is None else "{} / {}{}".format(
                b.checked[0], b.checked[1], "" if b.checked[0] == b.checked[1] else "  <-")
            rows.append((name, _word(b.same_code_before), _word(b.same_code_after), _word(b.same_action),
                         checked))
        widths = [max(len(r[i]) for r in rows) for i in range(5)]
        lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
        det = "detectors: {} and {} independent".format(*self.detectors)
        if self.same_detectors is not None:
            det += ", {} space".format("the same" if self.same_detectors else "a different")
        lines.append(det)
        lines.append(self.verdict())
        return "\n".join(lines)

    def first_difference(self) -> BlockComparison | None:
        """The first pair of blocks that differs (later ones often differ only as a consequence)."""
        return next((b for b in self.blocks if b.differs), None)

    def describe(self) -> str:
        """The table, then the details of the first differing (or else undecided) block and of the whole programs."""
        parts = [self.table()]
        first = self.first_difference()
        if first is not None:
            parts.append("first difference:\n" + first.describe())
        else:
            undecided = next((b for b in self.blocks if not b.ok), None)
            if undecided is not None:
                parts.append("first block not decided:\n" + undecided.describe())
        if not self.whole.ok or self.whole.notes:
            parts.append(self.whole.describe())
        parts += self.notes
        return "\n\n".join(parts)

    def __str__(self) -> str:
        return self.describe()


def compare_programs(first: Any, second: Any, level_first: int | None = None,
                     level_second: int | None = None, by: str = 'label') -> ProgramComparison:
    """Compare two :class:`~zxqec.qprogram.QProgram` s block by block (see the module docstring).

    ``level_first`` and ``level_second`` (default: the same as ``level_first``)
    are the levels whose blocks are paired in order; with ``level_first=None``
    only the whole programs and their detectors are compared. ``by`` is how
    qubits of the two codes are matched: ``'label'`` (program qubit) or
    ``'position'`` (physical position).
    """
    if by not in ('label', 'position'):
        raise ValueError("by must be 'label' or 'position'")
    blocks: list[BlockComparison] = []
    notes: list[str] = []
    da, db = _outcome_detectors(first), _outcome_detectors(second)
    if level_first is not None:
        level_second = level_first if level_second is None else level_second
        sa, sb = first.summary(level_first), second.summary(level_second)
        if len(sa.blocks) != len(sb.blocks):
            raise ValueError("the two levels have {} and {} blocks; they are compared block by block".format(
                len(sa.blocks), len(sb.blocks)))
        for i, (a, b) in enumerate(zip(sa.blocks, sb.blocks)):
            blk = _compare(i, a.label, b.label, a.code_in, a.code_out, a.action,
                           b.code_in, b.code_out, b.action, by)
            blk.checked = (_checked(a.measurements, da), _checked(b.measurements, db))
            blocks.append(blk)
    wa, wb = _whole(first), _whole(second)
    whole = _compare(None, 'first', 'second', wa[0], wa[1], wa[2], wb[0], wb[1], wb[2], by)
    from ..gpf import detector_rank
    ranks = (detector_rank(da), detector_rank(db))
    same: bool | None = None
    if _results(first) == _results(second):
        same = detector_rank(da + db) == ranks[0] == ranks[1]
    else:
        notes.append("the two programs name their measurement results differently, so only the numbers of "
                     "independent detectors are compared")
    return ProgramComparison(blocks, whole, ranks, same, notes)


# -- one pair ----------------------------------------------------------------------------------------

def _compare(index: int | None, name_a: str, name_b: str, in_a: Any, out_a: Any, act_a: Any,
             in_b: Any, out_b: Any, act_b: Any, by: str) -> BlockComparison:
    notes: list[str] = []
    what = "the programs" if index is None else "the block"
    before = _same_code(in_a, in_b, by, "before " + what, notes)
    after = _same_code(out_a, out_b, by, "after " + what, notes)
    action: bool | None = None
    if act_a is None or act_b is None:
        notes.append("a logical action could not be computed")
    elif not (_same_code(in_a, in_b, 'label', None, []) and _same_code(out_a, out_b, 'label', None, [])):
        notes.append("logical actions not compared: the logical basis is read from the codes, which "
                     "differ by program label")
    else:
        action, why = _same_action(act_a, act_b)
        if why:
            notes.append(why)
    return BlockComparison(index, name_a, name_b, before, after, action, notes)


def _same_code(a: Any, b: Any, by: str, where: str | None, notes: list[str]) -> bool | None:
    if a is None or b is None:
        if where:
            notes.append("the code {} could not be identified".format(where))
        return None
    if by == 'position' and (a.positions is None or b.positions is None):
        by = 'label'
    same = bool(a.same_code_as(b, by=by))
    if not same and where:
        notes.append("code {}: {}".format(where.split()[0], _difference(a, b, by)))
    return same


def _difference(a: Any, b: Any, by: str) -> str:
    """Stabilizers that one code has and the other has not (a few of each), or a different qubit set."""
    from ..codes.pauli import PauliString
    da, db = a.data(), b.data()
    ka = list(da.positions) if by == 'position' else list(da.qubits)
    kb = list(db.positions) if by == 'position' else list(db.qubits)
    if set(ka) != set(kb):
        return "on qubits {} in the first and {} in the second".format(sorted(ka), sorted(kb))
    where = [kb.index(q) for q in ka]            # qubit i of a is qubit where[i] of b

    def to_b(p: Any) -> Any:
        x, z = [0] * len(ka), [0] * len(ka)
        for i, j in enumerate(where):
            x[j], z[j] = p.x[i], p.z[i]
        return PauliString.from_arrays(x, z)

    def to_a(p: Any) -> Any:
        return PauliString.from_arrays([p.x[j] for j in where], [p.z[j] for j in where])
    only_a = [p for p in da.stabilizers if not db.contains(to_b(p))]
    only_b = [to_a(p) for p in db.stabilizers if not da.contains(to_a(p))]
    text = []
    if only_a:
        text.append("the first has {} and the second does not".format(", ".join(p.letters for p in only_a[:3])))
    if only_b:
        text.append("the second has {} and the first does not".format(", ".join(p.letters for p in only_b[:3])))
    return "; ".join(text) + " (qubits in the order {})".format(ka)


def _same_action(a: Any, b: Any) -> tuple[bool | None, str]:
    """Every class of runs of ``a`` equals one of ``b`` up to a Pauli, and the other way round."""
    from ..codes.logical import _relating_pauli
    sa, sb = a.stays_in_code(), b.stays_in_code()
    if sa is False and sb is False:
        return None, "not compared: both blocks leave their code, so neither is a logical operation"
    if sa is False or sb is False:
        return False, "the {} block leaves its code and the other does not".format("first" if sa is False else "second")
    if (a.k_in, a.k_out) != (b.k_in, b.k_out):
        return False, "{} -> {} logical qubits in the first, {} -> {} in the second".format(
            a.k_in, a.k_out, b.k_in, b.k_out)
    ma = [a.matrix(br.outcomes) for br in a.branches()]
    mb = [b.matrix(br.outcomes) for br in b.branches()]
    if not ma or not mb:
        return (not ma and not mb), "" if (not ma and not mb) else "one block is zero in every run"
    effect = sorted({i for act in (a, b) for ins, outs, _w in act.parts() if not outs for i in ins})

    def covered(xs: list, ys: list) -> bool:
        return all(any(_relating_pauli(x, y, a.k_in, a.k_out, effect) is not None for y in ys) for x in xs)
    same = covered(ma, mb) and covered(mb, ma)
    if same:
        return True, ""
    ops_a = sorted({str(br.operation) for br in a.branches()})
    ops_b = sorted({str(br.operation) for br in b.branches()})
    return False, "logical action: {} in the first, {} in the second".format(" / ".join(ops_a), " / ".join(ops_b))


# -- whole programs and detectors -------------------------------------------------------------------

def _whole(program: Any) -> tuple[Any, Any, Any]:
    try:
        act = program.logical_action()
    except (NotImplementedError, ValueError):
        return None, None, None
    return act.code_in, act.code_out, act


_RESET = re.compile(r'(s\d+_)?_?r\d+')


def _outcome_detectors(program: Any) -> list[Any]:
    """The detectors of the whole program on measurement results (not on PyZX's reset variables)."""
    from ..gpf import reduce_gpf
    res = reduce_gpf(program.quantum_graph())
    return [d for d in res.detectors if not any(_RESET.fullmatch(v.name) for v in d.free_vars())]


def _checked(report: Any, detectors: list[Any]) -> int:
    """How many results of a block some detector of the whole program mentions.

    Not the kinds ``fixed`` and ``tied`` of the report: a result the map depends
    on (``frame``, e.g. the readout that decides a correction) can be checked by
    a detector too. The results mentioned by a list of detectors are those
    mentioned by the space they span, so this does not depend on the basis.
    """
    if report is None:
        return 0
    mentioned = {v.name for d in detectors for v in d.free_vars()}
    return sum(1 for name in report.measurements if name in mentioned)


def _results(program: Any) -> set[str]:
    """The names of the program's measurement results."""
    from ..gpf import boolean_variables
    return {n for n in boolean_variables(program.quantum_graph()) if not _RESET.fullmatch(n)}


def _word(value: bool | None) -> str:
    return {True: "same", False: "DIFFERENT", None: "not compared"}[value]
