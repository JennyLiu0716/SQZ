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

"""The level-up table: for every block of a level, what it does and whether it does its job.

For each block the table gives the three answers of the paper:

1. the code the qubits are in before and after it (:func:`~zxqec.codes.code_of`);
2. what it does to the encoded qubits (:func:`~zxqec.codes.logical_action`);
3. whether it does so in every run, and what happens to each of its
   measurements (:class:`~zxqec.gpf.MeasurementReport`, read in the context of
   the whole program).

A block's *role* (the stage type the programmer declared when levelling up)
adds checks that are specific to it: an encoder must enlarge the code without
changing the logical state, a syndrome-extraction round must leave both the
code and the logical state alone and measure only stabilizers, and so on.
A failed check is reported, not raised, so that one table shows every problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ['BlockSummary', 'ProgramSummary', 'summarize', 'transversality_violations']


@dataclass
class BlockSummary:
    index: int
    label: str
    role: str | None
    stages: tuple[int, ...]
    code_in: Any
    code_out: Any
    action: Any
    measurements: Any                          #: MeasurementReport restricted to the block, or None
    checks: list[tuple[bool, str]] = field(default_factory=list)
    circuit: Any = None                        #: the block's gates, on the program's qubits

    @property
    def ok(self) -> bool:
        return all(passed for passed, _ in self.checks)

    def operation(self) -> str:
        if self.action is None:
            return "?"
        if self.action.stays_in_code() is False:
            return "leaves the code"
        ops = sorted({"not recognised" if b.operation is None else str(b.operation) for b in self.action.branches()})
        return " / ".join(ops) if ops else "zero"

    def describe(self) -> str:
        lines = ["Block {} '{}'{} -- stages {}".format(
            self.index, self.label, " ({})".format(self.role) if self.role else "",
            ", ".join(map(str, self.stages)))]
        lines.append("  code before: " + _codes(self.code_in))
        lines.append("  code after:  " + _codes(self.code_out))
        if self.action is not None:
            lines.append("  logical action: " + self.action.verdict())
        if self.measurements is not None and self.measurements.measurements:
            kinds: dict[str, int] = {}
            for m in self.measurements.measurements.values():
                kinds[m.kind] = kinds.get(m.kind, 0) + 1
            lines.append("  measurements: {} ({})".format(
                len(self.measurements.measurements),
                ", ".join("{} {}".format(v, k) for k, v in sorted(kinds.items()))))
        for passed, msg in self.checks:
            lines.append("  {} {}".format("ok     " if passed else "PROBLEM", msg))
        return "\n".join(lines)


@dataclass
class ProgramSummary:
    level: int
    blocks: list[BlockSummary]

    @property
    def ok(self) -> bool:
        return all(b.ok for b in self.blocks)

    def table(self) -> str:
        rows = [("block", "role", "code before", "code after", "logical action", "every run?", "checks")]
        for b in self.blocks:
            rows.append((
                "{} {}".format(b.index, b.label), b.role or "-",
                _short(b.code_in), _short(b.code_out), b.operation(),
                "?" if b.action is None else "yes" if b.action.is_deterministic() else "no",
                "ok" if b.ok else "{} problem(s)".format(sum(1 for p, _ in b.checks if not p))))
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)

    def __str__(self) -> str:
        return self.table() + "\n\n" + "\n\n".join(b.describe() for b in self.blocks)


def summarize(program: Any, level: int) -> ProgramSummary:
    from ..codes import NotACodeError, logical_action
    from ..codes.logical import glue
    whole = program.measurement_report()
    out = []
    for i, block in enumerate(program.levels.blocks(level)):
        g = program.block_graph(level, i, classical=False)
        before = program.prefix_graph(block.first - 1)
        names = [m.name for m in whole.measurements.values() if m.stage in block.stages]
        meas = whole.restrict(names, scope="block {}".format(i)) if names else None
        label = block.name or str(block)
        code_in = code_out = action = None
        problem = None
        try:
            code_in = program._code(before)
            code_out = program._code(glue(before, g))
            action = logical_action(g, code_in, code_out, context=whole.detectors)  # runs the program can produce
        except (NotACodeError, NotImplementedError) as e:  # e.g. a code that depends on the run
            if code_out is None:
                problem = "the code {} the block could not be identified: {}".format(
                    "before" if code_in is None else "after", e)
            else:
                problem = "the logical action could not be computed: {}".format(e)
        s = BlockSummary(i, label, block.role, block.stages, code_in, code_out, action, meas,
                         circuit=program.block_circuit(level, i))
        s.checks = [(False, problem)] if problem else _checks(s)
        out.append(s)
    return ProgramSummary(level, out)


# -- checks per role ---------------------------------------------------------------------

def _checks(b: BlockSummary) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    act = b.action
    if act.result.is_zero:
        return [(False, "the block is the zero map in this program: its detectors contradict each other")]
    ops = {br.operation for br in act.branches()}
    if act.stays_in_code() is False:
        checks.append((False, "maps part of the code before it outside the code after it, so it is not a "
                              "logical operation"))
    if None in ops:
        checks.append((False, "in some run the block does something that is not a recognised logical "
                              "operation (for example it leaves the code)"))
    role = b.role
    k_in, k_out = _k(b.code_in), _k(b.code_out)
    if role == 'encoder':
        checks.append((_n(b.code_out) > _n(b.code_in) and k_in == k_out,
                       "encodes {} logical qubit(s) into {}".format(k_in, _short(b.code_out))))
        checks.append((ops == {'I'}, "leaves the logical state unchanged"))
    elif role == 'decoder':
        checks.append((_n(b.code_out) < _n(b.code_in) and k_in == k_out,
                       "decodes {} into {} physical qubit(s)".format(_short(b.code_in), _n(b.code_out))))
        checks.append((ops == {'I'}, "leaves the logical state unchanged"))
    elif role == 'syndrome extraction':
        checks.append((b.code_in.same_code_as(b.code_out), "stays in the same code"))
        checks.append((ops == {'I'}, "leaves the logical state unchanged (up to a Pauli frame)"))
        if b.measurements is not None:
            idle = [m.name for m in b.measurements.measurements.values() if m.kind == 'gauge']
            checks.append((not idle, "every outcome is either checked again or used" if not idle else
                           "outcome(s) {} affect nothing and are never checked again: the circuit may "
                           "measure an operator that is not a stabilizer of the code".format(", ".join(idle))))
        else:
            checks.append((False, "measures nothing"))
    elif role in ('logical gate', 'transversal gate'):
        names = sorted("an operation outside the named list" if o is None else str(o) for o in ops)
        checks.append((act.is_deterministic() and len(ops) == 1,
                       "implements {} in every run".format(" / ".join(names))))
        if role == 'transversal gate' and b.circuit is not None:
            bad = transversality_violations(b.circuit, b.code_in)
            checks.append((not bad, "transversal: no two qubits of one code block are connected by its gates"
                           if not bad else "not transversal: its gates connect qubits {} of the code block on "
                           "qubits {}".format(", ".join(map(str, bad[0][0])), _qubits(bad[0][1]))))
    elif role == 'correction':
        runs = ["{} {}".format(br.operation, br.when()) for br in act.branches()]
        checks.append((None not in ops, "applies " + "; ".join(runs)))
    elif role == 'logical measurement':
        checks.append((k_out < k_in, "measures {} logical qubit(s)".format(k_in - k_out)))
    elif role == 'state preparation':
        checks.append((_n(b.code_out) > _n(b.code_in) and act.is_deterministic(),
                       "{}".format(" / ".join(sorted(map(str, ops))))))
    elif role == 'injection':
        checks.append((act.is_deterministic(), act.verdict()))
    elif role == 'code switching':
        checks.append((not b.code_in.same_code_as(b.code_out) and k_in == k_out,
                       "switches {} to {}".format(_short(b.code_in), _short(b.code_out))))
        d_in, d_out = b.code_in.distance, b.code_out.distance
        if d_in is not None and d_out is not None:
            checks.append((d_out >= d_in, "keeps the code distance ({} to {})".format(d_in, d_out)))
        checks.append((act.is_deterministic() and ops == {'I'}, "leaves the logical state unchanged in every run"))
    elif role == 'cultivation':
        # the double-check of magic-state cultivation: measure the logical qubit in the basis of
        # the magic state, with an outcome that earlier outcomes predict
        checks.append((b.code_in.same_code_as(b.code_out), "stays in the same code"))
        basis = act.measured_basis()
        checks.append((basis is not None, "measures the logical qubit in the basis {{{}}}".format(
            ", ".join(basis)) if basis else "is not a logical measurement in a named basis"))
        if basis is not None and b.measurements is not None:
            deciding = sorted({str(v) for br in act.branches() for cond, _ in br.conditions
                               for v in _variables(cond)})
            kinds = {m.name: m.kind for m in b.measurements.measurements.values()}
            unchecked = [v for v in deciding if kinds.get(v) not in ('fixed', 'tied')]
            checks.append((not unchecked,
                           "its outcome ({}) is predicted by earlier outcomes, so the check can reject a "
                           "wrong state".format(", ".join(deciding)) if not unchecked else
                           "outcome(s) {} are not predicted by earlier outcomes: the check cannot tell a "
                           "right state from a wrong one".format(", ".join(unchecked))))
    return checks


def transversality_violations(circuit: Any, code: Any) -> list[tuple[list[int], list[int]]]:
    """Where the gates of ``circuit`` are not transversal for ``code``.

    Two qubits are connected when a multi-qubit gate acts on both, or through a
    chain of such gates (a fault on one can spread to the other). The circuit is
    transversal when every connected group holds at most one qubit of each code
    block of ``code``. Returns ``(qubits, block)`` for each group and block where
    it holds two or more: the qubits of the block in the group, and the block's
    qubits. Qubits are the program's labels, as in ``code.qubits``.
    """
    parent: dict[int, int] = {}

    def find(q: int) -> int:
        while parent.setdefault(q, q) != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q
    for gate in circuit.gates:
        qs = _gate_qubits(gate)
        for q in qs[1:]:
            parent[find(q)] = find(qs[0])
    groups: dict[int, set[int]] = {}
    for q in list(parent):
        groups.setdefault(find(q), set()).add(q)
    out = []
    for blk in code.blocks() if code is not None else []:
        members = set(int(q) for q in blk.qubits)
        for grp in groups.values():
            shared = sorted(grp & members)
            if len(shared) > 1:
                out.append((shared, sorted(members)))
    return out


def _gate_qubits(gate: Any) -> list[int]:
    """The qubits a gate acts on (PyZX gates name them control/target, ctrl1/ctrl2, or targets)."""
    qs: list[int] = []
    for name in ('control', 'ctrl1', 'ctrl2', 'target'):
        q = getattr(gate, name, None)
        if isinstance(q, int) and q not in qs:
            qs.append(q)
    for q in getattr(gate, 'targets', None) or []:
        if isinstance(q, int) and q not in qs:
            qs.append(q)
    return qs


def _qubits(qs: list[int]) -> str:
    return ",".join(map(str, qs))


def _variables(text: str) -> list[str]:
    """The outcome names in a parity written as text, e.g. ``'c[5] + c[24]'``."""
    import re
    return re.findall(r'[A-Za-z_][A-Za-z0-9_]*\[\d+\]|s\d+_r\d+', text)


def _when(outcomes: dict) -> str:
    ones = sorted(k for k, v in outcomes.items() if v)
    return "the outcomes are all 0" if not ones else " = ".join(ones) + " = 1"


def _k(code: Any) -> int:
    return code.data().k


def _n(code: Any) -> int:
    return code.data().n


def _short(code: Any) -> str:
    if code is None:
        return "?"
    parts = []
    for blk in code.blocks():
        n, k, d = blk.parameters
        parts.append("[[{},{},{}]]".format(n, k, d if d is not None else '?'))
    return " + ".join(parts) if parts else "none"


def _codes(code: Any) -> str:
    if code is None:
        return "not identified"
    parts = []
    for blk in code.blocks():
        n, k, d = blk.parameters
        parts.append("[[{},{},{}]] on qubits {}{}".format(
            n, k, d if d is not None else '?', ",".join(map(str, blk.qubits)),
            " (non-Clifford logical content)" if blk.non_clifford else ""))
    return "; ".join(parts) if parts else "no qubits"
