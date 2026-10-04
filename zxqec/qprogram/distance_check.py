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

"""Distance checking over the stages of a program (L. Yeh's pipeline).

:func:`check_distance` runs three steps.

1. :func:`split_by_gate_type` cuts the stages so that each is *Clifford*, a run of
   single-qubit *non-Clifford gates*, or single-qubit *non-Clifford measurements*
   or *initialisations*. A non-Clifford gate starts a measurement when the next
   operations on its qubit are single-qubit gates and then a measurement (the stage
   holds all of them), and ends an initialisation when the operations before it on
   its qubit are single-qubit gates after a reset or the qubit's initial state.
   Otherwise gates keep their order: a run is a stretch of consecutive gates of one
   kind, so the cuts follow the gate list (Stim's ``T_DAG 0 3 6`` is one run).
2. Every maximal run of Clifford stages gets its circuit distance,
   :func:`~zxqec.qprogram.circuit_distance` (between noiseless boundaries; needs
   ``stim`` and ``codedistance``). A run whose distance cannot be computed is
   reported with the reason.
3. Starting with the first non-Clifford gate stage, the stages from it up to and
   including the next one form a group. The group's diagram is reduced
   (:func:`~zxqec.gpf.reduce_gpf`) and split into terms
   (:func:`~zxqec.gpf.decompose_terms`), and the check is that every term maps each
   Pauli error to the same Pauli, in every run checked (all runs its detectors
   allow, or a seeded sample of ``max_runs`` when there are more). The hubs of phase
   gadgets are not split: split at its hub a gadget is a sum of Pauli strings, so
   the check would pass for any phase (a bare ``T`` splits into ``I`` and ``Z``,
   which keep every Pauli up to a sign). If it passes, the
   next group starts at the next non-Clifford gate stage after it; if not, the group
   is extended to the next non-Clifford gate stage (or, after the last one, to the
   end of the program) and checked again. A group that cannot be extended any
   further and does not pass is reported as failing. ``groups=`` gives the groups
   by hand instead, for protocols other than cultivation, where the user is
   expected to know which stages together preserve the distance.

Non-Clifford measurements and initialisations are post-selections: inside a group
they are part of its diagram (the check leaves effects and states out); outside
every group they are listed as not counted. The check is not proved to decide the
distance; on the d = 3 and d = 5 cultivation double-checks it agrees with the known
fault distances (Chan et al., arXiv:2609.17706).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Sequence

from pyzx.circuit import Circuit
from pyzx.spacetime import used_qubits

from .feedforward import ParityControlled
from .program import QProgram, Stage

__all__ = ['KINDS', 'SplitProgram', 'split_by_gate_type', 'GroupCheck', 'DistanceCheck', 'check_distance']

KINDS = ('Clifford', 'non-Clifford gates', 'non-Clifford measurement', 'non-Clifford initialisation')

_MEASURE = ('Measurement', 'PostSelect')
_PREPARE = ('Reset', 'InitAncilla')


def _clifford_phase(p: Any) -> bool:
    return (2 * Fraction(p).limit_denominator(10 ** 6)).denominator == 1


def _role(gate: Any) -> str:
    """``'measure'``, ``'prepare'``, ``'Clifford'`` or ``'non-Clifford'`` (a single-qubit non-Clifford gate)."""
    name = type(gate).__name__
    if name in _MEASURE:
        return 'measure'
    if name in _PREPARE:
        return 'prepare'
    if name == 'DiscardBit':
        return 'Clifford'
    if isinstance(gate, ParityControlled):
        return 'Clifford' if _clifford_phase(gate.angle) else 'non-Clifford'
    qubits = used_qubits(gate)
    try:
        c = Circuit(max(qubits) + 1 if qubits else 1)
        c.add_gate(gate)
        g = c.to_graph()
        clifford = all(_clifford_phase(g.phase(v)) for v in g.vertices())
    except Exception as e:                               # e.g. a symbolic phase
        raise ValueError("cannot tell whether {} is Clifford".format(gate)) from e
    if clifford:
        return 'Clifford'
    if len(qubits) > 1:
        raise NotImplementedError("{} is a non-Clifford gate on several qubits; the pipeline cuts stages at "
                                  "single-qubit non-Clifford gates only".format(gate))
    return 'non-Clifford'


def _label(gate: Any) -> str:
    if isinstance(gate, ParityControlled):
        return 'parity-controlled ' + gate.gate
    if type(gate).__name__ == 'T':
        return 'T^dag' if getattr(gate, 'adjoint', False) else 'T'
    return str(gate)


@dataclass
class SplitProgram:
    """The result of :func:`split_by_gate_type`."""
    program: QProgram                    #: the stages cut by gate type
    kinds: tuple[str, ...]               #: the kind of each stage, one of :data:`KINDS`
    origin: tuple[int, ...]              #: the stage of the original program each stage comes from

    def describe(self) -> str:
        lines = ['{} stages, cut by gate type:'.format(len(self.kinds))]
        for i, (kind, o) in enumerate(zip(self.kinds, self.origin)):
            lines.append('  {:>3}  {:28} {}'.format(i, kind, self.program.stage(i)))
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.describe()


def _bits_written(gate: Any) -> set[Any]:
    bit = getattr(gate, 'result_bit', None)
    return set() if bit is None else {bit}


def _bits_read(gate: Any) -> set[Any]:
    if isinstance(gate, ParityControlled):
        return {int(b[2:-1]) if b.startswith('c[') else b for b in gate.bits}
    return set()


def split_by_gate_type(program: QProgram) -> SplitProgram:
    """The program with its circuit stages cut by gate type (step 1 of the module docstring).

    A non-Clifford measurement stage holds the whole single-qubit measurement: the
    non-Clifford gate, the single-qubit gates after it and the measurement; an
    initialisation stage the reset, the single-qubit gates and the non-Clifford
    gate. They are gathered at the place of the non-Clifford gate, past gates on
    other qubits, which commute with them; if a gate in between reads or writes a
    classical bit the move would reorder, nothing is moved and the non-Clifford
    gate counts as a gate. The new program has the same qubits, bits, topology,
    mapping, durations and initial states and is the same map; only the scheduling,
    the stages gates fall in, and the names of reset variables (which carry their
    stage) change. Permutation stages are kept and count as Clifford.
    """
    stages = [program.stage(s) for s in range(program.num_stages)]
    seq = [(s, g) for s, st in enumerate(stages) if st.kind == 'circuit' for g in st.circuit.gates]
    roles = [_role(g) for _s, g in seq]
    initial = set(program._initial_states)

    def chain(i: int, step: int) -> tuple[list[int], str | None]:
        """From gate ``i``, the single-qubit gates on its qubit in direction ``step`` up to a measurement
        (forwards) or a reset (backwards): their indices and what ends the chain."""
        (q,) = tuple(used_qubits(seq[i][1]))
        out, j = [], i + step
        while 0 <= j < len(seq):
            qs = used_qubits(seq[j][1])
            if q in qs:
                if len(qs) != 1:
                    return out, None
                out.append(j)
                if roles[j] == ('measure' if step > 0 else 'prepare'):
                    return out, roles[j]
            j += step
        return out, ('start' if step < 0 and q in initial else None)

    def movable(unit: list[int], anchor: int) -> bool:
        """Gathering ``unit`` at ``anchor`` reorders no classical bit it reads or writes."""
        for k in unit:
            lo, hi = sorted((k, anchor))
            mine_r, mine_w = _bits_read(seq[k][1]), _bits_written(seq[k][1])
            for j in range(lo + 1, hi):
                if j in unit:
                    continue
                other = seq[j][1]
                if (mine_w & (_bits_read(other) | _bits_written(other))) or (mine_r & _bits_written(other)):
                    return False
        return True

    kind_at: dict[int, str] = {}          # anchor index -> kind of its unit
    unit_at: dict[int, list[int]] = {}    # anchor index -> the gates of its unit, in order
    taken: set[int] = set()
    for i, r in enumerate(roles):
        if r != 'non-Clifford' or i in taken:
            continue
        fwd, f_end = chain(i, 1)
        bwd, b_end = chain(i, -1)
        unit, kind = [i], 'non-Clifford gates'
        if f_end == 'measure':
            unit = (sorted(bwd) if b_end in ('prepare', 'start') else []) + [i] + fwd
            kind = 'non-Clifford measurement'
        elif b_end in ('prepare', 'start'):
            unit, kind = sorted(bwd) + [i], 'non-Clifford initialisation'
        if len(unit) > 1 and (not movable(unit, i) or taken & set(unit)):
            unit, kind = [i], 'non-Clifford gates'
        kind_at[i], unit_at[i] = kind, unit
        taken |= set(unit)

    new = QProgram(program.qubits, bits=program.bits, topology=program.topology,
                   mapping=dict(enumerate(program.mapping)), gate_durations=program.gate_durations,
                   initial_states=dict(program._initial_states))
    kinds: list[str] = []
    origin: list[int] = []
    position = {id(g): k for k, (_s, g) in enumerate(seq)}
    for s, st in enumerate(stages):
        if st.kind != 'circuit':
            new._append(Stage(len(new._stages), st.kind, st.circuit.copy(), st.mapping, perm=st.perm,
                              name=st.name, visible=st.visible))
            kinds.append('Clifford')
            origin.append(s)
            continue
        runs: list[tuple[str, list[Any]]] = []
        for g in st.circuit.gates:
            k = position[id(g)]
            if k in unit_at:
                kind, gates = kind_at[k], [seq[j][1] for j in unit_at[k]]
            elif k in taken:
                continue                                     # gathered with its non-Clifford gate
            else:
                kind, gates = 'Clifford', [g]
            if runs and runs[-1][0] == kind:
                runs[-1][1].extend(gates)
            else:
                runs.append((kind, list(gates)))
        if not runs:
            runs = [('Clifford', [])]
        n_clifford = sum(k == 'Clifford' for k, _ in runs)
        part = 0
        for k, gs in runs:
            c = Circuit(program.qubits, bit_amount=program.bits)
            for g in gs:
                c.add_gate(g.copy())
            if len(runs) == 1:
                name = str(st)
            elif k == 'Clifford':
                part += 1
                name = '{} (part {} of {})'.format(st, part, n_clifford) if n_clifford > 1 else str(st)
            else:
                qubits = sorted({q for g in gs for q in used_qubits(g)})
                labels = sorted({_label(g) for g in gs if _role(g) == 'non-Clifford'})
                name = '{}: {} on {}{}'.format(st, '/'.join(labels), ', '.join(map(str, qubits)),
                                               '' if k == 'non-Clifford gates' else ' ({})'.format(k.split()[-1]))
            new._append(Stage(len(new._stages), 'circuit', c, st.mapping, name=name))
            kinds.append(k)
            origin.append(s)
    return SplitProgram(new, tuple(kinds), tuple(origin))


@dataclass
class GroupCheck:
    """One group of stages checked in step 3."""
    first: int
    last: int
    passes: bool | None                  #: every term maps each Pauli error to the same Pauli, in every run checked
    runs: int = 0                        #: runs checked
    all_runs: bool = True                #: whether those were all the runs the detectors allow
    of_runs: int = 0                     #: how many runs the detectors allow
    single_qubit: bool | None = None     #: every term of the non-Clifford parts single-qubit, in every run checked
    inputs: int = 0
    outputs: int = 0
    note: str = ''

    def describe(self) -> str:
        if self.note and self.passes is None and not self.runs:
            return 'stages {}-{}: {} ({})'.format(self.first, self.last, self.passes, self.note)
        runs = ('the only run' if self.runs == 1 else 'all {} runs'.format(self.runs)) if self.all_runs else \
            '{} sampled runs of {}'.format(self.runs, self.of_runs)
        return 'stages {}-{}: {} ({}; {} inputs, {} outputs; non-Clifford parts single-qubit: {}){}'.format(
            self.first, self.last, self.passes, runs, self.inputs, self.outputs, self.single_qubit,
            '; ' + self.note if self.note else '')


@dataclass
class DistanceCheck:
    """The result of :func:`check_distance`."""
    split: SplitProgram
    clifford: list[tuple[int, int, Any]]          #: (first, last, CircuitDistance or the reason it was not computed)
    groups: list[GroupCheck]                      #: the final groups: passed, or the last attempt that failed
    attempts: list[GroupCheck] = field(default_factory=list)   #: every group checked, in order
    not_counted: list[int] = field(default_factory=list)       #: post-selection stages outside every group

    @property
    def passes(self) -> bool | None:
        """Every group passes (True when there is no non-Clifford gate stage); False if one fails."""
        values = [g.passes for g in self.groups]
        return False if False in values else (None if None in values else True)

    @property
    def clifford_distance(self) -> int | None:
        """The smallest circuit distance found for the Clifford runs (None if none was computed)."""
        found = [r.distance for _f, _l, r in self.clifford if hasattr(r, 'distance') and r.distance is not None]
        return min(found) if found else None

    def describe(self) -> str:
        lines = [self.split.describe(), 'Clifford stages, circuit distance (noiseless boundaries):']
        for first, last, r in self.clifford:
            inside = next((g for g in self.groups if g.first <= first and last <= g.last), None)
            what = r if isinstance(r, str) else str(r).replace('full circuit distance of stages {}-{}: '.format(
                first, last), '')
            lines.append('  stages {}-{}: {}{}'.format(first, last, what, '' if inside is None else
                                                       ' (inside the group {}-{})'.format(inside.first, inside.last)))
        if not self.clifford:
            lines.append('  (none)')
        lines.append('non-Clifford groups, every term maps each Pauli error to the same Pauli:')
        for a in self.attempts:
            lines.append('  ' + a.describe())
        if not self.attempts:
            lines.append('  (no non-Clifford gate stage)')
        if self.not_counted:
            lines.append('not counted (non-Clifford measurements and initialisations outside the groups): '
                         'stages {}'.format(', '.join(map(str, self.not_counted))))
        lines.append('groups pass: {}; smallest distance of the Clifford runs: {}'.format(
            self.passes, self.clifford_distance))
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.describe()


def _check_group(program: QProgram, first: int, last: int, labelling: str, max_runs: int, rng: random.Random,
                 max_terms: int, max_legs: int) -> GroupCheck:
    from ..gpf import (boolean_variables, consistent_assignments, decompose_terms, detector_rank, reduce_gpf,
                       sample_consistent_assignment)

    g = program.stages_graph(first, last, classical=False)
    check = GroupCheck(first, last, None, inputs=len(g.inputs()), outputs=len(g.outputs()))
    res = reduce_gpf(g)
    if res.is_zero:
        check.note = 'the diagram is zero in every run'
        return check
    names = sorted(set(boolean_variables(g)) | {str(v) for c in res.constraints for v in c.free_vars()})
    free = len(names) - detector_rank(res.constraints)
    check.of_runs = 2 ** free
    if check.of_runs <= max_runs:
        runs = consistent_assignments(res.constraints, names)
    else:
        runs = [sample_consistent_assignment(res.constraints, names, rng) for _ in range(max_runs)]
        check.all_runs = False
    check.runs = len(runs)
    same, single = [], []
    for run in runs:
        d = decompose_terms(res, run, labelling, split_gadgets=False, max_terms=max_terms, max_legs=max_legs)
        same.append(d.same_error_action)
        single.append(d.non_clifford_single_qubit)
    check.passes = False if False in same else (None if None in same else True)
    check.single_qubit = False if False in single else (None if None in single else True)
    return check


def check_distance(program: QProgram, *, distance: bool = True, kind: str = 'full',
                   groups: Sequence[tuple[int, int]] | None = None, labelling: str = 'fewest links',
                   max_runs: int = 256, seed: int = 0, max_terms: int = 1024,
                   max_legs: int = 8, **distance_params: Any) -> DistanceCheck:
    """L. Yeh's distance-checking pipeline on ``program`` (see the module docstring).

    ``distance=False`` skips the circuit distances of step 2 (which need ``stim`` and
    ``codedistance``); ``kind`` and ``distance_params`` go to
    :func:`~zxqec.qprogram.circuit_distance`. ``groups`` are stage ranges of the
    *split* program to check instead of the automatic grouping (run
    :func:`split_by_gate_type` first to see its stages). ``labelling``,
    ``max_terms`` and ``max_legs`` go to :func:`~zxqec.gpf.decompose_terms` (gadget
    hubs are never split, see the module docstring); a group with more than ``max_runs`` runs is
    checked on that many sampled runs (random seed ``seed``).
    """
    from .distance import circuit_distance

    split = split_by_gate_type(program)
    p, kinds = split.program, split.kinds
    last_stage = len(kinds) - 1

    clifford: list[tuple[int, int, Any]] = []
    i = 0
    while i <= last_stage:
        if kinds[i] != 'Clifford':
            i += 1
            continue
        j = i
        while j + 1 <= last_stage and kinds[j + 1] == 'Clifford':
            j += 1
        if distance:
            try:
                clifford.append((i, j, circuit_distance(p, i, j, kind, **distance_params)))
            except Exception as e:                       # reported, not hidden
                clifford.append((i, j, 'not computed: {}: {}'.format(type(e).__name__, e)))
        i = j + 1

    rng = random.Random(seed)
    args = (labelling, max_runs, rng, max_terms, max_legs)
    attempts: list[GroupCheck] = []
    final: list[GroupCheck] = []
    if groups is not None:
        for first, last in groups:
            c = _check_group(p, first, last, *args)
            attempts.append(c)
            final.append(c)
    else:
        gates = [s for s, k in enumerate(kinds) if k == 'non-Clifford gates']
        a = 0
        while a < len(gates):
            first = gates[a]
            b = a + 1
            while True:
                last = gates[b] if b < len(gates) else last_stage
                c = _check_group(p, first, last, *args)
                attempts.append(c)
                if c.passes or last == last_stage:
                    final.append(c)
                    break
                b += 1                                   # extend to the next non-Clifford gate stage
            a = next((k for k, s in enumerate(gates) if s > last), len(gates))
    covered = {s for c in final for s in range(c.first, c.last + 1)}
    not_counted = [s for s, k in enumerate(kinds) if k in KINDS[2:] and s not in covered]
    return DistanceCheck(split, clifford, final, attempts, not_counted)
