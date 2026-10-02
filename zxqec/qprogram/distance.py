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

"""Circuit distance of Clifford stages, with Stim and Mark Webster's ``codedistance`` library.

The *circuit distance* of some consecutive stages of a program is the smallest
number of faults, under a circuit-level noise model, that changes the logical
information carried through them while no detector notices. It is a property of
the circuit, not only of the codes: question 1 gives the distance of the code at
every cut, which a bad gate schedule can undercut.

:func:`circuit_distance` writes the stages as a Stim circuit:

1. noiseless products of Paulis (``MPP``) measuring the stabilizers of the code
   before the stages and one of its logical operators, which puts Stim's initial
   ``|0...0>`` into the code, and the Pauli of every fixed qubit there (e.g. a
   helper prepared in ``|+>`` that the stages use later); if the stages start the
   program, its initial states are prepared instead;
2. the stages, gate by gate, with noise after every operation:
   ``'full'``: ``DEPOLARIZE1`` / ``DEPOLARIZE2`` after one- and two-qubit gates,
   ``X_ERROR`` after every reset and before every measurement; ``'Z'``: only
   ``Z_ERROR`` after every gate and reset (Z faults); ``'X'``: only ``X_ERROR``
   after every gate and reset and before every measurement (X faults);
3. noiseless ``MPP`` measuring the stabilizers of the code after the stages and
   one logical operator (X, Y or Z).

The detectors and the observable are not written by hand: the noiseless circuit
is read back with :func:`~zxqec.qprogram.program_from_stim` and reduced, and its
detectors (every parity of results that is the same in every run) become
``DETECTOR`` lines; the one that contains the final logical measurement becomes
the observable. This works whatever the stages do to the logical qubit, a
logical Hadamard included, and Stim checks that every detector and the
observable are deterministic. :func:`codedistance.circuitDistance` then finds
the lightest undetectable fault set (by default with ``'MIPDist'``, an exact
integer program; ``codedistance``'s Chromobius filter is switched off). Each
input logical (X, then Z) is paired with every output logical whose final value
is deterministic, and the smallest distance is reported.

The stages must be Clifford: a ``T`` gate, or a correction controlled by
outcomes other than a Pauli, raises :class:`NotCliffordError`. The Z and X
distances need *phase-free* stages (CNOTs, resets, measurements and Paulis; a
Hadamard only as part of an X-basis reset or measurement), else
:class:`NotPhaseFreeError` is raised. One logical qubit after the stages is
supported.

**Non-Clifford circuits.** A ``Z`` fault commutes with ``T`` and ``T^dag``, so in
a phase-free circuit whose only non-Clifford gates are single-qubit ``T``-type
gates, a Z fault flips the same results as in the circuit with those gates
removed (:func:`without_gates`). The Z distance of the circuit is then the Z
distance of the circuit without them, measured against the logical operators of
the code the real circuit ends in (``code_out``; for cultivation, the logical
X), provided that the two circuits have the same detectors and that no Z fault
can trigger a fed-forward X correction. :func:`z_distance_through_t` checks both
and refuses otherwise. The Z distance is an upper bound on the full circuit
distance. The X distance found by removing the gates is not the X distance of
the real circuit, since an X fault before a ``T`` becomes a Clifford error.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pyzx.circuit import Circuit

from .program import QProgram

__all__ = ["circuit_distance", "z_distance_through_t", "CircuitDistance", "Fault", "without_gates",
           "NotCliffordError", "NotPhaseFreeError", "TRemovalError"]

KINDS = ('full', 'Z', 'X')


class NotCliffordError(ValueError):
    """The stages contain a gate that Stim cannot simulate (non-Clifford, or a non-Pauli conditional gate)."""


class NotPhaseFreeError(ValueError):
    """A Z or X distance was asked of stages with a Hadamard or S gate that mixes X and Z faults."""


class TRemovalError(ValueError):
    """The Z distance of stages without their T gates is not shown to be the Z distance of the stages."""


@dataclass(frozen=True)
class Fault:
    """One fault of a fault set: the stage, the operation it follows (or the measurement it flips), the Pauli."""
    stage: int
    operation: str
    pauli: str

    def __str__(self) -> str:
        return "stage {}: {} {}".format(self.stage, self.pauli, self.operation)


@dataclass
class CircuitDistance:
    """The circuit distance of some stages under one kind of fault."""
    kind: str                                         #: 'full', 'Z' or 'X'
    first: int
    last: int
    distance: int | None                              #: smallest over the logical pairs; None if no fault set flips one
    by_logical: dict[tuple[str, str], int | None] = field(default_factory=dict)   #: (input, output logical) -> distance
    circuits: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)  #: the annotated Stim circuits
    detectors: dict[tuple[str, str], int] = field(default_factory=dict)  #: number of detectors in each circuit
    method: str = 'MIPDist'

    def explain(self, key: tuple[str, str] | None = None) -> list[Fault]:
        """A lightest undetectable fault set, found by Stim's own search in the circuit of ``key``.

        ``key`` is an (input, output logical) pair of :attr:`by_logical`; by default
        one with the smallest distance. Stim's search is exhaustive over the fault
        sets it explores; its result has :attr:`distance` faults in the cases
        tested, and is a fault set that no detector sees in any case. The circuit
        has a ``TICK`` before every stage, so each fault's tick gives its stage.
        """
        import stim
        if key is None:
            key = min((k for k, d in self.by_logical.items() if d is not None), key=lambda k: self.by_logical[k])
        qc = stim.Circuit(self.circuits[key])
        found = qc.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=max(4, (self.by_logical[key] or 0) + 1),
            dont_explore_edges_with_degree_above=9999, dont_explore_edges_increasing_symptom_degree=False)
        out = []
        for err in found:
            loc = err.circuit_error_locations[0]
            at = loc.stack_frames[0].instruction_offset
            qubits = ' '.join(str(t.gate_target.value) for t in loc.instruction_targets.targets_in_range)
            after = qc[at + 1] if at + 1 < len(qc) else None
            if qc[at].name == 'X_ERROR' and after is not None and after.name == 'M':
                operation = 'before M {} (flips its result)'.format(qubits)
            else:                                       # every gate is followed by its own noise line
                gate = qc[at - 1]
                operation = 'after {} {}'.format(gate.name, ' '.join(str(t.value) for t in gate.targets_copy()))
            stage = self.first + loc.tick_offset - 1
            out.append(Fault(stage, operation, ' '.join(str(p) for p in loc.flipped_pauli_product)))
        return out

    def __str__(self) -> str:
        pairs = ", ".join("{} -> {}: {}".format(a, b, d) for (a, b), d in self.by_logical.items())
        return "{} circuit distance of stages {}-{}: {} ({})".format(self.kind, self.first, self.last,
                                                                     self.distance, pairs)


# -- the stages as Stim text ---------------------------------------------------------------

def _qubits_of(g: Any) -> tuple[int, ...]:
    if type(g).__name__ in ('CNOT', 'CZ'):
        return (g.control, g.target)
    return (g.target,)


def _check_phase_free(program: QProgram, first: int, last: int) -> None:
    sequence = [(s, g) for s in range(0, last + 1) for g in program.stage_circuit(s).gates]
    for i, (s, g) in enumerate(sequence):
        if s < first:
            continue
        name = type(g).__name__
        if name == 'S':
            raise NotPhaseFreeError("stage {}: {} mixes X and Z faults".format(s, g))
        if name != 'HAD':
            continue
        q = g.target
        before = next((h for _, h in reversed(sequence[:i]) if q in _qubits_of(h)), None)
        after = next((h for _, h in sequence[i + 1:] if q in _qubits_of(h)), None)
        prepares = type(before).__name__ == 'Reset' or before is None     # after a reset, or the initial |0>
        if not prepares and type(after).__name__ != 'Measurement':
            raise NotPhaseFreeError("stage {}: a Hadamard on qubit {} that is not part of an X-basis reset or "
                                    "measurement mixes X and Z faults".format(s, q))


def _result_name(m: Any) -> str:
    return m.result_symbol if m.result_symbol is not None else 'c[{}]'.format(m.result_bit)


def _stage_lines(program: QProgram, first: int, last: int, kind: str, p: float,
                 keep: tuple[str, ...] = ()) -> list[tuple[int, Any]]:
    """(stage, line) for stages first..last with noise; measurements and fed-forward Paulis as tuples.

    ``keep=('T',)`` writes ``T`` / ``T_DAG`` gates (without noise) instead of refusing them; such a circuit is
    only read back by zxqec, never given to Stim.
    """
    out: list[Any] = []
    where: list[int] = []

    def noise1(q: int, flips: bool = False) -> None:
        if kind == 'full':
            out.append('X_ERROR({}) {}'.format(p, q) if flips else 'DEPOLARIZE1({}) {}'.format(p, q))
        else:
            out.append('{}_ERROR({}) {}'.format(kind, p, q))

    for s in range(first, last + 1):
        for g in program.stage_circuit(s).gates:
            where.extend([s] * (len(out) - len(where)))
            name = type(g).__name__
            if name in ('HAD', 'NOT', 'Z') or (name == 'S'):
                op = {'HAD': 'H', 'NOT': 'X', 'Z': 'Z'}.get(name) or ('S_DAG' if g.adjoint else 'S')
                out.append('{} {}'.format(op, g.target))
                noise1(g.target)
            elif name in ('CNOT', 'CZ'):
                out.append('{} {} {}'.format('CX' if name == 'CNOT' else 'CZ', g.control, g.target))
                if kind == 'full':
                    out.append('DEPOLARIZE2({}) {} {}'.format(p, g.control, g.target))
                else:
                    noise1(g.control)
                    noise1(g.target)
            elif name == 'Reset':
                out.append('R {}'.format(g.target))
                noise1(g.target, flips=True)
            elif name == 'Measurement':
                if kind in ('full', 'X'):
                    out.append('X_ERROR({}) {}'.format(p, g.target))
                out.append(('M', g.target, _result_name(g)))
            elif name == 'ParityControlled':
                if g.gate not in ('X', 'Z'):
                    raise NotCliffordError("stage {}: {} is a correction Stim cannot express (only Paulis fed forward "
                                           "from outcomes)".format(s, g))
                out.append(('FF', g.gate, g.target, g.bits))
            elif name in keep:
                out.append('{} {}'.format('T_DAG' if g.adjoint else 'T', g.target))
            else:
                raise NotCliffordError("stage {}: {} is not a Clifford gate this function supports".format(s, g))
        where.extend([s] * (len(out) - len(where)))
    return list(zip(where, out))


def _mpp(letters: str, labels: Sequence[int]) -> str:
    return '*'.join('{}{}'.format(a, q) for a, q in zip(letters, labels) if a != 'I')


def _fixed_letter(state: str) -> str:
    """The Pauli whose eigenstate a fixed qubit is in, from its label in :meth:`StabilizerCode.fixed_qubits`."""
    if state in ('0', '1'):
        return 'Z'
    if state in ('+', '-'):
        return 'X'
    if state in ('+i', '-i') or state.endswith(') i'):
        return 'Y'
    return 'X' if state.startswith('(-1)^(') else 'Z'    # a sign that depends on outcomes


def _circuit(program: QProgram, first: int, last: int, kind: str, p: float, code_in: Any, code_out: Any,
             in_letters: str | None, out_letters: str, fixed_in: dict[Any, str],
             keep: tuple[str, ...] = ()) -> tuple[str, str, list[str]]:
    """(noiseless text, noisy text, the name of every measurement record in order); a TICK before every stage."""
    records: list[str] = []
    clean: list[str] = []
    noisy: list[str] = []

    def both(line: str) -> None:
        clean.append(line)
        noisy.append(line)

    if first == 0:
        for q, state in sorted(program._initial_states.items()):
            both('R {}'.format(q))
            if state == '+':
                both('H {}'.format(q))
    for i, (q, state) in enumerate(sorted(fixed_in.items())):
        both('MPP {}{}'.format(_fixed_letter(state), int(q)))      # its sign is recorded, as for the stabilizers
        records.append('in_fixed_{}'.format(i))
    labels_in = [int(q) for q in code_in.qubits]
    for i, s in enumerate(code_in.stabilizers):
        both('MPP ' + _mpp(s.letters, labels_in))
        records.append('in_stabilizer_{}'.format(i))
    if in_letters is not None:
        both('MPP ' + _mpp(in_letters, labels_in))
        records.append('in_logical')
    stage = None
    for s, line in _stage_lines(program, first, last, kind, p, keep):
        while stage is None or stage < s:                 # a TICK before every stage, empty ones included
            stage = first if stage is None else stage + 1
            both('TICK')
        if isinstance(line, str):
            noisy.append(line)
            if not line.startswith(('DEPOLARIZE', 'X_ERROR', 'Z_ERROR')):
                clean.append(line)
        elif line[0] == 'M':
            both('M {}'.format(line[1]))
            records.append(line[2])
        else:
            _, pauli, target, bits = line
            for b in bits:
                if b not in records:
                    raise NotCliffordError("a correction on qubit {} is controlled by {}, which is not measured in "
                                           "these stages".format(target, b))
                back = len(records) - max(i for i, r in enumerate(records) if r == b)
                both('C{} rec[-{}] {}'.format(pauli, back, target))
    while stage is None or stage <= last:                # TICKs for trailing empty stages, then the output boundary
        stage = first if stage is None else stage + 1
        both('TICK')
    labels_out = [int(q) for q in code_out.qubits]
    for i, s in enumerate(code_out.stabilizers):
        both('MPP ' + _mpp(s.letters, labels_out))
        records.append('out_stabilizer_{}'.format(i))
    both('MPP ' + _mpp(out_letters, labels_out))
    records.append('out_logical')
    return '\n'.join(clean) + '\n', '\n'.join(noisy) + '\n', records


def _detector_rows(clean: str, n: int) -> list[list[int]]:
    """The detectors zxqec derives from the noiseless circuit ``clean``, as rows over its ``n`` results."""
    from ..gpf import reduce_gpf
    from .stim_text import program_from_stim
    program, _ = program_from_stim(clean)
    rows = []
    for d in reduce_gpf(program.quantum_graph()).detectors:
        names = {v.name for v in d.free_vars()}
        if not all(x.startswith('c[') for x in names):
            continue                                      # constraints on PyZX's reset variables
        row = [0] * n
        for coeff, term in d.terms:
            if term.vars and int(coeff) % 2:
                row[int(term.vars[0][0].name[2:-1])] ^= 1
        if any(row):
            rows.append(row)
    return rows


def _split_off(rows: list[list[int]], col: int) -> tuple[list[int] | None, list[list[int]]]:
    """(a row with a 1 in ``col``, or None; the rows combined with it so that no other has a 1 there)."""
    pivot = next((r for r in rows if r[col]), None)
    if pivot is None:
        return None, rows
    return pivot, [[a ^ b for a, b in zip(r, pivot)] if r[col] else r for r in rows if r is not pivot]


def _rank(rows: list[list[int]]) -> int:
    basis: list[int] = []
    for r in rows:
        v = int(''.join(map(str, r)) or '0', 2)
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)


def _annotated(clean: str, noisy: str, records: list[str]) -> tuple[str | None, int]:
    """``noisy`` with the detectors and the observable that zxqec derives from ``clean``; None if the final
    logical measurement is random (then it cannot be an observable)."""
    n = len(records)
    rows = _detector_rows(clean, n)
    pivot, detectors = _split_off(rows, records.index('out_logical'))
    if pivot is None:
        return None, len(rows)
    rec = lambda row: ' '.join('rec[-{}]'.format(n - i) for i, b in enumerate(row) if b)
    lines = [noisy.rstrip('\n')] + ['DETECTOR ' + rec(r) for r in detectors if any(r)]
    lines.append('OBSERVABLE_INCLUDE(0) ' + rec(pivot))
    return '\n'.join(lines) + '\n', sum(1 for r in detectors if any(r))


def _letters(code: Any, which: str) -> str:
    x, z = code.logical_x[0].letters, code.logical_z[0].letters
    if which == 'X':
        return x
    if which == 'Z':
        return z
    return ''.join({('X', 'Z'): 'Y', ('X', 'I'): 'X', ('I', 'Z'): 'Z', ('I', 'I'): 'I'}[(a, b)] for a, b in zip(x, z))


# -- the distance ----------------------------------------------------------------------------

def circuit_distance(program: QProgram, first: int = 0, last: int | None = None, kind: str | Sequence[str] = 'full',
                     p: float = 1e-3, method: str = 'MIPDist', code_in: Any = None, code_out: Any = None,
                     params: dict[str, Any] | None = None) -> Any:
    """The circuit distance of stages ``first`` to ``last`` (see the module docstring).

    ``kind`` is ``'full'``, ``'Z'`` or ``'X'``, or a sequence of them (then a dict
    ``kind -> CircuitDistance`` is returned). ``code_in`` / ``code_out`` default to
    the codes the program is in before and after the stages; give ``code_out``
    when the logical to protect is not that of the stages' own output, as for
    a circuit with its ``T`` gates removed. ``p`` only sets the probabilities in
    the Stim circuit; ``method`` and ``params`` go to
    :func:`codedistance.circuitDistance`. Needs the ``stim`` and ``codedistance``
    packages.
    """
    if not isinstance(kind, str):
        return {k: circuit_distance(program, first, last, k, p, method, code_in, code_out, params) for k in kind}
    if kind not in KINDS:
        raise ValueError("kind must be one of {}, got {!r}".format(KINDS, kind))
    try:
        import codedistance
        import stim
    except ImportError as e:
        raise ImportError("circuit_distance needs the packages stim and codedistance "
                          "(pip install codedistance)") from e
    last = program.num_stages - 1 if last is None else last
    if not 0 <= first <= last < program.num_stages:
        raise ValueError("stages {}..{} are not in the program's {} stages".format(first, last, program.num_stages))
    _stage_lines(program, first, last, kind, p)           # raises NotCliffordError early
    if kind in ('Z', 'X'):
        _check_phase_free(program, first, last)
    cut = program.code_at(first - 1)
    fixed_in = cut.fixed_qubits() if first > 0 else {}  # at the start, the initial states are prepared instead
    code_in = cut.data() if code_in is None else code_in
    code_out = program.code_at(last).data() if code_out is None else code_out
    if code_out.k != 1:
        raise NotImplementedError("circuit_distance: one logical qubit after the stages is supported, the code "
                                  "after them has k = {}; give code_out".format(code_out.k))
    if code_in.k > 1:
        raise NotImplementedError("circuit_distance: at most one logical qubit before the stages is supported")
    result = CircuitDistance(kind, first, last, None, method=method)
    for a in (('X', 'Z') if code_in.k else (None,)):
        in_letters = _letters(code_in, a) if a else None
        for b in ('X', 'Y', 'Z'):
            clean, noisy, records = _circuit(program, first, last, kind, p, code_in, code_out, in_letters,
                                             _letters(code_out, b), fixed_in)
            text, ndet = _annotated(clean, noisy, records)
            if text is None:
                continue                                  # that output logical is random: no observable
            key = (a or '-', b)
            qc = stim.Circuit(text)
            result.circuits[key] = text
            result.detectors[key] = ndet
            dem = qc.detector_error_model()
            if not any(t.is_logical_observable_id() for inst in dem.flattened() if inst.type == 'error'
                       for t in inst.targets_copy()):
                result.by_logical[key] = None             # no fault of this kind can flip the observable
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                found = codedistance.circuitDistance(qc, method, params=dict(params or {}, filterCircuit=False))
            result.by_logical[key] = int(found['d'])
    found_d = [d for d in result.by_logical.values() if d is not None]
    result.distance = min(found_d) if found_d else None
    return result


# -- removing gates ------------------------------------------------------------------------------

def without_gates(program: QProgram, gates: Iterable[str] = ('T',)) -> QProgram:
    """A copy of ``program`` with every gate whose class is named in ``gates`` removed (``'T'`` covers ``T^dag``).

    The stages, their names and permutations are kept; levels are not. Used for
    the Z distance of circuits whose only non-Clifford gates are single-qubit
    ``T``-type gates (see the module docstring); :func:`z_distance_through_t`
    does that after checking that removing them is exact.
    """
    drop = set(gates)
    stages = program.stages
    initial = list(stages[0].mapping) if stages else list(range(program.qubits))
    if stages and stages[0].kind == 'permutation':
        assert stages[0].perm is not None
        initial = [0] * program.qubits
        for k in range(program.qubits):
            initial[stages[0].perm[k]] = stages[0].mapping[k]
    out = QProgram(program.qubits, bits=program.bits, topology=program.topology,
                   mapping=dict(enumerate(initial)), gate_durations=program.gate_durations,
                   initial_states=program._initial_states)
    for st in stages:
        if st.kind == 'permutation':
            assert st.perm is not None
            out.add_permutation(list(st.perm), name=st.name, visible=st.visible)
            continue
        c = Circuit(program.qubits, bit_amount=program.bits)
        for g in st.circuit.gates:
            if type(g).__name__ not in drop:
                c.add_gate(g)
        out.add_circuit(c, name=st.name)
    return out


# -- the Z distance through T gates -----------------------------------------------------------------

def _x_basis_results(program: QProgram, last: int) -> set[str]:
    """The results of stages 0..last measured in the X basis (a Hadamard right before the measurement)."""
    previous: dict[int, Any] = {}
    out: set[str] = set()
    for s in range(last + 1):
        for g in program.stage_circuit(s).gates:
            if type(g).__name__ == 'Measurement' and type(previous.get(g.target)).__name__ == 'HAD':
                out.add(_result_name(g))
            for q in _qubits_of(g):
                previous[q] = g
    return out


def z_distance_through_t(program: QProgram, first: int = 0, last: int | None = None, p: float = 1e-3,
                         method: str = 'MIPDist', params: dict[str, Any] | None = None) -> CircuitDistance:
    """The Z circuit distance of stages whose only non-Clifford gates are ``T`` and ``T^dag``.

    A Z fault commutes with ``T`` and ``T^dag``, so it flips the same results with
    and without them. The Z distance of the stages is then that of the stages
    without their ``T`` gates (:func:`without_gates`), measured against the
    logical operators of the code the real stages end in, provided that

    1. the two have the same detectors: the detectors derived from the noiseless
       circuits with and without the ``T`` gates, between the same boundary
       checks, span the same space (the final logical measurement left out);
    2. no Z fault can trigger a fed-forward X correction, which would be an X
       error in front of later ``T`` gates: every X correction is controlled by
       results of Z-basis measurements, which Z faults do not flip.

    Both are checked, and :class:`TRemovalError` is raised if one fails. Any
    other non-Clifford gate raises :class:`NotCliffordError`, and stages that
    are not phase-free without their ``T`` gates :class:`NotPhaseFreeError`.
    The X distance cannot be found this way: an X fault in front of a ``T`` gate
    becomes a Clifford error that the circuit without ``T`` does not have.
    """
    last = program.num_stages - 1 if last is None else last
    if not 0 <= first <= last < program.num_stages:
        raise ValueError("stages {}..{} are not in the program's {} stages".format(first, last, program.num_stages))
    _stage_lines(program, first, last, 'Z', p, keep=('T',))        # raises NotCliffordError for any other gate
    stripped = without_gates(program)
    _check_phase_free(stripped, first, last)
    code_in = program.code_at(first - 1).data()
    code_out = program.code_at(last).data()
    if code_out.k != 1 or code_in.k > 1:
        raise NotImplementedError("z_distance_through_t: one logical qubit after the stages and at most one before "
                                  "are supported (k = {} before, {} after)".format(code_in.k, code_out.k))
    x_basis = _x_basis_results(stripped, last)
    for s in range(first, last + 1):
        for g in stripped.stage_circuit(s).gates:
            if type(g).__name__ == 'ParityControlled' and g.gate == 'X' and set(g.bits) & x_basis:
                raise TRemovalError("stage {}: the X correction on qubit {} is controlled by {}, measured in the X "
                                    "basis, so a Z fault can trigger it, and the X error it then makes does not "
                                    "commute with T".format(s, g.target, ', '.join(sorted(set(g.bits) & x_basis))))
    fixed_in = stripped.code_at(first - 1).fixed_qubits() if first > 0 else {}
    out_letters = _letters(code_out, 'X')
    for a in (('X', 'Z') if code_in.k else (None,)):
        in_letters = _letters(code_in, a) if a else None
        real, _, records = _circuit(program, first, last, 'Z', p, code_in, code_out, in_letters, out_letters,
                                    fixed_in, keep=('T',))
        bare, _, _ = _circuit(stripped, first, last, 'Z', p, code_in, code_out, in_letters, out_letters, fixed_in)
        col = records.index('out_logical')
        with_t = _split_off(_detector_rows(real, len(records)), col)[1]
        without_t = _split_off(_detector_rows(bare, len(records)), col)[1]
        r1, r2, both = _rank(with_t), _rank(without_t), _rank(with_t + without_t)
        if not r1 == r2 == both:
            raise TRemovalError("the stages have {} independent detectors with their T gates and {} without them, "
                                "{} together{}: removing the T gates changes what the detectors see".format(
                                    r1, r2, both, '' if a is None else ' (input logical {})'.format(a)))
    return circuit_distance(stripped, first, last, 'Z', p, method, code_in, code_out, params)
