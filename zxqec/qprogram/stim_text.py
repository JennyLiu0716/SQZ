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

"""Programs from circuits written in Stim's text format.

Stim's format is the common way to publish error-correction circuits. This
module reads the gate subset those circuits use, plus ``T`` and ``T_DAG``,
which some cultivation circuits add although Stim itself cannot simulate them,
into a :class:`QProgram`.

* ``R`` / ``RX`` reset to ``|0>`` / ``|+>``; ``M`` / ``MX`` measure in the Z / X
  basis; ``H``, ``S``, ``S_DAG``, ``T``, ``T_DAG``, ``X``, ``Z``, ``CX``, ``CZ``.
* Measurement ``i`` (counting from 0 over the whole file) writes the outcome
  variable ``c[i]``.
* ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` are *not* used to build the program;
  they are returned separately, translated to lists of outcome names, so that
  the detectors the authors wrote by hand can be compared with the ones the
  reduction derives. ``QUBIT_COORDS`` are returned too; ``TICK`` is ignored
  (the program schedules its own gates).

The program is cut into stages at every point where a qubit that has been
measured is reset again, so that every measurement ends a stage and every
reset starts one, which is how :class:`QProgram` treats reused qubits.
Additional cut points can be given as indices of instructions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, NOT, Measurement, Reset, S, T, Z

from .program import QProgram

__all__ = ['StimCircuit', 'parse_stim', 'program_from_stim']

_ANNOTATIONS = {'TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS', 'SHIFT_COORDS'}
_SUPPORTED = {'R', 'RX', 'M', 'MX', 'H', 'S', 'S_DAG', 'T', 'T_DAG', 'X', 'Z', 'CX', 'CNOT', 'CZ'}


@dataclass
class Instruction:
    name: str
    args: tuple[float, ...]
    targets: tuple[str, ...]


@dataclass
class StimCircuit:
    """A parsed circuit: its instructions, measurements, and annotations."""
    instructions: list[Instruction]
    qubits: int
    measurements: int
    coords: dict[int, tuple[float, ...]] = field(default_factory=dict)
    detectors: list[list[str]] = field(default_factory=list)       #: outcome names of each DETECTOR
    detector_coords: list[tuple[float, ...]] = field(default_factory=list)
    observables: dict[int, list[str]] = field(default_factory=dict)


_LINE = re.compile(r'^\s*([A-Z_]+)(\(([^)]*)\))?\s*(.*)$')


def parse_stim(text: str) -> StimCircuit:
    """Parse Stim-format text (with ``T`` / ``T_DAG``); raises on unsupported instructions."""
    instrs: list[Instruction] = []
    coords: dict[int, tuple[float, ...]] = {}
    detectors: list[list[str]] = []
    dcoords: list[tuple[float, ...]] = []
    observables: dict[int, list[str]] = {}
    count = 0
    top = -1
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        m = _LINE.match(line)
        if m is None:
            raise ValueError("cannot parse line {!r}".format(raw))
        name, args_text, targets_text = m.group(1), m.group(3), m.group(4)
        args = tuple(float(a) for a in args_text.split(',')) if args_text else ()
        targets = tuple(targets_text.split())
        if name in ('DETECTOR', 'OBSERVABLE_INCLUDE'):
            recs = []
            for t in targets:
                r = re.fullmatch(r'rec\[-(\d+)\]', t)
                if r is None:
                    raise ValueError("unsupported target {!r} in {}".format(t, name))
                idx = count - int(r.group(1))
                if idx < 0:
                    raise ValueError("{} refers to a measurement before the first one".format(name))
                recs.append('c[{}]'.format(idx))
            if name == 'DETECTOR':
                detectors.append(recs)
                dcoords.append(args)
            else:
                observables.setdefault(int(args[0]) if args else 0, []).extend(recs)
            continue
        if name == 'QUBIT_COORDS':
            coords[int(targets[0])] = args
            continue
        if name in _ANNOTATIONS:
            continue
        if name not in _SUPPORTED:
            raise ValueError("unsupported instruction {} (supported: {})".format(name, sorted(_SUPPORTED)))
        qs = [int(t) for t in targets]
        top = max([top] + qs)
        if name in ('CX', 'CNOT', 'CZ') and len(qs) % 2:
            raise ValueError("{} needs an even number of targets".format(name))
        if name in ('M', 'MX'):
            count += len(qs)
        instrs.append(Instruction(name, args, tuple(targets)))
    qubits = max(top + 1, max(coords, default=-1) + 1)
    return StimCircuit(instrs, qubits, count, coords, detectors, dcoords, observables)


def _gates(ins: Instruction, first_bit: int) -> list[Any]:
    qs = [int(t) for t in ins.targets]
    n = ins.name
    if n == 'R':
        return [Reset(q) for q in qs]
    if n == 'RX':
        return [g for q in qs for g in (Reset(q), HAD(q))]
    if n == 'M':
        return [Measurement(q, first_bit + i) for i, q in enumerate(qs)]
    if n == 'MX':
        return [g for i, q in enumerate(qs) for g in (HAD(q), Measurement(q, first_bit + i))]
    if n == 'H':
        return [HAD(q) for q in qs]
    if n in ('S', 'S_DAG'):
        return [S(q, adjoint=(n == 'S_DAG')) for q in qs]
    if n in ('T', 'T_DAG'):
        return [T(q, adjoint=(n == 'T_DAG')) for q in qs]
    if n == 'X':
        return [NOT(q) for q in qs]
    if n == 'Z':
        return [Z(q) for q in qs]
    pairs = list(zip(qs[::2], qs[1::2]))
    if n in ('CX', 'CNOT'):
        return [CNOT(c, t) for c, t in pairs]
    if n == 'CZ':
        return [CZ(c, t) for c, t in pairs]
    raise AssertionError(n)


def program_from_stim(text: str | StimCircuit, names: Sequence[str] | None = None,
                      extra_cuts: Sequence[int] = ()) -> tuple[QProgram, StimCircuit]:
    """A :class:`QProgram` for a Stim-format circuit, and the parsed circuit.

    Every qubit starts in ``|0>`` (a leading ``R`` is then a no-op, a leading
    ``RX`` a Hadamard). The program is cut into stages as described in the
    module docstring; ``names`` names them in order (``stage 0``, ... by
    default), and ``extra_cuts`` adds cuts before the given instruction indices.
    """
    circ = text if isinstance(text, StimCircuit) else parse_stim(text)
    n, nbits = circ.qubits, circ.measurements
    cuts = set(extra_cuts)
    measured: set[int] = set()
    for i, ins in enumerate(circ.instructions):
        qs = {int(t) for t in ins.targets}
        if ins.name in ('R', 'RX') and qs & measured:
            cuts.add(i)
            measured.clear()
        if ins.name in ('M', 'MX'):
            measured |= qs
    bounds = [0] + sorted(c for c in cuts if 0 < c < len(circ.instructions)) + [len(circ.instructions)]
    p = QProgram(n, bits=nbits, initial_states={q: '0' for q in range(n)})
    bit = 0
    for s, (a, b) in enumerate(zip(bounds, bounds[1:])):
        c = Circuit(n, bit_amount=nbits)
        for ins in circ.instructions[a:b]:
            for g in _gates(ins, bit):
                c.add_gate(g)
            if ins.name in ('M', 'MX'):
                bit += len(ins.targets)
        name = names[s] if names is not None and s < len(names) else 'stage {}'.format(s)
        p.add_circuit(c, name=name)
    return p, circ
