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
* ``CX rec[-k] q`` and ``CZ rec[-k] q`` (Pauli corrections fed forward from a
  measurement) become :class:`~zxqec.qprogram.ParityControlled` ``X`` / ``Z`` on
  ``q``, conditioned on that outcome.
* ``MPP`` measures products of Paulis, such as ``MPP X0*X1 Z2*Y3`` (``!`` in
  front of a product inverts its result). Each product is measured with a helper
  qubit of its own, numbered after the circuit's qubits and starting in ``|0>``:
  every factor is turned to the Z basis (``H`` for X, ``S_DAG`` then ``H`` for
  Y), copied onto the helper by a ``CNOT``, and turned back after the helper is
  measured. The helper's outcome is the product's ``c[i]``.
* The program is the noiseless circuit: noise channels (``DEPOLARIZE1``,
  ``X_ERROR``, ...) are skipped, and so is the flip probability of a noisy
  measurement such as ``MX(0.001)``.
* ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` are *not* used to build the program;
  they are returned separately, translated to lists of outcome names, so that
  the detectors the authors wrote by hand can be compared with the ones the
  reduction derives. ``QUBIT_COORDS`` are returned too; ``TICK`` is ignored
  (the program schedules its own gates).
* The gates of consecutive ``T`` and ``T_DAG`` instructions form one layer
  (:func:`~zxqec.qprogram.same_layer`): they are placed on one time step, as
  in the file, rather than each as soon as its qubit is free. A transversal
  ``T`` layer then sits on one time step of the diagram, so that a slice or a
  cut at a time step holds all of it. ``layers`` chooses the instructions.

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

from .feedforward import ParityControlled
from .program import QProgram, same_layer

__all__ = ['StimCircuit', 'parse_stim', 'program_from_stim']

_ANNOTATIONS = {'TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS', 'SHIFT_COORDS'}
_SUPPORTED = {'R', 'RX', 'M', 'MX', 'MPP', 'H', 'S', 'S_DAG', 'T', 'T_DAG', 'X', 'Z', 'CX', 'CNOT', 'CZ'}
_NOISE = {'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'PAULI_CHANNEL_1', 'PAULI_CHANNEL_2',
          'E', 'ELSE_CORRELATED_ERROR', 'CORRELATED_ERROR', 'HERALDED_ERASE', 'HERALDED_PAULI_CHANNEL_1'}
_REC = re.compile(r'rec\[-(\d+)\]')
_FACTOR = re.compile(r'([XYZ])(\d+)')


def _product(target: str) -> tuple[bool, list[tuple[str, int]]]:
    """``'!X0*Y3'`` -> ``(True, [('X', 0), ('Y', 3)])``: whether the result is inverted, and the factors."""
    inverted = target.startswith('!')
    factors = []
    for f in (target[1:] if inverted else target).split('*'):
        m = _FACTOR.fullmatch(f)
        if m is None:
            raise ValueError("cannot read {!r} as a product of Paulis in MPP".format(target))
        factors.append((m.group(1), int(m.group(2))))
    if len({q for _p, q in factors}) != len(factors):
        raise ValueError("MPP product {!r} names a qubit twice".format(target))
    return inverted, factors


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


_LINE = re.compile(r'^\s*([A-Z_][A-Z0-9_]*)(\(([^)]*)\))?\s*(.*)$')


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
        if name in _ANNOTATIONS or name in _NOISE:
            continue
        if name not in _SUPPORTED:
            raise ValueError("unsupported instruction {} (supported: {})".format(name, sorted(_SUPPORTED)))
        if name in ('CX', 'CNOT', 'CZ'):
            fixed = []
            for t in targets:                       # rec[-k] is the outcome c[count - k]
                r = _REC.fullmatch(t)
                fixed.append('c[{}]'.format(count - int(r.group(1))) if r else t)
            targets = tuple(fixed)
            if len(targets) % 2:
                raise ValueError("{} needs an even number of targets".format(name))
            for c, t in zip(targets[::2], targets[1::2]):
                if c.startswith('c[') and t.startswith('c['):
                    raise ValueError("{} between two measurement records".format(name))
                if t.startswith('c[') and name != 'CZ':
                    raise ValueError("{} with a measurement record as target".format(name))
        elif any(_REC.fullmatch(t) for t in targets):
            raise ValueError("{} does not take measurement records".format(name))
        if name == 'MPP':
            qs = [q for t in targets for _p, q in _product(t)[1]]
            count += len(targets)                   # one result per product
        else:
            qs = [int(t) for t in targets if not t.startswith('c[')]
            if name in ('M', 'MX'):
                count += len(qs)
        top = max([top] + qs)
        instrs.append(Instruction(name, args, tuple(targets)))
    qubits = max(top + 1, max(coords, default=-1) + 1)
    return StimCircuit(instrs, qubits, count, coords, detectors, dcoords, observables)


def _gates(ins: Instruction, first_bit: int, first_helper: int = -1) -> list[Any]:
    n = ins.name
    if n == 'MPP':
        out = []
        for i, t in enumerate(ins.targets):          # product i is measured on the helper first_helper + i
            inverted, factors = _product(t)
            a = first_helper + i
            for p, q in factors:                     # to the Z basis
                out += {'X': [HAD(q)], 'Y': [S(q, adjoint=True), HAD(q)], 'Z': []}[p]
            out += [CNOT(q, a) for _p, q in factors]
            out += [NOT(a)] if inverted else []
            out.append(Measurement(a, first_bit + i))
            for p, q in factors:                     # and back
                out += {'X': [HAD(q)], 'Y': [HAD(q), S(q)], 'Z': []}[p]
        return out
    if n in ('CX', 'CNOT', 'CZ') and any(t.startswith('c[') for t in ins.targets):
        out: list[Any] = []
        for c, t in zip(ins.targets[::2], ins.targets[1::2]):
            if c.startswith('c['):                  # a Pauli fed forward from the outcome c
                out.append(ParityControlled('X' if n in ('CX', 'CNOT') else 'Z', int(t), [c]))
            elif t.startswith('c['):                # CZ is symmetric
                out.append(ParityControlled('Z', int(c), [t]))
            else:
                out.append(CNOT(int(c), int(t)) if n in ('CX', 'CNOT') else CZ(int(c), int(t)))
        return out
    qs = [int(t) for t in ins.targets]
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
                      extra_cuts: Sequence[int] = (),
                      layers: Sequence[str] = ('T', 'T_DAG')) -> tuple[QProgram, StimCircuit]:
    """A :class:`QProgram` for a Stim-format circuit, and the parsed circuit.

    Every qubit starts in ``|0>`` (a leading ``R`` is then a no-op, a leading
    ``RX`` a Hadamard). The program is cut into stages as described in the
    module docstring; ``names`` names them in order (``stage 0``, ... by
    default), and ``extra_cuts`` adds cuts before the given instruction indices.
    The helper qubits of ``MPP`` products come after the circuit's qubits, one
    per product in the order of the file. Consecutive instructions named in
    ``layers`` are one layer of gates, placed on one time step (``()`` places
    every gate as soon as its qubits are free).
    """
    circ = text if isinstance(text, StimCircuit) else parse_stim(text)
    n, nbits = circ.qubits, circ.measurements
    helpers = sum(len(ins.targets) for ins in circ.instructions if ins.name == 'MPP')
    cuts = set(extra_cuts)
    measured: set[int] = set()
    for i, ins in enumerate(circ.instructions):
        if ins.name == 'MPP':
            continue                                # measures fresh helpers only, which are never reset
        qs = {int(t) for t in ins.targets if not t.startswith('c[')}
        if ins.name in ('R', 'RX') and qs & measured:
            cuts.add(i)
            measured.clear()
        if ins.name in ('M', 'MX'):
            measured |= qs
    bounds = [0] + sorted(c for c in cuts if 0 < c < len(circ.instructions)) + [len(circ.instructions)]
    p = QProgram(n + helpers, bits=nbits, initial_states={q: '0' for q in range(n + helpers)})
    bit, helper = 0, n
    layer, previous = 0, None
    for s, (a, b) in enumerate(zip(bounds, bounds[1:])):
        c = Circuit(n + helpers, bit_amount=nbits)
        for ins in circ.instructions[a:b]:
            gates = _gates(ins, bit, helper)
            if ins.name in layers:
                if previous not in layers:
                    layer += 1
                same_layer(gates, ('stim', layer))
            previous = ins.name
            for g in gates:
                c.add_gate(g)
            if ins.name in ('M', 'MX'):
                bit += len(ins.targets)
            elif ins.name == 'MPP':
                bit += len(ins.targets)
                helper += len(ins.targets)
        name = names[s] if names is not None and s < len(names) else 'stage {}'.format(s)
        p.add_circuit(c, name=name)
    return p, circ
