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

"""Finding the Pauli corrections that make a protocol do the same thing in every run.

A protocol with measurements does a different map in each run. When the
difference is a Pauli on its outputs that the outcomes determine (a *Pauli
frame*), applying that Pauli, controlled by those outcomes, makes every run do
what the run with every outcome 0 does. :func:`find_corrections` reads the Paulis
off the normal forms:

1. In the Z-X normal form of the program (:func:`~zxqec.gpf.reduce_gpf`) each
   output has an X-spider. The part of its phase that depends on the outcomes,
   ``(c[i] + c[j] + ...) pi``, is an X on that output when that parity is 1: the
   X corrections.
2. With those applied, the X-Z normal form (:func:`~zxqec.gpf.x_z_normal_form`)
   has a Z-spider on each output: the outcome phases of the Z-spiders have been
   copied across to the outputs. The part of each that depends on the outcomes is
   a Z on that output: the Z corrections.
3. With both applied, the program is checked to be the same map, up to a scalar,
   in every run its detectors allow as in the run with every outcome 0.

An outcome phase on the outputs of a normal form can be a check of the code the
outputs are in, which needs no correction: when no correction, or the X
corrections alone, already pass the check of step 3, that is what is returned.

Phases are first written modulo the detectors (:func:`~zxqec.gpf.phases_modulo`),
so a parity that is 0 in every run gives no correction. A difference that is not
a Pauli -- an outcome that changes a non-Pauli phase, as in the injection of a
``T`` gate, whose correction is an ``S`` -- is not corrected; the result then
says that the program still depends on its outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyzx.utils import EdgeType, VertexType

from .feedforward import ParityControlled
from .program import QProgram

__all__ = ['Corrections', 'find_corrections']


@dataclass
class Corrections:
    """Pauli corrections at the end of a program, each controlled by the parity of some outcomes."""
    x: dict[int, tuple[str, ...]] = field(default_factory=dict)   #: program qubit -> outcomes of its X correction
    z: dict[int, tuple[str, ...]] = field(default_factory=dict)   #: program qubit -> outcomes of its Z correction
    complete: bool = False          #: with them, every run the detectors allow is the run with every outcome 0

    def gates(self) -> list[ParityControlled]:
        """The corrections as gates, the X corrections first: append them as a stage."""
        return ([ParityControlled('X', q, list(bits)) for q, bits in sorted(self.x.items())]
                + [ParityControlled('Z', q, list(bits)) for q, bits in sorted(self.z.items())])

    def describe(self) -> str:
        lines = ["{} on qubit {} when {} = 1".format(letter, q, " + ".join(bits))
                 for letter, part in (('X', self.x), ('Z', self.z)) for q, bits in sorted(part.items())]
        if not lines:
            lines = ["no Pauli correction is needed" if self.complete else "no Pauli correction found"]
        lines.append("with them, every run is the run with every outcome 0" if self.complete else
                     "with them, the program still depends on its outcomes (a difference that is not a Pauli)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def _output_paulis(h: Any, letters: dict[int, str]) -> dict[int, tuple[str, tuple[str, ...]]]:
    """For each output of the normal form ``h``: (the Pauli its spider's phase is, the outcomes of that phase)."""
    from ..codes.phasepoly import affine_parts
    from ..spacetime.subdiagram import get_logical_qubit
    out = {}
    for b in h.outputs():
        (s,) = list(h.neighbors(b))
        if h.type(s) not in letters:
            continue
        try:
            names, _const = affine_parts(h.phase(s))
        except ValueError:                     # not a parity of outcomes: no Pauli correction reads it
            continue
        if names:
            letter = letters[h.type(s)]
            if h.edge_type(h.edge(b, s)) == EdgeType.HADAMARD:
                letter = 'Z' if letter == 'X' else 'X'
            out[get_logical_qubit(h, b)] = (letter, tuple(sorted(names)))
    return out


def _every_run_the_same(program: QProgram, after: int, gates: list[ParityControlled]) -> bool:
    """Whether ``program`` with ``gates`` after stage ``after`` is, in every run its detectors allow, the all-0 run."""
    from ..gpf import boolean_variables, equal_for_all_assignments, reduce_gpf
    from .errors import _with_stage
    g = _with_stage(program, after, gates, name='corrections').quantum_graph() if gates else program.quantum_graph()
    names = boolean_variables(g)
    reference = g.substitute_variables({n: 0 for n in names}) if names else g
    return equal_for_all_assignments(g, reference, constraints=reduce_gpf(g).constraints)


def find_corrections(program: QProgram) -> Corrections:
    """The Pauli corrections, at the end of ``program``, that make every run do the same (module docstring).

    Returns :class:`Corrections`; ``Corrections.gates()`` are the corrections as
    :class:`ParityControlled` gates on the program's qubits, to append as a last
    stage, and ``complete`` says whether the program with them is the same map in
    every run its detectors allow.
    """
    from ..gpf import phases_modulo, reduce_gpf, x_z_normal_form
    from .errors import _with_stage

    letters = {VertexType.X: 'X', VertexType.Z: 'Z'}
    last = program.num_stages - 1
    found = Corrections()

    res = reduce_gpf(program.quantum_graph())                    # 1. the Z-X normal form: X on the outputs
    first = _output_paulis(phases_modulo(res.to_graph(), res.constraints), letters)
    for q, (letter, bits) in first.items():
        (found.x if letter == 'X' else found.z)[q] = bits
    after_first = Corrections(dict(found.x), dict(found.z))
    step = _with_stage(program, last, found.gates(), name='corrections') if first else program

    xz, constraints = x_z_normal_form(step.quantum_graph())     # 2. the X-Z normal form: Z on the outputs
    second = _output_paulis(phases_modulo(xz, constraints), letters)
    for q, (letter, bits) in second.items():
        part = found.x if letter == 'X' else found.z
        part[q] = tuple(sorted(set(part.get(q, ())) ^ set(bits)))
        if not part[q]:
            del part[q]

    # 3. the same map in every run? None, then the X ones, then all: an output phase can be a check of the code
    tried: list[tuple[dict[int, tuple[str, ...]], dict[int, tuple[str, ...]]]] = []
    for candidate in (Corrections(), after_first, found):
        if (candidate.x, candidate.z) in tried:
            continue
        tried.append((candidate.x, candidate.z))
        if _every_run_the_same(program, last, candidate.gates()):
            candidate.complete = True
            return candidate
    return found
