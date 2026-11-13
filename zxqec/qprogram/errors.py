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

"""What Pauli errors at chosen places do to a protocol that checks its outcomes, by trying every pattern.

A protocol such as magic-state distillation keeps a run only when its checks
pass: when every detector of the noiseless program (a parity of outcomes that is
0 in every noiseless run) reads 0. :func:`pauli_errors` puts a Pauli (``'Z'``,
``'X'`` or ``'Y'``) on some of the given qubits right after a given stage, for
every pattern of those qubits, reduces each program (:func:`~zxqec.gpf.reduce_gpf`)
and sorts the pattern into one of three kinds:

* *caught* -- no run passes the checks: the detectors of the program with the
  errors and those of the noiseless program cannot all hold;
* *harmless* -- runs pass, and in them the program with the errors is the same
  map as the noiseless one (up to a scalar);
* *wrong* -- runs pass, and the map is a different one: an error the checks miss.

A run is a value of the results. The checks are the detectors on results; a
detector of the noiseless program on a value that nobody reads (what a reset
throws away) is a fact about the noiseless map, not a check, and such values are
summed over as a mixture when the maps are compared, not compared one by one.

For errors that happen independently with probability ``p`` at each place, the
counts give the probability that a run passes the checks and the probability
that a run that passes is wrong (:meth:`PauliErrors.acceptance`,
:meth:`PauliErrors.output_error`); the lowest weight with a wrong pattern gives
the leading term. Every pattern is a program to reduce, so this is for small
protocols: ``2^n`` patterns for ``n`` places, fewer with ``max_weight``. Whether
the errors are independent Paulis -- for magic states, what twirling them would
make them -- is the model, not something checked here.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pyzx.circuit import Circuit
from pyzx.circuit.gates import NOT, Y, Z

from .program import QProgram

__all__ = ['PauliErrors', 'pauli_errors']

_GATES = {'X': NOT, 'Y': Y, 'Z': Z}


@dataclass
class PauliErrors:
    """The patterns of Pauli errors at some places, sorted by what the checks do with them."""
    letter: str                                        #: the Pauli at each place
    after: int                                         #: the stage the errors follow
    places: tuple[int, ...]                            #: the qubits that can have an error
    counts: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    #: weight -> (patterns, caught, harmless, wrong)
    wrong: dict[int, list[tuple[int, ...]]] = field(default_factory=dict)   #: weight -> the wrong patterns

    @property
    def complete(self) -> bool:
        """Whether every weight up to the number of places was tried."""
        return set(self.counts) == set(range(len(self.places) + 1))

    def leading(self) -> tuple[int, int] | None:
        """``(weight, number of wrong patterns)`` for the lowest weight with a wrong pattern, or None."""
        for w in sorted(self.counts):
            if self.counts[w][3]:
                return w, self.counts[w][3]
        return None

    def acceptance(self, p: float) -> float:
        """The probability that a run passes the checks (from the weights tried)."""
        n = len(self.places)
        return sum((harmless + wrong) * p ** w * (1 - p) ** (n - w)
                   for w, (_total, _caught, harmless, wrong) in self.counts.items())

    def output_error(self, p: float) -> float:
        """The probability that a run that passes the checks is wrong (from the weights tried)."""
        n = len(self.places)
        bad = sum(wrong * p ** w * (1 - p) ** (n - w) for w, (_t, _c, _h, wrong) in self.counts.items())
        return bad / self.acceptance(p)

    def describe(self) -> str:
        lines = ["{} errors after stage {} on qubits {}:".format(
            self.letter, self.after, ", ".join(map(str, self.places))),
                 "  weight  patterns  caught  harmless  wrong"]
        for w in sorted(self.counts):
            total, caught, harmless, wrong = self.counts[w]
            lines.append("  {:6}  {:8}  {:6}  {:8}  {:5}".format(w, total, caught, harmless, wrong))
        lead = self.leading()
        if lead is None:
            lines.append("no pattern tried passes the checks and changes the output")
        else:
            lines.append("lowest weight that passes the checks and changes the output: {} ({} pattern(s)); "
                         "output error {} p^{} to leading order".format(lead[0], lead[1], lead[1], lead[0]))
        if not self.complete:
            lines.append("(weights up to {} of {} tried)".format(max(self.counts), len(self.places)))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def _with_stage(program: QProgram, after: int, gates: list[Any], name: str = 'errors') -> QProgram:
    """A copy of ``program`` with a stage ``name`` of ``gates`` (on program qubits) right after stage ``after``."""
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
            out.add_permutation(list(st.perm), name=st.name, visible=st.visible, cross_inside=st.cross_inside)
        else:
            c = Circuit(program.qubits, bit_amount=program.bits)
            for g in st.circuit.gates:
                c.add_gate(g)
            out.add_circuit(c, name=st.name)
        if st.index == after:
            c = Circuit(program.qubits, bit_amount=program.bits)
            for g in gates:
                c.add_gate(g)
            out.add_circuit(c, name=name)
    return out


def pauli_errors(program: QProgram, after: int, qubits: Iterable[int], letter: str = 'Z',
                 max_weight: int | None = None) -> PauliErrors:
    """Every pattern of ``letter`` errors on ``qubits`` right after stage ``after``, sorted (see the module docstring).

    ``qubits`` are program qubits; ``max_weight`` limits the number of errors in a
    pattern (default: all of them). A pattern is caught when the detectors of the
    program with the errors contradict those of the noiseless program, harmless when
    the two programs are the same map (up to a scalar) in every run that passes the
    checks, and wrong otherwise.
    """
    from ..gpf import reduce_gpf, unsatisfiable
    from .faults import _compare_runs, _noiseless_runs, _result_names, _unit_scalar, _variables
    if letter not in _GATES:
        raise ValueError("letter must be 'X', 'Y' or 'Z', not {!r}".format(letter))
    if not 0 <= after < program.num_stages:
        raise IndexError("no stage {} ({} stages)".format(after, program.num_stages))
    places = tuple(qubits)
    base = program.quantum_graph()
    results = _result_names(program)
    detectors = list(reduce_gpf(base).detectors)
    clean = [d for d in detectors if _variables(d) <= results]          # the checks
    noiseless = _noiseless_runs(_unit_scalar(base), detectors, results)
    top = len(places) if max_weight is None else min(max_weight, len(places))
    out = PauliErrors(letter, after, places)
    for w in range(top + 1):
        caught = harmless = 0
        wrong: list[tuple[int, ...]] = []
        patterns = list(itertools.combinations(places, w))
        for pattern in patterns:
            g = _with_stage(program, after, [_GATES[letter](q) for q in pattern]).quantum_graph()
            res = reduce_gpf(g)
            checks = clean + list(res.detectors)
            verdict = 'caught' if res.is_zero or unsatisfiable(checks) else _compare_runs(
                _unit_scalar(g), checks, noiseless, results)
            if verdict == 'caught':
                caught += 1
            elif verdict == 'harmless':
                harmless += 1
            else:
                wrong.append(pattern)
        out.counts[w] = (len(patterns), caught, harmless, len(wrong))
        out.wrong[w] = wrong
    return out
