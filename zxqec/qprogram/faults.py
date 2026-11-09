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

"""A Pauli error after every gate, as formulas in Boolean switches, with one reduction of the diagram.

After every gate (except a measurement), each qubit the gate acts on gets two
Boolean *switches*, ``x<k>`` and ``z<k>`` for place ``k``: the error there is
``X^x Z^z``. :func:`error_formulas` writes down what the switches do without
reducing a diagram per error:

* every switch is pushed forward through the gates that follow it, with the
  rules that move a Pauli past a Clifford gate (the ZX ``pi``-copy rules): a
  Hadamard swaps its X and Z parts, a CNOT copies X from control to target and
  Z from target to control, an ``S`` adds its X part to its Z part;
* at a measurement, the X part of what arrives flips the result
  (:attr:`ErrorFormulas.flips`, the parity of the switches that reach it), and the
  qubit's switches end there, as at a reset;
* at a ``T`` (or ``T^dag``), an X part that arrives passes through and turns the
  gate into its adjoint, ``T^dag = S^dag T``; the gate gets a *flip variable*
  ``f<i>`` (:attr:`ErrorFormulas.t_flips`), whose value is the parity of those
  switches;
* what is left at the end is a Pauli on each output qubit
  (:attr:`ErrorFormulas.frame`), with variables ``ox<q>`` and ``oz<q>``.

Each step is an equality of maps (up to a global phase), so the noisy program
is the *abstract program* -- the program with ``S^dag`` (``S`` after a
``T^dag``) controlled by each flip variable and the output Pauli controlled by
the frame variables -- with every result ``c`` read as ``c`` plus its flip.
The abstract program is reduced once (:func:`~zxqec.gpf.reduce_gpf`); it has the
flip and frame variables only, not the switches, so the reduction is as small as
the noiseless one.

Every pattern of errors then has a *signature*: the results it flips, the ``T``
gates it flips and the output Pauli, each the parity of its switches. What the
pattern does depends on the signature alone, and is decided as in
:func:`~zxqec.qprogram.pauli_errors`: *caught* when no run passes the checks (the
detectors of the noiseless program), *harmless* when the runs that pass give
the same map as the noiseless program, *wrong* otherwise
(:meth:`ErrorFormulas.classify`). The maps are compared on every input of the
program: when an input is not yet in a code (data qubits whose first checks
read random values), an error that acts like a Pauli on that input before the
program starts counts as wrong. For a fault distance, start from fixed states or
from an encoder. A detector of the noiseless program reads the
parity of the switches in the flips of its results
(:meth:`ErrorFormulas.detector_switches`); errors that reach a ``T`` as an X
can also change which results are random, which the reduced diagram decides.
On top of this, :meth:`ErrorFormulas.single_errors` lists every single Pauli
error, :meth:`ErrorFormulas.lowest_wrong` searches the fewest faults that are
wrong (the fault distance, by trying all patterns up to a weight), and
:meth:`ErrorFormulas.sample` draws faults with given probabilities and counts
what happens, by evaluating the formulas rather than simulating a state. A
*fault* is a Pauli error after one gate on the qubits it acts on, as in the
usual circuit noise (15 kinds after a CNOT), or, with ``per='place'``, on one
of them (:meth:`ErrorFormulas.single_faults`).
:meth:`ErrorFormulas.classify_directly` decides a pattern the slow way, with the
error gates in the program, to check the formulas.

Gates: ``HAD``, ``CNOT``, ``CZ``, ``S``, ``S^dag`` (``S*``), ``T``, ``T^dag``, the
Paulis, ``Measurement`` (in the Z basis), ``Reset`` and feed-forward Pauli
corrections (:class:`~zxqec.qprogram.ParityControlled` with an X or Z of angle
``pi``, which fire wrongly when a result they read is flipped); permutation
stages leave the switches on their program qubits. Other gates raise
``NotImplementedError``, and so does a result written by two measurements.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from pyzx.circuit import Circuit

from .feedforward import ParityControlled
from .program import QProgram

__all__ = ['Place', 'TFlip', 'ErrorFormulas', 'error_formulas']

_PAULIS = ('X', 'Y', 'Z')


@dataclass(frozen=True)
class Place:
    """A place where an error can happen: right after gate ``gate`` of stage ``stage``, on ``qubit``."""
    index: int
    stage: int
    gate: int
    name: str                                       #: the gate's name (``'CNOT'``, ``'HAD'``, ...)
    qubit: int                                      #: program qubit

    @property
    def x(self) -> str:
        """The switch of the X part of the error here."""
        return 'x{}'.format(self.index)

    @property
    def z(self) -> str:
        """The switch of the Z part of the error here."""
        return 'z{}'.format(self.index)


@dataclass(frozen=True)
class TFlip:
    """A ``T`` (or ``T^dag``) gate that an X error can reach, turning it into its adjoint."""
    name: str                                       #: the flip variable, ``f<i>``
    stage: int
    gate: int
    qubit: int
    adjoint: bool                                   #: the gate is ``T^dag``
    switches: frozenset[str]                        #: its value is the parity of these switches


@dataclass
class ErrorFormulas:
    """What a Pauli error after any gate does to a program (see the module docstring)."""
    program: QProgram
    places: list[Place]
    flips: dict[str, frozenset[str]]                #: result -> the switches whose parity flips it
    t_flips: list[TFlip]
    frame: dict[int, tuple[frozenset[str], frozenset[str]]]   #: output qubit -> (X switches, Z switches)
    abstract: QProgram                              #: the program with flip and frame variables
    reduced: Any                                    #: :func:`reduce_gpf` of the abstract program
    clean: list[Any]                                #: the detectors of the noiseless program
    base: Any                                       #: the noiseless program, reduced (a graph)
    decoder: Any = None                             #: the ideal decoder of the output code (``final_check``)
    _verdicts: dict[int, str] = field(default_factory=dict, repr=False)
    _base_decoded: Any = field(default=None, repr=False)
    _flip_checks: dict[int, Any] = field(default_factory=dict, repr=False)
    _base_runs: Any = field(default=None, repr=False)
    _noisy_cache: dict[int, Any] = field(default_factory=dict, repr=False)
    _shares: dict[int, dict[str, float]] = field(default_factory=dict, repr=False)

    # -- signatures ---------------------------------------------------------------------

    def __post_init__(self) -> None:
        # one bit per result, per flip variable and per frame variable; a switch's mask is what it flips
        self._columns: list[str] = (sorted(self.flips, key=_result_order) + [t.name for t in self.t_flips]
                                    + ['{}{}'.format(o, q) for q in sorted(self.frame) for o in ('ox', 'oz')])
        bit = {c: 1 << i for i, c in enumerate(self._columns)}
        self._mask: dict[str, int] = {}
        for c, sw in self.flips.items():
            for v in sw:
                self._mask[v] = self._mask.get(v, 0) ^ bit[c]
        for t in self.t_flips:
            for v in t.switches:
                self._mask[v] = self._mask.get(v, 0) ^ bit[t.name]
        for q, (xs, zs) in self.frame.items():
            for v in xs:
                self._mask[v] = self._mask.get(v, 0) ^ bit['ox{}'.format(q)]
            for v in zs:
                self._mask[v] = self._mask.get(v, 0) ^ bit['oz{}'.format(q)]
        self._t_bits = sum(bit[t.name] for t in self.t_flips)
        self._frame_names = frozenset('{}{}'.format(o, q) for q in self.frame for o in ('ox', 'oz'))
        # the checks are the detectors on results; the others hold a value that nobody reads (what a reset throws
        # away) and are facts about the noiseless map, not checks
        self._checks = [d for d in self.clean if _variables(d) <= set(self.flips)]
        self._detector_masks = [sum(bit[c] for c in _variables(d) if c in bit) for d in self._checks]

    def switches(self, errors: dict[int, str]) -> set[str]:
        """The switches that are on for ``errors``, a map from place index to ``'X'``, ``'Y'`` or ``'Z'``."""
        on: set[str] = set()
        for k, letter in errors.items():
            if letter not in _PAULIS:
                raise ValueError("an error is 'X', 'Y' or 'Z', not {!r}".format(letter))
            place = self.places[k]
            if letter in 'XY':
                on.add(place.x)
            if letter in 'YZ':
                on.add(place.z)
        return on

    def mask(self, errors: dict[int, str]) -> int:
        """The signature of ``errors`` as a bit mask (see :meth:`signature`)."""
        m = 0
        for v in self.switches(errors):
            m ^= self._mask.get(v, 0)
        return m

    def signature(self, errors: dict[int, str]) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        """``(flipped results, flipped T gates, output frame variables that are 1)`` for ``errors``."""
        return self._decode(self.mask(errors))

    def _decode(self, m: int) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        on = [c for i, c in enumerate(self._columns) if m >> i & 1]
        ts = {t.name for t in self.t_flips}
        return (frozenset(c for c in on if c in self.flips), frozenset(c for c in on if c in ts),
                frozenset(c for c in on if c in self._frame_names))

    def detector_switches(self) -> list[tuple[Any, frozenset[str]]]:
        """Each detector of the noiseless program with the switches whose parity flips it.

        A detector is a parity of results; an error flips it when it flips an odd
        number of them. This is the whole story for errors that do not reach a
        ``T`` gate as an X; the others are decided by :meth:`classify`.
        """
        out = []
        for d in self.clean:
            names = _variables(d)
            s: set[str] = set()
            for c in names:
                s ^= set(self.flips.get(c, frozenset()))
            out.append((d, frozenset(s)))
        return out

    def formula(self, what: str | Iterable[str]) -> str:
        """A formula as text: the sum of the switches that flip ``what``, in the order of the places.

        ``what`` is a result (``'c[0]'``), a ``T`` flip variable (``'f0'``), an output
        frame variable (``'ox0'``, ``'oz0'``) or a set of switches (as
        :meth:`detector_switches` gives them). ``what`` is flipped when the sum is odd;
        ``'0'`` when no switch reaches it.
        """
        if isinstance(what, str):
            flips = {t.name: t.switches for t in self.t_flips}
            if what in self.flips:
                switches: Iterable[str] = self.flips[what]
            elif what in flips:
                switches = flips[what]
            elif what in self._frame_names:
                xs, zs = self.frame[int(what[2:])]
                switches = xs if what.startswith('ox') else zs
            else:
                raise KeyError("{!r} is not a result, a T flip or an output frame variable".format(what))
        else:
            switches = what
        names = sorted(switches, key=lambda v: (int(v[1:]), v[0]))
        return ' + '.join(names) if names else '0'

    def describe(self, errors: dict[int, str]) -> str:
        """A pattern of errors in words, one fault per gate, joined by ``'; '``.

        A fault reads like ``'X on qubit 7, Y on qubit 8 after the CNOT (stage 4, gate 11)'``.
        """
        self.switches(errors)                                # the letters are checked there
        by_gate: dict[tuple[int, int], list[str]] = {}
        names: dict[tuple[int, int], str] = {}
        for k in sorted(errors):
            place = self.places[k]
            by_gate.setdefault((place.stage, place.gate), []).append('{} on qubit {}'.format(errors[k], place.qubit))
            names[place.stage, place.gate] = place.name
        if not by_gate:
            return 'no errors'
        return '; '.join('{} after the {} (stage {}, gate {})'.format(', '.join(what), names[key], *key)
                         for key, what in sorted(by_gate.items()))

    # -- verdicts ----------------------------------------------------------------------

    def classify(self, errors: dict[int, str]) -> str:
        """``'caught'``, ``'harmless'`` or ``'wrong'`` for a pattern of errors (see the module docstring)."""
        return self._verdict(self.mask(errors))

    def _verdict(self, m: int) -> str:
        if m in self._verdicts:
            return self._verdicts[m]
        flipped = m & self._t_bits
        if not flipped and any(bin(m & d).count('1') % 2 for d in self._detector_masks):
            verdict = 'caught'                           # no T flipped: a noiseless detector reads 1 in every run
        elif flipped and self._caught_with_flips(m, flipped):
            verdict = 'caught'
        else:
            verdict = self._reduce_verdict(self._decode(m))
        self._verdicts[m] = verdict
        return verdict

    def _caught_with_flips(self, m: int, flipped: int) -> bool:
        # With these T gates flipped, the abstract program is reduced once (per pattern of flipped gates); its
        # detectors are linear in the results and the frame variables. A pattern is caught when, with its results
        # shifted and its frame variables set, they contradict the noiseless detectors. Each combination of
        # detectors whose left-hand sides cancel gives a parity of the pattern's bits that must equal a constant.
        if flipped not in self._flip_checks:
            self._flip_checks[flipped] = self._flip_functionals(flipped)
        checks = self._flip_checks[flipped]
        if checks is None:
            return False
        return any((bin(m & mask).count('1') + const) % 2 for mask, const in checks)

    def _flip_functionals(self, flipped: int) -> list[tuple[int, int]] | None:
        import numpy as np

        from ..codes.gf2 import nullspace
        from ..gpf import reduce_gpf
        on = {c for i, c in enumerate(self._columns) if flipped >> i & 1}
        values: dict[str, Any] = {t.name: (1 if t.name in on else 0) for t in self.t_flips}
        g = self.reduced.to_graph().substitute_variables(values)
        res = reduce_gpf(g)
        if res.is_zero:
            return [(0, 1)]                              # every run is the zero map: always caught
        noisy = [d for d in (_substitute(d, values) for d in self.reduced.detectors) if not _is_zero(d)]
        noisy += list(res.detectors)
        frame_vars = self._frame_names
        rows: list[tuple[dict[str, int], int]] = []
        for d in self._checks + noisy:
            row = _affine_row(d)
            if row is None:
                return None                              # a product of variables: no linear shortcut
            rows.append(row)
        unknown = sorted({v for coeffs, _c in rows for v in coeffs if v not in frame_vars})
        a = np.array([[coeffs.get(v, 0) for v in unknown] for coeffs, _c in rows], dtype=np.uint8).reshape(
            len(rows), len(unknown))
        bit = {c: 1 << i for i, c in enumerate(self._columns)}
        n_clean = len(self._checks)
        out = []
        for y in nullspace(a.T) if len(unknown) else np.eye(len(rows), dtype=np.uint8):
            const, mask = 0, 0
            for i in np.flatnonzero(y):
                coeffs, c = rows[i]
                const ^= c
                if i >= n_clean:                         # a noisy detector: the shift of its results and frame
                    for v, k in coeffs.items():
                        if k and v in bit:
                            mask ^= bit[v]
            out.append((mask, const))
        return out

    def _noisy(self, sig: tuple[frozenset[str], frozenset[str], frozenset[str]]) -> tuple[Any, list[Any]]:
        # The program with the errors of a signature: the abstract program's reduced diagram with the T flips and
        # the output Pauli set and each flipped result c read as c + 1, reduced again, and its own constraints
        # (which runs it can give; the checks are not among them). None when it is the zero map.
        from pyzx.symbolic import new_var

        from ..gpf import reduce_gpf
        results, ts, out = sig
        g = self.reduced.to_graph()
        values: dict[str, Any] = {t.name: (1 if t.name in ts else 0) for t in self.t_flips}
        for q in self.frame:
            for o in ('ox', 'oz'):
                values['{}{}'.format(o, q)] = 1 if '{}{}'.format(o, q) in out else 0
        for c in results:
            values[c] = new_var(c, True, g.var_registry) + 1
        g = g.substitute_variables(values)
        shifted = [_substitute(d, values) for d in self.reduced.detectors]
        res = reduce_gpf(g)
        own = [d for d in shifted if not _is_zero(d)] + list(res.detectors)
        return (None if res.is_zero else res.to_graph()), own

    def _reduce_verdict(self, sig: tuple[frozenset[str], frozenset[str], frozenset[str]]) -> str:
        from ..gpf import unsatisfiable
        noisy, own = self._noisy(sig)
        if sig[1]:                                           # T gates flipped: outcomes may need it again
            self._noisy_cache[self._mask_of(sig)] = (noisy, own)
        checks = self._checks + own
        if noisy is None or unsatisfiable(checks):
            return 'caught'
        return self._runs_verdict(noisy, checks)

    def _runs_verdict(self, noisy: Any, checks: list[Any]) -> str:
        # The noisy map against the noiseless one, run by run (see _compare_runs). With the final check, both
        # outputs are decoded by the ideal decoder of the output code; a run that decodes to zero fails it and is
        # thrown away.
        a_g = _unit_scalar(noisy)
        if self.decoder is not None:
            a_g.compose(self.decoder)
        return _compare_runs(a_g, checks, self._noiseless(), set(self.flips), drop_zero=self.decoder is not None)

    def _noiseless(self) -> Any:
        # the runs of the noiseless program (decoded with the final check), once
        if self._base_runs is None:
            b_g = _unit_scalar(self.base)
            if self.decoder is not None:
                b_g.compose(self.decoder)
            self._base_runs = _noiseless_runs(b_g, self.clean, set(self.flips))
        return self._base_runs

    def _mask_of(self, sig: tuple[frozenset[str], frozenset[str], frozenset[str]]) -> int:
        bit = {c: 1 << i for i, c in enumerate(self._columns)}
        return sum(bit[c] for part in sig for c in part)

    def switched_program(self) -> QProgram:
        """The program with its switches in it: after every place ``k``, ``Z`` and ``X`` controlled by ``z<k>`` and
        ``x<k>`` (:class:`ParityControlled`), so that its diagram is the noisy program for every value of the
        switches. :meth:`pushed_graph` is the same map with the switches moved to the results and the outputs."""
        at = self._places_at()

        def extra(stage: int, j: int, g: Any) -> list[Any]:
            return [h for place in at.get((stage, j), [])
                    for h in (ParityControlled('Z', place.qubit, [place.z]),
                              ParityControlled('X', place.qubit, [place.x]))]
        return _copy(self.program, extra)

    def pushed_graph(self, frame: bool = True) -> Any:
        """The diagram of :meth:`switched_program` with every switch pushed to the end.

        It is the diagram of the abstract program (the program, an ``S^dag`` controlled by each ``T`` flip variable,
        and the output Pauli controlled by the frame variables) with each result ``c`` read as ``c`` plus its formula
        and each frame variable replaced by its formula. Phases that are multiples of ``pi`` add their switches
        modulo 2, so a sum of switches is their parity there. A ``T`` flip variable controls an ``S^dag``, a quarter
        turn, where a sum is not the parity: it stays a variable ``f<i>``, whose value is its formula
        (:meth:`formula`). Equal to the diagram of :meth:`switched_program` up to a scalar, for every value of the
        switches (with the ``f<i>`` set to their formulas). With ``frame=False`` the Pauli left on each output stays
        controlled by its frame variables ``ox<q>`` and ``oz<q>``, whose values are their formulas: the same diagram,
        shorter to draw.
        """
        from pyzx.symbolic import new_var
        g = self.abstract.quantum_graph()

        def total(switches: Iterable[str]) -> Any:
            out: Any = 0
            for v in sorted(switches, key=lambda v: (int(v[1:]), v[0])):
                out = out + new_var(v, True, g.var_registry)
            return out
        values: dict[str, Any] = {c: new_var(c, True, g.var_registry) + total(sw) for c, sw in self.flips.items() if sw}
        if frame:
            for q, (xs, zs) in self.frame.items():
                values['ox{}'.format(q)] = total(xs)
                values['oz{}'.format(q)] = total(zs)
        return g.substitute_variables(values)

    def program_with(self, errors: dict[int, str]) -> QProgram:
        """The program with the error gates of ``errors`` (``'X'``, ``'Y'`` or ``'Z'`` at a place) put in after their
        gates: the program that :meth:`classify_directly` reduces."""
        from pyzx.circuit.gates import NOT, Y, Z
        self.switches(errors)                                # the letters are checked there
        gate = {'X': NOT, 'Y': Y, 'Z': Z}
        at: dict[tuple[int, int], list[Any]] = {}
        for k, letter in sorted(errors.items()):
            place = self.places[k]
            at.setdefault((place.stage, place.gate), []).append(gate[letter](place.qubit))
        return _copy(self.program, lambda stage, j, g: at.get((stage, j), []))

    def _places_at(self) -> dict[tuple[int, int], list[Place]]:
        at: dict[tuple[int, int], list[Place]] = {}
        for place in self.places:
            at.setdefault((place.stage, place.gate), []).append(place)
        return at

    def outcomes(self, errors: dict[int, str]) -> dict[str, float]:
        """How a pattern of errors ends, as shares of its runs: ``{'caught': ..., 'harmless': ..., 'wrong': ...}``.

        Each run the program with the errors can give (a value of the results) is
        weighted by its probability, the squared norm of its output. A run in which a
        check fires is caught. With the final check, a run that passes the checks is
        also caught with the probability that the perfect measurement of the output
        code finds its output outside the code; what passes both is harmless when the
        logical state is the noiseless one of that run, and wrong otherwise. :meth:`classify`
        says ``'caught'`` when the first share is 1 and ``'wrong'`` when the last is above
        0, so a pattern that flips a ``T`` gate can be ``'harmless'`` or ``'wrong'`` while
        a share of its runs is caught. The weights assume that the reduced diagram's
        scalar has the same size in every run (its phase may depend on the results).
        """
        return dict(self._outcomes(self.mask(errors)))

    def _outcomes(self, m: int) -> dict[str, float]:
        import numpy

        from ..gpf import boolean_variables, evaluate_phase
        if m in self._shares:
            return self._shares[m]
        if self._verdict(m) == 'caught':
            self._shares[m] = {'caught': 1.0, 'harmless': 0.0, 'wrong': 0.0}
            return self._shares[m]
        noisy, own = self._noisy_cache.pop(m, None) or self._noisy(self._decode(m))
        if noisy is None:
            self._shares[m] = {'caught': 1.0, 'harmless': 0.0, 'wrong': 0.0}
            return self._shares[m]
        b_keys, base_runs = self._noiseless()
        zero = _zero_size(base_runs)
        results = set(self.flips)
        full = _unit_scalar(noisy)
        decoded = None
        if self.decoder is not None:
            decoded = _unit_scalar(noisy)
            decoded.compose(self.decoder)
        names = set(boolean_variables(full)) | (set(boolean_variables(decoded)) if decoded is not None else set())
        keys = sorted({v for v in names if v in results} | {v for d in self._checks for v in _variables(d)}
                      | set(b_keys))
        inner = sorted(names - results)
        runs_full = _runs(full, own, keys, inner)
        runs_decoded = _runs(decoded, own, keys, inner) if decoded is not None else runs_full
        at = [keys.index(v) for v in b_keys]
        scale = self._decoder_scale() if decoded is not None else 1.0
        shares = {'caught': 0.0, 'harmless': 0.0, 'wrong': 0.0}
        for key, (choi, _size) in runs_full.items():
            weight = float(numpy.real(numpy.trace(choi)))
            values = dict(zip(keys, key))
            if any(evaluate_phase(d, values) for d in self._checks):
                shares['caught'] += weight
                continue
            a, _a_size = runs_decoded[key]
            passed = float(numpy.real(numpy.trace(a))) / scale if decoded is not None else weight
            passed = min(passed, weight)
            shares['caught'] += weight - passed
            if passed < 1e-12 * max(weight, 1e-300):
                continue
            b, b_size = base_runs.get(tuple(key[i] for i in at), (None, 0.0))
            same = b_size > zero and numpy.allclose(a / numpy.trace(a), b / numpy.trace(b), atol=1e-8)
            shares['harmless' if same else 'wrong'] += passed
        total = sum(shares.values())
        self._shares[m] = ({k: v / total for k, v in shares.items()} if total > 0
                           else {'caught': 1.0, 'harmless': 0.0, 'wrong': 0.0})
        return self._shares[m]

    def _decoder_scale(self) -> float:
        # the decoder is the ideal decoder up to a scalar: D D^dag = s * I, and |D psi|^2 = s |psi|^2 in the code
        import numpy

        from ..gpf import boolean_variables
        if not hasattr(self, '_scale'):
            g = self.decoder.adjoint()
            g.compose(self.decoder)
            names = boolean_variables(g)
            m = (g.substitute_variables({v: 0 for v in names}) if names else g).to_matrix(preserve_scalar=True)
            self._scale = float(numpy.real(numpy.trace(m))) / m.shape[0]
        return self._scale

    def classify_directly(self, errors: dict[int, str]) -> str:
        """The verdict for ``errors`` the slow way, to check :meth:`classify`.

        The error gates are put into the program at their places, the whole program
        is reduced, and the verdict is decided by the same rules. The checks of the
        noiseless program are taken on its results only: a constraint on a value
        that a reset throws away (a reset variable) holds for the noiseless program,
        not for the noisy one, whose own constraints come from its reduction.
        """
        from ..gpf import reduce_gpf, unsatisfiable
        res = reduce_gpf(self.program_with(errors).quantum_graph())
        checks = self._checks + list(res.detectors)
        if res.is_zero or unsatisfiable(checks):
            return 'caught'
        return self._runs_verdict(res.to_graph(), checks)

    def single_errors(self) -> list[tuple[Place, str, str, int]]:
        """Every single Pauli error: ``(place, letter, verdict, number of detectors it flips)``.

        The count is of the noiseless detectors whose parity of results it flips;
        for an error that reaches a ``T`` as an X, the verdict also takes into
        account the results that the flip makes random.
        """
        rows = []
        dets = self.detector_switches()
        for k, place in enumerate(self.places):
            for letter in _PAULIS:
                on = self.switches({k: letter})
                fired = sum(1 for _d, s in dets if len(s & on) % 2)
                rows.append((place, letter, self.classify({k: letter}), fired))
        return rows

    def single_faults(self, per: str = 'gate') -> list[tuple[Place, list[dict[int, str]]]]:
        """Where a fault can happen, and its kinds: ``(first place, [pattern, ...])`` per gate or per place.

        ``per='gate'`` (the usual circuit noise): a fault is a Pauli error after one
        gate on the qubits it acts on, other than the identity -- 3 kinds after a
        one-qubit gate, 15 after a CNOT. ``per='place'``: a Pauli error on one qubit
        after one gate, 3 kinds per place; a CNOT then has two places, and an error
        on both of its qubits counts as two.
        """
        if per == 'place':
            return [(place, [{place.index: letter} for letter in _PAULIS]) for place in self.places]
        if per != 'gate':
            raise ValueError("per is 'gate' or 'place', not {!r}".format(per))
        gates: dict[tuple[int, int], list[Place]] = {}
        for place in self.places:
            gates.setdefault((place.stage, place.gate), []).append(place)
        out = []
        for places in gates.values():
            kinds = [{pl.index: letter for pl, letter in zip(places, letters) if letter != 'I'}
                     for letters in itertools.product(('I',) + _PAULIS, repeat=len(places))]
            out.append((places[0], [e for e in kinds if e]))
        return out

    def lowest_wrong(self, max_weight: int = 2, per: str = 'gate') -> tuple[int | None, list[dict[int, str]]]:
        """The fewest faults (at most ``max_weight``) that are wrong, and those patterns.

        Every pattern of faults at distinct gates (:meth:`single_faults`; at distinct
        places with ``per='place'``), of weight 1, 2, ..., is classified until a weight
        has a wrong pattern; ``(None, [])`` if none up to ``max_weight`` is. This is the
        fault distance when it is at most ``max_weight``. Patterns are compared by
        their signature, so each signature is decided once.
        """
        groups = [[(e, self.mask(e)) for e in kinds] for _place, kinds in self.single_faults(per)]
        for w in range(1, max_weight + 1):
            wrong = []
            for combo in itertools.combinations(groups, w):
                for choice in itertools.product(*combo):
                    m = 0
                    for _e, mk in choice:
                        m ^= mk
                    if self._verdict(m) == 'wrong':
                        pattern: dict[int, str] = {}
                        for e, _mk in choice:
                            pattern.update(e)
                        wrong.append(pattern)
            if wrong:
                return w, wrong
        return None, []

    def sample(self, p: float | dict[str, float] | Callable[[Place], float], shots: int,
               seed: int | None = None, per: str = 'gate') -> dict[str, int]:
        """Draw faults and count the verdicts: ``{'caught': ..., 'harmless': ..., 'wrong': ...}``.

        After each gate, with probability ``p`` a fault happens, each of its kinds
        (:meth:`single_faults`) equally likely: ``X``, ``Y`` or ``Z`` with ``p/3`` each
        after a one-qubit gate, each of the 15 two-qubit Paulis with ``p/15`` after a
        CNOT. With ``per='place'``, the same on each qubit of the gate separately.
        ``p`` is a number, a map from gate name to a number (a gate not in it has no
        faults), or a function of the gate's first :class:`Place`. No state is
        simulated: each draw is decided by its signature. Faults that flip no ``T``
        gate are Pauli errors and do the same in every run, so the verdict
        (:meth:`classify`) is the run's. Faults that flip a ``T`` gate can be caught in
        some runs and not in others; for them the run's end is drawn from the shares
        of :meth:`outcomes`. The share of ``'wrong'`` among the runs that are not
        ``'caught'`` estimates the logical error rate after the checks.
        """
        rng = random.Random(seed)
        faults = self.single_faults(per)
        prob = _fault_probabilities(p, faults)
        masks = [[self.mask(e) for e in kinds] for _place, kinds in faults]
        counts = {'caught': 0, 'harmless': 0, 'wrong': 0}
        for _ in range(shots):
            m = 0
            for k, pk in enumerate(prob):
                if pk and rng.random() < pk:
                    m ^= masks[k][rng.randrange(len(masks[k]))]
            verdict = self._verdict(m)
            if verdict != 'caught' and m & self._t_bits:
                shares, r = self._outcomes(m), rng.random()
                verdict = ('caught' if r < shares['caught'] else
                           'harmless' if r < shares['caught'] + shares['harmless'] else 'wrong')
            counts[verdict] += 1
        return counts

    def probability(self, patterns: Iterable[dict[int, str]], p: float | dict[str, float] | Callable[[Place], float],
                    per: str = 'gate', outcome: str | None = None) -> float:
        """The probability that the faults of a run are exactly one of ``patterns``, in the noise of :meth:`sample`.

        A pattern happens when each gate it has a fault at has exactly that fault
        (probability ``p/k`` for a gate with ``k`` kinds, :meth:`single_faults`) and every
        other gate has none (``1 - p``). The empty pattern ``{}`` gives the probability
        that a run has no fault at all. For the wrong patterns that :meth:`lowest_wrong`
        returns, the sum is the leading term of the probability that a run is wrong
        (patterns with more faults add terms with higher powers of ``p``), which
        :meth:`sample` would need very many runs to see when ``p`` is small. Each
        pattern counts once, however it is written; a pattern that is not a set of
        faults at distinct gates raises ``ValueError``. With ``outcome`` (``'caught'``,
        ``'harmless'`` or ``'wrong'``), each pattern counts with its share of that end
        (:meth:`outcomes`): for the wrong patterns and ``'wrong'``, the leading term of
        the probability that a run is kept and wrong.
        """
        faults = self.single_faults(per)
        prob = _fault_probabilities(p, faults)
        group: dict[int, int] = {}                           # place -> its gate (or place) among the faults
        for i, (_place, kinds) in enumerate(faults):
            for e in kinds:
                for k in e:
                    group[k] = i
        none = 1.0
        for pk in prob:
            none *= 1.0 - pk
        seen: set[frozenset[tuple[int, str]]] = set()
        total = 0.0
        for pattern in patterns:
            key = frozenset(pattern.items())
            if key in seen:
                continue
            seen.add(key)
            at: dict[int, dict[int, str]] = {}
            for k, letter in pattern.items():
                if k not in group:
                    raise ValueError("place {} is not a place of a fault".format(k))
                at.setdefault(group[k], {})[k] = letter
            self.switches(pattern)                           # the letters are checked there
            term = none
            for i, fault in at.items():
                kinds = faults[i][1]
                if fault not in kinds:
                    raise ValueError("{} is not one fault after one gate".format(fault))
                if prob[i] >= 1.0:
                    raise ValueError("a gate with fault probability 1 has no run without a fault")
                term *= prob[i] / len(kinds) / (1.0 - prob[i])
            if outcome is not None:
                if outcome not in ('caught', 'harmless', 'wrong'):
                    raise ValueError("outcome is 'caught', 'harmless' or 'wrong', not {!r}".format(outcome))
                term *= self._outcomes(self.mask(pattern))[outcome]
            total += term
        return total


def _fault_probabilities(p: float | dict[str, float] | Callable[[Place], float],
                         faults: list[tuple[Place, list[dict[int, str]]]]) -> list[float]:
    """The probability of a fault at each gate (or place) of ``faults``: ``p`` a number, a map from gate name to a
    number (0 for a gate not in it), or a function of the gate's first place."""
    if callable(p):
        return [float(p(place)) for place, _kinds in faults]
    if isinstance(p, dict):
        return [float(p.get(place.name, 0.0)) for place, _kinds in faults]
    return [float(p)] * len(faults)


def _affine_row(p: Any) -> tuple[dict[str, int], int] | None:
    """A constraint ``p = 0`` as ``({variable: 1}, constant)`` with ``sum of variables = constant``, or None
    when it has a product of variables."""
    if not hasattr(p, 'terms'):
        return {}, int(Fraction(p)) % 2
    coeffs: dict[str, int] = {}
    const = 0
    for c, t in p.terms:
        if Fraction(c) % 2 == 0:
            continue
        if not t.vars:
            const ^= 1
        elif len(t.vars) == 1 and t.vars[0][1] == 1:
            name = t.vars[0][0].name
            coeffs[name] = coeffs.get(name, 0) ^ 1
        else:
            return None
    return {v: 1 for v, k in coeffs.items() if k}, const


def _noiseless_runs(g: Any, constraints: list[Any], results: set[str]) -> tuple[list[str], dict[Any, Any]]:
    """The runs of the noiseless diagram ``g`` (see :func:`_runs`), keyed by the ``results`` it reads."""
    from ..gpf import boolean_variables
    names = set(boolean_variables(g))
    keys = sorted(v for v in names if v in results)
    return keys, _runs(g, constraints, keys, sorted(names - results))


def _compare_runs(g: Any, constraints: list[Any], noiseless: tuple[list[str], dict[Any, Any]], results: set[str],
                  drop_zero: bool = False) -> str:
    """``'harmless'``, ``'wrong'`` or ``'caught'``: the diagram ``g`` against the noiseless runs, run by run.

    A run is a value of the results that ``constraints`` allow. A value that nobody reads (a variable of ``g``
    not in ``results``, such as what a reset throws away) is summed over as a mixture, not compared value by
    value: each run's map is a Choi matrix (:func:`_runs`), compared up to a scalar. A run in which ``g`` is the
    zero map (:func:`_zero_size`) is thrown away when ``drop_zero`` (a final check fails); otherwise it must be zero
    in the noiseless runs too. ``'caught'`` when no run is left.
    """
    import numpy

    from ..gpf import boolean_variables
    b_keys, base = noiseless
    zero = _zero_size(base)
    names = set(boolean_variables(g))
    keys = sorted({v for v in names if v in results} | set(b_keys))
    at = [keys.index(v) for v in b_keys]
    passed = 0
    for key, (a, a_size) in _runs(g, constraints, keys, sorted(names - results)).items():
        a_zero = a_size <= zero
        if drop_zero and a_zero:
            continue
        passed += 1
        b, b_size = base.get(tuple(key[i] for i in at), (None, 0.0))
        if a_zero != (b_size <= zero):
            return 'wrong'
        if not a_zero and not numpy.allclose(a / numpy.trace(a), b / numpy.trace(b), atol=1e-8):
            return 'wrong'
    return 'harmless' if passed else 'caught'


def _zero_size(noiseless: dict[Any, tuple[Any, float]]) -> float:
    """The size (see :func:`_runs`) at or below which a run's map is zero, from the ``noiseless`` runs.

    After :func:`_unit_scalar` a size means nothing by itself: the largest entry of the noiseless map decoded is 0.03
    for d = 3 cultivation and 4e-12 for d = 5, and the rounding errors of a zero map lie some 16 orders of magnitude
    below the map. So the size is taken relative to the largest noiseless run, one number for every pattern: a
    pattern whose runs are all zero (it always fails the final check) must not be measured against itself.
    """
    return 1e-9 * max((size for _choi, size in noiseless.values()), default=0.0)


def _result_names(program: QProgram) -> set[str]:
    """The names of the results of ``program``'s measurements."""
    return {_result_name(g, st.index) for st in program.stages if st.kind != 'permutation'
            for g in st.circuit.gates if g.name == 'Measurement'}


def _runs(g: Any, constraints: list[Any], keys: Sequence[str],
          inner: Sequence[str]) -> dict[tuple[int, ...], tuple[Any, float]]:
    """``{values of the results in keys: (Choi matrix, largest entry of the tensors)}``, over the runs allowed.

    A run is allowed when some values of the other variables of ``constraints`` satisfy them. The ``inner``
    variables of ``g`` (values that nobody reads) are summed as a mixture: the Choi matrix of a run is the sum
    of ``|t><t|`` over their allowed values, ``t`` the tensor of ``g``. One contraction per distinct map.
    """
    import numpy

    branches: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    for subs in _allowed(constraints, list(keys) + list(inner)):
        branches.setdefault(tuple(subs[v] for v in keys), set()).add(tuple(subs[v] for v in inner))
    from ..gpf import boolean_variables
    reads = [v for v in list(keys) + list(inner) if v in set(boolean_variables(g))]
    tensors: dict[tuple[int, ...], Any] = {}                # one contraction per value of what g reads
    out: dict[tuple[int, ...], tuple[Any, float]] = {}
    for key, values in branches.items():
        choi: Any = 0
        size = 0.0
        for inner_values in sorted(values):
            subs = dict(zip(keys, key))
            subs.update(zip(inner, inner_values))
            read = tuple(subs[v] for v in reads)
            if read not in tensors:
                used = {v: subs[v] for v in reads}
                h = g.substitute_variables(used) if used else g
                tensors[read] = h.to_tensor(preserve_scalar=True).reshape(-1)
            t = tensors[read]
            choi = choi + numpy.outer(t, t.conj())
            size = max(size, float(numpy.abs(t).max()))
        out[key] = (choi, size)
    return out


def _allowed(constraints: list[Any], keep: list[str]) -> list[dict[str, int]]:
    """The values of the variables ``keep`` for which some values of the other variables satisfy ``constraints``.

    The linear constraints are solved over GF(2) after the other variables are eliminated from them; a
    constraint with a product of variables is checked on each solution, its variables kept.
    """
    from ..gpf import evaluate_phase
    rows: list[tuple[set[str], int]] = []
    products = []
    for c in constraints:
        row = _affine_row(c)
        if row is None:
            products.append(c)
        else:
            rows.append((set(row[0]), row[1]))
    keep = list(keep) + sorted({v for c in products for v in _variables(c)} - set(keep))
    for v in sorted({v for vs, _c in rows for v in vs} - set(keep)):       # eliminate the other variables
        at = next((i for i, (vs, _c) in enumerate(rows) if v in vs), None)
        if at is None:
            continue
        pvs, pc = rows.pop(at)
        rows = [(vs ^ pvs, c ^ pc) if v in vs else (vs, c) for vs, c in rows]
    order = {v: i for i, v in enumerate(keep)}
    pivots: dict[str, tuple[set[str], int]] = {}                          # reduced row echelon form
    for vs, c in rows:
        vs = set(vs)
        for p, (pvs, pc) in pivots.items():
            if p in vs:
                vs, c = vs ^ pvs, c ^ pc
        if not vs:
            if c:
                return []                                                  # no values satisfy them
            continue
        p = min(vs, key=order.__getitem__)
        for q, (qvs, qc) in list(pivots.items()):
            if p in qvs:
                pivots[q] = (qvs ^ vs, qc ^ c)
        pivots[p] = (vs, c)
    free = [v for v in keep if v not in pivots]
    out = []
    for bits in itertools.product([0, 1], repeat=len(free)):
        subs = dict(zip(free, bits))
        for p, (vs, c) in pivots.items():
            subs[p] = (c + sum(subs[v] for v in vs if v != p)) % 2
        if not any(evaluate_phase(c, subs) for c in products):
            out.append(subs)
    return out


def _unit_scalar(g: Any) -> Any:
    """``g`` with scalar 1: the maps are compared up to a scalar, and whether a run can occur (a zero map) is decided
    by the detectors, so the scalar -- which can keep a phase in the outcomes after the reduction -- is not needed."""
    from pyzx.graph.scalar import Scalar
    h = g.copy()
    h.scalar = Scalar()
    return h


def _result_order(name: str) -> tuple[str, int]:
    head, _, rest = name.partition('[')
    return head, int(rest.rstrip(']')) if rest.rstrip(']').isdigit() else 0


def _variables(p: Any) -> set[str]:
    try:
        return {v.name for _c, t in p.terms for v, _e in t.vars}
    except AttributeError:
        return set()


def _is_zero(p: Any) -> bool:
    if hasattr(p, 'terms'):
        return all(Fraction(c) % 2 == 0 for c, _t in p.terms)
    return Fraction(p) % 2 == 0


def _substitute(p: Any, values: dict[str, Any]) -> Any:
    if not hasattr(p, 'substitute'):
        return p
    mapping = {}
    for _c, t in p.terms:
        for v, _e in t.vars:
            if v.name in values:
                mapping[v] = values[v.name]
    return p.substitute(mapping) if mapping else p


def _copy(program: QProgram, extra: Callable[[int, int, Any], list[Any]],
          final: list[Any] | None = None) -> QProgram:
    """A copy of ``program`` with the gates ``extra(stage, index, gate)`` after each gate, and ``final`` at the end."""
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
        for j, g in enumerate(st.circuit.gates):
            c.add_gate(g)
            for h in extra(st.index, j, g):
                c.add_gate(h)
        out.add_circuit(c, name=st.name)
    if final:
        c = Circuit(program.qubits, bit_amount=program.bits)
        for h in final:
            c.add_gate(h)
        out.add_circuit(c, name='output frame')
    return out


def _result_name(g: Any, stage: int) -> str:
    """The variable of a measurement's result, as PyZX names it (its ``result_symbol``, else ``c[<bit>]``, else
    ``m<qubit>``) and the program renames PyZX's own names (``s<stage>_m<qubit>``)."""
    from .program import _AUTO_VAR
    if getattr(g, 'result_symbol', None) is not None:
        name = g.result_symbol
    elif g.result_bit is not None:
        name = 'c[{}]'.format(g.result_bit)
    else:
        name = 'm{}'.format(g.target)
    return 's{}_{}'.format(stage, name) if _AUTO_VAR.match(name) else name


def _qubits_of(g: Any) -> list[int]:
    if isinstance(g, ParityControlled):
        return [g.target]
    qs = [getattr(g, a) for a in ('control', 'ctrl1', 'ctrl2') if hasattr(g, a)] + [g.target]
    return list(dict.fromkeys(qs))


def error_formulas(program: QProgram, final_check: bool = True, code_out: Any = None,
                   stages: Iterable[int] | None = None) -> ErrorFormulas:
    """The switches of a Pauli error after every gate of ``program``, pushed to the results and outputs.

    With ``final_check`` (the default), the output is checked at the end by a perfect
    measurement of the stabilizers of the code it is in (:meth:`QProgram.code_at`),
    as fault distances are usually defined: a run whose output leaves the code is
    caught, and a pattern is wrong only when it changes the logical state. Both
    outputs are decoded with the ideal decoder of that code and compared. Without
    it, the outputs themselves are compared, so any error on the output after the
    last check is wrong. The final check needs the output to be in a code; if it is
    not (for example, the program ends with a post-selection), ``code_at`` raises
    :class:`~zxqec.codes.NotACodeError`, and ``final_check=False`` is the option.

    ``code_out`` is the code to check against instead of ``code_at``: a
    :class:`~zxqec.codes.StabilizerCode` on the output qubits, in order. It is
    needed when the output is a known stabilizer state, which ``code_at`` reads as
    a code with no logical qubit (``k = 0``): against such a code a wrong state
    cannot be told from a caught one, so it is refused. (d = 3 cultivation with
    ``S`` in place of ``T`` is an example; the code of the real circuit, with its
    logical qubit, is the one to give.) A code that the noiseless program does not
    pass is refused too.

    ``stages`` are the stages with errors (all of them by default); the gates of
    the other stages are perfect, as an encoder often is in a model of the checks
    that follow it.
    """
    from ..gpf import reduce_gpf
    from ..spacetime import get_logical_qubit

    n = program.qubits
    xs: dict[int, set[str]] = {q: set() for q in range(n)}
    zs: dict[int, set[str]] = {q: set() for q in range(n)}
    flips: dict[str, frozenset[str]] = {}
    t_flips: dict[tuple[int, int], TFlip] = {}
    places: list[Place] = []
    noisy = set(range(program.num_stages)) if stages is None else set(stages)
    if not noisy <= set(range(program.num_stages)):
        raise ValueError("error_formulas: the program has stages 0 to {}, not {}".format(
            program.num_stages - 1, sorted(noisy - set(range(program.num_stages)))))
    for st in program.stages:
        if st.kind == 'permutation':
            continue
        for j, g in enumerate(st.circuit.gates):
            name, t = g.name, g.target
            if name == 'HAD':
                xs[t], zs[t] = zs[t], xs[t]
            elif name == 'CNOT':
                xs[t] ^= xs[g.control]
                zs[g.control] ^= zs[t]
            elif name == 'CZ':
                zs[t] ^= xs[g.control]
                zs[g.control] ^= xs[t]
            elif name == 'S':
                zs[t] ^= xs[t]
            elif name in ('NOT', 'Z', 'Y'):
                pass
            elif name == 'T':
                if xs[t]:
                    key = (st.index, j)
                    t_flips[key] = TFlip('f{}'.format(len(t_flips)), st.index, j, t, bool(getattr(g, 'adjoint', False)),
                                         frozenset(xs[t]))
            elif name == 'Measurement':
                result = _result_name(g, st.index)
                if result in flips:
                    raise NotImplementedError("error_formulas: the result {} is written by two measurements"
                                              .format(result))
                flips[result] = frozenset(xs[t])
                xs[t], zs[t] = set(), set()
                continue
            elif name == 'Reset':
                xs[t], zs[t] = set(), set()
            elif isinstance(g, ParityControlled) and g.angle == 1:
                s: set[str] = set()
                for bit in g.bits:
                    s ^= set(flips.get(bit, frozenset()))
                (xs if g.axis == 'X' else zs)[t] ^= s
            else:
                raise NotImplementedError("error_formulas: gate {!r} is not supported".format(str(g)))
            if st.index not in noisy:
                continue
            for q in _qubits_of(g):
                place = Place(len(places), st.index, j, name, q)
                places.append(place)
                xs[q] ^= {place.x}
                zs[q] ^= {place.z}

    graph = program.quantum_graph()
    outputs: list[int] = []
    for b in graph.outputs():
        logical = get_logical_qubit(graph, b)
        assert logical is not None, "an output of the program's diagram has no logical qubit"
        outputs.append(logical)
    outputs.sort()
    frame = {q: (frozenset(xs[q]), frozenset(zs[q])) for q in outputs}

    def extra(stage: int, j: int, g: Any) -> list[Any]:
        flip = t_flips.get((stage, j))
        if flip is None:
            return []
        return [ParityControlled('S' if flip.adjoint else 'S^dag', flip.qubit, [flip.name])]

    final = [h for q in outputs for h in (ParityControlled('X', q, ['ox{}'.format(q)]),
                                          ParityControlled('Z', q, ['oz{}'.format(q)]))]
    abstract = _copy(program, extra, final)
    reduced = reduce_gpf(abstract.quantum_graph())
    clean_res = reduce_gpf(graph)
    decoder = None
    if final_check:
        code = program.code_at(program.num_stages - 1) if code_out is None else code_out
        if list(code.qubits) != outputs:
            error = NotImplementedError if code_out is None else ValueError
            raise error("error_formulas: the output code's qubits {} are not the outputs {} in order"
                        .format(list(code.qubits), outputs))
        if code.k == 0:
            raise ValueError("error_formulas: the output code has no logical qubit, so a wrong output cannot be "
                             "told from a caught one; give code_out, a code on the outputs with a logical qubit")
        decoder = code.encoder().adjoint()
    ef = ErrorFormulas(program, places, flips, sorted(t_flips.values(), key=lambda f: int(f.name[1:])), frame,
                       abstract, reduced, list(clean_res.detectors), clean_res.to_graph(), decoder)
    if decoder is not None and ef.classify({}) != 'harmless':
        raise ValueError("error_formulas: the noiseless program does not pass the final check against this code")
    return ef
