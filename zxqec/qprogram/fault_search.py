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

"""The fewest faults that are wrong, found with a solver instead of by trying every pattern.

:meth:`ErrorFormulas.lowest_wrong <zxqec.qprogram.ErrorFormulas.lowest_wrong>`
classifies every pattern of faults up to a weight. For d = 3 cultivation and
weight 2 that is about 200 s; beyond that it is too slow. :func:`search_lowest_wrong`
asks a solver for a light pattern that *can* be wrong, and lets
:meth:`ErrorFormulas.classify` decide. It works in four steps.

* *A pattern that flips no* ``T`` *gate is a Pauli error.* Its verdict is linear in
  its signature (:meth:`ErrorFormulas.mask`). It is caught when it flips a check of
  the noiseless program, or a stabilizer of the output code (the final check).
  Otherwise it is wrong when its logical part is. A final-check row is a
  stabilizer, plus the results in its sign. A logical row is logical X or Z,
  plus a part on the results, which is learned from a few verdicts
  (:func:`branch_model`).
* *A flipped* ``T`` *gate is the gate followed by* ``S^dag`` (or ``S``), and
  ``S^dag = a I + b Z``. So a pattern that flips the ``T`` gates ``F`` is a sum of
  *branches*: its Pauli part, with a ``Z`` right after the ``T`` gates of a
  subset ``E`` of ``F``. A branch is a Pauli error. If every branch is caught or
  harmless, the runs that pass are the noiseless ones, up to a number. So a
  pattern that is wrong has a wrong branch. This is argued, not proved: when a
  value that nobody reads (a reset) is summed as a mixture, the numbers could
  differ between its values. On d = 3 cultivation every wrong pattern of 2
  faults has a wrong branch.
* *The solver* chooses faults (at most one per gate) and a subset ``E``, so that
  the branch passes every check and is a logical error, with as few faults as
  it can.
* *A lazy loop.* The solver's pattern is classified. If it is wrong, its weight
  is the answer. If not, the solver is told to skip it (a *cut*), and runs
  again. When the solver finds nothing up to ``max_weight``, no pattern of that
  many faults is wrong.

*Two cuts* (``cut``). With ``'signature'`` (the default) the solver skips the
signature of the pattern: patterns with one signature get one verdict. With
``'coset'`` it skips the pattern's *class*: every signature with the same flipped
``T`` gates and the same value on every row (:meth:`BranchModel.coset`). A class
holds many signatures, so far fewer solves are needed. Why a class has one
verdict:

* Let ``h`` be a Pauli signature that every row reads as 0
  (:meth:`BranchModel.harmless_basis`). Two signatures of one class differ by
  such an ``h``.
* The program with the signature ``m + h`` is the program with ``m``, with its
  results shifted by those of ``h``, and then ``h``'s Pauli on the outputs (this
  is how :mod:`~zxqec.qprogram.faults` builds every noisy program).
* ``h`` flips no check, so a run of ``m + h`` passes the checks when the
  shifted run of ``m`` does.
* ``h`` flips no stabilizer of the final check, so after decoding, its Pauli is
  a logical Pauli ``U``. Its logical rows read 0, so ``U`` is the identity, or the
  logical Pauli that is harmless.
* So each run of ``m + h`` is the shifted run of ``m``, with ``U`` applied after
  decoding. ``h`` alone is harmless, so the noiseless runs are related by the
  same ``U``. Two outputs are equal exactly when they are equal after ``U``.

This rests on the rows being right for Pauli errors (checked on random patterns
when the model is built) and on the formulas of
:mod:`~zxqec.qprogram.faults`. It does not need the branch argument.

With ``'coset'`` there is a second kind of cut. The first time a set ``F`` of
flipped ``T`` gates comes up, every signature with the flips ``F`` that makes one
of the checks of the program with ``F`` flipped read 1 is cut. ``classify`` calls
all of them caught, by the same rule (``ErrorFormulas._flip_functionals``), so this
needs no new argument.

The coset claim was checked against ``classify``, which does not use it. On d = 3
cultivation, the 239 signatures of at most 2 faults that have a wrong branch fall
into 106 classes, and each class has one verdict. Adding a random ``h`` changed no
verdict, in 4068 tries (1200 of them on the 12 wrong signatures). A search that
lists every wrong signature with coset cuts finds exactly the 12 signatures of the
48 wrong pairs that ``lowest_wrong(2)`` finds. On d = 5 cultivation, the 2367 such
signatures fall into 651 classes, each with one verdict, and 400 random ``h``
changed no verdict. On a small program whose output has a harmless logical ``Z``
(a bit-flip code started in ``|0>``, with ``T`` and ``T^dag`` pairs), every signature
of at most 2 faults was classified (1187 of them, 694 with ``T`` flips): each of
the 180 classes has one verdict. A program where ``Y`` is harmless and faults flip
``T`` gates was not tried.

So: a wrong pattern that the search returns is checked by ``classify``. A weight
it rules out rests on the branch argument, on the cut (the coset cut needs the
argument above), and on the solver.

The model is checked when it is built, and an error is raised instead of a wrong
answer: a ``Z`` right after a ``T`` gate must flip no ``T`` gate (so a branch is a
Pauli error); the model's verdict must agree with ``classify`` on random Pauli
patterns; and the output code must have one logical qubit.

Three solvers (``solver``): ``'cp-sat'`` (the default) and ``'scip'`` come with
OR-Tools; ``'sat'`` is an incremental SAT solver (CaDiCaL from python-sat). The SAT
solver keeps what it learned between solves, which suits the lazy loop. On d = 5
cultivation a solve took 0.1 to 0.5 s on average with SAT, and 1 to 3 s with
CP-SAT. With ``solver='sat'`` and ``cut='coset'``, no pattern of at most 2 faults
was wrong after 200 to 340 s (429 solves), and a wrong pattern of 3 faults came
after 34 minutes (3798 solves); with CP-SAT and signature cuts, weight 2 was not
done after 20 minutes. CP-SAT is slow with coset cuts (they add rows of XORs).
SCIP gave false answers ("infeasible" with solutions left) while its dual
reductions were on; they are off now. Both packages come with the optional
dependency ``codedistance`` (``pip install zxqec[distance]``); they are imported
only when a search runs.
"""

from __future__ import annotations

import itertools
import math
import random
import threading
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from .faults import ErrorFormulas, _affine_row

__all__ = ['BranchModel', 'Candidate', 'SolveRecord', 'SearchResult', 'branch_model', 'search_lowest_wrong']

_LETTERS = ('I', 'X', 'Y', 'Z')
_NO_ORTOOLS = ("search_lowest_wrong needs OR-Tools (pip install ortools); it comes with the optional dependency "
               "codedistance: pip install zxqec[distance]")
_NO_PYSAT = ("solver='sat' needs python-sat (pip install python-sat); it comes with the optional dependency "
             "codedistance: pip install zxqec[distance]")
_CUTS = ('signature', 'coset')


def _bits(m: int) -> Iterator[int]:
    """The positions of the ones of ``m``, lowest first."""
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


def _par(a: int, b: int) -> int:
    """The parity of ``a & b``."""
    return bin(a & b).count('1') & 1


def _array(masks: list[int], cols: list[int]) -> Any:
    import numpy as np
    return np.array([[m >> c & 1 for c in cols] for m in masks], dtype=np.uint8).reshape(len(masks), len(cols))


def _mask(vector: Any, cols: list[int]) -> int:
    return sum(1 << c for c, b in zip(cols, vector) if b)


@dataclass
class BranchModel:
    """The rows of "a branch passes every check and is a logical error", over the bits of a signature.

    A *row* is a mask over :attr:`columns`; a pattern's bit for it is the parity of
    the row and the pattern's signature. Built by :func:`branch_model`.
    """
    ef: ErrorFormulas
    code: Any                                       #: the output code (a :class:`~zxqec.codes.StabilizerCode`)
    per: str                                        #: ``'gate'`` or ``'place'``, as in ``single_faults``
    columns: list[str]                              #: the bits of a signature, as in ``ErrorFormulas.mask``
    detectors: list[int]                            #: one row per check of the noiseless program
    final_rows: list[int]                           #: one row per stabilizer of the output code
    final_names: list[str]                          #: the stabilizers, as text
    lx: int                                         #: 1 when a Pauli error anticommutes with logical X
    lz: int                                         #: 1 when a Pauli error anticommutes with logical Z
    harmless_logicals: tuple[str, ...]              #: the logical Paulis that are harmless (none, or one)
    z_after_t: list[int]                            #: the signature of a Z right after each T gate
    choices: list[tuple[int, list[tuple[int, dict[int, str]]]]]   #: (signature, [(gate, fault), ...])
    groups: list[list[int]]                         #: choices at one gate (when they are not merged)
    kinds: int                                      #: fault kinds with a nonempty signature
    dropped: int                                    #: fault kinds whose signature is empty
    sample: dict[str, int]                          #: verdicts of the random Pauli patterns checked
    probes: int                                     #: verdicts used to learn the logical rows
    seconds: float                                  #: time to build and check the model
    _harmless_pairs: frozenset[tuple[int, int]] = field(default=frozenset(), repr=False)

    @property
    def wrong_logicals(self) -> tuple[str, ...]:
        """The logical Paulis that are wrong: ``('X', 'Y', 'Z')`` for a ``T`` state."""
        return tuple(p for p in 'XYZ' if p not in self.harmless_logicals)

    @property
    def wrong_kind(self) -> str:
        """``'any'`` (X, Y and Z are wrong), ``'xor'`` (X and Z; Y is harmless), or ``'not <P>'``."""
        if not self.harmless_logicals:
            return 'any'
        return 'xor' if self.harmless_logicals == ('Y',) else 'not ' + self.harmless_logicals[0]

    @cached_property
    def t_columns(self) -> list[int]:
        """The column of each ``T`` flip variable."""
        return [self.columns.index(t.name) for t in self.ef.t_flips]

    @cached_property
    def coset_rows(self) -> list[int]:
        """The rows that a class fixes: the checks, the final check, and the logical rows.

        The logical rows are ``lx`` and ``lz`` when no logical Pauli is harmless;
        otherwise the one row that reads 0 on the harmless one (``lx + lz`` when
        ``Y`` is harmless).
        """
        if not self.harmless_logicals:
            logical = [self.lx, self.lz]
        else:
            hx, hz = next(p for p in self._harmless_pairs if p != (0, 0))
            logical = [(self.lx if hz else 0) ^ (self.lz if hx else 0)]
        return self.detectors + self.final_rows + logical

    def coset(self, mask: int) -> tuple[int, int]:
        """The class of a signature: ``(its T flips as a mask, its bits on the coset rows)``.

        Two signatures of one class have one verdict (see the module docstring).
        """
        bits = 0
        for k, row in enumerate(self.coset_rows):
            if _par(row, mask):
                bits |= 1 << k
        return mask & self.ef._t_bits, bits

    @cached_property
    def harmless_basis(self) -> list[int]:
        """A basis of the Pauli signatures that every coset row reads as 0.

        Each is a harmless Pauli error that flips no ``T`` gate. Adding one to a
        signature keeps its class.
        """
        from ..codes.gf2 import nullspace
        cols = [j for j in range(len(self.columns)) if not self.ef._t_bits >> j & 1]
        return [_mask(v, cols) for v in nullspace(_array(self.coset_rows, cols))]

    def names(self, row: int) -> list[str]:
        """The columns of a row (or of a signature), in order."""
        return [self.columns[j] for j in _bits(row)]

    def logical(self, mask: int) -> tuple[int, int]:
        """``(lx, lz)`` of a Pauli signature: ``(0, 1)`` is a logical X, ``(1, 0)`` a Z, ``(1, 1)`` a Y."""
        return _par(self.lx, mask), _par(self.lz, mask)

    def verdict(self, mask: int) -> str:
        """``'caught'``, ``'harmless'`` or ``'wrong'`` for a signature that flips no ``T`` gate, from the rows."""
        if mask & self.ef._t_bits:
            raise ValueError("this signature flips T gates; its branches are Pauli errors (see branches)")
        if any(_par(r, mask) for r in self.detectors) or any(_par(r, mask) for r in self.final_rows):
            return 'caught'
        return 'harmless' if self.logical(mask) in self._harmless_pairs else 'wrong'

    def branches(self, mask: int) -> list[tuple[tuple[str, ...], int]]:
        """Every branch of a signature: ``(T flips with a Z after them, the branch's signature)``.

        A signature that flips ``k`` T gates has ``2^k`` branches.
        """
        flipped = [i for i, c in enumerate(self.t_columns) if mask >> c & 1]
        base = mask & ~self.ef._t_bits
        out = []
        for r in range(len(flipped) + 1):
            for e in itertools.combinations(flipped, r):
                b = base
                for i in e:
                    b ^= self.z_after_t[i]
                out.append((tuple(self.ef.t_flips[i].name for i in e), b))
        return out

    def wrong_branch(self, mask: int) -> int | None:
        """A branch of a signature that the rows call wrong, or None; by linear algebra, not by listing branches."""
        flipped = [i for i, c in enumerate(self.t_columns) if mask >> c & 1]
        base = mask & ~self.ef._t_bits
        zs = [self.z_after_t[i] for i in flipped]
        # rows: sum of e_k (the Z after flipped gate k) times its effect = the base's effect, for every check
        pivots: list[tuple[int, int, int]] = []      # (pivot bit, row over e, right-hand side)
        for row in self.detectors + self.final_rows:
            lhs = sum(1 << k for k, z in enumerate(zs) if _par(row, z))
            rhs = _par(row, base)
            for p, prow, prhs in pivots:
                if lhs >> p & 1:
                    lhs, rhs = lhs ^ prow, rhs ^ prhs
            if not lhs:
                if rhs:
                    return None                      # no branch passes every check
                continue
            p = lhs.bit_length() - 1
            pivots = [(q, qrow ^ lhs, qrhs ^ rhs) if qrow >> p & 1 else (q, qrow, qrhs) for q, qrow, qrhs in pivots]
            pivots.append((p, lhs, rhs))
        free = [k for k in range(len(zs)) if all(p != k for p, _r, _c in pivots)]

        def branch(on: set[int]) -> int:
            e = set(on)
            for p, prow, prhs in pivots:
                if (prhs + sum(1 for k in on if prow >> k & 1)) % 2:
                    e.add(p)
            b = base
            for k in e:
                b ^= zs[k]
            return b
        b0 = branch(set())
        start = self.logical(b0)
        steps = []                                   # free choices that change the logical part
        for k in free:
            d = tuple(x ^ y for x, y in zip(self.logical(branch({k})), start))
            if any(d):
                steps.append((k, d))
        for n in range(min(len(steps), 2) + 1):
            for combo in itertools.combinations(steps, n):
                pair = start
                for _k, d in combo:
                    pair = (pair[0] ^ d[0], pair[1] ^ d[1])
                if pair not in self._harmless_pairs:
                    b = branch({k for k, _d in combo})
                    assert self.verdict(b) == 'wrong'
                    return b
        return None

    def pattern(self, chosen: list[int]) -> dict[int, str]:
        """The faults of the choices ``chosen``, each at its own gate.

        A merged choice can sit at several gates; the gates are matched to the
        choices so that no two share one (augmenting paths).
        """
        owner: dict[int, int] = {}                   # gate -> the position in chosen that sits there
        at: list[dict[int, dict[int, str]]] = [{} for _ in chosen]   # per position: gate -> its fault there
        for k, i in enumerate(chosen):
            for gate, fault in self.choices[i][1]:
                at[k].setdefault(gate, fault)

        def seat(k: int, seen: set[int]) -> bool:
            for gate in at[k]:
                if gate in seen:
                    continue
                seen.add(gate)
                if gate not in owner or seat(owner[gate], seen):
                    owner[gate] = k
                    return True
            return False
        for k in range(len(chosen)):
            if not seat(k, set()):
                raise RuntimeError("branch_model: the chosen faults cannot sit at distinct gates; a lightest pattern "
                                   "always can (two faults at one gate are one fault there)")
        out: dict[int, str] = {}
        for gate, k in sorted(owner.items(), key=lambda gk: gk[1]):
            out.update(at[k][gate])
        return out

    def describe(self) -> str:
        """The model in plain words: its sizes, its rows and its self-check."""
        def results(row: int) -> str:
            names = [c for c in self.names(row) if c in self.ef.flips]
            return ' + '.join(names) if names else 'none'
        merged = len(self.choices) < self.kinds
        checked = sum(self.sample.values())
        lines = [
            "Output code: {} on qubits {}.".format(_parameters(self.code), ', '.join(map(str, self.code.qubits))),
            "Checks: {} detectors of the noiseless program, {} final-check rows (one per stabilizer).".format(
                len(self.detectors), len(self.final_rows)),
            "Logical rows: the X row also reads the results {}; the Z row reads {}.".format(
                results(self.lx), results(self.lz)),
            "Wrong logical errors: {}{}.".format(', '.join(self.wrong_logicals),
                                                 ' ({} is harmless)'.format(', '.join(self.harmless_logicals))
                                                 if self.harmless_logicals else ' (none is harmless)'),
            "T gates: {} (a Z right after each flips no T gate).".format(len(self.z_after_t)),
            "Fault choices: {} from {} fault kinds ({}per {}); {} kinds with an empty signature dropped.".format(
                len(self.choices), self.kinds, 'one per signature, ' if merged else '', self.per, self.dropped),
            ("Self-check: {} random Pauli patterns agree with classify ({}); {} probes learned the logical rows."
             .format(checked, ', '.join('{} {}'.format(n, v) for v, n in sorted(self.sample.items())), self.probes)
             if checked else "Self-check: none (sample=0); {} probes learned the logical rows.".format(self.probes)),
            "Built in {:.1f} s.".format(self.seconds)]
        return '\n'.join(lines)


def _parameters(code: Any) -> str:
    # n and k only: the distance of a large code takes long to compute
    return '[[{}, {}]]'.format(code.n, code.k)


def _code_text(code: Any) -> tuple[Any, ...]:
    """A code as text, to tell two codes apart: qubits, stabilizers (with their signs) and logical Paulis."""
    return (tuple(code.qubits), tuple(str(g) for g in code.stabilizers), tuple(str(p) for p in code.logical_x),
            tuple(str(p) for p in code.logical_z))


def branch_model(ef: ErrorFormulas, per: str = 'gate', *, code_out: Any = None, sample: int = 48, seed: int = 0,
                 merge: bool = True) -> BranchModel:
    """The rows that :func:`search_lowest_wrong` gives the solver, learned and checked against ``ef``.

    ``code_out`` is the output code of the final check: the code that ``error_formulas``
    was given, or by default the program's code at its last stage. ``sample`` random
    Pauli patterns are classified both ways, and a disagreement raises ``RuntimeError``
    (``sample=0`` skips this, with a warning).
    With ``merge`` (the default), the fault kinds with one signature are one choice: a
    lightest pattern never has two faults with one signature (they cancel), nor two
    faults at one gate (their product is one fault there).

    Raises ``ValueError`` without the final check, or for an output code that does not
    have exactly one logical qubit; ``NotImplementedError`` for a stabilizer sign with a
    product of results, or a ``T`` gate with no place after it (``stages`` left it out).
    """
    import numpy as np

    from ..codes.gf2 import nullspace, solve

    start = time.perf_counter()
    if sample < 0:
        raise ValueError("branch_model: sample is at least 0")
    if sample == 0:
        warnings.warn("branch_model: sample=0, so the model is not checked against classify", stacklevel=2)
    program = ef.program
    if ef.decoder is None:
        raise ValueError("branch_model: the search needs the final check; build the formulas with "
                         "error_formulas(program) or error_formulas(program, code_out=...)")
    code = program.code_at(program.num_stages - 1) if code_out is None else code_out
    if code.k != 1:
        hint = (" (when error_formulas was given code_out, give the same code here)"
                if code_out is None and code.k == 0 else '')
        raise ValueError("branch_model: the output code has {} logical qubits; the search handles one{}".format(
            code.k, hint))
    outputs = sorted(ef.frame)
    if list(code.qubits) != outputs:
        raise ValueError("branch_model: the output code's qubits {} are not the outputs {} in order".format(
            list(code.qubits), outputs))
    columns = list(ef._columns)
    col = {c: i for i, c in enumerate(columns)}
    t_bits = ef._t_bits
    result_cols = [i for i, c in enumerate(columns) if c in ef.flips]
    frame_cols = [i for i, c in enumerate(columns) if c in ef._frame_names]

    def frame_row(p: Any) -> int:                     # parity with a frame mask: does that Pauli anticommute with p
        row = 0
        for j, q in enumerate(outputs):
            if p.z[j]:
                row |= 1 << col['ox{}'.format(q)]
            if p.x[j]:
                row |= 1 << col['oz{}'.format(q)]
        return row

    def frame_mask(p: Any) -> int:                    # the frame mask of the Pauli p on the outputs
        m = 0
        for j, q in enumerate(outputs):
            if p.x[j]:
                m |= 1 << col['ox{}'.format(q)]
            if p.z[j]:
                m |= 1 << col['oz{}'.format(q)]
        return m

    # -- the final check: one row per stabilizer, with the results of its sign
    final_rows, final_names = [], []
    for g in code.stabilizers:
        row = frame_row(g)
        affine = _affine_row(g.sign)
        if affine is None:
            raise NotImplementedError("branch_model: the sign of the stabilizer {} has a product of results".format(g))
        for v in affine[0]:
            if v not in ef.flips:
                raise ValueError("branch_model: the sign of the stabilizer {} reads {!r}, which is not a result"
                                 .format(g, v))
            row ^= 1 << col[v]
        final_rows.append(row)
        final_names.append(str(g))
    detectors = list(ef._detector_masks)
    if any(r & t_bits for r in detectors + final_rows):
        raise RuntimeError("branch_model: a check reads a T flip variable")

    # -- a Z right after each T gate: a Pauli error that flips no T gate
    at: dict[tuple[int, int], list[Any]] = {}
    for pl in ef.places:
        at.setdefault((pl.stage, pl.gate), []).append(pl)
    z_after_t = []
    for t in ef.t_flips:
        places = at.get((t.stage, t.gate))
        if not places:
            raise NotImplementedError("branch_model: the T gate {} (stage {}, gate {}) has no place after it (its "
                                      "stage has no errors)".format(t.name, t.stage, t.gate))
        z = ef.mask({places[0].index: 'Z'})
        if z & t_bits:
            raise RuntimeError("branch_model: a Z right after the T gate {} flips the T gates {}, so its branches are "
                               "not Pauli errors".format(t.name, sorted(ef._decode(z)[1])))
        z_after_t.append(z)

    # -- which logical Paulis are harmless, from the frame alone
    lx_frame, lz_frame = frame_row(code.logical_x[0]), frame_row(code.logical_z[0])
    letter = {'I': 0, 'X': frame_mask(code.logical_x[0]), 'Z': frame_mask(code.logical_z[0])}
    letter['Y'] = letter['X'] ^ letter['Z']

    def pair(m: int) -> tuple[int, int]:
        return _par(lx_frame, m), _par(lz_frame, m)
    probes = 0
    alone = {p: ef._verdict(letter[p]) for p in 'XYZ'}
    probes += 3
    if 'caught' in alone.values():
        raise RuntimeError("branch_model: a logical Pauli on the outputs is caught ({}); the rows do not fit".format(
            alone))
    harmless = tuple(p for p in 'XYZ' if alone[p] == 'harmless')
    if len(harmless) > 1:
        raise RuntimeError("branch_model: the logical Paulis {} are all harmless; not handled".format(harmless))
    harmless_pairs = frozenset({(0, 0)} | {pair(letter[p]) for p in harmless})

    # -- the results' part of the logical rows: probes on a basis of the results that fire no check
    if result_cols:
        basis = nullspace(_array(detectors, result_cols)) if detectors else np.eye(len(result_cols), dtype=np.uint8)
    else:
        basis = np.zeros((0, 0), dtype=np.uint8)
    free = [_mask(v, result_cols) for v in basis]
    on_frame = _array(final_rows, frame_cols)

    def fix(m: int) -> int:                           # a frame mask with the same final-check bits as m
        syndrome = np.array([_par(r, m) for r in final_rows], dtype=np.uint8)
        x = solve(on_frame, syndrome)
        if x is None:
            raise RuntimeError("branch_model: no Pauli on the outputs undoes these final-check bits")
        return _mask(x, frame_cols)
    lam_x, lam_z = [], []
    for r in free:
        p = fix(r)
        seen = {q: ef._verdict(r ^ p ^ letter[q]) for q in _LETTERS}
        probes += 4
        ok = sorted(q for q, v in seen.items() if v == 'harmless')
        if 'caught' in seen.values() or len(ok) != len(harmless_pairs):
            raise RuntimeError("branch_model: results {} with the outputs {}: verdicts {} do not fit the rows".format(
                [columns[j] for j in _bits(r)], [columns[j] for j in _bits(p)], seen))
        h = pair(letter[ok[0]])
        if {(h[0] ^ a, h[1] ^ b) for a, b in harmless_pairs} != {pair(letter[q]) for q in ok}:
            raise RuntimeError("branch_model: the harmless logical Paulis {} after the results {} are not a coset"
                               .format(ok, [columns[j] for j in _bits(r)]))
        shift = pair(p ^ letter[ok[0]])
        lam_x.append(shift[0])
        lam_z.append(shift[1])
    lx, lz = lx_frame, lz_frame
    if free:
        rows = _array(free, result_cols)
        ax = solve(rows, np.array(lam_x, dtype=np.uint8))
        az = solve(rows, np.array(lam_z, dtype=np.uint8))
        assert ax is not None and az is not None
        lx ^= _mask(ax, result_cols)
        lz ^= _mask(az, result_cols)

    # -- the fault kinds, as choices for the solver
    choices: list[tuple[int, list[tuple[int, dict[int, str]]]]] = []
    groups: list[list[int]] = []
    index: dict[int, int] = {}
    kinds = dropped = 0
    for gate, (_place, faults) in enumerate(ef.single_faults(per)):
        group = []
        for fault in faults:
            m = ef.mask(fault)
            if not m:
                dropped += 1
                continue
            kinds += 1
            if merge and m in index:
                choices[index[m]][1].append((gate, fault))
                continue
            index[m] = len(choices)
            group.append(len(choices))
            choices.append((m, [(gate, fault)]))
        if not merge and len(group) > 1:
            groups.append(group)

    model = BranchModel(ef, code, per, columns, detectors, final_rows, final_names, lx, lz, harmless, z_after_t,
                        choices, groups, kinds, dropped, {}, probes, 0.0, harmless_pairs)

    # -- the self-check: the model against classify on random Pauli patterns
    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    for n in range(sample):
        m = 0
        for r in free:
            if rng.random() < 0.5:
                m ^= r
        for c in frame_cols:
            if rng.random() < 0.5:
                m ^= 1 << c
        if n % 4 == 3:                                # random results: mostly caught by a check
            for c in result_cols:
                if rng.random() < 0.1:
                    m ^= 1 << c
        elif n % 2 == 1:                              # the final check undone: harmless or wrong
            m ^= fix(m)
        library, mine = ef._verdict(m), model.verdict(m)
        if library != mine:
            raise RuntimeError("branch_model: for the signature {} classify says {!r} and the model {!r}".format(
                ef._decode(m), library, mine))
        counts[library] += 1
    model.sample = dict(counts)
    model.seconds = time.perf_counter() - start
    return model


@dataclass(frozen=True)
class Candidate:
    """A pattern the solver proposed, and its verdict."""
    pattern: dict[int, str]
    signature: tuple[frozenset[str], frozenset[str], frozenset[str]]   #: as ``ErrorFormulas.signature``
    verdict: str
    seconds: float                                  #: the time ``classify`` took


@dataclass(frozen=True)
class SolveRecord:
    """One run of the solver: ``'optimal'``, ``'infeasible'`` or ``'time limit'``, and what it proposed."""
    status: str
    weight: int | None                              #: the fewest faults left, when proved
    seconds: float
    candidates: tuple[Candidate, ...]


@dataclass
class SearchResult:
    """What :func:`search_lowest_wrong` found. Every count of faults rests on the branch argument."""
    weight: int | None                              #: the fewest faults that are wrong, when found
    pattern: dict[int, str] | None                  #: a wrong pattern of that many faults
    verdict: str | None                             #: ``classify`` of it (``'wrong'``)
    finished: bool                                  #: False when the time limit stopped the search
    max_weight: int
    lower_bound: int                                #: every wrong pattern has at least this many faults
    solver: str
    batch: int
    solves: list[SolveRecord]
    model: BranchModel
    model_seconds: float                            #: building the model (0 when it was given)
    solver_seconds: float
    classify_seconds: float
    total_seconds: float
    cut: str = 'signature'                          #: ``'signature'`` or ``'coset'``
    check_cuts: int = 0                             #: cuts from a check with T gates flipped (``'coset'``)
    wrong: list[Candidate] = field(default_factory=list)   #: the wrong candidates (all of them with all_wrong)

    @property
    def candidates(self) -> list[Candidate]:
        return [c for s in self.solves for c in s.candidates]

    @property
    def rejected(self) -> int:
        """Signatures that the solver proposed and ``classify`` did not call wrong."""
        return sum(1 for c in self.candidates if c.verdict != 'wrong')

    @property
    def rejected_per_weight(self) -> dict[int, int]:
        out: Counter[int] = Counter()
        for s in self.solves:
            for c in s.candidates:
                if c.verdict != 'wrong' and s.weight is not None:
                    out[s.weight] += 1
        return dict(sorted(out.items()))

    @property
    def ruled_out(self) -> list[int]:
        """The numbers of faults that are never wrong (up to ``max_weight``)."""
        return list(range(1, min(self.lower_bound, self.max_weight + 1)))

    def describe(self) -> str:
        """The result in plain words."""
        ef = self.model.ef
        if self.weight is not None:
            head = "{} faults are wrong: {}.".format(self.weight, ef.describe(self.pattern or {}))
            if self.weight > 1:
                head += " No pattern of {} is wrong.".format(_fewer(self.weight - 1))
            if len(self.wrong) > 1:
                head += " {} wrong signatures listed{}.".format(
                    len(self.wrong), '' if self.finished else ' (time limit: maybe not all)')
        elif self.finished:
            head = "No pattern of {} is wrong.".format(_fewer(self.max_weight))
        else:
            head = "Not finished (time limit). "
            head += ("No pattern of {} is wrong; {} not decided.".format(_fewer(self.lower_bound - 1), _plural(
                self.lower_bound)) if self.lower_bound > 1 else "Nothing ruled out yet.")
        per = ', '.join('{} at weight {}'.format(n, w) for w, n in self.rejected_per_weight.items())
        how = ', '.join([self.solver] + (['batch {}'.format(self.batch)] if self.batch > 1 else [])
                        + (['coset cuts, {} from checks'.format(self.check_cuts)] if self.cut == 'coset' else []))
        tail = ("{} solves ({}), {} candidates rejected{}; model {:.1f} s, solver {:.1f} s, classify {:.1f} s, "
                "total {:.1f} s.".format(len(self.solves), how, self.rejected, ' ({})'.format(per) if per else '',
                                         self.model_seconds, self.solver_seconds, self.classify_seconds,
                                         self.total_seconds))
        return head + '\n' + tail


def _plural(n: int) -> str:
    return '{} fault{}'.format(n, '' if n == 1 else 's')


def _fewer(n: int) -> str:
    return _plural(n) if n == 1 else 'at most ' + _plural(n)


# -- the solvers -----------------------------------------------------------------------------------------------------
#
# Each solver has the same model: a Boolean y per choice, a bit s per column of the signature (the parity of the y
# that reach it), an e per T gate (a Z after it, only when the gate is flipped), every check and final-check row of
# the branch at parity 0, its logical part not a harmless one, and at most max_weight choices. A cut forbids an
# assignment of some bits: ``_forbid([(bit, value), ...])`` adds "not all of these bits have these values". For the
# coset cuts, a bit per set of flipped T gates that is 1 when the T bits are those, and a bit per coset row.

class _Backend:
    model: BranchModel
    s: list[Any]

    def _setup(self, model: BranchModel, coset: bool) -> None:
        self.model, self.coset = model, coset
        self._parities: dict[int, Any] = {}
        self._flips: dict[int, Any] = {}

    def _xor(self, lits: list[Any]) -> Any:           # a new bit equal to the parity of lits
        raise NotImplementedError

    def _new_bit(self) -> Any:
        raise NotImplementedError

    def _forbid(self, pairs: list[tuple[Any, int]]) -> None:
        raise NotImplementedError

    def _parity(self, mask: int) -> Any:
        """The bit for the parity of the signature on ``mask`` (made once); None when no choice reaches it."""
        if mask not in self._parities:
            lits = [self.s[j] for j in _bits(mask) if self.s[j] is not None]
            self._parities[mask] = None if not lits else lits[0] if len(lits) == 1 else self._xor(lits)
        return self._parities[mask]

    def _signature_pairs(self, mask: int) -> list[tuple[Any, int]]:
        return [(b, mask >> j & 1) for j, b in enumerate(self.s) if b is not None]

    def _flip_pairs(self, flips: int) -> list[tuple[Any, int]] | None:
        # the T bits equal to flips; None when that is impossible (a flipped gate no choice reaches)
        pairs = []
        for c in self.model.t_columns:
            if self.s[c] is None:
                if flips >> c & 1:
                    return None
                continue
            pairs.append((self.s[c], flips >> c & 1))
        return pairs

    def _flip_bit(self, flips: int) -> Any:
        """A bit that is 1 when the T bits are ``flips`` (made once); None when they never are."""
        if flips not in self._flips:
            pairs = self._flip_pairs(flips)
            bit = None
            if pairs is not None:
                bit = self._new_bit()
                self._forbid(pairs + [(bit, 0)])     # the T bits are not flips, or the bit is 1
            self._flips[flips] = bit
        return self._flips[flips]

    def _class_pairs(self, mask: int, flip_bit: bool = True) -> list[tuple[Any, int]]:
        flips = mask & self.model.ef._t_bits
        if flip_bit:
            pairs = [(self._flip_bit(flips), 1)]
        else:
            pairs = self._flip_pairs(flips) or []
        for row in self.model.coset_rows:
            b = self._parity(row)
            if b is not None:
                pairs.append((b, _par(row, mask)))
        return pairs

    def _exclusion(self, mask: int, flip_bit: bool = True) -> list[tuple[Any, int]]:
        return self._class_pairs(mask, flip_bit) if self.coset else self._signature_pairs(mask)

    def _key(self, mask: int) -> Any:
        return self.model.coset(mask) if self.coset else mask

    def cut(self, mask: int) -> None:
        """Forbid one signature."""
        self._forbid(self._signature_pairs(mask))

    def cut_class(self, mask: int) -> None:
        """Forbid the class of a signature: the same T flips and the same bits on the coset rows."""
        self._forbid(self._class_pairs(mask))

    def cut_check(self, flips: int, row: int, const: int) -> None:
        """Forbid the signatures with exactly these T flips whose parity on ``row`` is not ``const``."""
        f = self._flip_bit(flips)
        if f is None:
            return
        b = self._parity(row)
        if b is None:
            if const:
                self._forbid([(f, 1)])                # the parity is always 0, not const: these flips are out
            return
        self._forbid([(f, 1), (b, 1 ^ const)])

    def _make_coset_bits(self) -> None:
        for row in self.model.coset_rows:
            self._parity(row)


class _CpSat(_Backend):
    """The model in CP-SAT: XOR constraints, at-most-one per gate, clauses for the cuts."""

    def __init__(self, model: BranchModel, max_weight: int, seed: int, workers: int, coset: bool = False) -> None:
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise ImportError(_NO_ORTOOLS) from exc
        self._setup(model, coset)
        self.cp = cp_model
        self.seed, self.workers = seed, workers
        m = cp_model.CpModel()
        self.m = m
        self.y = [m.new_bool_var('y{}'.format(i)) for i in range(len(model.choices))]
        for group in model.groups:
            m.add_at_most_one([self.y[i] for i in group])
        by_column: list[list[int]] = [[] for _ in model.columns]
        for i, (mask, _faults) in enumerate(model.choices):
            for j in _bits(mask):
                by_column[j].append(i)
        self.s = []
        for j, idx in enumerate(by_column):            # the signature, bit by bit
            if not idx:
                self.s.append(None)                    # no fault reaches this bit
                continue
            b = m.new_bool_var('s{}'.format(j))
            m.add_bool_xor([self.y[i] for i in idx] + [b.negated()])
            self.s.append(b)
        self.e = []
        for i, c in enumerate(model.t_columns):        # a Z after T gate i, only if gate i is flipped
            v = m.new_bool_var('e{}'.format(i))
            if self.s[c] is None:
                m.add_bool_or([v.negated()])
            else:
                m.add_implication(v, self.s[c])
            self.e.append(v)

        def lits(row: int) -> list[Any]:
            return ([self.s[j] for j in _bits(row) if self.s[j] is not None]
                    + [self.e[i] for i, z in enumerate(model.z_after_t) if _par(row, z)])
        for row in model.detectors + model.final_rows:
            ls = lits(row)
            if ls:
                m.add_bool_xor([ls[0].negated()] + ls[1:])  # parity 0
        px, pz = m.new_bool_var('px'), m.new_bool_var('pz')
        m.add_bool_xor(lits(model.lx) + [px.negated()])
        m.add_bool_xor(lits(model.lz) + [pz.negated()])
        for hx, hz in model._harmless_pairs:           # the logical part is not a harmless one
            m.add_bool_or([px.negated() if hx else px, pz.negated() if hz else pz])
        self.total = cp_model.LinearExpr.sum(self.y)
        m.add(self.total <= max_weight)
        m.minimize(self.total)
        if coset:
            self._make_coset_bits()                    # before any clone, so that more() can cut classes

    def _xor(self, lits: list[Any]) -> Any:
        v = self.m.new_bool_var('')
        self.m.add_bool_xor(lits + [v.negated()])
        return v

    def _new_bit(self) -> Any:
        return self.m.new_bool_var('')

    def _forbid(self, pairs: list[tuple[Any, int]], m: Any = None) -> None:
        (self.m if m is None else m).add_bool_or([b.negated() if val else b for b, val in pairs])

    def _solver(self, time_left: float | None, workers: int) -> Any:
        solver = self.cp.CpSolver()
        solver.parameters.num_workers = workers
        solver.parameters.random_seed = self.seed
        if time_left is not None:
            solver.parameters.max_time_in_seconds = max(time_left, 0.01)
        return solver

    def at_least(self, weight: int) -> None:
        self.m.add(self.total >= weight)

    def solve(self, time_left: float | None) -> tuple[str, int | None, list[int], float]:
        solver = self._solver(time_left, self.workers)
        status = solver.solve(self.m)
        if status == self.cp.OPTIMAL:
            return ('optimal', int(round(solver.objective_value)),
                    [i for i, v in enumerate(self.y) if solver.boolean_value(v)], 0.0)
        if status == self.cp.INFEASIBLE:
            return 'infeasible', None, [], 0.0
        if status in (self.cp.FEASIBLE, self.cp.UNKNOWN):
            return 'time limit', None, [], float(solver.best_objective_bound)
        raise RuntimeError("CP-SAT: status {}".format(solver.status_name(status)))

    def more(self, weight: int, count: int, exclude: set[int], time_left: float | None) -> list[list[int]]:
        # more patterns of this weight with new signatures (new classes with coset cuts): feasibility solves on a
        # copy of the model, each one cut there (listing every solution instead would also list every subset E of
        # one pattern)
        clone = self.m.clone()
        clone.clear_objective()

        def there(pairs: list[tuple[Any, int]]) -> list[tuple[Any, int]]:
            return [(clone.get_bool_var_from_proto_index(b.index), val) for b, val in pairs]
        y = [clone.get_bool_var_from_proto_index(v.index) for v in self.y]
        clone.add(self.cp.LinearExpr.sum(y) == weight)
        for m in exclude:
            self._forbid(there(self._exclusion(m, False)), clone)
        found: list[list[int]] = []
        deadline = time.perf_counter() + time_left if time_left is not None else None
        while len(found) < count:
            left = deadline - time.perf_counter() if deadline is not None else None
            if left is not None and left <= 0:
                break
            solver = self._solver(left, self.workers)
            if solver.solve(clone) not in (self.cp.OPTIMAL, self.cp.FEASIBLE):
                break
            chosen = [i for i, v in enumerate(y) if solver.boolean_value(v)]
            m = 0
            for i in chosen:
                m ^= self.model.choices[i][0]
            found.append(chosen)
            self._forbid(there(self._exclusion(m, False)), clone)
        return found


class _Scip(_Backend):
    """The model as an integer program for SCIP: a parity as ``sum - 2 k = bit`` with an integer ``k``."""

    def __init__(self, model: BranchModel, max_weight: int, seed: int, workers: int, coset: bool = False) -> None:
        try:
            from ortools.linear_solver import pywraplp  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(_NO_ORTOOLS) from exc
        s = pywraplp.Solver.CreateSolver('SCIP')
        if s is None:
            raise ImportError("OR-Tools was built without SCIP")
        self._setup(model, coset)
        self.lp, self.solver = pywraplp, s
        # with its dual reductions on, SCIP (OR-Tools 9.15) answered "infeasible" for some feasible models here,
        # depending on the seed and on the order of the constraints; without them it did not
        s.SetSolverSpecificParametersAsString('randomization/randomseedshift = {}\nmisc/allowstrongdualreds = FALSE\n'
                                              'misc/allowweakdualreds = FALSE\n'.format(seed))
        s.SetNumThreads(workers)
        self.y = [s.BoolVar('y{}'.format(i)) for i in range(len(model.choices))]
        for group in model.groups:
            s.Add(s.Sum([self.y[i] for i in group]) <= 1)
        by_column: list[list[int]] = [[] for _ in model.columns]
        for i, (mask, _faults) in enumerate(model.choices):
            for j in _bits(mask):
                by_column[j].append(i)
        half = max_weight // 2
        self.s = []
        for j, idx in enumerate(by_column):
            if not idx:
                self.s.append(None)
                continue
            b = s.BoolVar('s{}'.format(j))
            k = s.IntVar(0, min(len(idx) // 2, half), 'ks{}'.format(j))
            s.Add(s.Sum([self.y[i] for i in idx]) - 2 * k == b)
            self.s.append(b)
        self.e = []
        for i, c in enumerate(model.t_columns):
            v = s.BoolVar('e{}'.format(i))
            s.Add(v <= (self.s[c] if self.s[c] is not None else 0))
            self.e.append(v)

        def lits(row: int) -> list[Any]:
            return ([self.s[j] for j in _bits(row) if self.s[j] is not None]
                    + [self.e[i] for i, z in enumerate(model.z_after_t) if _par(row, z)])
        for n, row in enumerate(model.detectors + model.final_rows):
            ls = lits(row)
            if ls:
                k = s.IntVar(0, len(ls) // 2, 'k{}'.format(n))
                s.Add(s.Sum(ls) == 2 * k)
        px, pz = s.BoolVar('px'), s.BoolVar('pz')
        for name, row, bit in (('lx', model.lx, px), ('lz', model.lz, pz)):
            ls = lits(row)
            k = s.IntVar(0, len(ls) // 2, 'k' + name)
            s.Add(s.Sum(ls) - 2 * k == bit)
        for hx, hz in model._harmless_pairs:
            s.Add((1 - px if hx else px) + (1 - pz if hz else pz) >= 1)
        self.total = s.Sum(self.y)
        s.Add(self.total <= max_weight)
        s.Minimize(self.total)
        if coset:
            self._make_coset_bits()

    def _new_bit(self) -> Any:
        return self.solver.BoolVar('')

    def _xor(self, lits: list[Any]) -> Any:
        s = self.solver
        v = s.BoolVar('')
        k = s.IntVar(0, len(lits) // 2, '')
        s.Add(s.Sum(lits) - 2 * k == v)
        return v

    def _forbid(self, pairs: list[tuple[Any, int]]) -> None:
        self.solver.Add(self.solver.Sum([(1 - b) if val else b for b, val in pairs]) >= 1)

    def at_least(self, weight: int) -> None:
        self.solver.Add(self.total >= weight)

    def _chosen(self) -> list[int]:
        return [i for i, v in enumerate(self.y) if v.solution_value() > 0.5]

    def solve(self, time_left: float | None) -> tuple[str, int | None, list[int], float]:
        s = self.solver
        s.SetTimeLimit(int(max(time_left, 0.01) * 1000) if time_left is not None else 0)
        # every solve starts from scratch (a re-solve was the first suspect for the false answers above)
        params = self.lp.MPSolverParameters()
        params.SetIntegerParam(params.INCREMENTALITY, params.INCREMENTALITY_OFF)
        status = s.Solve(params)
        if status == self.lp.Solver.OPTIMAL:
            return 'optimal', int(round(s.Objective().Value())), self._chosen(), 0.0
        if status == self.lp.Solver.INFEASIBLE:
            return 'infeasible', None, [], 0.0
        if status in (self.lp.Solver.FEASIBLE, self.lp.Solver.NOT_SOLVED):
            return 'time limit', None, [], float(s.Objective().BestBound())
        raise RuntimeError("SCIP: status {}".format(status))

    def more(self, weight: int, count: int, exclude: set[int], time_left: float | None) -> list[list[int]]:
        # the other solutions SCIP kept from the last solve, at the same weight, with new signatures (classes)
        found: list[list[int]] = []
        seen = {self._key(m) for m in exclude}
        while len(found) < count and self.solver.NextSolution():
            chosen = self._chosen()
            if len(chosen) != weight:
                continue
            m = 0
            for i in chosen:
                m ^= self.model.choices[i][0]
            if self._key(m) not in seen:
                seen.add(self._key(m))
                found.append(chosen)
        return found


class _Sat(_Backend):
    """The model as clauses for an incremental SAT solver (python-sat).

    A parity is a chain of XORs, the number of choices a totalizer. The weight goes
    up one at a time: the solver is asked for a solution with at most ``level``
    choices, and ``level`` grows when there is none. Cuts are clauses added between
    solves, and the solver keeps what it learned. Every solution found at ``level``
    has exactly ``level`` choices: a lighter one would have been found at a lower
    level, where fewer cuts were in place.
    """

    def __init__(self, model: BranchModel, max_weight: int, seed: int, workers: int, coset: bool = False) -> None:
        try:
            from pysat.card import ITotalizer  # type: ignore[import-untyped]
            from pysat.solvers import NoSuchSolverError, Solver  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(_NO_PYSAT) from exc
        self._setup(model, coset)
        self.top = 0
        self.sat: Any = None                          # clauses are kept in a list until the solver is made
        self.clauses: list[list[int]] = []
        self.y = [self._new() for _ in model.choices]
        for group in model.groups:
            for a, b in itertools.combinations(group, 2):
                self.clauses.append([-self.y[a], -self.y[b]])
        by_column: list[list[int]] = [[] for _ in model.columns]
        for i, (mask, _faults) in enumerate(model.choices):
            for j in _bits(mask):
                by_column[j].append(i)
        self.s = []
        for idx in by_column:
            lits = [self.y[i] for i in idx]
            self.s.append(None if not lits else lits[0] if len(lits) == 1 else self._xor(lits))
        self.e = []
        for c in model.t_columns:
            v = self._new()
            self.clauses.append([-v] + ([self.s[c]] if self.s[c] is not None else []))
            self.e.append(v)

        def branch_bit(row: int) -> Any:            # the row on the branch: its parity on s, plus the e it reads
            ls = [self._parity(row)] + [self.e[i] for i, z in enumerate(model.z_after_t) if _par(row, z)]
            ls = [b for b in ls if b is not None]
            return None if not ls else ls[0] if len(ls) == 1 else self._xor(ls)
        for row in model.detectors + model.final_rows:
            b = branch_bit(row)
            if b is not None:
                self.clauses.append([-b])
        px, pz = branch_bit(model.lx), branch_bit(model.lz)
        for hx, hz in model._harmless_pairs:          # the logical part is not a harmless one
            self._forbid([(p, h) for p, h in ((px, hx), (pz, hz)) if p is not None]
                         if all(p is not None or not h for p, h in ((px, hx), (pz, hz))) else None)
        if coset:
            self._make_coset_bits()
        self.max_weight = max_weight
        self.total = ITotalizer(lits=self.y, ubound=max_weight, top_id=self.top)
        self.top = self.total.top_id
        for name in ('cadical195', 'glucose4'):
            try:
                self.sat = Solver(name=name, bootstrap_with=self.clauses + self.total.cnf.clauses)
                self.name = name
                break
            except NoSuchSolverError:
                continue
        if self.sat is None:
            raise ImportError("python-sat has neither CaDiCaL 1.9.5 nor Glucose 4")
        self.clauses = []
        self.level = 1

    def _new(self) -> int:
        self.top += 1
        return self.top

    _new_bit = _new

    def _xor(self, lits: list[Any]) -> Any:
        acc = lits[0]
        for b in lits[1:]:
            v = self._new()
            self._add([[-v, acc, b], [-v, -acc, -b], [v, -acc, b], [v, acc, -b]])
            acc = v
        return acc

    def _add(self, clauses: list[list[int]]) -> None:
        if self.sat is None:
            self.clauses.extend(clauses)
        else:
            for c in clauses:
                self.sat.add_clause(c)

    def _forbid(self, pairs: list[tuple[Any, int]] | None, selector: int | None = None) -> None:
        # None: nothing to forbid (a bit that is always 0 would have to be 1)
        if pairs is None:
            return
        clause = [(-b if val else b) for b, val in pairs] + ([-selector] if selector else [])
        if not clause:                               # forbid everything: an empty clause, as two units
            z = self._new()
            self._add([[z], [-z]])
            return
        self._add([clause])

    def _at_most(self, level: int) -> list[int]:
        return [-self.total.rhs[level]] if level < len(self.total.rhs) else []

    def _run(self, assumptions: list[int], time_left: float | None) -> bool | None:
        # None when the time limit stopped it. CaDiCaL cannot be interrupted: the limit is checked between solves
        if time_left is not None and time_left <= 0:
            return None
        if time_left is None or self.name == 'cadical195':
            return self.sat.solve(assumptions=assumptions)
        timer = threading.Timer(time_left, self.sat.interrupt)
        timer.start()
        try:
            return self.sat.solve_limited(assumptions=assumptions, expect_interrupt=True)
        finally:
            timer.cancel()
            self.sat.clear_interrupt()

    def _chosen(self) -> list[int]:
        model = self.sat.get_model()
        return [i for i, v in enumerate(self.y) if model[v - 1] > 0]

    def at_least(self, weight: int) -> None:
        self.level = max(self.level, weight)

    def solve(self, time_left: float | None) -> tuple[str, int | None, list[int], float]:
        deadline = time.perf_counter() + time_left if time_left is not None else None
        while self.level <= self.max_weight:
            left = deadline - time.perf_counter() if deadline is not None else None
            found = self._run(self._at_most(self.level), left)
            if found is None:
                return 'time limit', None, [], float(self.level)
            if found:
                chosen = self._chosen()
                if len(chosen) != self.level:
                    raise RuntimeError("SAT: a solution with {} choices at level {}".format(len(chosen), self.level))
                return 'optimal', self.level, chosen, 0.0
            self.level += 1
        return 'infeasible', None, [], 0.0

    def more(self, weight: int, count: int, exclude: set[int], time_left: float | None) -> list[list[int]]:
        # more solutions at this weight, each new one excluded by a clause that only holds while a selector is
        # assumed; the selector is switched off at the end
        selector = self._new()
        for m in exclude:
            self._forbid(self._exclusion(m), selector)
        found: list[list[int]] = []
        deadline = time.perf_counter() + time_left if time_left is not None else None
        while len(found) < count:
            left = deadline - time.perf_counter() if deadline is not None else None
            if not self._run(self._at_most(weight) + [selector], left):
                break
            chosen = self._chosen()
            m = 0
            for i in chosen:
                m ^= self.model.choices[i][0]
            found.append(chosen)
            self._forbid(self._exclusion(m), selector)
        self._add([[-selector]])
        return found


_SOLVERS: dict[str, Any] = {'cp-sat': _CpSat, 'scip': _Scip, 'sat': _Sat}


def search_lowest_wrong(ef: ErrorFormulas, max_weight: int = 3, *, per: str = 'gate', code_out: Any = None,
                        solver: str = 'cp-sat', batch: int = 1, cut: str = 'signature', all_wrong: bool = False,
                        time_limit: float | None = None, seed: int = 0, workers: int = 1,
                        log: Callable[[str], None] | None = None, model: BranchModel | None = None) -> SearchResult:
    """The fewest faults (at most ``max_weight``) that are wrong, found with a solver (see the module docstring).

    The same question as :meth:`ErrorFormulas.lowest_wrong`, with faults per gate or
    per place (``per``). The solver proposes the lightest pattern with a wrong branch;
    ``classify`` decides it; a pattern that is not wrong is cut, and the solver runs
    again. ``cut='signature'`` (the default) cuts its signature. ``cut='coset'`` cuts
    its class, and the signatures that a check with its ``T`` gates flipped catches
    (see the module docstring). With ``batch`` above 1, up to that many patterns
    of the lightest weight, with distinct signatures (classes), are found before they
    are classified: with CP-SAT and SAT by more solves, with SCIP from the solutions
    it kept.

    ``solver`` is ``'cp-sat'`` (the default), ``'scip'`` or ``'sat'``; ``seed`` and
    ``workers`` are passed to the first two. ``time_limit`` (seconds, for the whole
    search, the model included) stops the search with ``finished=False`` and the
    weights ruled out so far; the SAT solver is stopped only between solves.
    ``code_out`` is the output code (see :func:`branch_model`), and ``model`` a model
    built before, for ``ef`` and ``per``. ``log`` gets one line per solve.

    With ``all_wrong``, the search does not stop at the first wrong pattern. It goes
    on up to ``max_weight`` and lists every wrong signature that has a wrong branch
    (:attr:`SearchResult.wrong`); a wrong one is cut by its signature only.
    """
    if solver not in _SOLVERS:
        raise ValueError("solver is 'cp-sat', 'scip' or 'sat', not {!r}".format(solver))
    if cut not in _CUTS:
        raise ValueError("cut is 'signature' or 'coset', not {!r}".format(cut))
    if max_weight < 1 or batch < 1:
        raise ValueError("max_weight and batch are at least 1")
    start = time.perf_counter()
    deadline = start + time_limit if time_limit is not None else None
    if model is None:
        model = branch_model(ef, per, code_out=code_out, seed=seed)
        model_seconds = model.seconds
        if log:
            log("model: {} choices, {} rows, built in {:.1f} s".format(
                len(model.choices), len(model.detectors) + len(model.final_rows) + 2, model.seconds))
    elif model.ef is not ef:
        raise ValueError("search_lowest_wrong: the model was built for other error formulas")
    elif model.per != per:
        raise ValueError("search_lowest_wrong: the model has faults per {!r}, not per {!r}".format(model.per, per))
    elif code_out is not None and code_out is not model.code and _code_text(code_out) != _code_text(model.code):
        raise ValueError("search_lowest_wrong: the model was built for another output code")
    else:
        model_seconds = 0.0
    backend = _SOLVERS[solver](model, max_weight, seed, workers, cut == 'coset')
    t_bits = ef._t_bits
    lower = 1                                         # every wrong pattern has at least this many faults
    floor, bounded = 1, 0                             # what is left has at least floor faults; told to the solver
    records: list[SolveRecord] = []
    wrong: list[Candidate] = []
    solver_seconds = classify_seconds = 0.0
    weight = pattern = verdict = None
    finished = False
    check_cuts = 0

    seen_flips: set[int] = set()

    def checks_of(m: int) -> int:
        # the first time a set of flipped T gates is seen: cut every signature with those flips that makes one of
        # its checks read 1 (classify calls them all caught); the number of cuts
        flips = m & t_bits
        if cut != 'coset' or not flips or flips in seen_flips:
            return 0
        seen_flips.add(flips)
        if flips not in ef._flip_checks:
            ef._flip_checks[flips] = ef._flip_functionals(flips)
        checks = ef._flip_checks[flips] or []
        for row, const in checks:
            backend.cut_check(flips, row, const)
        return len(checks)

    while True:
        left = deadline - time.perf_counter() if deadline is not None else None
        if left is not None and left <= 0:
            if log:
                log("time limit before solve {}".format(len(records) + 1))
            break
        if floor > bounded:                           # cuts only remove patterns: the lightest never gets lighter
            backend.at_least(floor)
            bounded = floor
        t0 = time.perf_counter()
        status, w, chosen, bound = backend.solve(left)
        sols = [chosen] if status == 'optimal' else []
        if status == 'optimal' and batch > 1:
            assert w is not None
            first = 0
            for i in chosen:
                first ^= model.choices[i][0]
            left = deadline - time.perf_counter() if deadline is not None else None
            sols += backend.more(w, batch - 1, {first}, left)
        seconds = time.perf_counter() - t0
        solver_seconds += seconds
        if status == 'infeasible':
            records.append(SolveRecord(status, None, seconds, ()))
            finished = True
            if weight is None:
                lower = max_weight + 1
            if log:
                log("solve {}: nothing left up to {} ({:.2f} s)".format(len(records), _plural(max_weight), seconds))
            break
        if status == 'time limit':
            records.append(SolveRecord(status, None, seconds, ()))
            if weight is None and math.isfinite(bound):
                lower = max(lower, math.ceil(bound - 1e-6))
            if log:
                log("solve {}: time limit ({:.2f} s); at least {}".format(len(records), seconds, _plural(lower)))
            break
        assert w is not None
        floor = w
        if weight is None:
            lower = w
        candidates = []
        stop = False
        for sol in sols:
            p = model.pattern(sol)
            m = ef.mask(p)
            cached = m in ef._noisy_cache
            t0 = time.perf_counter()
            v = ef.classify(p)
            dt = time.perf_counter() - t0
            if not cached:
                ef._noisy_cache.pop(m, None)          # kept for outcomes(), which can rebuild it; not needed here
            classify_seconds += dt
            c = Candidate(p, ef._decode(m), v, dt)
            candidates.append(c)
            check_cuts += checks_of(m)
            if v == 'wrong':
                if weight is None:
                    weight, pattern, verdict = w, p, v
                wrong.append(c)
                if not all_wrong:
                    finished = stop = True
                    break
                backend.cut(m)                        # the signature only: the rest of its class is listed too
                continue
            if cut == 'coset':
                backend.cut_class(m)
            else:
                backend.cut(m)
        records.append(SolveRecord(status, w, seconds, tuple(candidates)))
        if log:
            log("solve {}: weight {} ({:.2f} s); {}".format(
                len(records), w, seconds, '; '.join('{} [{} T flips] ({:.2f} s)'.format(c.verdict, len(c.signature[1]),
                                                                                         c.seconds)
                                                    for c in candidates)))
        if stop:
            break
    total = time.perf_counter() - start
    if log:
        log("done: weight {}, finished {}, at least {}, {} rejected, {} wrong, {} check cuts, {:.1f} s".format(
            weight, finished, _plural(lower), sum(1 for r in records for c in r.candidates if c.verdict != 'wrong'),
            len(wrong), check_cuts, total))
    return SearchResult(weight, pattern, verdict, finished, max_weight, lower, solver, batch, records, model,
                        model_seconds, solver_seconds, classify_seconds, total, cut, check_cuts, wrong)
