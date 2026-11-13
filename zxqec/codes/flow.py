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

"""Naming a Clifford class of runs without a matrix: the Clifford branch of the decompilation flow.

The paper's methodology (its flow chart) takes the logical map
``L = E ; P ; E^dag`` and, when ``L`` is Clifford, brings it to a normal form
instead of evaluating it. This module does that for one class of runs: the
results are fixed, every phase is a multiple of ``pi/2``, and the map is read
through its Choi state (inputs bent into outputs), a stabilizer state whose
stabilizers :func:`~zxqec.codes.code_of` reads by Gaussian elimination. The
stabilizer group, in reduced row echelon form, with the sign of each row, fixes
the map up to a scalar, for an isometry (the iAPNF case) and for any other
Clifford map, such as a projector or a measurement, alike.

:class:`LogicalAction.branches` asks the same four questions of each class that
it asks of a matrix, and gets the same answers:

* :func:`identify` -- the name from the fixed list and the Pauli correction, with
  the same order of trying Paulis (lightest first) and names as
  :func:`~zxqec.codes.logical._identify`; the named maps' groups are computed once
  from their small matrices, and only the Clifford ones can match;
* :func:`factorise` -- the split into code blocks: a stabilizer state is a product
  over a split of its qubits exactly when the stabilizers supported inside the
  parts generate the whole group (a rank count);
* :func:`relating_pauli` -- the Pauli that relates a class to the first class:
  two maps with the same group differ by a Pauli, which flips the signs of the
  rows it anticommutes with, so it solves a linear system over GF(2); when the
  solution is not unique the lightest is taken, in the order the matrix route
  tries them;
* :func:`is_unitary` -- no stabilizer is supported on the outputs alone or the
  inputs alone, and there are as many inputs as outputs.

A class that is not Clifford (a ``T`` gate left), the zero map, or a scalar is
named by its matrix, as before.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Any

import numpy as np

from .gf2 import nullspace, rank, rref

__all__ = ['StabMap', 'stab_map_of', 'identify', 'factorise', 'relating_pauli', 'is_unitary', 'is_clifford_graph']


class StabMap:
    """A Clifford map ``k_in -> k_out`` as the stabilizer group of its Choi state (outputs first, then inputs).

    ``paulis`` are the rows of the group's reduced row echelon form, as signed
    Pauli strings on ``k_out + k_in`` qubits (:class:`~zxqec.codes.iapnf._Pauli`, whose
    products keep the sign); ``key`` is the unsigned form, ``signs`` the 0/1 signs.
    """

    def __init__(self, k_in: int, k_out: int, gens: list[Any]) -> None:
        self.k_in, self.k_out = k_in, k_out
        self.n = k_in + k_out
        self.paulis = _echelon(gens, self.n)
        self.rows = np.array([np.concatenate([p.x, p.z]) for p in self.paulis], np.uint8).reshape(
            len(self.paulis), 2 * self.n)
        self.signs = np.array([int(p.sign()) % 2 for p in self.paulis], np.uint8)
        self.key = self.rows.tobytes() + bytes([self.n])


def stab_map_of(h: Any, k_in: int, k_out: int) -> StabMap:
    """The :class:`StabMap` of a Clifford diagram with no Boolean variables, read off its Choi state."""
    from .iapnf import _Pauli
    from .tableau import _choi_generators
    gens = _choi_generators(h)
    return StabMap(k_in, k_out, [_Pauli.of(p) for p in gens])


def is_clifford_graph(h: Any) -> bool:
    from .tableau import _is_clifford_angle
    return all(_is_clifford_angle(h.phase(v)) for v in h.vertices())


# -- the four questions ---------------------------------------------------------------------------

def identify(d: StabMap) -> tuple[str | None, str, str]:
    """As :func:`~zxqec.codes.logical._identify`: ``(name, Pauli on outputs, Pauli on inputs)``."""
    lib = _library(d.k_in, d.k_out)
    cands = [(name, g) for name, g in lib if g.key == d.key]
    if not cands:
        return None, '', ''
    on_inputs = d.k_out == 0                       # an effect: the Pauli acts before it, on the inputs
    k = d.k_in if on_inputs else d.k_out
    for letters in _lightest_first(k):
        q = _pauli_vector(letters, d, on_inputs)
        flips = _omega(q, d.rows)
        for name, g in cands:
            if np.array_equal(d.signs, g.signs ^ flips):
                return (name, '', letters) if on_inputs else (name, letters, 'I' * d.k_in)
    return None, '', ''


def factorise(d: StabMap, parts: list[tuple[list[int], list[int]]]) -> list[StabMap] | None:
    """As :func:`~zxqec.codes.logical._factorise`: one map per part, or ``None`` if ``d`` is not a product."""
    from .iapnf import _Pauli, _product
    columns = []
    for ins, outs in parts:
        columns.append(list(outs) + [d.k_out + i for i in ins])
    flat = sorted(c for cols in columns for c in cols)
    if flat != list(range(d.n)):
        return None                                # the parts do not cover each qubit once
    out, total = [], 0
    for (ins, outs), cols in zip(parts, columns):
        other = [c for c in range(d.n) if c not in cols]
        other2 = other + [d.n + c for c in other]
        if other:
            combos = nullspace(d.rows[:, other2].T.copy()) if len(d.paulis) else np.zeros((0, 0), np.uint8)
        else:
            combos = np.eye(len(d.paulis), dtype=np.uint8)
        gens = []
        for c in combos:
            p = _product(d.paulis, c, d.n)
            gens.append(_Pauli(p.x[cols], p.z[cols], p.e, p.s))
        sub = StabMap(len(ins), len(outs), gens)
        total += len(sub.paulis)
        out.append(sub)
    if total != len(d.paulis):
        return None
    return out


def relating_pauli(d: StabMap, ref: StabMap, effect_inputs: list[int]) -> tuple[str, str] | None:
    """As :func:`~zxqec.codes.logical._relating_pauli`: ``(Pauli on outputs, Pauli on inputs)`` with ``d = Q ref Q'``."""
    if d.key != ref.key:
        return None
    delta = d.signs ^ ref.signs
    k_in, k_out = d.k_in, d.k_out
    identity_in = 'I' * k_in
    # first a Pauli on the outputs alone: a linear system; unique for a unitary, else the lightest that fits
    a = _omega_matrix(d.rows, list(range(k_out)), d.n)
    sols = _solutions(a, delta)
    if sols is not None:
        x0, kernel = sols
        if kernel.shape[0] == 0:
            return _letters(x0, k_out), identity_in
        for letters in (_lightest_first(k_out) if k_out else ['']):
            if np.array_equal(_omega(_pauli_vector(letters, d, False), d.rows), delta):
                return letters, identity_in
    # then a Pauli on the outputs with one on the inputs of measured parts, in the matrix route's order
    ins = []
    for letters in itertools.product('IXYZ', repeat=len(effect_inputs)):
        full = ['I'] * k_in
        for i, c in zip(effect_inputs, letters):
            full[i] = c
        ins.append(''.join(full))
    if identity_in not in ins:
        ins.append(identity_in)
    for qo in (_lightest_first(k_out) if k_out else ['']):
        vo = _omega(_pauli_vector(qo, d, False), d.rows)
        for qi in ins:
            if np.array_equal(vo ^ _omega(_pauli_vector(qi, d, True), d.rows), delta):
                return qo, qi
    return None


def is_unitary(d: StabMap) -> bool:
    """As :func:`~zxqec.codes.logical._is_unitary`: a unitary's Choi state has no stabilizer on one side alone."""
    if d.k_in != d.k_out:
        return False
    outs, ins = list(range(d.k_out)), list(range(d.k_out, d.n))
    for side in (outs, ins):
        other = [c for c in range(d.n) if c not in side]
        other2 = other + [d.n + c for c in other]
        if len(d.paulis) and nullspace(d.rows[:, other2].T.copy()).shape[0]:
            return False
    return len(d.paulis) == d.n


# -- helpers --------------------------------------------------------------------------------------

def _echelon(gens: list[Any], n: int) -> list[Any]:
    """The reduced row echelon form of the group, as signed Paulis (columns: x then z)."""
    rows = [g for g in gens]
    vec = lambda p: np.concatenate([p.x, p.z]).astype(np.uint8)
    out, r = [], 0
    rows = list(rows)
    for col in range(2 * n):
        piv = next((i for i in range(r, len(rows)) if vec(rows[i])[col]), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and vec(rows[i])[col]:
                rows[i] = rows[i] * rows[r]
        r += 1
    return [p for p in rows[:r]]


def _lightest_first(k: int) -> list[str]:
    return [''.join(t) for t in sorted(itertools.product('IXYZ', repeat=k),
                                       key=lambda t: (sum(c != 'I' for c in t), t))]


def _pauli_vector(letters: str, d: StabMap, on_inputs: bool) -> np.ndarray:
    """``(x | z)`` on all ``n`` qubits of the Choi state for a Pauli on the outputs (or the inputs)."""
    v = np.zeros(2 * d.n, np.uint8)
    off = d.k_out if on_inputs else 0
    for j, c in enumerate(letters):
        if c in 'XY':
            v[off + j] = 1
        if c in 'ZY':
            v[d.n + off + j] = 1
    return v


def _omega(q: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """For each row, whether the Pauli ``q`` anticommutes with it (flips its sign)."""
    if not len(rows):
        return np.zeros(0, np.uint8)
    n = rows.shape[1] // 2
    return ((rows[:, :n].astype(np.int64) @ q[n:].astype(np.int64)
             + rows[:, n:].astype(np.int64) @ q[:n].astype(np.int64)) % 2).astype(np.uint8)


def _omega_matrix(rows: np.ndarray, qubits: list[int], n: int) -> np.ndarray:
    """The matrix of ``q -> omega(q, rows)`` for Paulis ``q`` on ``qubits``: unknowns ``x`` then ``z`` there."""
    cols = [rows[:, n + j] for j in qubits] + [rows[:, j] for j in qubits]
    return np.array(cols, np.uint8).T.reshape(len(rows), 2 * len(qubits))


def _solutions(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """One solution of ``a x = b`` over GF(2) and a basis of the kernel, or ``None``."""
    from .gf2 import solve
    cols = a.shape[1]
    if a.shape[0] == 0:
        return np.zeros(cols, np.uint8), np.eye(cols, dtype=np.uint8)
    x = solve(a.copy(), b.copy())
    if x is None:
        return None
    return x, nullspace(a.copy())


def _letters(x: np.ndarray, k: int) -> str:
    return ''.join({(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}[(int(x[j]), int(x[k + j]))] for j in range(k))


@lru_cache(maxsize=None)
def _library(k_in: int, k_out: int) -> tuple[tuple[str, StabMap], ...]:
    """The Clifford maps of :func:`~zxqec.codes.logical._library`, in its order, as :class:`StabMap`."""
    from . import logical as L
    from .iapnf import _Pauli
    out = []
    for name, m in L._library(k_in, k_out).items():
        gens = _stabilizers_of_matrix(np.asarray(m, dtype=complex).reshape(2 ** k_out, 2 ** k_in), k_in + k_out)
        if gens is not None:
            out.append((name, StabMap(k_in, k_out, [_Pauli.hermitian(x, z, s) for x, z, s in gens])))
    return tuple(out)


def _stabilizers_of_matrix(m: np.ndarray, n: int) -> list[tuple[np.ndarray, np.ndarray, int]] | None:
    """The stabilizers of the Choi vector of ``m`` (outputs first), or ``None`` if it is not a stabilizer state."""
    psi = m.reshape(-1)
    norm = np.linalg.norm(psi)
    if norm < 1e-12:
        return None
    psi = psi / norm
    mats = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]), 'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.diag([1, -1])}
    found = []
    for letters in itertools.product('IXYZ', repeat=n):
        p = np.array([[1]])
        for c in letters:
            p = np.kron(p, mats[c])
        e = np.vdot(psi, p @ psi)
        if abs(abs(e) - 1) < 1e-9:
            x = np.array([c in 'XY' for c in letters], np.uint8)
            z = np.array([c in 'ZY' for c in letters], np.uint8)
            found.append((x, z, 0 if e.real > 0 else 1))
    if len(found) != 2 ** n:
        return None
    vecs = np.array([np.concatenate([x, z]) for x, z, _s in found], np.uint8)
    if rank(vecs) != n:
        return None
    return found
