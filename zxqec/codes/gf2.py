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

"""Small linear-algebra helpers over GF(2), on 0/1 numpy arrays.

Everything here is textbook Gaussian elimination; it is kept separate from the
split matrices of :mod:`zxqec.gpf.matrix`, which carry phases along with the
rows and follow the thesis's pivot rules.
"""

from __future__ import annotations

import numpy as np

__all__ = ['as_matrix', 'rref', 'rank', 'nullspace', 'reduce_mod', 'in_rowspace', 'solve',
           'complement_basis']


def as_matrix(rows: object, ncols: int) -> np.ndarray:
    """A 2D uint8 array from a list of rows (possibly empty) with ``ncols`` columns."""
    a = np.array(rows, dtype=np.uint8).reshape(-1, ncols) if ncols else np.zeros((len(rows), 0), np.uint8)  # type: ignore[arg-type]
    return a % 2


def rref(a: np.ndarray, track: bool = False) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Reduced row echelon form of ``a`` over GF(2).

    Returns ``(R, pivots, T)``: the non-zero rows of the RREF, the pivot column of
    each, and (if ``track``) the matrix ``T`` with ``T @ a = R`` over GF(2), so a
    caller can tell which original rows each row of ``R`` combines. Without
    ``track``, ``T`` is an empty array.
    """
    m = (a.copy() % 2).astype(np.uint8)
    rows, cols = m.shape
    t = np.eye(rows, dtype=np.uint8) if track else np.zeros((0, 0), np.uint8)
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        hit = next((i for i in range(r, rows) if m[i, c]), None)
        if hit is None:
            continue
        if hit != r:
            m[[r, hit]] = m[[hit, r]]
            if track:
                t[[r, hit]] = t[[hit, r]]
        for i in range(rows):
            if i != r and m[i, c]:
                m[i] ^= m[r]
                if track:
                    t[i] ^= t[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return m[:r], pivots, (t[:r] if track else t)


def rank(a: np.ndarray) -> int:
    return len(rref(a)[1])


def nullspace(a: np.ndarray) -> np.ndarray:
    """A basis (as rows) of ``{x : a @ x = 0}`` over GF(2), in a canonical order."""
    cols = a.shape[1]
    r, pivots, _ = rref(a)
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        x = np.zeros(cols, dtype=np.uint8)
        x[f] = 1
        for row, p in zip(r, pivots):
            if row[f]:
                x[p] = 1
        basis.append(x)
    return as_matrix(basis, cols)


def reduce_mod(v: np.ndarray, r: np.ndarray, pivots: list[int]) -> np.ndarray:
    """``v`` with the pivot columns of the RREF ``(r, pivots)`` cleared: its normal form modulo the row space."""
    v = v.copy() % 2
    for row, p in zip(r, pivots):
        if v[p]:
            v ^= row
    return v


def in_rowspace(v: np.ndarray, r: np.ndarray, pivots: list[int]) -> bool:
    return not reduce_mod(v, r, pivots).any()


def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """One solution ``x`` of ``a @ x = b`` over GF(2), or ``None``."""
    rows, cols = a.shape
    aug = np.concatenate([a % 2, (b % 2).reshape(rows, 1)], axis=1).astype(np.uint8)
    r, pivots, _ = rref(aug)
    if cols in pivots:
        return None
    x = np.zeros(cols, dtype=np.uint8)
    for row, p in zip(r, pivots):
        x[p] = row[cols]
    return x


def complement_basis(sub: np.ndarray, whole: np.ndarray) -> np.ndarray:
    """A canonical basis of ``rowspace(whole)`` modulo ``rowspace(sub)``.

    Each row of ``whole`` is reduced modulo ``sub`` (pivot columns of the RREF
    of ``sub`` cleared), and the reduced rows are brought to RREF. The result
    depends only on the two row spaces, not on the rows given.
    """
    cols = whole.shape[1]
    rs, ps, _ = rref(sub) if sub.size else (np.zeros((0, cols), np.uint8), [], None)
    reduced = as_matrix([reduce_mod(v, rs, ps) for v in whole], cols)
    return rref(reduced)[0]
