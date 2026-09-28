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

"""Pauli strings with a (possibly symbolic) sign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ['PauliString']

_LETTER = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}


@dataclass(frozen=True)
class PauliString:
    """The operator ``(-1)^sign * P_1 (x) ... (x) P_n``.

    ``x[j]`` and ``z[j]`` give the letter on qubit ``j`` (``Y`` when both are 1).
    ``sign`` is 0, 1, or a Boolean polynomial in measurement outcomes: the sign
    of a stabilizer that a measurement has projected onto.
    """
    x: tuple[int, ...]
    z: tuple[int, ...]
    sign: Any = 0

    @classmethod
    def from_arrays(cls, x: Any, z: Any, sign: Any = 0) -> 'PauliString':
        return cls(tuple(int(b) % 2 for b in x), tuple(int(b) % 2 for b in z), sign)

    @classmethod
    def from_str(cls, s: str, sign: Any = 0) -> 'PauliString':
        """``'XXIZ'`` style, with an optional leading ``+`` or ``-``."""
        if s[:1] in '+-':
            sign, s = (1 if s[0] == '-' else 0), s[1:]
        inv = {v: k for k, v in _LETTER.items()}
        return cls(tuple(inv[c][0] for c in s), tuple(inv[c][1] for c in s), sign)

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def letters(self) -> str:
        return ''.join(_LETTER[(a, b)] for a, b in zip(self.x, self.z))

    @property
    def weight(self) -> int:
        return sum(1 for a, b in zip(self.x, self.z) if a or b)

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(j for j, (a, b) in enumerate(zip(self.x, self.z)) if a or b)

    @property
    def symplectic(self) -> np.ndarray:
        """The row ``(x | z)`` over GF(2)."""
        return np.array(self.x + self.z, dtype=np.uint8)

    def commutes_with(self, other: 'PauliString') -> bool:
        s = sum(a * d + b * c for a, b, c, d in zip(self.x, self.z, other.x, other.z))
        return s % 2 == 0

    def restricted(self, columns: list[int]) -> 'PauliString':
        return PauliString(tuple(self.x[j] for j in columns), tuple(self.z[j] for j in columns), self.sign)

    def sign_str(self) -> str:
        s = self.sign
        if isinstance(s, int) or (hasattr(s, 'free_vars') and not s.free_vars()):
            v = int(s) if isinstance(s, int) else _constant(s)
            return '+' if v % 2 == 0 else '-'
        return '(-1)^({})'.format(s)

    def __str__(self) -> str:
        return self.sign_str() + self.letters


def _constant(p: Any) -> int:
    try:
        return int(sum(c for c, _ in p.terms)) % 2
    except TypeError:
        return 0


def phase_sum(items: Any) -> Any:
    """The sum mod 2 of signs that are 0/1 integers or Boolean polynomials.

    Returns an ``int`` when the result is constant, so plain codes keep plain signs.
    """
    total: Any = 0
    for s in items:
        if isinstance(s, (int, np.integer)):
            total = (total + int(s)) % 2 if isinstance(total, int) else total + int(s)
        else:
            total = s + total if isinstance(total, int) else total + s
    if hasattr(total, 'free_vars') and not total.free_vars():
        return _constant(total)
    if not isinstance(total, int) and hasattr(total, 'terms') and not total.terms:
        return 0
    return total
