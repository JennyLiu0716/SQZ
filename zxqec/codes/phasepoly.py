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

"""Phase polynomials of a layer of phase gadgets, for deciding which X-strings survive it.

In GPF every Z-spider carries a bit ``z_i`` and a phase gadget with leaf angle
``a`` (units of pi) and hub phase ``s`` (a parity of outcomes) contributes the
phase ``a * (XOR of its bits  XOR  s)``. Flipping the bits in a set ``A`` changes
the total phase by

    Delta_A(z, c) = sum over gadgets that see an odd number of A's bits of  a * (1 - 2 y),

with ``y`` that gadget's XOR. Writing XOR as an integer polynomial,
``y1 ^ ... ^ yn = sum over non-empty T of (-2)^(|T|-1) prod_T y``, makes
``Delta_A`` a multilinear polynomial with rational coefficients modulo 2; terms
whose coefficient becomes even vanish, so for angles ``p / 2^e`` only monomials
of degree at most ``e`` survive. The X-string for ``A`` is a stabilizer exactly
when no surviving monomial contains a bit ``z_i``: then the gadgets change only
a sign, which may depend on the outcomes. This is how a pair of T-sized gadgets
that together form a conditional ``S`` is recognised as harmless for a code
whose weights make ``S`` transversal, while a lone ``T`` gadget is not.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Any

__all__ = ['Gadget', 'delta', 'affine_parts', 'to_pyzx']

Monomial = frozenset
Poly = dict  # Monomial -> Fraction (mod 2)


class Gadget:
    """A Z-phase gadget: leaf angle, the Z-spider bits it reads (any hashable names), and the outcome parity on its hub."""

    def __init__(self, angle: Fraction, bits: list[Any], sign_vars: list[str], sign_const: int) -> None:
        self.angle = Fraction(angle) % 2
        self.bits = list(bits)
        self.sign_vars = list(sign_vars)
        self.sign_const = sign_const % 2

    def __repr__(self) -> str:
        return "Gadget({}, {}, {}+{})".format(self.angle, self.bits, self.sign_vars, self.sign_const)


def _add(p: Poly, mono: frozenset, coeff: Fraction) -> None:
    c = (p.get(mono, Fraction(0)) + coeff) % 2
    if c:
        p[mono] = c
    else:
        p.pop(mono, None)


def _max_degree(angle: Fraction) -> int:
    """Largest |T| for which ``angle * 2^|T|`` is not a multiple of 2."""
    d = 0
    while (angle * 2 ** (d + 1)) % 2 != 0:
        d += 1
        if d > 16:
            raise ValueError("phase {} is not a dyadic multiple of pi".format(angle))
    return d


def delta(gadgets: list[Gadget], flipped: set[str]) -> Poly:
    """``Delta_A`` for the set ``A = flipped`` of Z-spider bits (see the module docstring)."""
    out: Poly = {}
    for g in gadgets:
        if len(set(g.bits) & flipped) % 2 == 0:
            continue
        variables = g.bits + g.sign_vars
        sgn = -1 if g.sign_const else 1          # a * (1 - 2y) with y -> 1 - y when the constant is 1
        _add(out, frozenset(), sgn * g.angle)
        for size in range(1, _max_degree(g.angle) + 1):
            coeff = sgn * g.angle * Fraction(-2) ** size
            if coeff % 2 == 0:
                continue
            for t in combinations(variables, size):
                if len(set(t)) == size:
                    _add(out, frozenset(t), coeff)
    return out


def affine_parts(p: Any) -> tuple[list[str], int]:
    """The variables and constant of an affine Boolean polynomial (a parity of outcomes)."""
    if isinstance(p, (int, Fraction)) or not hasattr(p, 'terms'):
        c = Fraction(p) % 2
        if c not in (0, 1):
            raise ValueError("hub phase {} is not a parity".format(p))
        return [], int(c)
    names: list[str] = []
    const = 0
    for coeff, term in p.terms:
        if Fraction(coeff) % 2 == 0:
            continue
        if Fraction(coeff) % 2 != 1:
            raise ValueError("hub phase {} is not a parity".format(p))
        if not term.vars:
            const ^= 1
        elif len(term.vars) == 1 and term.vars[0][1] == 1:
            names.append(term.vars[0][0].name)
        else:
            raise ValueError("hub phase {} is not affine".format(p))
    return names, const


def to_pyzx(p: Poly, registry: Any) -> Any:
    """A phase-polynomial dictionary as a PyZX phase (``int`` if constant)."""
    from pyzx.symbolic import new_var
    total: Any = 0
    for mono, coeff in p.items():
        if not mono:
            total = total + coeff
            continue
        term: Any = None
        for name in sorted(mono):
            v = new_var(name, True, registry)
            term = v if term is None else term * v
        total = term * coeff + total if not isinstance(total, int) or total else term * coeff
    if isinstance(total, Fraction) and total.denominator == 1:
        return int(total) % 2
    return total
