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

"""Corrections controlled by the parity of measurement outcomes (feed-forward).

Error correction and gate teleportation both end with a classically controlled
correction: "apply ``X`` if these outcomes have odd parity", "apply ``S`` if the
logical readout was 1". PyZX's ``ConditionalGate`` covers ``if (register ==
value)``, which cannot say "odd parity of seven bits"; :class:`ParityControlled`
can.

In the diagram the parity ``p`` (a sum of outcome variables, in units of pi) is
written into phases, so the correction stays symbolic like the outcomes:

* ``X`` or ``Z``: one spider on the wire with phase ``p``;
* a rotation by ``a`` about ``Z`` (``S``, ``S^dag``, ``T``, ...): a spider
  ``Z(a/2)`` on the wire and a phase gadget with leaf ``-a/2`` whose hub carries
  ``p``. A hub phase of ``pi`` flips the sign of the gadget, so the two halves
  cancel when ``p = 0`` and add up to ``a`` when ``p = 1`` (up to a global
  phase ``e^{-i a p/2}``);
* rotations about ``X`` likewise with the colours swapped.

The phases stay affine in the outcomes, which is what :func:`~zxqec.gpf.reduce_gpf`
needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from pyzx.circuit.gates import Gate
from pyzx.symbolic import new_var
from pyzx.utils import VertexType

__all__ = ['ParityControlled']

#: gate name -> (axis, angle in units of pi)
_ROTATIONS: dict[str, tuple[str, Fraction]] = {
    'X': ('X', Fraction(1)), 'Z': ('Z', Fraction(1)),
    'S': ('Z', Fraction(1, 2)), 'S^dag': ('Z', Fraction(3, 2)),
    'T': ('Z', Fraction(1, 4)), 'T^dag': ('Z', Fraction(7, 4)),
    'SX': ('X', Fraction(1, 2)), 'SX^dag': ('X', Fraction(3, 2)),
}


class ParityControlled(Gate):
    """Apply ``gate`` to qubit ``target`` if the XOR of the outcome bits ``bits`` is 1.

    ``gate`` is one of ``'X'``, ``'Z'``, ``'S'``, ``'S^dag'``, ``'T'``,
    ``'T^dag'``, ``'SX'``, ``'SX^dag'``, or a pair ``('Z' | 'X', angle)`` with
    the angle in units of pi. ``bits`` are outcome names as they appear in the
    diagram, e.g. ``['c[6]', 'c[7]']`` (``c[k]`` is the outcome of
    ``Measurement(q, k)``).
    """
    name = 'ParityControlled'

    def __init__(self, gate: str | tuple[str, Any], target: int, bits: Sequence[str] | str) -> None:
        if isinstance(gate, tuple):
            axis, angle = gate[0], Fraction(gate[1]) % 2
            label = '{}({})'.format(axis, angle)
        elif gate in _ROTATIONS:
            axis, angle = _ROTATIONS[gate]
            label = gate
        else:
            raise ValueError("unknown correction {!r}; use one of {} or ('Z'|'X', angle)".format(
                gate, sorted(_ROTATIONS)))
        if axis not in ('X', 'Z'):
            raise ValueError("axis must be 'X' or 'Z'")
        self.gate = label
        self.axis = axis
        self.angle = angle
        self.target = target
        self.bits = (bits,) if isinstance(bits, str) else tuple(bits)
        if not self.bits:
            raise ValueError("a parity-controlled gate needs at least one outcome bit")

    def __str__(self) -> str:
        return "if({}) {}({})".format(" ^ ".join(self.bits), self.gate, self.target)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, ParityControlled) and self.target == other.target
                and self.axis == other.axis and self.angle == other.angle
                and self.bits == other.bits)

    def to_qasm(self) -> str:
        raise TypeError("a parity-controlled gate has no OpenQASM 2 form")

    def to_graph(self, g: Any, q_mapper: Any, c_mapper: Any) -> None:
        parity: Any = None
        for name in self.bits:
            v = new_var(name, True, g.var_registry)
            parity = v if parity is None else parity + v
        wire = VertexType.Z if self.axis == 'Z' else VertexType.X
        other = VertexType.X if self.axis == 'Z' else VertexType.Z
        r = q_mapper.next_row(self.target)
        if self.angle == 1:
            self.graph_add_node(g, q_mapper, wire, self.target, r, parity)
        else:
            half = self.angle / 2
            w = self.graph_add_node(g, q_mapper, wire, self.target, r, half)
            q = q_mapper.to_qubit(self.target)
            hub = g.add_vertex(other, q + 0.35, r, parity)
            leaf = g.add_vertex(wire, q + 0.7, r, (-half) % 2)
            g.add_edge((w, hub))
            g.add_edge((hub, leaf))
        q_mapper.advance_next_row(self.target)
