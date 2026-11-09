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

A ``ConditionalGate`` around a rotation by an angle that is not a multiple of pi
(OpenQASM's ``if(c==1) s q[0];``) is written the same way when a
:class:`~zxqec.qprogram.QProgram` builds its diagram. PyZX writes it as one spider
with phase (condition) * angle, which is not affine in the outcomes, and
:func:`~zxqec.codes.code_of` cannot read it. The condition ``register == value``
is a product of the register's bits (``c[i]``, or ``1 + c[i]`` where the value
has a 0), and a product of ``k`` bits is a sum of parities,

    x_1 ... x_k = sum over non-empty S of (-1)^(|S|+1) / 2^(k-1) * (XOR of the x_i in S),

so the rotation becomes ``2^k - 1`` rotations, each controlled by a parity as
above: ``S`` when ``c == 3`` is ``T`` when ``c[0]``, ``T`` when ``c[1]`` and
``T^dag`` when ``c[0] + c[1]``. With a register of one bit it is one gadget
whose hub carries the condition.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations
from typing import Any

from pyzx.circuit.gates import ConditionalGate, Gate, XPhase, ZPhase
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
        _controlled_rotation(self, g, q_mapper, self.axis, self.angle, _parity(self.bits, 0, g))


class _ConditionalRotation(Gate):
    """PyZX's ``ConditionalGate`` around a Z- or X-rotation by a non-Pauli angle, as parity-controlled rotations.

    See the module docstring; :meth:`parities` lists the rotations. Only the
    diagram uses it (:func:`_gadget_form`): the stage keeps the ``ConditionalGate``.
    """
    name = 'ConditionalRotation'

    def __init__(self, gate: ConditionalGate) -> None:
        inner = gate.inner_gate
        self.gate = gate
        self.target = inner.target
        self.axis = 'Z' if isinstance(inner, ZPhase) else 'X'
        self.angle = Fraction(inner.phase) % 2
        if hasattr(gate, 'layer'):
            self.layer = gate.layer                     # see same_layer

    def __str__(self) -> str:
        return str(self.gate)

    def parities(self) -> list[tuple[Fraction, tuple[str, ...], int]]:
        """``(angle, outcomes, constant)`` per non-empty set of the register's bits: the angle applied when the parity is 1."""
        k = self.gate.register_size
        names = ['{}[{}]'.format(self.gate.condition_register, i) for i in range(k)]
        zeros = {i for i in range(k) if not self.gate.condition_value >> i & 1}     # literal 1 + c[i]
        out = []
        for size in range(1, k + 1):
            share = (self.angle * (-1) ** (size + 1) / 2 ** (k - 1)) % 2
            for bits in combinations(range(k), size):
                out.append((share, tuple(names[i] for i in bits), len(zeros.intersection(bits)) % 2))
        return out

    def to_graph(self, g: Any, q_mapper: Any, c_mapper: Any) -> None:
        for angle, names, const in self.parities():
            _controlled_rotation(self, g, q_mapper, self.axis, angle, _parity(names, const, g))


def _gadget_form(gate: Gate) -> Gate:
    """``gate`` as a :class:`_ConditionalRotation` if it is a ``ConditionalGate`` around a non-Pauli Z- or X-rotation.

    Any other gate is returned as it is: also a ``ConditionalGate`` around a
    Pauli (PyZX's phase is then the condition times pi, which ``reduce_gpf`` reads)
    or around a rotation by a symbolic angle.
    """
    if not isinstance(gate, ConditionalGate) or not isinstance(gate.inner_gate, (ZPhase, XPhase)):
        return gate
    phase = gate.inner_gate.phase
    if not isinstance(phase, (int, float, Fraction)) or Fraction(phase).denominator == 1:
        return gate
    return _ConditionalRotation(gate)


def _parity(names: Sequence[str], const: int, g: Any) -> Any:
    """The phase ``c_1 + ... + c_m + const`` (units of pi) of the outcomes ``names``."""
    parity: Any = None
    for name in names:
        v = new_var(name, True, g.var_registry)
        parity = v if parity is None else parity + v
    return parity + 1 if const else parity


def _controlled_rotation(gate: Gate, g: Any, q_mapper: Any, axis: str, angle: Fraction, parity: Any) -> None:
    """Write a rotation by ``angle`` (units of pi) about ``axis`` on ``gate.target``, applied when ``parity`` is 1."""
    wire = VertexType.Z if axis == 'Z' else VertexType.X
    other = VertexType.X if axis == 'Z' else VertexType.Z
    r = q_mapper.next_row(gate.target)
    if angle == 1:
        gate.graph_add_node(g, q_mapper, wire, gate.target, r, parity)
    else:
        half = angle / 2
        w = gate.graph_add_node(g, q_mapper, wire, gate.target, r, half)
        q = q_mapper.to_qubit(gate.target)
        hub = g.add_vertex(other, q + 0.35, r, parity)
        leaf = g.add_vertex(wire, q + 0.7, r, (-half) % 2)
        g.add_edge((w, hub))
        g.add_edge((hub, leaf))
    q_mapper.advance_next_row(gate.target)
