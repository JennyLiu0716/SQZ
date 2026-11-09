"""Corrections written in OpenQASM as conditional non-Pauli rotations (``if(c==1) s q[0];``), in the gadget form of ParityControlled."""
import itertools
from fractions import Fraction

import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import NOT, SX, ConditionalGate, S, T, XPhase, Z, ZPhase

from zxqec.gpf import equal_for_all_assignments, reduce_gpf, same_map
from zxqec.qprogram import ParityControlled, QProgram
from zxqec.qprogram.feedforward import _gadget_form

_QASM = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{}];\ncreg c[{}];\n'
_INJECT = 'h q[1]; t q[1];\ncx q[0], q[1];\nmeasure q[1] -> c[0];\n'


def _program(qasm, qubits=2, bits=1, gates=(), **kwargs):
    """``qasm`` (and then ``gates``) as one stage, with every qubit but 0 starting in |0>."""
    p = QProgram(qubits, bits=bits, initial_states={q: '0' for q in range(1, qubits)}, **kwargs)
    c = Circuit.from_qasm(_QASM.format(qubits, bits) + qasm)
    for gate in gates:
        c.add_gate(gate)
    p.add_circuit(c, name='stage')
    return p


def _runs(action):
    return [(b.outcomes, b.operation, b.correction) for b in action.branches()]


@pytest.mark.parametrize('qasm, name', [('s', 'S'), ('sdg', 'S^dag'), ('t', 'T'), ('tdg', 'T^dag')])
def test_t_injection_corrected_in_openqasm_as_with_parity_controlled(qasm, name):
    # before, code_of raised NotImplementedError ("a gadget with a symbolic angle") for the OpenQASM correction
    p = _program(_INJECT + 'if(c==1) {} q[0];\n'.format(qasm))
    ref = _program(_INJECT, gates=[ParityControlled(name, 0, ['c[0]'])])
    action, expected = p.logical_action(), ref.logical_action()
    assert action.verdict() == expected.verdict()
    assert _runs(action) == _runs(expected)
    assert equal_for_all_assignments(p.quantum_graph(), ref.quantum_graph(), preserve_scalar=True)
    assert str(p.code_at(0)) == str(ref.code_at(0))
    if qasm == 's':
        assert action.verdict() == 'deterministic: every run implements T'


def test_x_rotation_as_with_parity_controlled():
    p = _program('h q[1];\nmeasure q[1] -> c[0];\nif(c==1) sx q[0];\n')
    ref = _program('h q[1];\nmeasure q[1] -> c[0];\n', gates=[ParityControlled('SX', 0, ['c[0]'])])
    assert p.logical_action().verdict() == ref.logical_action().verdict()
    assert equal_for_all_assignments(p.quantum_graph(), ref.quantum_graph(), preserve_scalar=True)


def test_whole_register_condition_as_parities():
    # S when c == 3 is S^(c0 c1) = T^c0 T^c1 T^dag^(c0 + c1): three ParityControlled gadgets
    measure = 'h q[1]; h q[2];\nmeasure q[1] -> c[0];\nmeasure q[2] -> c[1];\n'
    p = _program(measure + 'if(c==3) s q[0];\n', qubits=3, bits=2)
    ref = _program(measure, qubits=3, bits=2, gates=[ParityControlled('T', 0, ['c[0]']), ParityControlled('T', 0, ['c[1]']),
                                                     ParityControlled('T^dag', 0, ['c[0]', 'c[1]'])])
    action, expected = p.logical_action(), ref.logical_action()
    assert action.verdict() == expected.verdict()
    assert _runs(action) == _runs(expected)
    assert equal_for_all_assignments(p.quantum_graph(), ref.quantum_graph(), preserve_scalar=True)
    assert {(b.outcomes['c[0]'], b.outcomes['c[1]']): b.operation for b in action.branches()} == {
        (0, 0): 'I', (1, 0): 'I', (0, 1): 'I', (1, 1): 'S'}


@pytest.mark.parametrize('value', range(4))
def test_whole_register_condition_on_a_t_injection(value):
    # the correction S when c == value on a two-bit register; c[1] reads an ancilla in |0>, so it is always 0.
    # The injection gives T when c[0] = 0 and T^dag when c[0] = 1. With c == 1 the correction is the right one;
    # with c == 0 it acts on the wrong run (S T = Z T^dag); with c == 2 or 3 it never acts.
    p = _program(_INJECT + 'measure q[2] -> c[1];\nif(c=={}) s q[0];\n'.format(value), qubits=3, bits=2)
    action = p.logical_action()
    assert [str(d) for d in action.detectors] == ['c[1]']
    never = 'not deterministic: the operation depends on the outcomes (T when c[0] = 0; T^dag when c[0] = 1)'
    assert action.verdict() == [
        'deterministic up to a Pauli frame: every run implements T^dag followed by a Pauli correction (Z, none) '
        'that the outcomes determine', 'deterministic: every run implements T', never, never][value]
    p.code_at(0)                                 # code_of reads the diagram
    assert not reduce_gpf(p.quantum_graph()).is_zero


_INNER = [S(0), S(0, adjoint=True), T(0), T(0, adjoint=True), SX(0), SX(0, adjoint=True), ZPhase(0, Fraction(3, 8)),
          XPhase(0, Fraction(5, 4))]


@pytest.mark.parametrize('size', [1, 2, 3])
def test_every_run_applies_the_rotation_exactly_when_the_register_equals_the_value(size):
    # checked run by run against the gate itself or nothing, not against another diagram of the condition
    for value in range(1 << size):
        for inner in _INNER:
            p = QProgram(1, bits=size)
            c = Circuit(1, bit_amount=size)
            c.add_gate(ConditionalGate('c', value, inner, size))
            p.add_circuit(c, name='correction')
            g = p.quantum_graph()
            applied, nothing = Circuit(1), Circuit(1)
            applied.add_gate(inner)
            for bits in itertools.product([0, 1], repeat=size):
                run = g.substitute_variables({'c[{}]'.format(i): b for i, b in enumerate(bits)})
                hit = sum(b << i for i, b in enumerate(bits)) == value
                assert same_map(run, (applied if hit else nothing).to_graph()), (value, str(inner), bits)


def test_paulis_and_other_gates_are_left_to_pyzx():
    for gate in (ConditionalGate('c', 3, Z(0), 2), ConditionalGate('c', 1, NOT(0), 1), ConditionalGate('c', 1, ZPhase(0, 2), 1),
                 S(0)):
        assert _gadget_form(gate) is gate


def test_the_correction_takes_the_time_of_a_conditional_gate():
    # the H after the correction starts when a ConditionalGate would end (around a Z, which stays PyZX's gate)
    for durations in ({'ConditionalGate': 3}, {ConditionalGate: 2}, {}):
        ticks = []
        for gate in ('s', 'z'):
            p = _program(_INJECT + 'if(c==1) {} q[0];\nh q[0];\n'.format(gate), gate_durations=durations)
            p.graph()
            ticks.append(p.stages[0].ticks)
        assert ticks[0] == ticks[1]
