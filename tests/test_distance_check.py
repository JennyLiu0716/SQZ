"""zxqec.qprogram.check_distance: L. Yeh's distance-checking pipeline over the stages of a program."""
import os

import pytest
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement, T

from zxqec.gpf import boolean_variables, equal_for_all_assignments
from zxqec.qprogram import ParityControlled, QProgram, check_distance, program_from_stim, split_by_gate_type

HERE = os.path.dirname(__file__)
CULTIVATION = os.path.join(HERE, '..', 'examples', 'circuits', 'cultivation')


def _stage(*gates, qubits=3, bits=2):
    c = Circuit(qubits, bit_amount=bits)
    for g in gates:
        c.add_gate(g)
    return c


def _mixed_program():
    # a T after a CNOT (a gate), H then T on a qubit that starts in |0> (an initialisation), T then H and a
    # measurement with a CNOT on other qubits in between (a measurement), and a T after a fed-forward Z
    p = QProgram(3, bits=2, initial_states={2: '0'})
    p.add_circuit(_stage(HAD(0), CNOT(0, 1), T(1), CNOT(0, 1)), name='a')
    p.add_circuit(_stage(HAD(2), T(2, adjoint=True), CNOT(2, 0)), name='b')
    p.add_circuit(_stage(T(0), CNOT(1, 2), HAD(0), Measurement(0, result_bit=0)), name='c')
    p.add_circuit(_stage(ParityControlled('Z', 1, ['c[0]']), T(1)), name='d')
    return p


def test_splitting_by_gate_type_keeps_the_map():
    p = _mixed_program()
    split = split_by_gate_type(p)
    assert split.kinds == ('Clifford', 'non-Clifford gates', 'Clifford', 'non-Clifford initialisation', 'Clifford',
                           'non-Clifford measurement', 'Clifford', 'Clifford', 'non-Clifford gates')
    assert split.origin == (0, 0, 0, 1, 1, 2, 2, 3, 3)
    measured = split.program.stage(5).circuit.gates            # the whole measurement, gathered past the CNOT
    assert [type(g).__name__ for g in measured] == ['T', 'HAD', 'Measurement']
    a, b = p.quantum_graph(), split.program.quantum_graph()
    names = sorted(set(boolean_variables(a)) | set(boolean_variables(b)))
    assert equal_for_all_assignments(a, b, names, [])


def test_a_measurement_is_not_gathered_past_a_gate_that_reads_its_bit():
    p = QProgram(2, bits=2)
    p.add_circuit(_stage(T(0), ParityControlled('X', 1, ['c[1]']), HAD(0), Measurement(0, result_bit=1),
                         qubits=2), name='a')
    assert split_by_gate_type(p).kinds == ('non-Clifford gates', 'Clifford')


def test_stages_graph_is_the_block_of_those_stages():
    p = _mixed_program()
    level = p.group_stages([(1, 2, 'b and c')])
    index = next(k for k, blk in enumerate(p.levels.blocks(level)) if blk.name == 'b and c')
    a, b = p.stages_graph(1, 2, classical=False), p.block_graph(level, index, classical=False)
    names = sorted(set(boolean_variables(a)) | set(boolean_variables(b)))
    assert equal_for_all_assignments(a, b, names, [])
    with pytest.raises(ValueError):
        p.stages_graph(2, 1)


def test_a_group_is_extended_until_its_terms_map_paulis_alike():
    # T on qubit 0, then T on qubit 1, then both undone: the first group (T on 0 up to T on 1) turns X errors into
    # non-Paulis; extended to the stage that undoes them it is the identity on qubits 0 and 1
    p = QProgram(3)
    p.add_circuit(_stage(T(0), bits=0), name='T0')
    p.add_circuit(_stage(HAD(2), bits=0), name='H2')
    p.add_circuit(_stage(T(1), bits=0), name='T1')
    p.add_circuit(_stage(HAD(2), bits=0), name='H2 again')
    p.add_circuit(_stage(T(0, adjoint=True), T(1, adjoint=True), bits=0), name='undo')
    r = check_distance(p, distance=False)
    assert [(a.first, a.last, a.passes) for a in r.attempts] == [(0, 2, False), (0, 4, True)]
    assert r.passes is True and [(g.first, g.last) for g in r.groups] == [(0, 4)]
    assert r.groups[0].inputs == r.groups[0].outputs == 3


def test_cultivation_with_real_t_fails_the_check_and_with_s_has_nothing_to_check():
    with open(os.path.join(CULTIVATION, 'd3_cultivation.stim')) as f:
        text = f.read()
    r = check_distance(program_from_stim(text)[0], distance=False)
    assert r.split.kinds.count('non-Clifford gates') == 2 and r.split.kinds[2] == 'non-Clifford measurement'
    # the T^dag layer up to the T layer: "nothing" and T X T^dag on every code qubit; extended to the end, the same
    assert [(a.first, a.last, a.passes, a.all_runs) for a in r.attempts] == [(6, 9, False, True), (6, 10, False, True)]
    assert all(a.single_qubit is True for a in r.attempts)
    assert r.passes is False and r.not_counted == [2]
    assert 'groups pass: False' in r.describe()

    swap = {'T': 'S', 'T_DAG': 'S_DAG'}
    text_s = '\n'.join(swap[l.split(' ', 1)[0]] + ' ' + l.split(' ', 1)[1] if l.split(' ', 1)[0] in swap else l
                       for l in text.splitlines())
    r = check_distance(program_from_stim(text_s)[0], distance=False)
    assert set(r.split.kinds) == {'Clifford'} and r.attempts == [] and r.passes is True

    r = check_distance(program_from_stim(text)[0], distance=False, groups=[(6, 9)])
    assert [(a.first, a.last, a.passes) for a in r.attempts] == [(6, 9, False)]


def test_the_clifford_runs_of_cultivation_get_their_circuit_distance():
    pytest.importorskip('stim')
    pytest.importorskip('codedistance')
    with open(os.path.join(CULTIVATION, 'd3_cultivation.stim')) as f:
        r = check_distance(program_from_stim(f.read())[0])
    found = {(first, last): res for first, last, res in r.clifford}
    assert found[(3, 5)].distance == 3                      # the code between the injection and the T^dag layer
    assert 'k = 0' in found[(0, 1)]                         # before the injection there is no logical qubit
    assert r.clifford_distance == 3
