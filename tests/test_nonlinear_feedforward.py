"""Corrections conditioned on several outcomes at once ("Z if a = 1 and b = 0"), whose phases are products of outcomes."""
import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, NOT, Z, ConditionalGate, Measurement, Reset

from zxqec.gpf import boolean_variables, phases_modulo, reduce_gpf
from zxqec.qprogram import QProgram


def _rounds(error, rounds=2):
    """The repetition-code round of the thesis (X0X1, X1X2 on ancillas 3, 4) with the minimum-weight Z correction."""
    p = QProgram(5, bits=4, initial_states={3: '0', 4: '0'})
    stages = [('round 1', 'c', [HAD(3), HAD(4)])]
    if rounds == 2:
        stages += [('error', None, error), ('round 2', 'd', [Reset(3), Reset(4), HAD(3), HAD(4)])]
    for name, reg, gates in stages:
        if reg is not None:
            gates = gates + [CNOT(3, 0), CNOT(3, 1), CNOT(4, 1), CNOT(4, 2), HAD(3), HAD(4),
                             Measurement(3, result_symbol=reg + '[0]'), Measurement(4, result_symbol=reg + '[1]'),
                             ConditionalGate(reg, 1, Z(0), 2), ConditionalGate(reg, 3, Z(1), 2),
                             ConditionalGate(reg, 2, Z(2), 2)]
        c = Circuit(5, bit_amount=4)
        for g in gates:
            c.add_gate(g)
        p.add_circuit(c, name=name)
    return p


def test_branches_of_a_product_phase():
    # before the fix, the product c[0]*c[1] made every class of runs look impossible ("never succeeds")
    action = _rounds([], rounds=1).logical_action()
    assert len(action.branches()) == 4
    assert 'never succeeds' not in action.verdict()


_QASM = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[3];\n'


def _register_condition(after=''):
    """c[0], c[1] measured on |+>, then X on qubit 2 if c == 3 (the whole register, c[2] included) and c[2] measured."""
    p = QProgram(4, bits=3, initial_states={0: '+', 1: '+', 2: '0'})
    p.add_circuit(Circuit.from_qasm(_QASM + 'measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n'), name='m')
    p.add_circuit(Circuit.from_qasm(_QASM + 'if(c==3) x q[2];\nmeasure q[2] -> c[2];\n' + after), name='check')
    return p


def test_branches_with_a_product_detector():
    # before the fix, the linear solver refused the detector, the refusal was read as "no such run", and the
    # block "never succeeded", although it is the identity on the data qubit for c = 000, 010, 100
    action = _register_condition().logical_action(0, 1)
    assert [str(d) for d in action.detectors] == ['c[0]⋅c[1] + c[0]⋅c[1]⋅c[2] + c[2]']
    (branch,) = action.branches()
    assert (branch.operation, branch.correction, branch.outcomes) == ('I', '', {'c[0]': 0, 'c[1]': 0, 'c[2]': 0})
    assert action.verdict() == 'deterministic: every run implements I'


def test_branches_with_a_product_detector_and_a_product_phase():
    # a Z on the data qubit if c == 1 as well: its phase is a product too, and listing the runs raised ValueError
    action = _register_condition('if(c==1) z q[3];\n').logical_action(0, 1)
    assert [(b.outcomes, b.operation, b.correction) for b in action.branches()] == [
        ({'c[0]': 0, 'c[1]': 0, 'c[2]': 0}, 'I', ''), ({'c[0]': 1, 'c[1]': 0, 'c[2]': 0}, 'I', 'Z')]
    assert action.is_deterministic()


def test_corrected_and_logical_errors():
    for error, left, syndrome in (([], 'I', 'd[0], d[1]'), ([Z(0)], 'I', 'd[0] + 1, d[1]'),
                                  ([Z(0), Z(1)], 'Z', 'd[0], d[1] + 1'), ([NOT(1)], 'X', 'd[0], d[1]')):
        p = _rounds(error)
        level = p.group_stages([(0, 0, 'round 1'), (1, 2, 'error + round 2')])
        action = p.logical_action(level, 1)
        assert {b.correction or 'I' for b in action.branches()} == {left}
        assert action.is_deterministic()
        assert ', '.join(sorted(str(d) for d in reduce_gpf(p.quantum_graph()).detectors)) == syndrome
        assert p.code_at(0).describe().splitlines()[1].strip() == 'stabilizer +XXI'


def test_phases_modulo_fixes_outcomes_in_products():
    res = reduce_gpf(_rounds([Z(0)]).quantum_graph())
    raw = {str(v) for v in boolean_variables(res.graph)}
    assert {'d[0]', 'd[1]'} <= raw                              # round 2's outcomes are still in the phases
    shown = phases_modulo(res.graph, res.constraints)
    left = {str(v) for vtx in shown.vertices() if hasattr(shown.phase(vtx), 'free_vars')
            for v in shown.phase(vtx).free_vars()}
    assert left == {'c[0]', 'c[1]'}                             # the detectors fix d[0] = 1, d[1] = 0


_QASM_CD = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{}];\ncreg c[2];\ncreg d[1];\n'


def _copy_if_both(qubits=3, before='', after=''):
    """c[0], c[1] measured on |+>, X on qubit 2 if both are 1, and qubit 2 measured into d[0]: d[0] = c[0]*c[1]."""
    p = QProgram(qubits, bits=3, initial_states={0: '+', 1: '+', 2: '0'})
    qasm = _QASM_CD.format(qubits)
    if before:
        p.add_circuit(Circuit.from_qasm(qasm + before), name='before')
    p.add_circuit(Circuit.from_qasm(qasm + 'measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n'), name='m')
    p.add_circuit(Circuit.from_qasm(qasm + 'if(c==3) x q[2];\nmeasure q[2] -> d[0];\n'), name='check')
    if after:
        p.add_circuit(Circuit.from_qasm(qasm + after), name='after')
    return p


def test_a_product_detector_can_make_the_diagram_zero():
    # post-selecting the copy of c[0]*c[1] on 1 and a copy of c[0] on 0 asks for c[0]*c[1] = 1 and c[0] = 0, which no
    # run satisfies; the linear constraint c[0] alone has a solution, so before the fix the diagram was not zero and
    # code_of read a code off it
    import numpy as np
    from pyzx.circuit.gates import PostSelect
    from pyzx.symbolic import new_var
    from zxqec.codes import NotACodeError, code_of
    from zxqec.gpf import constraint_system, unsatisfiable
    c0, c1 = new_var('c[0]', True), new_var('c[1]', True)
    assert unsatisfiable([c0 * c1 + 1, c0]) and not unsatisfiable([c0 * c1 + 1, c0 + c1])
    assert len(constraint_system([c0 * c1 + 1, c0])[1]) == 1           # the matrix still holds the linear one only
    p = QProgram(5, bits=3, initial_states={0: '+', 1: '+', 2: '0', 3: '0', 4: '0'})
    p.add_circuit(Circuit.from_qasm(_QASM_CD.format(5) + 'cx q[0], q[3];\nmeasure q[0] -> c[0];\n'
                                    'measure q[1] -> c[1];\nif(c==3) x q[2];\n'), name='m')
    ps = Circuit(5, bit_amount=3)
    ps.add_gate(PostSelect(2, '1'))
    ps.add_gate(PostSelect(3, '0'))
    p.add_circuit(ps, name='post-select')
    g = p.quantum_graph()
    res = reduce_gpf(g)
    assert sorted(str(c) for c in res.constraints) == ['c[0]', 'c[0]⋅c[1] + 1']
    assert res.is_zero
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        assert np.abs(g.substitute_variables({'c[0]': a, 'c[1]': b}).to_tensor(preserve_scalar=True)).max() < 1e-12
    with pytest.raises(NotACodeError):
        code_of(g)


def test_an_error_that_contradicts_a_product_detector_is_caught():
    # the noiseless detector is c[0]*c[1] + d[0]; an X on qubit 2 before the check gives c[0]*c[1] + d[0] + 1. Their
    # sum is 1, so no run passes both; before the fix neither was solved for, and the pattern counted as harmless
    from zxqec.qprogram import pauli_errors
    e = pauli_errors(_copy_if_both(), 0, [2], letter='X')
    assert e.counts == {0: (1, 0, 1, 0), 1: (1, 1, 0, 0)}


def test_reduce_modulo_keeps_the_linear_constraints_next_to_a_product():
    from pyzx.symbolic import new_var
    from zxqec.gpf import reduce_modulo
    c0, c1, c2 = (new_var('c[{}]'.format(i), True) for i in range(3))
    assert str(reduce_modulo(c1, [c0 + c1, c0 * c2])) == 'c[0]'           # before the fix: c[1], c[0] + c[1] was lost
    assert str(reduce_modulo(c1, [c0 * c2])) == 'c[1]'


def test_distance_check_lists_and_samples_the_runs_of_a_product_detector():
    # T and T-dagger on qubit 3 around the copy of c[0]*c[1]: the detector d[0] = c[0]*c[1] allows 4 of the 8 runs.
    # A run that violates it is refused by decompose_terms, so every run listed or sampled is allowed.
    import random
    from zxqec.qprogram import check_distance
    from zxqec.qprogram.distance_check import _check_group
    p = _copy_if_both(4, 't q[3];\n', 'tdg q[3];\n')
    listed = _check_group(p, 0, 3, 'fewest links', 256, random.Random(0), 1024, 8)
    assert (listed.passes, listed.runs, listed.all_runs, listed.of_runs) == (True, 4, True, 4)
    sampled = _check_group(p, 0, 3, 'fewest links', 3, random.Random(0), 1024, 8)
    assert (sampled.passes, sampled.runs, sampled.all_runs, sampled.of_runs) == (True, 3, False, 4)
    # the whole pipeline gets there too: the ConditionalGate (X if c == 3) is a Clifford stage
    r = check_distance(p, distance=False)
    assert r.split.kinds == ('non-Clifford gates', 'Clifford', 'Clifford', 'non-Clifford gates')
    assert [(a.first, a.last, a.passes, a.runs, a.of_runs) for a in r.attempts] == [(0, 3, True, 4, 4)]
