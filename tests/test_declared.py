"""Declared detectors checked against the derived ones (zxqec.qprogram.check_detectors).

A declared detector is ``deterministic`` when the derived detectors imply it,
``not deterministic`` when two allowed runs give it different values, and
``unknown`` otherwise. The runs reported as evidence are checked here against
the original program diagram, by tensor contraction, not against the reduction.
"""
import os

import numpy as np
import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, NOT, Measurement, Reset

from zxqec.qprogram import QProgram, check_detectors, program_from_stim

CULTIVATION = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'cultivation',
                           'd3_cultivation.stim')


def _repetition():
    """|+> in the 3-qubit repetition code; two rounds of the checks Z0Z1 (c0, c2) and Z1Z2 (c1, c3)
    with an X on qubit 0 between them; then the data read out in the Z basis (c4, c5, c6)."""
    p = QProgram(5, bits=7, initial_states={0: '+', 1: '0', 2: '0', 3: '0', 4: '0'})

    def checks(first):
        c = Circuit(5, bit_amount=7)
        for g in (CNOT(0, 3), CNOT(1, 3), Measurement(3, result_symbol='c[%d]' % first), Reset(3),
                  CNOT(1, 4), CNOT(2, 4), Measurement(4, result_symbol='c[%d]' % (first + 1)), Reset(4)):
            c.add_gate(g)
        return c
    c = Circuit(5, bit_amount=7)
    c.add_gate(CNOT(0, 1))
    c.add_gate(CNOT(0, 2))
    p.add_circuit(c, name='encode')
    p.add_circuit(checks(0), name='round 1')
    c = checks(2)
    c.gates.insert(0, NOT(0))
    p.add_circuit(c, name='X on qubit 0, round 2')
    c = Circuit(5, bit_amount=7)
    for q in range(3):
        c.add_gate(Measurement(q, result_symbol='c[%d]' % (4 + q)))
    p.add_circuit(c, name='read out')
    return p


REPETITION_DETECTORS = [['c[0]'], ['c[1]'], ['c[0]', 'c[2]'], ['c[1]', 'c[3]'],
                        ['c[2]', 'c[4]', 'c[5]'], ['c[3]', 'c[5]', 'c[6]']]


def _witness_is_real(program, det):
    """Both runs are allowed by the program itself (non-zero), and the parity differs between them."""
    g = program.quantum_graph()
    parities = []
    for run in det.witness:
        assert np.abs(g.substitute_variables(run).to_tensor(preserve_scalar=True)).max() > 1e-9
        parities.append(sum(run[n] for n in det.results) % 2)
    return parities[0] != parities[1]


def test_the_detectors_of_a_repetition_code_and_their_values():
    # the X on qubit 0 flips Z0Z1 between the rounds, so c0 + c2 is 1 in every run
    chk = check_detectors(_repetition(), REPETITION_DETECTORS)
    assert chk.ok and chk.derived == 6 and chk.undeclared == []
    assert [d.value for d in chk.declared] == [0, 0, 1, 0, 0, 0]


def test_a_random_result_is_not_deterministic_and_the_evidence_is_checked():
    p = _repetition()
    chk = check_detectors(p, [['c[4]'], [5, 6]])          # a data readout alone; and c5 + c6 by index
    lone, pair = chk.declared
    assert lone.status == 'not deterministic' and _witness_is_real(p, lone)
    assert pair.status == 'deterministic' and pair.results == ['c[5]', 'c[6]'] and pair.value == 0


def test_checks_the_file_does_not_declare_are_listed():
    chk = check_detectors(_repetition(), REPETITION_DETECTORS[:2])
    assert chk.ok and chk.derived == 6 and len(chk.undeclared) == 4


def test_a_name_that_is_not_a_result_is_refused():
    with pytest.raises(ValueError):
        check_detectors(_repetition(), [['c[9]']])


def test_all_26_declared_detectors_of_cultivation_are_deterministic():
    p, circ = program_from_stim(open(CULTIVATION).read())
    chk = check_detectors(p, circ.detectors)
    assert chk.ok and len(chk.declared) == 26 and chk.derived == 26 and chk.undeclared == []
    assert {d.value for d in chk.declared} == {0}


def test_without_the_gadget_layer_two_are_unknown_not_non_deterministic():
    # the two detectors that hold only because the T gates cancel: with the gadgets held fixed they
    # are not derived, and no run breaks them either, so the answer is 'unknown'
    p, circ = program_from_stim(open(CULTIVATION).read())
    chk = check_detectors(p, circ.detectors, gadgets=False)
    assert chk.count('deterministic') == 24 and chk.count('unknown') == 2 and chk.count('not deterministic') == 0
    assert all('c[24]' in d.results for d in chk.declared if d.status == 'unknown')


def test_a_cultivation_detector_with_a_result_left_out_is_not_deterministic():
    p, circ = program_from_stim(open(CULTIVATION).read())
    wrong = [list(d) for d in circ.detectors]
    wrong[9] = [x for x in wrong[9] if x != 'c[12]']
    chk = check_detectors(p, wrong)
    assert [d.index for d in chk.declared if d.status != 'deterministic'] == [9]
    assert chk.declared[9].status == 'not deterministic' and _witness_is_real(p, chk.declared[9])
