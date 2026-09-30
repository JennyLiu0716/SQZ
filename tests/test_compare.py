"""Comparing two programs block by block (zxqec.qprogram.compare_programs).

Two programs are logically equivalent when, block by block, the codes before and
after agree, the logical operations agree up to a Pauli frame, and the whole
programs have the same detectors. The cases below are variants of the two case
studies whose differences are known.
"""
import os
import tempfile

import pyzx as zx
from pyzx.circuit.gates import HAD, Reset

from . import test_case_study_fold_h as F
from .programs import steane_t_injection
from .test_codes import FOLD_GROUPS
from zxqec.qprogram import QProgram, compare_programs

STEANE = [(0, 0, 'encode data', 'encoder'), (1, 2, 'prepare magic', 'state preparation'),
          (3, 3, 'check magic', 'syndrome extraction'), (4, 4, 'transversal CNOT', 'logical gate'),
          (5, 5, 'read out magic', 'logical measurement'), (6, 6, 'correct', 'correction')]


def _miswired_fold_h():
    """The fold-transversal H with the thesis's original wiring error in ft_h_5.qasm."""
    text = open(os.path.join(F.CIRCUITS, 'ft_h_5.qasm')).read()
    path = os.path.join(tempfile.mkdtemp(), 'ft_h_5_miswired.qasm')
    with open(path, 'w') as f:
        f.write(text.replace("cx q[4], q[14];", "cx q[2], q[14];", 1))
    p = QProgram(F.N, bits=F.BITS, initial_states={**{q: '0' for q in range(1, 10)}, 18: '0',
                                                   **{a: '+' for a in F.X_ANC + F.Z_ANC}})
    p.add_circuit(F.load('ft_h'))
    p.add_circuit(F.load('ft_h_2'))
    p.add_circuit(F.with_prefix(F.load('ft_h_3'), [Reset(a) for a in F.X_ANC]))
    p.add_circuit(F.load('ft_h_3.2'))
    p.add_permutation([0, 3, 6, 9, 2, 5, 8, 1, 4, 7] + list(range(10, 19)))
    p.add_circuit(F.with_prefix(zx.Circuit.load(path), [Reset(a) for a in F.Z_ANC]), physical=True)
    p.add_circuit(F.with_prefix(F.load('ft_h_5.2'), [g for a in F.X_ANC for g in (Reset(a), HAD(a))]),
                  physical=True)
    p.add_circuit(F.with_prefix(F.load('ft_h_6'), [Reset(a) for a in F.Z_ANC]), physical=True)
    return p


def _steane_pair(**variant):
    a, b = steane_t_injection(), steane_t_injection(**variant)
    return compare_programs(a, b, a.group_stages(STEANE), b.group_stages(STEANE))


def test_the_miswired_check_is_found_by_its_lost_detectors():
    # every code and every logical action agree: the next round measures X2X3 again and restores
    # the code. What differs is the checking: two detectors fewer, both lost in the round with the
    # wrong check. Its result c[13] is then mentioned by no detector; X2X3 (c[18]) is random but
    # still compared with the final readout of qubits 2 and 3, so it still counts as checked
    a, b = F.fold_transversal_h(), _miswired_fold_h()
    c = compare_programs(a, b, a.group_stages(FOLD_GROUPS), b.group_stages(FOLD_GROUPS))
    assert not c.ok
    assert all(x.same_code_before and x.same_code_after and x.same_action for x in c.blocks)
    assert c.detectors == (24, 22) and c.same_detectors is False
    assert [x.index for x in c.blocks if x.checked[0] != x.checked[1]] == [4]
    assert c.blocks[4].checked == (8, 7)


def test_a_program_is_equivalent_to_itself_by_position():
    a, b = F.fold_transversal_h(), F.fold_transversal_h()
    c = compare_programs(a, b, a.group_stages(FOLD_GROUPS), b.group_stages(FOLD_GROUPS), by='position')
    assert c.ok and c.detectors == (24, 24) and c.same_detectors


def test_measuring_the_z_checks_first_is_equivalent():
    assert _steane_pair(z_checks_first=True).ok


def test_s_instead_of_s_dagger_is_equivalent_up_to_a_pauli_frame():
    # S = Z S^dag on each qubit: the correction differs by a logical Pauli, which is only tracked
    assert _steane_pair(correction='S').ok


def test_a_missing_correction_changes_the_logical_action():
    a, b = steane_t_injection(), steane_t_injection(correct=False)
    la = a.level_up(a.group_stages(STEANE), 1, 5, name='inject T', role='injection')
    lb = b.level_up(b.group_stages(STEANE[:5]), 1, 4, name='inject T', role='injection')
    c = compare_programs(a, b, la, lb)
    assert not c.ok
    inject = c.blocks[1]
    assert inject.same_code_before and inject.same_code_after and inject.same_action is False
    assert any('T / T^dag' in n for n in inject.notes)
    # the readout decides the correction (kind 'frame') and is checked by detectors all the same:
    # the same 13 results are checked in both
    assert inject.checked == (13, 13)


def test_a_check_wired_to_the_wrong_qubit_changes_the_code_after_it():
    c = _steane_pair(first_z_check=(0, 3, 4, 6))
    assert not c.ok
    first = c.first_difference()
    assert first.index == 2 and first.same_code_before and first.same_code_after is False
    assert any(n.startswith('code after:') for n in first.notes)


def _s_if_m():
    """Steane, then S on qubit 0 only when a random bit m is 1: the code after differs between runs."""
    from pyzx.circuit import Circuit
    from pyzx.circuit.gates import Measurement
    from .programs import steane_encoder
    from zxqec.qprogram import ParityControlled
    p = QProgram(8, bits=1, initial_states={**{q: '0' for q in range(7) if q != 3}, 7: '+'})
    c = Circuit(8, bit_amount=1)
    for gate in steane_encoder(list(range(7))):
        c.add_gate(gate)
    c.add_gate(Measurement(7, result_symbol='m'))
    p.add_circuit(c, name='encode, and a random bit')
    s = Circuit(8, bit_amount=1)
    s.add_gate(ParityControlled('S', 0, ['m']))
    p.add_circuit(s, name='S if m = 1')
    return p


def test_what_cannot_be_computed_is_not_decided_rather_than_different():
    # the code after 'S if m = 1' differs between runs, so it and the block's logical action are
    # unknown: the program compared with itself has nothing different, and must not be reported as
    # different
    a, b = _s_if_m(), _s_if_m()
    c = compare_programs(a, b, 0)
    assert c.blocks[0].ok
    assert c.blocks[1].same_code_before and c.blocks[1].same_code_after is None and c.blocks[1].same_action is None
    assert not c.ok and not c.differs and c.first_difference() is None
    assert c.verdict().startswith("not decided")


def test_programs_in_a_code_that_is_not_css_are_compared_in_full():
    # the [[5,1,3]] code: its logical action is computed, so the program equals itself
    from .test_noncss import _wheel_program
    a, b = _wheel_program(), _wheel_program()
    assert compare_programs(a, b, 0).ok
