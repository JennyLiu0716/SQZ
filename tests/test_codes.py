"""Questions 1-3: the code at a cut, the logical action of a block, and the level-up table."""
import itertools
import os
import tempfile
from fractions import Fraction

import numpy as np
import pytest
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, T
from pyzx.circuit.graphparser import circuit_to_graph
from pyzx.circuit.gates import Reset

from . import test_case_study_fold_h as F
from .programs import READOUT, steane_encoder, steane_t_injection
from zxqec.codes import PauliString, code_of
from zxqec.codes.phasepoly import Gadget, delta
from zxqec.gpf import same_map
from zxqec.qprogram import ParityControlled, QProgram

FOLD_GROUPS = [(0, 0, 'encode + round 1', 'encoder'), (1, 1, 'transversal H', 'logical gate'),
               (2, 3, 'round in code 2', 'syndrome extraction'), (4, 4, 'rotate', 'logical gate'),
               (5, 6, 'round in code 1', 'syndrome extraction'), (7, 7, 'checks + decode', 'decoder')]


def _steane_encoder_graph(prep_t=False):
    c = Circuit(7)
    if prep_t:
        c.add_gate(HAD(3))
        c.add_gate(T(3))
    for g in steane_encoder(list(range(7))):
        c.add_gate(g)
    return circuit_to_graph(c, initialize_qubits=[prep_t or q != 3 for q in range(7)])


# -- question 1: which code -------------------------------------------------------------

def test_steane_encoder_gives_the_steane_code():
    code = code_of(_steane_encoder_graph())
    assert code.parameters == (7, 1, 3)
    assert code.is_css
    assert sorted(p.weight for p in code.stabilizers) == [4] * 6
    assert code.logical_x[0].weight == code.logical_z[0].weight == 3
    for s in code.stabilizers:                      # the logical operators commute with the stabilizers
        assert s.commutes_with(code.logical_x[0]) and s.commutes_with(code.logical_z[0])
    assert not code.logical_x[0].commutes_with(code.logical_z[0])


def test_canonical_encoder_is_the_circuit_encoder():
    g = _steane_encoder_graph()
    assert same_map(code_of(g).encoder(), g)


def test_encoded_magic_state_is_the_steane_code_with_a_non_clifford_state():
    code = code_of(_steane_encoder_graph(prep_t=True))
    assert code.parameters == (7, 1, 3)
    assert code.non_clifford


def test_first_round_outcomes_are_the_signs_of_the_measured_checks():
    p = F.fold_transversal_h()
    code = p.code_at(0).data()
    assert code.parameters == (9, 1, 3)
    signs = {str(s.sign) for s in code.stabilizers if any(s.x)}
    assert signs == {'c[1]', 'c[2]', 'c[3]', 'c[4]'}     # one X check per outcome of round 1


def test_fold_transversal_h_returns_to_its_code_on_the_hardware():
    p = F.fold_transversal_h()
    start = p.code_at(0)
    assert not p.code_at(1).same_code_as(start, by='position')   # the dual code after H
    assert not p.code_at(3).same_code_as(start, by='position')
    assert p.code_at(4).same_code_as(start, by='position')       # rotated back
    assert not p.code_at(4).same_code_as(start, by='label')      # though every qubit moved


def test_pauli_string_roundtrip():
    p = PauliString.from_str('-XYZI')
    assert str(p) == '-XYZI' and p.weight == 3 and p.support == (0, 1, 2)


# -- question 2: the logical action -----------------------------------------------------------

def test_transversal_h_is_logical_h_and_the_rounds_are_identities():
    p = F.fold_transversal_h()
    assert p.logical_action(0, 1).operations() == {'H'}
    for s in (2, 3, 4, 5, 6):
        assert p.logical_action(0, s).operations() == {'I'}


def test_whole_fold_transversal_h_is_h_up_to_a_pauli_frame():
    act = F.fold_transversal_h().logical_action()
    assert act.is_deterministic() and act.operations() == {'H'}
    assert {b.correction for b in act.branches()} == {'', 'X', 'Y', 'Z'}


# -- feed-forward and question 3 ------------------------------------------------------------------

@pytest.mark.parametrize('gate, matrix', [
    ('Z', np.diag([1, -1])), ('X', np.array([[0, 1], [1, 0]])),
    ('S', np.diag([1, 1j])), ('S^dag', np.diag([1, -1j])), ('T', np.diag([1, np.exp(1j * np.pi / 4)]))])
def test_parity_controlled_gate_applies_the_gate_on_odd_parity(gate, matrix):
    c = Circuit(1)
    c.add_gate(ParityControlled(gate, 0, ['a', 'b']))
    g = circuit_to_graph(c)
    for a, b in itertools.product((0, 1), repeat=2):
        m = g.substitute_variables({'a': a, 'b': b}).to_matrix(preserve_scalar=True)
        want = matrix if (a ^ b) else np.eye(2)
        k = np.argmax(abs(want))
        assert np.allclose(m, m.flat[k] / want.flat[k] * want)
        assert np.isclose(abs(m.flat[k]), abs(want.flat[k]))


def test_injection_without_correction_is_t_or_t_dagger():
    act = steane_t_injection(correct=False).logical_action()
    assert act.operations() == {'T', 'T^dag'}
    assert not act.is_deterministic()


def test_injection_with_correction_is_t_in_every_run():
    act = steane_t_injection(correct=True).logical_action()
    assert act.operations() == {'T'} and act.is_deterministic()


def test_correction_is_s_on_odd_readout_parity():
    p = steane_t_injection()
    act = p.logical_action(0, p.num_stages - 1)
    by_value = {b.conditions[0][1]: b.operation for b in act.branches()}
    assert by_value == {0: 'I', 1: 'S'}
    assert set(act.branches()[0].conditions[0][0].replace(' ', '').split('+')) == set(READOUT)


def _has_bit(d):
    return any(v.startswith('z') for mono in d for v in mono)


def test_a_lone_t_gadget_breaks_an_x_string():
    # flipping the bit under a T gadget changes the phase by pi/4 - (pi/2) z: it depends on z
    assert _has_bit(delta([Gadget(Fraction(1, 4), ['z0'], [], 0)], {'z0'}))


def test_conditional_s_on_a_cat_state_keeps_xxxx():
    # the cat state |0000> + |1111> is one bit z0 copied to four qubits; a conditional S^dag on each
    # qubit is a pair of gadgets (3pi/4, and -3pi/4 with the outcome c on its hub). XXXX flips z0 under
    # all eight gadgets and the change of phase no longer depends on z0: XXXX stays a stabilizer.
    gadgets = [Gadget(Fraction(3, 4), ['z0'], [], 0) for _ in range(4)]
    gadgets += [Gadget(Fraction(5, 4), ['z0'], ['c'], 0) for _ in range(4)]
    assert not _has_bit(delta(gadgets, {'z0'}))


def test_conditional_s_on_independent_qubits_breaks_xxxx():
    # four independent bits: S^dag on each turns XXXX into YYYY, and ZZZZ is not a stabilizer here
    gadgets = [Gadget(Fraction(3, 4), ['z{}'.format(i)], [], 0) for i in range(4)]
    gadgets += [Gadget(Fraction(5, 4), ['z{}'.format(i)], ['c'], 0) for i in range(4)]
    assert _has_bit(delta(gadgets, {'z0', 'z1', 'z2', 'z3'}))


# -- the level-up table ------------------------------------------------------------------------------

def test_summary_of_fold_transversal_h():
    p = F.fold_transversal_h()
    s = p.summary(p.group_stages(FOLD_GROUPS))
    assert s.ok
    assert [b.operation() for b in s.blocks] == ['I', 'H', 'I', 'I', 'I', 'I']


def test_summary_flags_the_miswired_check_of_the_thesis_circuit():
    text = open(os.path.join(F.CIRCUITS, 'ft_h_5.qasm')).read()
    assert "cx q[4], q[14];" in text
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
    s = p.summary(p.group_stages(FOLD_GROUPS))
    bad = [b for b in s.blocks if not b.ok]
    assert [b.label for b in bad] == ['round in code 1']
    assert any('not a stabilizer' in msg for ok, msg in bad[0].checks if not ok)
    assert bad[0].action.operations() == {'I'}        # the logical state survives; only a check is wasted


def test_summary_of_steane_injection_at_two_levels():
    p = steane_t_injection()
    fine = p.group_stages([(0, 0, 'encode data', 'encoder'), (1, 2, 'prepare magic', 'state preparation'),
                           (3, 3, 'check magic', 'syndrome extraction'), (4, 4, 'CNOT', 'logical gate'),
                           (5, 5, 'read out', 'logical measurement'), (6, 6, 'correct', 'correction')])
    s = p.summary(fine)
    assert s.ok
    ops = [b.operation() for b in s.blocks]
    assert ops[0] == 'I' and 'T|+>' in ops[1] and ops[3] == 'CNOT(0->1)' and 'Z measurement' in ops[4]
    coarse = p.level_up(fine, 1, 5, name='inject T', role='injection')
    s2 = p.summary(coarse)
    assert s2.ok and [b.operation() for b in s2.blocks] == ['I', 'T']


# -- the edge of the CSS fragment ------------------------------------------------------------

def _steane_then(*gates):
    from pyzx.circuit.gates import S
    c = Circuit(7)
    for g in steane_encoder(list(range(7))):
        c.add_gate(g)
    for g in gates:
        c.add_gate(g)
    return circuit_to_graph(c, initialize_qubits=[q != 3 for q in range(7)])


def test_one_s_gate_gives_the_steane_code_conjugated_by_s():
    from pyzx.circuit.gates import S
    # one S gate makes the Steane code non-CSS: X on qubit 0 becomes Y. The code is still
    # [[7,1,3]], but no choice of Hadamards makes it CSS; code_of reads the mixed stabilizers.
    steane = code_of(_steane_then())
    code = code_of(_steane_then(S(0)))
    assert code.parameters == (7, 1, 3) and not code.is_css
    conjugated = []
    for p in steane.stabilizers:
        z = list(p.z)
        z[0] ^= p.x[0]                           # S X S^dag = Y, S Z S^dag = Z
        conjugated.append(PauliString.from_arrays(p.x, z, p.sign))
    assert all(code.contains(p) for p in conjugated)
    assert any('Y' in p.letters for p in code.stabilizers)


def test_transversal_s_keeps_the_steane_code_and_is_clifford():
    from pyzx.circuit.gates import S
    code = code_of(_steane_then(*[S(q) for q in range(7)]))
    assert code.parameters == (7, 1, 3)
    assert not code.non_clifford                    # S is Clifford: no non-Clifford logical content


def test_t_on_one_qubit_is_non_clifford_not_non_css():
    code = code_of(_steane_then(T(0)))              # T breaks stabilizers in a non-Clifford way
    assert code.non_clifford


def test_summary_reports_what_it_cannot_compute_instead_of_crashing():
    # the code after one S is read (a non-CSS [[7,1,3]]), but its ideal encoder, and so the
    # logical action of the block, is only built for CSS codes so far
    from pyzx.circuit.gates import S
    p = QProgram(7, initial_states={q: '0' for q in range(7) if q != 3})
    c = Circuit(7)
    for g in steane_encoder(list(range(7))):
        c.add_gate(g)
    p.add_circuit(c, name='encode')
    s1 = Circuit(7)
    s1.add_gate(S(0))
    p.add_circuit(s1, name='one S')
    s = p.summary(0)
    assert s.blocks[0].ok
    assert not s.blocks[1].ok
    assert s.blocks[1].code_out.parameters == (7, 1, 3) and not s.blocks[1].code_out.is_css
    assert any('logical action' in msg and 'CSS' in msg for ok, msg in s.blocks[1].checks if not ok)


# -- a block must map the code into the code: the logical action is computed up to a scalar ---------

def _reed_muller_15():
    """The [[15,1,3]] quantum Reed-Muller code: X-stabilizers are the rows of the 4 x 15 matrix whose
    column j is the binary expansion of j + 1, the logical X is all ones."""
    from pyzx.graph import Graph
    from pyzx.utils import VertexType
    rows = np.array([[((j + 1) >> b) & 1 for j in range(15)] for b in range(4)], dtype=np.uint8)
    g = Graph()
    xs, outs = [], []
    for j in range(15):
        x = g.add_vertex(VertexType.X, j, 2)
        o = g.add_vertex(VertexType.BOUNDARY, j, 3)
        g.add_edge((x, o))
        xs.append(x)
        outs.append(o)
    i = g.add_vertex(VertexType.BOUNDARY, 0, 0)
    zl = g.add_vertex(VertexType.Z, 0, 1)
    g.add_edge((i, zl))
    for x in xs:
        g.add_edge((zl, x))
    for r in rows:
        z = g.add_vertex(VertexType.Z, 1, 1)
        for j in np.flatnonzero(r):
            g.add_edge((z, xs[j]))
    g.set_inputs((i,))
    g.set_outputs(tuple(outs))
    return code_of(g)


def _t_on(code, qubits):
    from zxqec.codes.logical import set_logical_qubit
    c = Circuit(code.n)
    for q in qubits:
        c.add_gate(T(q))
    b = c.to_graph()
    for v, q in zip(b.inputs(), code.qubits):
        set_logical_qubit(b, v, q)
    for v, q in zip(b.outputs(), code.qubits):
        set_logical_qubit(b, v, q)
    return b


def test_transversal_t_on_the_15_qubit_code_is_logical_t_dagger():
    from zxqec.codes import logical_action
    code = _reed_muller_15()
    assert code.parameters == (15, 1, 3)
    act = logical_action(_t_on(code, range(15)), code, code)
    assert act.stays_in_code() is True
    assert act.operations() == {'T^dag'} and act.is_deterministic()


def test_t_on_part_of_the_code_leaves_it_instead_of_being_named_a_gate():
    # between the ideal encoders this block is 0.75 T^dag up to a phase: up to scalar it looks like
    # T^dag, but a run stays in the code with probability 9/16 only
    from zxqec.codes import logical_action
    code = _reed_muller_15()
    act = logical_action(_t_on(code, range(7)), code, code)
    assert act.stays_in_code() is False
    assert not act.is_deterministic()
    assert act.verdict().startswith("leaves the code")


def test_every_block_of_the_steane_injection_stays_in_its_code():
    # including the correction, whose conditional S changes the logical part of some stabilizers of
    # the check's Choi state between runs; only the stabilizers common to all runs are needed
    s = steane_t_injection().summary(0)
    assert all(b.action.stays_in_code() is True for b in s.blocks)


def test_a_code_that_differs_between_runs_can_be_read_as_its_common_part():
    from pyzx.circuit.gates import Measurement
    from zxqec.codes import OutcomeDependentCodeError
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
    g = p.prefix_graph(1)
    with pytest.raises(OutcomeDependentCodeError):
        code_of(g)
    common = code_of(g, common_to_all_runs=True)
    # the Z-type stabilizers do not change, and the X-types through qubit 0 do (X becomes Y there)
    steane = code_of(_steane_then())
    assert all(common.contains(z) for z in steane.stabilizers if not any(z.x))
    assert not any(common.contains(x) for x in steane.stabilizers if x.x[0])


def test_every_block_of_the_fold_transversal_h_stays_in_its_code():
    p = F.fold_transversal_h()
    s = p.summary(p.group_stages(FOLD_GROUPS))
    assert all(b.action.stays_in_code() is True for b in s.blocks)
