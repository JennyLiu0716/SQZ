"""zxqec.gpf.decompose_terms: a reduced diagram as a sum of terms over its hidden spiders (L. Yeh's decomposition)."""
import os

import pytest
import pyzx as zx

from zxqec.gpf import boolean_variables, consistent_assignments, decompose_terms, reduce_gpf
from zxqec.qprogram import program_from_stim

HERE = os.path.dirname(__file__)
CULTIVATION = os.path.join(HERE, '..', 'examples', 'circuits', 'cultivation')
GROUPS_D3 = [(0, 0, 'first checks', 'state preparation'), (1, 1, 'inject T', 'injection'),
             (2, 2, 'checks', 'syndrome extraction'), (3, 4, 'double-check', 'cultivation')]


def _reduced_block(p, groups, index):
    g = p.block_graph(p.group_stages(groups), index, classical=False)
    res = reduce_gpf(g)
    names = sorted(set(boolean_variables(g)) | {str(v) for c in res.constraints for v in c.free_vars()})
    return res, consistent_assignments(res.constraints, names)


def _double_check(gate):
    with open(os.path.join(CULTIVATION, 'd3_cultivation.stim')) as f:
        text = f.read()
    if gate == 'S':                                              # Stim's version: S and S_DAG for T and T_DAG
        swap = {'T': 'S', 'T_DAG': 'S_DAG'}
        text = '\n'.join(swap[l.split(' ', 1)[0]] + ' ' + l.split(' ', 1)[1] if l.split(' ', 1)[0] in swap else l
                         for l in text.splitlines())
    return _reduced_block(program_from_stim(text)[0], GROUPS_D3, 3)


def _matrix(g):
    return zx.tensor_to_matrix(zx.tensorfy(g, preserve_scalar=True), len(g.inputs()), len(g.outputs()))


def _names(term):
    return {p.inputs[0]: p.name for p in term.pieces if p.inputs and p.outputs}


@pytest.mark.parametrize('gate, flip, same', [('T', {'T X T^dag', 'T^dag X T'}, False), ('S', {'Y', 'X'}, True)])
def test_the_d3_double_check_links_qubits_as_reduced_and_is_single_qubit_with_fewest_links(gate, flip, same):
    # the double-check of d = 3 cultivation has one hidden Z-spider; as reduce_gpf labels it ('the value of output
    # 14'), both terms link the code qubits; relabelled as z + input 14 they are "nothing" and the same single-qubit
    # gate on every code qubit (Z on qubit 6 when the check's outcome is 1). With real T that gate is T X T^dag,
    # which turns X errors into Y errors, so the two terms treat errors differently; with S it is a Pauli
    res, runs = _double_check(gate)
    for run in runs:
        as_reduced = decompose_terms(res, run, 'as reduced')
        main = next(p for p in as_reduced.parts if p.outputs)
        assert main.hidden == ['Z-spider z1'] and len(main.nonzero) == 2
        assert all(not t.single_qubit for t in main.nonzero) and as_reduced.single_qubit is False
        assert as_reduced.same_error_action is None        # these terms are projections, not unitaries: no answer

        fewest = decompose_terms(res, run, 'fewest links')
        main = next(p for p in fewest.parts if p.outputs)
        assert main.hidden == ['Z-spider z1 (+ inputs 14)'] and main.outputs == (0, 3, 6, 8, 9, 11, 14)
        nothing, gate_term = (_names(t) for t in main.nonzero)
        assert set(nothing.values()) <= {'I', 'Z'} and set(gate_term.values()) <= flip
        assert fewest.single_qubit is True and fewest.same_error_action is same
        assert {p.errors['X'] for p in main.nonzero[1].pieces} == ({'X'} if same else {'Y'})
        # only the T version has a non-Clifford part, and its terms are single-qubit Cliffords: not enough
        assert main.clifford is (gate == 'S')
        assert fewest.non_clifford_single_qubit is (None if gate == 'S' else True)


def test_the_terms_of_each_part_add_up_to_the_part():
    # the exact terms (before their own reduction) are a decomposition of the diagram, in both labellings
    res, runs = _double_check('T')
    for labelling in ('as reduced', 'fewest links'):
        for part in decompose_terms(res, runs[0], labelling).parts:
            total = sum(_matrix(t.diagram) for t in part.terms)
            assert zx.compare_tensors(total, _matrix(part.diagram), preserve_scalar=False)


def test_clifford_blocks_read_as_their_gates():
    # transversal H of the fold-transversal Hadamard: H on every data qubit; the rotation: the identity on every
    # program qubit (a permutation stage only moves physical positions); the transversal CNOT of the Steane
    # injection: seven pieces, each linking one pair of qubits
    import sys
    sys.path.insert(0, os.path.join(HERE, '..', 'examples'))
    import matplotlib
    matplotlib.use('Agg')
    import interactive_fold_h_3d as fth
    from .programs import steane_t_injection
    from .test_codes import FOLD_GROUPS
    from .test_compare import STEANE

    fold = fth.build_program()
    res, runs = _reduced_block(fold, FOLD_GROUPS, 1)
    d = decompose_terms(res, runs[0])
    data = [p for p in d.parts if p.inputs and p.inputs[0] in range(1, 10)]
    assert len(data) == 9 and all(p.terms[0].pieces[0].name == 'H' for p in data)

    res, runs = _reduced_block(fold, FOLD_GROUPS, 3)
    pieces = [pc for p in decompose_terms(res, runs[0]).parts for pc in p.terms[0].pieces]
    assert len(pieces) == 10 and all(pc.name == 'I' and pc.inputs == pc.outputs for pc in pieces)

    res, runs = _reduced_block(steane_t_injection(), STEANE, 3)
    d = decompose_terms(res, runs[0])
    assert sorted(p.inputs for p in d.parts) == [(i, i + 7) for i in range(7)]
    assert d.single_qubit is False


def test_the_pauli_action_of_a_term_that_links_qubits():
    # a CNOT is one Clifford part with one term that links its two qubits: X on the control and Z on the target
    # spread to both qubits, as conjugation by the CNOT says; a T gate turns X into something that is not a Pauli
    from pyzx.circuit import Circuit
    from pyzx.circuit.gates import CNOT, T
    c = Circuit(2)
    c.add_gate(CNOT(0, 1))
    d = decompose_terms(c.to_graph())
    (part,) = d.parts
    assert part.clifford is True and part.single_qubit is False and d.non_clifford_single_qubit is None
    assert {k: v for k, v in part.terms[0].pauli_action().items() if k[1] != 'Y'} == {
        (0, 'X'): 'X0 X1', (0, 'Z'): 'Z0', (1, 'X'): 'X1', (1, 'Z'): 'Z0 Z1'}
    assert d.same_error_action is True
    c = Circuit(1)
    c.add_gate(T(0))
    d = decompose_terms(c.to_graph())
    assert d.parts[0].clifford is False and d.non_clifford_single_qubit is True
    assert d.parts[0].terms[0].pauli_action()[(0, 'X')] == 'not a Pauli' and d.same_error_action is False
    # split at its hub the T gadget is I + Z, which keep every Pauli up to a sign: why the check leaves hubs alone
    assert decompose_terms(c.to_graph(), split_gadgets=True).same_error_action is True


def test_a_term_that_stays_connected_but_is_a_product_is_split_by_its_matrix():
    # a phase gadget split at its hub gives I x I and Z x Z: the reduced term stays connected, its matrix decides
    import numpy as np
    from pyzx.circuit import Circuit
    from pyzx.circuit.gates import CNOT, T
    from zxqec.gpf.terms import _factors
    c = Circuit(2)
    for g in (CNOT(0, 1), T(1), CNOT(0, 1)):
        c.add_gate(g)
    d = decompose_terms(c.to_graph(), split_gadgets=True)
    assert [[pc.name for pc in t.pieces] for t in d.parts[0].nonzero] == [['I', 'I'], ['Z', 'Z']]
    assert d.single_qubit is True
    c = Circuit(2)                                       # a CNOT that is not undone still links the qubits
    for g in (CNOT(0, 1), T(1)):
        c.add_gate(g)
    assert decompose_terms(c.to_graph(), split_gadgets=True).single_qubit is False
    # the factors multiply back to the matrix, the linked rest kept as one piece, whatever the scale
    x = np.array([[0, 1], [1, 0]])
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    (a, b) = _factors(np.kron(x, cnot) * 1e-12, (0, 1, 2), (0, 1, 2))
    assert (a[0], a[1], b[0], b[1]) == ((0,), (0,), (1, 2), (1, 2))
    assert np.allclose(np.kron(a[2], b[2]), np.kron(x, cnot) * 1e-12)


def test_a_part_with_too_many_terms_is_reported_but_not_split():
    # the rounds of checks of d = 5 cultivation: one part with 18 hidden spiders (2^18 terms)
    from .test_cultivation import GROUPS_D5
    with open(os.path.join(CULTIVATION, 'd5_cultivation.stim')) as f:
        p = program_from_stim(f.read())[0]
    res, runs = _reduced_block(p, GROUPS_D5, 3)
    d = decompose_terms(res, runs[0])
    big = [part for part in d.parts if part.terms is None]
    assert [len(part.hidden) for part in big] == [18]
    assert d.complete is False and d.single_qubit is None and 'not split' in d.table()
    assert d.same_error_action is None

    # the d = 5 double-check, relabelled: "nothing" and T X T^dag or T^dag X T on the code qubits, so the terms
    # turn an X error into different Paulis. Each piece carries the term's (tiny) scalar; the check is scale-free
    res, runs = _reduced_block(p, GROUPS_D5, 4)
    d = decompose_terms(res, runs[0], 'fewest links')
    main = next(part for part in d.parts if part.same_error_action is False)
    names = [{pc.name for pc in t.pieces} for t in main.nonzero]
    assert names[0] == {'I'} and names[1] == {'T X T^dag', 'T^dag X T'}
    assert all(pc.errors is not None for t in main.nonzero for pc in t.pieces)
    assert d.same_error_action is False and d.non_clifford_single_qubit is True


def test_runs_and_labellings_are_checked():
    res, runs = _double_check('T')
    with pytest.raises(ValueError, match='labelling'):
        decompose_terms(res, runs[0], 'fewest')
    with pytest.raises(ValueError, match='give a value'):
        decompose_terms(res, {})
    bad = dict(runs[0])
    detector = next(c for c in res.constraints if c.free_vars())
    flip = next(iter(detector.free_vars())).name
    bad[flip] ^= 1
    with pytest.raises(ValueError, match='violates'):
        decompose_terms(res, bad)
