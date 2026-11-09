"""Circuit distance of Clifford stages (zxqec.qprogram.distance), with Stim and codedistance."""
import os

import pytest

pytest.importorskip('stim')
pytest.importorskip('codedistance')

import stim  # noqa: E402

from zxqec.qprogram import (NotCliffordError, NotPhaseFreeError, ParityControlled, QProgram,  # noqa: E402
                            TRemovalError, circuit_distance, program_from_stim, without_gates,
                            z_distance_through_t)

from .programs import steane_t_injection  # noqa: E402

HERE = os.path.dirname(__file__)
CULTIVATION = os.path.join(HERE, '..', 'examples', 'circuits', 'cultivation')


def _fold_h(miswired, cnot_order='thesis'):
    import sys
    sys.path.insert(0, os.path.join(HERE, '..', 'examples'))
    import matplotlib
    matplotlib.use('Agg')
    import interactive_fold_h_3d as fth
    return fth.build_program(miswired=miswired, cnot_order=cnot_order)


def test_fold_transversal_h_rounds_have_circuit_distance_2_and_the_miswired_ones_1():
    # stages 1-6 go from the encoded [[9,1,3]] code to the encoded code; every code on the way has distance 3,
    # but two faults on check ancillas half way through their CNOTs (hook errors) make a logical error, and the
    # mis-wired check of the thesis's files lets a single fault through
    correct = circuit_distance(_fold_h(False), 1, 6)
    assert correct.distance == 2 and set(correct.by_logical) == {('X', 'Z'), ('Z', 'X')}   # the logical H
    assert circuit_distance(_fold_h(True), 1, 6).distance == 1


def test_n_order_in_two_check_stages_gives_the_fold_transversal_h_circuit_distance_3():
    # the thesis's weight-4 checks put their last two CNOTs on a horizontal pair, so a fault half way spreads along
    # code 2's logical X (stage 3, X checks) and code 1's logical Z (stage 5, Z checks); in N order the pair is
    # vertical. The noiseless program is the same, and each change fixes its own round
    from zxqec.qprogram import compare_programs
    thesis, n_order = _fold_h(False), _fold_h(False, 'N')
    assert compare_programs(thesis, n_order, 0).ok
    assert circuit_distance(n_order, 1, 6).distance == 3
    assert circuit_distance(n_order, 1, 3).distance == circuit_distance(n_order, 5, 6).distance == 3
    assert circuit_distance(thesis, 1, 3).distance == circuit_distance(thesis, 5, 6).distance == 2


def test_helpers_still_unused_at_the_cut_are_prepared_at_the_input_boundary():
    # helpers 14-17 start in |+> and are first used in stage 3; from stage 1 on, the input boundary measures X on
    # each of them (and Z on the unused qubit 18), else the X checks of stage 3 measure nothing and lose their
    # detectors (24 instead of 29)
    r = circuit_distance(_fold_h(False), 1, 6)
    lines = r.circuits[('X', 'Z')].splitlines()
    assert lines[:5] == ['MPP X14', 'MPP X15', 'MPP X16', 'MPP X17', 'MPP Z18']
    assert r.detectors[('X', 'Z')] == 29


def test_detectors_and_observable_are_deterministic_for_stim():
    # Stim refuses a circuit whose detectors or observable are random; the ones zxqec derives are accepted
    r = circuit_distance(_fold_h(False), 1, 6)
    for text in r.circuits.values():
        stim.Circuit(text).detector_error_model()


def test_non_clifford_and_non_phase_free_stages_are_refused():
    with pytest.raises(NotCliffordError):
        circuit_distance(steane_t_injection(correct=True))
    with pytest.raises(NotPhaseFreeError):
        circuit_distance(_fold_h(False), 1, 6, kind='Z')          # the transversal H mixes X and Z faults


def test_z_distance_of_cultivation_with_the_t_gates_removed():
    # Z faults commute with T, so the Z distance is that of the circuit without its T gates, measured against
    # the logical X of the code the real circuit ends in
    with open(os.path.join(CULTIVATION, 'd3_cultivation.stim')) as f:
        d3 = program_from_stim(f.read())[0]
    with open(os.path.join(CULTIVATION, 'd5_cultivation.stim')) as f:
        d5 = program_from_stim(f.read())[0]
    for program, expected in ((d3, 3), (d5, 4)):
        code = program.code_at(program.num_stages - 1).data()
        r = circuit_distance(without_gates(program), kind='Z', code_out=code)
        assert r.distance == expected and set(r.by_logical) == {('-', 'X')}
    with pytest.raises(NotCliffordError):
        circuit_distance(d3)                                      # with its T gates


def test_z_distance_through_t_checks_its_conditions_on_cultivation():
    # both cultivation circuits pass the two checks (same detectors with and without T; no X correction that a
    # Z fault can trigger) and give the Z distances above
    for name, expected in (('d3', 3), ('d5', 4)):
        with open(os.path.join(CULTIVATION, name + '_cultivation.stim')) as f:
            r = z_distance_through_t(program_from_stim(f.read())[0])
        assert r.kind == 'Z' and r.distance == expected and set(r.by_logical) == {('-', 'X')}


def _two_qubit_program(gates):
    # qubit 1 is the logical input, qubit 0 a helper in |+>
    from pyzx.circuit import Circuit
    p = QProgram(2, bits=1, initial_states={0: '+'})
    c = Circuit(2, bit_amount=1)
    for g in gates:
        c.add_gate(g)
    p.add_circuit(c, name='only stage')
    return p


def test_several_logical_qubits_each_get_their_observables():
    # a Steane block (input on qubit 3) next to a bare qubit 7: transversal X on the block has circuit distance 3,
    # but one gate on the bare qubit lets a single fault flip its logical
    from pyzx.circuit import Circuit
    from pyzx.circuit.gates import NOT

    from .programs import steane_encoder
    for bare_gate, expected in ((False, 3), (True, 1)):
        p = QProgram(8, initial_states={q: '0' for q in range(7) if q != 3})
        c = Circuit(8)
        for g in steane_encoder(list(range(7))):
            c.add_gate(g)
        p.add_circuit(c)
        c = Circuit(8)
        for q in range(8 if bare_gate else 7):
            c.add_gate(NOT(q))
        p.add_circuit(c)
        with pytest.raises(NotImplementedError, match='one logical qubit'):
            circuit_distance(p, 1, 1)                      # by default
        r = circuit_distance(p, 1, 1, several_logicals=True)
        assert r.distance == expected
        assert ('X', 'XX') in r.by_logical and ('Z', 'ZZ') in r.by_logical    # all inputs as X / Z, outputs per qubit


def test_z_distance_through_t_refuses_when_removing_t_is_not_exact():
    from pyzx.circuit.gates import HAD, T, Measurement
    # T T turns |+> into |+i>: its X measurement is random, without the T gates it is fixed (a detector more)
    with pytest.raises(TRemovalError, match='detectors'):
        z_distance_through_t(_two_qubit_program([T(0), T(0), HAD(0), Measurement(0, result_bit=0)]))
    # an X correction controlled by an X-basis result: a Z fault triggers it, and the X error meets a T
    with pytest.raises(TRemovalError, match='X correction'):
        z_distance_through_t(_two_qubit_program([HAD(0), Measurement(0, result_bit=0),
                                                 ParityControlled('X', 1, ['c[0]']), T(1)]))
    # the same with the correction written in OpenQASM, if(c==1) x q[1];
    from pyzx.circuit.gates import NOT, ConditionalGate
    with pytest.raises(TRemovalError, match='X correction'):
        z_distance_through_t(_two_qubit_program([HAD(0), Measurement(0, result_bit=0),
                                                 ConditionalGate('c', 1, NOT(1), 1), T(1)]))


_REPETITION_QASM = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[7];\ncreg a[1];\ncreg b[1];\ncreg c[1];\ncreg d[1];\n'


def _repetition_rounds(parity_controlled):
    # the bit-flip repetition code on qubits 0-2 (qubit 0 the input), two rounds of Z0 Z1 and Z1 Z2 checks with fresh
    # ancillas (3-4, then 5-6), each round followed by X on qubit 0 if its first check gave 1 and X on qubit 2 if its
    # second did: if(a==1) x q[0]; in OpenQASM, or the same correction as ParityControlled('X', 0, ['a[0]'])
    from pyzx.circuit import Circuit
    from pyzx.circuit.gates import ConditionalGate
    p = QProgram(7, bits=4, initial_states={q: '0' for q in range(1, 7)})
    p.add_circuit(Circuit.from_qasm(_REPETITION_QASM + 'cx q[0], q[1];\ncx q[0], q[2];\n'), name='encode')
    for s, t, a, b in ((3, 4, 'a', 'b'), (5, 6, 'c', 'd')):
        c = Circuit.from_qasm(_REPETITION_QASM + (
            'cx q[0], q[{s}];\ncx q[1], q[{s}];\ncx q[1], q[{t}];\ncx q[2], q[{t}];\n'
            'measure q[{s}] -> {a}[0];\nmeasure q[{t}] -> {b}[0];\n'
            'if({a}==1) x q[0];\nif({b}==1) x q[2];\n').format(s=s, t=t, a=a, b=b))
        if parity_controlled:
            gates = [ParityControlled('X', g.target, [g.condition_register + '[0]'])
                     if isinstance(g, ConditionalGate) else g for g in c.gates]
            c = Circuit(7, bit_amount=4)
            for g in gates:
                c.add_gate(g)
        p.add_circuit(c, name='round')
    return p


def test_an_openqasm_pauli_correction_is_fed_forward_as_a_parity_controlled_one():
    # if(a==1) x q[0]; becomes Stim's CX rec[-2] 0, as ParityControlled('X', 0, ['a[0]']) does: the same circuits
    # and distances (X faults: 3, the repetition code's; any fault: 1, a Z fault on a data qubit). Before, the
    # ConditionalGate was "not a Clifford gate this function supports"
    qasm, parity = _repetition_rounds(False), _repetition_rounds(True)
    assert type(qasm.stage_circuit(1).gates[-1]).__name__ == 'ConditionalGate'
    for kind, expected in (('full', 1), ('X', 3)):
        a, b = circuit_distance(qasm, 1, 2, kind), circuit_distance(parity, 1, 2, kind)
        assert a.distance == b.distance == expected and a.by_logical == b.by_logical
        assert a.circuits == b.circuits
    assert 'M 4\nCX rec[-2] 0\nCX rec[-1] 2\n' in a.circuits[('Z', 'Z')]


def test_one_fault_that_no_detector_sees_gives_z_distance_1():
    # a Z fault on a data qubit of the bit-flip repetition code flips its logical X and no detector, and so does every
    # other Z fault that flips anything: the detector error model has the one error L0. codedistance's MIPDist reads a
    # lightest fault set that takes every distinct error as no solution, and gave 0
    r = circuit_distance(_repetition_rounds(True), 1, 2, 'Z')
    dem = stim.Circuit(r.circuits[('X', 'X')]).detector_error_model()
    assert [str(e).split(' ')[1:] for e in dem.flattened() if e.type == 'error'] == [['L0']]
    assert r.distance == r.by_logical[('X', 'X')] == 1 and r.by_logical[('Z', 'Z')] is None
    faults = r.explain()
    assert len(faults) == 1 and faults[0].pauli in ('Z0', 'Z1', 'Z2')
    # MIPDist also gives 0 when the lightest set is two errors and there are no others, and when there is none
    from zxqec.qprogram.distance import _every_error_distance
    for errors, expected in ((['L0'], 1), (['D0 L0', 'D0'], 2), (['D0 L0'], None), (['D0 L0', 'D1'], None)):
        assert _every_error_distance(stim.DetectorErrorModel(
            '\n'.join('error(0.1) ' + e for e in errors))) == expected


def test_openqasm_corrections_stim_cannot_feed_forward_are_refused():
    # Stim feeds a Pauli forward from outcomes: if(c==1) z q[0]; (also as rz(-pi)) on a register of one bit is written
    # as ParityControlled('Z', 0, ['c[0]']) is; a rotation that is not a Pauli, a condition on an outcome being 0,
    # or on a register of several bits (a product of outcomes), is refused
    from pyzx.circuit import Circuit
    from pyzx.circuit.gates import HAD, Measurement

    def program(bits, correction):
        p = QProgram(2, bits=bits, initial_states={1: '0'})
        p.add_circuit(Circuit.from_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[{}];\n'
                                        'h q[1];\nmeasure q[1] -> c[0];\n{}'.format(bits, correction)), name='a')
        return p
    p = QProgram(2, bits=1, initial_states={1: '0'})
    c = Circuit(2, bit_amount=1)
    for g in (HAD(1), Measurement(1, result_bit=0), ParityControlled('Z', 0, ['c[0]'])):
        c.add_gate(g)
    p.add_circuit(c, name='a')
    expected = circuit_distance(p).circuits
    for correction in ('if(c==1) z q[0];\n', 'if(c==1) rz(-pi) q[0];\n'):
        assert circuit_distance(program(1, correction)).circuits == expected
    for bits, correction, reason in ((1, 'if(c==1) s q[0];\n', 'only Paulis'),
                                     (1, 'if(c==0) z q[0];\n', r'when c\[0\] is 0'),
                                     (2, 'if(c==1) z q[0];\n', 'product of the 2 outcomes of c'),
                                     (2, 'if(c==3) z q[0];\n', 'product of the 2 outcomes of c')):
        with pytest.raises(NotCliffordError, match=reason):
            circuit_distance(program(bits, correction))


def test_gidneys_published_d5_circuit_has_the_same_operations_and_fault_distance_4():
    # Gidney et al.'s own d = 5 file (S for T): the same CNOTs, resets, measurements and noise as SOFT's, and
    # Stim alone finds an undetectable logical error of weight 4 on it, as on SOFT's (graph-like search: 5);
    # 4 is the fault distance d = 5 cultivation is designed to have before its escape stage
    with open(os.path.join(CULTIVATION, 'd5_cultivation_gidney_s.stim')) as f:
        gidney = stim.Circuit(f.read())
    swap = {'T': 'S', 'T_DAG': 'S_DAG'}
    lines = []
    for name in ('d5_cultivation.stim', 'd5_final_check.stim'):
        with open(os.path.join(CULTIVATION, name)) as f:
            lines += f.read().splitlines()
    soft = stim.Circuit('\n'.join(swap[l.split(' ', 1)[0]] + ' ' + l.split(' ', 1)[1] if l.split(' ', 1)[0] in swap
                                  else l for l in lines))

    def operations(c, names):
        return [str(op) for op in c.flattened() if op.name in names
                and not any(t.is_measurement_record_target for t in op.targets_copy())]
    for names in ({'CX'}, {'R', 'RX', 'M', 'MX'}, {'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Z_ERROR'}):
        assert operations(gidney, names) == operations(soft, names)
    lightest = gidney.search_for_undetectable_logical_errors(dont_explore_detection_event_sets_with_size_above=5,
                                                             dont_explore_edges_with_degree_above=9999,
                                                             dont_explore_edges_increasing_symptom_degree=False)
    assert len(lightest) == 4 and len(gidney.shortest_graphlike_error()) == 5


def test_without_gates_keeps_the_rest_of_the_program():
    with open(os.path.join(CULTIVATION, 'd3_cultivation.stim')) as f:
        p = program_from_stim(f.read())[0]
    q = without_gates(p)
    assert q.num_stages == p.num_stages
    for s in range(p.num_stages):
        kept = [str(g) for g in p.stage_circuit(s).gates if type(g).__name__ != 'T']
        assert [str(g) for g in q.stage_circuit(s).gates] == kept


def test_explain_names_the_hook_error_of_the_fold_transversal_h():
    # the two faults of the lightest fault set sit after CNOTs from two X-check ancillas onto qubit 5, in the stage
    # of X checks after the transversal H
    program = _fold_h(False)
    faults = circuit_distance(program, 1, 6).explain()
    assert len(faults) == 2 and {f.stage for f in faults} == {3} and str(program.stage(3)) == 'X checks (code 2)'
    assert sorted(f.operation for f in faults) == ['after CX 15 5', 'after CX 16 5']
