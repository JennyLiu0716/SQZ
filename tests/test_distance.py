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
