"""error_formulas: a Pauli error after every gate as Boolean switches, pushed to the results and outputs, reduced once.

Every verdict is checked against the direct way (``classify_directly``): put the error gates into the program at
their places, reduce the whole program, and decide caught / harmless / wrong the same way.
"""
import dataclasses
import itertools
import os

import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, Measurement, Reset, T

from zxqec.qprogram import ParityControlled, QProgram, error_formulas, pauli_errors, program_from_stim
from zxqec.qprogram import faults as faults_module

PATH_D5 = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'cultivation', 'd5_cultivation.stim')


def _circuit(n, bits, gates):
    c = Circuit(n, bit_amount=bits)
    for g in gates:
        c.add_gate(g)
    return c


def _repetition():
    # three-qubit bit-flip code: data 0-2 (qubit 0 the input), helpers 3 and 4 measure Z0Z1 and Z1Z2, two rounds
    p = QProgram(5, bits=4, initial_states={1: '0', 2: '0', 3: '0', 4: '0'})
    p.add_circuit(_circuit(5, 4, [CNOT(0, 1), CNOT(0, 2)]), name='encode')
    for k, name in ((0, 'round 1'), (2, 'round 2')):
        p.add_circuit(_circuit(5, 4, [Reset(3), Reset(4), CNOT(0, 3), CNOT(1, 3), CNOT(1, 4), CNOT(2, 4),
                                      Measurement(3, k), Measurement(4, k + 1)]), name=name)
    return p


def _t_program():
    # one data qubit and a magic state: T on the helper, a CNOT, the helper measured, an S correction left out;
    # then the data checked against a second copy (CNOT onto a fresh helper, measured)
    p = QProgram(3, bits=2, initial_states={1: '+', 2: '0'})
    p.add_circuit(_circuit(3, 2, [T(1), CNOT(0, 1), Measurement(1, 0)]), name='inject')
    p.add_circuit(_circuit(3, 2, [HAD(0), CZ(0, 2), HAD(0), CNOT(0, 2), Measurement(2, 1)]), name='check')
    return p


def _t_check():
    # a T and a T^dag that cancel, checked: |+>, T, T^dag, H, measured (always 0 without errors); a second qubit
    # carries the output. An X between the two gates turns them into S, and the check becomes random.
    p = QProgram(2, bits=1, initial_states={0: '+'})
    p.add_circuit(_circuit(2, 1, [HAD(0), HAD(0), T(0), T(0).to_adjoint(), CNOT(1, 0), HAD(0), Measurement(0, 0)]),
                  name='T and T^dag')
    return p


def _reset_live():
    # a helper prepared, used, and reset again without being measured: the reset throws its old value away
    p = QProgram(2, bits=1, initial_states={1: '0'})
    p.add_circuit(_circuit(2, 1, [HAD(1), HAD(1), CNOT(1, 0)]), name='use')
    p.add_circuit(_circuit(2, 1, [Reset(1), CNOT(0, 1), Measurement(1, 0)]), name='reset, then read')
    return p


def test_the_places_and_what_reaches_the_results():
    p = _repetition()
    ef = error_formulas(p)
    # 2 encoder CNOTs, and per round 2 resets and 4 CNOTs, each with a place on every qubit it acts on
    assert len(ef.places) == 2 * 2 + 2 * (2 + 4 * 2)
    assert [pl.name for pl in ef.places[:4]] == ['CNOT'] * 4
    # an X on data qubit 1 between the rounds (after the last CNOT of round 1 on it) flips both checks of round 2
    k = max(pl.index for pl in ef.places if pl.stage == 1 and pl.qubit == 1)
    assert ef.signature({k: 'X'})[0] == frozenset({'c[2]', 'c[3]'})
    assert ef.signature({k: 'Z'}) == (frozenset(), frozenset(), frozenset({'oz1'}))
    # the detectors c[0] + c[2] and c[1] + c[3]: an X on qubit 1 between the rounds is in both
    on = ef.switches({k: 'X'})
    fired = [d for d, s in ef.detector_switches() if len(s & on) % 2]
    assert len(fired) == 2


def test_a_bit_flip_code_does_not_see_a_phase_flip():
    p = _repetition()
    ef = error_formulas(p)
    rows = ef.single_errors()
    assert {v for _p, letter, v, _d in rows if letter == 'X'} <= {'caught', 'harmless'}
    assert any(v == 'wrong' for _p, letter, v, _d in rows if letter == 'Z')
    weight, wrong = ef.lowest_wrong(max_weight=2, per='place')
    assert weight == 1 and all(set(e.values()) <= {'Z', 'Y'} for e in wrong)
    # a fault after a CNOT can hit both of its qubits: still one fault
    weight, wrong = ef.lowest_wrong(max_weight=1)
    assert weight == 1 and any(len(e) == 2 for e in wrong)


@pytest.mark.parametrize('build, final_check', [(_repetition, True), (_repetition, False), (_t_program, False),
                                                (_t_check, False), (_reset_live, False)],
                         ids=['repetition, final check', 'repetition', 'T injection', 'T and T^dag checked',
                              'a reset of a qubit that was not measured'])
def test_every_single_error_and_pair_agrees_with_the_direct_way(build, final_check):
    # (the T program measures its data qubit at the end, so its output is in no code to check)
    p = build()
    ef = error_formulas(p, final_check=final_check)
    singles = [(k, letter) for k in range(len(ef.places)) for letter in 'XYZ']
    for k, letter in singles:
        assert ef.classify({k: letter}) == ef.classify_directly({k: letter}), (ef.places[k], letter)
    step = 1 if len(singles) < 40 else 11                # every pair for the small programs
    for (k1, l1), (k2, l2) in itertools.islice(itertools.combinations(singles, 2), 0, None, step):
        if k1 != k2:
            errors = {k1: l1, k2: l2}
            assert ef.classify(errors) == ef.classify_directly(errors), errors


def test_a_t_gate_reached_by_an_x_is_flipped():
    p = _t_program()
    ef = error_formulas(p, final_check=False)
    assert len(ef.t_flips) == 0                        # nothing comes before the T: no error can reach it
    q = QProgram(2, bits=1, initial_states={1: '+'})
    q.add_circuit(_circuit(2, 1, [HAD(1), HAD(1), T(1), CNOT(0, 1), Measurement(1, 0)]), name='inject')
    ef2 = error_formulas(q, final_check=False)
    (flip,) = ef2.t_flips
    # the Z part of the error after the first H (the second H makes it an X) and the X part of the one after it
    assert flip.qubit == 1 and not flip.adjoint and flip.switches == {ef2.places[0].z, ef2.places[1].x}
    k = ef2.places[1].index                            # after the second H on qubit 1
    assert ef2.signature({k: 'X'})[1] == frozenset({flip.name})
    assert ef2.signature({k: 'Z'})[1] == frozenset()


def test_feed_forward_corrections_misfire_on_a_flipped_result():
    p = QProgram(2, bits=1, initial_states={1: '0'})
    p.add_circuit(_circuit(2, 1, [CNOT(0, 1), Measurement(1, 0), ParityControlled('X', 0, ['c[0]'])]), name='m')
    ef = error_formulas(p, final_check=False)
    k = ef.places[1].index                             # after the CNOT, on the helper
    assert ef.places[1].qubit == 1
    # an X on the helper flips c[0], and the correction then puts an X on qubit 0
    assert ef.signature({k: 'X'})[0] == frozenset({'c[0]'})
    assert 'ox0' in ef.signature({k: 'X'})[2]


def test_sampling():
    ef = error_formulas(_repetition())
    counts = ef.sample(0.0, 50, seed=1)
    assert counts == {'caught': 0, 'harmless': 50, 'wrong': 0}
    counts = ef.sample(0.2, 400, seed=1)
    assert sum(counts.values()) == 400 and counts['wrong'] > 0 and counts['caught'] > 0
    only_resets = ef.sample({'Reset': 0.3}, 200, seed=2)
    assert sum(only_resets.values()) == 200
    assert sum(ef.sample(0.2, 100, seed=1, per='place').values()) == 100
    assert ef.sample(lambda place: 0.0, 10, seed=3)['harmless'] == 10


def test_unsupported_gates_are_refused():
    from pyzx.circuit.gates import Tofolli
    p = QProgram(3)
    p.add_circuit(_circuit(3, 0, [Tofolli(0, 1, 2)]), name='toffoli')
    with pytest.raises(NotImplementedError):
        error_formulas(p, final_check=False)


def test_errors_that_reach_t_gates_are_decided_by_the_reduced_diagram():
    ef = error_formulas(_t_check(), final_check=False)
    assert [t.adjoint for t in ef.t_flips] == [False, True]
    assert len(ef.clean) == 1                          # the check always reads 0
    between = next(pl.index for pl in ef.places if pl.name == 'T' and pl.qubit == 0)   # after the T, before T^dag
    assert ef.signature({between: 'X'})[1] == frozenset({ef.t_flips[1].name})
    assert ef.classify({between: 'Z'}) == 'caught'     # a Z there flips the check
    before = ef.places[1].index                        # after the second H: an X reaches both gates
    assert ef.signature({before: 'X'})[1] == frozenset(t.name for t in ef.t_flips)


def test_faults_per_gate_and_per_place():
    ef = error_formulas(_repetition())
    per_gate = ef.single_faults()
    assert len(per_gate) == 2 + 2 * (2 + 4)                      # one per gate
    assert sorted({len(kinds) for _p, kinds in per_gate}) == [3, 15]
    assert len(ef.single_faults('place')) == len(ef.places)
    with pytest.raises(ValueError):
        ef.single_faults('qubit')


def test_formulas_as_text():
    p = _repetition()
    ef = error_formulas(p)
    k = max(pl.index for pl in ef.places if pl.stage == 1 and pl.qubit == 1)
    assert 'x{}'.format(k) in ef.formula('c[2]').split(' + ')
    assert ef.formula(set()) == '0'
    assert ef.formula('oz1') == ef.formula(ef.frame[1][1])
    with pytest.raises(KeyError):
        ef.formula('c[9]')


def test_results_keep_their_register_names():
    # OpenQASM with two registers, as in the paper's running example: round 1 writes c, round 2 writes d
    header = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\ncreg c[2];\ncreg d[2];\n'
    round_1 = ('h q[3]; h q[4]; cx q[3], q[0]; cx q[3], q[1]; cx q[4], q[1]; cx q[4], q[2]; h q[3]; h q[4];'
               'measure q[3] -> c[0]; measure q[4] -> c[1];')
    round_2 = 'reset q[3]; reset q[4];' + round_1.replace('c[', 'd[')
    p = QProgram(5, bits=4, initial_states={3: '0', 4: '0'})
    p.add_circuit(Circuit.from_qasm(header + round_1), name='round 1')
    p.add_circuit(Circuit.from_qasm(header + round_2), name='round 2')
    ef = error_formulas(p)
    assert sorted(ef.flips) == ['c[0]', 'c[1]', 'd[0]', 'd[1]']
    k = max(pl.index for pl in ef.places if pl.stage == 0 and pl.qubit == 1)   # data qubit 1 between the rounds
    assert ef.signature({k: 'Z'})[0] == frozenset({'d[0]', 'd[1]'}) and ef.classify({k: 'Z'}) == 'caught'
    assert ef.classify({k: 'X'}) == 'wrong'                      # the phase-flip code does not see X


def test_a_result_written_twice_is_refused():
    p = QProgram(1, bits=1)
    p.add_circuit(_circuit(1, 1, [Measurement(0, 0), Measurement(0, 0)]), name='twice')
    with pytest.raises(NotImplementedError):
        error_formulas(p, final_check=False)


def test_an_output_without_a_logical_qubit_needs_code_out():
    def build(start):
        p = QProgram(5, bits=4, initial_states=start)
        p.add_circuit(_circuit(5, 4, [CNOT(0, 1), CNOT(0, 2)]), name='encode')
        p.add_circuit(_circuit(5, 4, [CNOT(0, 3), CNOT(1, 3), CNOT(1, 4), CNOT(2, 4), Measurement(3, 0),
                                      Measurement(4, 1)]), name='round')
        return p
    helpers = {1: '0', 2: '0', 3: '0', 4: '0'}
    fixed = build({0: '0', **helpers})                            # |000>: a stabilizer state, k = 0
    assert fixed.code_at(1).k == 0
    with pytest.raises(ValueError):
        error_formulas(fixed)
    code = build(helpers).code_at(1)                              # the bit-flip code, with its logical qubit
    ef = error_formulas(fixed, code_out=code)
    x0 = ef.places[0].index                                       # after the first CNOT, on qubit 0
    assert ef.classify({x0: 'Z'}) == 'harmless' and ef.classify({x0: 'X'}) == 'caught'
    # X on the three data qubits after their last CNOT: |111>, the logical |1> instead of |0>, which no check sees
    last = {q: max(pl.index for pl in ef.places if pl.qubit == q) for q in (0, 1, 2)}
    assert ef.classify({k: 'X' for k in last.values()}) == 'wrong'


def test_a_pattern_in_words():
    ef = error_formulas(_repetition())
    cnot = [pl.index for pl in ef.places[:2]]                     # the first CNOT, on qubits 0 and 1
    assert ef.describe({cnot[0]: 'X', cnot[1]: 'Y', ef.places[-1].index: 'Z'}) == (
        'X on qubit 0, Y on qubit 1 after the CNOT (stage 0, gate 0); Z on qubit {} after the {} (stage 2, gate {})'
        .format(ef.places[-1].qubit, ef.places[-1].name, ef.places[-1].gate))
    assert ef.describe({}) == 'no errors'


def test_errors_in_some_stages_only():
    p = _repetition()
    ef = error_formulas(p, stages=[2])                            # the encoder and round 1 are perfect
    assert {pl.stage for pl in ef.places} == {2} and len(ef.places) == 2 + 4 * 2
    full = error_formulas(p)
    k = next(pl for pl in ef.places if pl.name == 'CNOT' and pl.qubit == 1)
    same = next(pl for pl in full.places if (pl.stage, pl.gate, pl.qubit) == (k.stage, k.gate, k.qubit))
    for letter in 'XYZ':
        assert ef.signature({k.index: letter}) == full.signature({same.index: letter})
        assert ef.classify({k.index: letter}) == ef.classify_directly({k.index: letter})
    with pytest.raises(ValueError):
        error_formulas(p, stages=[5])


def test_an_error_that_a_reset_throws_away_is_harmless():
    ef = error_formulas(_reset_live(), final_check=False)
    before = next(pl.index for pl in ef.places if pl.stage == 0 and pl.name == 'CNOT' and pl.qubit == 1)
    # an X on the helper after the CNOT: the reset throws it away; the value it throws away is now 1, which nobody
    # reads, so both ways must sum over it instead of taking it as a check
    assert ef.classify({before: 'X'}) == ef.classify_directly({before: 'X'}) == 'harmless'


def test_pauli_errors_sums_over_what_a_reset_throws_away():
    # pauli_errors adds a stage for the errors, so the stages after it are numbered one higher: the reset variable
    # of the first reset below (s1_r0 without errors) is s2_r0 with them, the name the second reset has without
    # errors. The noiseless constraint s2_r0 = 0 is about the second reset, and must not be used as a check on
    # the first: the X is thrown away by the reset, so the pattern is harmless (it was counted as caught)
    from zxqec.qprogram import pauli_errors
    p = QProgram(2, initial_states={1: '0'})
    p.add_circuit(_circuit(2, 0, [HAD(1), HAD(1), CNOT(1, 0)]), name='use')
    p.add_circuit(_circuit(2, 0, [Reset(1)]), name='reset')
    p.add_circuit(_circuit(2, 0, [Reset(1)]), name='reset again')
    assert pauli_errors(p, 0, [1], 'X').counts[1] == (1, 0, 1, 0)      # (patterns, caught, harmless, wrong)


@pytest.mark.parametrize('build, final_check', [(_repetition, True), (_t_check, False), (_t_program, False)],
                         ids=['repetition', 'T and T^dag checked', 'T injection'])
def test_the_switched_program_is_the_pushed_diagram(build, final_check):
    # for random values of the switches (and every value of the results), the program with the switches in it is
    # the diagram with the switches pushed to the end; a T flip variable keeps its name, set to its formula
    import random

    from zxqec.gpf import boolean_variables, equal_for_all_assignments
    ef = error_formulas(build(), final_check=final_check)
    switched, pushed = ef.switched_program().quantum_graph(), ef.pushed_graph()
    names = [pl.x for pl in ef.places] + [pl.z for pl in ef.places]
    rng = random.Random(7)
    for _ in range(12):
        values = {v: rng.randint(0, 1) for v in names}
        flips = {t.name: sum(values[v] for v in t.switches) % 2 for t in ef.t_flips}
        assert equal_for_all_assignments(switched.substitute_variables(values),
                                         pushed.substitute_variables({**values, **flips}))
    kept = {v for v in boolean_variables(ef.pushed_graph(frame=False)) if v[:2] in ('ox', 'oz')}
    assert kept == {'{}{}'.format(o, q) for q in ef.frame for o in ('ox', 'oz')}     # frame=False keeps their names


def test_a_pattern_put_into_the_program():
    ef = error_formulas(_repetition())
    k = ef.places[0].index
    gates = [g.name for st in ef.program_with({k: 'Y'}).stages for g in st.circuit.gates]
    assert gates.count('Y') == 1 and len(gates) == 1 + sum(len(st.circuit.gates) for st in ef.program.stages)


def test_the_probability_of_patterns_of_faults():
    ef = error_formulas(_repetition())
    faults = ef.single_faults()
    p = 0.01
    none = (1 - p) ** len(faults)
    assert ef.probability([{}], p) == pytest.approx(none)
    cnot = next(kinds for _place, kinds in faults if len(kinds) == 15)
    one = next(kinds for _place, kinds in faults if len(kinds) == 3)
    assert ef.probability([cnot[0]], p) == pytest.approx(none / (1 - p) * p / 15)
    assert ef.probability([{**cnot[0], **one[1]}], p) == pytest.approx(none / (1 - p) ** 2 * (p / 15) * (p / 3))
    assert ef.probability([cnot[0], dict(cnot[0])], p) == ef.probability([cnot[0]], p)     # counted once
    # the probabilities of all patterns of at most one fault and of no fault add up to P(at most one fault)
    singles = [e for _place, kinds in faults for e in kinds]
    at_most_one = none * (1 + len(faults) * p / (1 - p))
    assert ef.probability([{}] + singles, p) == pytest.approx(at_most_one)
    # a gate-dependent p, as in sample
    assert ef.probability([{}], {'CNOT': p}) == pytest.approx((1 - p) ** sum(len(k) == 15 for _pl, k in faults))
    with pytest.raises(ValueError):
        ef.probability([{999: 'X'}], p)                   # no such place


def test_outcomes_are_shares_of_runs():
    # Clifford errors: all or nothing, as classify says
    ef = error_formulas(_repetition())
    for pl in ef.places:
        for letter in 'XYZ':
            shares = ef.outcomes({pl.index: letter})
            assert shares[ef.classify({pl.index: letter})] == pytest.approx(1.0)
    assert ef.outcomes({}) == pytest.approx({'caught': 0.0, 'harmless': 1.0, 'wrong': 0.0})
    # an X between a T and a T^dag turns them into S: the check (an X measurement of S|+>) fires half of the time
    ef = error_formulas(_t_check(), final_check=False)
    between = next(pl.index for pl in ef.places if pl.name == 'T' and pl.qubit == 0)
    assert ef.classify({between: 'X'}) == 'harmless'
    assert ef.outcomes({between: 'X'}) == pytest.approx({'caught': 0.5, 'harmless': 0.5, 'wrong': 0.0})
    # sampling uses the shares: always that fault (X, Y or Z, a third each); Z is caught, X and Y half of the time
    counts = ef.sample(lambda place: 1.0 if place.index == between else 0.0, 3000, seed=4)
    assert counts['caught'] / 3000 == pytest.approx(2 / 3, abs=0.04) and counts['wrong'] == 0
    both = ef.probability([{between: 'X'}], 0.5)
    assert ef.probability([{between: 'X'}], 0.5, outcome='caught') == pytest.approx(0.5 * both)
    with pytest.raises(ValueError):
        ef.probability([{}], 0.1, outcome='kept')


# -- whether a run's map is zero is relative to the noiseless map ----------------------------------------------------

def _with_a_sign_flipped(code, signs):
    # the same code with the sign of one stabilizer flipped: what the program gives lies outside it
    flipped = list(getattr(code, signs))
    flipped[0] = flipped[0] + 1
    return dataclasses.replace(code, **{signs: flipped})


def _cube(t_layer):
    # the [[8,3,2]] code on a cube: qubits 1, 2, 3 are the logical inputs, or |+++> with the T layer (a logical CCZ).
    # With the T layer its X stabilizer holds only because the T phases cancel, so a code with that sign flipped
    # gives rounding errors rather than an exact zero.
    start = {q: '0' for q in range(4, 8)}
    start.update({q: '+' for q in (0, 1, 2, 3)} if t_layer else {0: '+'})
    p = QProgram(8, initial_states=start)
    p.add_circuit(_circuit(8, 0, [CNOT(0, 4), CNOT(0, 5), CNOT(0, 6), CNOT(0, 7),
                                  CNOT(1, 4), CNOT(1, 6), CNOT(1, 7), CNOT(0, 1),
                                  CNOT(2, 4), CNOT(2, 5), CNOT(2, 7), CNOT(0, 2),
                                  CNOT(3, 4), CNOT(3, 5), CNOT(3, 6), CNOT(0, 3)]), name='encode')
    if t_layer:
        p.add_circuit(_circuit(8, 0, [T(1), T(2), T(3), T(4), T(5, adjoint=True), T(6, adjoint=True),
                                      T(7, adjoint=True), T(0, adjoint=True)]), name='T layer')
    return p


@pytest.mark.parametrize('factor', [1e-12, 1e6], ids=['small tensors', 'large tensors'])
def test_a_verdict_does_not_depend_on_the_size_of_the_tensors(monkeypatch, factor):
    # after _unit_scalar the size of a tensor is arbitrary: 0.03 at d = 3, 4e-12 at d = 5. Every run of the same
    # program with all its sizes multiplied by one factor gives the same verdicts and the same shares.
    p = _repetition()
    plain = error_formulas(p)
    errors = [{k: letter} for k in range(len(plain.places)) for letter in 'XYZ']
    verdicts = [plain.classify(e) for e in errors]
    assert set(verdicts) == {'caught', 'harmless', 'wrong'}
    shares = [plain.outcomes(e) for e in errors]
    counts = pauli_errors(p, 1, [0, 1, 2], letter='X').counts
    runs = faults_module._runs

    def scaled(*args):
        return {key: (choi * factor ** 2, size * factor) for key, (choi, size) in runs(*args).items()}
    monkeypatch.setattr(faults_module, '_runs', scaled)
    ef = error_formulas(p)
    assert [ef.classify(e) for e in errors] == verdicts
    assert [ef.classify_directly(e) for e in errors] == verdicts
    for e, share in zip(errors, shares):
        assert ef.outcomes(e) == pytest.approx(share)
    assert pauli_errors(p, 1, [0, 1, 2], letter='X').counts == counts


def test_d5_cultivation_passes_the_final_check():
    # the largest entry of its noiseless decoded map is about 4e-12: once it counted as the zero map, and the program
    # was refused as if it failed its own final check
    p, _circuit_d5 = program_from_stim(open(PATH_D5).read())
    ef = error_formulas(p)
    assert ef.classify({}) == 'harmless'

    def fault(stage, gate, qubit, letter):
        return {next(pl.index for pl in ef.places if (pl.stage, pl.gate, pl.qubit) == (stage, gate, qubit)): letter}
    # a Z on an output qubit, which only the final check sees: all its runs are zero, and it is measured against the
    # noiseless map, not against its own largest run (which would make it wrong)
    z_output = fault(7, 29, 29, 'Z')
    assert ef.signature(z_output) == (frozenset(), frozenset(), frozenset({'oz29'}))
    assert ef.classify(z_output) == ef.classify_directly(z_output) == 'caught'
    # a Z that the checks and the logical qubit do not see, and one that also flips T gates
    z_stabilizer = fault(0, 21, 8, 'Z')
    assert ef.classify(z_stabilizer) == ef.classify_directly(z_stabilizer) == 'harmless'
    flips_t = fault(0, 0, 7, 'Z')
    assert ef.signature(flips_t)[1] and ef.classify(flips_t) == ef.classify_directly(flips_t) == 'harmless'
    # the same program against a code it does not pass is still refused
    with pytest.raises(ValueError, match='does not pass the final check'):
        error_formulas(p, code_out=_with_a_sign_flipped(p.code_at(p.num_stages - 1), 'sx_signs'))


@pytest.mark.filterwarnings('error::RuntimeWarning')           # a map that is exactly zero is never divided by its trace
@pytest.mark.parametrize('build, code, signs', [
    (_repetition, lambda: _repetition().code_at(2), 'sz_signs'),
    (lambda: _cube(True), lambda: _cube(False).code_at(0), 'sx_signs'),
    (lambda: _cube(True), lambda: _cube(False).code_at(0), 'sz_signs')],
                         ids=['bit-flip code', 'T layer, X sign (rounding errors)', 'T layer, Z sign (exact zero)'])
def test_a_code_the_noiseless_program_does_not_pass_is_refused(build, code, signs):
    p = build()
    assert error_formulas(p, code_out=code()).classify({}) == 'harmless'
    with pytest.raises(ValueError, match='does not pass the final check'):
        error_formulas(p, code_out=_with_a_sign_flipped(code(), signs))
