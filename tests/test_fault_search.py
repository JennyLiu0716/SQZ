"""search_lowest_wrong: the fewest faults that are wrong, proposed by a solver and decided by classify.

The answers are checked against what is known: d = 3 cultivation needs 2 faults with real T gates (lowest_wrong(2),
about 200 s, finds 48 wrong pairs) and 3 with S in place of T (as circuit_distance says), and on small programs the
solver agrees with lowest_wrong. Weights and verdicts are asserted, never one particular pattern. Most searches use
the SAT solver, which is the fastest here; CP-SAT and SCIP run on the small programs and once each on d = 3.
"""
import importlib.util
import itertools
import os
import sys

import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement, Reset, T

from zxqec.qprogram import (QProgram, branch_model, error_formulas, program_from_stim, search_lowest_wrong)

pytest.importorskip('ortools')
needs_sat = pytest.mark.skipif(importlib.util.find_spec('pysat') is None, reason='python-sat is not installed')

PATH_D3 = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'cultivation', 'd3_cultivation.stim')
SOLVERS = ['cp-sat', 'scip'] + (['sat'] if importlib.util.find_spec('pysat') else [])


def _text():
    with open(PATH_D3) as fh:
        return fh.read()


@pytest.fixture(scope='module')
def real():
    program, _circuit = program_from_stim(_text())
    ef = error_formulas(program)
    return program, ef, branch_model(ef)


@pytest.fixture(scope='module')
def searched(real):
    # one search on d = 3 with signature cuts, shared by the tests that look at its candidates
    pytest.importorskip('pysat')
    _program, ef, model = real
    return search_lowest_wrong(ef, 3, solver='sat', model=model)


@pytest.fixture(scope='module')
def with_s(real):
    # S in place of T: a Clifford circuit whose output is a known stabilizer state, so the real circuit's code (with
    # its logical qubit) is the one to check against
    program, _circuit = program_from_stim(_text().replace('T_DAG ', 'S_DAG ').replace('\nT ', '\nS '))
    code = real[0].code_at(real[0].num_stages - 1)
    ef = error_formulas(program, code_out=code)
    return code, ef, branch_model(ef, code_out=code, sample=16)


def _circuit(n, bits, gates):
    c = Circuit(n, bit_amount=bits)
    for g in gates:
        c.add_gate(g)
    return c


def _repetition():
    # the bit-flip code of tests/test_faults.py: data 0-2 (qubit 0 the input), helpers 3 and 4, two rounds
    p = QProgram(5, bits=4, initial_states={1: '0', 2: '0', 3: '0', 4: '0'})
    p.add_circuit(_circuit(5, 4, [CNOT(0, 1), CNOT(0, 2)]), name='encode')
    for k, name in ((0, 'round 1'), (2, 'round 2')):
        p.add_circuit(_circuit(5, 4, [Reset(3), Reset(4), CNOT(0, 3), CNOT(1, 3), CNOT(1, 4), CNOT(2, 4),
                                      Measurement(3, k), Measurement(4, k + 1)]), name=name)
    return p


def _phase_flip_with_t_pairs():
    # the phase-flip code (data 0-2, helpers 3 and 4, two rounds) with a T and a T^dag on qubit 1 before the first
    # round's checks and on qubit 0 before the second's: they cancel without errors, an X between them does not
    p = QProgram(5, bits=4, initial_states={1: '0', 2: '0', 3: '0', 4: '0'})
    had = [HAD(0), HAD(1), HAD(2)]
    p.add_circuit(_circuit(5, 4, [CNOT(0, 1), CNOT(0, 2)] + had), name='encode')
    for k, q, name in ((0, 1, 'round 1'), (2, 0, 'round 2')):
        p.add_circuit(_circuit(5, 4, [T(q), T(q, adjoint=True), Reset(3), Reset(4)] + had
                               + [CNOT(0, 3), CNOT(1, 3), CNOT(1, 4), CNOT(2, 4)] + had
                               + [Measurement(3, k), Measurement(4, k + 1)]), name=name)
    return p


def _cube(t_layer):
    # the [[8,3,2]] code on a cube (three logical qubits), as in tests/test_faults.py
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


def _wrong_signatures(ef, model, max_weight, per):
    # every signature of at most max_weight faults that has a wrong branch and that classify calls wrong
    groups = [sorted({ef.mask(e) for e in kinds}) for _place, kinds in ef.single_faults(per)]
    seen = set()
    for k in range(1, max_weight + 1):
        for combo in itertools.combinations(groups, k):
            for choice in itertools.product(*combo):
                m = 0
                for x in choice:
                    m ^= x
                seen.add(m)
    return {m for m in seen if m and model.wrong_branch(m) is not None and ef._verdict(m) == 'wrong'}


def test_the_model_of_d3_cultivation(real, with_s):
    _program, ef, model = real
    assert len(model.detectors) == len(ef._detector_masks) == 26
    assert len(model.final_rows) == 6 and len(model.z_after_t) == len(ef.t_flips) == 15
    assert model.wrong_kind == 'any' and model.wrong_logicals == ('X', 'Y', 'Z')
    assert sum(model.sample.values()) == 48 and {'caught', 'harmless', 'wrong'} <= set(model.sample)
    # the final check reads the results in the stabilizers' signs: c[4], c[27], c[5] + c[28]
    assert {c for r in model.final_rows for c in model.names(r) if c in ef.flips} == {'c[4]', 'c[27]', 'c[5]',
                                                                                      'c[28]'}
    assert 'harmless 0 wrong' not in model.describe()
    # a class is fixed by the 26 checks, the 6 final-check rows and the two logical rows
    assert len(model.coset_rows) == 34 and len(model.harmless_basis) == 11
    # with S in place of T the output is a Y-type logical state: Y is harmless, X and Z are wrong
    _code, _ef_s, model_s = with_s
    assert model_s.wrong_kind == 'xor' and model_s.harmless_logicals == ('Y',)
    assert not model_s.z_after_t
    assert model_s.coset_rows[-1] == model_s.lx ^ model_s.lz


def test_d3_cultivation_with_real_t_needs_two_faults(real, searched):
    _program, ef, model = real
    for r, classes in ((searched, 8), (search_lowest_wrong(ef, 3, model=model), 8),       # SAT, then the defaults
                       (search_lowest_wrong(ef, 3, solver='sat', cut='coset', model=model), 5)):
        assert (r.weight, r.verdict, r.finished, r.lower_bound, r.ruled_out) == (2, 'wrong', True, 2, [1])
        assert ef.classify(r.pattern) == 'wrong'
        assert len(r.pattern) <= 4 and r.rejected_per_weight[1] == classes   # the 8 single faults with a harmless
        # sum fall into 5 classes
        assert model.wrong_branch(ef.mask(r.pattern)) is not None   # the lightest wrong patterns have a wrong branch
    assert ef.classify_directly(r.pattern) == 'wrong'
    assert '2 faults are wrong' in r.describe() and 'coset cuts' in r.describe()


def test_several_candidates_per_solve(real):
    _program, ef, model = real
    r = search_lowest_wrong(ef, 3, batch=8, model=model)
    assert r.weight == 2 and ef.classify(r.pattern) == 'wrong'
    assert max(len(s.candidates) for s in r.solves) > 1
    assert len({c.signature for c in r.candidates}) == len(r.candidates)        # one candidate per signature


@needs_sat
def test_several_classes_per_solve(real):
    _program, ef, model = real
    r = search_lowest_wrong(ef, 3, solver='sat', cut='coset', batch=4, model=model)
    assert r.weight == 2 and ef.classify(r.pattern) == 'wrong'
    assert max(len(s.candidates) for s in r.solves) > 1
    classes = [model.coset(ef.mask(c.pattern)) for c in r.candidates]
    assert len(set(classes)) == len(classes)                                    # one candidate per class


@needs_sat
def test_s_in_place_of_t_needs_three_faults(with_s):
    code, ef, model = with_s
    r = search_lowest_wrong(ef, 3, solver='sat', cut='coset', code_out=code, model=model)
    assert (r.weight, r.finished) == (3, True)
    assert ef.classify(r.pattern) == ef.classify_directly(r.pattern) == 'wrong'
    r = search_lowest_wrong(ef, 2, solver='scip', model=model)
    assert (r.weight, r.pattern, r.finished, r.lower_bound, r.ruled_out) == (None, None, True, 3, [1, 2])
    assert 'No pattern of at most 2 faults is wrong' in r.describe()


def test_a_class_has_one_verdict(real, searched):
    # a Pauli signature that every coset row reads as 0 changes no verdict: here on one candidate of each verdict
    _program, ef, model = real
    picked = {}
    for c in searched.candidates:
        picked.setdefault(c.verdict, ef.mask(c.pattern))
    assert set(picked) == {'caught', 'harmless', 'wrong'}
    h = 0
    for k, b in enumerate(model.harmless_basis):
        h ^= b if k % 2 == 0 else 0
    assert h and not h & ef._t_bits and model.coset(h) == (0, 0)
    for verdict, m in picked.items():
        assert model.coset(m ^ h) == model.coset(m)
        assert ef._verdict(m ^ h) == verdict


@pytest.mark.parametrize('per', ['gate', 'place'])
@pytest.mark.parametrize('merge', [True, False])
def test_a_small_program_agrees_with_lowest_wrong(per, merge):
    ef = error_formulas(_repetition())
    weight, wrong = ef.lowest_wrong(max_weight=2, per=per)
    model = branch_model(ef, per, merge=merge)
    for solver in SOLVERS:
        for cut in ('signature', 'coset'):
            r = search_lowest_wrong(ef, 2, per=per, solver=solver, cut=cut, model=model)
            assert r.weight == weight == 1 and r.finished
            assert frozenset(r.pattern.items()) in {frozenset(p.items()) for p in wrong}


def test_every_wrong_signature_is_listed():
    # with all_wrong the search goes on after the first wrong pattern; it must list every wrong signature with a
    # wrong branch, whatever the solver and the cut. SCIP with its dual reductions on missed one of them here (per
    # place, seed 0, signature cuts): it said "infeasible" with a wrong signature left
    ef = error_formulas(_phase_flip_with_t_pairs())
    model = branch_model(ef, 'place')
    truth = _wrong_signatures(ef, model, 2, 'place')
    weight, _wrong = ef.lowest_wrong(max_weight=2, per='place')
    assert weight == 1 and len(truth) > 10
    for solver, cut in itertools.product(SOLVERS, ('signature', 'coset')):
        if (solver, cut) != ('scip', 'coset'):     # SCIP is the slow one here
            r = search_lowest_wrong(ef, 2, per='place', solver=solver, cut=cut, all_wrong=True, model=model)
            assert r.finished and r.weight == r.lower_bound == weight and r.ruled_out == list(range(1, weight))
            assert {ef.mask(c.pattern) for c in r.wrong} == truth, (solver, cut)
            assert all(ef.classify(c.pattern) == 'wrong' for c in r.wrong)


@pytest.mark.parametrize('per', ['gate', 'place'])
def test_a_known_logical_state_where_z_is_harmless(per):
    # the bit-flip code of tests/test_faults.py started in |000>, the logical |0>: a logical Z does nothing to it
    def build(start):
        p = QProgram(5, bits=4, initial_states=start)
        p.add_circuit(_circuit(5, 4, [CNOT(0, 1), CNOT(0, 2)]), name='encode')
        p.add_circuit(_circuit(5, 4, [CNOT(0, 3), CNOT(1, 3), CNOT(1, 4), CNOT(2, 4), Measurement(3, 0),
                                      Measurement(4, 1)]), name='round')
        return p
    helpers = {1: '0', 2: '0', 3: '0', 4: '0'}
    code = build(helpers).code_at(1)
    ef = error_formulas(build({0: '0', **helpers}), code_out=code)
    model = branch_model(ef, per, code_out=code)
    assert model.harmless_logicals == ('Z',) and model.wrong_kind == 'not Z'
    assert model.coset_rows[-1] == model.lz
    weight, wrong = ef.lowest_wrong(max_weight=3, per=per)
    for solver, cut in (('cp-sat', 'signature'), (SOLVERS[-1], 'coset')):
        r = search_lowest_wrong(ef, 3, per=per, model=model, solver=solver, cut=cut)
        assert r.weight == weight and r.finished and ef.classify(r.pattern) == 'wrong'
        assert frozenset(r.pattern.items()) in {frozenset(p.items()) for p in wrong}


def test_wrong_branch_agrees_with_listing_the_branches(real):
    _program, ef, model = real
    masks = sorted({ef.mask(fault) for _place, kinds in ef.single_faults() for fault in kinds} - {0})
    flipping = [m for m in masks if m & ef._t_bits]
    assert flipping
    for m in flipping[::4] + masks[:40]:
        listed = any(model.verdict(b) == 'wrong' for _e, b in model.branches(m))
        assert (model.wrong_branch(m) is not None) == listed, ef._decode(m)


def test_choices_at_shared_gates_are_matched(real):
    # a merged choice can sit at several gates; when its first gate is the only gate of another chosen fault, the
    # faults are still put at distinct gates (a greedy choice of the first free gate failed here)
    _program, ef, model = real
    only = {}
    for i, (_m, faults) in enumerate(model.choices):
        if len({g for g, _f in faults}) == 1:
            only.setdefault(faults[0][0], i)
    a, b = next((i, only[faults[0][0]]) for i, (_m, faults) in enumerate(model.choices)
                if len({g for g, _f in faults}) > 1 and faults[0][0] in only)
    p = model.pattern([a, b])
    assert ef.mask(p) == model.choices[a][0] ^ model.choices[b][0]
    assert len({(ef.places[k].stage, ef.places[k].gate) for k in p}) == 2


def test_a_time_limit_stops_the_search(real):
    _program, ef, model = real
    for solver in SOLVERS:
        r = search_lowest_wrong(ef, 3, time_limit=0.05, model=model, solver=solver)
        assert not r.finished and r.weight is None and r.pattern is None
        assert 'Not finished' in r.describe()


def test_what_the_search_refuses(real, with_s):
    _program, ef, model = real
    # three logical qubits
    cube = error_formulas(_cube(True), code_out=_cube(False).code_at(0))
    with pytest.raises(ValueError, match='3 logical qubits'):
        branch_model(cube, code_out=_cube(False).code_at(0))
    # S in place of T without its code: the program's own output code has no logical qubit
    with pytest.raises(ValueError, match='give the same code here'):
        branch_model(with_s[1])
    # no final check
    with pytest.raises(ValueError, match='final check'):
        branch_model(error_formulas(_repetition(), final_check=False))
    with pytest.raises(ValueError, match='solver'):
        search_lowest_wrong(ef, model=model, solver='gurobi')
    with pytest.raises(ValueError, match='cut'):
        search_lowest_wrong(ef, model=model, cut='class')
    with pytest.raises(ValueError, match='other error formulas'):
        search_lowest_wrong(with_s[1], model=model)
    # a model given with other faults, or for another output code, is not used silently
    with pytest.raises(ValueError, match="per 'gate', not per 'place'"):
        search_lowest_wrong(ef, model=model, per='place')
    import dataclasses
    unsigned = dataclasses.replace(model.code, sx_signs=[0] * len(model.code.sx_signs))
    with pytest.raises(ValueError, match='another output code'):
        search_lowest_wrong(ef, model=model, code_out=unsigned)
    with pytest.warns(UserWarning, match='sample=0'):
        assert 'Self-check: none' in branch_model(error_formulas(_repetition()), sample=0).describe()


def test_a_model_that_does_not_fit_is_refused(real):
    # the output code with the results dropped from its signs: its final-check rows miss them, the verdicts of
    # classify (which uses the code error_formulas was given) do not fit, and the model is refused, not used
    program, ef, _model = real
    import dataclasses
    code = program.code_at(program.num_stages - 1)
    unsigned = dataclasses.replace(code, sx_signs=[0] * len(code.sx_signs))
    with pytest.raises(RuntimeError, match='do not fit'):
        branch_model(ef, code_out=unsigned)


def test_without_ortools_or_python_sat(real, monkeypatch):
    _program, ef, model = real
    monkeypatch.setitem(sys.modules, 'ortools.sat.python', None)
    with pytest.raises(ImportError, match='ortools'):
        search_lowest_wrong(ef, model=model)
    monkeypatch.setitem(sys.modules, 'pysat.solvers', None)
    with pytest.raises(ImportError, match='python-sat'):
        search_lowest_wrong(ef, model=model, solver='sat')
