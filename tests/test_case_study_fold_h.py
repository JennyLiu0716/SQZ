"""Regression test: the fold-transversal logical Hadamard case study with symbolic outcomes."""
import os
import random

import pytest
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import HAD, Reset
from pyzx.utils import VertexType

from zxqec.gpf import (boolean_variables, evaluate_phase, has_boundary_hadamard, is_gpf, is_leaf,
                       reduce_gpf, sample_consistent_assignment)
from zxqec.qprogram import QProgram

CIRCUITS = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'ft_h')
N, BITS = 19, 35
X_ANC, Z_ANC = [10, 11, 12, 13], [14, 15, 16, 17]


def load(name):
    return zx.Circuit.load(os.path.join(CIRCUITS, name + '.qasm'))


def with_prefix(circ, gates):
    c = Circuit(N, bit_amount=BITS)
    for g in gates:
        c.add_gate(g)
    c.add_circuit(circ)
    return c


def fold_transversal_h():
    p = QProgram(N, bits=BITS,
                 initial_states={**{q: '0' for q in range(1, 10)}, 18: '0', **{a: '+' for a in X_ANC + Z_ANC}})
    p.add_circuit(load('ft_h'), name='encode + X checks')
    p.add_circuit(load('ft_h_2'), name='transversal H')
    p.add_circuit(with_prefix(load('ft_h_3'), [Reset(a) for a in X_ANC]), name='Z checks (code 2)')
    p.add_circuit(load('ft_h_3.2'), name='X checks (code 2)')
    p.add_permutation([0, 3, 6, 9, 2, 5, 8, 1, 4, 7] + list(range(10, 19)), name='rotate 90')
    p.add_circuit(with_prefix(load('ft_h_5'), [Reset(a) for a in Z_ANC]), name='Z checks', physical=True)
    p.add_circuit(with_prefix(load('ft_h_5.2'), [g for a in X_ANC for g in (Reset(a), HAD(a))]),
                  name='X checks', physical=True)
    p.add_circuit(with_prefix(load('ft_h_6'), [Reset(a) for a in Z_ANC]), name='Z checks + decode', physical=True)
    return p


@pytest.fixture(scope='module')
def reduced():
    p = fold_transversal_h()
    g = p.quantum_graph()
    return p, g, reduce_gpf(g)


def test_program_shape(reduced):
    p, g, _ = reduced
    assert p.num_stages == 8 and p.stage(4).kind == 'permutation'
    assert len(g.inputs()) == 1 and len(g.outputs()) == 1           # logical in (q0) -> logical out (q18)
    assert len(boolean_variables(g)) == 34                           # c[25] is unused in the files


def test_reduces_to_hadamard_with_pauli_frame(reduced):
    _, g, res = reduced
    r = res.graph
    assert is_gpf(r) and not res.is_zero
    spiders = [v for v in r.vertices() if r.type(v) != VertexType.BOUNDARY]
    assert len(spiders) == 2 and not any(is_leaf(r, v) for v in spiders)
    z, x = sorted(spiders, key=lambda v: r.type(v) != VertexType.Z)
    assert r.type(z) == VertexType.Z and r.type(x) == VertexType.X and r.connected(z, x)
    # the Hadamard sits on the input wire, so the diagram reads H first and the frame after it:
    # that is the order in which a decoder applies the corrections
    assert sum(has_boundary_hadamard(r, b) for b in r.inputs()) == 1
    assert sum(has_boundary_hadamard(r, b) for b in r.outputs()) == 0
    # outcome-dependent Pauli frame on both layers
    assert {str(v) for v in r.phase(z).free_vars()} == {'c[26]', 'c[29]', 'c[32]'}   # the logical X readout
    assert {str(v) for v in r.phase(x).free_vars()} == {'c[0]', 'c[1]', 'c[2]', 'c[3]', 'c[4]'}


def test_detectors(reduced):
    _, g, res = reduced
    dets = {frozenset(str(v) for v in c.free_vars()) for c in res.detectors}
    assert len(res.detectors) == 24
    # every outcome is either fixed by a detector or part of the Pauli frame
    frame = {'c[0]', 'c[1]', 'c[2]', 'c[3]', 'c[4]', 'c[26]', 'c[29]', 'c[32]'}
    assert set().union(*dets) | frame == set(boolean_variables(g))
    for expected in [{'c[9]'}, {'c[17]'}, {'c[18]'}, {'c[6]', 'c[2]'}, {'c[13]', 'c[3]'}, {'c[14]', 'c[1]'},
                     {'c[27]', 'c[28]'}, {'c[33]', 'c[32]'},
                     {'c[34]', 'c[32]', 'c[31]', 'c[29]', 'c[27]', 'c[26]'}]:
        assert frozenset(expected) in dets


def test_all_zero_outcomes_give_hadamard(reduced):
    _, g, res = reduced
    h = Circuit(1); h.add_gate(HAD(0))
    zero = {n: 0 for n in boolean_variables(g)}
    assert zx.compare_tensors(res.to_graph().substitute_variables(zero), h.to_graph(), preserve_scalar=False)
    g0 = g.substitute_variables(zero)
    zx.full_reduce(g0)
    assert zx.compare_tensors(g0, h.to_graph(), preserve_scalar=False)


def test_consistent_outcomes_agree_with_full_reduce(reduced):
    _, g, res = reduced
    names = boolean_variables(g)
    rng = random.Random(2)
    for _ in range(3):
        a = sample_consistent_assignment(res.constraints, names, rng)
        assert all(evaluate_phase(c, a) == 0 for c in res.constraints)
        ga = g.substitute_variables(a)
        zx.full_reduce(ga)
        assert zx.compare_tensors(ga, res.to_graph().substitute_variables(a), preserve_scalar=False)


def test_impossible_outcome_is_zero(reduced):
    _, g, res = reduced
    bad = {n: 0 for n in boolean_variables(g)}
    bad['c[9]'] = 1
    assert reduce_gpf(g.substitute_variables(bad)).is_zero


def _gf2_rank(polys, names):
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    for p in polys:
        r = [0] * len(names)
        for var in p.free_vars():
            r[idx[str(var)]] ^= 1
        rows.append(r)
    rank = 0
    for col in range(len(names)):
        piv = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i][col]:
                rows[i] = [a ^ b for a, b in zip(rows[i], rows[rank])]
        rank += 1
    return rank


def test_detectors_agree_with_pyzx_full_reduce(reduced):
    # Independent oracle: PyZX's own simplifier on the symbolic master graph puts every factor
    # 1 + e^{i pi p} it meets into g.scalar.phasenodes; those p span the same GF(2) space as our detectors.
    _, g, res = reduced
    names = boolean_variables(g)
    h = g.copy()
    zx.full_reduce(h)
    pyzx_dets = list(h.scalar.phasenodes)
    assert len(pyzx_dets) == 24
    r_pyzx, r_ours = _gf2_rank(pyzx_dets, names), _gf2_rank(res.detectors, names)
    assert r_pyzx == r_ours == 24
    assert _gf2_rank(pyzx_dets + res.detectors, names) == 24


def test_every_detector_is_necessary(reduced):
    # violating exactly one detector (all others satisfied) makes the protocol's map the zero map
    import numpy as np
    _, g, res = reduced
    names = boolean_variables(g)
    rng = random.Random(3)
    for k, d in enumerate(res.constraints):
        others = [c for j, c in enumerate(res.constraints) if j != k] + [d + 1]
        a = sample_consistent_assignment(others, names, rng)
        ga = g.substitute_variables(a)
        zx.full_reduce(ga)
        assert np.allclose(ga.to_tensor(preserve_scalar=True), 0)


def test_signs_in_the_latest_round():
    """code_at(..., outcomes='latest'): a check measured by several rounds has its sign in the last round's result.

    The detectors c[5] = c[1], ..., c[16] = c[2] say that the later rounds measure the same values, so both codes
    are the same in every run without faults, written in different results."""
    from zxqec.gpf import equal_for_all_assignments, reduce_gpf, reduce_modulo
    p = fold_transversal_h()
    p.graph()
    signs = {}
    for stage in (0, 3, 6):
        early, late = p.code_at(stage).data(), p.code_at(stage, outcomes='latest').data()
        signs[stage] = sorted(str(s.sign) for s in late.stabilizers if str(s.sign) != '0')
        dets = reduce_gpf(p.prefix_graph(stage)).detectors
        for a, b in zip(early.stabilizers, late.stabilizers):
            assert a.x == b.x and a.z == b.z                    # the same generators
            assert reduce_modulo(a.sign, dets) == reduce_modulo(b.sign, dets)   # the same sign in every such run
        assert equal_for_all_assignments(early.encoder(), late.encoder(), constraints=dets)
    assert signs[0] == ['c[1]', 'c[2]', 'c[3]', 'c[4]']
    assert signs[3] == ['c[5]', 'c[6]', 'c[7]', 'c[8]']             # the Z checks of code 2, measured in stage 2
    assert signs[6] == ['c[13]', 'c[14]', 'c[15]', 'c[16]']         # the same checks, measured again after the turn
    with pytest.raises(ValueError):
        p.code_at(0, outcomes='middle')


def test_measured_operators_are_the_checks_each_round_reads():
    """QProgram.measured_operators: each result of a round of checks with the check it measures (logical labels)."""
    p = fold_transversal_h()

    def letters(op):
        return ''.join(op.letters)[1:10]                 # data qubits 1..9

    first = p.measured_operators(0)
    assert letters(first['c[1]']) == 'XXIXXIIII'         # the first round: X checks on |0> data (random results)
    assert letters(first['c[2]']) == 'IXXIIIIII'
    z2, x2 = p.measured_operators(2), p.measured_operators(3)
    assert {k: letters(v) for k, v in z2.items()} == {'c[5]': 'ZZIZZIIII', 'c[6]': 'IZZIIIIII', 'c[7]': 'IIIIZZIZZ',
                                                       'c[8]': 'IIIIIIZZI'}
    assert {k: letters(v) for k, v in x2.items()} == {'c[9]': 'XIIXIIIII', 'c[10]': 'IXXIXXIII',
                                                       'c[11]': 'IIIXXIXXI', 'c[12]': 'IIIIIXIIX'}
    assert all(v.sign == 0 and not any(v.x[10:] + v.z[10:]) for v in list(z2.values()) + list(x2.values()))
    assert p.measured_operators(4) == {}                 # a permutation measures nothing
    last = p.measured_operators(7)
    assert last['c[26]'] is None                         # a data qubit read out after the decoding: no fixed operator


def test_draw_measurement_puts_each_result_on_its_check():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = fold_transversal_h()
    coords = {q: ((q - 1) % 3 + 1, (q - 1) // 3 + 1) for q in range(1, 10)}
    for stage, kind, names in ((3, 'X', ['c[10]', 'c[11]', 'c[12]', 'c[9]']), (6, 'X', ['c[17]', 'c[18]', 'c[19]', 'c[20]'])):
        code = p.code_at(stage)
        bare = code.draw_measurement(coords, kind, label_size=7).axes[0]
        assert not bare.texts                            # these checks are + in every run
        ax = code.draw_measurement(coords, kind, results=p.measured_operators(stage), label_size=7).axes[0]
        assert sorted(t.get_text() for t in ax.texts) == names
    plt.close('all')


def test_gpf_checks_gives_the_round_of_code_2_as_its_checks():
    # 'gpf' writes the Z checks of code 2 as sums (c[5] + c[7] + c[8], ...) that bend three data wires across the
    # grid; 'gpf_checks' gives one spider per check, joined to that check's data qubits, and every data wire straight
    from zxqec.spacetime import get_logical_qubit
    plaquettes = {'c[5]': [1, 2, 4, 5], 'c[6]': [2, 3], 'c[7]': [5, 6, 8, 9], 'c[8]': [7, 8]}
    for rewrite, bent in (('gpf', 3), ('gpf_checks', 0)):
        p = fold_transversal_h()
        level = p.group_stages([(2, 2, 'Z checks (code 2)', None, rewrite)], one_level=True)
        g = p.reduced_block(level, 2).graph
        ins = {next(iter(g.neighbors(b))): get_logical_qubit(g, b) for b in g.inputs()}
        outs = {next(iter(g.neighbors(b))) for b in g.outputs()}
        assert sum(1 for s in outs if len([u for u in g.neighbors(s) if u in ins]) != 1) == bent
        if rewrite == 'gpf_checks':
            checks = {str(g.phase(v)): sorted(ins[u] for u in g.neighbors(v)) for v in g.vertices()
                      if g.type(v) != VertexType.BOUNDARY and v not in ins and v not in outs}
            assert checks == plaquettes
