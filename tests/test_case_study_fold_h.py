"""Regression test: thesis case study 1 (fold-transversal logical Hadamard) with symbolic outcomes."""
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
