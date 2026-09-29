"""Codes that are not CSS: odd Hadamard cycles in the normal form, and mixed stabilizers.

Every stabilizer the library reports is checked against the diagram itself:
with ``E`` the linear map of the diagram in an allowed run (outcomes set to
values the detectors allow), a stabilizer ``P`` must satisfy ``P E = E``, sign
included. That uses only PyZX's tensor contraction and numpy, not the
reduction being tested.
"""
import itertools
import random

import numpy as np
import pytest
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, S, Measurement, Reset
from pyzx.circuit.graphparser import circuit_to_graph

from .programs import steane_encoder
from zxqec.codes import NotACodeError, OutcomeDependentCodeError, PauliString, code_of
from zxqec.gpf import (boolean_variables, evaluate_phase, is_gpf, push_hadamards_to_boundary, reduce_gpf,
                       restore_boundary_hadamards, to_gpf)
from zxqec.qprogram import ParityControlled, QProgram

_P = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]), 'Y': np.array([[0, -1j], [1j, 0]]),
      'Z': np.diag([1, -1])}


def _matrix(p, values):
    m = np.array([[1]])
    for ch in p.letters:
        m = np.kron(m, _P[ch])
    sign = evaluate_phase(p.sign, values) if hasattr(p.sign, 'free_vars') else int(p.sign)
    return (-1) ** (sign % 2) * m


def _allowed_runs(g):
    """``(assignment, matrix)`` for every outcome assignment the detectors allow."""
    names = boolean_variables(g)
    res = reduce_gpf(g)
    for values in itertools.product([0, 1], repeat=len(names)):
        a = dict(zip(names, values))
        if any(evaluate_phase(c, a) for c in res.constraints):
            continue
        e = g.substitute_variables(a).to_matrix() if names else g.to_matrix()
        if not np.allclose(e, 0):
            yield a, e


def _assert_code_fixes_its_image(g, code):
    stabs = code.stabilizers
    assert len(stabs) == code.n - code.k
    runs = 0
    for a, e in _allowed_runs(g):
        runs += 1
        for p in stabs:
            assert np.allclose(_matrix(p, a) @ e, e), (str(p), a)
    assert runs > 0
    for i, (x, z) in enumerate(zip(code.logical_x, code.logical_z)):
        assert all(x.commutes_with(p) and z.commutes_with(p) for p in stabs)
        assert not x.commutes_with(z)
        for j, (x2, z2) in enumerate(zip(code.logical_x, code.logical_z)):
            if i != j:
                assert x.commutes_with(x2) and x.commutes_with(z2) and z.commutes_with(z2)


def _random_clifford(rng, n, depth):
    c = Circuit(n)
    for _ in range(depth):
        r = rng.random()
        a, b = rng.sample(range(n), 2)
        if r < 0.25:
            c.add_gate(HAD(a))
        elif r < 0.45:
            c.add_gate(S(a))
        elif r < 0.7:
            c.add_gate(CZ(a, b))
        else:
            c.add_gate(CNOT(a, b))
    return c


# -- step 1: the normal form accepts any diagram of Z- and X-spiders ------------------------------

def test_odd_hadamard_cycles_become_cz_gadgets():
    rng = random.Random(1)
    odd = 0
    for _ in range(60):
        g = _random_clifford(rng, rng.randint(2, 5), rng.randint(4, 25)).to_graph()
        try:
            push_hadamards_to_boundary(g.copy())
        except ValueError:
            odd += 1                                  # rejected before; accepted now
        h = to_gpf(g)
        assert is_gpf(h)
        assert zx.compare_tensors(restore_boundary_hadamards(h), g, preserve_scalar=False)
        assert zx.compare_tensors(reduce_gpf(g).to_graph(), g, preserve_scalar=False)
    assert odd > 30


def test_the_old_behaviour_is_still_available():
    c = Circuit(2)
    c.add_gate(CNOT(0, 1)); c.add_gate(HAD(0)); c.add_gate(CNOT(0, 1))
    with pytest.raises(ValueError):
        to_gpf(c.to_graph(), cz_gadgets=False)


def test_detectors_are_exact_with_clifford_phases():
    # zero branch <=> a reported detector is violated, on random circuits with S gates and
    # mid-circuit measurements (S before a measurement is what needed the closed-form sums)
    rng = random.Random(8)
    for _ in range(25):
        n = rng.randint(2, 4)
        c = Circuit(n, bit_amount=4)
        m = 0
        for _ in range(rng.randint(6, 20)):
            r = rng.random()
            a, b = rng.sample(range(n), 2)
            if r < 0.2:
                c.add_gate(HAD(a))
            elif r < 0.4:
                c.add_gate(S(a))
            elif r < 0.6:
                c.add_gate(CZ(a, b))
            elif r < 0.85:
                c.add_gate(CNOT(a, b))
            elif m < 4:
                c.add_gate(Measurement(a, result_symbol='c[%d]' % m))
                c.add_gate(Reset(a))
                m += 1
        g = circuit_to_graph(c, initialize_qubits=[False] * n)
        names = boolean_variables(g)
        res = reduce_gpf(g)
        for values in itertools.product([0, 1], repeat=len(names)):
            a = dict(zip(names, values))
            violated = res.is_zero or any(evaluate_phase(k, a) for k in res.constraints)
            zero = np.allclose(g.substitute_variables(a).to_tensor(preserve_scalar=True), 0)
            assert zero == violated, a


# -- step 2: reading codes that are not CSS --------------------------------------------------------

def test_ring_graph_state():
    c = Circuit(5)
    for q in range(5):
        c.add_gate(HAD(q))
    for q in range(5):
        c.add_gate(CZ(q, (q + 1) % 5))
    g = circuit_to_graph(c, initialize_qubits=[True] * 5)
    code = code_of(g)
    assert (code.n, code.k) == (5, 0) and not code.is_css
    _assert_code_fixes_its_image(g, code)
    for q in range(5):                                # the graph-state stabilizers Z X Z
        letters = ['I'] * 5
        letters[q], letters[(q - 1) % 5], letters[(q + 1) % 5] = 'X', 'Z', 'Z'
        assert code.contains(''.join(letters))


def _wheel_program():
    """The [[5,1,3]] code: an input joined to the 5-ring by CZs, then measured in the X basis."""
    p = QProgram(6, bits=1, initial_states={q: '+' for q in range(1, 6)})
    c = Circuit(6, bit_amount=1)
    for q in range(1, 6):
        c.add_gate(CZ(q, q % 5 + 1))
    for q in range(1, 6):
        c.add_gate(CZ(0, q))
    c.add_gate(HAD(0))
    c.add_gate(Measurement(0, result_symbol='c'))
    p.add_circuit(c, name='wheel')
    return p


def test_five_qubit_code():
    p = _wheel_program()
    code = p.code_at(0)
    assert code.parameters == (5, 1, 3) and not code.is_css
    _assert_code_fixes_its_image(p.prefix_graph(0), code)
    shifts = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ', 'ZZXIX']
    assert all(code.contains(s) for s in shifts)
    assert all(s.letters in shifts for s in code.stabilizers)    # the textbook generators


def test_steane_code_after_an_s_gate_matches_the_tensor():
    c = Circuit(7)
    for gate in steane_encoder(list(range(7))):
        c.add_gate(gate)
    c.add_gate(S(0))
    g = circuit_to_graph(c, initialize_qubits=[q != 3 for q in range(7)])
    code = code_of(g)
    assert code.parameters == (7, 1, 3) and not code.is_css
    _assert_code_fixes_its_image(g, code)


def test_an_s_gate_controlled_by_an_outcome_is_refused():
    # S on one qubit in the runs with m = 1 only: the code itself differs between runs
    p = QProgram(8, bits=1, initial_states={**{q: '0' for q in range(7) if q != 3}, 7: '+'})
    c = Circuit(8, bit_amount=1)
    for gate in steane_encoder(list(range(7))):
        c.add_gate(gate)
    c.add_gate(Measurement(7, result_symbol='m'))
    p.add_circuit(c, name='encode, and a random bit')
    s = Circuit(8, bit_amount=1)
    s.add_gate(ParityControlled('S', 0, ['m']))
    p.add_circuit(s, name='S if m = 1')
    with pytest.raises(OutcomeDependentCodeError):
        p.code_at(1)


def test_random_clifford_programs():
    # random programs of one to three stages with inputs, prepared qubits and measurements;
    # every code read must fix the program's image in every allowed run, and every refusal
    # must be a program that really is not an isometry
    rng = random.Random(12)
    read = non_css = 0
    for _ in range(40):
        n = rng.randint(3, 5)
        k = rng.randint(0, 2)
        p = QProgram(n, bits=6, initial_states={q: rng.choice(['0', '+']) for q in range(k, n)})
        m = 0
        for stage in range(rng.randint(1, 3)):
            c = Circuit(n, bit_amount=6)
            for _ in range(rng.randint(3, 10)):
                r = rng.random()
                a, b = rng.sample(range(n), 2)
                if r < 0.2:
                    c.add_gate(HAD(a))
                elif r < 0.4:
                    c.add_gate(S(a))
                elif r < 0.6:
                    c.add_gate(CZ(a, b))
                elif r < 0.88:
                    c.add_gate(CNOT(a, b))
                elif m < 4 and a >= k:
                    c.add_gate(Measurement(a, result_symbol='c[%d]' % m))
                    c.add_gate(Reset(a))
                    m += 1
            p.add_circuit(c, name='stage %d' % stage)
        g = p.prefix_graph(p.num_stages - 1)
        try:
            code = code_of(g)
        except NotACodeError:
            isometry = True
            for _a, e in _allowed_runs(g):
                gram = e.conj().T @ e
                isometry &= bool(np.allclose(gram, gram[0, 0] * np.eye(gram.shape[0])))
            assert not isometry
            continue
        _assert_code_fixes_its_image(g, code)
        read += 1
        non_css += not code.is_css
    assert read > 30 and non_css > 10


def test_a_helper_qubit_in_plus_i_is_a_fixed_qubit():
    # S|+> = |+i>: a single-qubit Y stabilizer, next to a code that is not CSS
    c = Circuit(7)
    c.add_gate(HAD(0))
    c.add_gate(S(0))
    for q in range(1, 6):
        c.add_gate(HAD(q))
    for q in range(1, 6):
        c.add_gate(CZ(q, q % 5 + 1))
    g = circuit_to_graph(c, initialize_qubits=[True] * 6 + [False])
    code = code_of(g)
    assert code.fixed_qubits() == {0: '+i'}
    _assert_code_fixes_its_image(g, code)
