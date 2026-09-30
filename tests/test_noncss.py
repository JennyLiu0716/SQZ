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


# -- step 3: the ideal encoder and the logical action in codes that are not CSS ---------------------

def _reference_encoder(code, values, rng):
    """``E`` from its definition with numpy alone: ``Xbar^b`` applied to a code state with every ``Zbar = +1``."""
    n, k = code.n, code.k
    proj = np.eye(2 ** n, dtype=complex)
    for p in list(code.stabilizers) + list(code.logical_z):
        proj = proj @ (np.eye(2 ** n) + _matrix(p, values)) / 2
    psi = proj @ (rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n))
    cols = []
    for bits in itertools.product([0, 1], repeat=k):
        v = psi
        for i, b in enumerate(bits):
            if b:
                v = _matrix(code.logical_x[i], values) @ v
        cols.append(v)
    return np.array(cols).T


def _proportional(a, b):
    i = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    return abs(a[i]) > 1e-9 and np.allclose(a * (b[i] / a[i]), b, atol=1e-8)


def _sign_values(code, rng):
    names = {v.name for p in code.stabilizers if hasattr(p.sign, 'free_vars') for v in p.sign.free_vars()}
    return {name: int(rng.integers(2)) for name in sorted(names)}


def _encoder_matrix(e, values):
    return (e.substitute_variables(values) if values else e).to_matrix()


def _random_program(rng, n, k, stages, gates):
    p = QProgram(n, bits=6, initial_states={q: rng.choice(['0', '+']) for q in range(k, n)})
    m = 0
    for stage in range(stages):
        c = Circuit(n, bit_amount=6)
        for _ in range(rng.randint(*gates)):
            r = rng.random()
            a, b = rng.sample(range(n), 2)
            if r < 0.25:
                c.add_gate(HAD(a))
            elif r < 0.5:
                c.add_gate(S(a))
            elif r < 0.65:
                c.add_gate(CZ(a, b))
            elif r < 0.9:
                c.add_gate(CNOT(a, b))
            elif m < 3 and a >= k:
                c.add_gate(Measurement(a, result_symbol='c[%d]' % m))
                c.add_gate(Reset(a))
                m += 1
        p.add_circuit(c, name='stage %d' % stage)
    return p


def test_the_ideal_encoder_of_a_code_that_is_not_css_is_its_definition():
    # E X_i = Xbar_i E, E Z_i = Zbar_i E and the image is the code space with its signs: compared with
    # a matrix built from those properties alone
    rng = np.random.default_rng(1)
    c = Circuit(7)
    for gate in steane_encoder(list(range(7))):
        c.add_gate(gate)
    c.add_gate(S(0))
    codes = [_wheel_program().code_at(0), code_of(circuit_to_graph(c, initialize_qubits=[q != 3 for q in range(7)]))]
    prng = random.Random(12)
    mixed = 0
    while sum(not x.is_css for x in codes) < 25:
        p = _random_program(prng, prng.randint(3, 5), prng.randint(1, 2), prng.randint(1, 3), (3, 10))
        try:
            code = code_of(p.prefix_graph(p.num_stages - 1))
        except NotACodeError:
            continue
        if not code.is_css and code.k:
            codes.append(code)
            mixed += any(any(x.z) for x in code.logical_x) or any(any(z.x) for z in code.logical_z)
    for code in codes:
        if code.is_css:
            continue
        values = _sign_values(code, rng)
        assert _proportional(_encoder_matrix(code.encoder(), values), _reference_encoder(code, values, rng))
    assert mixed > 0                     # logical operators that are neither pure X nor pure Z


def test_the_choi_construction_gives_the_css_encoder_too():
    # the general construction, used only for codes that are not CSS, agrees with the two-layer
    # CSS encoder, outcome-dependent signs included
    from zxqec.codes.code import _choi_encoder
    from . import test_case_study_fold_h as F
    from .programs import steane_t_injection
    rng = np.random.default_rng(2)
    for code in (F.fold_transversal_h().code_at(0).data(), steane_t_injection().code_at(3)):
        assert code.is_css
        for _ in range(2):
            values = _sign_values(code, rng)
            assert _proportional(_encoder_matrix(_choi_encoder(code), values),
                                 _encoder_matrix(code.encoder(), values))


def _reorder(m, labels, order):
    """The rows of ``m``, indexed by qubits in the order ``labels``, re-indexed in the order ``order``."""
    n = len(labels)
    if list(labels) == list(order):
        return m
    t = m.reshape([2] * n + [m.shape[1]])
    return t.transpose([list(labels).index(q) for q in order] + [n]).reshape(m.shape)


def _definition_holds(act, rng):
    """``B E_in = E_out L`` in every class of runs, with ``E`` from :func:`_reference_encoder`."""
    from zxqec.spacetime.subdiagram import get_logical_qubit
    blk = act.block
    ins = [get_logical_qubit(blk, b) for b in blk.inputs()]
    outs = [get_logical_qubit(blk, b) for b in blk.outputs()]
    for br in act.branches():
        values = {k: int(v) for k, v in br.outcomes.items()}
        lhs = _encoder_matrix(blk, values) @ _reorder(_reference_encoder(act.code_in, values, rng),
                                                      act.code_in.qubits, ins)
        rhs = _reorder(_reference_encoder(act.code_out, values, rng), act.code_out.qubits, outs) @ \
            act.matrix(br.outcomes)
        if not _proportional(lhs, rhs):
            return False
    return True


def test_logical_actions_in_codes_that_are_not_css_are_those_of_the_definition():
    # every stage of random multi-stage Clifford programs with inputs and measurements: the tool's
    # logical action L satisfies B E_in = E_out L in every class of runs (Proposition on the logical
    # action), with E built from the codes by numpy alone
    prng = random.Random(7)
    rng = np.random.default_rng(3)
    checked = non_css = 0
    while non_css < 30:
        p = _random_program(prng, prng.randint(3, 5), prng.randint(1, 2), prng.randint(2, 3), (2, 8))
        for s in range(1, p.num_stages):
            try:
                act = p.logical_action(0, s)
            except (NotACodeError, OutcomeDependentCodeError):
                continue
            assert act.stays_in_code() is True       # the code after a stage is read from its image
            assert _definition_holds(act, rng)
            checked += 1
            non_css += not (act.code_in.is_css and act.code_out.is_css)
    assert checked >= non_css >= 30


def _on_every_qubit(code, gates, qubits=None):
    from zxqec.codes.logical import set_logical_qubit
    c = Circuit(code.n)
    for j in range(code.n) if qubits is None else qubits:
        for gate in gates:
            c.add_gate(gate(j))
    b = c.to_graph()
    for v, q in zip(b.inputs(), code.qubits):
        set_logical_qubit(b, v, q)
    for v, q in zip(b.outputs(), code.qubits):
        set_logical_qubit(b, v, q)
    return b


def test_transversal_cliffords_on_the_five_qubit_code():
    # the [[5,1,3]] code is kept by the order-3 Clifford applied to every qubit (HS or SH), and by
    # none of H, S, HSH; T on one qubit leaves it too. Where the tool says a block stays in the code,
    # B E_in = E_out L holds, and where it says the block leaves, it does not
    from pyzx.circuit.gates import T
    from zxqec.codes import logical_action
    rng = np.random.default_rng(4)
    code = _wheel_program().code_at(0).data()
    for gates, name in (([HAD, S], 'SH'), ([S, HAD], 'HS')):
        act = logical_action(_on_every_qubit(code, gates), code, code)
        assert act.stays_in_code() is True and act.is_deterministic() and act.operations() == {name}
        assert _definition_holds(act, rng)
    for gates, qubits in (([HAD], None), ([S], None), ([HAD, S, HAD], None), ([T], [0])):
        act = logical_action(_on_every_qubit(code, gates, qubits), code, code)
        assert act.stays_in_code() is False and act.verdict().startswith("leaves the code")
        assert not _definition_holds(act, rng)


def test_logical_gates_in_the_steane_code_after_an_s_gate():
    # S on qubit 0 moves the Steane code to a code that is not CSS without changing the logical
    # state; S^dag(0) H^7 S(0) keeps that code and is its logical H
    rng = np.random.default_rng(5)
    p = QProgram(7, initial_states={q: '0' for q in range(7) if q != 3})
    c = Circuit(7)
    for gate in steane_encoder(list(range(7))):
        c.add_gate(gate)
    p.add_circuit(c, name='encode')
    c = Circuit(7)
    c.add_gate(S(0))
    p.add_circuit(c, name='one S')
    c = Circuit(7)
    for _ in range(3):
        c.add_gate(S(0))
    for q in range(7):
        c.add_gate(HAD(q))
    c.add_gate(S(0))
    p.add_circuit(c, name='H in the new code')
    one_s, h = p.logical_action(0, 1), p.logical_action(0, 2)
    assert one_s.code_in.is_css and not one_s.code_out.is_css and one_s.operations() == {'I'}
    assert not h.code_in.is_css and h.code_in.same_code_as(h.code_out) and h.operations() == {'H'}
    assert one_s.is_deterministic() and h.is_deterministic()
    assert _definition_holds(one_s, rng) and _definition_holds(h, rng)
