"""Clifford logical gates without matrices: the Clifford fast path of code_of, the tableau and its name, and the
Clifford part of a non-Clifford gate (zxqec/codes/tableau.py)."""
import random
from fractions import Fraction

import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, S, T
from pyzx.symbolic import new_var
from pyzx.utils import VertexType

import zxqec.codes.code as code_module
from zxqec.codes import (NotCliffordError, clifford_plus_phases, code_of, iap_normal_form, logical_action)
from zxqec.gpf import boolean_variables, same_map

from . import test_case_study_fold_h as F
from .programs import steane_t_injection
from .test_ccz import FROM_X, FROM_Y, FROM_Z, GHZ, P, _program as _ccz
from .test_code_switching import _switch
from .test_shor import _shor
from .test_teleportation import _teleport_cx


def _isometry(n, k, depth, rng, symbolic=False):
    """A random Clifford isometry from k inputs to n outputs (the circuits of the paper's cost appendix)."""
    c = Circuit(n)
    for _ in range(depth):
        r = rng.random()
        if r < .3:
            q = rng.randrange(n)
            c.add_gate(HAD(q) if rng.random() < .5 else S(q))
        else:
            a, b = rng.sample(range(n), 2)
            c.add_gate(CNOT(a, b) if r < .65 else CZ(a, b))
    g = c.to_graph()
    for b in list(g.inputs())[k:]:
        g.set_type(b, VertexType.X)
    g.set_inputs(tuple(list(g.inputs())[:k]))
    if symbolic:                                   # outcome-dependent signs: pi c on a few spiders
        spiders = [v for v in g.vertices() if g.type(v) in (VertexType.Z, VertexType.X)]
        for j in range(rng.randint(1, 3)):
            v = rng.choice(spiders)
            g.set_phase(v, g.phase(v) + new_var('c[{}]'.format(j), True, g.var_registry))
    return g


def _code_text(g, **kw):
    try:
        return str(code_of(g.copy(), **kw))
    except Exception as e:                         # noqa: BLE001
        return 'error {}: {}'.format(type(e).__name__, e)


def test_clifford_fast_path_reads_the_same_code_as_listing_the_gadget_patterns(monkeypatch):
    # the surviving X-strings of a Clifford gadget layer are the kernel of one matrix; the old way listed 2^r
    # patterns. Both must print the same code, with signs that depend on outcomes too.
    rng = random.Random(3)
    real = code_module._clifford_passing
    for _ in range(40):
        n = rng.randint(3, 10)
        g = _isometry(n, rng.randint(0, min(2, n - 1)), rng.randint(n, 5 * n), rng, symbolic=rng.random() < 0.4)
        new = [_code_text(g), _code_text(g, generators='echelon'), _code_text(g, common_to_all_runs=True)]
        monkeypatch.setattr(code_module, '_clifford_passing', lambda *a, **k: None)
        old = [_code_text(g), _code_text(g, generators='echelon'), _code_text(g, common_to_all_runs=True)]
        monkeypatch.setattr(code_module, '_clifford_passing', real)
        assert new == old


def test_clifford_code_reading_has_no_limit_on_gadget_parities():
    # 32 qubits, 17 independent parities read by pi/2 gadgets: above the old limit of 14 (it raised
    # NotImplementedError), now Gaussian elimination
    g = _isometry(32, 1, 160, random.Random(100 * 32 + 2))
    code = code_of(g, generators='echelon')
    assert (code.n, code.k) == (32, 1) and len(code.stabilizers) == 31
    assert iap_normal_form(g, generators='echelon').k == 1


def test_iap_normal_form_does_not_depend_on_the_generators():
    rng = random.Random(5)
    for _ in range(20):
        n = rng.randint(3, 9)
        g = _isometry(n, rng.randint(0, min(2, n - 1)), rng.randint(n, 4 * n), rng)
        assert iap_normal_form(g.copy()) == iap_normal_form(g.copy(), generators='echelon')


@pytest.mark.parametrize('build', [_shor, F.fold_transversal_h, _switch, _teleport_cx])
def test_tableau_names_the_case_studies_as_the_matrix_does(build):
    act = build().logical_action()
    name, _correction = act.tableau().name()
    assert {b.operation for b in act.branches()} == {name}


def test_fold_h_tableau_gives_the_pauli_frame_as_a_formula_in_the_outcomes():
    tab = F.fold_transversal_h().logical_action().tableau()
    name, correction = tab.name()
    assert name == 'H'
    assert correction.startswith('X^(') and ' Z^(' in correction and correction.endswith(' on 0')
    assert {str(r) for r in tab.x_images} | {str(r) for r in tab.z_images} != {'+X', '+Z'}   # signed by outcomes


def test_tableau_names_gates_beyond_three_logical_qubits():
    # k = 6 repetition codes, one round of checks, then a transversal logical CNOT inside each pair of codes;
    # the matrix route names gates on at most three logical qubits
    from pyzx.circuit.gates import Measurement
    from zxqec.qprogram import QProgram
    k, d = 6, 3
    n_data, n_anc = k * d, k * (d - 1)
    blocks = [list(range(i * d, (i + 1) * d)) for i in range(k)]
    ancs = [list(range(n_data + i * (d - 1), n_data + (i + 1) * (d - 1))) for i in range(k)]
    p = QProgram(n_data + n_anc, bits=n_anc, initial_states={a: '0' for a in range(n_data, n_data + n_anc)})
    c = Circuit(n_data + n_anc, bit_amount=n_anc)
    bit = 0
    for b, a in zip(blocks, ancs):
        for anc, (d0, d1) in zip(a, zip(b, b[1:])):
            c.add_gate(HAD(anc)); c.add_gate(CNOT(anc, d0)); c.add_gate(CNOT(anc, d1)); c.add_gate(HAD(anc))
            c.add_gate(Measurement(anc, bit)); bit += 1
    p.add_circuit(c, name='checks')
    gate = Circuit(n_data + n_anc, bit_amount=n_anc)
    for i in range(0, k, 2):
        for q0, q1 in zip(blocks[i], blocks[i + 1]):
            gate.add_gate(CNOT(q0, q1))
    p.add_circuit(gate, name='gate')
    lev = p.group_stages([(0, 0, 'checks', 'syndrome extraction'), (1, 1, 'gate', 'logical gate')], one_level=True)
    name, correction = p.logical_action(lev, 1).tableau().name()
    assert name == ' (x) '.join('CNOT(0->1) on logical qubits {},{}'.format(i, i + 1) for i in range(0, k, 2))
    assert correction == ''


def test_tableau_refuses_a_non_clifford_gate():
    with pytest.raises(NotCliffordError):
        steane_t_injection().logical_action().tableau()


def _plug(cp):
    u = cp.clifford.copy()
    outs = list(u.outputs())
    for b, (a, kind) in zip(outs[cp.k_out:], cp.phases):
        u.set_type(b, VertexType.Z if kind == 'Z' else VertexType.X)
        u.set_phase(b, a)
    u.set_outputs(tuple(outs[:cp.k_out]))
    return u


def _zero(g):
    names = boolean_variables(g)
    return g.substitute_variables({v: 0 for v in names}) if names else g


def test_clifford_part_of_transversal_t_on_the_15_qubit_code():
    from pyzx.css import generate_css_encoder_graph
    from pyzx.linalg import Mat2
    sx = Mat2([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
               [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    enc, _ = generate_css_encoder_graph(sx, Mat2([[1] * 15]), 'Z-X')
    enc.auto_detect_io()
    code = code_of(enc)
    tt = Circuit(15)
    for q in range(15):
        tt.add_gate(T(q))
    act = logical_action(tt.to_graph(), code, code, via='encoder')
    cp = act.clifford_part()
    assert cp.phases == [(Fraction(7, 4), 'Z')]          # one T-dagger effect
    assert cp.is_isometry
    assert [str(p) for p in cp.tableau.x_images] == ['+XX'] and [str(p) for p in cp.tableau.z_images] == ['+ZI']
    assert [str(p) for p in cp.tableau.stabilizers] == ['+ZZ']
    assert same_map(_zero(act._modulo_detectors()), _zero(_plug(cp)))


@pytest.mark.parametrize('build', [steane_t_injection, lambda: _ccz([GHZ, FROM_X, FROM_Y, FROM_Z, P])])
def test_clifford_part_is_an_isometry_and_gives_the_gate_back(build):
    act = build().logical_action()
    cp = act.clifford_part()
    assert cp.phases and all(a % Fraction(1, 2) != 0 for a, _k in cp.phases)
    assert cp.is_isometry and cp.normal_form is not None
    assert same_map(_zero(act._modulo_detectors()), _zero(_plug(cp)))


def test_clifford_part_of_random_clifford_plus_t_unitaries():
    r = random.Random(2)
    for n in (2, 3, 4):
        c = Circuit(n)
        for _ in range(12):
            x = r.random()
            q = r.randrange(n)
            a, b = r.sample(range(n), 2)
            c.add_gate(T(q) if x < .3 else HAD(q) if x < .5 else S(q) if x < .6 else CNOT(a, b) if x < .85
                       else CZ(a, b))
        g = c.to_graph()
        cp = clifford_plus_phases(g)
        assert cp.is_isometry
        assert same_map(g, _plug(cp))
