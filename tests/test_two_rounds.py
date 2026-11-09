"""The paper's running example, step by step (examples/TwoRoundsStepByStep.ipynb).

Two rounds of repetition-code checks with a Pauli error between them. Each step is checked twice: its
output is what the notebook prints, and it agrees with a calculation that does not take the tool's own
route -- tensors of the circuit's diagram composed with PyZX, or a state-vector simulation of round 2
that does not use ZX at all (tests/statevector.py).
"""
import itertools

import numpy as np
import pytest
from pyzx.circuit import Circuit
from pyzx.utils import VertexType

from zxqec.codes import PauliString, stabilizes
from zxqec.gpf import (boolean_variables, consistent_assignments, equal_for_all_assignments, evaluate_phase,
                       reduce_gpf, to_gpf_steps)
from zxqec.qprogram import QProgram
from zxqec.spacetime.subdiagram import get_logical_qubit

from .statevector import PAULI, H, X, Z, apply1, apply_cx

HEADER = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\n'
          'creg c[2];\ncreg d[2];\n')
ROUND = ('h q[3]; h q[4];\ncx q[3], q[0]; cx q[3], q[1];\ncx q[4], q[1]; cx q[4], q[2];\nh q[3]; h q[4];\n'
         'measure q[3] -> c[0];\nmeasure q[4] -> c[1];\n')
CORRECTION = 'if(c==1) z q[0];\nif(c==3) z q[1];\nif(c==2) z q[2];\n'
ROUND_2 = 'reset q[3]; reset q[4];\n' + ROUND.replace('c[', 'd[')
CORRECTION_2 = CORRECTION.replace('if(c', 'if(d')
QASM = {'X': 'x q[{0}];\n', 'Y': 'z q[{0}]; x q[{0}];\n', 'Z': 'z q[{0}];\n'}
ERRORS = {'none': []}
ERRORS.update({l + str(q): [(l, q)] for q in range(3) for l in 'XYZ'})
ERRORS.update({l1 + str(q1) + l2 + str(q2): [(l1, q1), (l2, q2)]
               for q1, q2 in itertools.combinations(range(3), 2) for l1 in 'XYZ' for l2 in 'XYZ'})

# what the notebook prints for E = Z0 Z1
CODES = [
    "[[3, 1, 1]] CSS code on qubits 0, 1, 2\n    stabilizer (-1)^(c[0])XXI\n    stabilizer (-1)^(c[1])IXX\n"
    "    logical X XII   logical Z ZZZ",
    "[[3, 1, 1]] CSS code on qubits 0, 1, 2\n    stabilizer +XXI\n    stabilizer +XIX\n"
    "    logical X XII   logical Z ZZZ",
    "[[3, 1, 1]] CSS code on qubits 0, 1, 2\n    stabilizer +XXI\n    stabilizer -XIX\n"
    "    logical X XII   logical Z ZZZ",
    "[[3, 1, 1]] CSS code on qubits 0, 1, 2\n    stabilizer +XXI\n    stabilizer -XIX\n"
    "    logical X XII   logical Z ZZZ",
    "[[3, 1, 1]] CSS code on qubits 0, 1, 2\n    stabilizer (-1)^(d[0])XXI\n    stabilizer (-1)^(d[1] + 1)IXX\n"
    "    logical X XII   logical Z ZZZ",
]
# the notebook's last table: (what is left, what round 2 reads) -> errors
TABLE = {
    ('I', '00'): ['none', 'X0X1', 'X0X2', 'X1X2'], ('I', '01'): ['Z2', 'X0Y2', 'X1Y2'],
    ('I', '10'): ['Z0', 'Y0X1', 'Y0X2'], ('I', '11'): ['Z1', 'X0Y1', 'Y1X2'],
    ('X', '00'): ['X0', 'X1', 'X2'], ('X', '01'): ['Y2', 'X0Z2', 'X1Z2'],
    ('X', '10'): ['Y0', 'Z0X1', 'Z0X2'], ('X', '11'): ['Y1', 'X0Z1', 'Z1X2'],
    ('Y', '01'): ['Y0Z1', 'Z0Y1'], ('Y', '10'): ['Y1Z2', 'Z1Y2'], ('Y', '11'): ['Y0Z2', 'Z0Y2'],
    ('Z', '01'): ['Y0Y1', 'Z0Z1'], ('Z', '10'): ['Y1Y2', 'Z1Z2'], ('Z', '11'): ['Y0Y2', 'Z0Z2'],
}

# the code states of C_in (stabilizers X0X1 and X0X2, logical Z = ZZZ, logical X = X0): even and odd parity
ZERO_BAR = np.zeros((2, 2, 2), complex)
for bits in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
    ZERO_BAR[bits] = 0.5
ONE_BAR = np.flip(ZERO_BAR, axis=0).copy()                 # X on qubit 0


def _program(error):
    p = QProgram(5, bits=4, initial_states={3: '0', 4: '0'})
    text = ''.join(QASM[l].format(q) for l, q in error)
    for name, piece in [('round 1', ROUND), ('correct 1', CORRECTION), ('error', text),
                        ('round 2', ROUND_2), ('correct 2', CORRECTION_2)]:
        p.add_circuit(Circuit.from_qasm(HEADER + piece), name=name)
    return p


def _action(p):
    return p.logical_action(p.group_stages([(0, 1, 'round 1'), (2, 4, 'error + round 2')]), 1)


@pytest.fixture(scope='module')
def example():
    p = _program(ERRORS['Z0Z1'])
    return p, _action(p)


def _signed(stabilizer, run):
    """``stabilizer`` with its sign evaluated in ``run``."""
    s = stabilizer.sign
    value = evaluate_phase(s, run) if hasattr(s, 'free_vars') else int(s)
    return PauliString.from_str(('-' if value % 2 else '+') + stabilizer.letters)


def _at(g, run):
    names = boolean_variables(g)
    return g.substitute_variables({n: run[n] for n in names}) if names else g.copy()


def _runs(names):
    return [dict(zip(names, bits)) for bits in itertools.product((0, 1), repeat=len(names))]


def _pauli(letters):
    m = np.eye(1, dtype=complex)
    for l in letters:
        m = np.kron(m, PAULI[l])
    return m


def _proportional(a, b):
    k = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    return abs(b[k]) > 1e-9 and np.allclose(a, a[k] / b[k] * b, atol=1e-9) and abs(abs(a[k] / b[k]) - 1) < 1e-9


# -- Question 1 --------------------------------------------------------------------------------------

def test_question_1_the_code_after_every_stage(example):
    p, _ = example
    assert [str(p.code_at(s)).strip() for s in range(p.num_stages)] == CODES
    # every stabilizer, with its sign, leaves the circuit's own diagram unchanged in every run
    for s in range(p.num_stages):
        g, code = p.prefix_graph(s), p.code_at(s)
        for run in _runs(boolean_variables(g)):
            h = _at(g, run)
            for stab in code.stabilizers:
                assert stabilizes(h, _signed(stab, run))


# -- Question 2, step 1: the encoder, the decoder and the composite -----------------------------------

def test_the_encoder_of_c_in(example):
    p, _ = example
    E = p.code_at(1).encoder()
    assert sum(1 for v in E.vertices() if E.type(v) != VertexType.BOUNDARY) == 6 and E.num_edges() == 9
    M = np.asarray(E.to_matrix(preserve_scalar=True))
    assert np.allclose(M.conj().T @ M, np.eye(2))                                  # an isometry
    assert np.allclose(M @ X, _pauli('XII') @ M)                                   # E o X = Xbar o E
    assert np.allclose(M @ Z, _pauli('ZZZ') @ M)                                   # E o Z = Zbar o E
    for stab in ('XXI', 'XIX'):
        assert np.allclose(_pauli(stab) @ M, M)                                    # onto the code space
    expected = np.stack([ZERO_BAR.reshape(8), ONE_BAR.reshape(8)], axis=1)
    assert _proportional(M, expected)                                              # the one encoder, up to phase


def test_the_decoder_of_c_out(example):
    p, _ = example
    code = p.code_at(4)
    D = code.encoder().adjoint()
    phases = sorted(str(D.phase(v)) for v in D.vertices() if D.type(v) == VertexType.Z and str(D.phase(v)) != '0')
    assert phases == ['d[0]', 'd[1] + 1']                                          # the signs of C_out
    for run in _runs(['d[0]', 'd[1]']):
        M = np.asarray(_at(code.encoder(), run).to_matrix(preserve_scalar=True))
        Mdag = np.asarray(_at(D, run).to_matrix(preserve_scalar=True))
        assert np.allclose(Mdag, M.conj().T)                                       # the adjoint
        assert np.allclose(M.conj().T @ M, np.eye(2))
        assert np.allclose(M @ X, _pauli('XII') @ M) and np.allclose(M @ Z, _pauli('ZZZ') @ M)
        for stab in code.stabilizers:
            s = _signed(stab, run)
            assert np.allclose((-1) ** int(s.sign) * _pauli(s.letters) @ M, M)
    # in the run that occurs (d[0] = 0, d[1] = 1) both signs are +: the same code space as C_in
    M = np.asarray(_at(code.encoder(), {'d[0]': 0, 'd[1]': 1}).to_matrix(preserve_scalar=True))
    assert _proportional(M, np.asarray(p.code_at(1).encoder().to_matrix(preserve_scalar=True)))


def test_the_composite_is_decoder_block_encoder(example):
    p, a = example
    B = a.block
    assert [get_logical_qubit(B, b) for b in B.inputs()] == [0, 1, 2]
    assert [get_logical_qubit(B, b) for b in B.outputs()] == [0, 1, 2]
    assert B.num_vertices() == 29 and a.composite.num_vertices() == 37
    g = p.code_at(1).encoder()
    g.compose(B.copy())
    g.compose(p.code_at(4).encoder().adjoint())
    assert equal_for_all_assignments(a.composite, g)                               # PyZX's compose, not glue


# -- Question 2, step 2: reduce ----------------------------------------------------------------------

def test_the_reduction_and_its_detectors(example):
    _, a = example
    assert [g.num_vertices() for _what, g in to_gpf_steps(a.composite.copy())] == [37, 19, 15, 15, 4, 4]
    assert a.graph.num_vertices() == 4
    assert sorted(str(d) for d in a.detectors) == ['d[0]', 'd[1] + 1']
    # the detectors are right: the composite is the zero map exactly in the runs they forbid (with its scalar:
    # PyZX's tensor without the scalar can be non-zero for a zero map)
    for run in _runs(['d[0]', 'd[1]']):
        t = np.asarray(_at(a.composite, run).to_tensor(preserve_scalar=True))
        assert (np.abs(t).max() > 1e-9) == (run == {'d[0]': 0, 'd[1]': 1})
    assert equal_for_all_assignments(a.composite, a.graph, constraints=a.detectors)


# -- Question 2, steps 3 and 4: name, check ----------------------------------------------------------

def test_one_class_named_identity_with_correction_z(example):
    _, a = example
    (b,) = a.branches()
    assert b.when() == 'in every run' and b.outcomes == {'d[0]': 0, 'd[1]': 1}
    assert (b.operation, b.correction) == ('I', 'Z')
    assert np.allclose(np.round(a.matrix(b.outcomes).real, 9) + 0.0, [[1, 0], [0, -1]])
    # independently: the composite itself, in that run, is a multiple of Z
    m = np.asarray(_at(a.composite, b.outcomes).to_matrix(preserve_scalar=False))
    assert _proportional(m / np.linalg.norm(m, 2), Z)
    assert a.verdict() == ('deterministic up to a Pauli frame: every run implements I followed by a Pauli '
                           'correction (Z) that the outcomes determine')


def test_the_block_keeps_the_code(example):
    p, a = example
    assert a.stays_in_code() is True
    # independently: in the run that occurs, every stabilizer of C_out with its sign fixes B E_in
    run = {'d[0]': 0, 'd[1]': 1}
    g = p.code_at(1).encoder()
    g.compose(a.block.copy())
    h = _at(g, run)
    for stab in p.code_at(4).stabilizers:
        assert stabilizes(h, _signed(stab, run))


# -- Question 3 --------------------------------------------------------------------------------------

def test_question_3_the_runs_and_the_verdict(example):
    p, a = example
    detectors = reduce_gpf(p.quantum_graph()).detectors
    assert sorted(str(d) for d in detectors) == ['d[0]', 'd[1] + 1']
    runs = consistent_assignments(detectors, ['c[0]', 'c[1]', 'd[0]', 'd[1]'])
    assert sorted((r['c[0]'], r['c[1]']) for r in runs) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all((r['d[0]'], r['d[1]']) == (0, 1) for r in runs)                     # round 1 random, round 2 fixed
    assert a.is_deterministic()


# -- all 36 errors, against a state-vector simulation of round 2 -------------------------------------

def _simulate(error, logical):
    """Round 2 and its correction on the code state ``logical`` after ``error``, without ZX.

    Returns what round 2 reads (both outcomes must be certain) and the data qubits afterwards.
    """
    state = np.zeros((2,) * 5, complex)
    state[:, :, :, 0, 0] = logical                                                # ancillas in |0>
    for l, q in error:
        for u in ((Z, X) if l == 'Y' else (PAULI[l],)):                          # y as in QASM: z, then x
            state = apply1(state, u, q)
    for q in (3, 4):
        state = apply1(state, H, q)
    for c, t in [(3, 0), (3, 1), (4, 1), (4, 2)]:
        state = apply_cx(state, c, t)
    for q in (3, 4):
        state = apply1(state, H, q)
    reads = []
    for q in (3, 4):
        p1 = float(np.sum(np.abs(np.take(state, 1, axis=q)) ** 2))
        assert min(p1, 1 - p1) < 1e-9                                             # a certain outcome
        reads.append(int(round(p1)))
    data = state[:, :, :, reads[0], reads[1]]
    fix = {(1, 0): 0, (1, 1): 1, (0, 1): 2}.get(tuple(reads))
    if fix is not None:
        data = apply1(data, Z, fix)
    return '{}{}'.format(*reads), data


def test_all_36_errors_against_a_state_vector_simulation():
    table = {}
    for name, error in ERRORS.items():
        reads, out0 = _simulate(error, ZERO_BAR)
        _, out1 = _simulate(error, ONE_BAR)
        L = np.array([[np.vdot(bar, out) for out in (out0, out1)] for bar in (ZERO_BAR, ONE_BAR)])
        assert np.allclose(np.abs(np.linalg.det(L)), 1)                          # stays in the code
        (left,) = [l for l in 'IXYZ' if _proportional(L, PAULI[l])]
        p = _program(error)
        a = _action(p)
        tool_reads = ','.join('{}{}'.format(r['d[0]'], r['d[1]'])
                              for r in consistent_assignments(reduce_gpf(p.quantum_graph()).detectors,
                                                              ['d[0]', 'd[1]']))
        assert tool_reads == reads, name
        assert a.stays_in_code() is True and a.is_deterministic(), name
        assert {(b.operation, b.correction or 'I') for b in a.branches()} == {('I', left)}, name
        table.setdefault((left, reads), []).append(name)
    assert len(ERRORS) - 1 == 36 and table == TABLE
