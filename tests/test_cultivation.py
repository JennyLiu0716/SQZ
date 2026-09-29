"""Magic-state cultivation (d = 3), from the circuit of Wan, Zhong and Zapirain in Stim's format.

The claims of the tool are checked against the detectors the authors wrote by hand and against an
independent state-vector simulation of the same circuit.
"""
import os

import numpy as np
import pytest

from zxqec.codes.gf2 import as_matrix, rank
from zxqec.gpf import evaluate_phase, is_leaf, reduce_gpf
from zxqec.qprogram import program_from_stim

from .statevector import expectation, run

PATH = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'cultivation', 'd3_cultivation.stim')
GROUPS = [(0, 0, 'first checks', 'state preparation'), (1, 1, 'inject T', 'injection'),
          (2, 2, 'checks', 'syndrome extraction'), (3, 4, 'double-check', 'cultivation')]


@pytest.fixture(scope='module')
def program():
    with open(PATH) as f:
        return program_from_stim(f.read())


def _span(polys, names):
    rows = []
    for d in polys:
        v = np.zeros(len(names) + 1, dtype=np.uint8)
        if isinstance(d, list):
            for n in d:
                v[names.index(n)] ^= 1
        else:
            for coeff, term in d.terms:
                if int(coeff) % 2:
                    if term.vars:
                        v[names.index(term.vars[0][0].name)] ^= 1
                    else:
                        v[-1] ^= 1
        rows.append(v)
    return as_matrix(rows, len(names) + 1)


def _measurement_detectors(res, n):
    """Our detectors on measurement outcomes, leaving out the two about PyZX's reset variables."""
    out = [d for d in res.detectors if all(str(v).startswith('c[') for v in d.free_vars())]
    assert len(res.detectors) - len(out) == 2                 # s1_r0 = 0, s3_r0 = 0: resets of |0>
    return out


def test_derived_detectors_equal_the_declared_ones(program):
    p, circ = program
    res = reduce_gpf(p.quantum_graph())
    names = ['c[{}]'.format(i) for i in range(circ.measurements)]
    ours, theirs = _span(_measurement_detectors(res, circ.measurements), names), _span(circ.detectors, names)
    assert rank(theirs) == 26
    assert rank(ours) == rank(theirs) == rank(np.concatenate([ours, theirs], 0))


def test_without_the_gadget_layer_the_double_check_detectors_are_missed(program):
    p, circ = program
    res = reduce_gpf(p.quantum_graph(), gadgets=False)
    names = ['c[{}]'.format(i) for i in range(circ.measurements)]
    ours = _span(_measurement_detectors(res, circ.measurements), names)
    missed = [i for i, d in enumerate(circ.detectors)
              if rank(np.concatenate([ours, _span([d], names)], 0)) > rank(ours)]
    assert len(missed) == 2
    assert all('c[24]' in circ.detectors[i] for i in missed)   # both involve the double-check outcome


def test_the_circuit_reduces_to_one_t_dagger_gadget(program):
    p, _ = program
    h = reduce_gpf(p.quantum_graph()).graph
    leaves = [v for v in h.vertices() if is_leaf(h, v)]
    assert [str(h.phase(v)) for v in leaves] == ['7/4']


def test_code_and_logical_state(program):
    p, _ = program
    code = p.code_at(p.num_stages - 1).data()
    assert code.parameters == (7, 1, 3) and code.non_clifford
    act = p.logical_action()
    assert act.is_deterministic() and act.operations() == {'prepares T^dag|+>'}
    assert {b.correction for b in act.branches()} == {'', 'Z'}


def test_level_up_table(program):
    p, _ = program
    s = p.summary(p.group_stages(GROUPS))
    assert s.ok
    assert s.blocks[3].action.measured_basis() == ('T^dag|+>', 'T^dag|->')
    whole = p.summary(p.level_up(p.group_stages(GROUPS), 0, 3, name='cultivation', role='state preparation'))
    assert whole.ok and [b.operation() for b in whole.blocks] == ['prepares T^dag|+>']


def test_independent_simulation_agrees(program):
    p, circ = program
    code = p.code_at(p.num_stages - 1).data()
    act = p.logical_action()
    data = [int(q) for q in code.qubits]
    lx, lz = code.logical_x[0].letters, code.logical_z[0].letters
    ly = ''.join({('X', 'Z'): 'Y', ('X', 'I'): 'X', ('I', 'Z'): 'Z', ('I', 'I'): 'I'}[(x, z)] for x, z in zip(lx, lz))
    # i X Z = Y per qubit, times i^(number of Y)... the overall sign of i*Xbar*Zbar:
    n_y = ly.count('Y')
    sign_y = (1j * (-1j) ** n_y).real
    t_dag_plus = np.array([np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0])
    (frame_parity, _v), = act.branches()[1].conditions
    rng = np.random.default_rng(5)
    seen = set()
    for _ in range(8):
        state, outs = run(circ, rng)
        a = {'c[{}]'.format(i): o for i, o in enumerate(outs)}
        for d in circ.detectors:                                   # the authors' detectors hold
            assert sum(outs[int(n[2:-1])] for n in d) % 2 == 0
        for s in code.stabilizers:                                  # the tool's stabilizers and signs
            want = (-1) ** (s.sign if isinstance(s.sign, int) else evaluate_phase(s.sign, a))
            assert abs(expectation(state, s.letters, data) - want) < 1e-9
        bloch = np.array([expectation(state, lx, data).real, sign_y * expectation(state, ly, data).real,
                          expectation(state, lz, data).real])
        flip = sum(a[v.strip()] for v in frame_parity.split('+')) % 2
        seen.add(flip)
        assert np.allclose(bloch, t_dag_plus * (np.array([-1, -1, 1]) if flip else 1), atol=1e-9)
    assert seen == {0, 1}                                          # both classes of runs were sampled


def test_each_drawn_step_equals_the_circuit_and_ends_at_the_expected_diagram(program):
    # Part 4 of examples/Cultivation.ipynb: every picture is checked against the circuit diagram
    # by tensor contraction, in every run the detectors allow, and the last one is the expected
    # diagram (T-dagger|+> into the encoder of the code, with the Z correction) spider for spider
    import sys
    from fractions import Fraction
    import pyzx as zx
    from zxqec.gpf import boolean_variables, restore_boundary_hadamards, to_gpf
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
    from zx_steps import allowed_runs, encoded_state, encoder_form, layer_rows, output_phases
    p, _circ = program
    g = p.quantum_graph()
    res = reduce_gpf(g)
    names = sorted(set(boolean_variables(g)) | {str(v) for c in res.constraints for v in c.free_vars()})
    runs = allowed_runs(res.constraints, names)
    assert len(runs) == 32
    code = p.code_at(p.num_stages - 1).data()
    steps = [restore_boundary_hadamards(to_gpf(g)), res.to_graph(),
             encoder_form(res.graph, [list(code.logical_x[0].x)] + [list(r) for r in code.sx])]
    expected = encoded_state(code, Fraction(7, 4), ['c[4]', 'c[5]', 'c[6]', 'c[7]', 'c[14]'])
    for h in steps + [expected]:
        assert all(zx.compare_tensors(g.substitute_variables(a), h.substitute_variables(a), preserve_scalar=False)
                   for a in runs)
    assert layer_rows(steps[2], res.constraints) == layer_rows(expected, res.constraints)
    assert output_phases(steps[2], res.constraints) == output_phases(expected, res.constraints)


def test_in_the_codes_variables_the_gadgets_have_lias_block_form(program):
    # Part 4, step 2 of examples/Cultivation.ipynb: in the variables L | S1 S2 S3 | h1 h2 the 7 T gates
    # read the code's columns (P_L P_S), the 7 T-dagger gates the same columns plus h1 + h2, and all 15
    # add up to one T-dagger gadget on L (H = (1)) plus Pauli terms in c[14]
    import sys
    from fractions import Fraction
    from pyzx.utils import VertexType
    from zxqec.gpf import phase_polynomial, to_gpf
    from zxqec.gpf.gadgets import _xor_terms
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
    from zx_steps import encoder_form
    p, _circ = program
    code = p.code_at(p.num_stages - 1).data()
    rows = [list(code.logical_x[0].x)] + [list(r) for r in code.sx]
    m = encoder_form(to_gpf(p.quantum_graph()), rows, fuse=False)
    zs = sorted(v for v in m.vertices() if m.type(v) == VertexType.Z and not is_leaf(m, v))
    assert len(zs) == 6                                          # L, S1, S2, S3 and two hidden variables
    gadgets = []
    for leaf in (v for v in m.vertices() if is_leaf(m, v)):
        (hub,) = list(m.neighbors(leaf))
        gadgets.append((str(m.phase(leaf)), tuple(int(m.connected(hub, z)) for z in zs)))
    columns = {tuple(int(r[j]) for r in rows) for j in range(code.n)}
    assert {r[:4] for a, r in gadgets if a == '1/4'} == columns
    assert {r[:4] for a, r in gadgets if a == '7/4' and r[4:] == (1, 1)} == columns
    poly, _ = phase_polynomial(m)
    rest = {mono: c for mono, c in poly.items()
            if not (c == 1 and sum(isinstance(v, tuple) for v in mono) == 1)}
    assert rest == _xor_terms([('z', zs[0])], Fraction(7, 4))


def test_every_block_grouped_as_in_the_paper_stays_in_its_code(program):
    p, _circ = program
    s = p.summary(p.group_stages(GROUPS))
    assert all(b.action.stays_in_code() is True for b in s.blocks)
    assert p.logical_action().stays_in_code() is True
