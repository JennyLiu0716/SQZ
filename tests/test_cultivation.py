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
    from fractions import Fraction
    from pyzx.symbolic import new_var
    from pyzx.utils import VertexType
    from zxqec.gpf import (boolean_variables, change_variables, consistent_assignments, equal_for_all_assignments,
                           phases_modulo, restore_boundary_hadamards, to_gpf)
    p, _circ = program
    g = p.quantum_graph()
    res = reduce_gpf(g)
    names = sorted(set(boolean_variables(g)) | {str(v) for c in res.constraints for v in c.free_vars()})
    assert len(consistent_assignments(res.constraints, names)) == 32
    code = p.code_at(p.num_stages - 1).data()
    steps = [restore_boundary_hadamards(to_gpf(g)), res.to_graph(),
             change_variables(res.graph, [list(code.logical_x[0].x)] + [list(r) for r in code.sx])]
    expected = code.encoder()
    (b,) = list(expected.inputs())
    (z,) = list(expected.neighbors(b))
    expected.remove_vertex(b)
    expected.set_inputs(())
    phase = Fraction(7, 4)
    for name in ['c[4]', 'c[5]', 'c[6]', 'c[7]', 'c[14]']:
        phase = phase + new_var(name, True, expected.var_registry)
    expected.set_phase(z, expected.phase(z) + phase)
    for h in steps + [expected]:
        assert equal_for_all_assignments(g, h, names, res.constraints)
    # and step 4 and the expected diagram have the same spiders, edges and phases (Appendix E of the paper)
    def wiring(h):
        h = phases_modulo(h, res.constraints)
        outs = [next(iter(h.neighbors(b))) for b in h.outputs()]
        zs = [v for v in h.vertices() if h.type(v) == VertexType.Z and not is_leaf(h, v)]
        return (sorted((str(h.phase(z)), tuple(int(h.connected(z, w)) for w in outs)) for z in zs),
                [str(h.phase(w)) for w in outs])
    assert wiring(steps[2]) == wiring(expected)


def test_in_the_codes_variables_the_gadgets_have_lias_block_form(program):
    # Part 4, step 2 of examples/Cultivation.ipynb: in the variables L | S1 S2 S3 | h1 h2 the 7 T gates
    # read the code's columns (P_L P_S), the 7 T-dagger gates the same columns plus h1 + h2, and all 15
    # add up to one T-dagger gadget on L (H = (1)) plus Pauli terms in c[14]
    from fractions import Fraction
    from pyzx.utils import VertexType
    from zxqec.gpf import change_variables, phase_polynomial, to_gpf
    from zxqec.gpf.gadgets import _xor_terms
    p, _circ = program
    code = p.code_at(p.num_stages - 1).data()
    rows = [list(code.logical_x[0].x)] + [list(r) for r in code.sx]
    m = change_variables(to_gpf(p.quantum_graph()), rows, fuse=False)
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


def test_step_4_on_the_triangle_of_the_steane_code(program):
    # Part 4 of examples/Cultivation.ipynb (in 3D) and Appendix E of the paper: of the 7! orderings of the
    # code qubits onto the textbook triangle, six put the Z-spiders wired to four qubits on the three faces;
    # with the first, step 4 in the run where every outcome is 0 is T-dagger|+> in the textbook Steane code
    # built from the triangle alone, whichever weight-3 logical X its logical spider is wired to
    import itertools
    from fractions import Fraction
    import pyzx as zx
    from pyzx.graph import Graph
    from pyzx.utils import VertexType
    from zxqec.gpf import boolean_variables, change_variables
    faces = [(0, 3, 4, 5), (1, 3, 4, 6), (2, 5, 4, 6)]
    p, _circ = program
    g = p.quantum_graph()
    res = reduce_gpf(g)
    code = p.code_at(p.num_stages - 1).data()
    step4 = change_variables(res.graph, [list(code.logical_x[0].x)] + [list(r) for r in code.sx])
    outs = [next(iter(step4.neighbors(b))) for b in step4.outputs()]
    wired = [frozenset(j for j, w in enumerate(outs) if step4.connected(z, w))
             for z in step4.vertices() if step4.type(z) == VertexType.Z]
    orderings = [perm for perm in itertools.permutations(range(7))          # output j -> triangle qubit perm[j]
                 if {frozenset(perm[j] for j in w) for w in wired if len(w) == 4} == {frozenset(f) for f in faces}]
    assert len(orderings) == 6
    output_of = {q: j for j, q in enumerate(orderings[0])}

    def textbook(side):
        t = Graph()
        xs, bs = [], []
        for j in range(7):
            xs.append(t.add_vertex(VertexType.X))
            bs.append(t.add_vertex(VertexType.BOUNDARY))
            t.add_edge((xs[j], bs[j]))
        for spider_qubits, phase in [(side, Fraction(7, 4))] + [(f, 0) for f in faces]:
            z = t.add_vertex(VertexType.Z, phase=phase)
            for q in spider_qubits:
                t.add_edge((z, xs[output_of[q]]))
        t.set_inputs(())
        t.set_outputs(tuple(bs))
        return t

    variables = set(boolean_variables(g)) | {str(v) for c in res.constraints for v in c.free_vars()}
    zero = step4.substitute_variables({n: 0 for n in variables})
    sides = [s for s in itertools.combinations(range(7), 3) if all(len(set(s) & set(f)) % 2 == 0 for f in faces)]
    assert len(sides) == 7
    assert all(zx.compare_tensors(zero, textbook(s), preserve_scalar=False) for s in sides)


def test_the_double_check_on_its_own(program):
    # examples/CultivationDoubleCheck.ipynb and Appendix E of the paper. The double-check block, without the
    # discard of helper qubit 1's old state by its reset (a separate piece of the block's diagram), maps the
    # seven code qubits to themselves; its detectors are the file's last six DETECTOR lines; reduced, it has a
    # T-dagger gadget on each input and a T gadget reading exactly what each output reads; and it equals, up to a
    # scalar, Z_6^c24 (I + (-1)^c24 H^7) with H = (X + Y)/sqrt(2), in every run its detectors allow
    from fractions import Fraction
    from functools import reduce
    import pyzx as zx
    from pyzx.utils import VertexType
    from zxqec.gpf import boolean_variables, consistent_assignments
    from zxqec.spacetime import get_logical_qubit
    p, circ = program
    level = p.group_stages(GROUPS)
    g = p.block_graph(level, 3, classical=False)
    left, pieces = set(g.vertices()), []
    while left:
        stack, piece = [left.pop()], set()
        while stack:
            v = stack.pop()
            if v not in piece:
                piece.add(v)
                stack.extend(g.neighbors(v))
        left -= piece
        pieces.append(piece)
    (discard,) = [x for x in pieces if not x & set(g.outputs())]
    assert [get_logical_qubit(g, b) for b in g.inputs() if b in discard] == [1]
    g.set_inputs(tuple(b for b in g.inputs() if b not in discard))
    g.remove_vertices(list(discard))
    qubits = [get_logical_qubit(g, b) for b in g.inputs()]
    assert qubits == [get_logical_qubit(g, b) for b in g.outputs()] == [0, 3, 6, 8, 9, 11, 14]

    res = reduce_gpf(g)
    assert ({frozenset(d) for d in circ.detectors[-6:]}
            == {frozenset(v.name for v in d.free_vars()) for d in res.detectors})
    h = res.graph
    on_input = {next(iter(h.neighbors(b))) for b in h.inputs()}
    read_by_output = {frozenset(w for w in h.neighbors(next(iter(h.neighbors(b)))) if h.type(w) == VertexType.Z)
                      for b in h.outputs()}
    gadgets = []
    for leaf in (v for v in h.vertices() if is_leaf(h, v)):
        (hub,) = list(h.neighbors(leaf))
        gadgets.append((h.phase(leaf), frozenset(w for w in h.neighbors(hub) if w != leaf)))
    t_dag = [r for a, r in gadgets if a == Fraction(7, 4)]
    t = [r for a, r in gadgets if a == Fraction(1, 4)]
    assert len(gadgets) == 14 and len(t_dag) == 7 and len(t) == 7
    assert {r for r in t_dag} == {frozenset([z]) for z in on_input}
    assert set(t) == read_by_output

    i2, x, y, z = np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])
    hxy = reduce(np.kron, [(x + y) / np.sqrt(2)] * 7)
    names = sorted(set(boolean_variables(g)) | {str(v) for c in res.constraints for v in c.free_vars()})
    runs = consistent_assignments(res.constraints, names)
    assert len(runs) == 2
    for a in runs:
        c24 = a['c[24]']
        expected = reduce(np.kron, [z if (q == 6 and c24) else i2 for q in qubits]) @ (np.eye(2 ** 7) + (-1) ** c24 * hxy)
        block = zx.tensor_to_matrix(zx.tensorfy(g.substitute_variables(a), preserve_scalar=False), 7, 7)
        assert zx.compare_tensors(block, expected, preserve_scalar=False)
