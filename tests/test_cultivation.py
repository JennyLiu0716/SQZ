"""Magic-state cultivation (d = 3), from the circuit of Wan, Zhong and Zapirain in Stim's format.

The claims of the tool are checked against the detectors the authors wrote by hand and against an
independent state-vector simulation of the same circuit. The last tests run d = 5 cultivation with real
T gates (42 qubits, from haoliri0/SOFT): there the claims are checked against the declared detectors, the
authors' own final check (detectors and observable), and T-dag|+> in the colour code of the faces of that
check, by tensor contraction of the reduced diagram and of the whole circuit in one run (no simulation at
42 qubits).
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


# -- d = 5 with real T gates (Part 6 of examples/Cultivation.ipynb) ---------------------------------

PATH_D5 = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'cultivation', 'd5_cultivation.stim')


@pytest.fixture(scope='module')
def program_d5():
    with open(PATH_D5) as f:
        return program_from_stim(f.read())


def test_d5_codes_detectors_and_logical_state(program_d5):
    from zxqec.qprogram import check_detectors
    p, circ = program_d5
    assert (circ.qubits, circ.measurements, len(circ.detectors), p.num_stages) == (42, 93, 89, 8)
    codes = [[(b.parameters, b.non_clifford) for b in p.code_at(s).blocks()] for s in range(p.num_stages)]
    assert codes[0] == codes[2][:1] == [((7, 1, 3), True)]
    assert [k for (_n, k, _d), _nc in codes[2][1:]] == [0] * 6                  # six pairs for the growth
    assert all(codes[s] == [((19, 1, 5), True)] for s in (3, 4, 5, 7))
    chk = check_detectors(p, circ.detectors)
    assert chk.count('deterministic') == 89 and chk.derived == 89 and not chk.undeclared
    assert all(d.value == 0 for d in chk.declared)
    act = p.logical_action()
    assert act.is_deterministic() and act.operations() == {'prepares T^dag|+>'}
    assert {b.correction for b in act.branches()} == {''}


def test_d5_gadgets_in_the_codes_variables_are_one_t_dagger_on_the_logical(program_d5):
    # the growth feeds X corrections forward from c[21], c[23], c[25], which the detectors fix to 0; written
    # modulo the detectors and in the code's variables, the 53 T-type gadgets add up to 7pi/4 on L alone, and
    # the phases of the hidden variables are the detectors that need the T gates
    from fractions import Fraction
    from pyzx.utils import VertexType
    from zxqec.gpf import change_variables, phase_polynomial, phases_modulo
    p, circ = program_d5
    res = reduce_gpf(p.quantum_graph(), gadgets=False)
    assert {'c[21]', 'c[23]', 'c[25]'} <= {str(c) for c in res.constraints}
    code = p.code_at(p.num_stages - 1).data()
    rows = [list(code.logical_x[0].x)] + [list(r) for r in code.sx]
    m = change_variables(phases_modulo(res.graph, res.constraints), rows, fuse=False)
    zs = sorted(v for v in m.vertices() if m.type(v) == VertexType.Z and not is_leaf(m, v))
    poly, gadgets = phase_polynomial(m)
    assert len(gadgets) == 53 and len(zs) == 23
    assert {mono: c % 2 for mono, c in poly.items() if c % 2} == {frozenset([('z', zs[0])]): Fraction(7, 4)}
    names = ['c[{}]'.format(i) for i in range(circ.measurements)]
    on_outcomes = [d for d in res.detectors if all(str(v).startswith('c[') for v in d.free_vars())]
    ours = np.concatenate([_span(on_outcomes, names), _span([m.phase(z) for z in zs[len(rows):]], names)], 0)
    theirs = _span(circ.detectors, names)
    assert rank(_span(on_outcomes, names)) == 76
    assert rank(ours) == rank(theirs) == rank(np.concatenate([ours, theirs], 0)) == 89


# the faces that the first MPP of the authors' final check (d5_final_check.stim) measures
FACES5 = [[0, 3, 7, 11], [9, 13, 15, 19, 21, 27], [17, 25, 32, 36], [3, 7, 9, 13], [15, 21, 23, 29],
          [19, 25, 27, 32, 34, 38], [7, 11, 13, 17, 19, 25], [21, 27, 29, 34], [32, 36, 38, 40]]
SIDE5 = [0, 3, 9, 15, 23]
PATH_D5_CHECK = os.path.join(os.path.dirname(PATH_D5), 'd5_final_check.stim')


def test_d5_passes_the_authors_final_check():
    # the published file is the protocol followed by the final check: the 18 stabilizers measured by MPP, a DETECTOR
    # for each, then T gates and the product of Y, which ends OBSERVABLE_INCLUDE(0); all of it is deterministic
    from zxqec.qprogram import check_detectors
    with open(PATH_D5) as f, open(PATH_D5_CHECK) as g:
        p, circ = program_from_stim(f.read() + g.read())
    assert (circ.measurements, len(circ.detectors), len(circ.observables[0])) == (112, 107, 11)
    first_mpp = next(ins for ins in circ.instructions if ins.name == 'MPP')
    for letter in 'XZ':
        assert (sorted(sorted(int(f[1:]) for f in t.split('*')) for t in first_mpp.targets if t[0] == letter)
                == sorted(map(sorted, FACES5)))
    chk = check_detectors(p, circ.detectors)
    assert chk.count('deterministic') == 107 and all(d.value == 0 for d in chk.declared)
    assert chk.undeclared == ['c[111] = 0']                     # the final product of Y, 0 on its own
    (obs,) = check_detectors(p, [circ.observables[0]]).declared
    assert obs.status == 'deterministic' and obs.value == 0


def test_d5_is_t_dagger_in_the_colour_code_of_the_authors_final_check(program_d5):
    # Part 6, 'The result in 3D': the code is the colour code of the faces; in the faces' variables the reduced
    # diagram is T-dag|+> into that code with no outcome left; and the whole circuit equals that state in the run
    # where every outcome is 0 (tensor contraction), while T|+> does not
    from fractions import Fraction
    from pyzx.graph import Graph
    from pyzx.utils import VertexType
    from zxqec.codes import PauliString
    from zxqec.gpf import (boolean_variables, change_variables, phases_modulo, same_map, simplify_gadgets)
    p, _circ = program_d5
    code = p.code_at(p.num_stages - 1).data()
    qubits = [int(q) for q in code.qubits]
    on = lambda support: [int(q in support) for q in qubits]
    faces = ([PauliString.from_arrays(on(f), [0] * 19) for f in FACES5]
             + [PauliString.from_arrays([0] * 19, on(f)) for f in FACES5])
    assert all(code.contains(s) for s in faces) and len(code.stabilizers) == 18
    assert rank(np.array([s.symplectic for s in faces], dtype=np.uint8)) == 18
    assert on(SIDE5) == [int(b) for b in code.logical_x[0].x]

    g = p.quantum_graph()
    res = reduce_gpf(g)
    rows = [on(SIDE5)] + [on(f) for f in FACES5]
    m = change_variables(res.graph, rows, fuse=False)
    while simplify_gadgets(m):
        pass
    m = phases_modulo(change_variables(m, rows), res.constraints)
    assert not any(is_leaf(m, v) for v in m.vertices())
    assert not any(getattr(m.phase(v), 'free_vars', lambda: [])() for v in m.vertices())
    xs = [next(iter(m.neighbors(b))) for b in m.outputs()]
    wiring = sorted((str(m.phase(z)), tuple(q for q, x in zip(qubits, xs) if m.connected(z, x)))
                    for z in m.vertices() if m.type(z) == VertexType.Z)
    assert wiring == sorted([('7/4', tuple(SIDE5))] + [('0', tuple(sorted(f, key=qubits.index))) for f in FACES5])

    def textbook(angle):
        t = Graph()
        xv, bs = [], []
        for _q in qubits:
            xv.append(t.add_vertex(VertexType.X))
            bs.append(t.add_vertex(VertexType.BOUNDARY))
            t.add_edge((xv[-1], bs[-1]))
        for support in FACES5 + [SIDE5]:
            s = t.add_vertex(VertexType.Z)
            for q in support:
                t.add_edge((s, xv[qubits.index(q)]))
        t.add_edge((t.add_vertex(VertexType.Z, phase=angle), s))
        t.set_inputs(())
        t.set_outputs(tuple(bs))
        return t

    zero = {n: 0 for n in set(boolean_variables(g)) | {str(v) for c in res.constraints for v in c.free_vars()}}
    assert all(evaluate_phase(c, zero) == 0 for c in res.constraints)
    assert same_map(m.substitute_variables(zero), textbook(Fraction(7, 4)))
    circuit = g.substitute_variables(zero)
    assert same_map(circuit, textbook(Fraction(7, 4)))
    assert not same_map(circuit, textbook(Fraction(1, 4)))


GROUPS_D5 = [(0, 0, 'inject T', 'injection'), (1, 2, 'double-check d=3', 'cultivation'), (3, 3, 'grow', None),
             (4, 5, 'checks', 'syndrome extraction'), (6, 7, 'double-check d=5', 'cultivation')]


def test_d5_double_check_block_is_a_logical_measurement(program_d5):
    # as a block, the d = 5 double-check is a measurement in {T-dag|+>, T-dag|->} split by its own outcome
    # c[73] alone: outcomes measured before it that the program's detectors fix (c[21], which appears in a
    # sign of the code before it, c[65], c[70]) are read as their values, and no parity is counted twice
    p, _circ = program_d5
    s = p.summary(p.group_stages(GROUPS_D5))
    block = s.blocks[4]
    assert block.ok and block.action.measured_basis() == ('T^dag|+>', 'T^dag|->')
    assert sorted(br.conditions for br in block.action.branches()) == [[('c[73]', 0)], [('c[73]', 1)]]
    assert [b.ok for b in s.blocks] == [True, False, True, True, True]
    # the d = 3 double-check ends in the stage that also prepares the six pairs of the growth, so the code
    # after the block is not the code before it; its other two checks pass
    assert [ok for ok, _why in s.blocks[1].checks] == [False, True, True]
