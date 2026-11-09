"""The transversal CCZ of the [[8,3,2]] code (L. Yeh's thesis, Eqs. (6.44)-(6.46); examples/TransversalCCZ.ipynb and
examples/TransversalCCZStepByStep.ipynb)."""
from fractions import Fraction

from pyzx.circuit import Circuit
from pyzx.circuit.gates import CCZ, CNOT, T, PostSelect

from zxqec.gpf import equal_for_all_assignments, phase_polynomial, reduce_gpf
from zxqec.qprogram import QProgram

#: qubit k is the thesis's qubit [8, 7, 6, 4, 1, 2, 3, 5][k]; x, y, z enter on 1, 2, 3; wires in the thesis's order
WIRES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 0: 7}
CORNERS = ['000', '100', '010', '001', '111', '011', '101', '110']
GHZ = [CNOT(0, 4), CNOT(0, 5), CNOT(0, 6), CNOT(0, 7)]
FROM_X = [CNOT(1, 4), CNOT(1, 6), CNOT(1, 7), CNOT(0, 1)]
FROM_Y = [CNOT(2, 4), CNOT(2, 5), CNOT(2, 7), CNOT(0, 2)]
FROM_Z = [CNOT(3, 4), CNOT(3, 5), CNOT(3, 6), CNOT(0, 3)]
P = [T(1), T(2), T(3), T(4), T(5, adjoint=True), T(6, adjoint=True), T(7, adjoint=True), T(0, adjoint=True)]


def _program(stages):
    p = QProgram(8, mapping=WIRES, initial_states={0: '+', 4: '0', 5: '0', 6: '0', 7: '0'})
    for gates in stages:
        c = Circuit(8)
        for g in gates:
            c.add_gate(g)
        p.add_circuit(c)
    return p


def _ccz():
    c = Circuit(3)
    c.add_gate(CCZ(0, 1, 2))
    return c.to_graph()


def test_the_encoder_is_the_cube_code_and_the_t_layer_a_transversal_ccz():
    p = _program([GHZ, FROM_X, FROM_Y, FROM_Z, P])
    code = p.code_at(3)
    assert code.parameters == (8, 3, 2)
    # each qubit holds w + its corner . (x, y, z): the logical X of x is X on the corners with a 1 for x
    for i in range(3):
        assert code.logical_x[i].letters.replace('I', '.') in (
            ''.join('X' if c[i] == '1' else '.' for c in CORNERS), ''.join('.' if c[i] == '1' else 'X' for c in CORNERS))
    level = p.group_stages([(0, 3, 'encoder', 'encoder'), (4, 4, 'T layer', 'transversal gate')])
    s = p.summary(level)
    assert [str(b.action.branches()[0].operation) for b in s.blocks] == ['I', 'CCZ']
    assert all(ok for b in s.blocks for ok, _ in b.checks)
    assert p.logical_action().verdict() == 'deterministic: every run implements CCZ'


def test_the_whole_circuit_is_the_encoder_after_seven_gadgets_with_phase_polynomial_xyz():
    h = reduce_gpf(_program([GHZ, FROM_X, FROM_Y, FROM_Z, P]).quantum_graph()).graph
    names = {next(iter(h.neighbors(b))): v for b, v in zip(h.inputs(), 'xyz')}
    poly, gadgets = phase_polynomial(h)
    assert len(gadgets) == 7
    assert {'*'.join(sorted(names[t[1]] for t in m)): str(c) for m, c in poly.items()} == {'x*y*z': '1'}


def test_eqs_6_45_and_6_46_level_by_level():
    decode = GHZ[::-1] + [PostSelect(0, '+')] + [PostSelect(q, '0') for q in (4, 5, 6, 7)]
    p = _program([GHZ, FROM_X, FROM_Y, FROM_Z, P, FROM_Z[::-1], FROM_Y[::-1], FROM_X[::-1], decode])
    circuit = p.quantum_graph()
    assert equal_for_all_assignments(circuit, _ccz())
    z = p.level_up(0, 3, 5, rewrite='phase gadgets')
    y = p.level_up(z, 2, 4, rewrite='phase gadgets')
    x = p.level_up(y, 1, 3, rewrite='phase gadgets')
    w = p.level_up(x, 0, 2, rewrite='phase gadgets')
    nest = p.level_up(w, 0, 0, rewrite='spider nest')
    pictures = [p.reduced_level(level).graph for level in (z, y, x, w, nest)]
    assert all(equal_for_all_assignments(circuit, g) for g in pictures)
    assert [sum(1 for v in g.vertices() if g.type(v) != 0) for g in pictures] == [75, 63, 49, 18, 11]
    assert equal_for_all_assignments(pictures[-1], _ccz())
    # level 4: eight gadgets on x, y, z and w, the one on w alone drawn as the phase -pi/4 of w's spider;
    # level 5: seven on x, y, z, those on one wire drawn as the phase pi/4 of its spider
    for g, hubs, phases in ((pictures[3], 7, [Fraction(7, 4)]), (pictures[4], 4, [Fraction(1, 4)] * 3)):
        drawn = [v for v in g.vertices() if g.type(v) == 2 and any(g.vertex_degree(u) == 1 for u in g.neighbors(v))]
        on_wires = [g.phase(v) for v in g.vertices() if g.type(v) == 1 and g.vertex_degree(v) > 1 and g.phase(v)]
        assert len(drawn) == hubs and sorted(on_wires) == phases
