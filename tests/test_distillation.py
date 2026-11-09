"""Magic-state distillation with the [[8,3,2]] code, eight T states to one CCZ state (L. Yeh's thesis,
Eqs. (6.47)-(6.49); examples/MagicStateDistillation.ipynb and examples/MagicStateDistillationStepByStep.ipynb)."""
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CCZ, CNOT, HAD, T, Measurement, PostSelect

from zxqec.gpf import equal_for_all_assignments, reduce_gpf
from zxqec.qprogram import QProgram, pauli_errors

WIRES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 0: 7}
GHZ = [CNOT(0, 4), CNOT(0, 5), CNOT(0, 6), CNOT(0, 7)]
FROM_X = [CNOT(1, 4), CNOT(1, 6), CNOT(1, 7), CNOT(0, 1)]
FROM_Y = [CNOT(2, 4), CNOT(2, 5), CNOT(2, 7), CNOT(0, 2)]
FROM_Z = [CNOT(3, 4), CNOT(3, 5), CNOT(3, 6), CNOT(0, 3)]
P = [T(1), T(2), T(3), T(4), T(5, adjoint=True), T(6, adjoint=True), T(7, adjoint=True), T(0, adjoint=True)]
P_DAGGER = [T(1, adjoint=True), T(2, adjoint=True), T(3, adjoint=True), T(4, adjoint=True), T(5), T(6), T(7), T(0)]
DECODE = FROM_Z[::-1] + FROM_Y[::-1] + FROM_X[::-1] + GHZ[::-1]
CHECK = [HAD(0), Measurement(0, 0)] + [Measurement(q, k + 1) for k, q in enumerate((4, 5, 6, 7))]
PROJECT = [PostSelect(0, '+')] + [PostSelect(q, '0') for q in (4, 5, 6, 7)]


def _program(stages, initial, bits=5):
    p = QProgram(8, bits=bits, mapping=WIRES, initial_states=initial)
    for gates in stages:
        c = Circuit(8, bit_amount=bits)
        for g in gates:
            c.add_gate(g)
        p.add_circuit(c)
    return p


def _factory():
    return _program([GHZ, FROM_X, FROM_Y, FROM_Z, P, DECODE, CHECK], {q: '0' if q >= 4 else '+' for q in range(8)})


def _ccz_state():
    p = QProgram(3, initial_states={0: '+', 1: '+', 2: '+'})
    c = Circuit(3)
    c.add_gate(CCZ(0, 1, 2))
    p.add_circuit(c)
    return p.quantum_graph()


def test_the_factory_prepares_the_ccz_state_and_its_checks_are_detectors():
    p = _factory()
    assert p.logical_action().verdict() == 'deterministic: every run implements prepares CCZ|+++>'
    level = p.group_stages([(0, 3, None, 'state preparation'), (4, 4, None, 'transversal gate'), (5, 6, None, 'decoder')])
    s = p.summary(level)
    assert [str(b.action.branches()[0].operation) for b in s.blocks] == ['a non-zero scalar', 'prepares CCZ|+++>', 'I']
    assert all(ok for b in s.blocks for ok, _ in b.checks)
    detectors = reduce_gpf(p.quantum_graph()).detectors
    assert sorted(str(d) for d in detectors) == ['c[0]', 'c[1] + c[3] + c[2]', 'c[2]', 'c[3]', 'c[4] + c[3] + c[2]']
    assert equal_for_all_assignments(p.quantum_graph(), _ccz_state(), constraints=detectors)


def test_decoding_the_magic_states_directly_agrees_where_every_result_is_0():
    direct = _program([P_DAGGER, DECODE, CHECK], {q: '+' for q in range(8)})
    detectors = reduce_gpf(_factory().quantum_graph()).detectors
    assert equal_for_all_assignments(direct.quantum_graph(), _factory().quantum_graph(), constraints=detectors)
    assert not reduce_gpf(direct.quantum_graph()).detectors        # nothing fixes its results


def test_z_errors_on_the_magic_states_leave_28_p_squared():
    e = pauli_errors(_factory(), 4, range(8), letter='Z')
    assert e.counts == {0: (1, 0, 1, 0), 1: (8, 8, 0, 0), 2: (28, 0, 0, 28), 3: (56, 56, 0, 0), 4: (70, 0, 14, 56),
                        5: (56, 56, 0, 0), 6: (28, 0, 0, 28), 7: (8, 8, 0, 0), 8: (1, 0, 1, 0)}
    assert e.leading() == (2, 28) and e.complete
    assert abs(e.output_error(1e-3) / (28 * 1e-6) - 1) < 0.01
    partial = pauli_errors(_factory(), 4, range(8), letter='Z', max_weight=2)
    assert not partial.complete and partial.leading() == (2, 28)


def test_eqs_6_48_and_6_49_level_by_level():
    magic = _program([P_DAGGER, DECODE + PROJECT], {q: '+' for q in range(8)}, bits=0)
    decoded = _program([[], DECODE + PROJECT, [CCZ(1, 2, 3)]], {q: '+' for q in range(8)}, bits=0)
    assert magic.logical_action().verdict() == 'deterministic: every run implements prepares CCZ|+++>'
    assert equal_for_all_assignments(magic.quantum_graph(), decoded.quantum_graph())          # Eq. (6.48)
    notation = decoded.group_stages([(0, 0, None, None, 'thesis notation')])
    normal_form = decoded.level_up(notation, 1, 1, rewrite='x-z gpf')
    copied = decoded.level_up(normal_form, 0, 1, rewrite='copy and fuse')
    h_box = decoded.level_up(copied, 0, 1, rewrite='remove identities')
    circuit = decoded.quantum_graph()
    pictures = [decoded.reduced_level(level).graph for level in (notation, normal_form, copied, h_box)]
    assert all(equal_for_all_assignments(circuit, g) for g in pictures)
    assert [sum(1 for v in g.vertices() if g.type(v) != 0) for g in pictures] == [49, 24, 7, 1]
    # level 2: E^dag with an X-spider on each of its eight inputs; level 4: the H-box alone, on the three outputs
    xz = decoded.reduced_block(normal_form, 1).graph
    assert all(xz.type(next(iter(xz.neighbors(i)))) == 2 for i in xz.inputs())
    (box,) = [v for v in pictures[-1].vertices() if pictures[-1].type(v) != 0]
    assert pictures[-1].type(box) == 3 and sorted(pictures[-1].neighbors(box)) == sorted(pictures[-1].outputs())
    assert equal_for_all_assignments(magic.quantum_graph(), pictures[-1])
