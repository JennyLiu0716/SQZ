"""Shor's nine-qubit code, and the bit-flip code of L. Yeh's thesis, Section 4.2 (examples/ShorCode.ipynb and
examples/ShorCodeStepByStep.ipynb)."""
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Z, Measurement, PostSelect
from pyzx.css import generate_css_encoder_graph
from pyzx.linalg import Mat2
from pyzx.utils import EdgeType, VertexType

from zxqec.codes import code_of
from zxqec.gpf import equal_for_all_assignments, reduce_gpf
from zxqec.qprogram import ParityControlled, QProgram, circuit_distance, pauli_errors

ENCODER = [CNOT(0, 3), CNOT(0, 6), HAD(0), HAD(3), HAD(6)] + [CNOT(b, b + k) for b in (0, 3, 6) for k in (1, 2)]
Z_CHECKS = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]
X_CHECKS = {'in order': [(0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8)],
            'alternating': [(0, 3, 1, 4, 2, 5), (3, 6, 4, 7, 5, 8)]}


def _program(n, bits, initial, stages):
    p = QProgram(n, bits=bits, initial_states=initial)
    for name, gates in stages:
        c = Circuit(n, bit_amount=bits)
        for gate in gates:
            c.add_gate(gate)
        p.add_circuit(c, name=name)
    return p


def _shor(order='alternating'):
    return _program(17, 8, {q: '0' for q in range(1, 17)}, [
        ('encoder', ENCODER),
        ('Z checks', [g for k, (a, b) in enumerate(Z_CHECKS) for g in (CNOT(a, 9 + k), CNOT(b, 9 + k), Measurement(9 + k, k))]),
        ('X checks', [g for k, qs in enumerate(X_CHECKS[order])
                      for g in [HAD(15 + k)] + [CNOT(15 + k, q) for q in qs] + [HAD(15 + k), Measurement(15 + k, 6 + k)]])])


def _spiders(g):
    return [v for v in g.vertices() if g.type(v) != VertexType.BOUNDARY]


def test_a_single_z_gets_through_the_bit_flip_code():
    p = _program(5, 2, {1: '0', 2: '0', 3: '0', 4: '0'}, [
        ('encoder', [CNOT(0, 1), CNOT(0, 2)]),
        ('checks', [CNOT(0, 3), CNOT(1, 3), Measurement(3, 0), CNOT(1, 4), CNOT(2, 4), Measurement(4, 1)])])
    assert p.code_at(0).parameters == (3, 1, 1)
    x, z = pauli_errors(p, 0, range(3), 'X'), pauli_errors(p, 0, range(3), 'Z')
    assert x.counts[1] == (3, 3, 0, 0) and x.leading() == (3, 1)
    assert z.counts[1] == (3, 0, 0, 3) and z.counts[2] == (3, 0, 3, 0)


def test_shors_code_and_the_round_of_checks():
    p = _shor()
    css, _ = generate_css_encoder_graph(Mat2([[1] * 6 + [0] * 3, [0] * 3 + [1] * 6]), Mat2([[1] * 3 + [0] * 6]), 'Z-X')
    css.auto_detect_io()
    code = p.code_at(0)
    assert code.parameters == (9, 1, 3) and code.same_code_as(code_of(css))
    # the tool's logical operators are all X and all Z: Shor's |0> is the tool's |+>, his encoder an H
    assert code.logical_x[0].letters[:9] == 'XXXIIIIII' and code.logical_z[0].letters[:9] == 'ZIIZIIZII'
    level = p.group_stages([(0, 0, 'encoder', 'encoder'), (1, 2, 'checks', 'syndrome extraction')])
    ops = [str(b.action.branches()[0].operation) for b in p.summary(level).blocks]
    assert ops == ['H', 'I'] and all(ok for ok, _ in p.summary(level).blocks[1].checks)
    assert p.code_at(2).same_code_as(code) and all(str(s.sign) == '0' for s in p.code_at(2).stabilizers)


def test_the_whole_circuit_is_the_encoder_after_an_h():
    p = _shor()
    whole = reduce_gpf(p.quantum_graph())
    assert sorted(str(d) for d in whole.detectors) == ['c[{}]'.format(k) for k in range(8)]
    encoder = _program(9, 0, {q: '0' for q in range(1, 9)}, [('encoder', ENCODER)])
    h = whole.to_graph()
    assert equal_for_all_assignments(h, encoder.quantum_graph())
    (b,) = h.inputs()
    (s,) = h.neighbors(b)
    assert h.edge_type(h.edge(b, s)) == EdgeType.HADAMARD and h.type(s) == VertexType.Z
    assert sorted(h.qubit(w) for w in h.neighbors(s) if w != b) == [6, 7, 8]


def test_errors_on_one_or_two_qubits_are_noticed():
    p = _shor()
    x, z = pauli_errors(p, 0, range(9), 'X', max_weight=3), pauli_errors(p, 0, range(9), 'Z', max_weight=3)
    assert x.leading() == (3, 3) and z.leading() == (3, 27)
    assert z.counts[2] == (36, 27, 9, 0)


def test_the_order_of_the_x_checks_sets_the_circuit_distance():
    assert circuit_distance(_shor('in order'), 1, 2, kind='full').distance == 1
    assert circuit_distance(_shor('alternating'), 1, 2, kind='full').distance == 3
    hook = circuit_distance(_shor('in order'), 1, 2, kind='X').explain()
    assert len(hook) == 1 and hook[0].pauli == 'X16'


def test_the_thesis_diagrams_level_by_level_as_in_the_step_by_step_notebook():
    # Eq. (4.5): one Z-spider, the input copied onto three outputs
    p = _program(3, 0, {1: '0', 2: '0'}, [('encoder', [CNOT(0, 1), CNOT(0, 2)])])
    last = p.level_up(p.level_up(0, 0, 0, rewrite='copy and fuse'), 0, 0, rewrite='remove identities')
    g = p.reduced_level(last).graph
    assert [g.type(v) for v in _spiders(g)] == [VertexType.Z] and equal_for_all_assignments(p.quantum_graph(), g)
    # Eq. (4.8): |a>|b> through the check post-selected on 0 is |a>|b> times (1 + e^{i pi (a+b)})/2
    p = _program(4, 1, {0: '0', 1: '0', 3: '0'}, [
        ('|a>, |b>', [ParityControlled('X', 0, ['a']), ParityControlled('X', 1, ['b'])]),
        ('Z Z', [CNOT(0, 3), CNOT(1, 3), PostSelect(3, '0')])])
    last = p.level_up(p.level_up(0, 0, 1, rewrite='spider_simp'), 0, 0, rewrite='copy results')
    g = p.reduced_level(last).graph
    assert equal_for_all_assignments(p.quantum_graph(), g)
    assert sorted(str(g.phase(v)) for v in _spiders(g)) == ['a', 'b'] and 'a + b' in str(g.scalar)
    # Shor's encoder: a Z-spider joined by Hadamard edges to three Z-spiders with three outputs each;
    # a Z on qubit 4 stops at the spider of its block
    for stages in ([('encoder', ENCODER)], [('encoder', ENCODER), ('Z on qubit 4', [Z(4)])]):
        p = _program(9, 0, {q: '0' for q in range(1, 9)}, stages)
        last = p.level_up(0, 0, len(stages) - 1, rewrite='copy and fuse')
        for rewrite in ('remove identities', 'spider_simp'):
            last = p.level_up(last, 0, 0, rewrite=rewrite)
        g = p.reduced_level(last).graph
        assert equal_for_all_assignments(p.quantum_graph(), g)
        spiders = _spiders(g)
        assert len(spiders) == 4 and all(g.type(v) == VertexType.Z for v in spiders)
        assert sum(1 for e in g.edges() if g.edge_type(e) == EdgeType.HADAMARD) == 3
        assert sorted(str(g.phase(v)) for v in spiders) == ['0'] * (5 - len(stages)) + ['1'] * (len(stages) - 1)
