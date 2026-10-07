"""Gate teleportation (L. Yeh's thesis, Proposition 6.41 and Eq. (6.42)): a logical CX between the two logical
qubits of one block of the 2x2 toric code, through an ancilla block (examples/GateTeleportation.ipynb)."""
import itertools

import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, SWAP, Measurement, Reset
from pyzx.utils import VertexType

from zxqec.gpf import phases_modulo, reduce_gpf, to_gpf_steps
from zxqec.qprogram import ParityControlled, QProgram

#: the toric code on a 2x2 torus: qubit 2(2y+x) is the horizontal edge from (x, y), qubit 2(2y+x)+1 the vertical one
STARS = [[0, 1, 2, 5], [0, 2, 3, 7], [1, 4, 5, 6], [3, 4, 6, 7]]
PLAQUETTES = [[0, 1, 3, 4], [1, 2, 3, 6], [0, 4, 5, 7], [2, 5, 6, 7]]
X_C, Z_C = [0, 4], [0, 2]
ENCODER = [CNOT(2, 6), CNOT(5, 7), HAD(0), CNOT(0, 2), CNOT(0, 4), CNOT(0, 6), HAD(1), CNOT(1, 4), CNOT(1, 5),
           CNOT(1, 6), HAD(3), CNOT(3, 4), CNOT(3, 6), CNOT(3, 7)]
REFLECT = [(0, 1), (2, 5), (3, 4), (6, 7)]          #: (x, y) -> (y, x): swaps the two logical qubits


def _toric():
    c = Circuit(8)
    for g in ENCODER:
        c.add_gate(g)
    return c


def _stage(gates, n):
    c = Circuit(n, bit_amount=10)
    for g in gates:
        c.add_gate(g)
    return c


def _teleport_cx(twice=False, corrected=True):
    """Data block 0-7 (inputs c on 2, t on 5), ancilla 8-15 in |+0>, helper 16 (and 17 when c is measured twice)."""
    n = 18 if twice else 17
    p = QProgram(n, bits=10, initial_states={**{q: '0' for q in range(n) if q not in (2, 5)}, 10: '+'})
    for block in (range(8), range(8, 16)):
        c = Circuit(n, bit_amount=10)
        c.add_circuit(_toric(), mask=list(block), bit_mask=[])
        p.add_circuit(c)
    gates = [CNOT(8 + j, j) for j in range(8)]
    for k in range(2 if twice else 1):
        gates += [CNOT(q, 16 + k) for q in Z_C] + [Measurement(16 + k, k)]
    if corrected:
        gates += [ParityControlled('X', b + q, ['c[0]']) for q in X_C for b in (8, 0)]
    p.add_circuit(_stage(gates, n), name='teleport c into the ancilla')
    p.add_permutation([(8 + a, 8 + b) for a, b in REFLECT], cycles=True)
    p.add_circuit(_stage([CNOT(8 + j, j) for j in range(8)], n), name='transversal CX', physical=True)
    p.add_permutation([(8 + a, 8 + b) for a, b in REFLECT], cycles=True)
    gates = [CNOT(8 + j, j) for j in range(8)] + [HAD(8 + j) for j in range(8)]
    gates += [Measurement(8 + j, 2 + j) for j in range(8)]
    gates += [ParityControlled('Z', q, ['c[{}]'.format(2 + j) for j in X_C]) for q in Z_C]
    p.add_circuit(_stage(gates, n), name='teleport c back')
    return p


def _wiring(g):
    """For each Z-spider of a normal form: its input (or None) and the outputs it is joined to."""
    outs = {next(iter(g.neighbors(b))): j for j, b in enumerate(g.outputs())}
    ins = {next(iter(g.neighbors(b))): j for j, b in enumerate(g.inputs())}
    return sorted((ins.get(v, -1), tuple(sorted(outs[w] for w in g.neighbors(v) if w in outs)))
                  for v in g.vertices() if g.type(v) == VertexType.Z)


def test_eq_6_42_on_logical_qubits_rewrites_to_a_cx_and_the_ancillas_t_in_0():
    # wires: c, t of the data block, c, t of the ancilla; measure c, prepare it again, as drawn in the thesis
    p = QProgram(4, bits=2, initial_states={2: '+', 3: '0'})
    for gates, physical in (([CNOT(2, 0), CNOT(3, 1), Measurement(0, 0), Reset(0), ParityControlled('X', 2, ['c[0]'])],
                             False), ([CNOT(2, 0), CNOT(3, 1)], True),
                            ([CNOT(2, 0), CNOT(3, 1), HAD(2), Measurement(2, 1), ParityControlled('Z', 0, ['c[1]'])],
                             False)):
        if physical:
            p.add_permutation([(2, 3)], cycles=True)
        c = Circuit(4, bit_amount=2)
        for g in gates:
            c.add_gate(g)
        p.add_circuit(c, physical=physical)
        if physical:
            p.add_permutation([(2, 3)], cycles=True)
    assert p.logical_action().verdict() == 'deterministic: every run implements CNOT(0->1)'
    steps = to_gpf_steps(p.quantum_graph())
    last = steps[-1][1]
    assert [len([v for v in h.vertices() if h.type(v) != VertexType.BOUNDARY]) for _, h in steps] == [33, 14, 12, 12, 4, 5]
    assert _wiring(last) == [(0, (0, 1)), (1, (1,))]            # c onto c and t, t onto t: a CX
    xs = [v for v in last.vertices() if last.type(v) == VertexType.X]
    assert len(xs) == 3 and all(last.phase(v) == 0 for v in last.vertices())   # the third X-spider: |0> on output 2


def test_the_whole_circuit_is_the_encoder_after_a_cx():
    p = _teleport_cx()
    whole = reduce_gpf(p.quantum_graph())
    nf = phases_modulo(whole.graph, whole.detectors)
    assert {str(nf.phase(v)) for v in nf.vertices()} == {'0'}
    wiring = _wiring(nf)
    assert wiring[3:] == [(0, (0, 1, 3, 4)), (1, (1, 3))]       # c onto X_c and X_t, t onto X_t
    stars = {frozenset(s) for s in STARS}
    assert all(frozenset(q) in stars or any(frozenset(q) == a ^ b for a in stars for b in stars)
               for i, q in wiring if i == -1) and sum(1 for i, _ in wiring if i == -1) == 3


def test_the_whole_program_is_a_cx_from_c_to_t_in_every_run():
    for twice in (False, True):
        act = _teleport_cx(twice).logical_action()
        assert act.is_deterministic() and {str(b.operation) for b in act.branches()} == {'CNOT(0->1)'}


def test_c_moves_into_the_ancilla_and_back():
    p = _teleport_cx()
    data, ancilla = p.code_at(2).blocks()
    assert data.parameters == ancilla.parameters == (8, 1, 2)
    assert data.contains('ZIZIIIII')                      # Z of c: c is reset to |0> in the data block
    assert data.logical_x[0].letters == 'IXIXIIII'        # which keeps t
    assert ancilla.logical_x[0].letters == 'XIIIXIII'     # and the ancilla holds c
    assert all(str(s.sign) == '0' for s in p.code_at(2).stabilizers)
    assert p.code_at(6).data().same_code_as(p.code_at(0).data())


def test_the_steps_in_the_tools_numbering_of_the_logical_qubits():
    # logical qubits are numbered by the qubits that carry them: in the middle t (data block) comes first
    p = _teleport_cx()
    level = p.group_stages([(2, 2, 'in'), (3, 5, 'CX', 'transversal gate'), (6, 6, 'back')])
    s = p.summary(level)
    ops = [str(b.action.branches()[0].operation) for b in s.blocks[2:]]
    assert ops == ['SWAP', 'CNOT(1->0)', 'SWAP']
    assert all(ok for b in s.blocks for ok, _ in b.checks)


def test_without_the_corrections_t_ends_with_an_x_when_c_was_measured_1():
    p = _teleport_cx(corrected=False)
    assert sorted({str(s.sign) for s in p.code_at(2).stabilizers}) == ['0', 'c[0]']
    act = p.logical_action()
    assert act.verdict().startswith('deterministic up to a Pauli frame')
    # the X_c missing on the ancilla and on the data block cancel on c; the one carried by the CX lands on t
    assert {(b.when(), b.operation, b.correction) for b in act.branches()} == {
        ('when c[0] = 0', 'CNOT(0->1)', ''), ('when c[0] = 1', 'CNOT(0->1)', 'IX')}


def test_the_frame_left_in_the_reduced_diagram_is_zero_modulo_the_detectors():
    report = _teleport_cx().measurement_report()
    from zxqec.gpf import reduce_modulo
    assert report.frame and all(str(reduce_modulo(ph, report.detectors)) == '0' for _, ph in report.frame)


def test_the_reflection_is_a_logical_swap_and_no_ordering_of_the_qubits_is_a_cx():
    p = QProgram(8, initial_states={q: '0' for q in range(8) if q not in (2, 5)})
    p.add_circuit(_toric())
    c = Circuit(8)
    for a, b in REFLECT:
        c.add_gate(SWAP(a, b))
    p.add_circuit(c)
    assert p.logical_action().verdict() == 'deterministic: every run implements SWAP'

    # every ordering of the eight qubits that keeps the stars and the plaquettes, and what it does to X_c, X_t
    def span(rows):
        out = {0}
        for r in rows:
            out |= {v ^ r for v in out}
        return out
    mask = lambda qs: sum(1 << q for q in qs)
    stars, plaquettes = span(mask(s) for s in STARS), span(mask(s) for s in PLAQUETTES)
    logical = {mask(X_C): 'c', mask([1, 3]): 't', mask(X_C) ^ mask([1, 3]): 'ct'}
    actions = set()
    for perm in itertools.permutations(range(8)):
        move = lambda m: sum(1 << perm[q] for q in range(8) if m >> q & 1)
        if {move(s) for s in stars} != stars or {move(s) for s in plaquettes} != plaquettes:
            continue
        image = lambda m: next(name for x, name in logical.items() if any(move(m) ^ x == s for s in stars))
        actions.add((image(mask(X_C)), image(mask([1, 3]))))
    assert actions == {('c', 't'), ('t', 'c')}            # the identity and the swap; a CX would map X_c to X_c X_t


def test_circuit_distance_needs_c_measured_twice():
    pytest.importorskip('stim')
    pytest.importorskip('codedistance')
    from zxqec.qprogram import circuit_distance
    once = circuit_distance(_teleport_cx(), 2, 6, several_logicals=True)
    assert once.distance == 1
    assert [str(f) for f in once.explain()] == ['stage 2: X16 before M 16 (flips its result)']
    twice = circuit_distance(_teleport_cx(twice=True), 2, 6, kind=('full', 'Z', 'X'), several_logicals=True)
    assert {k: d.distance for k, d in twice.items()} == {'full': 2, 'Z': 2, 'X': 2}


def test_eq_6_42_level_by_level_as_in_the_step_by_step_notebook():
    # one rewrite per level (examples/GateTeleportationStepByStep.ipynb); each level is the circuit's map
    from zxqec.gpf import equal_for_all_assignments
    p = QProgram(4, bits=2, initial_states={0: '+', 1: '0'})
    for gates, physical in (([CNOT(0, 2), CNOT(1, 3), Measurement(2, 0), Reset(2), ParityControlled('X', 0, ['c[0]'])],
                             False), ([CNOT(0, 2), CNOT(1, 3)], True),
                            ([CNOT(0, 2), CNOT(1, 3), HAD(0), Measurement(0, 1), ParityControlled('Z', 2, ['c[1]'])],
                             False)):
        if physical:
            p.add_permutation([(0, 1)], cycles=True)
        c = Circuit(4, bit_amount=2)
        for gate in gates:
            c.add_gate(gate)
        p.add_circuit(c, physical=physical)
        if physical:
            p.add_permutation([(0, 1)], cycles=True)
    circuit = p.quantum_graph()
    runs = reduce_gpf(circuit).constraints
    assert [str(x) for x in runs] == ['s0_r0 + c[0]']
    notation = p.group_stages([(s, s, None, None, 'thesis notation') for s in range(5)])
    copied_in = p.level_up(notation, 0, 0, rewrite='copy and fuse')
    wire_in = p.level_up(copied_in, 0, 0, rewrite='remove identities')
    through_cx = p.level_up(wire_in, 0, 3, rewrite='copy and fuse')
    copied_back = p.level_up(through_cx, 0, 1, rewrite='copy and fuse')
    done = p.level_up(copied_back, 0, 0, rewrite='remove identities')
    assert (notation, done) == (5, 10)
    pictures = [p.reduced_level(level).graph for level in (notation, copied_in, wire_in, through_cx, copied_back, done)]
    assert all(equal_for_all_assignments(circuit, h, constraints=runs) for h in pictures)
    # the five diagrams of Eq. (6.42) have 19, 15, 10, 6 and 3 spiders; level 7 (12) lies between the second and third
    assert [sum(1 for v in h.vertices() if h.type(v) != VertexType.BOUNDARY) for h in pictures] == [19, 15, 12, 10, 6, 3]
    # the last: a CX from c (data wire 2) to t (wire 3), and |0> on the ancilla's t
    last = pictures[-1]
    spiders = [v for v in last.vertices() if last.type(v) != VertexType.BOUNDARY]
    (z,) = [v for v in spiders if last.type(v) == VertexType.Z]
    x = [v for v in spiders if last.type(v) == VertexType.X]
    (target,) = [v for v in x if last.vertex_degree(v) == 3]
    (zero,) = [v for v in x if last.vertex_degree(v) == 1]
    assert last.connected(z, target) and last.vertex_degree(z) == 3
    assert all(last.phase(v) == 0 for v in spiders)
    assert [last.type(w) for w in last.neighbors(zero)] == [VertexType.BOUNDARY]
