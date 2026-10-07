"""Code switching from the Steane code to the [[15,1,3]] Reed-Muller code (examples/CodeSwitching.ipynb)."""
import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement
from pyzx.css import generate_css_encoder_graph
from pyzx.linalg import Mat2

from zxqec.codes import code_of
from zxqec.qprogram import ParityControlled, QProgram

STEANE = [CNOT(2, 4), CNOT(2, 5), HAD(0), CNOT(0, 2), CNOT(0, 4), CNOT(0, 6), HAD(1), CNOT(1, 2), CNOT(1, 5),
          CNOT(1, 6), HAD(3), CNOT(3, 4), CNOT(3, 5), CNOT(3, 6)]
CUBE = [HAD(7), CNOT(7, 10), CNOT(7, 12), CNOT(7, 13), HAD(8), CNOT(8, 10), CNOT(8, 12), CNOT(8, 14),
        HAD(9), CNOT(9, 10), CNOT(9, 13), CNOT(9, 14), HAD(11), CNOT(11, 12), CNOT(11, 13), CNOT(11, 14)]
CHECKED_FACES = [[8, 10, 12, 14], [7, 9, 11, 13], [9, 10, 13, 14], [11, 12, 13, 14]]
GAUGE = [[2, 6, 10, 14], [4, 6, 12, 14], [5, 6, 13, 14]]
FIX = [[3, 4, 5, 6], [1, 2, 5, 6], [0, 2, 4, 6]]


def _switch(checked=True, fix=True):
    p = QProgram(22, bits=7, initial_states={**{q: '0' for q in range(22) if q != 2}, 18: '+', 19: '+', 20: '+', 21: '+'})
    stages = [('Steane code', list(STEANE)), ('cube state', list(CUBE))]
    if checked:
        for k, face in enumerate(CHECKED_FACES):
            stages[1][1].extend([CNOT(18 + k, q) for q in face] + [HAD(18 + k), Measurement(18 + k, 3 + k)])
    gauge = []
    for k, op in enumerate(GAUGE):
        gauge += [CNOT(q, 15 + k) for q in op] + [Measurement(15 + k, k)]
    stages.append(('measure the gauge operators', gauge))
    stages.append(('gauge fixing', [ParityControlled('X', q, ['c[%d]' % k]) for k, face in enumerate(FIX) for q in face]
                   if fix else []))
    for name, gates in stages:
        c = Circuit(22, bit_amount=7)
        for g in gates:
            c.add_gate(g)
        p.add_circuit(c, name=name)
    return p


def _reed_muller():
    rows = [[(c >> b) & 1 for c in range(1, 16)] for b in range(4)]
    g, _ = generate_css_encoder_graph(Mat2(rows), Mat2([[1] * 15]), 'Z-X')
    g.auto_detect_io()
    return code_of(g)


def test_the_codes_and_the_logical_action():
    p = _switch()
    before = p.code_at(1).restricted(list(range(15)))
    assert [b.parameters[:2] for b in before.blocks()] == [(7, 1), (8, 0)] and before.distance == 3
    after = p.code_at(3)
    assert after.same_code_as(_reed_muller())                          # the same checks (compared up to signs)
    assert all(str(s.sign) == '0' for s in after.stabilizers)          # and every sign is +
    signs = [str(s.sign) for s in p.code_at(2).stabilizers if str(s.sign) not in ('0', '1')]
    assert sorted(signs) == ['c[0]', 'c[1]', 'c[2]']                 # before the fix the measured checks carry the outcomes
    level = p.group_stages([(0, 0, 'Steane code', 'encoder'), (1, 1, 'cube state', 'state preparation'),
                            (2, 3, 'switch', 'code switching')])
    s = p.summary(level)
    assert s.ok and s.blocks[2].action.verdict() == 'deterministic: every run implements I'
    assert [ok for ok, _ in s.blocks[2].checks] == [True, True, True]


def test_without_the_corrections_the_code_has_random_signs():
    code = _switch(fix=False).code_at(3)
    assert code.same_code_as(_reed_muller())                           # the right checks ...
    assert sorted(str(s.sign) for s in code.stabilizers if str(s.sign) != '0') == ['c[0]', 'c[1]', 'c[2]']


def test_a_block_that_keeps_its_code_is_not_a_code_switch():
    p = QProgram(7, initial_states={q: '0' for q in range(7) if q != 2})
    for name, gates in (('Steane code', STEANE), ('nothing', [HAD(0), HAD(0)])):
        c = Circuit(7)
        for g in gates:
            c.add_gate(g)
        p.add_circuit(c, name=name)
    level = p.group_stages([(0, 0, 'Steane code', 'encoder'), (1, 1, 'nothing', 'code switching')])
    s = p.summary(level)
    assert not s.ok and s.blocks[1].checks[0][0] is False


def test_circuit_distance_needs_the_cube_state_checked():
    pytest.importorskip('stim')
    pytest.importorskip('codedistance')
    from zxqec.qprogram import circuit_distance
    assert circuit_distance(_switch(checked=False), 1, 3).distance == 2
    for kind, d in (('full', 3), ('Z', 3), ('X', 5)):
        assert circuit_distance(_switch(), 1, 3, kind=kind).distance == d
