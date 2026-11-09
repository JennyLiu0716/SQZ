"""Shor's code levelled up twice (examples/ShorCodeConcatenated.ipynb): one round of error correction at both levels.

The expected answers are derived by hand from the two codes, not from the tool. Level 1 reads the error and the
inner round in the three bit-flip codes (checks ZZ, logical Z = Z on any one qubit of a block): an X on one qubit
is corrected, a Z on qubit q is the logical Z of its block, two Zs in one block are a check, and two Xs in one
block fool the inner correction into X on the whole block. Level 2 reads the error and both rounds in Shor's code:
the outer checks see a logical Z of one block and correct it, so every single-qubit error is fixed; two Zs in
different blocks look like a Z on the third block, which completes the logical Z; X on a whole block is the
logical X.
"""
import pytest
from pyzx.circuit import Circuit

from zxqec.codes import logical_action
from zxqec.qprogram import QProgram

HEADER = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[17];\n'
          'creg s0[2];\ncreg s1[2];\ncreg s2[2];\ncreg o[2];\n')
ENCODER = ("cx q[0], q[3]; cx q[0], q[6];\nh q[0]; h q[3]; h q[6];\n"
           + "".join("cx q[{0}], q[{1}]; cx q[{0}], q[{2}];\n".format(b, b + 1, b + 2) for b in (0, 3, 6)))
INNER_CHECKS = "".join("cx q[{}], q[{h}]; cx q[{}], q[{h}];\nmeasure q[{h}] -> s{}[{}];\n".format(a, b, k, i, h=9 + 2 * k + i)
                       for k, base in enumerate((0, 3, 6)) for i, (a, b) in enumerate(((base, base + 1), (base + 1, base + 2))))
INNER_CORRECTIONS = "".join("if(s{0}==1) x q[{1}];\nif(s{0}==3) x q[{2}];\nif(s{0}==2) x q[{3}];\n".format(k, b, b + 1, b + 2)
                            for k, b in enumerate((0, 3, 6)))
OUTER_CHECKS = "".join("h q[{h}];\n".format(h=h) + "".join("cx q[{}], q[{}];\n".format(h, d) for d in data)
                       + "h q[{h}];\nmeasure q[{h}] -> o[{i}];\n".format(h=h, i=i)
                       for i, (h, data) in enumerate(((15, range(0, 6)), (16, range(3, 9)))))
OUTER_CORRECTIONS = "if(o==1) z q[0];\nif(o==3) z q[3];\nif(o==2) z q[6];\n"
QASM = {'X': 'x q[{0}];\n', 'Y': 'z q[{0}]; x q[{0}];\n', 'Z': 'z q[{0}];\n'}
BLOCKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def _two_levels(error):
    p = QProgram(17, bits=8, initial_states={q: '0' for q in range(1, 17)})
    for name, text in [('encoder', ENCODER), ('error', error), ('inner checks', INNER_CHECKS),
                       ('inner corrections', INNER_CORRECTIONS), ('outer checks', OUTER_CHECKS),
                       ('outer corrections', OUTER_CORRECTIONS)]:
        p.add_circuit(Circuit.from_qasm(HEADER + text), name=name)
    level1 = p.level_up(0, 1, 3, 'error + inner round')
    level2 = p.level_up(level1, 1, 3, 'error + both rounds')
    first = logical_action(p.block_graph(level1, 1, classical=False),
                           p.code_at(0).separated(BLOCKS + [[q] for q in p.code_at(0).fixed_qubits()]),
                           p.code_at(3).separated(BLOCKS + [[q] for q in p.code_at(3).fixed_qubits()]))
    both = p.logical_action(level2, 1)
    return p, level1, level2, first, both


def _left(action, identity):
    assert action.is_deterministic() and action.stays_in_code() is True
    assert {b.operation for b in action.branches()} == {'I'}
    (left,) = {b.correction or identity for b in action.branches()}
    return left


def test_the_codes_of_the_two_levels():
    p, level1, level2, _first, _both = _two_levels('')
    outer = p.code_at(0).data()
    assert (outer.n, outer.k) == (9, 1) and outer.is_css
    inner = p.code_at(0).separated(BLOCKS + [[q] for q in p.code_at(0).fixed_qubits()]).data()
    assert (inner.n, inner.k) == (9, 3)
    assert sorted(str(s) for s in inner.stabilizers) == sorted(
        str(s) for s in outer.stabilizers if set(str(s).lstrip('+-')) <= {'I', 'Z'})     # the ZZ checks only
    assert (level1, level2) == (1, 2)
    assert [p.levels.block(level1, k).name for k in range(p.levels.level_size(level1))] == [
        'encoder', 'error + inner round', 'outer checks', 'outer corrections']
    assert [p.levels.block(level2, k).name for k in range(p.levels.level_size(level2))] == [
        'encoder', 'error + both rounds']


@pytest.mark.parametrize('q', range(9))
@pytest.mark.parametrize('letter', 'XYZ')
def test_every_single_qubit_error_is_fixed_at_the_level_that_sees_it(letter, q):
    _p, _l1, _l2, first, both = _two_levels(QASM[letter].format(q))
    block = ['I', 'I', 'I']
    if letter in 'YZ':                     # the Z part passes the bit-flip code: the logical Z of its block
        block[q // 3] = 'Z'
    assert _left(first, 'III') == ''.join(block)
    assert _left(both, 'I') == 'I'         # the outer code fixes it


@pytest.mark.parametrize('error, after1, after2', [
    ('x q[0];\nx q[1];\n', 'XII', 'X'),     # two Xs in one block: X on the whole block, Shor's logical X
    ('x q[0];\nx q[3];\n', 'III', 'I'),     # in two blocks: each block corrects its own
    ('z q[0];\nz q[1];\n', 'III', 'I'),     # Z Z in one block is a check
    ('z q[0];\nz q[3];\n', 'ZZI', 'Z'),     # in two blocks: taken for a Z on the third, a logical Z
])
def test_two_qubit_errors(error, after1, after2):
    _p, _l1, _l2, first, both = _two_levels(error)
    assert _left(first, 'III') == after1
    assert _left(both, 'I') == after2


def test_levelling_up_twice_with_rewrites():
    # with the encoder in the blocks, so that the helpers are prepared and measured inside them: level 1 (encoder, Z4,
    # inner round) is Shor's encoder with the outer check on qubits 3-8 negated (the error survives, as a sign the
    # outer checks read); level 2 (everything) is Shor's encoder; both the same map as the circuit
    from pyzx.utils import VertexType
    from zxqec.gpf import equal_for_all_assignments, phases_modulo
    p = QProgram(17, bits=8, initial_states={q: '0' for q in range(1, 17)})
    for name, text in [('encoder', ENCODER), ('error', 'z q[4];\n'), ('inner checks', INNER_CHECKS),
                       ('inner corrections', INNER_CORRECTIONS), ('outer checks', OUTER_CHECKS),
                       ('outer corrections', OUTER_CORRECTIONS)]:
        p.add_circuit(Circuit.from_qasm(HEADER + text), name=name)
    level1 = p.level_up(0, 0, 3, 'encoder + error + inner round', rewrite='gpf')
    level2 = p.level_up(level1, 0, 2, 'encoder + error + both rounds', rewrite='gpf')
    for level in (level1, level2):
        reduced = p.reduced_level(level)
        assert equal_for_all_assignments(reduced.graph, p.quantum_graph(), constraints=reduced.detectors)
    block = p.reduced_level(level1).blocks[0]
    g = phases_modulo(block.graph, block.detectors)
    outs = list(g.outputs())
    negated = [sorted(outs.index(o) for w in g.neighbors(v) for o in g.neighbors(w) if o in outs)
               for v in g.vertices() if g.type(v) == VertexType.Z and str(g.phase(v)) != '0']
    assert negated == [[3, 4, 5, 6, 7, 8]]
    encoder = QProgram(9, initial_states={q: '0' for q in range(1, 9)})
    encoder.add_circuit(Circuit.from_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[9];\n' + ENCODER), name='encoder')
    reduced = p.reduced_level(level2)
    assert equal_for_all_assignments(reduced.graph, encoder.quantum_graph(), constraints=reduced.detectors)
