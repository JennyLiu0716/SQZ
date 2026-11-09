"""code_layers (what code_of reads), stabilizes (a tensor check of a stabilizer) and gf2.span."""
import numpy as np
import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement

from zxqec.codes import PauliString, code_layers, code_of, stabilizes
from zxqec.codes.gf2 import as_matrix, in_rowspace, nullspace, rank, rref, span
from zxqec.qprogram import QProgram

CHECKS = {0: (0, 3, 4, 5), 1: (1, 3, 4, 6), 2: (2, 4, 5, 6)}


def _encoder() -> QProgram:
    c = Circuit(7)
    for g in [CNOT(3, 5), CNOT(3, 6)] + [CNOT(s, t) for s, check in CHECKS.items() for t in check if t != s]:
        c.add_gate(g)
    p = QProgram(7, initial_states={q: '+' if q <= 3 else '0' for q in range(7) if q != 3})
    p.add_circuit(c, name='encode')
    return p


def test_layers_of_the_steane_encoder():
    g = _encoder().prefix_graph(0)
    layers = code_layers(g)
    code = code_of(g)
    assert layers.input_rows.shape == (1, 7) and layers.check_rows.shape == (3, 7) and not layers.hubs
    written = as_matrix([[int(q in check) for q in range(7)] for check in CHECKS.values()], 7)
    assert (rref(layers.check_rows)[0] == rref(written)[0]).all()
    # what code_of prints spans the same groups as what it reads
    assert rank(np.vstack([layers.check_rows, code.sx])) == rank(code.sx) == 3
    assert rank(np.vstack([layers.z_rows(), code.sz])) == rank(code.sz) == 3
    assert not ((np.vstack([layers.input_rows, layers.check_rows]) @ layers.z_rows().T) % 2).any()
    assert layers.check_phases == [0, 0, 0] and layers.output_phases == [0] * 7
    names = layers.names
    assert names[layers.input_spiders[0]] == 'in 0'
    assert [names[v] for v in layers.check_spiders] == ['c1', 'c2', 'c3']
    assert [names[v] for v in layers.output_spiders] == ['out {}'.format(q) for q in range(7)]
    text = layers.describe()
    assert 'in 0' in text and 'c3' in text and str(layers) == text
    fig = layers.draw()
    texts = {t.get_text() for t in fig.axes[0].texts}
    assert {'in 0', 'c1', 'c2', 'c3'} | {str(q) for q in range(7)} <= texts


def test_rows_of_m_without_gadgets_are_in_reduced_echelon_form():
    # without phase gadgets every check spider is a free row of the Z layer, so M comes out of
    # reduce_gpf in reduced row echelon form (columns in reduce_gpf's order) and is a basis
    from pyzx.utils import VertexType
    from zxqec.gpf import split_matrix
    header = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\ncreg c[2];\n'
    rep = QProgram(5, bits=2, initial_states={3: '0', 4: '0'})
    rep.add_circuit(Circuit.from_qasm(header + 'h q[3]; h q[4];\ncx q[3], q[0]; cx q[3], q[1];\n'
                                      'cx q[4], q[1]; cx q[4], q[2];\nh q[3]; h q[4];\n'
                                      'measure q[3] -> c[0];\nmeasure q[4] -> c[1];\n'))
    rep.add_circuit(Circuit.from_qasm(header + 'if(c==1) z q[0];\nif(c==3) z q[1];\nif(c==2) z q[2];\n'))
    for g in [_encoder().prefix_graph(0), rep.prefix_graph(0), rep.prefix_graph(1)]:
        layers = code_layers(g)
        assert not layers.hubs
        m = split_matrix(layers.gpf.graph, VertexType.Z)
        assert set(layers.check_spiders) == {m.rows[i] for i in m.free_rows()}
        ext = m.num_external
        rows = [m.entries[m.rows.index(v)][ext:] for v in layers.check_spiders]
        leads = [r.index(1) for r in rows]
        assert all(sum(r[c] for r in rows) == 1 for c in leads)
        assert rank(layers.check_rows) == len(layers.check_spiders)


def test_echelon_generators():
    # generators='echelon' keeps the group's reduced row echelon basis: the same group, logical
    # operators and distance as the default, only other generators printed
    from pyzx.circuit.gates import CZ
    from zxqec.codes.gf2 import rref
    header = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\ncreg c[2];\n'
    rep = QProgram(5, bits=2, initial_states={3: '0', 4: '0'})
    rep.add_circuit(Circuit.from_qasm(header + 'h q[3]; h q[4];\ncx q[3], q[0]; cx q[3], q[1];\n'
                                      'cx q[4], q[1]; cx q[4], q[2];\nh q[3]; h q[4];\n'
                                      'measure q[3] -> c[0];\nmeasure q[4] -> c[1];\n'))
    ring = QProgram(6, bits=1, initial_states={q: '+' for q in range(1, 6)})
    c = Circuit(6, bit_amount=1)
    c.add_gate(HAD(0))
    for q in range(1, 6):
        c.add_gate(CZ(q, q % 5 + 1))
    for q in range(1, 6):
        c.add_gate(CZ(0, q))
    c.add_gate(HAD(0))
    c.add_gate(Measurement(0, result_symbol='c[0]'))
    ring.add_circuit(c)
    for p in (rep, _encoder(), ring):
        lightest, echelon = p.code_at(0), p.code_at(0, generators='echelon')
        assert lightest.same_code_as(echelon)
        assert [str(x) for x in lightest.logical_x + lightest.logical_z] == \
               [str(x) for x in echelon.logical_x + echelon.logical_z]
        assert lightest.distance == echelon.distance
        assert (echelon.sx == rref(echelon.sx)[0]).all()
    assert [str(s) for s in rep.code_at(0, generators='echelon').stabilizers] == \
           ['(-1)^(c[0] + c[1])XIX', '(-1)^(c[1])IXX']                     # the rows of M
    assert [str(s) for s in rep.code_at(0).stabilizers] == ['(-1)^(c[0])XXI', '(-1)^(c[1])IXX']
    with pytest.raises(ValueError):
        code_of(rep.prefix_graph(0), generators='shortest')


def test_stabilizes_checks_the_circuit():
    g = _encoder().prefix_graph(0)
    code = code_of(g)
    assert all(stabilizes(g, s) for s in code.stabilizers)
    assert not stabilizes(g, code.logical_x[0]) and not stabilizes(g, code.logical_z[0])
    assert not stabilizes(g, '-XXXIXII')                 # the sign counts
    assert stabilizes(g, 'YYIYIII') is False             # logical Y
    with pytest.raises(ValueError):
        stabilizes(g, 'XX')


def test_stabilizes_signs_and_y():
    # the Bell state |00> + |11>: XX and ZZ, and YY = -(XX)(ZZ)
    p = QProgram(2, initial_states={0: '+', 1: '0'})
    c = Circuit(2)
    c.add_gate(CNOT(0, 1))
    p.add_circuit(c)
    g = p.prefix_graph(0)
    assert stabilizes(g, 'XX') and stabilizes(g, 'ZZ') and stabilizes(g, '-YY')
    assert not stabilizes(g, 'YY') and not stabilizes(g, '-XX')


def test_stabilizes_after_substitution():
    # a substituted diagram has vertex numbers with gaps; the check must use the copy's own outputs
    p = QProgram(3, bits=1, initial_states={0: '0', 1: '0', 2: '0'})
    c = Circuit(3, bit_amount=1)
    for gate in (HAD(2), CNOT(2, 0), CNOT(2, 1), HAD(2), Measurement(2, result_symbol='m')):
        c.add_gate(gate)
    p.add_circuit(c)
    g = p.prefix_graph(0)
    for m in (0, 1):
        run = g.substitute_variables({'m': m})
        assert stabilizes(run, PauliString.from_str('XX', m)) and not stabilizes(run, PauliString.from_str('XX', 1 - m))
        assert stabilizes(run, 'ZZ')


def test_layers_signs_from_outcomes():
    # a measured X check: its spider carries the outcome, the sign of the stabilizer
    p = QProgram(3, bits=1, initial_states={0: '0', 1: '0', 2: '0'})
    c = Circuit(3, bit_amount=1)
    c.add_gate(HAD(2))
    c.add_gate(CNOT(2, 0))
    c.add_gate(CNOT(2, 1))
    c.add_gate(HAD(2))
    c.add_gate(Measurement(2, result_symbol='m'))
    p.add_circuit(c)
    layers = code_layers(p.prefix_graph(0))
    code = code_of(p.prefix_graph(0))
    assert layers.input_rows.shape[0] == 0 and layers.check_rows.tolist() == [[1, 1]]
    assert [str(p) for p in layers.check_phases] == ['m'] and layers.output_phases == [0, 0]
    assert str(code.stabilizers[0]) == '(-1)^(m)XX'


def test_span():
    a = as_matrix([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]], 4)     # rank 2
    s = span(a)
    assert s.shape == (4, 4) and not s[0].any()
    assert [int(v.sum()) for v in s] == sorted(int(v.sum()) for v in s)
    assert all(in_rowspace(v, *rref(a)[:2]) for v in s)
    assert span(np.zeros((0, 3), np.uint8)).shape == (1, 3)
    assert len(span(nullspace(as_matrix([[1, 1, 1]], 3)))) == 4
