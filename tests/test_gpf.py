"""Tests for zxqec.gpf: GPF extraction and reduction with phases."""
import random
from fractions import Fraction

import pytest
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, T, XPhase, ZPhase
from pyzx.graph import Graph
from pyzx.symbolic import Poly, new_var
from pyzx.utils import EdgeType, VertexType

from zxqec.gpf import (
    SplitMatrix, boolean_variables, equal_for_all_assignments, has_boundary_hadamard, is_gpf,
    is_leaf, is_stem, push_hadamards_to_boundary, reduce_gpf, reduce_rows, restore_boundary_hadamards,
    split_matrix, to_gpf,
)
from zxqec.gpf.matrix import phase_mod2


def spiders(g):
    return [v for v in g.vertices() if g.type(v) != VertexType.BOUNDARY]


def random_cnot_layers(rng, qubits, n_gates):
    c = Circuit(qubits)
    for _ in range(n_gates):
        a, b = rng.sample(range(qubits), 2)
        c.add_gate(CNOT(a, b))
    return c


def conditional_pauli_circuit(rng, qubits, n_gates, names=('a', 'b')):
    """CNOT circuit with random Z(a pi) / X(b pi) Paulis on symbolic Boolean variables."""
    c = random_cnot_layers(rng, qubits, n_gates)
    g = c.to_graph()
    variables = {n: new_var(n, True, g.var_registry) for n in names}
    # sprinkle Boolean Paulis onto some spiders (adding to their phase)
    for v in spiders(g):
        if rng.random() < 0.4:
            g.set_phase(v, phase_mod2(g.phase(v) + variables[rng.choice(names)]))
    return g


# -- shape ------------------------------------------------------------------

def test_phase_free_random_circuits():
    rng = random.Random(7)
    for qubits, n in [(2, 4), (3, 8), (4, 12), (5, 16)]:
        c = random_cnot_layers(rng, qubits, n)
        g = c.to_graph()
        gpf = to_gpf(g)
        assert is_gpf(gpf)
        assert zx.compare_tensors(gpf, g, preserve_scalar=False)
        res = reduce_gpf(g)
        assert is_gpf(res.graph) and res.constraints == [] and not res.is_zero
        assert zx.compare_tensors(res.graph, g, preserve_scalar=False)
        # a unitary on n qubits needs exactly one Z and one X spider per wire
        assert len(spiders(res.graph)) == 2 * qubits


def test_transversal_hadamard_layers():
    rng = random.Random(3)
    c = random_cnot_layers(rng, 3, 6)
    for q in range(3):
        c.add_gate(HAD(q))
    c.add_circuit(random_cnot_layers(rng, 3, 6))
    g = c.to_graph()
    gpf = to_gpf(g)
    assert is_gpf(gpf)                                    # no internal Hadamard edges left
    assert any(has_boundary_hadamard(gpf, b) for b in list(gpf.inputs()) + list(gpf.outputs()))
    assert zx.compare_tensors(restore_boundary_hadamards(gpf), g, preserve_scalar=False)
    res = reduce_gpf(g)
    assert zx.compare_tensors(res.to_graph(), g, preserve_scalar=False)


def test_odd_hadamard_cycle_is_rejected():
    c = Circuit(2)
    c.add_gate(CNOT(0, 1)); c.add_gate(HAD(0)); c.add_gate(CNOT(0, 1))   # H on one leg of a loop
    with pytest.raises(ValueError):
        push_hadamards_to_boundary(c.to_graph())


# -- Boolean phases ----------------------------------------------------------

def test_boolean_paulis_random():
    rng = random.Random(11)
    for trial in range(6):
        g = conditional_pauli_circuit(rng, 3, 8)
        names = boolean_variables(g)
        gpf = to_gpf(g)
        assert is_gpf(gpf)
        assert equal_for_all_assignments(gpf, g, names)
        res = reduce_gpf(g)
        assert equal_for_all_assignments(res.to_graph(), g, names, constraints=res.constraints)
        assert len(spiders(res.graph)) == 6


def test_isolate_paulis_reproduces_thesis_form():
    rng = random.Random(5)
    g = conditional_pauli_circuit(rng, 3, 8)
    names = boolean_variables(g)
    thesis = to_gpf(g, isolate_paulis=True)
    assert is_gpf(thesis)
    # every non-boundary spider that carries a phase is a leaf, all others are phase-free
    for v in spiders(thesis):
        if thesis.phase(v) != 0:
            assert is_leaf(thesis, v) and thesis.vertex_degree(v) == 1
    assert equal_for_all_assignments(thesis, g, names)
    res = reduce_gpf(g, isolate_paulis=True)
    assert equal_for_all_assignments(res.to_graph(), g, names, constraints=res.constraints)


def test_non_pauli_phases_become_gadgets():
    c = Circuit(2)
    c.add_gate(CNOT(0, 1)); c.add_gate(T(1)); c.add_gate(CNOT(1, 0)); c.add_gate(ZPhase(0, Fraction(1, 3)))
    g = c.to_graph()
    gpf = to_gpf(g)
    assert is_gpf(gpf)
    leaves = [v for v in spiders(gpf) if is_leaf(gpf, v)]
    assert sorted(str(gpf.phase(v)) for v in leaves) == ['1/3', '1/4']
    for v in leaves:
        assert gpf.vertex_degree(v) == 1
        (stem,) = gpf.neighbors(v)
        assert is_stem(gpf, stem) and gpf.type(stem) != gpf.type(v) and gpf.phase(stem) == 0
    assert zx.compare_tensors(gpf, g, preserve_scalar=False)
    res = reduce_gpf(g)
    assert zx.compare_tensors(res.to_graph(), g, preserve_scalar=False)
    assert len([v for v in spiders(res.graph) if is_leaf(res.graph, v)]) == 2


# -- detectors and zero diagrams ---------------------------------------------

REPEATED_XX = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[2];
h q[2];
cx q[2],q[0];
cx q[2],q[1];
h q[2];
measure q[2] -> c[0];
reset q[2];
h q[2];
cx q[2],q[0];
cx q[2],q[1];
h q[2];
measure q[2] -> c[1];
"""


def test_repeated_syndrome_gives_detector():
    circ = zx.qasm(REPEATED_XX)
    circ.initialize_qubits([False, False, True])
    g = circ.to_graph()
    res = reduce_gpf(g)
    assert not res.is_zero
    dets = [str(d) for d in res.detectors]
    assert any(set(d.replace(' ', '').split('+')) == {'c[0]', 'c[1]'} for d in dets), dets
    assert equal_for_all_assignments(res.to_graph(), g, constraints=res.constraints)
    assert not equal_for_all_assignments(res.to_graph(), g)      # forbidden outcomes differ


def test_contradictory_outcomes_give_zero_diagram():
    circ = zx.qasm(REPEATED_XX)
    circ.initialize_qubits([False, False, True])
    g = circ.to_graph()
    res = reduce_gpf(g.substitute_variables({'c[0]': 0, 'c[1]': 1}))   # impossible outcome pair
    assert res.is_zero
    res = reduce_gpf(g.substitute_variables({'c[0]': 1, 'c[1]': 1}))   # consistent outcomes
    assert not res.is_zero
    assert all(isinstance(c, Poly) for c in res.constraints)           # only the reset randomness remains symbolic


# -- the matrix layer ----------------------------------------------------------

def test_reduce_rows_only_uses_free_rows():
    a = new_var('a', True)
    b = new_var('b', True)
    # columns: [ext0 | int0 int1 int2]; rows: anchored (ext0), free (phase a), free (phase b)
    m = SplitMatrix(VertexType.Z, rows=[0, 1, 2], external=['in'], internal=['x0', 'x1', 'x2'],
                    entries=[[1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]],
                    phases=[0, a, b], anchored=[True, False, False])
    pivots = reduce_rows(m)
    assert pivots == [1, 2]
    assert m.entries[0] == [1, 0, 0, 1]                      # anchored row absorbed the free rows
    assert m.phases[0] == a                                  # ... and only row 1, so its phase is a
    assert m.entries[1] == [0, 1, 0, 0] and m.entries[2] == [0, 0, 1, 1]
    assert str(phase_mod2(m.phases[1])) in ('a + b', 'b + a') and m.phases[2] == b
    # an anchored row is never a pivot even when it is the only one with a 1
    m2 = SplitMatrix(VertexType.Z, rows=[0, 1], external=['in'], internal=['x0'],
                     entries=[[1, 1], [0, 0]], phases=[0, 0], anchored=[True, False])
    assert reduce_rows(m2) == [] and m2.entries[0] == [1, 1]


def test_split_matrix_roles_for_gadgets():
    c = Circuit(2)
    c.add_gate(CNOT(0, 1)); c.add_gate(T(1))
    gpf = to_gpf(c.to_graph())
    mz = split_matrix(gpf, VertexType.Z)
    stems = [v for v in gpf.vertices() if is_stem(gpf, v)]
    assert stems and all(s in mz.external for s in stems if gpf.type(s) == VertexType.X)
    assert all(not is_leaf(gpf, r) for r in mz.rows)
    # rows connected to an input or to a gadget stem are anchored
    for r, anchored in zip(mz.rows, mz.anchored):
        touches_ext = any(gpf.connected(r, c) for c in mz.external)
        assert anchored == touches_ext


def test_phase_mod2_normalisation():
    a = new_var('a', True)
    assert phase_mod2(a + 1 + 1) == a
    assert phase_mod2(a + a) == 0 and phase_mod2(3) == 1 and phase_mod2(Fraction(0)) == 0
    assert isinstance(phase_mod2(a + 1), Poly)
