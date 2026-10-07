"""Tests for zxqec.gpf: GPF extraction and reduction with phases."""
import random
from fractions import Fraction

import pytest
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, T, XPhase, ZPhase
from pyzx.graph import Graph
from pyzx.symbolic import Poly, new_var
from pyzx.utils import EdgeType, VertexType, is_pauli

from zxqec.gpf import (
    same_map,
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


def test_non_pauli_phase_on_a_boundary_spider():
    # a degree-one spider on a boundary wire is a state, not a gadget leaf: its phase must be
    # unfused onto a gadget, or the boundary padding would put two same-coloured spiders in a row
    for ty, phase in [(VertexType.Z, Fraction(1, 4)), (VertexType.X, Fraction(1, 4)),
                      (VertexType.Z, Fraction(1, 2)), (VertexType.X, Fraction(3, 4))]:
        g = Graph()
        v = g.add_vertex(ty, qubit=0, row=1, phase=phase)
        i = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
        o = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=2)
        g.add_edge((i, v)); g.add_edge((v, o))
        g.set_inputs((i,)); g.set_outputs((o,))
        h = to_gpf(g)
        assert is_gpf(h), f"{ty} with phase {phase}: {[(w, h.type(w), str(h.phase(w))) for w in h.vertices()]}"
        # an X-phase is recoloured to a Z-phase between two boundary Hadamards (see hadamard.py)
        assert zx.compare_tensors(g, restore_boundary_hadamards(h), preserve_scalar=False)

    # the same with the spider only on one side (a state)
    g = Graph()
    v = g.add_vertex(VertexType.Z, qubit=0, row=1, phase=Fraction(1, 4))
    o = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=2)
    g.add_edge((v, o)); g.set_inputs(()); g.set_outputs((o,))
    h = to_gpf(g)
    assert is_gpf(h) and zx.compare_tensors(g, h, preserve_scalar=False)


def test_same_map_handles_zero_diagrams():
    # pyzx.compare_tensors normalises by the scalar, so a zero map compares unequal to itself;
    # same_map (and equal_for_all_assignments through it) must not be fooled by that
    z = Graph()                                  # <0|+...> style inconsistency: the zero map
    a = z.add_vertex(VertexType.Z, qubit=0, row=1, phase=1)
    b = z.add_vertex(VertexType.X, qubit=0, row=2)
    o = z.add_vertex(VertexType.BOUNDARY, qubit=0, row=3)
    z.add_edge((a, b)); z.add_edge((a, b)); z.add_edge((b, o))
    z.set_inputs(()); z.set_outputs((o,))
    import numpy as np
    assert np.abs(z.to_tensor(preserve_scalar=True)).max() < 1e-12      # it really is zero
    assert same_map(z, z.copy())
    assert equal_for_all_assignments(z, to_gpf(z))


def test_zero_verdict_uses_the_constraint_system():
    # {c, c + 1} has no solution although neither entry is the constant 1: every branch is zero
    from zxqec.gpf import constraint_system, detector_rank, unsatisfiable
    g = Graph()
    reg = g.var_registry
    c = new_var("c", True, reg)
    i = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
    o = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=2)
    w = g.add_vertex(VertexType.Z, qubit=0, row=1)
    g.add_edge((i, w)); g.add_edge((w, o))
    g.add_vertex(VertexType.Z, qubit=1, row=1, phase=c)            # isolated: 1 + (-1)^c
    g.add_vertex(VertexType.Z, qubit=2, row=1, phase=c + 1)        # isolated: 1 + (-1)^(c+1)
    g.set_inputs((i,)); g.set_outputs((o,))
    res = reduce_gpf(g)
    assert {str(x) for x in res.constraints} == {'c', 'c + 1'}
    assert res.is_zero                                             # neither constraint is constant
    assert unsatisfiable(res.constraints) and detector_rank(res.constraints) == 1
    for value in (0, 1):
        import numpy as np
        assert np.abs(g.substitute_variables({'c': value}).to_tensor(preserve_scalar=True)).max() < 1e-12


def test_isolated_pair_with_two_non_pauli_phases_is_not_dropped():
    # <X(3pi/2)|Z(pi/2)> is exactly zero, so such a pair may not be discarded as a unit scalar
    from zxqec.gpf.reduce import _remove_isolated_pairs
    for a, b, zero in [(Fraction(1, 2), Fraction(3, 2), True), (Fraction(3, 2), Fraction(1, 2), True),
                       (Fraction(1, 4), Fraction(1, 4), False), (1, Fraction(1, 4), False)]:
        g = Graph()
        u = g.add_vertex(VertexType.Z, qubit=0, row=0, phase=a)
        v = g.add_vertex(VertexType.X, qubit=0, row=1, phase=b)
        g.add_edge((u, v)); g.set_inputs(()); g.set_outputs(())
        import numpy as np
        value = complex(g.to_tensor(preserve_scalar=True))
        assert (abs(value) < 1e-12) == zero
        _remove_isolated_pairs(g)
        kept = g.num_vertices() == 2
        assert kept == (not is_pauli(a) and not is_pauli(b)), (a, b, kept)


def test_two_gadgets_on_one_parity_keep_their_detector():
    # Two gadgets on the same row with leaves pi/4 and 3pi/4 hide a Pauli phase: the branch c = 0
    # is the zero map, so c + 1 is a detector. It is found because identity removal collapses the
    # degree-two hubs and fuses the leaves into the row before the phases are isolated again.
    import numpy as np
    g = Graph()
    c = new_var("c", True, g.var_registry)
    i = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
    o = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=3)
    w = g.add_vertex(VertexType.Z, qubit=0, row=1)
    g.add_edge((i, w)); g.add_edge((w, o))                      # an identity wire alongside
    row = g.add_vertex(VertexType.Z, qubit=1, row=1, phase=c)
    for leaf_phase in (Fraction(1, 4), Fraction(3, 4)):
        hub = g.add_vertex(VertexType.X, qubit=1, row=1.5)
        leaf = g.add_vertex(VertexType.Z, qubit=1, row=2, phase=leaf_phase)
        g.add_edge((row, hub)); g.add_edge((hub, leaf))
    g.set_inputs((i,)); g.set_outputs((o,))
    assert np.abs(g.substitute_variables({'c': 0}).to_tensor(preserve_scalar=True)).max() < 1e-12
    assert np.abs(g.substitute_variables({'c': 1}).to_tensor(preserve_scalar=True)).max() > 1e-9
    res = reduce_gpf(g)
    assert [str(x) for x in res.constraints] == ['c + 1']


def test_identity_removal_before_pushing_hadamards():
    # An odd-Hadamard cycle through an identity spider denotes a Pauli Z; removing the identity
    # first turns the Hadamard edge into a phase, so the diagram is accepted rather than rejected
    g = Graph()
    i = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
    o = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=4)
    z1 = g.add_vertex(VertexType.Z, qubit=0, row=1)
    z2 = g.add_vertex(VertexType.Z, qubit=0, row=3)
    x = g.add_vertex(VertexType.X, qubit=1, row=2)
    g.add_edge((i, z1)); g.add_edge((z2, o)); g.add_edge((z1, x)); g.add_edge((x, z2))
    g.add_edge((z1, z2), EdgeType.HADAMARD)
    g.set_inputs((i,)); g.set_outputs((o,))
    h = to_gpf(g)                                                # no longer raises
    assert is_gpf(h) and zx.compare_tensors(g, h, preserve_scalar=False)


# -- runs, phases modulo the detectors, change of variables, layout ----------------------------------

def _outcomes(names, g=None):
    reg = g.var_registry if g is not None else None
    return {n: new_var(n, True, reg) for n in names}


def test_consistent_assignments_are_exactly_the_solutions():
    import itertools
    from zxqec.gpf import consistent_assignments, evaluate_phase
    v = _outcomes(['a', 'b', 'c', 'd'])
    constraints = [v['a'] + v['b'], v['b'] + v['c'] + 1]
    names = ['a', 'b', 'c', 'd']
    brute = [dict(zip(names, bits)) for bits in itertools.product([0, 1], repeat=4)
             if not any(evaluate_phase(c, dict(zip(names, bits))) for c in constraints)]
    found = consistent_assignments(constraints, names)
    assert sorted(map(sorted, map(dict.items, found))) == sorted(map(sorted, map(dict.items, brute)))
    assert len(found) == 4                                    # 4 variables, 2 independent constraints
    assert consistent_assignments([v['a'], v['a'] + 1], names) == []          # inconsistent
    assert consistent_assignments([v['a'] + v['d']], ['a']) == [{'a': 0}]     # d is not listed: it is 0
    with pytest.raises(ValueError):
        consistent_assignments([v['a'] * v['b']], names)


def test_equal_for_all_assignments_only_visits_the_allowed_runs():
    # 40 outcomes tied by 39 constraints: 2 runs, where trying every assignment would take 2^40
    g = Graph()
    names = ['c[{}]'.format(i) for i in range(40)]
    v = _outcomes(names, g)
    z = g.add_vertex(VertexType.Z, phase=sum((v[n] for n in names[1:]), v[names[0]]))
    b = g.add_vertex(VertexType.BOUNDARY)
    g.add_edge((z, b))
    g.set_outputs((b,))
    constraints = [v[names[i]] + v[names[i + 1]] for i in range(39)]
    assert equal_for_all_assignments(g, g.copy(), names, constraints)


def test_reduce_modulo_writes_a_parity_in_the_earliest_outcomes():
    from zxqec.gpf import reduce_modulo
    v = _outcomes(['c[1]', 'c[2]', 'c[3]'])
    reduced = reduce_modulo(v['c[1]'] + v['c[3]'] + Fraction(9, 4), [v['c[3]'] + v['c[2]'] + 1])
    assert str(reduced) == str(v['c[1]'] + v['c[2]'] + Fraction(5, 4))        # c3 = c2 + 1; 9/4 + 1 = 5/4 mod 2
    assert reduce_modulo(v['c[3]'] + v['c[2]'], [v['c[3]'] + v['c[2]']]) == 0
    product = v['c[1]'] * v['c[2]']
    assert reduce_modulo(product, [v['c[1]']]) is product                    # not a parity: unchanged


def test_phases_modulo_returns_a_rewritten_copy():
    from zxqec.gpf import phases_modulo
    g = Graph()
    v = _outcomes(['c[1]', 'c[2]'], g)
    z = g.add_vertex(VertexType.Z, phase=v['c[2]'])
    h = phases_modulo(g, [v['c[2]'] + v['c[1]']])
    assert str(h.phase(z)) == 'c[1]' and str(g.phase(z)) == 'c[2]'


def test_outcomes_fixed_by_the_constraints_leave_the_gadget_hubs():
    # T on the parity y + c and T-dagger on y, where a spider with phase c makes c = 0 a constraint:
    # in the only run that can occur the two cancel, and they must not survive as an S controlled by c
    g = Graph()
    c = new_var('c', True, g.var_registry)
    i = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
    row = g.add_vertex(VertexType.Z, qubit=0, row=1)
    o = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=3)
    g.add_edge((i, row)); g.add_edge((row, o))
    for hub_phase, leaf_phase in ((c, Fraction(1, 4)), (0, Fraction(7, 4))):
        hub = g.add_vertex(VertexType.X, qubit=1, row=1.5, phase=hub_phase)
        leaf = g.add_vertex(VertexType.Z, qubit=1, row=2, phase=leaf_phase)
        g.add_edge((row, hub)); g.add_edge((hub, leaf))
    g.add_vertex(VertexType.Z, qubit=2, row=1, phase=c)
    g.set_inputs((i,)); g.set_outputs((o,))
    res = reduce_gpf(g)
    assert [str(x) for x in res.constraints] == ['c']
    assert not any(is_leaf(res.graph, v) for v in res.graph.vertices())
    assert equal_for_all_assignments(g, res.graph, ['c'], res.constraints)


def test_only_fixed_outcomes_are_substituted_in_a_hub_parity():
    from zxqec.gpf.reduce import _substitute_in_parity
    v = _outcomes(['c[1]', 'c[2]', 'c[3]'])
    assert str(_substitute_in_parity(v['c[3]'] + v['c[1]'] + v['c[2]'], {'c[1]': 0})) == 'c[3] + c[2]'
    assert _substitute_in_parity(v['c[3]'] + v['c[1]'], {'c[1]': 1, 'c[3]': 1}) == 0
    assert _substitute_in_parity(v['c[2]'] + v['c[3]'], {'c[1]': 1}) is None          # nothing to substitute
    assert _substitute_in_parity(v['c[1]'] * v['c[2]'], {'c[1]': 0}) is None          # not a parity


def _two_bit_state(t_on_first=True):
    # sum over a, b of |a, b, a + b>, with T on qubit 0 when asked
    c = Circuit(3)
    if t_on_first:
        c.add_gate(T(0))
    c.add_gate(CNOT(0, 2))
    c.add_gate(CNOT(1, 2))
    g = c.to_graph()
    g.apply_state('++0')
    return g


def test_change_variables_sets_the_rows_and_keeps_the_map():
    from zxqec.gpf import change_variables
    g = _two_bit_state(t_on_first=False)
    d = to_gpf(g)
    for rows in ([[1, 1, 0], [0, 1, 1]], [[1, 0, 1], [1, 1, 0]]):
        h = change_variables(d, rows)
        outs = [next(iter(h.neighbors(b))) for b in h.outputs()]
        zs = sorted(v for v in h.vertices() if h.type(v) == VertexType.Z and not is_leaf(h, v))
        assert [[int(h.connected(z, w)) for w in outs] for z in zs] == rows
        assert same_map(h, g)
    with pytest.raises(ValueError):
        change_variables(d, [[1, 0, 0], [0, 1, 1]])               # [1, 0, 0] is not in the span


def test_change_variables_fuses_a_gadget_that_reads_one_spider():
    from zxqec.gpf import change_variables
    g = _two_bit_state()                                          # T on the bit a
    d = to_gpf(g)
    assert sum(is_leaf(d, v) for v in d.vertices()) == 1
    rows = [[1, 0, 1], [0, 1, 1]]                                 # a and b
    kept = change_variables(d, rows, fuse=False)
    fused = change_variables(d, rows)
    assert sum(is_leaf(kept, v) for v in kept.vertices()) == 1
    assert sum(is_leaf(fused, v) for v in fused.vertices()) == 0
    assert any(fused.phase(v) == Fraction(1, 4) for v in fused.vertices() if fused.type(v) == VertexType.Z)
    assert same_map(kept, g) and same_map(fused, g)


def test_layout_gpf_puts_the_layers_side_by_side():
    from zxqec.gpf import layout_gpf
    d = to_gpf(_two_bit_state())
    layout_gpf(d)
    outs = list(d.outputs())
    xs = [next(iter(d.neighbors(b))) for b in outs]
    zs = [v for v in d.vertices() if d.type(v) == VertexType.Z and not is_leaf(d, v)]
    assert [d.qubit(x) for x in xs] == [0, 1, 2]                  # output j on line j
    assert all(d.row(z) == 1 for z in zs)
    assert all(d.row(x) > 1 for x in xs) and all(d.row(b) == d.row(xs[0]) + 1 for b in outs)
    for leaf in (v for v in d.vertices() if is_leaf(d, v)):
        (hub,) = list(d.neighbors(leaf))
        assert d.qubit(leaf) == d.qubit(hub) and d.row(leaf) > d.row(hub)


def test_to_gpf_steps_records_the_steps_of_to_gpf():
    # the diagram after each step of to_gpf: the first is the input, the last to_gpf's result, and every one is the
    # same map (up to a scalar)
    import pyzx as zx
    from pyzx.circuit import Circuit
    from pyzx.circuit.gates import CNOT, HAD, T
    from zxqec.gpf import to_gpf, to_gpf_steps
    c = Circuit(3)
    for g in (HAD(0), CNOT(0, 1), T(1), CNOT(1, 2), HAD(2), CNOT(0, 2)):
        c.add_gate(g)
    g = c.to_graph()
    steps = to_gpf_steps(g)
    assert [what for what, _ in steps][0] == 'the diagram' and len(steps) == 6
    final, direct = steps[-1][1], to_gpf(g)
    assert final.num_vertices() == direct.num_vertices() and final.num_edges() == direct.num_edges()
    assert all(zx.compare_tensors(h, g, preserve_scalar=False) for _, h in steps)


def test_reduce_gpf_of_a_reduced_diagram():
    # A reduced diagram carries the gadget roles of its reduction. Reducing it again first removes every phase-free
    # hub of degree two and fuses its leaf with the spider the hub read, which must not stay marked as a leaf: to_gpf
    # drops the old roles. The result is the same map on every run its detectors allow
    from .programs import steane_t_injection
    c = Circuit(2)
    for gate in (T(0), CNOT(0, 1), T(1)):
        c.add_gate(gate)
    for g in (reduce_gpf(c.to_graph()).to_graph(), steane_t_injection().logical_action().graph):
        assert any(is_leaf(g, v) for v in g.vertices())
        res = reduce_gpf(g)
        assert is_gpf(res.graph) and not res.is_zero
        assert equal_for_all_assignments(g, res.to_graph(), constraints=res.detectors)


def test_copy_leaf_copies_a_measurement_result_and_refuses_what_the_rule_does_not_cover():
    from zxqec.gpf import copy_leaf
    # an X-leaf with phase c[0] on a Z-spider with three other legs: Z-measurement result c[0] of a copied qubit
    g = Graph()
    a = new_var('c[0]', True, g.var_registry)
    ins = [g.add_vertex(VertexType.BOUNDARY, q, 0) for q in range(2)]
    outs = [g.add_vertex(VertexType.BOUNDARY, q, 3) for q in range(2)]
    z = g.add_vertex(VertexType.Z, 0, 1)
    x = g.add_vertex(VertexType.X, 1, 1)
    leaf = g.add_vertex(VertexType.X, 0, 2, a)
    g.add_edges([(ins[0], z), (z, outs[0]), (z, x), (ins[1], x), (x, outs[1]), (z, leaf)])
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    h = g.clone()
    new = copy_leaf(h, leaf)
    assert set(new) == {ins[0], outs[0], x} and all(h.phase(u) == a and h.vertex_degree(u) == 1 for u in new.values())
    assert equal_for_all_assignments(g, h)
    # the X copy that reaches the CX target fuses with it
    assert zx.rewrite_rules.fuse_rule.fuse(h, x, new[x]) and h.phase(x) == a
    assert equal_for_all_assignments(g, h)
    with pytest.raises(ValueError, match='one leg'):
        copy_leaf(g.clone(), z)
    k = g.clone()
    k.set_phase(z, Fraction(1, 2))
    with pytest.raises(ValueError, match='phase-0 spider'):
        copy_leaf(k, leaf)
    k = g.clone()
    k.set_type(leaf, VertexType.Z)
    with pytest.raises(ValueError, match='other colour'):
        copy_leaf(k, leaf)
    k = g.clone()
    k.set_phase(leaf, Fraction(1, 4))
    with pytest.raises(ValueError, match='not 0, pi'):
        copy_leaf(k, leaf)
