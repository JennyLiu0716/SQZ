"""The gadget layer simplified as a whole (zxqec.gpf.gadgets)."""
import cmath
import itertools
import random
from fractions import Fraction

import numpy as np
import pyzx as zx
from pyzx.graph import Graph
from pyzx.symbolic import new_var
from pyzx.utils import EdgeType, VertexType

from zxqec.gpf import (boolean_variables, is_gpf, is_leaf, normalize_gadget_angles, open_leaves, phase_polynomial,
                       reduce_gpf, same_map, simplify_gadgets, to_gpf)
from zxqec.gpf.normal_form import GPF_ROLE_KEY, LEAF


def _layer(nz, gadgets):
    """``nz`` Z-spiders, each between an input and an output, with gadgets (angle, reads, hub vars, leaf vars)."""
    g = Graph()
    reg = g.var_registry
    zs, ins, outs = [], [], []
    for i in range(nz):
        b = g.add_vertex(VertexType.BOUNDARY, i, 0)
        z = g.add_vertex(VertexType.Z, i, 1)
        x = g.add_vertex(VertexType.X, i, 2)
        o = g.add_vertex(VertexType.BOUNDARY, i, 3)
        g.add_edge((b, z)); g.add_edge((z, x)); g.add_edge((x, o))
        zs.append(z); ins.append(b); outs.append(o)
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    for angle, reads, hub_vars, leaf_vars in gadgets:
        hp, lp = 0, Fraction(angle)
        for c in hub_vars:
            hp = hp + new_var(c, True, reg)
        for c in leaf_vars:
            lp = lp + new_var(c, True, reg)
        hub = g.add_vertex(VertexType.X, 5, 5, hp)
        leaf = g.add_vertex(VertexType.Z, 6, 6, lp)
        g.set_vdata(leaf, GPF_ROLE_KEY, LEAF)
        g.add_edge((hub, leaf))
        for r in reads:
            g.add_edge((hub, zs[r]))
    return g


def _same_for_all_outcomes(a, b):
    names = sorted(set(boolean_variables(a)) | set(boolean_variables(b)))
    for vals in itertools.product((0, 1), repeat=len(names)):
        asg = dict(zip(names, vals))
        if not zx.compare_tensors(a.substitute_variables(asg) if names else a,
                                  b.substitute_variables(asg) if names else b, preserve_scalar=False):
            return False
    return True


def _leaves(g):
    return sum(1 for v in g.vertices() if g.vdata(v, GPF_ROLE_KEY, None) == LEAF)


def test_random_layers_keep_their_map():
    rng = random.Random(3)
    for _ in range(80):
        nz = rng.randint(1, 4)
        gadgets = [(Fraction(rng.choice([1, 2, 3, 5, 6, 7]), 4), rng.sample(range(nz), rng.randint(1, nz)),
                    rng.sample(['a', 'b'], rng.randint(0, 1)), rng.sample(['a', 'b'], rng.randint(0, 1)))
                   for _ in range(rng.randint(1, 5))]
        g = _layer(nz, gadgets)
        h = g.copy()
        simplify_gadgets(h)
        assert _same_for_all_outcomes(g, h)


def test_t_and_t_dagger_on_one_parity_cancel():
    g = _layer(2, [(Fraction(1, 4), [0, 1], [], []), (Fraction(7, 4), [0, 1], [], [])])
    assert simplify_gadgets(g) and _leaves(g) == 0


def test_a_triorthogonal_layer_is_the_identity():
    # thesis Fig. 4.11: 16 pi/4 gadgets on 5 bits whose parity matrix is triorthogonal
    m_t = [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
           [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
           [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
           [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]]
    g = _layer(5, [(Fraction(1, 4), [i for i in range(5) if m_t[i][j]], [], []) for j in range(16)])
    assert phase_polynomial(g)[0] == {}
    h = g.copy()
    simplify_gadgets(h)
    assert _leaves(h) == 0 and _same_for_all_outcomes(g, h)


def test_the_fewest_gadgets_are_chosen():
    # S on three bits, S^dag on each pair and T on all three: one T^dag gadget on the three
    g = _layer(3, [(Fraction(1, 2), [i], [], []) for i in range(3)]
               + [(Fraction(3, 2), list(p), [], []) for p in itertools.combinations(range(3), 2)]
               + [(Fraction(1, 4), [0, 1, 2], [], [])])
    h = g.copy()
    simplify_gadgets(h)
    assert _leaves(h) == 1 and _same_for_all_outcomes(g, h)


def test_simplifying_twice_changes_nothing():
    g = _layer(3, [(Fraction(1, 4), [0, 1], ['a'], []), (Fraction(3, 4), [1, 2], [], ['b'])])
    simplify_gadgets(g)
    assert not simplify_gadgets(g)


def _read_spiders(g):
    return [next(iter(g.neighbors(b))) for b in g.inputs()]


def _angles(g):
    return sorted(Fraction(g.phase(ls[0])) for _hub, ls in phase_polynomial(g)[1]
                  if not hasattr(g.phase(ls[0]), 'terms'))


def test_normalized_angles_keep_the_map():
    rng = random.Random(7)
    for _ in range(80):
        nz = rng.randint(1, 4)
        gadgets = [(Fraction(rng.choice([1, 2, 3, 5, 6, 7]), 4), rng.sample(range(nz), rng.randint(1, nz)),
                    rng.sample(['a', 'b'], rng.randint(0, 1)), rng.sample(['a', 'b'], rng.randint(0, 1)))
                   for _ in range(rng.randint(1, 5))]
        g = _layer(nz, gadgets)
        h = g.copy()
        normalize_gadget_angles(h)
        assert _same_for_all_outcomes(g, h)
        assert all(not Fraction(1, 2) < a <= Fraction(3, 2) for a in _angles(h))
        assert not normalize_gadget_angles(h)


def test_five_pi_over_four_is_pi_over_four_and_a_pi_on_each_read_spider():
    g = _layer(3, [(Fraction(5, 4), [0, 2], ['f'], [])])
    assert normalize_gadget_angles(g)
    assert _angles(g) == [Fraction(1, 4)]
    assert [g.phase(z) for z in _read_spiders(g)] == [1, 0, 1]


def test_seven_pi_over_four_and_pi_over_two_stay():
    g = _layer(2, [(Fraction(7, 4), [0, 1], [], []), (Fraction(1, 2), [0], [], [])])
    assert not normalize_gadget_angles(g)


def test_an_angle_of_pi_leaves_no_gadget():
    g = _layer(2, [(Fraction(1), [0, 1], ['a'], [])])
    h = g.copy()
    assert normalize_gadget_angles(h)
    assert not phase_polynomial(h)[1] and [h.phase(z) for z in _read_spiders(h)] == [1, 1]
    assert _same_for_all_outcomes(g, h)


def test_outcomes_on_the_leaf_stay_on_the_leaf():
    g = _layer(2, [(Fraction(3, 2), [0, 1], [], ['c'])])
    h = g.copy()
    assert normalize_gadget_angles(h)
    (_hub, (leaf,)), = phase_polynomial(h)[1]
    assert {v.name for v in h.phase(leaf).free_vars()} == {'c'} and _same_for_all_outcomes(g, h)


def test_a_circuit_in_gpf_stays_in_gpf_with_normalized_angles():
    c = zx.Circuit(3)
    for gate in [("T", 0), ("CNOT", 0, 1), ("T", 1), ("Z", 1), ("CNOT", 1, 2), ("S", 2), ("Z", 2), ("T", 2)]:
        c.add_gate(*gate)
    d = to_gpf(c.to_graph())
    h = d.copy()
    assert normalize_gadget_angles(h)
    assert is_gpf(h) and zx.compare_tensors(d, h, preserve_scalar=False)


def _two_t_gates():
    # T on qubit 0 and T^dag on the parity of qubits 0 and 1, from |+>|+>: two gadgets after reduce_gpf
    c = zx.Circuit(2)
    for q in range(2):
        c.add_gate("HAD", q)
    c.add_gate("T", 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("T", 1, adjoint=True)
    c.add_gate("CNOT", 0, 1)
    g = c.to_graph()
    g.apply_state('00')
    return reduce_gpf(g, gadgets=False).to_graph()


def test_opened_leaves_leave_a_clifford_diagram_with_one_output_per_leaf():
    g = _two_t_gates()
    n = sum(1 for v in g.vertices() if is_leaf(g, v))
    h, new = open_leaves(g)
    assert n == 2 and len(new) == n and list(h.outputs()) == list(g.outputs()) + new
    assert all(Fraction(h.phase(v)).denominator <= 2 for v in h.vertices())


def test_plugging_the_leaves_back_gives_the_diagram():
    g = _two_t_gates()
    leaves = sorted(v for v in g.vertices() if is_leaf(g, v))
    h, _new = open_leaves(g)
    t = h.to_tensor(preserve_scalar=True)               # outputs in order: the old ones, then the new ones
    for leaf in reversed(leaves):                       # the last output with <0| + e^(i a pi)<1|
        t = np.tensordot(t, np.array([1, cmath.exp(1j * np.pi * float(g.phase(leaf)))]), axes=([t.ndim - 1], [0]))
    assert zx.compare_tensors(t, g.to_tensor(preserve_scalar=True), preserve_scalar=False)


def test_a_z_on_an_opened_wire_is_a_pi_on_the_spiders_its_hub_reads():
    g = _two_t_gates()
    h, (b0, _b1) = open_leaves(g)
    (hub,) = list(h.neighbors(b0))
    with_z = h.clone()
    with_z.remove_edge(with_z.edge(hub, b0))
    z = with_z.add_vertex(VertexType.Z, phase=1)
    with_z.add_edge((hub, z))
    with_z.add_edge((z, b0))
    moved = h.clone()
    for v in h.neighbors(hub):
        if h.type(v) == VertexType.Z:
            moved.add_to_phase(v, 1)
    assert same_map(with_z, moved)
    assert not same_map(with_z, h)


def test_hubs_that_are_not_of_gpf_shape_are_left_alone():
    # two leaves on one hub: the hub fixes only the parity of the reads and the other leaf
    g = _layer(2, [(Fraction(5, 4), [0, 1], [], [])])
    (hub,) = [v for v in g.vertices() if g.type(v) == VertexType.X and g.row(v) == 5]
    extra = g.add_vertex(VertexType.Z, 7, 7, Fraction(5, 4))
    g.set_vdata(extra, GPF_ROLE_KEY, LEAF)
    g.add_edge((hub, extra))
    h = g.copy()
    assert not normalize_gadget_angles(h) and _same_for_all_outcomes(g, h)
    # a Hadamard edge from the hub to a spider it reads
    g = _layer(2, [(Fraction(5, 4), [0], [], [])])
    (hub,) = [v for v in g.vertices() if g.type(v) == VertexType.X and g.row(v) == 5]
    z = next(w for w in g.neighbors(hub) if not is_leaf(g, w))
    g.remove_edge(g.edge(hub, z))
    g.add_edge((hub, z), EdgeType.HADAMARD)
    assert not normalize_gadget_angles(g.copy())
