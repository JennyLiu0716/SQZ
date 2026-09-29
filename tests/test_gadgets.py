"""The gadget layer simplified as a whole (zxqec.gpf.gadgets)."""
import itertools
import random
from fractions import Fraction

import pyzx as zx
from pyzx.graph import Graph
from pyzx.symbolic import new_var
from pyzx.utils import VertexType

from zxqec.gpf import boolean_variables, phase_polynomial, simplify_gadgets
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
