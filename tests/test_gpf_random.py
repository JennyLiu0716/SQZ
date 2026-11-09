"""Property-based checks of the GPF pipeline on random diagrams (fixed seed).

For every diagram: ``to_gpf`` must produce a diagram in GPF shape that is equal
to the input for *every* assignment of the Boolean variables, and ``reduce_gpf``
must be equal on the assignments its constraints allow. Zero diagrams are
compared with the scalar kept, because PyZX normalises by the scalar and that is
meaningless when the map is zero.
"""
import itertools
import random
from fractions import Fraction

from pyzx.graph import Graph
from pyzx.symbolic import new_var
from pyzx.utils import EdgeType, VertexType

from zxqec.gpf import boolean_variables, is_gpf, reduce_gpf, restore_boundary_hadamards, same_map, to_gpf


def random_diagram(rng, phases):
    """A random diagram; ``phases`` is 'none', 'pauli', 'boolean' or 'mixed'."""
    g = Graph()
    zs = [g.add_vertex(VertexType.Z, qubit=i, row=1) for i in range(rng.randint(1, 4))]
    xs = [g.add_vertex(VertexType.X, qubit=i, row=2) for i in range(rng.randint(1, 4))]
    spiders = zs + xs
    ins, outs = [], []
    for i in range(rng.randint(0, 2)):
        b = g.add_vertex(VertexType.BOUNDARY, qubit=i, row=0)
        g.add_edge((b, rng.choice(spiders)), EdgeType.SIMPLE)
        ins.append(b)
    for i in range(rng.randint(1, 3)):
        b = g.add_vertex(VertexType.BOUNDARY, qubit=i, row=3)
        g.add_edge((b, rng.choice(spiders)), EdgeType.SIMPLE)
        outs.append(b)
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    for _ in range(rng.randint(len(spiders), 2 * len(spiders))):
        a, b = rng.sample(spiders, 2)
        if not g.connected(a, b):
            g.add_edge((a, b), EdgeType.SIMPLE)
    n = 0
    for v in spiders:
        r = rng.random()
        if phases == 'pauli' and r < 0.4:
            g.set_phase(v, 1)
        elif phases == 'boolean' and r < 0.4:
            n += 1
            g.set_phase(v, new_var("c{}".format(n), True, g.var_registry))
        elif phases == 'mixed':
            if r < 0.25:
                n += 1
                g.set_phase(v, new_var("c{}".format(n), True, g.var_registry))
            elif r < 0.4:
                g.set_phase(v, Fraction(1, 4))          # non-Pauli: must become a gadget
    return g


def assignments(names):
    for values in itertools.product([0, 1], repeat=len(names)):
        yield dict(zip(names, values))


def test_to_gpf_on_random_diagrams():
    rng = random.Random(2026)
    for phases in ('none', 'pauli', 'boolean', 'mixed'):
        for _ in range(30):
            g = random_diagram(rng, phases)
            names = boolean_variables(g)
            if len(names) > 4:
                continue
            h = to_gpf(g)
            assert is_gpf(h), (phases, [(v, h.type(v), str(h.phase(v))) for v in h.vertices()])
            h = restore_boundary_hadamards(h)             # recolouring X-phases leaves boundary Hadamards
            for a in assignments(names):
                assert same_map(g.substitute_variables(a) if names else g,
                                h.substitute_variables(a) if names else h), (phases, a)


def test_detectors_are_sound_and_complete_on_random_diagrams():
    """Every violated constraint must give a zero branch, and every allowed branch must be non-zero.

    Comparing ``D`` with ``D'`` on the allowed branches (the test below) passes
    even when a reported detector is false, because those branches are skipped,
    and when one is missing, because then both sides vanish. This is the check
    that catches both. Completeness is asserted only for diagrams whose phases
    are all Pauli: with non-Pauli gadgets the reduction is sound but not
    complete (two gadgets sharing a parity can hide a detector).
    """
    import numpy as np
    rng = random.Random(4242)
    for phases in ('none', 'pauli', 'boolean'):
        for _ in range(30):
            g = random_diagram(rng, phases)
            names = boolean_variables(g)
            if len(names) > 4:
                continue
            res = reduce_gpf(g)
            for a in assignments(names):
                branch = g.substitute_variables(a) if names else g
                zero = np.abs(branch.to_tensor(preserve_scalar=True)).max() < 1e-12
                violated = res.is_zero or any(evaluate(c, a) for c in res.constraints)
                if violated:
                    assert zero, ("a reported detector excludes a branch that is not zero", phases, a)
                else:
                    assert not zero, ("a zero branch is not excluded by any detector", phases, a)


def test_reduce_gpf_on_random_diagrams():
    rng = random.Random(7)
    for phases in ('none', 'pauli', 'boolean', 'mixed'):
        for _ in range(30):
            g = random_diagram(rng, phases)
            names = boolean_variables(g)
            if len(names) > 4:
                continue
            res = reduce_gpf(g)
            assert is_gpf(res.graph)
            for a in assignments(names):
                if any(evaluate(c, a) for c in res.constraints):
                    continue                              # an outcome that cannot occur
                assert same_map(g.substitute_variables(a) if names else g,
                                res.to_graph().substitute_variables(a) if names else res.to_graph()), (phases, a)


def test_reduce_gpf_leaves_the_free_rows_in_reduced_echelon_form():
    # the promise of reduce_gpf's docstring, which code_of relies on: in each layer, every free row
    # (no input, gadget or non-Pauli phase on it) is non-empty and has a leading 1 in a column where
    # every other row of the layer, free or not, has a 0; so the free rows are independent and
    # one more Gauss-Jordan pass changes nothing
    import copy
    from zxqec.gpf import reduce_rows, split_matrix
    rng = random.Random(11)
    checked = 0
    for phases in ('none', 'pauli', 'boolean', 'mixed'):
        for _ in range(100):
            res = reduce_gpf(random_diagram(rng, phases))
            if res.is_zero:
                continue
            checked += 1
            for side in (VertexType.Z, VertexType.X):
                m = split_matrix(res.graph, side)
                ext = m.num_external
                for i in m.free_rows():
                    internal = m.entries[i][ext:]
                    assert any(internal), (phases, side, "an empty free row")
                    lead = ext + internal.index(1)
                    assert not any(m.entries[j][lead] for j in range(len(m.entries)) if j != i), (phases, side)
                again = copy.deepcopy(m)
                reduce_rows(again)
                assert again.entries == m.entries, (phases, side)
    assert checked > 300


def evaluate(poly, assignment):
    from zxqec.gpf import evaluate_phase
    return evaluate_phase(poly, assignment)
