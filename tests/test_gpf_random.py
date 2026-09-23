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

from zxqec.gpf import boolean_variables, is_gpf, reduce_gpf, same_map, to_gpf


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
            for a in assignments(names):
                assert same_map(g.substitute_variables(a) if names else g,
                                h.substitute_variables(a) if names else h), (phases, a)


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


def evaluate(poly, assignment):
    from zxqec.gpf import evaluate_phase
    return evaluate_phase(poly, assignment)
