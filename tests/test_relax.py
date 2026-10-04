"""zxqec.spacetime.relax: diagrams laid out where a physical energy is lowest, with no coordinates given."""
import numpy as np
import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, T

from zxqec.gpf import reduce_gpf
from zxqec.spacetime import RELAX_METHODS, draw_relaxed, draw_relaxed_3d, relax_layout

pytest.importorskip('networkx')
pytest.importorskip('scipy')


def _diagram():
    # a normal form with two connected parts: a T gadget on a parity of qubits 0 and 1, and qubit 2 alone
    c = Circuit(3)
    for g in (HAD(0), CNOT(0, 1), T(1), CNOT(0, 1), HAD(2)):
        c.add_gate(g)
    return reduce_gpf(c.to_graph()).to_graph()


@pytest.mark.parametrize('method', RELAX_METHODS)
@pytest.mark.parametrize('dim', [2, 3])
def test_every_vertex_gets_a_reproducible_place(method, dim):
    g = _diagram()
    pos = relax_layout(g, dim, method, seed=3)
    assert set(pos) == set(g.vertices()) and all(len(p) == dim for p in pos.values())
    assert pos == relax_layout(g, dim, method, seed=3)          # seeded: the same layout every time
    lengths = [np.linalg.norm(np.subtract(pos[a], pos[b])) for a, b in (g.edge_st(e) for e in g.edges())]
    assert 0.5 < np.mean(lengths) < 2.0                        # an edge is about one unit long


def test_connected_parts_are_side_by_side():
    import networkx as nx
    g = _diagram()
    graph = nx.Graph()
    graph.add_nodes_from(g.vertices())
    graph.add_edges_from(g.edge_st(e) for e in g.edges())
    parts = list(nx.connected_components(graph))
    assert len(parts) == 2
    pos = relax_layout(g, 2, gap=1.5)
    spans = sorted((min(pos[v][0] for v in p), max(pos[v][0] for v in p)) for p in parts)
    assert spans[1][0] - spans[0][1] == pytest.approx(1.5)


def test_bad_arguments_and_the_drawings():
    import matplotlib
    matplotlib.use('Agg')
    g = _diagram()
    with pytest.raises(ValueError):
        relax_layout(g, method='annealing')
    with pytest.raises(ValueError):
        relax_layout(g, dim=4)
    fig = draw_relaxed(g)
    assert fig.axes                                            # PyZX's own drawing
    fig = draw_relaxed_3d(g, show_phases=False)
    ax = fig.axes[0]
    assert ax.get_zlabel() == '' and list(ax.get_zticks()) == []   # the positions are neither space nor time
    import matplotlib.pyplot as plt
    plt.close('all')
