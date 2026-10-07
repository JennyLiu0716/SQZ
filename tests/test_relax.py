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


# -- codes by their shape --------------------------------------------------------------------------------

def _cube_faces():
    return [[1 if ((q >> axis) & 1) == side else 0 for q in range(8)] for axis in range(3) for side in (0, 1)]


def _reed_muller():
    return [[(c >> bit) & 1 for c in range(1, 16)] for bit in range(4)]


def test_check_diagrams_and_the_surface_code():
    from pyzx.css import generate_css_encoder_graph
    from pyzx.linalg import Mat2
    from pyzx.utils import VertexType
    from zxqec.codes import check_diagram, code_of, surface_code_checks
    from zxqec.codes.gf2 import rank
    for d in (2, 3, 4, 5):
        hx, hz = surface_code_checks(d)
        assert not ((hx.astype(int) @ hz.T.astype(int)) % 2).any()
        assert d * d - rank(hx) - rank(hz) == 1
    hx, hz = surface_code_checks(3)
    enc, _ = generate_css_encoder_graph(Mat2(hx.tolist()), Mat2([[1 if q % 3 == 0 else 0 for q in range(9)]]), 'Z-X')
    enc.auto_detect_io()
    code = code_of(enc)
    assert code.parameters == (9, 1, 3)
    g = check_diagram(code)                                    # from the code: its X-type generators
    kinds = [g.type(v) for v in g.vertices()]
    assert kinds.count(VertexType.Z) == 4 and kinds.count(VertexType.X) == 9 and len(g.outputs()) == 9
    h = check_diagram(hz, 'Z')                                  # Z-checks: the colours swapped
    assert [h.type(v) for v in h.vertices()].count(VertexType.X) == 4
    with pytest.raises(ValueError):
        check_diagram(hx, 'Y')


def test_outputs_point_outward_and_the_shapes_of_codes():
    from itertools import combinations
    from pyzx.utils import VertexType
    from zxqec.codes import check_diagram
    g = check_diagram(_cube_faces(), 'Z')
    pos = relax_layout(g, 3, boundaries='outward')
    for b in g.outputs():                                       # each output one unit from its spider
        (v,) = g.neighbors(b)
        assert abs(np.linalg.norm(np.subtract(pos[b], pos[v])) - 1) < 1e-9
    corners = [np.array(pos[v]) for v in g.vertices() if g.type(v) == VertexType.Z]
    d = sorted(np.linalg.norm(a - b) for a, b in combinations(corners, 2))
    d = np.array(d) / d[0]                                      # a cube: 12 edges, 12 face diagonals, 4 diagonals
    assert np.allclose(d[:12], 1, atol=1e-2) and np.allclose(d[12:24], 2 ** 0.5, atol=1e-2)
    assert np.allclose(d[24:], 3 ** 0.5, atol=1e-2)
    g = check_diagram(_reed_muller())                           # [[15,1,3]]: a tetrahedron
    pos = relax_layout(g, 3, boundaries='outward')
    qubits = sorted(v for v in g.vertices() if g.type(v) == VertexType.X)
    weight = {v: bin(c).count('1') for v, c in zip(qubits, range(1, 16))}
    middle = np.mean([pos[v] for v in qubits], axis=0)
    corners = [np.array(pos[v]) for v in qubits if weight[v] == 1]
    d = [np.linalg.norm(a - b) for a, b in combinations(corners, 2)]
    assert max(d) - min(d) < 1e-2 * max(d)                      # four corners, all equally far apart
    radius = {w: np.mean([np.linalg.norm(np.subtract(pos[v], middle)) for v in qubits if weight[v] == w])
              for w in (1, 2, 3, 4)}
    assert radius[1] > radius[2] > radius[3] > radius[4] and radius[4] < 1e-2 * radius[1]
    assert relax_layout(g, 3) == relax_layout(g, 3, boundaries='relax')
    with pytest.raises(ValueError):
        relax_layout(g, 3, boundaries='inward')


def test_shaded_drawings_and_views():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from zxqec.codes import check_diagram
    steane = [[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]]
    fig = draw_relaxed(check_diagram(steane), shade='Z', boundaries='outward')
    assert sum(1 for p in fig.axes[0].patches if p.get_alpha() == 0.35) == 3        # one face per check
    fig = draw_relaxed_3d(check_diagram(_reed_muller()), shade='Z', boundaries='outward', box=False,
                          views=[(0, 0), (30, 60)], titles=['a', 'b'])
    assert len(fig.axes) == 2 and [ax.get_title() for ax in fig.axes] == ['a', 'b']
    assert all(sum(isinstance(c, Poly3DCollection) for c in ax.collections) == 4 for ax in fig.axes)
    with pytest.raises(ValueError):
        draw_relaxed(check_diagram(steane), shade='Y')
    plt.close('all')
