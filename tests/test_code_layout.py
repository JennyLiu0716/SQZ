"""StabilizerCode.layout_checks and draw_layout: a code drawn on the hardware layout (examples/FoldTransversalHadamard)."""
import os

import matplotlib

matplotlib.use('Agg')

import pytest  # noqa: E402
from pyzx.circuit import Circuit  # noqa: E402
from pyzx.circuit.gates import CZ, HAD  # noqa: E402
from pyzx.css import generate_css_encoder_graph  # noqa: E402
from pyzx.linalg import Mat2  # noqa: E402

from zxqec.codes import code_of, surface_code_checks  # noqa: E402

COORDS = {q: (q % 3, q // 3) for q in range(9)}             # qubit 3 r + c at column c, row r
ROTATE = [2, 5, 8, 1, 4, 7, 0, 3, 6]                         # qubit q sits at position ROTATE[q] (a quarter turn)


def _surface_code(positions=None):
    hx, hz = surface_code_checks(3)
    enc, _ = generate_css_encoder_graph(Mat2(hx.tolist()), Mat2([[1 if q % 3 == 0 else 0 for q in range(9)]]), 'Z-X')
    enc.auto_detect_io()
    return code_of(enc, positions=positions), hx, hz


def _faces(h, where=range(9)):
    return sorted(tuple(sorted(where[int(j)] for j in row.nonzero()[0])) for row in h)


def test_the_checks_drawn_are_the_faces():
    code, hx, hz = _surface_code()
    checks = code.layout_checks(COORDS)
    found = sorted(tuple(sorted(c)) for c in checks['X']) + sorted(tuple(sorted(c)) for c in checks['Z'])
    # the faces of the surface code, X and Z in either order (the encoder decides which type is which)
    assert found in (_faces(hx) + _faces(hz), _faces(hz) + _faces(hx))
    assert {len(c) for c in checks['X'] + checks['Z']} == {2, 4}


def test_a_code_after_a_permutation_is_drawn_where_its_qubits_sit():
    code, hx, hz = _surface_code(positions=ROTATE)
    checks = code.layout_checks(COORDS)
    found = sorted(tuple(sorted(c)) for c in checks['X']) + sorted(tuple(sorted(c)) for c in checks['Z'])
    assert found in (_faces(hx, ROTATE) + _faces(hz, ROTATE), _faces(hz, ROTATE) + _faces(hx, ROTATE))


def test_the_drawing():
    code, _hx, _hz = _surface_code()
    fig = code.draw_layout(COORDS, title='the surface code')
    ax = fig.axes[0]
    assert len(ax.patches) == 8                               # four X checks and four Z checks
    assert sum(1 for line in ax.lines if line.get_linestyle() == '--') == 2      # logical X and Z
    assert sum(1 for line in ax.lines if line.get_marker() == 'o') == 9          # the qubits
    bare = code.draw_layout(COORDS, logicals=False)
    assert sum(1 for line in bare.axes[0].lines if line.get_linestyle() == '--') == 0


def test_missing_coordinates_and_codes_that_are_not_css():
    code, _hx, _hz = _surface_code()
    with pytest.raises(ValueError):
        code.layout_checks({q: COORDS[q] for q in range(8)})
    c = Circuit(2)
    for g in (HAD(0), HAD(1), CZ(0, 1)):
        c.add_gate(g)
    g = c.to_graph()
    g.apply_state('00')
    graph_state = code_of(g)                                  # stabilized by XZ and ZX
    assert not graph_state.is_css
    with pytest.raises(NotImplementedError):
        graph_state.layout_checks({0: (0, 0), 1: (1, 0)})


# -- the encoder drawn on the layout (paper: the levels figure of the fold-transversal Hadamard) -------------------

FOLD_H = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'ft_h')
GRID = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (1, 2), 5: (2, 2), 6: (3, 2), 7: (1, 3), 8: (2, 3), 9: (3, 3)}


def _fold_h_codes():
    """The code after the encoder and X checks (signs c[1..4] on its X checks), and after the transversal H."""
    import pyzx as zx
    from zxqec.qprogram import QProgram
    p = QProgram(19, bits=35, initial_states={**{q: '0' for q in range(1, 10)}, 18: '0',
                                             **{a: '+' for a in range(10, 18)}})
    for name in ('ft_h', 'ft_h_2'):
        p.add_circuit(zx.Circuit.load(os.path.join(FOLD_H, name + '.qasm')))
    return p.code_at(0), p.code_at(1)


def test_the_encoder_in_the_other_colours_is_the_same_map():
    from pyzx.utils import VertexType
    from zxqec.gpf import equal_for_all_assignments
    for code in _fold_h_codes():
        d = code.data()
        a, b = d.encoder(), d.encoder(checks='Z')
        assert equal_for_all_assignments(a, b)                  # every value of the signs c[1..4]
        assert equal_for_all_assignments(a.adjoint(), b.adjoint())
        # X-spiders on the Z checks, Z-spiders on the qubits
        assert sum(1 for v in b.vertices() if b.type(v) == VertexType.X) == 4 + d.k
        assert sum(1 for v in b.vertices() if b.type(v) == VertexType.Z) == d.n
    after_encoder, after_h = (c.data() for c in _fold_h_codes())
    assert not equal_for_all_assignments(after_encoder.encoder(), after_h.encoder(checks='Z'))
    with pytest.raises(ValueError):
        after_h.encoder(checks='Y')


def test_draw_encoder_puts_each_spider_where_it_is_drawn_by_hand():
    import matplotlib.pyplot as plt
    after_encoder, after_h = _fold_h_codes()
    spacing = 1.6
    for code, adjoint in ((after_encoder, False), (after_h, True)):
        ax = code.draw_encoder(GRID, adjoint=adjoint, label_size=7).axes[0]
        spiders = {(round(float(c.center[0]), 6), round(float(c.center[1]), 6))
                   for c in ax.patches if type(c).__name__ == 'Circle' and c.radius == 0.2}
        # every qubit's spider at its place (y down), the four checks with their signs as labels
        assert {(round(spacing * x, 6), round(-spacing * y, 6)) for x, y in GRID.values()} <= spiders
        assert len(spiders) == 9 + 4 + 1                       # the qubits, the X (or Z) checks, the logical
        assert sorted(t.get_text() for t in ax.texts) == ['c[1]', 'c[2]', 'c[3]', 'c[4]']
    # after the H the signs are on the Z checks, so 'auto' draws them as X-spiders, each with its sign
    ax = after_h.draw_encoder(GRID, label_size=7).axes[0]
    red = [c for c in ax.patches if type(c).__name__ == 'Circle' and c.radius == 0.2 and c.get_facecolor()[1] < 0.6]
    assert len(red) == 4 + 1                                    # the four Z checks and the logical input
    assert len(after_h.draw_encoder(GRID, checks='X', label_size=7).axes[0].texts) > 0
    plt.close('all')


def _css(hx, lx):
    enc, _ = generate_css_encoder_graph(Mat2(hx), Mat2(lx), 'Z-X')
    enc.auto_detect_io()
    return code_of(enc)


def _apart(ax):
    """No two spiders within 0.4 of each other, and no wire from a qubit running along that qubit's stub."""
    import numpy as np
    centres = [np.array(p.center, float) for p in ax.patches if type(p).__name__ == 'Circle']
    if any(np.linalg.norm(a - b) < 0.4 for i, a in enumerate(centres) for b in centres[i + 1:]):
        return False
    segs = [np.array(list(zip(line.get_xdata(), line.get_ydata())), float) for line in ax.lines]
    stubs = [sg for sg in segs if abs(np.linalg.norm(sg[1] - sg[0]) - 0.45) < 1e-6]
    for st in stubs:
        q, tip = (st[0], st[1]) if any(np.linalg.norm(st[0] - c) < 1e-6 for c in centres) else (st[1], st[0])
        for sg in segs:
            if sg is st or not any(np.linalg.norm(e - q) < 1e-6 for e in sg):
                continue
            other = sg[1] if np.linalg.norm(sg[0] - q) < 1e-6 else sg[0]
            d1, d2 = tip - q, other - q
            if abs(d1[0] * d2[1] - d1[1] * d2[0]) < 1e-6 and d1 @ d2 > 0:
                return False
    return True


def test_draw_encoder_keeps_its_spiders_and_wires_apart():
    """No spider on another, no wire along a stub: odd and even logicals, k > 1, blocks side by side, decoders."""
    import matplotlib.pyplot as plt
    import numpy as np
    hx = surface_code_checks(3)[0]
    two = np.zeros((2 * hx.shape[0], 18), int)
    two[:hx.shape[0], :9], two[hx.shape[0]:, 9:] = hx, hx
    codes = [
        (_surface_code(positions=ROTATE)[0], COORDS),
        (_css([[1, 1, 1, 1]], [[1, 1, 0, 0], [1, 0, 1, 0]]), {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}),   # [[4,2,2]]
        (_css(surface_code_checks(4)[0].tolist(), [[1 if q % 4 == 0 else 0 for q in range(16)]]),
         {q: (q % 4, q // 4) for q in range(16)}),                                                            # d = 4
        (_css([[1] * 8], [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]]),
         {0: (0, 0), 1: (3, 0), 2: (0, 3), 3: (3, 3), 4: (1, 1), 5: (2, 1), 6: (1, 2), 7: (2, 2)}),            # [[8,3,2]]
        (_css(two.tolist(), [[1 if q in (0, 3, 6) else 0 for q in range(18)], [1 if q in (9, 12, 15) else 0
                                                                               for q in range(18)]]),
         {q: (q % 3 + (0 if q < 9 else 3), (q % 9) // 3) for q in range(18)}),                                # two patches
    ]
    codes += [(c, GRID) for c in _fold_h_codes()]
    for c, coords in codes:
        for checks in ('X', 'Z'):
            for adjoint in (False, True):
                ax = c.draw_encoder(coords, adjoint=adjoint, checks=checks, wire_ends='tick', title='E').axes[0]
                assert _apart(ax), (c.n, checks, adjoint)
                assert ax.title.get_text() == 'E'
    plt.close('all')


def test_draw_encoder_wires_each_check_spider_to_its_face():
    """After the H, the stored X generators of the fold-H code are not all faces; the drawing uses the faces."""
    import matplotlib.pyplot as plt
    from zxqec.gpf import equal_for_all_assignments
    _after_encoder, after_h = _fold_h_codes()
    d = after_h.data()
    faces = d.layout_checks(GRID)
    stored = sorted(tuple(sorted(d.positions[j] for j in p.support)) for p in d.stabilizers if any(p.x))
    assert stored != sorted(tuple(sorted(f)) for f in faces['X'])    # the case this test is about
    spacing = 1.6
    place = {w: (round(spacing * x, 6), round(-spacing * y, 6)) for w, (x, y) in GRID.items()}
    for checks in ('X', 'Z'):
        ax = after_h.draw_encoder(GRID, checks=checks).axes[0]
        qubits = set(place.values())
        joined: dict = {}
        for line in ax.lines:
            ends = [(round(float(x), 6), round(float(y), 6)) for x, y in zip(*line.get_data())]
            for a, b in (ends, ends[::-1]):
                if b in qubits and a not in qubits:
                    joined.setdefault(a, set()).add(b)
        drawn = sorted(tuple(sorted(w for w, p in place.items() if p in qs)) for qs in joined.values())
        # each check spider on one face (and the logical spider on its operator)
        assert all(f in drawn for f in (tuple(sorted(f)) for f in faces[checks]))
    plt.close('all')


def test_measurement_is_the_projector_onto_the_signed_checks():
    """measurement(kind): the projector onto the states where every check of that kind has its sign, in every run."""
    import numpy as np
    from zxqec.gpf import evaluate_phase
    _after_encoder, after_h = _fold_h_codes()
    d = after_h.data()                                   # Z checks signed c[1]..c[4], X checks +
    eye, x, z = np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]]), np.diag([1.0, -1.0])
    for kind in ('X', 'Z'):
        g = d.measurement(kind)
        checks = [p for p in d.stabilizers if any(p.x if kind == 'X' else p.z)]
        for bits in ((0, 0, 0, 0), (1, 0, 1, 1), (0, 1, 1, 0)):
            assignment = {'c[{}]'.format(i + 1): b for i, b in enumerate(bits)}
            h = g.copy()
            for v in h.vertices():
                h.set_phase(v, evaluate_phase(h.phase(v), assignment))
            m = h.to_matrix()
            expected = np.eye(2 ** d.n)
            for p in checks:
                op = np.array([[1.0]])
                for j in range(d.n):
                    op = np.kron(op, (x if kind == 'X' else z) if j in p.support else eye)
                expected = expected @ (np.eye(2 ** d.n) + (-1) ** evaluate_phase(p.sign, assignment) * op) / 2
            k = np.flatnonzero(np.abs(expected) > 1e-9)[0]
            assert abs(m.flat[k]) > 1e-9
            assert np.allclose(m * (expected.flat[k] / m.flat[k]), expected)
    with pytest.raises(ValueError):
        d.measurement('Y')


def test_draw_measurement_has_no_logical_wire_and_stubs_on_both_sides():
    import matplotlib.pyplot as plt
    import numpy as np
    _after_encoder, after_h = _fold_h_codes()
    ax = after_h.draw_measurement(GRID, 'Z', label_size=7, wire_ends='tick').axes[0]
    spiders = [p for p in ax.patches if type(p).__name__ == 'Circle']
    assert len(spiders) == 9 + 4                                 # the qubits and the four Z checks, no logical spider
    stubs = [line for line in ax.lines if abs(np.hypot(*np.diff(np.array(line.get_data()), axis=1).ravel()) - 0.45) < 1e-6]
    assert len(stubs) == 18                                      # an input above and an output below each qubit
    assert sorted(t.get_text() for t in ax.texts) == ['c[1]', 'c[2]', 'c[3]', 'c[4]']
    assert _apart(ax)
    plt.close('all')


# -- zxqec.spacetime.code_layout: a reduced diagram in the faces' variables, drawn on the code's layout in 3D --------

CULTIVATION = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'cultivation')
TRIANGLE3 = [(0, 3, 6, 9), (3, 6, 8, 11), (6, 9, 11, 14)]          # the d = 3 triangle of the file: corners 0, 8, 14


def _in_faces(text, faces=None):
    """The program of a cultivation file, its code, the faces, and the reduced diagram in the faces' variables."""
    from zxqec.gpf import change_variables, phases_modulo, reduce_gpf, simplify_gadgets
    from zxqec.qprogram import program_from_stim
    p, circ = program_from_stim(text)
    code = p.code_at(p.num_stages - 1).data()
    faces = code.layout_checks(circ.coords)['X'] if faces is None else faces
    res = reduce_gpf(p.quantum_graph())
    rows = [list(code.logical_x[0].x)] + [[int(q in f) for q in code.qubits] for f in faces]
    m = change_variables(res.graph, rows, fuse=False)
    while simplify_gadgets(m):
        pass
    m = phases_modulo(change_variables(m, rows), res.constraints)
    return p, circ, code, faces, rows, m


@pytest.fixture(scope='module')
def d3():
    with open(os.path.join(CULTIVATION, 'd3_cultivation.stim')) as f:
        return _in_faces(f.read())


@pytest.fixture(scope='module')
def d5():
    with open(os.path.join(CULTIVATION, 'd5_cultivation.stim')) as f:
        return _in_faces(f.read())


def _mpp_faces():
    from zxqec.qprogram import parse_stim
    with open(os.path.join(CULTIVATION, 'd5_cultivation.stim')) as f, \
            open(os.path.join(CULTIVATION, 'd5_final_check.stim')) as g:
        circ = parse_stim(f.read() + g.read())
    first = next(ins for ins in circ.instructions if ins.name == 'MPP')
    return sorted(tuple(sorted(int(t[1:]) for t in term.split('*'))) for term in first.targets if term.startswith('X'))


def test_layout_checks_finds_the_faces_of_the_triangles(d3, d5):
    _p, _circ, code, faces, _rows, _m = d3
    assert sorted(faces) == TRIANGLE3
    count = {q: sum(q in f for f in faces) for q in code.qubits}
    assert sorted(q for q, c in count.items() if c == 1) == [0, 8, 14]         # the corners
    assert [q for q, c in count.items() if c == 3] == [6]                     # the middle
    assert sorted(tuple(sorted(f)) for f in d5[3]) == _mpp_faces()           # the faces the authors' final check measures


def _check_layout(lay, g, coords):
    import numpy as np
    from pyzx.spacetime import get_timestep
    from zxqec.spacetime import HEIGHTS
    # every vertex has a place, and the graph's qubit and timestep say the same
    assert set(lay.positions) == set(lay.graph.vertices()) == set(lay.roles)
    for v, (x, y, z) in lay.positions.items():
        assert lay.topology.coordinates(lay.graph.qubit(v)) == pytest.approx((x, y))
        if lay.roles[v] != 'output':
            assert get_timestep(lay.graph, v) == z == HEIGHTS[lay.roles[v]]
    # the qubit spiders at the file's coordinates, the outputs above them
    for v, j in lay.qubit_spiders.items():
        q = lay.qubits[j]
        assert lay.positions[v] == pytest.approx((coords[q][0], coords[q][1], 2))
    # each face spider in the middle of its face, wired to exactly its qubits
    for i, f in enumerate(lay.faces):
        (s,) = lay.face_spiders[i]
        assert lay.positions[s][:2] == pytest.approx(tuple(np.mean([coords[q][:2] for q in f], axis=0)))
        wired = {lay.qubits[lay.qubit_spiders[u]] for u in lay.graph.neighbors(s) if u in lay.qubit_spiders}
        assert wired == set(f)
        ring = [lay.graph.qubit(next(u for u, j in lay.qubit_spiders.items() if lay.qubits[j] == q)) for q in f]
        assert sum(lay.topology.are_adjacent(a, b) for a in ring for b in ring if a < b) == len(f)   # its sides
    # faces that share a qubit have different colours
    for i, f in enumerate(lay.faces):
        for j, e in enumerate(lay.faces[:i]):
            if set(f) & set(e):
                assert lay.colours[i] != lay.colours[j]
    # the input diagram is not changed
    assert g.num_vertices() == len(lay.positions) - len(lay.split)


def _outside_the_layout(lay, v):
    """Whether ``v`` lies outside the convex hull of the qubits (a point beyond some line through two qubits)."""
    import itertools
    import numpy as np
    pts = [np.array(lay.positions[u][:2]) for u in lay.qubit_spiders]
    p = np.array(lay.positions[v][:2])
    for a, b in itertools.combinations(pts, 2):
        n = np.array([-(b - a)[1], (b - a)[0]])
        side = [float(n @ (q - a)) for q in pts]
        if all(s <= 1e-9 for s in side) and n @ (p - a) > 1e-9:
            return True
        if all(s >= -1e-9 for s in side) and n @ (p - a) < -1e-9:
            return True
    return False


def test_code_layout_d3_is_the_steane_triangle(d3):
    from fractions import Fraction
    import pyzx as zx
    from zxqec.spacetime import code_layout
    _p, circ, code, faces, rows, m = d3
    lay = code_layout(m, code.qubits, circ.coords, faces)
    _check_layout(lay, m, circ.coords)
    roles = sorted(lay.roles.values())
    assert roles.count('face') == 3 and roles.count('logical') == 1 and roles.count('state') == 1
    assert roles.count('qubit') == roles.count('output') == 7
    (logical,) = [v for v, r in lay.roles.items() if r == 'logical']
    assert lay.names[logical] == 'L' and sorted(lay.names.values()) == ['L', 'S1', 'S2', 'S3']
    assert _outside_the_layout(lay, logical)
    # T-dag|+> split off L, under it; the outcome parity stays on L
    ((l, s),) = lay.split
    assert l == logical and lay.graph.phase(s) == Fraction(7, 4)
    assert lay.positions[s] == (lay.positions[l][0], lay.positions[l][1], 0.0)
    assert {str(v) for v in lay.graph.phase(l).free_vars()} == {'c[4]', 'c[5]', 'c[6]', 'c[7]', 'c[14]'}
    # the same map as the input, scalar included, in two runs
    from zxqec.gpf import boolean_variables
    names = set(boolean_variables(m)) | {v.name for v in getattr(m.scalar.phase, 'free_vars', set)()}
    assert {'c[4]', 'c[14]'} <= names
    for value in (0, 1):
        a = {n: value for n in names}
        assert zx.compare_tensors(lay.graph.substitute_variables(a), m.substitute_variables(a), preserve_scalar=True)


def test_code_layout_d5_is_the_colour_code_triangle(d5):
    from fractions import Fraction
    from zxqec.spacetime import code_layout
    _p, circ, code, faces, rows, m = d5
    lay = code_layout(m, code.qubits, circ.coords, faces)
    _check_layout(lay, m, circ.coords)
    roles = sorted(lay.roles.values())
    assert roles.count('face') == 9 and roles.count('logical') == 1 and roles.count('qubit') == 19
    assert len(set(lay.colours)) == 3                               # a colour code: three colours
    (logical,) = [v for v, r in lay.roles.items() if r == 'logical']
    wired = {lay.qubits[lay.qubit_spiders[u]] for u in lay.graph.neighbors(logical) if u in lay.qubit_spiders}
    assert wired == {0, 3, 9, 15, 23}                               # one side of the triangle
    assert _outside_the_layout(lay, logical)
    # the state split off: one vertex and one edge more, the phase moved from L to it
    ((l, s),) = lay.split
    assert m.phase(l) == Fraction(7, 4) and lay.graph.phase(l) == 0 and lay.graph.phase(s) == Fraction(7, 4)
    assert lay.graph.num_vertices() == m.num_vertices() + 1 and lay.graph.num_edges() == m.num_edges() + 1
    assert set(lay.graph.neighbors(s)) == {l}
    assert lay.positions[s][2] == 0 and lay.positions[l][2] == 1


def test_code_layout_with_faults_places_the_gadgets_under_their_qubits(d3):
    import numpy as np
    from zxqec.gpf import change_variables, phases_modulo
    from zxqec.qprogram import error_formulas
    from zxqec.spacetime import code_layout
    p, circ, code, faces, rows, _m = d3
    r = error_formulas(p).reduced
    m = phases_modulo(change_variables(r.graph, rows), r.constraints)
    lay = code_layout(m, code.qubits, circ.coords, faces)
    _check_layout(lay, m, circ.coords)
    assert lay.split == []                                          # the phase of L is a Pauli here
    roles = sorted(lay.roles.values())
    assert roles.count('hub') == roles.count('leaf') == 15 and roles.count('hidden') == 2
    assert sorted(n for v, n in lay.names.items() if lay.roles[v] == 'hidden') == ['h1', 'h2']
    for v in (v for v, r in lay.roles.items() if r == 'hidden'):
        assert _outside_the_layout(lay, v) and lay.positions[v][2] == 1
    spacing = 1.0
    under = 0
    for v in (v for v, r in lay.roles.items() if r == 'hub'):
        (leaf,) = [u for u in lay.graph.neighbors(v) if lay.roles[u] == 'leaf']
        assert lay.positions[leaf] == (lay.positions[v][0], lay.positions[v][1], 0.0)
        near = min(np.hypot(lay.positions[v][0] - lay.positions[x][0], lay.positions[v][1] - lay.positions[x][1])
                   for x in lay.qubit_spiders)
        under += near < 0.31 * spacing + 1e-9
    assert under == 14                                              # two per qubit; the injection's next to L
    # no two spiders in one place
    places = [tuple(round(c, 6) for c in pos) for v, pos in lay.positions.items()]
    assert len(set(places)) == len(places)
    assert len(lay.phase_list()) > 15


def test_code_layout_wants_the_faces_variables(d3):
    from zxqec.gpf import change_variables, reduce_gpf
    from zxqec.spacetime import code_layout
    p, circ, code, faces, _rows, m = d3
    rows = [list(code.logical_x[0].x)] + [list(r) for r in code.sx]       # code.sx: not all of them are faces
    other = change_variables(reduce_gpf(p.quantum_graph()).graph, rows)
    with pytest.raises(ValueError, match='change_variables'):
        code_layout(other, code.qubits, circ.coords, faces)
    with pytest.raises(ValueError):
        code_layout(m, code.qubits[:-1], circ.coords, faces)
    with pytest.raises(ValueError):
        code_layout(m, code.qubits, {q: circ.coords[q] for q in code.qubits[1:]}, faces)
    with pytest.raises(ValueError):
        code_layout(m, code.qubits, circ.coords, faces + [(0, 99)])


def test_draw_code_3d(d3, d5):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from zxqec.spacetime import code_layout, draw_code_3d
    for _p, circ, code, faces, _rows, m in (d3, d5):
        for view in (None, 'top', (20, -50)):
            fig = draw_code_3d(m, code.qubits, circ.coords, faces, *([view] if view else []), title='t')
            ax = fig.axes[0]
            assert sum(isinstance(c, Poly3DCollection) for c in ax.collections) == len(faces)
            texts = [t.get_text() for t in ax.texts]
            assert 'L' in texts and all('S{}'.format(i + 1) in texts for i in range(len(faces)))
            assert any('π' in t for t in texts)                       # the 7pi/4 of the state
            plt.close(fig)
        lay = code_layout(m, code.qubits, circ.coords, faces, y_down=True)
        assert all(lay.positions[v][1] == pytest.approx(-circ.coords[lay.qubits[j]][1])
                   for v, j in lay.qubit_spiders.items())
        for phases in ('all', 'constant', 'none'):
            plt.close(lay.draw(phases=phases, names=False, qubit_labels=False))
        with pytest.raises(ValueError):
            lay.draw(phases='some')
        plt.close(lay.draw_plane(title='the plane of the code'))


# -- after review: the input left alone, the regular triangle, in_face_variables, the recipe, labels and the list ----

EXAMPLES = os.path.join(os.path.dirname(__file__), '..', 'examples')


def _vertex_data(g):
    return {v: dict(g.vdata_dict(v)) for v in g.vertices()}


def test_code_layout_leaves_its_input_and_the_graphs_before_it_alone():
    """code_layout writes a timestep on every vertex of its copy; none of it may reach the input or what it came from.

    A fresh diagram, not the module fixtures: on those an earlier test may already have written the same timesteps.
    """
    import matplotlib.pyplot as plt
    from zxqec.gpf import change_variables, phases_modulo, reduce_gpf
    from zxqec.qprogram import program_from_stim
    from zxqec.spacetime import code_layout
    with open(os.path.join(CULTIVATION, 'd3_cultivation.stim')) as f:
        p, circ = program_from_stim(f.read())
    code = p.code_at(p.num_stages - 1).data()
    faces = code.layout_checks(circ.coords)['X']
    res = reduce_gpf(p.quantum_graph())
    rows = [list(code.logical_x[0].x)] + [[int(q in f) for q in code.qubits] for f in faces]
    m = phases_modulo(change_variables(res.graph, rows), res.constraints)     # PyZX clones share the vertex data
    before = (_vertex_data(res.graph), _vertex_data(m), {v: m.phase(v) for v in m.vertices()}, m.num_vertices())
    for regular in (False, True):
        lay = code_layout(m, code.qubits, circ.coords, faces, regular=regular)
        plt.close(lay.draw())
        assert all(lay.graph.vdata(v, 'timestep') is not None for v in lay.graph.vertices() if lay.roles[v] != 'output')
    assert (_vertex_data(res.graph), _vertex_data(m), {v: m.phase(v) for v in m.vertices()}, m.num_vertices()) == before


def _sides(at, faces):
    """The sides of the faces (pairs of qubits next to each other around a face), each with its number of faces."""
    import math
    import numpy as np
    sides: dict = {}
    for f in faces:
        c = np.mean([at[q] for q in f], axis=0)
        ring = sorted(f, key=lambda q: math.atan2(at[q][1] - c[1], at[q][0] - c[0]))
        for a, b in zip(ring, ring[1:] + ring[:1]):
            sides[frozenset((a, b))] = sides.get(frozenset((a, b)), 0) + 1
    return sides


def test_regular_triangle_coords_is_the_textbook_triangle(d3, d5):
    import numpy as np
    from zxqec.spacetime import regular_triangle_coords
    for (_p, circ, code, faces, _rows, _m), (top, left, right), n in ((d3, (8, 0, 14), 2), (d5, (23, 0, 40), 4)):
        at = regular_triangle_coords(circ.coords, faces)
        assert set(at) == set(code.qubits)
        a, b, c = (np.array(at[q]) for q in (top, left, right))
        # an equilateral triangle, flat at the bottom, a side n steps long, turning the same way as in the file
        assert np.linalg.norm(a - b) == pytest.approx(n) and np.linalg.norm(b - c) == pytest.approx(n)
        assert np.linalg.norm(c - a) == pytest.approx(n) and b[1] == pytest.approx(c[1]) and a[1] > b[1]
        file = [np.array(circ.coords[q], float) for q in (top, left, right)]
        turn = lambda p, q, r: np.sign((q - p)[0] * (r - p)[1] - (q - p)[1] * (r - p)[0])   # noqa: E731
        assert turn(a, b, c) == turn(*file)
        # the sides of the border 1 long; every inner qubit at the mean of its neighbours
        sides = _sides(at, faces)
        border = {q for s, k in sides.items() if k == 1 for q in s}
        assert all(np.linalg.norm(np.subtract(*(at[q] for q in s))) == pytest.approx(1) for s, k in sides.items()
                   if k == 1)
        assert len(border) == 3 * n
        for q in set(at) - border:
            near = [r for s in sides if q in s for r in s if r != q]
            assert np.allclose(at[q], np.mean([at[r] for r in near], axis=0))
    # d = 3: the Steane triangle; qubit 6 in the middle
    at = regular_triangle_coords(d3[1].coords, d3[3])
    assert np.allclose(at[6], np.mean([at[8], at[0], at[14]], axis=0))
    # y_down: the file read upside down, so the corner at the top is 0 (smallest y in the file) and it turns the other way
    down = regular_triangle_coords(d3[1].coords, d3[3], y_down=True)
    assert down[0][1] > down[8][1] == pytest.approx(down[14][1])
    flipped = {q: (x, -y) for q, (x, y) in d3[1].coords.items()}
    assert down == regular_triangle_coords(flipped, d3[3])
    # not a triangle: one square face (four corners), two triangles apart (two borders)
    with pytest.raises(ValueError):
        regular_triangle_coords({0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}, [(0, 1, 2, 3)])
    with pytest.raises(ValueError):
        regular_triangle_coords({**{q: c for q, c in d3[1].coords.items() if q in d3[2].qubits},
                                 **{100 + q: (c[0] + 10, c[1]) for q, c in d3[1].coords.items() if q in d3[2].qubits}},
                                d3[3] + [tuple(100 + q for q in f) for f in d3[3]])


def test_code_layout_on_the_regular_triangle(d3, d5):
    import matplotlib.pyplot as plt
    from zxqec.spacetime import code_layout, regular_triangle_coords
    for _p, circ, code, faces, _rows, m in (d3, d5):
        lay = code_layout(m, code.qubits, circ.coords, faces, regular=True)
        at = regular_triangle_coords(circ.coords, faces)
        _check_layout(lay, m, at)                                    # every check, with the regular places
        (logical,) = [v for v, r in lay.roles.items() if r == 'logical']
        assert _outside_the_layout(lay, logical)
        plt.close(lay.draw())
    # a qubit in no face has no regular place
    _p, circ, code, faces, _rows, m = d3
    with pytest.raises(ValueError):
        code_layout(m, code.qubits, circ.coords, faces[1:], regular=True)


def test_in_face_variables_is_the_recipe_in_one_call(d3, d5):
    from fractions import Fraction
    from zxqec.gpf import equal_for_all_assignments, reduce_gpf
    from zxqec.spacetime import code_layout, in_face_variables
    p, circ, code, faces, _rows, m = d3
    res = reduce_gpf(p.quantum_graph())
    before = _vertex_data(res.graph)
    g = in_face_variables(res, code, faces)
    assert equal_for_all_assignments(g, m, constraints=res.constraints)
    assert _vertex_data(res.graph) == before
    # d = 5: the 22 gadgets of the T gates become one phase, 7pi/4 on L; without the simplification they stay
    p5, circ5, code5, faces5, _rows5, _m5 = d5
    res5 = reduce_gpf(p5.quantum_graph())
    for simplify, hubs in ((True, 0), (False, 22)):
        lay = code_layout(in_face_variables(res5, code5, faces5, simplify=simplify), code5.qubits, circ5.coords, faces5,
                          regular=True)
        roles = list(lay.roles.values())
        assert roles.count('hub') == hubs and roles.count('face') == 9 and roles.count('logical') == 1
        if simplify:
            assert [lay.graph.phase(s) for _l, s in lay.split] == [Fraction(7, 4)]
    with pytest.raises(ValueError):
        in_face_variables(res, code, faces + [(0, 99)])


def test_in_face_variables_keeps_the_faults(d3):
    """With simplify=False, modulo=False only the variables change: every fault and outcome variable stays."""
    import matplotlib.pyplot as plt
    from zxqec.gpf import boolean_variables, change_variables
    from zxqec.qprogram import error_formulas
    from zxqec.spacetime import code_layout, in_face_variables
    p, circ, code, faces, rows, _m = d3
    r = error_formulas(p).reduced
    g = in_face_variables(r, code, faces, simplify=False, modulo=False)
    assert boolean_variables(g) == boolean_variables(r.graph)                  # 37 variables, none set to a value
    ref = change_variables(r.graph, rows)                                        # the same diagram, vertex by vertex
    assert set(g.vertices()) == set(ref.vertices()) and set(g.edges()) == set(ref.edges())
    assert all(str(g.phase(v)) == str(ref.phase(v)) and g.type(v) == ref.type(v) for v in g.vertices())
    lay = code_layout(g, code.qubits, circ.coords, faces, regular=True)
    roles = list(lay.roles.values())
    assert roles.count('hub') == roles.count('leaf') == 15 and roles.count('hidden') == 2
    # the two gadgets of a qubit side by side under it, the one without hidden spiders on the left
    for x in lay.qubit_spiders:
        column = {u for u in lay.graph.neighbors(x) if lay.roles[u] in ('face', 'logical')}
        hubs = [v for v, rl in lay.roles.items() if rl == 'hub'
                and {u for u in lay.graph.neighbors(v) if lay.roles[u] in ('face', 'logical')} == column]
        assert len(hubs) == 2
        plain, held = sorted(hubs, key=lambda v: sum(lay.roles[u] == 'hidden' for u in lay.graph.neighbors(v)))
        assert lay.positions[plain][0] < lay.positions[x][0] < lay.positions[held][0]
        assert lay.positions[plain][1] == pytest.approx(lay.positions[x][1])
    plt.close(lay.draw())


def _literal_block(lines):
    """The lines of a literal block: indented by four spaces, or empty, up to the first line that is not."""
    for line in lines:
        if line and not line.startswith('    '):
            return
        yield line


def test_the_recipe_in_the_module_docstring_runs_for_d3_and_d5(monkeypatch):
    import importlib
    from fractions import Fraction
    import matplotlib.pyplot as plt
    doc = importlib.import_module('zxqec.spacetime.code_layout').__doc__
    block = doc.split('::\n\n', 1)[1].split('\n\n', 1)
    lines = (block[0] + '\n\n' + block[1]).splitlines()
    code = '\n'.join(line[4:] for line in _literal_block(lines))
    assert 'd3_cultivation.stim' in code
    monkeypatch.chdir(EXAMPLES)
    for d, faces in (('d3', 3), ('d5', 9)):
        ns: dict = {}
        exec(code.replace('d3_cultivation', d + '_cultivation'), ns)
        lay = ns['lay']
        roles = list(lay.roles.values())
        assert roles.count('face') == faces and roles.count('hub') == 0 and roles.count('logical') == 1
        assert [lay.graph.phase(s) for _l, s in lay.split] == [Fraction(7, 4)]
        plt.close(ns['fig'])


def _label_boxes(fig):
    """The box on the screen of every text in ``fig``, 3D texts where they are drawn (after ``fig.canvas.draw()``)."""
    from matplotlib import cbook
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Text3D
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text() or not t.get_visible():
                continue
            if isinstance(t, Text3D):
                x, y, _ = proj3d.proj_transform(t._x, t._y, t._z, ax.M)
                with cbook._setattr_cm(t, _x=x, _y=y):
                    boxes.append((t.get_text(), t.get_window_extent(renderer)))
            else:
                boxes.append((t.get_text(), t.get_window_extent(renderer)))
    return boxes


def _overlaps(boxes):
    import itertools
    return [(a, b) for (a, p), (b, q) in itertools.combinations(boxes, 2)
            if min(p.x1, q.x1) - max(p.x0, q.x0) > 1 and min(p.y1, q.y1) - max(p.y0, q.y0) > 1]


def test_draw_keeps_the_labels_apart_and_the_phase_list_in_the_figure(d3, d5):
    import matplotlib.pyplot as plt
    from zxqec.qprogram import error_formulas
    from zxqec.spacetime import code_layout, in_face_variables
    # without faults: no two labels on each other, in both views, at d = 3 and d = 5
    for _p, circ, code, faces, _rows, m in (d3, d5):
        lay = code_layout(m, code.qubits, circ.coords, faces, regular=True)
        for view in (None, 'top'):
            fig = lay.draw(view)
            fig.canvas.draw()
            boxes = _label_boxes(fig)
            names = [t for t, _b in boxes]
            assert 'L' in names and all('S{}'.format(i + 1) in names for i in range(len(faces)))
            assert _overlaps(boxes) == []
            plt.close(fig)
    # with faults: 24 long phases in the list, every line inside the figure, also in an axes given
    p, circ, code, faces, _rows, _m = d3
    lay = code_layout(in_face_variables(error_formulas(p).reduced, code, faces, simplify=False, modulo=False),
                      code.qubits, circ.coords, faces, regular=True)
    assert len(lay.phase_list()) >= 20
    given = plt.figure(figsize=(8, 6.5))
    for fig in (lay.draw(), lay.draw(ax=given.add_subplot(111, projection='3d'))):
        fig.canvas.draw()
        inside = fig.bbox
        listed = [b for t, b in _label_boxes(fig) if '\n' in t]
        assert len(listed) == 1
        assert all(inside.x0 - 1 <= b.x0 and b.x1 <= inside.x1 + 1 and inside.y0 - 1 <= b.y0 and b.y1 <= inside.y1 + 1
                   for _t, b in _label_boxes(fig))
        plt.close(fig)
