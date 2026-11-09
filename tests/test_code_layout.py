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
