"""Smoke tests for zxqec.spacetime.drawing (rendering with the Agg backend)."""
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement, Reset, T

from zxqec.qprogram import QProgram
from zxqec.spacetime import (VIEWS, Topology, draw_graph, draw_graph_3d, draw_program, draw_program_3d,
                             draw_program_slices, draw_program_views, draw_slice, draw_slices, slice_vertices)


def _program(topology=None, mapping=None):
    rnd = Circuit(3)
    for g in [Reset(2), HAD(2), CNOT(2, 0), CNOT(2, 1), HAD(2), Measurement(2)]:
        rnd.add_gate(g)
    t = Circuit(3); t.add_gate(T(0))
    p = QProgram(3, topology=topology, mapping=mapping, initial_states={2: '0'},
                 gate_durations={"CNOT": 2})
    p.add_circuit(rnd, name="round 1"); p.add_circuit(t, name="T"); p.add_circuit(rnd, name="round 2")
    p.level_up(0, 0, 1, name="round + T")
    return p


def test_draw_program_time_axis_and_blocks():
    p = _program()
    fig = draw_program(p, level=1, figsize=(10, 3))
    assert fig is not None and len(fig.axes) == 1
    # one shaded span per (block, stage tick range): block 0 has two stages, block 1 one
    spans = [a for a in fig.axes[0].patches if type(a).__name__ == 'Rectangle' and a.get_alpha() == 0.15]
    assert len(spans) == 3
    assert draw_program(p, time_axis=False, show_time=False) is not None
    assert draw_program(p, level=0, time_axis=False) is not None


def test_draw_graph_puts_every_spider_at_its_timestep():
    # the static counterpart of draw_web, for a diagram cut out of a program
    import matplotlib.pyplot as plt
    p = _program()
    g = p.stage_graph(1, classical=False)
    fig = draw_graph(g, figsize=(4, 2))
    assert len(fig.axes) == 1
    from zxqec.spacetime.drawing import time_positions
    xs = sorted({round(x, 6) for x in time_positions(g).values()})
    drawn = sorted({round(float(c.center[0]), 6) for c in fig.axes[0].patches if type(c).__name__ == 'Circle'})
    assert drawn and drawn == xs                        # PyZX's circles, at the spiders' time positions
    assert draw_graph(g, time_axis=False, show_time=False) is not None
    plt.close('all')


def test_draw_graph_names_a_spider_just_left_of_it():
    import matplotlib.pyplot as plt
    p = _program()
    g = p.stage_graph(1, classical=False)
    v = next(v for v in g.vertices() if g.type(v) == 1)
    fig = draw_graph(g, time_axis=False, show_time=False, names={v: 'ell'}, label_size=9)
    (text,) = [t for t in fig.axes[0].texts if t.get_text() == 'ell']
    x, y = text.get_position()
    assert (x, y) == (g.row(v) - 0.3, -g.qubit(v)) and text.get_fontsize() == 9
    plt.close('all')


def test_draw_graph_clear_labels_cover_no_edge_spider_or_label():
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.transforms import Bbox
    from zxqec.gpf import to_gpf
    c = Circuit(3)
    for q in (0, 1):
        c.add_gate('CNOT', q, 2)
    c.add_gate('T', 2)
    c.add_gate('T', 0)
    for q in (0, 1):
        c.add_gate('CNOT', q, 2)
    g = to_gpf(c.to_graph())
    spiders = [next(iter(g.neighbors(b))) for b in g.inputs()]
    names = {v: 'in{}'.format(i) for i, v in enumerate(spiders)}
    fig = draw_graph(g, time_axis=False, show_time=False, names=names, label_size=9, label_side='clear')
    ax = fig.axes[0]
    fig.draw_without_rendering()
    inv = ax.transData.inverted()
    boxes = {t.get_text(): Bbox(inv.transform(t.get_window_extent().get_points())) for t in ax.texts}
    assert set(names.values()) <= set(boxes) and any('π/4' in s or 'pi' in s for s in boxes)
    shapes = [Path(line.get_xydata()) for line in ax.lines]
    shapes += [p.get_patch_transform().transform_path(p.get_path()) for p in ax.patches]
    for text, box in boxes.items():
        assert not any(s.intersects_bbox(box, filled=True) for s in shapes), text
        assert not any(box.overlaps(o) for t, o in boxes.items() if t != text), text
    for v, name in names.items():                  # each name stays near its spider
        cx, cy = boxes[name].x0 + boxes[name].width / 2, boxes[name].y0 + boxes[name].height / 2
        assert abs(cx - g.row(v)) < 2.5 and abs(cy + g.qubit(v)) < 1.5
    plt.close('all')


def test_draw_graph_names_the_right_spider_on_the_time_axis_after_a_vertex_is_gone():
    # with time_axis=True the drawing is of a copy, numbered afresh; a gap in g's numbering must not move names
    import matplotlib.pyplot as plt
    from zxqec.spacetime.drawing import time_positions
    p = _program()
    g = p.stage_graph(0, classical=False)
    g.remove_vertex(min(g.vertices()))
    names = {v: 'n{}'.format(v) for v in g.vertices() if g.type(v) == 1}
    fig = draw_graph(g, names=names)
    at = {t.get_text(): t.get_position() for t in fig.axes[0].texts}
    t = time_positions(g)
    assert names and all(at[name] == (t[v] - 0.3, -g.qubit(v)) for v, name in names.items())
    plt.close('all')


def test_a_slice_is_one_time_step_flat():
    import matplotlib.pyplot as plt
    from pyzx.spacetime import get_timestep
    line = Topology.line(3)
    p = _program(topology=line)
    g = p.quantum_graph()
    ticks = sorted({get_timestep(g, v) for v in slice_vertices(g, t=3)})
    assert ticks == [3]
    with pytest.raises(ValueError):
        slice_vertices(g)
    fig = draw_slice(g, line, t=3)
    xs = sorted(round(float(c.center[0]), 6) for c in fig.axes[0].patches if type(c).__name__ == 'Circle')
    # the spiders of tick 3, each at the x of its wire on the line
    assert xs == sorted(round(float(line.coordinates(g.qubit(v))[0]), 6) for v in slice_vertices(g, t=3))
    a, b = draw_slice(g, line, t=0).axes[0], draw_slice(g, line, t=9).axes[0]
    assert a.get_xlim() == b.get_xlim()                 # the same window for every slice of one diagram
    # by height: a range of heights of draw_graph_3d's positions, here the ticks 3 to 4
    assert set(slice_vertices(g, z=(3, 4.9), topology=line)) == set(slice_vertices(g, t=3)) | set(slice_vertices(g, t=4))
    p.graph()
    lo, hi = p.stage(1).ticks
    fig = draw_program_slices(p, stage=1, cols=2)
    assert len(fig.axes) == hi - lo + 1
    assert len(draw_slices(g, line, [0, 1, 2]).axes) == 3
    plt.close('all')


def test_draw_program_3d_with_topology():
    topo = Topology.grid(2, 2)
    p = _program(topology=topo, mapping={0: 0, 1: 3, 2: 1})
    fig = draw_program_3d(p, level=1, labels=True)
    assert fig is not None and fig.axes[0].name == '3d'
    assert draw_program_3d(p, show_topology=False, show_phases=False) is not None


def test_draw_graph_3d_without_topology_and_untimed():
    p = _program()
    assert draw_graph_3d(p.graph()) is not None
    plain = Circuit(2); plain.add_gate(CNOT(0, 1))
    assert draw_graph_3d(plain.to_graph()) is not None          # rows used as time
    line = Topology.line(3)
    assert draw_graph_3d(_program(topology=line).graph(), line, blocks=[("b", [(0, 3)])]) is not None


def test_phases_beside_their_spiders_are_not_under_them_from_above():
    # from the top, a phase written under its spider in time lands on the spider; beside it, it does not
    import matplotlib
    matplotlib.use('Agg')
    topo = Topology.grid(2, 2)
    g = _program(topology=topo, mapping={0: 0, 1: 3, 2: 1}).graph()

    def phases(beside):
        ax = draw_graph_3d(g, topo, view='top', phases_beside=beside, finish=False).axes[0]
        texts = [t for t in ax.texts if t.get_color() == 'blue']
        assert texts
        return texts

    under, beside = phases(False), phases(True)
    assert all(t.get_horizontalalignment() == 'center' for t in under)
    assert all(t.get_horizontalalignment() == 'left' and t.get_transform() is not t.axes.transData for t in beside)
    import matplotlib.pyplot as plt
    plt.close('all')


def test_hadamard_boxes_draw_each_hadamard_edge_as_a_yellow_square():
    import matplotlib
    from matplotlib.colors import to_rgba
    from pyzx.utils import EdgeType, VertexType
    matplotlib.use('Agg')
    g = _program().graph()
    hadamards = sum(1 for e in g.edges() if g.edge_type(e) == EdgeType.HADAMARD)
    assert hadamards

    # a phase-free spider with one Hadamard and one plain edge is the gate itself: drawn as the box, not as a spider
    gates = sum(1 for v in g.vertices() if g.type(v) in (VertexType.Z, VertexType.X) and g.phase(v) == 0
                and sorted(g.edge_type(e) for e in g.incident_edges(v)) == [EdgeType.SIMPLE, EdgeType.HADAMARD])
    assert gates

    def drawn(boxes):
        ax = draw_graph_3d(g, hadamard_boxes=boxes, finish=False).axes[0]
        count = lambda color: sum(1 for c in ax.collections if len(c.get_facecolor()) and
                                  tuple(c.get_facecolor()[0]) == to_rgba(color))
        blue = sum(1 for line in ax.lines if line.get_color() == '#0099ff')
        return count('yellow'), blue, count('#ccffcc') + count('#ff8888')

    yellow, blue, spiders = drawn(False)
    assert (yellow, blue) == (0, hadamards)
    assert drawn(True) == (hadamards, 0, spiders - gates)
    import matplotlib.pyplot as plt
    plt.close('all')


def test_draw_blocks_3d_draws_every_block_from_above_bare():
    import matplotlib
    matplotlib.use('Agg')
    from zxqec.spacetime import draw_blocks_3d
    topo = Topology.grid(2, 2)
    p = _program(topology=topo, mapping={0: 0, 1: 3, 2: 1})
    level = p.group_stages([(0, 0, 'first', None, 'gpf')], one_level=True)
    n = p.levels.level_size(level)
    fig = draw_blocks_3d(p, level, finish=False)
    assert len(fig.axes) == n
    for k, ax in enumerate(fig.axes):
        assert ax.elev == 90 and not ax._axis3don                   # from above, no panes or axes (3D's own flag)
        assert ax.get_title().startswith(str(p.block(level, k)))
    assert all(ax.get_xlim() == fig.axes[0].get_xlim() for ax in fig.axes)   # all at one scale
    assert draw_graph_3d(p.graph(), topo, finish=False).axes[0]._axis3don    # not bare by default
    rows = draw_blocks_3d(p, level, columns=1, finish=False)                  # one block per row
    assert [ax.get_subplotspec().rowspan.start for ax in rows.axes] == list(range(n))
    import matplotlib.pyplot as plt
    plt.close('all')


def test_draw_blocks_3d_places_phases_where_they_cover_least():
    # every phase written beside a spider gets a side from the placement pass, and the pass is repeatable
    import matplotlib
    matplotlib.use('Agg')
    from zxqec.spacetime import draw_blocks_3d
    from zxqec.spacetime.drawing import _place_labels
    topo = Topology.grid(2, 2)
    p = _program(topology=topo, mapping={0: 0, 1: 3, 2: 1})
    level = p.group_stages([(0, 0, 'first', None, 'gpf')], one_level=True)
    fig = draw_blocks_3d(p, level, view=(70, -80), time_scale=0.25, phases_beside=True, finish=False)
    labels = [t for ax in fig.axes for t, _ in ax._zx_drawn['labels']]
    assert labels and all(hasattr(t, '_zx_side') for t in labels)
    for ax in fig.axes:                                # placing again keeps what it chose: it is settled
        before = [t._zx_side for t, _ in ax._zx_drawn['labels']]
        _place_labels(ax)
        assert [t._zx_side for t, _ in ax._zx_drawn['labels']] == before
    import matplotlib.pyplot as plt
    plt.close('all')


def test_save_panels_saves_each_block_cropped_to_its_own_drawing(tmp_path):
    # one file per panel and format, each smaller than the whole figure; the figure is left as it was
    import matplotlib
    matplotlib.use('Agg')
    from pypdf import PdfReader
    from zxqec.spacetime import draw_blocks_3d, save_panels
    topo = Topology.grid(2, 2)
    p = _program(topology=topo, mapping={0: 0, 1: 3, 2: 1})
    level = p.group_stages([(0, 0, 'first', None, 'gpf')], one_level=True)
    fig = draw_blocks_3d(p, level, finish=False)
    titles = [ax.get_title() for ax in fig.axes]
    sizes = save_panels(fig, str(tmp_path / 'panel-{k}.{ext}'), formats=('pdf', 'png'))
    assert len(sizes) == len(fig.axes) == p.levels.level_size(level)
    whole = fig.get_size_inches()
    for k, (w, h) in enumerate(sizes):
        assert (tmp_path / 'panel-{}.png'.format(k)).exists()
        box = PdfReader(str(tmp_path / 'panel-{}.pdf'.format(k))).pages[0].mediabox
        assert float(box.width) == pytest.approx(72 * w, abs=1.5) and float(box.height) == pytest.approx(72 * h, abs=1.5)
        assert 0 < w < whole[0] and 0 < h <= whole[1]
    assert all(ax.get_visible() and ax.title.get_visible() for ax in fig.axes)
    assert [ax.get_title() for ax in fig.axes] == titles
    import matplotlib.pyplot as plt
    plt.close('all')


def test_crossing_wires_draws_a_permutation_without_its_identity_spiders():
    from pyzx.utils import VertexType
    from zxqec.spacetime import block_layout_3d, level_layout_3d
    grid = Topology.grid(3, 3)                         # a transversal H, then a quarter turn of the grid
    layer = Circuit(9)
    for q in range(9):
        layer.add_gate(HAD(q))
    p = QProgram(9, topology=grid)
    p.add_circuit(layer, name='transversal H')
    p.add_permutation([6, 3, 0, 7, 4, 1, 8, 5, 2], name='quarter turn')
    after = Circuit(9)
    after.add_gate(T(4))
    p.add_circuit(after, name='T')
    level, turn = 0, 1

    def spiders(g):
        return sum(1 for v in g.vertices() if g.type(v) != VertexType.BOUNDARY)

    g, _, _ = block_layout_3d(p, level, turn)
    bare, pos, _ = block_layout_3d(p, level, turn, crossing_wires=True)
    assert spiders(g) > 0 and spiders(bare) == 0
    assert all(len(list(bare.neighbors(b))) == 1 for b in bare.inputs())      # each wire still goes through
    whole, _, _ = level_layout_3d(p, level)
    fewer, _, _ = level_layout_3d(p, level, crossing_wires=True)
    assert spiders(whole) - spiders(fewer) == spiders(g)                      # only the turn's spiders go


def test_named_views_and_multi_view():
    topo = Topology.grid(2, 2)
    p = _program(topology=topo, mapping={0: 0, 1: 3, 2: 1})
    for name in VIEWS:
        fig = draw_program_3d(p, level=1, view=name)
        assert fig.axes[0].elev == VIEWS[name][0] and fig.axes[0].azim == VIEWS[name][1]
    fig = draw_program_3d(p, view=(10, 20), elev=30)
    assert fig.axes[0].elev == 30 and fig.axes[0].azim == 20
    with pytest.raises(ValueError):
        draw_program_3d(p, view='underneath')
    fig = draw_program_views(p, level=1)
    assert len(fig.axes) == 3 and [a.get_title() for a in fig.axes] == ['default', 'top', 'front']
    assert len(draw_program_views(p, views=['side', (45, 45)]).axes) == 2


def test_view_buttons_turn_the_figure():
    p = _program(topology=Topology.grid(2, 2), mapping={0: 0, 1: 3, 2: 1})
    fig = draw_program_3d(p, buttons=True)
    buttons = fig._zxqec_view_buttons
    assert [b.label.get_text() for b in buttons] == list(VIEWS)
    ax = fig.axes[0]
    from matplotlib.backend_bases import MouseEvent
    for b, name in zip(buttons, VIEWS):
        x, y = b.ax.transAxes.transform((0.5, 0.5))          # click the centre of the button
        for kind in ('button_press_event', 'button_release_event'):
            fig.canvas.callbacks.process(kind, MouseEvent(kind, fig.canvas, x, y, button=1))
        assert (ax.elev, ax.azim) == VIEWS[name]


def test_draw_program_even_rows():
    # a measurement sits half a row after the gate before it; even_rows draws it a whole row later
    import matplotlib
    matplotlib.use('Agg')
    from pyzx.circuit import Circuit
    from zxqec.qprogram import QProgram
    from zxqec.spacetime.drawing import draw_program, program_layout
    p = QProgram(3, bits=1, initial_states={2: '0'})
    p.add_circuit(Circuit.from_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[1];\n'
                                    'h q[2];\ncx q[2], q[0];\ncx q[2], q[1];\nh q[2];\nmeasure q[2] -> c[0];\n'))
    rows = {v: program_layout(p, False).row(v) for v in program_layout(p, False).vertices()}
    assert any(r != int(r) for r in rows.values())                       # PyZX's half rows
    fig = draw_program(p, time_axis=False, show_time=False, even_rows=True)
    xs = sorted({round(c.center[0], 6) for c in fig.axes[0].patches if hasattr(c, 'center')})
    assert xs and all(x == int(x) for x in xs)                           # every spider on a whole row
    assert {v: program_layout(p, False).row(v) for v in program_layout(p, False).vertices()} == rows


def test_block_spans_are_where_draw_program_shades_the_blocks():
    # every block's intervals, in the drawing's coordinates, on a time axis and with even rows
    from zxqec.spacetime import block_spans
    p = _program()
    for options in ({}, {'time_axis': False, 'even_rows': True}):
        fig = draw_program(p, level=1, show_time=False, **options)
        shaded = [(round(a.get_x(), 9), round(a.get_x() + a.get_width(), 9)) for a in fig.axes[0].patches
                  if type(a).__name__ == 'Rectangle' and a.get_alpha() == 0.15]
        spans = block_spans(p, 1, **options)
        assert [label for label, _ranges in spans] == ["round + T", "round 2"]
        assert [(round(x0, 9), round(x1, 9)) for _label, ranges in spans for x0, x1 in ranges] == shaded


def test_draw_program_labels_above():
    # labels_above puts every block label above the top row of spiders, in a strip added to the figure;
    # without it the labels are where they were, over the top row
    p = _program()
    names = {"round + T", "round 2"}
    def labels_and_tops(fig):
        ax = fig.axes[0]
        labels = [t.get_position()[1] for t in ax.texts if t.get_text() in names]
        tops = [c.center[1] + c.radius for c in ax.patches if hasattr(c, 'center') and hasattr(c, 'radius')]
        return labels, max(tops), ax.get_ylim()
    labels, top, (_lo, hi) = labels_and_tops(draw_program(p, level=1, time_axis=False, show_time=False))
    above, top2, (_lo2, hi2) = labels_and_tops(draw_program(p, level=1, time_axis=False, show_time=False,
                                                            labels_above=True))
    assert len(labels) == len(above) == 2 and top == top2
    assert all(y == hi - 0.15 for y in labels)                         # unchanged by default
    assert hi2 == hi + 0.5 and all(y > top for y in above)             # above every spider


def test_draw_program_labels_above_on_two_lines():
    # a narrow block with a long label: the label after it goes one line up instead of running into it,
    # and the strip grows by that line; the labels that fit stay on the lowest line, where they were
    rnd = Circuit(3)
    for g in [Reset(2), HAD(2), CNOT(2, 0), CNOT(2, 1), HAD(2), Measurement(2)]:
        rnd.add_gate(g)
    t = Circuit(3); t.add_gate(T(0))
    p = QProgram(3, initial_states={2: '0'}, gate_durations={"CNOT": 2})
    p.add_circuit(rnd, name="round 1"); p.add_circuit(t, name="a T gate on qubit 0, a long name")
    p.add_circuit(rnd, name="round 2")
    plain = draw_program(p, level=0, time_axis=False, show_time=False)
    hi = plain.axes[0].get_ylim()[1]
    fig = draw_program(p, level=0, time_axis=False, show_time=False, labels_above=True)
    ax = fig.axes[0]
    y = {t.get_text(): t.get_position()[1] for t in ax.texts}
    lowest = hi + 0.5 - 0.15
    assert y["round 1"] == lowest and y["a T gate on qubit 0, a long name"] == lowest
    assert y["round 2"] > lowest                                        # one line up
    assert ax.get_ylim()[1] > hi + 0.5                                  # the strip has grown


def test_draw_program_shows_some_blocks_with_bigger_labels():
    import matplotlib
    matplotlib.use('Agg')
    from pyzx.circuit import Circuit
    from pyzx.circuit.gates import CNOT, HAD, T
    from zxqec.qprogram import QProgram
    from zxqec.spacetime import draw_graph, draw_program
    p = QProgram(2)
    for k in range(3):
        c = Circuit(2)
        c.add_gate(HAD(0))
        c.add_gate(T(1))
        c.add_gate(CNOT(0, 1))
        p.add_circuit(c, name='part {}'.format(k))
    whole = draw_program(p, 0, time_axis=False, show_time=False, even_rows=True, figsize=(9, 2))
    one = draw_program(p, 0, time_axis=False, show_time=False, even_rows=True, figsize=(9, 2), blocks=[1],
                       label_size=13)
    lo, hi = one.axes[0].get_xlim()
    assert hi - lo < (whole.axes[0].get_xlim()[1] - whole.axes[0].get_xlim()[0]) / 2
    shown = [t for t in one.axes[0].texts if t.get_visible()]
    assert [t.get_text() for t in shown if t.get_color() != 'blue'] == ['part 1']
    big = draw_graph(p.quantum_graph(), time_axis=False, show_time=False, label_size=13)
    labels = [t for t in big.axes[0].texts if t.get_color() == 'blue']
    assert labels and all(t.get_fontsize() == 13 and t.get_verticalalignment() == 'top' for t in labels)
    plain = draw_graph(p.quantum_graph(), time_axis=False, show_time=False)
    below = [t.get_position()[1] for t in plain.axes[0].texts if t.get_color() == 'blue']
    assert [t.get_position()[1] for t in labels] == pytest.approx([y + 0.25 for y in below])   # just under the spider
    right = draw_graph(p.quantum_graph(), time_axis=False, show_time=False, label_size=13, label_side='right')
    beside = [t for t in right.axes[0].texts if t.get_color() == 'blue']
    assert [t.get_position()[1] for t in beside] == pytest.approx([y + 0.5 for y in below])    # on the spider's row
    assert all(t.get_horizontalalignment() == 'left' for t in beside)
    with pytest.raises(ValueError):
        draw_graph(p.quantum_graph(), label_size=13, label_side='above')
    with pytest.raises(ValueError):
        draw_program(p, blocks=[1])


def test_draw_checks_marks_the_results_each_check_reads():
    import matplotlib
    matplotlib.use('Agg')
    from pyzx.symbolic import new_var
    from pyzx.graph.base import VarRegistry
    from zxqec.spacetime import draw_checks
    reg = VarRegistry()
    c = {k: new_var('c[{}]'.format(k), True, reg) for k in (0, 1, 2, 10)}
    fig = draw_checks([c[2] + c[10], c[0] + c[1]])
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ['0', '1', '2', '10']    # by the number in the name
    grid = ax.images[0].get_array()
    assert grid.tolist() == [[1, 1, 0, 0], [0, 0, 1, 1]]                           # rows by their first result
    assert ax.get_xlabel() == 'result c[k]'


def test_a_stage_flat_on_the_layout():
    """draw_stage_layout: a transversal H is a stub through an H box on each qubit, a quarter turn moves the wires."""
    import matplotlib.pyplot as plt
    from zxqec.spacetime import draw_stage_layout
    grid = Topology.grid(3, 3)
    layer = Circuit(9)
    for q in range(9):
        layer.add_gate(HAD(q))
    turn = [6, 3, 0, 7, 4, 1, 8, 5, 2]
    p = QProgram(9, topology=grid)
    p.add_circuit(layer, name='transversal H')
    p.add_permutation(turn, name='quarter turn')
    p.add_circuit(Circuit(9), name='nothing')
    ax = draw_stage_layout(p, 0).axes[0]
    boxes = [r for r in ax.patches if type(r).__name__ == 'Rectangle' and r.get_facecolor()[:3] == (1.0, 1.0, 0.0)]
    assert len(boxes) == 9                                       # one Hadamard box on each qubit
    assert not [c for c in ax.patches if type(c).__name__ == 'Circle' and c.radius == 0.2]   # no spiders left
    ax = draw_stage_layout(p, 1, spacing=1.0).axes[0]
    paths = [r.get_path() for r in ax.patches if type(r).__name__ == 'PathPatch']
    # each moved wire: a stub above its old place, a line (or a curve) to its new place, a stub below it
    ends = sorted(tuple(sorted((round(float(x), 6), round(float(y), 6)) for x, y in path.vertices[[1, -2]]))
                  for path in paths)
    moved = sorted(tuple(sorted((round(float(grid.coordinates(j)[0]), 6), round(-float(grid.coordinates(j)[1]), 6))
                                for j in (k, turn[k]))) for k in range(9) if turn[k] != k)   # y down
    assert ends == moved
    assert all(path.vertices[0][1] - path.vertices[1][1] == pytest.approx(0.45) for path in paths)
    assert sum(1 for path in paths if 3 in path.codes) == 4      # the sides of the square, bowed off the middle places
    assert len(ax.lines) == 1                                    # the middle qubit, which stays: a stub
    bare = draw_stage_layout(p, 0, wire_ends='tick', title='H').axes[0]
    assert not [c for c in bare.patches if type(c).__name__ == 'Circle']       # no dots at the wire ends
    fig = bare.figure
    fig.canvas.draw()
    box = bare.title.get_window_extent()
    assert bare.title.get_text() == 'H' and box.y1 <= fig.bbox.y1 + 1      # the title is inside the figure
    with pytest.raises(ValueError):
        draw_stage_layout(p, 0, wires=[])
    plt.close('all')


def test_draw_graph_phases_constant_keeps_numbers_and_drops_phases_with_variables():
    import matplotlib.pyplot as plt
    from fractions import Fraction
    from pyzx.graph import Graph
    from pyzx.symbolic import new_var
    from pyzx.utils import VertexType
    g = Graph()
    a = g.add_vertex(VertexType.Z, qubit=0, row=0, phase=Fraction(7, 4))
    b = g.add_vertex(VertexType.X, qubit=1, row=1)
    g.set_phase(b, new_var('c', True, g.var_registry) + 1)
    g.add_edge((a, b))

    def blue(fig):
        return sorted(t.get_text() for t in fig.axes[0].texts if t.get_color() == 'blue')

    every = blue(draw_graph(g, time_axis=False, show_time=False))
    assert len(every) == 2 and '7π/4' in every                                  # the default is unchanged
    assert blue(draw_graph(g, time_axis=False, show_time=False, phases='constant')) == ['7π/4']
    assert blue(draw_graph(g, time_axis=False, show_time=False, phases='none', label_size=9)) == []
    with pytest.raises(ValueError):
        draw_graph(g, time_axis=False, show_time=False, phases='some')
    fig = draw_graph(g, time_axis=False, show_time=False, names_right={b: 'g1'}, phases='constant')
    (text,) = [t for t in fig.axes[0].texts if t.get_text() == 'g1']
    assert text.get_position()[0] > g.row(b) and text.get_horizontalalignment() == 'left'
    plt.close('all')
