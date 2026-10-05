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
