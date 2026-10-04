"""Simplification during level-up: blocks with rewrites (zxqec.qprogram.rewrites, QProgram.reduced_block)."""
import os

import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement, Reset, T
from pyzx.utils import VertexType

from zxqec.gpf import boolean_variables, equal_for_all_assignments
from zxqec.qprogram import ParityControlled, QProgram
from zxqec.qprogram.rewrites import REWRITES, register_rewrite

HERE = os.path.dirname(__file__)


def _stage(*gates, qubits=3, bits=2):
    c = Circuit(qubits, bit_amount=bits)
    for g in gates:
        c.add_gate(g)
    return c


def _small_program():
    # three qubits, five stages: a Bell pair and a CNOT, a measurement with a fed-forward correction and a reset,
    # a T gate, a permutation, and a last CNOT
    p = QProgram(3, bits=2, initial_states={1: '0', 2: '0'})
    p.add_circuit(_stage(HAD(1), CNOT(1, 2), CNOT(0, 1)), name='entangle')
    p.add_circuit(_stage(Measurement(1, result_bit=0), ParityControlled('X', 2, ['c[0]']), Reset(1)), name='measure')
    p.add_circuit(_stage(T(2), CNOT(2, 1)), name='T')
    p.add_permutation([0, 2, 1], name='swap')
    p.add_circuit(_stage(CNOT(0, 2), HAD(0)), name='last')
    return p


def _equal(p, level, index):
    """The rewritten block equals the cut block, up to a scalar, on the runs its removed detectors allow."""
    rb = p.reduced_block(level, index)
    cut = p.block_graph(level, index, classical=False)
    names = sorted(set(boolean_variables(cut)) | set(boolean_variables(rb.graph)) |
                   {str(v) for d in rb.detectors for v in d.free_vars()})
    return equal_for_all_assignments(cut, rb.graph, names, rb.detectors)


def test_without_rewrites_a_block_is_its_children_glued():
    # no rewrite anywhere: the glued children are the cut diagram, at every level
    p = _small_program()
    l1 = p.group_stages([(0, 1, 'first'), (2, 4, 'second')])
    l2 = p.level_up(l1, 0, 1, 'all')
    for level in (0, l1, l2):
        for index in range(p.levels.level_size(level)):
            assert p.reduced_block(level, index).rewrite is None and _equal(p, level, index)
    assert p.block(l2, 0).rewrite is None and '<rewrite' not in p.describe_level(l2)


@pytest.mark.parametrize('rewrite', sorted(REWRITES))
def test_every_rewrite_keeps_the_map(rewrite):
    p = _small_program()
    level = p.group_stages([(0, 2, 'first three', None, rewrite), (3, 4, 'last two', None, rewrite)])
    for index in range(2):
        rb = p.reduced_block(level, index)
        assert rb.rewrite == rewrite and _equal(p, level, index)
    assert '<rewrite: {}>'.format(rewrite) in p.describe_level(level)


def test_rewrites_compose_level_by_level():
    # the second level's block is glued from the first level's rewritten blocks, then rewritten again
    p = _small_program()
    l1 = p.group_stages([(0, 1, 'measure', None, 'gpf'), (2, 4, 'rest', None, 'full_reduce')])
    l2 = p.level_up(l1, 0, 1, 'all', rewrite='gpf')
    rb = p.reduced_block(l2, 0)
    assert [c.rewrite for c in rb.children] == ['gpf', 'full_reduce']
    assert rb.spiders_before == sum(c.spiders_after for c in rb.children)
    assert rb.spiders_after <= rb.spiders_before and _equal(p, l2, 0)
    # a rewrite of a single stage: level up one block on its own
    l3 = p.level_up(0, 2, 2, 'T alone', rewrite='spider_simp')
    assert _equal(p, l3, 2)


def test_a_level_rewritten_is_its_blocks_glued_and_keeps_the_map():
    # the whole program with every block of a level replaced by its rewritten diagram
    p = _small_program()
    l1 = p.group_stages([(0, 1, 'measure', None, 'gpf'), (2, 4, 'rest')])
    top = p.level_up(l1, 0, 1, 'all', rewrite='full_reduce')
    q = p.quantum_graph()
    for level in (0, l1, top):
        rl = p.reduced_level(level)
        assert len(rl.blocks) == p.levels.level_size(level)
        assert rl.spiders == sum(rb.spiders_after for rb in rl.blocks)
        assert set().union(*rl.vertices) == set(rl.graph.vertices())                  # every vertex from a block,
        assert sum(len(vs) for vs in rl.vertices) == rl.graph.num_vertices()           # from one block only
        names = sorted(set(boolean_variables(q)) | set(boolean_variables(rl.graph)) |
                       {str(v) for d in rl.detectors for v in d.free_vars()})
        assert equal_for_all_assignments(q, rl.graph, names, rl.detectors)
    assert p.reduced_level(0).spiders == sum(1 for v in q.vertices() if q.type(v) != VertexType.BOUNDARY)
    assert p.reduced_level(l1).detectors == p.reduced_block(l1, 0).detectors and 'level 2: 2 blocks' in \
        str(p.reduced_level(l1))

def test_a_level_in_3d_keeps_space_time_where_there_is_one():
    from pyzx.spacetime import get_timestep
    from zxqec.spacetime import level_layout_3d
    from zxqec.spacetime.drawing import _positions
    p = _small_program()
    q = p.quantum_graph()
    true = {v: tuple(round(c, 6) for c in xyz) for v, xyz in _positions(q, p.topology).items()}

    def spiders(g):
        return [v for v in g.vertices() if g.type(v) != VertexType.BOUNDARY]
    g0, pos0, _ = level_layout_3d(p, 0)                # nothing rewritten: every spider where the program draws it
    assert sorted(tuple(round(c, 6) for c in pos0[v]) for v in spiders(g0)) == sorted(true[v] for v in spiders(q))

    level = p.group_stages([(0, 1, 'measure', None, 'gpf'), (2, 4, 'rest')])
    g, pos, detectors = level_layout_3d(p, level)
    assert set(pos) == set(g.vertices()) and len(spiders(g)) == p.reduced_level(level).spiders
    names = sorted(set(boolean_variables(q)) | set(boolean_variables(g)) |
                   {str(v) for d in detectors for v in d.free_vars()})
    assert equal_for_all_assignments(q, g, names, detectors)      # the same map as reduced_level
    lo, hi = p.stage(0).ticks[0], p.stage(1).ticks[1]
    later = {true[v] for v in spiders(q) if get_timestep(q, v) > hi}
    for v in spiders(g):                               # 'rest' where it is; the rewritten block within its ticks
        assert tuple(round(c, 6) for c in pos[v]) in later or lo - 1 <= pos[v][2] <= hi + 1

def test_merged_blocks_with_a_rewrite():
    # merge_blocks takes a rewrite too (here two adjacent blocks: the swap and the last stage)
    p = _small_program()
    level = p.merge_blocks(0, 3, 4, 'swap + last', rewrite='gpf')
    assert level is not None and _equal(p, level, 3)


def test_unknown_and_registered_rewrites():
    p = _small_program()
    with pytest.raises(ValueError, match='unknown rewrite'):
        p.level_up(0, 0, 1, rewrite='no such rewrite')
    with pytest.raises(ValueError, match='unknown rewrite'):
        p.group_stages([(0, 1, 'a', None, 'no such rewrite')])
    register_rewrite('copy', lambda g: (g.copy(), []))
    try:
        level = p.level_up(0, 0, 1, rewrite='copy')
        assert _equal(p, level, 0)
    finally:
        del REWRITES['copy']


def test_the_cache_follows_the_program():
    p = _small_program()
    level = p.level_up(0, 0, 1, rewrite='gpf')
    first = p.reduced_block(level, 0)
    assert p.reduced_block(level, 0) is first                      # cached
    p.add_circuit(_stage(HAD(2)), name='one more')
    assert p.reduced_block(level, 0) is not first                  # the master graph changed


def test_the_fold_transversal_hadamard_level_by_level_gives_the_whole_reduction():
    # rewriting each block first and the whole program last gives what reducing the whole program at once gives:
    # 2 spiders and 24 detectors (Section 7.1 of the paper)
    import sys
    sys.path.insert(0, os.path.join(HERE, '..', 'examples'))
    import matplotlib
    matplotlib.use('Agg')
    import interactive_fold_h_3d as fth
    from .test_codes import FOLD_GROUPS
    p = fth.build_program()
    level = p.group_stages([g + (r,) for g, r in zip(FOLD_GROUPS, ['gpf', None, 'gpf', None, 'full_reduce', 'gpf'])])
    whole = p.level_up(level, 0, len(FOLD_GROUPS) - 1, 'whole', rewrite='gpf')
    rb = p.reduced_block(whole, 0)
    assert rb.spiders_after == 2 and len(rb.detectors) == 24
    assert rb.spiders_before == sum(c.spiders_after for c in rb.children)
    # the Pauli frame is written with other detector parities added; modulo the detectors it is the same
    from zxqec.gpf import phases_modulo, reduce_gpf
    direct = reduce_gpf(p.quantum_graph())

    def frame(g, detectors):
        h = phases_modulo(g, detectors)
        return sorted(str(h.phase(v)) for v in h.vertices() if h.type(v) != VertexType.BOUNDARY)
    assert frame(rb.graph, rb.detectors) == frame(direct.graph, direct.detectors) == \
        ['c[0] + c[1] + c[2] + c[3] + c[4]', 'c[26] + c[29] + c[32]']


def test_the_levels_window_moves_between_levels_and_blocks():
    import matplotlib
    matplotlib.use('Agg')
    from zxqec.spacetime import view_levels
    p = _small_program()
    l1 = p.group_stages([(0, 1, 'measure', None, 'gpf'), (2, 4, 'rest')])   # one new level per group
    top = p.level_up(l1, 0, 1, 'all', rewrite='full_reduce')
    fig = view_levels(p)
    view = fig.levels_view
    assert (view.level, view.index) == (top, 0) and 'rewrite: full_reduce' in view.describe()
    view.set_level(l1)
    assert view.index == 0 and "made from: 'entangle'; 'measure'" in view.describe()
    view.set_block(1)
    view.set_level(0)                                  # focus kept: the first stage of block 'rest'
    assert view.index == 2 and 'a stage of the circuit' in view.describe()
    view.set_plain(True)
    assert view.check.get_status() == [True]
    view.set_level(top + 3)                            # beyond the top: clamped
    assert view.level == top


def test_spiders_can_be_dragged_spread_out_and_reset():
    import types
    import matplotlib
    matplotlib.use('Agg')
    from zxqec.spacetime import view_levels
    p = _small_program()
    l1 = p.group_stages([(0, 1, 'measure', None, 'gpf'), (2, 4, 'rest')])
    fig = view_levels(p, l1, 0)
    view = fig.levels_view

    def drag(ax, picture, vertex, to):
        x, y = picture.to_px(*picture.positions[vertex])
        view._press(types.SimpleNamespace(inaxes=ax, xdata=x, ydata=y, button=1))
        x, y = picture.to_px(*to)
        view._release(types.SimpleNamespace(inaxes=ax, xdata=x, ydata=y, button=1))

    block = view._current['block']
    v = next(iter(block.positions))
    drag(view.ax_block, block, v, (7.5, -4.0))         # PyZX draws a spider at (row, -qubit)
    row, qubit = view.block_moves[(p.block(l1, 0), False)][v]
    assert (row, qubit) == pytest.approx((7.5, 4.0), abs=0.05) and type(row) is float
    assert view._current['block'].positions[v] == pytest.approx((7.5, -4.0), abs=0.05)

    middle = view._current['level']                     # the program as the level rewrites it
    assert len(middle.spans) == p.levels.level_size(l1)
    u = next(iter(middle.positions))
    drag(view.ax_level, middle, u, (2.0, -1.0))
    assert view.level_moves[l1][u] == pytest.approx((2.0, 1.0), abs=0.05)

    program = view._current['program']
    w = next(iter(program.positions))
    drag(view.ax_program, program, w, (3.0, -2.0))
    view.set_level(0)                                  # the move stays on every level
    assert view._current['program'].positions[w] == pytest.approx((3.0, -2.0), abs=0.05)

    far = program.to_px(-1000.0, 1000.0)               # nowhere near a spider: nothing to drag
    view._press(types.SimpleNamespace(inaxes=view.ax_program, xdata=far[0], ydata=far[1], button=1))
    assert view._drag is None

    view.set_level(l1)
    view.spread_block()
    moves = view.block_moves[(p.block(l1, 0), False)]
    assert set(moves) == set(p.reduced_block(l1, 0).graph.vertices())
    view.reset_positions()
    assert view.program_moves == {} and view.block_moves == {} and view.level_moves == {}

def test_the_levels_window_in_3d():
    import types
    import matplotlib
    matplotlib.use('Agg')
    from zxqec.spacetime import view_levels_3d
    p = _small_program()
    l1 = p.group_stages([(0, 1, 'measure', None, 'gpf'), (2, 4, 'rest')])
    top = p.level_up(l1, 0, 1, 'all', rewrite='full_reduce')
    fig = view_levels_3d(p)
    view = fig.levels_view
    assert (view.level, view.index) == (top, 0) and 'rewrite: full_reduce' in view.describe()
    assert view.ax_block.get_visible() and not view.ax_block3d.get_visible()     # rewritten: PyZX's drawing

    old, lines = list(view._slabs), len(view.ax_program.lines)
    view.set_level(l1)                                 # the program stays; only the slabs are replaced
    assert all(a.axes is None for a in old) and all(a.axes is view.ax_program for a in view._slabs)
    view.set_level(top)
    assert len(view.ax_program.lines) == lines

    view.set_level(l1)
    view.set_block(1)                                  # 'rest' has no rewrite: cut out, in 3D
    assert view.ax_block3d.get_visible() and view.ax_block3d.has_data() and not view.ax_block.get_visible()
    view.set_block(0)
    view.set_plain(True)                               # a rewritten block cut out: in 3D too
    assert view.ax_block3d.get_visible()
    assert view.ax_level.name == '3d'                  # the program as the level rewrites it, in 3D too
    lines = len(view.ax_level.lines)
    view.set_block(1)
    view.set_block(0)                                  # only the outline is replaced between blocks
    assert len(view.ax_level.lines) == lines
    view.set_view('top')                               # the level turns with the program
    assert (view.ax_program.elev, view.ax_level.elev, view.ax_block3d.elev) == (90, 90, 90)
    with pytest.raises(ValueError):
        view.set_view('below')

    press = types.SimpleNamespace(inaxes=view.ax_program, xdata=0.0, ydata=0.0, button=1)
    view._press(press)                                 # on a 3D picture the mouse turns it: nothing to drag
    assert view._drag is None
    view.set_plain(False)
    block = view._current['block']
    v = next(iter(block.positions))
    x, y = block.to_px(*block.positions[v])
    view._press(types.SimpleNamespace(inaxes=view.ax_block, xdata=x, ydata=y, button=1))
    x, y = block.to_px(5.0, -2.0)
    view._release(types.SimpleNamespace(inaxes=view.ax_block, xdata=x, ydata=y, button=1))
    assert view.block_moves[(p.block(l1, 0), False)][v] == pytest.approx((5.0, 2.0), abs=0.05)

def test_cultivation_level_by_level_finds_the_detectors_of_the_whole_reduction():
    # gpf on each block, then gpf on the four rewritten blocks glued: the detectors removed on the way span the
    # same space as those of reducing the whole circuit at once (26 on outcomes, 2 on PyZX's reset variables),
    # and 13 spiders are left, as in the direct reduction
    from zxqec.gpf import detector_rank, reduce_gpf
    from zxqec.qprogram import program_from_stim
    with open(os.path.join(HERE, '..', 'examples', 'circuits', 'cultivation', 'd3_cultivation.stim')) as f:
        p = program_from_stim(f.read())[0]
    level = p.group_stages([(0, 0, 'first checks', None, 'gpf'), (1, 1, 'inject T', None, 'gpf'),
                            (2, 2, 'checks', None, 'gpf'), (3, 4, 'double-check', None, 'gpf')])
    rb = p.reduced_block(p.level_up(level, 0, 3, 'whole', rewrite='gpf'), 0)
    direct = reduce_gpf(p.quantum_graph()).detectors
    found = list(rb.detectors)
    assert detector_rank(found) == detector_rank(direct) == detector_rank(found + list(direct)) == 28
    assert rb.spiders_after == 13
