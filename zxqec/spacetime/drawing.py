# zxqec - space-time ZX-diagrams for dynamic quantum error correction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drawing space-time ZX-diagrams: 1+1D with a time axis, and 2+1D in 3D.

Both views are matplotlib figures (matplotlib is imported lazily):

* :func:`draw_program` -- the master graph of a :class:`~zxqec.qprogram.QProgram`
  drawn with PyZX's 2D renderer, with the horizontal axis being *time* instead
  of PyZX's row index, and the blocks of a level shaded;
* :func:`draw_program_3d` / :func:`draw_graph_3d` -- the 2+1D view: every
  spider at ``(x, y, t)`` where ``(x, y)`` is the position of its physical
  qubit in a :class:`~zxqec.spacetime.Topology` and ``t`` its timestep, ZX
  edges as lines, the hardware connectivity drawn faintly on the first and
  last time layer, and the blocks of a level as translucent slabs;
* :func:`draw_slice`, :func:`draw_slices`, :func:`draw_program_slices` -- one time
  step (or height) of the 2+1D view, flat in the xy-plane, with PyZX's drawing;
* :func:`draw_level_3d` / :func:`level_layout_3d` -- the same for the program
  as a level rewrites it: blocks without a rewrite where they are in space-time,
  the spiders of a rewritten block laid out between the ends of its wires;
  :func:`draw_block_3d` / :func:`block_layout_3d` -- one block of a level, laid
  out the same way; :func:`draw_blocks_3d` -- every block of a level side by side.

Ported from the thesis ``draw_topology_3d`` without its button widgets. Every
function returns the figure: in a notebook it is displayed inline; in a script
call ``matplotlib.pyplot.show()`` to open it in a window, where the 3D views
can be rotated and zoomed with the mouse.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.spacetime import get_timestep, time_extent
from pyzx.utils import EdgeType, VertexType, phase_to_s

from .topology import Topology

if TYPE_CHECKING:
    from ..qprogram.program import QProgram

__all__ = ['draw_program', 'draw_graph', 'draw_checks', 'draw_program_3d', 'draw_graph_3d', 'draw_program_views',
           'slice_vertices', 'draw_slice', 'draw_slices', 'draw_program_slices',
           'draw_level_3d', 'level_layout_3d', 'draw_block_3d', 'block_layout_3d', 'draw_blocks_3d',
           'block_colors', 'block_spans', 'save_panels', 'VIEWS', 'time_positions']

#: Named viewpoints ``(elev, azim)`` for the 3D view: ``default`` is oblique,
#: ``top`` the bird's-eye view onto the hardware layout (time towards the
#: viewer), ``front`` and ``side`` look along ``y`` and ``x`` respectively so
#: that every time layer is a horizontal line.
VIEWS: dict[str, tuple[float, float]] = {
    'default': (25.0, -60.0),
    'top': (90.0, -90.0),
    'front': (0.0, -90.0),
    'side': (0.0, 0.0),
}

# PyZX's default spider colours.
_VERTEX_COLORS = {
    VertexType.Z: '#ccffcc',
    VertexType.X: '#ff8888',
    VertexType.H_BOX: 'yellow',
    VertexType.Z_BOX: '#ccffcc',
    VertexType.BOUNDARY: 'black',
}
_BLOCK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def _finish(fig: Any) -> Any:
    """Prepare a figure for return: closed in a notebook (where the returned
    object is displayed and an open figure would show twice), open otherwise so
    that ``plt.show()`` brings it up in a window (re-attaching a window if the
    figure was closed meanwhile)."""
    import matplotlib.pyplot as plt
    from pyzx.utils import get_mode

    if get_mode() == "notebook":
        plt.close(fig)
    elif fig.canvas.manager is None or not plt.fignum_exists(fig.number):
        manager = plt.figure().canvas.manager     # steal a fresh window manager
        if manager is not None:
            manager.canvas.figure = fig
            fig.set_canvas(manager.canvas)
    return fig


def block_colors(n: int) -> list[str]:
    """``n`` distinguishable colours for shading blocks (cycled beyond ten)."""
    return [_BLOCK_COLORS[i % len(_BLOCK_COLORS)] for i in range(n)]


def _block_tick_ranges(program: 'QProgram', level: int) -> list[tuple[str, list[tuple[int, int]]]]:
    """For every block of ``level``: its label and the tick ranges of its stages."""
    program.graph()                       # makes sure the stage ticks are known
    out = []
    for b in program.levels.blocks(level):
        ranges = [program.stage(s).ticks for s in b.stages if program.stage(s).ticks is not None]
        out.append((str(b), [r for r in ranges if r is not None]))
    return out


# ---------------------------------------------------------------------------
# 1+1D: PyZX's drawing on a time axis
# ---------------------------------------------------------------------------

def time_positions(g: BaseGraph[VT, ET]) -> dict[VT, float]:
    """Horizontal position of every vertex when time is the horizontal axis.

    A spider sits at its timestep; input boundaries one tick before the first
    timestep, outputs one tick after the last; untimed spiders at their row.
    Spiders that share a wire and a tick (a preparation and the first gate on
    it, a measurement and its outcome leaf, ...) are spread across the tick in
    the order of their PyZX rows so they do not overlap.
    """
    ext = time_extent(g)
    lo, hi = ext if ext is not None else (0, 0)
    ins, outs = set(g.inputs()), set(g.outputs())
    pos: dict[VT, float] = {}
    groups: dict[tuple[float, int], list[VT]] = {}
    for v in g.vertices():
        if v in ins:
            pos[v] = float(lo - 1)
        elif v in outs:
            pos[v] = float(hi + 1)
        else:
            t = get_timestep(g, v)
            if t is None:
                pos[v] = float(g.row(v))
            else:
                pos[v] = float(t)
                groups.setdefault((g.qubit(v), t), []).append(v)
    for members in groups.values():
        if len(members) > 1:
            members.sort(key=lambda v: (g.row(v), v))
            step = min(0.45, 0.59 / (len(members) - 1))     # > one node diameter apart, < a tick in total
            for k, v in enumerate(members):
                pos[v] = float(get_timestep(g, v)) + step * k  # type: ignore[arg-type]
    return pos


def _time_layout(g: BaseGraph[VT, ET]) -> BaseGraph[VT, ET]:
    """A copy of ``g`` laid out with time as the horizontal axis (see :func:`time_positions`).

    Every timed spider also gets the short vertex data ``t`` for display.
    """
    h = g.copy()
    if time_extent(h) is None:
        return h
    for v, x in time_positions(h).items():
        h.set_row(v, x)
        t = get_timestep(h, v)
        if t is not None and h.type(v) != VertexType.BOUNDARY:
            h.set_vdata(v, 't', t)
    return h


def program_layout(program: 'QProgram', time_axis: bool = True, classical: bool = False) -> BaseGraph[VT, ET]:
    """The graph :func:`draw_program` draws: the master graph (without classical wires unless
    ``classical``), laid out on a time axis if ``time_axis``. Its vertex numbers are the ones
    the ``positions`` argument of :func:`draw_program` refers to."""
    g = program.graph() if classical else program.quantum_graph()
    return _time_layout(g) if time_axis else g


def draw_program(program: 'QProgram', level: int | None = None, *,
                 time_axis: bool = True, show_time: bool = True, alpha: float = 0.15,
                 classical: bool = False, positions: Any = None, even_rows: bool = False,
                 labels_above: bool = False, blocks: Sequence[int] | None = None,
                 label_size: float | None = None, label_side: str = 'below', **kwargs: Any) -> Any:
    """Draw the master graph of ``program`` in 2D with matplotlib; returns the figure.

    With ``time_axis`` (default) spiders are placed at their timestep along the
    horizontal axis rather than at PyZX's row index, so the length of a wire is
    the time it spans. ``show_time`` adds PyZX's timestep labels and squiggly
    delays. If ``level`` is given, every block of that level is shaded over the
    ticks it occupies and labelled. ``classical=True`` also draws the
    classical-bit wires. ``positions`` maps vertices of :func:`program_layout`
    to ``(row, qubit)`` where they are drawn instead (spiders moved by hand,
    e.g. in :func:`~zxqec.spacetime.levels_view.view_levels`). With ``time_axis=False``,
    ``even_rows`` renumbers the rows in use 0, 1, 2, ... in their order, so spiders that PyZX
    puts half a row apart (a measurement right after a gate) are drawn a whole row apart,
    while spiders in the same row stay aligned. ``labels_above`` adds a strip above the
    top row and writes the block labels there, so that they do not run into its spiders
    (by default they are written over the top row). With ``level``, ``blocks`` draws only
    the blocks of that level with these indices: the drawing is cut to their ticks, so a
    wide program can be shown one part at a time. ``label_size`` is the font size of the
    phase labels (PyZX writes them at 8 points), for a drawing shown smaller than it is
    drawn, as in a paper; each label then hangs just below its spider, on a light
    background, so that a large label does not run into the spider, or with
    ``label_side='right'`` sits just right of it (for spiders one row apart). Remaining
    keyword arguments go to :func:`pyzx.draw_matplotlib` (e.g. ``figsize``, ``labels``).
    """
    from pyzx.drawing import draw_matplotlib
    from pyzx.utils import get_mode, settings

    g = program.graph() if classical else program.quantum_graph()
    h = program_layout(program, time_axis, classical)
    if even_rows:
        h = h.clone()                         # clone keeps the vertex numbers; never move the master graph
        order = {r: i for i, r in enumerate(sorted({h.row(v) for v in h.vertices()}))}
        for v in h.vertices():
            h.set_row(v, order[h.row(v)])
    if positions:
        h = h.clone()                         # clone keeps the vertex numbers; never move the master graph
        for v, (row, qubit) in positions.items():
            if v in h.vertex_set():
                h.set_row(v, row)
                h.set_qubit(v, qubit)
    # pyzx's draw_matplotlib calls plt.show() itself outside notebooks, which
    # would block before the shading is added; make it just return the figure.
    mode = get_mode()
    settings.mode = "notebook"
    try:
        fig = draw_matplotlib(h, show_time=show_time, **kwargs)
    finally:
        settings.mode = mode
    ax = fig.axes[0]
    if label_size is not None:
        _set_labels(ax, label_size, label_side)
    if blocks is not None and level is None:
        raise ValueError("draw_program: blocks needs a level")
    if level is not None:
        colors = block_colors(program.levels.level_size(level))
        ymin, ymax = ax.get_ylim()
        if labels_above:
            ymax += 0.5
            ax.set_ylim(ymin, ymax)
        placed = []                                # (x, label, colour) of each block's label
        shown: list[tuple[float, float]] = []      # the spans of the blocks drawn, with blocks
        for i, ((label, ranges), color) in enumerate(zip(_block_tick_ranges(program, level), colors)):
            for lo, hi in ranges:
                x0, x1 = (lo - 0.5, hi + 0.5) if time_axis else _row_span(h if even_rows else g, program, lo, hi)
                ax.axvspan(x0, x1, facecolor=color, alpha=alpha, zorder=-1)
                if blocks is not None and i in blocks:
                    shown.append((x0, x1))
            if ranges and (blocks is None or i in blocks):
                lo = ranges[0][0]
                x0 = lo - 0.5 if time_axis else _row_span(h if even_rows else g, program, lo, lo)[0]
                placed.append((x0 + 0.1, label, color))
        if blocks is not None:
            if not shown:
                raise ValueError("draw_program: level {} has no blocks {}".format(level, list(blocks)))
            left, right = min(a for a, _b in shown), max(b for _a, b in shown)
            ax.set_xlim(left, right)
            ax.set_adjustable('box')               # keep the scale; the axes get narrower instead
            for t in ax.texts:                     # labels are not clipped by the axes: hide those outside
                if not left - 0.25 <= t.get_position()[0] <= right + 0.25:
                    t.set_visible(False)
        if labels_above:
            _stagger_labels(fig, ax, placed, ymin, ymax)
        else:
            for x, label, color in placed:
                ax.text(x, ymax - 0.15, label, fontsize=7, color=color, va='top')
    return _finish(fig)


def block_spans(program: 'QProgram', level: int, *, time_axis: bool = True, even_rows: bool = False,
                classical: bool = False) -> list[tuple[str, list[tuple[float, float]]]]:
    """Where :func:`draw_program` shades the blocks of ``level``: for every block, its label and the
    horizontal intervals it covers (one per stage tick range), in the drawing's data coordinates.

    ``time_axis``, ``even_rows`` and ``classical`` are the options of :func:`draw_program` (drawn
    without ``positions``). With the axes of the drawing filling its figure, ``(x - left) /
    (right - left)`` for the axes' x-limits ``(left, right)`` is where a block starts or ends as a
    fraction of the figure's width: where a paper's figure can join the drawing to other pictures.
    """
    g = program.graph() if classical else program.quantum_graph()
    rows = program_layout(program, time_axis, classical)
    if even_rows:
        rows = rows.clone()
        order = {r: i for i, r in enumerate(sorted({rows.row(v) for v in rows.vertices()}))}
        for v in rows.vertices():
            rows.set_row(v, order[rows.row(v)])
    return [(label, [(lo - 0.5, hi + 0.5) if time_axis else _row_span(rows if even_rows else g, program, lo, hi)
                     for lo, hi in ranges])
            for label, ranges in _block_tick_ranges(program, level)]


def _stagger_labels(fig: Any, ax: Any, placed: list[tuple[float, str, str]], ymin: float, ymax: float) -> None:
    """Block labels above the top row, each on the lowest line where it clears the labels on its left.

    The lowest line is where a single line of labels goes; a label that would run into
    the one before it (a narrow block) goes one line up, and the strip grows to hold
    the lines. Labels that fit on one line are placed exactly as before. Growing the
    strip can change the scale of the drawing (its aspect is fixed), so the labels are
    measured again until the spacing settles."""
    texts = [ax.text(x, ymax - 0.15, label, fontsize=7, color=color, va='top') for x, label, color in placed]
    lines, step = [0] * len(texts), 0.0
    for attempt in range(5):
        if attempt:                                # once there are several lines: the scale the figure will have
            ax.apply_aspect()
        inv = ax.transData.inverted()
        ends: list[float] = []                     # right end of the last label on each line, lowest line first
        height = 0.0
        for i, t in enumerate(texts):
            bb = t.get_window_extent()             # the figure's own renderer, on any canvas
            (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            height = max(height, abs(y1 - y0))
            k = next((j for j, end in enumerate(ends) if end < x0 - 0.1), len(ends))
            if k == len(ends):
                ends.append(x1)
            else:
                ends[k] = x1
            lines[i] = k
        if len(ends) == 1:
            return                                 # one line: where block labels always go
        if step and abs(height * 1.3 - step) < 0.01 * step:
            break
        step = height * 1.3
        ax.set_ylim(ymin, ymax + step * (len(ends) - 1))
    for t, k in zip(texts, lines):
        t.set_y(ymax - 0.15 + k * step)


def draw_graph(g: BaseGraph[VT, ET], *, time_axis: bool = True, show_time: bool = True,
               label_size: float | None = None, label_side: str = 'below',
               names: dict[Any, str] | None = None, **kwargs: Any) -> Any:
    """PyZX's 2D drawing (``draw_matplotlib``) of a time-annotated diagram, time along the horizontal axis.

    The static counterpart of :func:`~zxqec.spacetime.draw_web`, for any diagram
    (a stage or block cut out of a program, a reduced diagram): every spider at its
    timestep (see :func:`time_positions`; ``time_axis=False`` keeps PyZX's rows), and
    with ``show_time`` PyZX's timestep labels and squiggly delays. ``label_size`` and
    ``label_side`` place the phase labels, as in :func:`draw_program`. ``names`` maps
    vertices of ``g`` to a name written just left of the spider (e.g. the bits of a
    Z-layer), at ``label_size`` (12 points when it is None). Keyword arguments go to ``draw_matplotlib`` (e.g. ``figsize``,
    ``labels``). Returns the figure.
    """
    from pyzx.drawing import draw_matplotlib
    from pyzx.utils import get_mode, settings

    h = _time_layout(g) if time_axis else g
    mode = get_mode()
    settings.mode = "notebook"                # PyZX would call plt.show() itself otherwise
    try:
        fig = draw_matplotlib(h, show_time=show_time, **kwargs)
    finally:
        settings.mode = mode
    if label_size is not None:
        _set_labels(fig.axes[0], label_size, label_side)
    at = dict(zip(g.vertices(), h.vertices()))     # g.copy() numbers the vertices afresh, in this order
    for v, name in (names or {}).items():          # PyZX puts a vertex at (row, -qubit)
        w = at[v]
        fig.axes[0].text(h.row(w) - 0.3, -h.qubit(w), name, ha='right', va='center',
                         fontsize=label_size if label_size is not None else 12)
    return _finish(fig)


def draw_checks(checks: Sequence[Any], names: Sequence[str] | None = None,
                figsize: tuple[float, float] = (8, 5.5)) -> Any:
    """Which results each check adds up, as a grid: one row per check, one column per result.

    ``checks`` are parities of results that are 0 in every run, such as the detectors
    of :func:`~zxqec.gpf.reduce_gpf`. A dark square in row ``i`` and column ``c`` means
    that check ``i`` reads result ``c``. ``names`` are the columns, by default every
    variable of the checks, ordered by the number in its name (``c[0], c[1], ...``);
    the rows are ordered by the first result they read. Returns the figure.
    """
    import re

    import matplotlib.pyplot as plt

    def number(name: str) -> tuple[str, int]:
        m = re.fullmatch(r'(.*?)\[?(\d+)\]?', name)
        return (m.group(1), int(m.group(2))) if m else (name, -1)

    read = [{v.name for v in c.free_vars()} if hasattr(c, 'free_vars') else set() for c in checks]
    cols = list(names) if names is not None else sorted(set().union(*read), key=number)
    rows = sorted(([1 if n in r else 0 for n in cols] for r in read),
                  key=lambda row: [-x for x in row])
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rows, cmap='Greys', vmin=0, vmax=1.3, aspect='equal')
    heads = {number(n)[0] for n in cols}
    short = len(heads) == 1 and all(number(n)[1] >= 0 for n in cols)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(number(n)[1]) if short else n for n in cols], fontsize=8,
                       rotation=0 if short and len(cols) <= 20 else 90)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([str(k + 1) for k in range(len(rows))], fontsize=8)
    ax.set_xlabel('result {}[k]'.format(heads.pop()) if short else 'result')
    ax.set_ylabel('check')
    ax.set_xticks([x - 0.5 for x in range(len(cols) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(rows) + 1)], minor=True)
    ax.grid(which='minor', color='#cccccc', linewidth=0.5)
    ax.tick_params(which='minor', length=0)
    return _finish(fig)


def _set_labels(ax: Any, size: float, side: str = 'below') -> None:
    """PyZX's phase labels at font ``size``, next to their spider, on a light background.

    PyZX writes a label with its baseline half a row below the spider (radius 0.2), so
    a large label runs up into it. With ``side='below'`` its top is a quarter of a row
    below the spider instead; with ``side='right'`` it sits just right of the spider,
    centred on its row. The background keeps it readable where edges pass under it.
    """
    if side not in ('below', 'right'):
        raise ValueError("label_side is 'below' or 'right', not {!r}".format(side))
    for t in ax.texts:
        if t.get_color() != 'blue':                # only PyZX's phase labels
            continue
        x, y = t.get_position()
        if side == 'below':
            t.set_position((x, y + 0.25))
            t.set_verticalalignment('top')
        else:
            t.set_position((x + 0.28, y + 0.5))
            t.set_horizontalalignment('left')
            t.set_verticalalignment('center')
        t.set_fontsize(size)
        t.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.3))


def _row_span(g: BaseGraph[VT, ET], program: 'QProgram', lo: int, hi: int) -> tuple[float, float]:
    """Row interval of the spiders scheduled on ticks ``lo..hi`` (for ``time_axis=False``)."""
    rows = [g.row(v) for v in g.vertices()
            if g.type(v) != VertexType.BOUNDARY and get_timestep(g, v) is not None
            and lo <= get_timestep(g, v) <= hi]  # type: ignore[operator]
    if not rows:
        return (lo, hi)
    return (min(rows) - 0.5, max(rows) + 0.5)


# ---------------------------------------------------------------------------
# 2+1D: spiders at (x, y, t)
# ---------------------------------------------------------------------------

def _positions(g: BaseGraph[VT, ET], topology: Topology | None) -> dict[VT, tuple[float, float, float]]:
    """``(x, y, t)`` of every vertex: topology coordinates of its physical wire and its timestep.

    Without a topology the wires are laid out on a line (``y = 0``). Vertices
    at fractional qubit coordinates (e.g. an H-box between two wires) are
    offset from the lower wire. Vertices without a timestep use their row.
    """
    times = time_positions(g)
    pos: dict[VT, tuple[float, float, float]] = {}
    for v in g.vertices():
        q = g.qubit(v)
        base = int(q // 1)
        frac = float(q - base)
        if topology is not None and base in topology:
            c = topology.coordinates(base)
            x, y = c[0], (c[1] if topology.dimension >= 2 else 0.0)
            x += 0.4 * frac
        else:
            x, y = float(base) + frac, 0.0
        pos[v] = (x, y, times[v])
    return pos


def draw_graph_3d(g: BaseGraph[VT, ET], topology: Topology | None = None, *,
                  blocks: Sequence[tuple[str, Sequence[tuple[int, int]]]] | None = None,
                  show_topology: bool = True, show_phases: bool = True, labels: bool = False,
                  spider_size: float = 60, figsize: tuple[float, float] = (8, 6),
                  view: str | tuple[float, float] = 'default',
                  elev: float | None = None, azim: float | None = None,
                  layer_lines: bool | None = None, buttons: bool = False,
                  ax: Any = None, finish: bool = True,
                  positions: dict[Any, tuple[float, float, float]] | None = None,
                  space_ticks: bool = True, phases_beside: bool = False, hadamard_boxes: bool = False,
                  bare: bool = False) -> Any:
    """Draw a time-annotated graph in 3D: space in the ``x``/``y`` plane, time upwards.

    ``topology`` gives the ``(x, y)`` position of every physical wire (the
    graph's ``qubit`` coordinate must be a node of it); without one the wires
    are spread along ``x``. ``blocks`` is a list of ``(label, tick_ranges)``
    drawn as translucent slabs. With ``show_topology`` the hardware graph is
    drawn faintly on the bottom and top time layers.

    ``view`` selects the viewpoint: a name from :data:`VIEWS` (``'default'``,
    ``'top'`` for the bird's-eye view, ``'front'``, ``'side'``) or an
    ``(elev, azim)`` pair; ``elev`` / ``azim`` override it. ``layer_lines``
    draws a faint guide line at every occupied timestep (on by default for
    the front and side views). The figure can be rotated with the mouse in
    any interactive matplotlib backend; ``buttons=True`` adds one button per
    named view below the plot that turns the figure to that viewpoint (meant
    for windows, not for static notebook output). Returns the figure.
    ``finish=False`` leaves the figure as it is (by default it is closed in a
    notebook, so that it is not shown twice), for drawing into a figure that
    stays open, such as a window. ``positions`` gives ``(x, y, t)`` for some
    vertices, used instead of their wire and timestep (as :func:`draw_level_3d`
    does for the spiders of a rewritten block). ``space_ticks=False`` leaves out
    the numbers on the ``x`` and ``y`` axes, for a topology whose coordinates are
    a layout rather than a chip's (:func:`~zxqec.spacetime.relaxed_topology`).
    ``phases_beside=True`` writes each phase a few points beside its spider on
    the screen, on a pale background, instead of just under it in time: from
    above (``view='top'``) a spider would cover it. Of eight sides around the
    spider it takes the one farthest from the spider's edges as seen from the
    viewpoint, the lower right when that is free.
    ``hadamard_boxes=True`` draws a Hadamard edge as a plain edge with a yellow
    square in its middle, and an H-box as a yellow square, as PyZX does in 2D,
    instead of a blue edge and a round marker; a phase-free spider with two
    edges, one of them a Hadamard edge, is a Hadamard gate on a wire and is drawn
    as that square alone, in its place. ``bare=True`` leaves out the grey panes,
    the grid and the axes with their numbers: only the diagram.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401  (registers the 3d projection)

    if isinstance(view, str):
        if view not in VIEWS:
            raise ValueError("unknown view {!r}; choose from {}".format(view, sorted(VIEWS)))
        v_elev, v_azim = VIEWS[view]
    else:
        v_elev, v_azim = view
    if elev is not None:
        v_elev = elev
    if azim is not None:
        v_azim = azim
    if layer_lines is None:
        layer_lines = abs(v_elev) < 1e-9          # front / side views

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    pos = _positions(g, topology)
    if positions:
        pos.update({v: p for v, p in positions.items() if v in pos})
    if not pos:
        return _finish(fig) if finish else fig

    # hardware connectivity on the bottom and top time layers
    ticks = sorted({int(z) for v, (_, _, z) in pos.items() if g.type(v) != VertexType.BOUNDARY})
    zs_all = [p[2] for p in pos.values()]
    if show_topology and topology is not None:
        for t in (min(zs_all), max(zs_all)):
            for a, b in topology.edge_list():
                ca, cb = topology.coordinates(a), topology.coordinates(b)
                ya = ca[1] if topology.dimension >= 2 else 0.0
                yb = cb[1] if topology.dimension >= 2 else 0.0
                ax.plot([ca[0], cb[0]], [ya, yb], [t, t], color='gray', alpha=0.35, linewidth=0.8)
            for n in topology:
                c = topology.coordinates(n)
                ax.scatter([c[0]], [c[1] if topology.dimension >= 2 else 0.0], [t],
                           c='gray', s=6, alpha=0.5, depthshade=False)

    # with hadamard_boxes, a phase-free spider on a wire with a Hadamard edge on one side is the Hadamard gate
    # itself (the spider only carries the edge): it is drawn as the yellow box, and its edge as a plain one
    gate_spiders: dict[Any, Any] = {}
    if hadamard_boxes:
        for v in pos:
            edges = list(g.incident_edges(v))
            kinds = [g.edge_type(e) for e in edges]
            if g.type(v) in (VertexType.Z, VertexType.X) and g.phase(v) == 0 and len(edges) == 2 \
                    and kinds.count(EdgeType.HADAMARD) == 1:
                gate_spiders[v] = edges[kinds.index(EdgeType.HADAMARD)]
    gate_edges = set(gate_spiders.values())
    box = dict(c='yellow', marker='s', s=spider_size * 1.6, edgecolors='black', linewidths=0.5,
               depthshade=False)                    # larger than a spider: seen around one on the same wire from above

    # what is drawn, for _place_labels: the phases written beside spiders, the markers and the edges
    drawn = ax.__dict__.setdefault('_zx_drawn', {'labels': [], 'points': [], 'segments': []})
    drawn['points'] += [(p, spider_size * (0.3 if g.type(v) == VertexType.BOUNDARY else 1.0)) for v, p in pos.items()]

    # ZX edges
    for e in g.edges():
        s, t_ = g.edge_st(e)
        (xs, ys, zs), (xt, yt, zt) = pos[s], pos[t_]
        drawn['segments'].append((pos[s], pos[t_]))
        hadamard = g.edge_type(e) == EdgeType.HADAMARD
        color = '#0099ff' if hadamard and not hadamard_boxes else 'black'
        ax.plot([xs, xt], [ys, yt], [zs, zt], color=color, linewidth=1.0, alpha=0.9)
        if hadamard and hadamard_boxes and e not in gate_edges:     # a yellow box in the middle of the edge
            ax.scatter([(xs + xt) / 2], [(ys + yt) / 2], [(zs + zt) / 2], **box)

    # spiders
    for v, (x, y, z) in pos.items():
        ty = g.type(v)
        if v in gate_spiders:
            ax.scatter([x], [y], [z], **box)
            continue
        size = spider_size * (0.3 if ty == VertexType.BOUNDARY else 1.0)
        ax.scatter([x], [y], [z], c=_VERTEX_COLORS.get(ty, 'black'), s=size,
                   marker='s' if hadamard_boxes and ty == VertexType.H_BOX else 'o',
                   edgecolors='black', linewidths=0.5, depthshade=False)
        if show_phases and ty != VertexType.BOUNDARY:
            ps = phase_to_s(g.phase(v), ty)
            if ps and phases_beside:                # shifted on the screen, so no viewpoint puts it under the spider
                from matplotlib.transforms import offset_copy
                angle = _free_side(g, v, pos, v_elev, v_azim)
                dx, dy = np.cos(angle), np.sin(angle)
                text = ax.text(x, y, z, ps, fontsize=7, color='blue',
                               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.75))
                _put_on_side(text, angle)
                drawn['labels'].append((text, (x, y, z)))
            elif ps:
                ax.text(x, y, z - 0.25, ps, fontsize=7, color='blue', ha='center', va='top')
        if labels:
            ax.text(x, y, z + 0.15, str(v), fontsize=6, color='gray', ha='center')

    # blocks as translucent slabs spanning the occupied region in space
    if blocks:
        _block_slabs(ax, *_space_extent(pos), blocks)

    if layer_lines and ticks:
        xs_all = [p[0] for p in pos.values()]
        ys_all = [p[1] for p in pos.values()]
        x0, x1 = min(xs_all) - 0.3, max(xs_all) + 0.3
        y0, y1 = min(ys_all) - 0.3, max(ys_all) + 0.3
        for t in ticks:
            ax.plot([x0, x1], [y0, y0], [t, t], color='gray', alpha=0.3, linewidth=0.6, linestyle=':')
            ax.plot([x0, x0], [y0, y1], [t, t], color='gray', alpha=0.3, linewidth=0.6, linestyle=':')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('time')
    if not space_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if 0 < len(ticks) <= 15:
        ax.set_zticks(ticks)                  # one tick label per occupied timestep
    elif ticks:
        from matplotlib.ticker import MaxNLocator
        ax.zaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    # hide the axis that collapses to a point in the flat viewpoints
    if abs(v_elev - 90) < 1e-9:
        ax.set_zticks([])
        ax.set_zlabel('')
    elif abs(v_elev) < 1e-9:
        collapsed = ax.yaxis if abs(v_azim % 180 - 90) < 1e-9 else ax.xaxis
        collapsed.set_ticks([])
        collapsed.set_label_text('')
    ax.view_init(elev=v_elev, azim=v_azim)
    if bare:
        ax.set_axis_off()
        ax.patch.set_visible(False)              # no background over a neighbouring picture's labels
    if buttons:
        _add_view_buttons(fig, ax)
    return _finish(fig) if finish else fig


def _put_on_side(text: Any, angle: float, gap: float = 5) -> None:
    """Write ``text`` ``gap`` points from its anchor towards the screen angle ``angle``, aligned away from it."""
    from matplotlib.transforms import offset_copy
    dx, dy = np.cos(angle), np.sin(angle)
    text.set_horizontalalignment('left' if dx > 0.3 else 'right' if dx < -0.3 else 'center')
    text.set_verticalalignment('bottom' if dy > 0.3 else 'top' if dy < -0.3 else 'center')
    text.set_transform(offset_copy(text.axes.transData, fig=text.axes.figure, x=gap * dx, y=gap * dy,
                                   units='points'))
    text._zx_side = angle


def _place_labels(ax: Any, gap: float = 5) -> None:
    """Move each phase written beside a spider to the side where it covers least, as the axes are now drawn.

    For every label in turn, each of the eight sides is scored on the screen:
    the area it shares with labels already placed, the spiders and boxes it
    covers, and the edges that cross it. The side with the lowest score wins,
    the side it has on a tie. Call it once the axes have their final size and
    viewpoint (:func:`draw_blocks_3d` does).
    """
    from mpl_toolkits.mplot3d import proj3d

    drawn = ax.__dict__.get('_zx_drawn')
    if not drawn or not drawn['labels']:
        return
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    proj = ax.get_proj()
    px = fig.dpi / 72                                 # pixels per point

    def screen(p: tuple[float, float, float]) -> np.ndarray:
        x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], proj)
        return ax.transData.transform((x, y))

    points = [(screen(p), np.sqrt(size) / 2 * px) for p, size in drawn['points']]
    segments = [(screen(a), screen(b)) for a, b in drawn['segments']]
    sides = [np.radians(-45 + 45 * k) for k in range(8)]
    placed: list[tuple[float, float, float, float]] = []

    def box(anchor: np.ndarray, w: float, h: float, angle: float) -> tuple[float, float, float, float]:
        dx, dy = np.cos(angle), np.sin(angle)
        x = anchor[0] + gap * px * dx - (0 if dx > 0.3 else w if dx < -0.3 else w / 2)
        y = anchor[1] + gap * px * dy - (0 if dy > 0.3 else h if dy < -0.3 else h / 2)
        return x, y, x + w, y + h

    def cost(b: tuple[float, float, float, float], own: np.ndarray) -> float:
        x0, y0, x1, y1 = b
        overlap = sum(max(0.0, min(x1, c1) - max(x0, c0)) * max(0.0, min(y1, d1) - max(y0, d0))
                      for c0, d0, c1, d1 in placed)
        covered = sum(1 for c, r in points if np.linalg.norm(c - own) > 1e-6
                      and x0 - r < c[0] < x1 + r and y0 - r < c[1] < y1 + r)
        crossed = sum(1 for a, z in segments
                      if any(x0 < q[0] < x1 and y0 < q[1] < y1 for q in (a + (z - a) * t for t in np.linspace(0, 1, 25))))
        return overlap / 4 + 40 * covered + 15 * crossed

    for text, anchor3d in drawn['labels']:
        anchor = screen(anchor3d)
        extent = text.get_window_extent(renderer)
        w, h = extent.width, extent.height
        start = getattr(text, '_zx_side', sides[0])
        best = min(sides, key=lambda a: (cost(box(anchor, w, h, a), anchor), abs(a - start) > 1e-9))
        _put_on_side(text, best, gap)
        placed.append(box(anchor, w, h, best))


def _free_side(g: BaseGraph, v: Any, pos: dict[Any, tuple[float, float, float]], elev: float,
               azim: float) -> float:
    """The screen angle (radians) of the side of ``v`` farthest from its edges, seen from ``(elev, azim)``.

    An edge's direction on the screen: its ``x``/``y`` part turned by the
    azimuth, and its time part upwards, shortened by how far the view looks
    down (time spans more units than space, so it counts as at most half a unit).
    Of the eight sides at 45 degrees the one whose least angle to an edge is
    largest wins; on a tie the lower right, then the order of the sides.
    """
    a, e = np.radians(azim), np.radians(elev)
    x, y, z = pos[v]
    edges = []
    for u in g.neighbors(v):
        dx, dy, dz = pos[u][0] - x, pos[u][1] - y, pos[u][2] - z
        dz = float(np.sign(dz)) * min(abs(dz), 0.5)
        sx = -dx * np.sin(a) + dy * np.cos(a)
        sy = (-dx * np.cos(a) - dy * np.sin(a)) * np.sin(e) + dz * np.cos(e)
        if abs(sx) + abs(sy) > 1e-9:
            edges.append(np.arctan2(sy, sx))
    sides = [np.radians(-45 + 45 * k) for k in range(8)]       # lower right first
    if not edges:
        return sides[0]

    def room(side: float) -> float:
        return min(abs((side - t + np.pi) % (2 * np.pi) - np.pi) for t in edges)

    best = max(room(s) for s in sides)
    return next(s for s in sides if room(s) > best - 1e-9)


def _space_extent(pos: dict[Any, tuple[float, float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """The ``x`` and ``y`` ranges of the occupied region, with a margin: what a block's slab spans."""
    xs_all = [p[0] for p in pos.values()]
    ys_all = [p[1] for p in pos.values()]
    return (min(xs_all) - 0.3, max(xs_all) + 0.3), (min(ys_all) - 0.3, max(ys_all) + 0.3)


def _block_slabs(ax: Any, xr: tuple[float, float], yr: tuple[float, float],
                 blocks: Sequence[tuple[str, Sequence[tuple[int, int]]]]) -> list[Any]:
    """Every block as a translucent slab over its tick ranges, labelled in its colour; returns the artists."""
    artists: list[Any] = []
    for (label, ranges), color in zip(blocks, block_colors(len(blocks))):
        for lo, hi in ranges:
            artists += _slab(ax, xr, yr, (lo - 0.5, hi + 0.5), color)
        if ranges:
            artists.append(ax.text(xr[0], yr[0], ranges[0][0] - 0.5, label, fontsize=7, color=color))
    return artists


def _add_view_buttons(fig: Any, ax: Any) -> None:
    """One button per named viewpoint along the bottom of ``fig``."""
    from matplotlib.widgets import Button

    names = list(VIEWS)
    width, gap, height = 0.11, 0.015, 0.045
    left = 0.5 - (len(names) * width + (len(names) - 1) * gap) / 2
    fig.subplots_adjust(bottom=0.12)
    widgets = []
    for i, name in enumerate(names):
        bax = fig.add_axes([left + i * (width + gap), 0.02, width, height])
        button = Button(bax, name)

        def turn(_event: Any, name: str = name) -> None:
            e, a = VIEWS[name]
            ax.view_init(elev=e, azim=a)
            fig.canvas.draw_idle()

        button.on_clicked(turn)
        widgets.append(button)
    fig._zxqec_view_buttons = widgets      # keep references alive, else the callbacks die


def draw_program_views(program: 'QProgram', level: int | None = None, *,
                       views: Sequence[str | tuple[float, float]] = ('default', 'top', 'front'),
                       figsize: tuple[float, float] | None = None, **kwargs: Any) -> Any:
    """Several viewpoints of the 2+1D view of a program side by side.

    ``views`` are names from :data:`VIEWS` or ``(elev, azim)`` pairs; the
    default shows the oblique, bird's-eye and front views. Other keyword
    arguments are passed to :func:`draw_graph_3d`. Returns the figure.
    """
    import matplotlib.pyplot as plt

    blocks = _block_tick_ranges(program, level) if level is not None else None
    n = len(views)
    fig = plt.figure(figsize=figsize or (6 * n, 6))
    g = program.quantum_graph()
    for i, view in enumerate(views):
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        draw_graph_3d(g, program.topology, blocks=blocks, view=view, ax=ax, **kwargs)
        ax.set_title(view if isinstance(view, str) else "elev {}, azim {}".format(*view), fontsize=10)
    return _finish(fig)


def _slab(ax: Any, xr: tuple[float, float], yr: tuple[float, float],
          zr: tuple[float, float], color: str, *, face_alpha: float = 0.06, linewidth: float = 0.6,
          linestyle: str = '-', line_alpha: float = 0.6) -> list[Any]:
    """A translucent slab on a 3D axes: tinted top and bottom faces plus its outline; returns the artists."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    (x0, x1), (y0, y1), (z0, z1) = xr, yr, zr
    bottom = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
    top = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    artists: list[Any] = []
    if face_alpha:
        artists.append(ax.add_collection3d(Poly3DCollection([bottom, top], facecolors=color, alpha=face_alpha,
                                                            linewidths=0)))
    line = dict(color=color, linewidth=linewidth, linestyle=linestyle, alpha=line_alpha)
    for face in (bottom, top):
        xs, ys, zs = zip(*(face + face[:1]))
        artists += ax.plot(xs, ys, zs, **line)
    for (x, y, _), (_, _, _) in zip(bottom, top):
        artists += ax.plot([x, x], [y, y], [z0, z1], **line)
    return artists


def draw_program_3d(program: 'QProgram', level: int | None = None, classical: bool = False,
                    **kwargs: Any) -> Any:
    """The 2+1D view of a program: :func:`draw_graph_3d` on its master graph.

    Uses the program's topology for the spatial layout and, if ``level`` is
    given, shades the blocks of that level; ``classical=True`` also draws the
    classical-bit wires. Keyword arguments are passed on.
    """
    blocks = _block_tick_ranges(program, level) if level is not None else None
    g = program.graph() if classical else program.quantum_graph()
    return draw_graph_3d(g, program.topology, blocks=blocks, **kwargs)


# ---------------------------------------------------------------------------
# slices: one time step (or one height) of a 2+1D drawing, flat
# ---------------------------------------------------------------------------

def slice_vertices(g: BaseGraph[VT, ET], *, t: int | None = None, z: float | tuple[float, float] | None = None,
                   positions: dict[Any, tuple[float, float, float]] | None = None,
                   topology: Topology | None = None) -> list[Any]:
    """The vertices in one slice of a 2+1D drawing: the spiders at timestep ``t``, or at height ``z``.

    ``z`` is a height (within a quarter of a unit) or a range ``(low, high)`` of
    heights, of the positions :func:`draw_graph_3d` uses (``positions`` overriding
    some of them, as in :func:`draw_level_3d`).
    """
    if (t is None) == (z is None):
        raise ValueError("give either a timestep t or a height z")
    if t is not None:
        return [v for v in g.vertices() if g.type(v) != VertexType.BOUNDARY and get_timestep(g, v) == t]
    pos = _positions(g, topology)
    if positions:
        pos.update({v: p for v, p in positions.items() if v in pos})
    lo, hi = (z - 0.25, z + 0.25) if isinstance(z, (int, float)) else z
    return [v for v in g.vertices() if lo <= pos[v][2] <= hi]


def draw_slice(g: BaseGraph[VT, ET], topology: Topology | None = None, *, t: int | None = None,
               z: float | tuple[float, float] | None = None,
               positions: dict[Any, tuple[float, float, float]] | None = None, show_topology: bool = True,
               extent: Any = None, title: str | None = None, figsize: tuple[float, float] = (4, 4),
               **kwargs: Any) -> Any:
    """One slice of the 2+1D view, flat: the spiders at timestep ``t`` (or height ``z``) at their ``(x, y)``.

    The spiders of :func:`slice_vertices`, with the edges between them, are drawn
    by PyZX's own ``draw_matplotlib`` at the positions :func:`draw_graph_3d` gives
    them in the plane (a qubit's place in ``topology``); edges to other time steps
    are left out, so a layer of CNOTs shows which qubits it couples. With
    ``show_topology`` the hardware graph is drawn faintly underneath. ``extent``,
    ``((x0, x1), (y0, y1))``, fixes the window (by default the whole layout of
    ``g``), so that slices of one diagram can be compared. Keyword arguments go to
    ``draw_matplotlib``. Returns the figure.
    """
    from pyzx.drawing import draw_matplotlib
    from pyzx.utils import get_mode, settings

    keep = set(slice_vertices(g, t=t, z=z, positions=positions, topology=topology))
    pos = _positions(g, topology)
    if positions:
        pos.update({v: p for v, p in positions.items() if v in pos})
    h = g.clone()                                       # clone keeps the vertex numbers
    h.remove_vertices([v for v in list(h.vertices()) if v not in keep])
    h.set_inputs(())
    h.set_outputs(())
    for v in h.vertices():
        h.set_row(v, pos[v][0])
        h.set_qubit(v, -pos[v][1])                      # PyZX draws a vertex at (row, -qubit)
    mode = get_mode()
    settings.mode = "notebook"                          # PyZX would call plt.show() itself otherwise
    try:
        fig = draw_matplotlib(h, figsize=figsize, **kwargs)
    finally:
        settings.mode = mode
    ax = fig.axes[0]
    if show_topology and topology is not None:
        for a, b in topology.edge_list():
            ca, cb = topology.coordinates(a), topology.coordinates(b)
            ya = ca[1] if topology.dimension >= 2 else 0.0
            yb = cb[1] if topology.dimension >= 2 else 0.0
            ax.plot([ca[0], cb[0]], [ya, yb], color='gray', alpha=0.35, linewidth=0.8, zorder=0)
    if extent is None and pos:
        extent = _space_extent(pos)
    if extent is not None:
        (x0, x1), (y0, y1) = extent
        ax.set_aspect('equal', adjustable='box')        # PyZX's axes adjust their limits instead
        ax.set_xlim(x0 - 0.5, x1 + 0.5)
        ax.set_ylim(y0 - 0.5, y1 + 0.5)
    ax.set_title(title if title is not None else ('t = {}'.format(t) if t is not None else 'height {}'.format(z)),
                 fontsize=9)
    return _finish(fig)


def draw_slices(g: BaseGraph[VT, ET], topology: Topology | None = None, ticks: Sequence[int] | None = None, *,
                cols: int = 4, panel: tuple[float, float] = (3, 3), **kwargs: Any) -> Any:
    """The slices of ``g`` at the timesteps ``ticks`` (by default every occupied one), side by side.

    Each panel is :func:`draw_slice` (PyZX's drawing, rendered to an image), all
    with the same window; ``cols`` panels per row, each ``panel`` inches. Keyword
    arguments go to :func:`draw_slice`. Returns the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if ticks is None:
        ticks = sorted({get_timestep(g, v) for v in g.vertices()
                        if g.type(v) != VertexType.BOUNDARY and get_timestep(g, v) is not None})
    ticks = list(ticks)
    extent = kwargs.pop('extent', None) or _space_extent(_positions(g, topology))
    images = []
    for t in ticks:
        f = draw_slice(g, topology, t=t, extent=extent, figsize=panel, **kwargs)
        f.dpi = 110
        canvas = FigureCanvasAgg(f)
        canvas.draw()
        images.append(np.asarray(canvas.buffer_rgba()).copy())
        plt.close(f)
    cols = max(1, min(cols, len(ticks)))
    rows = max(1, -(-len(ticks) // cols))
    fig = plt.figure(figsize=(panel[0] * cols, panel[1] * rows))
    for k, img in enumerate(images):
        ax = fig.add_subplot(rows, cols, k + 1)
        ax.imshow(img)
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    return _finish(fig)


def draw_program_slices(program: 'QProgram', stage: int | None = None, ticks: Sequence[int] | None = None,
                        **kwargs: Any) -> Any:
    """:func:`draw_slices` of a program's master graph: the timesteps of one ``stage``, or ``ticks``, or all.

    Each panel shows what happens at one time step, on the program's topology.
    Keyword arguments go to :func:`draw_slices`. Returns the figure.
    """
    g = program.quantum_graph()
    if stage is not None:
        if ticks is not None:
            raise ValueError("give a stage or ticks, not both")
        program.graph()                                 # makes sure the stage ticks are known
        span = program.stage(stage).ticks
        if span is None:
            raise ValueError("stage {} takes no time".format(stage))
        ticks = range(span[0], span[1] + 1)
    return draw_slices(g, program.topology, ticks, **kwargs)


def draw_stage_layout(program: 'QProgram', stage: int, wires: Sequence[int] | None = None, *, code: Any = None,
                      spacing: float = 1.6, figsize: tuple[float, float] | None = None, y_down: bool = True,
                      label_size: float | None = None, face_colors: dict[str, str] | None = None,
                      face_alpha: float = 0.35, wire_ends: str = 'dot', title: str | None = None) -> Any:
    """One stage of a program drawn flat on its hardware layout, as a transversal layer or a move of qubits is drawn by hand.

    The stage's spiders of the master graph (:meth:`QProgram.quantum_graph`) on the
    physical ``wires`` (all by default), each at its wire's place in the program's
    topology, one unit of the topology drawn ``spacing`` long. An edge to a spider
    outside the stage, or on a wire left out, becomes a wire end at that spider's
    place, and spiders that do nothing (phase 0 and two wires) are removed by PyZX's
    ``id_simp``. A wire end that is then where its neighbour is becomes a short stub,
    above if it stands for an earlier spider and below if for a later one: a wire that
    carries only a Hadamard is a stub through a Hadamard box, drawn 0.28 of a step wide.
    A wire that the stage moves (a permutation) comes in as a stub above its old place
    and leaves as a stub below its new one, the two joined by a line, bowed outwards
    where a straight line would pass through another place or lie on the stubs, and
    bowed apart where two wires trade places.
    Meant for a stage one timestep deep (a layer of one-qubit gates, a permutation):
    the spiders of one wire at several timesteps would be drawn on top of each other,
    and a two-qubit gate between places one above the other has its wire on their
    stubs. A Hadamard edge to a spider of the next or the last stage is drawn here too.
    Behind the diagram, the checks of ``code`` (a code on positions of the topology),
    drawn as by :meth:`~zxqec.codes.StabilizerCode.draw_encoder`, in ``face_colors``
    (``{'X': ..., 'Z': ...}``) at opacity ``face_alpha``. ``label_size``, ``y_down``,
    ``wire_ends`` and the size of the figure (0.3 inch per drawn unit without
    ``figsize``) as there. Returns the figure.
    """
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path
    from pyzx.drawing import draw_matplotlib
    from pyzx.graph import Graph
    from pyzx.simplify import id_simp
    from pyzx.utils import get_mode, settings

    from ..codes.code import _draw_check_faces
    from .subdiagram import get_stage

    topology = program.topology
    if topology is None:
        raise ValueError("the program has no topology")
    g = program.quantum_graph()
    pos = _positions(g, topology)
    flip = -spacing if y_down else spacing              # places are in the plot's (x, y): PyZX draws at (row, -qubit)

    def at(v: Any, dy: float = 0.0) -> tuple[float, float]:
        return (spacing * pos[v][0], flip * pos[v][1] + dy)

    keep = [v for v in g.vertices() if g.type(v) != VertexType.BOUNDARY and get_stage(g, v) == stage
            and (wires is None or int(g.qubit(v) // 1) in wires)]
    if not keep:
        raise ValueError("stage {} has no spiders on these wires".format(stage))
    h = Graph()
    for v in keep:
        for var in getattr(g.phase(v), 'free_vars', lambda: set())():
            h.var_registry.set_type(var.name, var.is_bool)
    new = {v: h.add_vertex(g.type(v), phase=g.phase(v)) for v in keep}
    place = {new[v]: at(v) for v in keep}
    time = {new[v]: pos[v][2] for v in keep}
    for v in keep:
        for w in g.neighbors(v):
            et = g.edge_type(g.edge(v, w))
            if w in new:
                if v < w:                               # each edge once
                    h.add_edge((new[v], new[w]), et)
                continue
            b = h.add_vertex(VertexType.BOUNDARY)
            place[b], time[b] = at(w), pos[w][2]
            h.add_edge((new[v], b), et)
    h.rebind_variables_to_registry()
    id_simp(h)
    spots = [place[v] for v in h.vertices()]           # the places, before the stubs move off them
    shift = {}
    for b in [v for v in h.vertices() if h.type(v) == VertexType.BOUNDARY]:
        (n,) = list(h.neighbors(b))
        if abs(place[b][0] - place[n][0]) + abs(place[b][1] - place[n][1]) < 1e-9:   # a stub, not a line
            shift[b] = 0.45 if time[b] < time[n] else -0.45
    for b, dy in shift.items():
        place[b] = (place[b][0], place[b][1] + dy)
    cx = sum(p[0] for p in place.values()) / len(place)
    cy = sum(p[1] for p in place.values()) / len(place)
    moved = []                                          # wires a permutation moves, drawn below with their stubs
    for e in list(h.edges()):
        s, t = h.edge_st(e)
        if h.edge_type(e) != EdgeType.SIMPLE or h.type(s) != VertexType.BOUNDARY or h.type(t) != VertexType.BOUNDARY:
            continue
        if s in shift or t in shift:                    # a wire that stays: already a stub
            continue
        if time[s] > time[t]:
            s, t = t, s                                 # from where it comes to where it goes
        moved.append((place[s], place[t]))
        h.remove_edge(e)
        place[s] = (place[s][0], place[s][1] + 0.45)    # its ends: a stub above its old place, below its new one
        place[t] = (place[t][0], place[t][1] - 0.45)
    for v in h.vertices():
        h.set_row(v, place[v][0])
        h.set_qubit(v, -place[v][1])
    mode = get_mode()
    settings.mode = "notebook"                          # PyZX would call plt.show() itself otherwise
    try:
        fig = draw_matplotlib(h, figsize=figsize or (3.2, 3.2), h_edge_draw='box', show_time=False)
    finally:
        settings.mode = mode
    ax = fig.axes[0]
    for (x0, y0), (x1, y1) in moved:
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        swapped = ((x1, y1), (x0, y0)) in moved         # two wires that trade places: bowed apart
        through = any(min(abs(px - x0) + abs(py - y0), abs(px - x1) + abs(py - y1)) > 1e-6
                      and 0.05 < ((px - x0) * (x1 - x0) + (py - y0) * (y1 - y0)) / length ** 2 < 0.95
                      and abs((px - x0) * (y1 - y0) - (py - y0) * (x1 - x0)) / length < 0.25 * spacing
                      for px, py in spots)              # a straight line would pass through another place
        along = abs(x1 - x0) < 1e-6                     # a straight line would lie on the stubs
        verts, codes = [(x0, y0 + 0.45), (x0, y0)], [Path.MOVETO, Path.LINETO]
        if swapped or through or along:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            nx, ny = -(y1 - y0), x1 - x0                # a normal, as long as the wire
            if not swapped and nx * (mx - cx) + ny * (my - cy) < 0:
                nx, ny = -nx, -ny                       # outwards (a swapped pair: each to its own side)
            verts += [(mx + 0.15 * nx, my + 0.15 * ny), (x1, y1)]
            codes += [Path.CURVE3, Path.CURVE3]
        else:
            verts.append((x1, y1))
            codes.append(Path.LINETO)
        verts.append((x1, y1 - 0.45))
        codes.append(Path.LINETO)
        ax.add_patch(PathPatch(Path(verts, codes), fill=False, edgecolor='black', linewidth=0.8, zorder=0))
    for p in list(ax.patches):                          # PyZX's Hadamard boxes, redrawn 0.28 of a step wide
        if type(p).__name__ == 'Rectangle' and p.get_facecolor()[:3] == (1.0, 1.0, 0.0):
            p.remove()
    side = 0.28 * spacing
    for e in h.edges():
        if h.edge_type(e) == EdgeType.HADAMARD:
            s, t = h.edge_st(e)
            mx, my = (place[s][0] + place[t][0]) / 2, (place[s][1] + place[t][1]) / 2
            ax.add_patch(Rectangle((mx - side / 2, my - side / 2), side, side, facecolor='yellow',
                                   edgecolor='black', linewidth=0.8, zorder=2))
    pts: dict[Any, tuple[float, float]] = {}
    if code is not None:
        d = code.data()
        where = list(d.positions) if d.positions is not None else list(d.qubits)
        coords = {w: topology.coordinates(w) for w in where}
        pts = {w: (spacing * float(c[0]), flip * float(c[1])) for w, c in coords.items()}
        colors = face_colors or {'X': _VERTEX_COLORS[VertexType.X], 'Z': _VERTEX_COLORS[VertexType.Z]}
        _draw_check_faces(ax, d.layout_checks(coords), pts,
                          {kind: dict(facecolor=colors[kind], edgecolor='white', alpha=face_alpha, lw=1.5, zorder=-1)
                           for kind in ('X', 'Z')})
    if label_size is not None:
        _set_labels(ax, label_size, 'right')
    _set_wire_ends(ax, wire_ends)
    _fit_layout(fig, ax, [place[v] for v in h.vertices()], list(pts.values()), 0.55 * spacing, figsize is None, title)
    return _finish(fig)


def _set_wire_ends(ax: Any, kind: str) -> None:
    """``'dot'`` keeps PyZX's dots at the ends of the wires (its boundaries); ``'tick'`` removes them."""
    if kind not in ('dot', 'tick'):
        raise ValueError("wire_ends is 'dot' or 'tick', not {!r}".format(kind))
    if kind == 'tick':
        for p in list(ax.patches):
            if type(p).__name__ == 'Circle' and p.radius == 0.1:
                p.remove()


def _fit_layout(fig: Any, ax: Any, spiders: list[tuple[float, float]], qubits: list[tuple[float, float]],
                faces: float, resize: bool, title: str | None = None) -> None:
    """The window of a diagram drawn on a layout: its ``spiders`` with a margin of 0.6, and the places of the
    ``qubits`` with a margin of ``faces`` (for the faces drawn around them). The axes fill the figure, at equal
    scale, below the ``title`` if there is one; with ``resize`` the drawing gets 0.3 inch per unit, so that such
    pictures share one scale, and the title 0.3 inch above it."""
    x0 = min([x - 0.6 for x, _ in spiders] + [x - faces for x, _ in qubits])
    x1 = max([x + 0.6 for x, _ in spiders] + [x + faces for x, _ in qubits])
    y0 = min([y - 0.6 for _, y in spiders] + [y - faces for _, y in qubits])
    y1 = max([y + 0.6 for _, y in spiders] + [y + faces for _, y in qubits])
    top = 1.0
    if resize:
        height = 0.3 * (y1 - y0)
        fig.set_size_inches(0.3 * (x1 - x0), height + (0.3 if title else 0.0))
        top = height / (height + 0.3) if title else 1.0
    elif title:
        top = 0.88
    ax.set_position([0, 0, 1, top])
    ax.set_aspect('equal', adjustable='box')            # PyZX's axes adjust their limits instead
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# 2+1D: the program as a level rewrites it
# ---------------------------------------------------------------------------

_POSITION = '_zxqec_position_3d'          # vertex data carrying (x, y, t) through the gluing


def level_layout_3d(program: 'QProgram', level: int, by_outcomes: bool = False, crossing_wires: bool = False,
                    min_stub: float = 0.0) -> tuple[BaseGraph, dict[Any, tuple[float, float, float]],
                                                    tuple[Any, ...]]:
    """The program as ``level`` rewrites it (:meth:`QProgram.reduced_level`), with ``(x, y, t)`` for every vertex.

    A rewritten diagram has no place in space-time, so the layout keeps what
    has one. A stage without a rewrite (and a block without one, made of such
    pieces) is where it is in the program: its spiders on their wires at their
    timesteps, as :func:`draw_program_3d` draws them. In a block with a rewrite
    only the ends of its wires are fixed, each where the wire enters or leaves
    the block (the wire's position, one tick before the block's first timestep
    or after its last). Every spider inside is laid out by its distances in the
    diagram to those ends: it sits over the wire ends nearest to it (their
    average position), at a time that divides the block in the ratio of its
    distances to the nearest input and the nearest output. In a normal form, the
    Z-spiders next to the inputs sit low over their wires, the X-spiders next to
    the outputs high over theirs, and the edges between them cross the block.
    Spiders that would coincide are moved apart in time by a quarter tick. These
    positions are a layout, not a claim about where or when anything happens.

    Returns the glued diagram (the same map as ``reduced_level(level).graph``),
    the positions, and every detector the level's blocks removed.

    ``by_outcomes=True`` puts a spider joined to no wire end whose phase is a
    single measurement outcome (a check after the rewrite ``'gpf_checks'``) where
    that outcome is measured in the block: on the measured qubit, at the time
    of the measurement, as the program draws it. ``crossing_wires=True`` draws a
    permutation stage as its wires going straight from their old places to their
    new ones, without the identity spider each carries in the program.
    ``min_stub`` keeps every spider of a rewritten block at least that fraction
    of the block's time span from its input and output ends, so each wire in
    and out shows as a stub of that length at least.
    """
    reduced = [program.reduced_block(level, k) for k in range(program.levels.level_size(level))]
    g = _glued([_positioned(program, rb, by_outcomes, crossing_wires, min_stub) for rb in reduced], program)
    return g, _take_positions(g), tuple(d for rb in reduced for d in rb.detectors)


def block_layout_3d(program: 'QProgram', level: int, index: int, by_outcomes: bool = False,
                    crossing_wires: bool = False, min_stub: float = 0.0) -> tuple[BaseGraph, dict[Any, tuple[float, float, float]], tuple[Any, ...]]:
    """Block ``index`` of ``level`` after its rewrite, laid out as in :func:`level_layout_3d`.

    The block (:meth:`QProgram.reduced_block`) is where it is in the picture of
    :func:`level_layout_3d`: without a rewrite its spiders on their wires at their
    timesteps, with one its spiders between the ends of its wires. Returns the
    block's diagram, the positions, and the detectors its rewrites removed.
    ``by_outcomes``, ``crossing_wires`` and ``min_stub`` are as there.
    """
    rb = program.reduced_block(level, index)
    g = _positioned(program, rb, by_outcomes, crossing_wires, min_stub)
    return g, _take_positions(g), tuple(rb.detectors)


def draw_block_3d(program: 'QProgram', level: int, index: int, by_outcomes: bool = False,
                  crossing_wires: bool = False, min_stub: float = 0.0, **kwargs: Any) -> Any:
    """The 2+1D view of block ``index`` of ``level`` after its rewrite, laid out by :func:`block_layout_3d`.

    The outcomes in the phases are written modulo the detectors the block's
    rewrites removed, as in :func:`draw_level_3d`. ``by_outcomes``,
    ``crossing_wires`` and ``min_stub`` are as in :func:`level_layout_3d`; other
    keyword arguments go to :func:`draw_graph_3d`. Returns the figure.
    """
    from ..gpf import phases_modulo

    g, pos, detectors = block_layout_3d(program, level, index, by_outcomes, crossing_wires, min_stub)
    return draw_graph_3d(phases_modulo(g, detectors), program.topology, positions=pos, **kwargs)


def draw_blocks_3d(program: 'QProgram', level: int, *, view: str | tuple[float, float] = 'top',
                   by_outcomes: bool = False, crossing_wires: bool = False, min_stub: float = 0.0,
                   show_topology: bool = False, bare: bool = True, time_scale: float = 1.0, columns: int | None = None,
                   panel_size: tuple[float, float] = (3.2, 3.4), finish: bool = True, **kwargs: Any) -> Any:
    """Every block of ``level`` after its rewrite side by side, each as :func:`draw_block_3d` draws it.

    One picture per block, in order, all from ``view`` (from above by default)
    and at the same scale, with the block's name and rewrite under it; an
    orthographic projection, so a grid stays a grid. By default bare (no panes,
    grid or axes) and without the hardware graph. With ``phases_beside`` each
    phase is then moved to the side of its spider where it covers the fewest
    other labels, spiders and edges. ``time_scale`` is the height
    of a block's time span against the width of the chip: small values flatten
    each block, so a view a little off the top shows the grid with short wires
    in and out. Every panel spans the same time range, so a time step is as tall
    in one as in another. ``columns`` puts that many blocks in a row and the rest in
    rows below (by default all in one row); a figure that fits the screen is
    not shrunk, which would leave the labels and markers at their size around a
    smaller diagram. Resizing the window places the labels again.
    ``by_outcomes``, ``crossing_wires`` and ``min_stub`` are as in
    :func:`level_layout_3d`; other keyword arguments go to :func:`draw_graph_3d`
    (``phases_beside``, ``hadamard_boxes``, ``show_phases``, ...). Returns the figure.
    """
    import matplotlib.pyplot as plt

    n = program.levels.level_size(level)
    layouts = [block_layout_3d(program, level, k, by_outcomes, crossing_wires, min_stub)[1] for k in range(n)]
    every = [p for pos in layouts for p in pos.values()]
    spans = [(min(p[2] for p in pos.values()), max(p[2] for p in pos.values())) for pos in layouts]
    height = max(hi - lo for lo, hi in spans) + 1     # one time range for every panel, half a step to spare
    (x0, x1), (y0, y1) = _space_extent({i: p for i, p in enumerate(every)})
    x1 += 1.8                                        # room for phases written to the right of a spider
    cols = n if columns is None else max(1, min(columns, n))
    rows = -(-n // cols)
    fig = plt.figure(figsize=(panel_size[0] * cols, panel_size[1] * rows))
    for k in range(n):
        ax = fig.add_subplot(rows, cols, k + 1, projection='3d')
        ax.set_proj_type('ortho')
        draw_block_3d(program, level, k, by_outcomes, crossing_wires, min_stub, ax=ax, view=view,
                      show_topology=show_topology,
                      bare=bare, finish=False, **kwargs)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        middle = sum(spans[k]) / 2
        ax.set_zlim(middle - height / 2, middle + height / 2)
        ax.set_box_aspect((x1 - x0, y1 - y0, time_scale * (x1 - x0 + y1 - y0) / 2))   # one scale in x and y
        block = program.block(level, k)
        ax.set_title("{}\n{}".format(block, block.rewrite or "no rewrite"), fontsize=9, y=-0.08)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.08 / rows, wspace=0.02, hspace=0.12)

    def place(_event: Any = None) -> None:           # once every panel has its size: labels where they fit
        for ax in fig.axes:
            _place_labels(ax)
        if _event is not None:
            fig.canvas.draw_idle()

    place()
    fig.canvas.mpl_connect('resize_event', place)
    return _finish(fig) if finish else fig


def save_panels(fig: Any, pattern: str, pad: float = 0.04, dpi: int = 200, formats: Sequence[str] = ('pdf',),
                titles: bool = False) -> list[tuple[float, float]]:
    """Save each panel (axes) of ``fig`` as a file of its own, cropped to what that panel draws.

    ``pattern`` names the files, with ``{k}`` for the panel's index and ``{ext}``
    for the format, e.g. ``'figures/block-{k}.{ext}'``; one file per panel and
    format. A panel's box is found from its ink: the figure is rasterized at
    ``dpi`` with only that panel visible, and the box is the non-white pixels,
    ``pad`` inches to spare (3D labels lie outside the axes' own box). Panel
    titles are left out unless ``titles``. Meant for the panels of
    :func:`draw_blocks_3d`, to place them one by one in a document. Returns each
    panel's (width, height) in inches; the figure is left as it was.
    """
    import io

    import numpy as np
    from matplotlib.transforms import Bbox

    axes = list(fig.axes)
    visible = [ax.get_visible() for ax in axes]
    title_shown = [ax.title.get_visible() for ax in axes]
    if not titles:
        for ax in axes:
            ax.title.set_visible(False)
    width, height = (fig.get_size_inches() * dpi).astype(int)
    sizes = []
    try:
        for k, ax in enumerate(axes):
            for other in axes:
                other.set_visible(other is ax)
            buf = io.BytesIO()
            fig.savefig(buf, format='rgba', dpi=dpi)
            pixels = np.frombuffer(buf.getvalue(), np.uint8).reshape(height, width, 4)
            rows, cols = np.nonzero((pixels[:, :, :3] < 245).any(axis=2))
            if not len(rows):
                raise ValueError("panel {} draws nothing".format(k))
            box = Bbox([[cols.min() / dpi - pad, (height - rows.max()) / dpi - pad],
                        [cols.max() / dpi + pad, (height - rows.min()) / dpi + pad]])
            for ext in formats:
                fig.savefig(pattern.format(k=k, ext=ext), bbox_inches=box, dpi=dpi, facecolor='white')
            sizes.append((box.width, box.height))
    finally:
        for ax, shown, title in zip(axes, visible, title_shown):
            ax.set_visible(shown)
            ax.title.set_visible(title)
    return sizes


def _take_positions(g: BaseGraph) -> dict[Any, tuple[float, float, float]]:
    """Every vertex's ``(x, y, t)``, taken out of its vertex data."""
    pos: dict[Any, tuple[float, float, float]] = {}
    for v in g.vertices():
        data = dict(g.vdata_dict(v))
        pos[v] = data.pop(_POSITION)
        g.set_vdata_dict(v, data)
    return pos


def draw_level_3d(program: 'QProgram', level: int, by_outcomes: bool = False, crossing_wires: bool = False,
                  min_stub: float = 0.0, **kwargs: Any) -> Any:
    """The 2+1D view of the program as ``level`` rewrites it, laid out by :func:`level_layout_3d`.

    The blocks of the level are drawn as slabs over the ticks they replace, and
    the outcomes in the phases are written modulo the detectors the rewrites
    removed (:func:`~zxqec.gpf.phases_modulo`). ``by_outcomes``, ``crossing_wires``
    and ``min_stub`` are as in :func:`level_layout_3d`; other keyword arguments
    go to :func:`draw_graph_3d`. Returns the figure.
    """
    from ..gpf import phases_modulo

    g, pos, detectors = level_layout_3d(program, level, by_outcomes, crossing_wires, min_stub)
    return draw_graph_3d(phases_modulo(g, detectors), program.topology, positions=pos,
                         blocks=_block_tick_ranges(program, level), **kwargs)


def _glued(parts: list[BaseGraph], program: 'QProgram') -> BaseGraph:
    g = parts[0]
    for h in parts[1:]:
        program._glue(g, h)                   # copies the vertex data, positions included
    return g


def _positioned(program: 'QProgram', rb: Any, by_outcomes: bool = False, crossing_wires: bool = False,
                min_stub: float = 0.0) -> BaseGraph:
    """A copy of a reduced block's diagram with every vertex's ``(x, y, t)`` in its vertex data."""
    if rb.block.rewrite is None and rb.children:
        return _glued([_positioned(program, c, by_outcomes, crossing_wires, min_stub) for c in rb.children], program)
    g = rb.graph.copy()
    topology = program.topology
    if rb.block.rewrite is None:              # a stage: where it is
        pos = _positions(g, topology)
        if crossing_wires and all(program.stage(s).kind == 'permutation' for s in rb.block.stages):
            for v in [v for v in g.vertices() if g.type(v) == VertexType.Z and g.phase(v) == 0
                      and g.vertex_degree(v) == 2
                      and all(g.edge_type(e) == EdgeType.SIMPLE for e in g.incident_edges(v))]:
                a, b = g.neighbors(v)         # the identity spider on a moved wire: the wire alone
                g.add_edge((a, b))
                g.remove_vertex(v)
                del pos[v]
    else:
        pos = _between_wire_ends(g, *_wire_ends(program, rb.block, g), min_stub)
        if by_outcomes:
            _over_outcomes(g, pos, _outcome_places(program, rb.block))
    for v, p in pos.items():
        g.set_vdata(v, _POSITION, p)
    return g


def _wire_ends(program: 'QProgram', block: Any, g: BaseGraph) -> tuple[dict[Any, tuple[float, float, float]],
                                                                         tuple[float, float]]:
    """Where each boundary of ``g`` (a rewritten block) is: the same wire end of the block's cut diagram; and
    the times of the block's inputs and outputs (one tick before its first timestep, one after its last).

    A wire end is over the spider it is joined to in the block: a cut next to a
    permutation sits on the far side of it, at the wire's place after the
    permutation, not where the wire is inside the block.
    """
    from .subdiagram import get_logical_qubit

    cut = program._without_classical(program._subdiagram(set(block.stages)))
    cpos = _positions(cut, program.topology)
    boundaries = set(cut.inputs()) | set(cut.outputs())
    for b in boundaries:
        inside = [u for u in cut.neighbors(b) if u not in boundaries]
        if inside:
            cpos[b] = (cpos[inside[0]][0], cpos[inside[0]][1], cpos[b][2])
    lo, hi = time_extent(cut) or (0, 0)
    ends = {('in', get_logical_qubit(cut, b)): cpos[b] for b in cut.inputs()}
    ends.update({('out', get_logical_qubit(cut, b)): cpos[b] for b in cut.outputs()})
    fixed = {}
    for side, boundaries in (('in', g.inputs()), ('out', g.outputs())):
        for b in boundaries:
            key = (side, get_logical_qubit(g, b))
            if key in ends:
                fixed[b] = ends[key]
    return fixed, (float(lo - 1), float(hi + 1))


def _outcome_places(program: 'QProgram', block: Any) -> dict[str, tuple[float, float, float]]:
    """The ``(x, y, t)`` of each measurement outcome of ``block``: where its measurement is in the block's cut
    diagram (not a correction controlled by it, which has the outcome in its phase too)."""
    cut = program._without_classical(program._subdiagram(set(block.stages)))
    cpos = _positions(cut, program.topology)
    return {cut.vdata(v, 'result_symbol'): cpos[v] for v in cut.vertices()
            if cut.vdata(v, 'outcome_type', None) == 'measurement' and cut.vdata(v, 'result_symbol', None)}


def _outcome_names(phase: Any) -> list[str]:
    free_vars = getattr(phase, 'free_vars', None)
    return sorted(str(x) for x in free_vars()) if free_vars is not None else []


def _over_outcomes(g: BaseGraph, pos: dict[Any, tuple[float, float, float]],
                   places: dict[str, tuple[float, float, float]]) -> None:
    """Move each spider joined to no boundary, with one outcome as its phase, to where that outcome is measured."""
    ends = set(g.inputs()) | set(g.outputs())
    for v in g.vertices():
        if v in ends or any(u in ends for u in g.neighbors(v)):
            continue
        names = _outcome_names(g.phase(v))
        if len(names) == 1 and names[0] in places:
            pos[v] = places[names[0]]


def _between_wire_ends(g: BaseGraph, fixed: dict[Any, tuple[float, float, float]],
                       times: tuple[float, float], min_stub: float = 0.0) -> dict[Any, tuple[float, float, float]]:
    """Every vertex not in ``fixed`` over the wire ends nearest to it, at a time set by its distances to the ends.

    ``fixed`` are the block's boundaries (inputs at ``times[0]``, outputs at
    ``times[1]``). A spider's ``(x, y)`` is the average of the positions of the
    boundaries at the least number of edges from it; its time divides the block
    in the ratio of its distances to the nearest input and the nearest output,
    kept at least ``min_stub`` of the block's span from either end. Spiders that
    would coincide are moved apart in time by a quarter tick.
    """
    from collections import deque

    def distances(sources: list[Any]) -> dict[Any, tuple[int, set[Any]]]:
        """Least number of edges from ``sources`` to every vertex, and which sources are that near."""
        out: dict[Any, tuple[int, set[Any]]] = {b: (0, {b}) for b in sources}
        queue = deque(sources)
        while queue:
            v = queue.popleft()
            d, near = out[v]
            for u in g.neighbors(v):
                if u not in out:
                    out[u] = (d + 1, set(near))
                    queue.append(u)
                elif out[u][0] == d + 1:
                    out[u][1].update(near)
        return out

    ins = [b for b in g.inputs() if b in fixed]
    outs = [b for b in g.outputs() if b in fixed]
    from_in, from_out = distances(ins), distances(outs)
    t0, t1 = times
    centre = tuple(sum(p[i] for p in fixed.values()) / len(fixed) for i in range(3)) if fixed else (0.0, 0.0, 0.0)
    pos = dict(fixed)
    free = [v for v in g.vertices() if v not in fixed]
    for v in free:
        din, near_in = from_in.get(v, (None, set()))
        dout, near_out = from_out.get(v, (None, set()))
        if din is None and dout is None:            # not connected to the block's wires
            pos[v] = (centre[0], centre[1], (t0 + t1) / 2)
            continue
        least = min(d for d in (din, dout) if d is not None)
        near = (near_in if din == least else set()) | (near_out if dout == least else set())
        x = sum(fixed[b][0] for b in near) / len(near)
        y = sum(fixed[b][1] for b in near) / len(near)
        if din is not None and dout is not None:
            frac = din / (din + dout)
        elif din is not None:                         # no output reachable: towards the end, never at it
            frac = din / (din + 1)
        else:
            frac = 1 / (dout + 1)                     # type: ignore[operator]
        frac = min(max(frac, min_stub), 1 - min_stub)
        pos[v] = (x, y, t0 + (t1 - t0) * frac)
    groups: dict[tuple[float, float, float], list[Any]] = {}
    for v in sorted(pos, key=lambda v: (v in fixed, v)):
        groups.setdefault(tuple(round(c, 2) for c in pos[v]), []).append(v)
    for members in groups.values():
        movable = [v for v in members if v not in fixed]
        shift = 0 if len(movable) == len(members) else 1
        for k, v in enumerate(movable):
            x, y, t = pos[v]
            pos[v] = (x, y, t + 0.25 * (k + shift - (len(movable) - 1 + shift) / 2))
    return pos
    centre = np.mean(np.array(list(fixed.values()), dtype=float), axis=0) if fixed else np.zeros(3)
    index = {v: i for i, v in enumerate(free)}
    pull = 1e-3
    a = np.eye(len(free)) * pull
    b = np.tile(centre * pull, (len(free), 1))
    for v in free:
        i = index[v]
        for u in g.neighbors(v):
            a[i, i] += 1
            if u in index:
                a[i, index[u]] -= 1
            else:
                b[i] += fixed[u]
    solved = np.linalg.solve(a, b)
    for v in free:
        pos[v] = tuple(float(c) for c in solved[index[v]])
    groups: dict[tuple[float, float, float], list[Any]] = {}
    for v in sorted(pos, key=lambda v: (v in fixed, v)):
        groups.setdefault(tuple(round(c, 2) for c in pos[v]), []).append(v)
    for members in groups.values():
        movable = [v for v in members if v not in fixed]
        shift = 0 if len(movable) == len(members) else 1
        for k, v in enumerate(movable):
            x, y, t = pos[v]
            pos[v] = (x, y, t + 0.25 * (k + shift - (len(movable) - 1 + shift) / 2))
    return pos
