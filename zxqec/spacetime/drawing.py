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
  the spiders of a rewritten block laid out between the ends of its wires.

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

__all__ = ['draw_program', 'draw_graph', 'draw_program_3d', 'draw_graph_3d', 'draw_program_views',
           'slice_vertices', 'draw_slice', 'draw_slices', 'draw_program_slices',
           'draw_level_3d', 'level_layout_3d', 'block_colors', 'VIEWS', 'time_positions']

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
                 classical: bool = False, positions: Any = None, **kwargs: Any) -> Any:
    """Draw the master graph of ``program`` in 2D with matplotlib; returns the figure.

    With ``time_axis`` (default) spiders are placed at their timestep along the
    horizontal axis rather than at PyZX's row index, so the length of a wire is
    the time it spans. ``show_time`` adds PyZX's timestep labels and squiggly
    delays. If ``level`` is given, every block of that level is shaded over the
    ticks it occupies and labelled. ``classical=True`` also draws the
    classical-bit wires. ``positions`` maps vertices of :func:`program_layout`
    to ``(row, qubit)`` where they are drawn instead (spiders moved by hand,
    e.g. in :func:`~zxqec.spacetime.levels_view.view_levels`). Remaining keyword
    arguments go to :func:`pyzx.draw_matplotlib` (e.g. ``figsize``, ``labels``).
    """
    from pyzx.drawing import draw_matplotlib
    from pyzx.utils import get_mode, settings

    g = program.graph() if classical else program.quantum_graph()
    h = program_layout(program, time_axis, classical)
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
    if level is not None:
        colors = block_colors(program.levels.level_size(level))
        ymin, ymax = ax.get_ylim()
        for (label, ranges), color in zip(_block_tick_ranges(program, level), colors):
            for lo, hi in ranges:
                x0, x1 = (lo - 0.5, hi + 0.5) if time_axis else _row_span(g, program, lo, hi)
                ax.axvspan(x0, x1, facecolor=color, alpha=alpha, zorder=-1)
            if ranges:
                lo = ranges[0][0]
                x0 = lo - 0.5 if time_axis else _row_span(g, program, lo, lo)[0]
                ax.text(x0 + 0.1, ymax - 0.15, label, fontsize=7, color=color, va='top')
    return _finish(fig)


def draw_graph(g: BaseGraph[VT, ET], *, time_axis: bool = True, show_time: bool = True, **kwargs: Any) -> Any:
    """PyZX's 2D drawing (``draw_matplotlib``) of a time-annotated diagram, time along the horizontal axis.

    The static counterpart of :func:`~zxqec.spacetime.draw_web`, for any diagram
    (a stage or block cut out of a program, a reduced diagram): every spider at its
    timestep (see :func:`time_positions`; ``time_axis=False`` keeps PyZX's rows), and
    with ``show_time`` PyZX's timestep labels and squiggly delays. Keyword arguments
    go to ``draw_matplotlib`` (e.g. ``figsize``, ``labels``). Returns the figure.
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
    return _finish(fig)


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
                  positions: dict[Any, tuple[float, float, float]] | None = None) -> Any:
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
    does for the spiders of a rewritten block).
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

    # ZX edges
    for e in g.edges():
        s, t_ = g.edge_st(e)
        (xs, ys, zs), (xt, yt, zt) = pos[s], pos[t_]
        color = '#0099ff' if g.edge_type(e) == EdgeType.HADAMARD else 'black'
        ax.plot([xs, xt], [ys, yt], [zs, zt], color=color, linewidth=1.0, alpha=0.9)

    # spiders
    for v, (x, y, z) in pos.items():
        ty = g.type(v)
        size = spider_size * (0.3 if ty == VertexType.BOUNDARY else 1.0)
        ax.scatter([x], [y], [z], c=_VERTEX_COLORS.get(ty, 'black'), s=size,
                   edgecolors='black', linewidths=0.5, depthshade=False)
        if show_phases and ty != VertexType.BOUNDARY:
            ps = phase_to_s(g.phase(v), ty)
            if ps:
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
    if buttons:
        _add_view_buttons(fig, ax)
    return _finish(fig) if finish else fig


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


# ---------------------------------------------------------------------------
# 2+1D: the program as a level rewrites it
# ---------------------------------------------------------------------------

_POSITION = '_zxqec_position_3d'          # vertex data carrying (x, y, t) through the gluing


def level_layout_3d(program: 'QProgram', level: int) -> tuple[BaseGraph, dict[Any, tuple[float, float, float]],
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
    """
    reduced = [program.reduced_block(level, k) for k in range(program.levels.level_size(level))]
    g = _glued([_positioned(program, rb) for rb in reduced], program)
    pos: dict[Any, tuple[float, float, float]] = {}
    for v in g.vertices():
        data = dict(g.vdata_dict(v))
        pos[v] = data.pop(_POSITION)
        g.set_vdata_dict(v, data)
    return g, pos, tuple(d for rb in reduced for d in rb.detectors)


def draw_level_3d(program: 'QProgram', level: int, **kwargs: Any) -> Any:
    """The 2+1D view of the program as ``level`` rewrites it, laid out by :func:`level_layout_3d`.

    The blocks of the level are drawn as slabs over the ticks they replace, and
    the outcomes in the phases are written modulo the detectors the rewrites
    removed (:func:`~zxqec.gpf.phases_modulo`). Keyword arguments go to
    :func:`draw_graph_3d`. Returns the figure.
    """
    from ..gpf import phases_modulo

    g, pos, detectors = level_layout_3d(program, level)
    return draw_graph_3d(phases_modulo(g, detectors), program.topology, positions=pos,
                         blocks=_block_tick_ranges(program, level), **kwargs)


def _glued(parts: list[BaseGraph], program: 'QProgram') -> BaseGraph:
    g = parts[0]
    for h in parts[1:]:
        program._glue(g, h)                   # copies the vertex data, positions included
    return g


def _positioned(program: 'QProgram', rb: Any) -> BaseGraph:
    """A copy of a reduced block's diagram with every vertex's ``(x, y, t)`` in its vertex data."""
    if rb.block.rewrite is None and rb.children:
        return _glued([_positioned(program, c) for c in rb.children], program)
    g = rb.graph.copy()
    topology = program.topology
    if rb.block.rewrite is None:              # a stage: where it is
        pos = _positions(g, topology)
    else:
        pos = _between_wire_ends(g, *_wire_ends(program, rb.block, g))
    for v, p in pos.items():
        g.set_vdata(v, _POSITION, p)
    return g


def _wire_ends(program: 'QProgram', block: Any, g: BaseGraph) -> tuple[dict[Any, tuple[float, float, float]],
                                                                         tuple[float, float]]:
    """Where each boundary of ``g`` (a rewritten block) is: the same wire end of the block's cut diagram; and
    the times of the block's inputs and outputs (one tick before its first timestep, one after its last)."""
    from .subdiagram import get_logical_qubit

    cut = program._without_classical(program._subdiagram(set(block.stages)))
    cpos = _positions(cut, program.topology)
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


def _between_wire_ends(g: BaseGraph, fixed: dict[Any, tuple[float, float, float]],
                       times: tuple[float, float]) -> dict[Any, tuple[float, float, float]]:
    """Every vertex not in ``fixed`` over the wire ends nearest to it, at a time set by its distances to the ends.

    ``fixed`` are the block's boundaries (inputs at ``times[0]``, outputs at
    ``times[1]``). A spider's ``(x, y)`` is the average of the positions of the
    boundaries at the least number of edges from it; its time divides the block
    in the ratio of its distances to the nearest input and the nearest output.
    Spiders that would coincide are moved apart in time by a quarter tick.
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
