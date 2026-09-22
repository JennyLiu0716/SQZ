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
  last time layer, and the blocks of a level as translucent slabs.

Ported from the thesis ``draw_topology_3d`` without its button widgets. Every
function returns the figure: in a notebook it is displayed inline; in a script
call ``matplotlib.pyplot.show()`` to open it in a window, where the 3D views
can be rotated and zoomed with the mouse.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.spacetime import get_timestep, time_extent
from pyzx.utils import EdgeType, VertexType, phase_to_s

from .topology import Topology

if TYPE_CHECKING:
    from ..qprogram.program import QProgram

__all__ = ['draw_program', 'draw_program_3d', 'draw_graph_3d', 'draw_program_views',
           'block_colors', 'VIEWS', 'time_positions']

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


def draw_program(program: 'QProgram', level: int | None = None, *,
                 time_axis: bool = True, show_time: bool = True, alpha: float = 0.15,
                 classical: bool = False, **kwargs: Any) -> Any:
    """Draw the master graph of ``program`` in 2D with matplotlib; returns the figure.

    With ``time_axis`` (default) spiders are placed at their timestep along the
    horizontal axis rather than at PyZX's row index, so the length of a wire is
    the time it spans. ``show_time`` adds PyZX's timestep labels and squiggly
    delays. If ``level`` is given, every block of that level is shaded over the
    ticks it occupies and labelled. ``classical=True`` also draws the
    classical-bit wires. Remaining keyword arguments go to
    :func:`pyzx.draw_matplotlib` (e.g. ``figsize``, ``labels``).
    """
    from pyzx.drawing import draw_matplotlib
    from pyzx.utils import get_mode, settings

    g = program.graph() if classical else program.quantum_graph()
    h = _time_layout(g) if time_axis else g
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
                  ax: Any = None) -> Any:
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
    if not pos:
        return _finish(fig)

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
        xs_all = [p[0] for p in pos.values()]
        ys_all = [p[1] for p in pos.values()]
        x0, x1 = min(xs_all) - 0.3, max(xs_all) + 0.3
        y0, y1 = min(ys_all) - 0.3, max(ys_all) + 0.3
        colors = block_colors(len(blocks))
        for (label, ranges), color in zip(blocks, colors):
            for lo, hi in ranges:
                _slab(ax, (x0, x1), (y0, y1), (lo - 0.5, hi + 0.5), color)
            if ranges:
                ax.text(x0, y0, ranges[0][0] - 0.5, label, fontsize=7, color=color)

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
    return _finish(fig)


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
          zr: tuple[float, float], color: str) -> None:
    """A translucent slab on a 3D axes: tinted top and bottom faces plus its outline."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    (x0, x1), (y0, y1), (z0, z1) = xr, yr, zr
    bottom = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
    top = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    ax.add_collection3d(Poly3DCollection([bottom, top], facecolors=color, alpha=0.06, linewidths=0))
    for face in (bottom, top):
        xs, ys, zs = zip(*(face + face[:1]))
        ax.plot(xs, ys, zs, color=color, linewidth=0.6, alpha=0.6)
    for (x, y, _), (_, _, _) in zip(bottom, top):
        ax.plot([x, x], [y, y], [z0, z1], color=color, linewidth=0.6, alpha=0.6)


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
