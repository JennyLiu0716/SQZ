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

"""A window for walking through the levels of a program, block by block.

:func:`view_levels` opens a matplotlib figure with

* on the left, the program as :func:`~zxqec.spacetime.drawing.draw_program` draws
  it, with the blocks of the selected level shaded, the selected block outlined
  and the blocks it was made from (one level down) outlined with dashes;
* in the middle, the whole program as the selected level rewrites it
  (:meth:`QProgram.reduced_level`): every block of the level replaced by its
  rewritten diagram and the pieces glued in order, as :func:`pyzx.draw_matplotlib`
  draws it, with each block's part shaded in its colour and the selected block
  outlined;
* on the right, the selected block's diagram as :func:`pyzx.draw_matplotlib` draws
  it: after the rewrites of its level-ups (:meth:`QProgram.reduced_block`), with
  the outcomes in its phases written modulo the detectors removed on the way
  (:func:`~zxqec.gpf.phases_modulo`, in the earliest outcomes possible, as the
  examples draw a normal form), or, with *plain diagram* ticked, as cut out of the
  master graph;
* below, what the block is: its name, role and rewrite, its stages, the blocks it
  was made from, its spider counts before and after the rewrite, and the
  detectors the rewrite removed;
* a slider for the level, buttons for the previous and next block, the *plain
  diagram* switch, *spread out block* (PyZX's force-directed layout,
  ``auto_layout_vertex_locs``, for a crowded block picture) and *reset
  positions*. The arrow keys move too (left/right: block, up/down: level).

**Spiders can be dragged** in all three pictures, with the mouse: press on a
spider, move, release. The spider is drawn at its new place together with its
labels (phase, measurement result, time), so labels that overlap can be pulled
apart. Moves are kept while the window is open (across levels for the program,
per level for the middle picture, per block for the right one) until *reset
positions*.

Moving the slider keeps the focus on the same stages: the new block is the one
of the new level that contains the first stage of the old one, so going down a
level shows what a block was made of, and going up shows what it became part of.

The pictures are the library's own drawings, rendered to images and shown in
the window; a moved spider changes the coordinates the drawing is made from
(PyZX draws every spider at its row and qubit), and the picture is drawn again.
From a terminal, ``plt.show()`` opens the window; in a notebook the figure is
shown as it starts (use an interactive matplotlib backend to move or drag).

:func:`view_levels_3d` is the same window with the program in 3D, as
:func:`~zxqec.spacetime.draw_program_3d` draws it, the blocks of a level as
slabs; it is turned with the mouse or the view buttons, and a block cut out of
the program is drawn in 3D on the right too. The middle picture is the program
as the level rewrites it in 3D (:func:`~zxqec.spacetime.draw_level_3d`): what
has a place in space-time is drawn there, and the spiders of a rewritten block,
which have none, are laid out between the ends of its wires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..qprogram import QProgram

__all__ = ['view_levels', 'LevelsView', 'view_levels_3d', 'LevelsView3D']

_DPI = 140                     # resolution of the rendered pictures (higher: sharper when zoomed)


class _Picture:
    """A rendered PyZX drawing: the image, the map between drawing and image coordinates, where each spider is."""

    def __init__(self, fig: Any, positions: dict[Any, tuple[float, float]]) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig.dpi = _DPI
        canvas = FigureCanvasAgg(fig)              # PyZX may hand back a closed figure
        canvas.draw()
        self.image = np.asarray(canvas.buffer_rgba()).copy()
        self.height = self.image.shape[0]
        ax = fig.axes[0]
        (x0, y0), (x1, y1) = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
        self.ax_, self.sx = x0, x1 - x0            # display x = ax_ + sx * x  (equal axes: linear)
        self.ay, self.sy = y0, y1 - y0
        bbox = ax.get_window_extent()
        self.rows = (self.height - bbox.y1, self.height - bbox.y0)
        self.positions = positions                 # vertex -> (x, y) in drawing coordinates
        plt.close(fig)

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        return self.ax_ + self.sx * x, self.height - (self.ay + self.sy * y)

    def from_px(self, px: float, py: float) -> tuple[float, float]:
        return (px - self.ax_) / self.sx, ((self.height - py) - self.ay) / self.sy

    def column(self, t: float) -> float:
        return self.to_px(t, 0.0)[0]

    def nearest(self, px: float, py: float) -> Any:
        """The spider drawn under the image point (px, py), or None."""
        radius = max(6.0, 0.3 * abs(self.sx))      # PyZX draws spiders with radius 0.2
        best, dist = None, radius
        for v, (x, y) in self.positions.items():
            qx, qy = self.to_px(x, y)
            d = ((qx - px) ** 2 + (qy - py) ** 2) ** 0.5
            if d < dist:
                best, dist = v, d
        return best


class LevelsView:
    """The state and drawing of :func:`view_levels` (``fig.levels_view``); its methods move it."""

    def __init__(self, program: 'QProgram', level: int, index: int, figsize: tuple[float, float],
                 program_figsize: tuple[float, float], block_figsize: tuple[float, float],
                 level_figsize: tuple[float, float]) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, Slider

        self.program = program
        self.level = level
        self.index = index
        self.plain = False
        self._program_figsize = program_figsize
        self._block_figsize = block_figsize
        self._level_figsize = level_figsize
        self._program_pictures: dict[int, _Picture] = {}
        self._level_pictures: dict[int, _Picture] = {}
        self._block_pictures: dict[tuple[Any, bool], _Picture] = {}
        self.program_moves: dict[Any, tuple[float, float]] = {}            #: vertex -> (row, qubit)
        self.level_moves: dict[int, dict[Any, tuple[float, float]]] = {}
        self.block_moves: dict[tuple[Any, bool], dict[Any, tuple[float, float]]] = {}
        self._layout: Any = None
        self._current: dict[str, _Picture] = {}
        self._drag: tuple[str, Any] | None = None
        self._marker: Any = None

        self.fig = plt.figure(figsize=figsize)
        self._make_axes()
        n = program.levels.num_levels
        self.slider = Slider(self.fig.add_axes([0.07, 0.04, 0.30, 0.03]), 'level', 0, max(n - 1, 1),
                             valinit=level, valstep=1)
        self.button_prev = Button(self.fig.add_axes([0.42, 0.025, 0.08, 0.05]), '< block')
        self.button_next = Button(self.fig.add_axes([0.51, 0.025, 0.08, 0.05]), 'block >')
        self.check = CheckButtons(self.fig.add_axes([0.61, 0.015, 0.13, 0.07]), ['plain diagram'], [False])
        self.button_spread = Button(self.fig.add_axes([0.75, 0.025, 0.11, 0.05]), 'spread out block')
        self.button_reset = Button(self.fig.add_axes([0.87, 0.025, 0.11, 0.05]), 'reset positions')
        self.slider.on_changed(lambda v: self.set_level(int(round(v))))
        self.button_prev.on_clicked(lambda _e: self.set_block(self.index - 1))
        self.button_next.on_clicked(lambda _e: self.set_block(self.index + 1))
        self.check.on_clicked(lambda _l: self.set_plain(self.check.get_status()[0]))
        self.button_spread.on_clicked(lambda _e: self.spread_block())
        self.button_reset.on_clicked(lambda _e: self.reset_positions())
        self.fig.canvas.mpl_connect('key_press_event', self._key)
        self.fig.canvas.mpl_connect('button_press_event', self._press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._motion)
        self.fig.canvas.mpl_connect('button_release_event', self._release)
        self.redraw()

    def _make_axes(self) -> None:
        self.ax_program = self.fig.add_axes([0.005, 0.36, 0.40, 0.58])
        self.ax_level = self.fig.add_axes([0.41, 0.36, 0.33, 0.58])
        self.ax_block = self.fig.add_axes([0.75, 0.36, 0.245, 0.58])
        self.ax_text = self.fig.add_axes([0.02, 0.11, 0.96, 0.23])

    # -- moving between blocks ---------------------------------------------

    def set_level(self, level: int) -> None:
        n = self.program.levels.num_levels
        level = min(max(level, 0), n - 1)
        if level == self.level:
            return
        first = self.program.block(self.level, self.index).first
        self.level = level
        blocks = self.program.levels.blocks(level)
        self.index = next(k for k, b in enumerate(blocks) if first in b.stages)
        if int(round(self.slider.val)) != level:
            self.slider.set_val(level)
        self.redraw()

    def set_block(self, index: int) -> None:
        index = min(max(index, 0), self.program.levels.level_size(self.level) - 1)
        if index != self.index:
            self.index = index
            self.redraw()

    def set_plain(self, plain: bool) -> None:
        if plain != self.plain:
            self.plain = plain
            if self.check.get_status()[0] != plain:      # set from code: tick the box too
                self.check.set_active(0)
            self.redraw()

    def _key(self, event: Any) -> None:
        moves = {'left': lambda: self.set_block(self.index - 1), 'right': lambda: self.set_block(self.index + 1),
                 'up': lambda: self.set_level(self.level + 1), 'down': lambda: self.set_level(self.level - 1)}
        if event.key in moves:
            moves[event.key]()

    # -- dragging spiders -------------------------------------------------------

    def move_spider(self, picture: str, vertex: Any, row: float, qubit: float) -> None:
        """Draw ``vertex`` of the ``'program'``, ``'level'`` or ``'block'`` picture at ``(row, qubit)`` from now on."""
        row, qubit = round(float(row), 3), round(float(qubit), 3)
        if picture == 'program':
            self.program_moves[vertex] = (row, qubit)
            self._program_pictures.clear()               # every level's picture shows the program
        elif picture == 'level':
            self.level_moves.setdefault(self.level, {})[vertex] = (row, qubit)
            self._level_pictures.pop(self.level, None)
        elif picture == 'block':
            key = (self.program.block(self.level, self.index), self.plain)
            self.block_moves.setdefault(key, {})[vertex] = (row, qubit)
            self._block_pictures.pop(key, None)
        else:
            raise ValueError("picture must be 'program', 'level' or 'block'")
        self.redraw()

    def spread_block(self, scale: float = 1.6, seed: int = 0) -> None:
        """Lay the block picture out with PyZX's force-directed layout (``auto_layout_vertex_locs``).

        Its positions become moves of the block, so spiders can still be dragged
        afterwards; ``seed`` makes the (random) start repeatable.
        """
        import random
        from pyzx.drawing import auto_layout_vertex_locs

        key = (self.program.block(self.level, self.index), self.plain)
        g = (self.program.block_graph(self.level, self.index, classical=False) if self.plain
             else self.program.reduced_block(self.level, self.index).graph)
        state = random.getstate()
        random.seed(seed)
        try:
            locs, _w, _h = auto_layout_vertex_locs(g)
        finally:
            random.setstate(state)
        self.block_moves[key] = {v: (round(scale * x, 3), round(scale * y, 3)) for v, (x, y) in locs.items()}
        self._block_pictures.pop(key, None)
        self.redraw()

    def reset_positions(self) -> None:
        """Forget every spider moved by hand."""
        self.program_moves.clear()
        self.level_moves.clear()
        self.block_moves.clear()
        self._program_pictures.clear()
        self._level_pictures.clear()
        self._block_pictures.clear()
        self.redraw()

    def _toolbar_busy(self) -> bool:
        tb = getattr(self.fig.canvas, 'toolbar', None)
        mode = getattr(tb, 'mode', None)
        return mode is not None and str(getattr(mode, 'value', mode)) != ''   # zoom or pan is active

    def _which(self, event: Any) -> str | None:
        if event.inaxes is self.ax_program:
            return 'program'
        if event.inaxes is self.ax_level:
            return 'level'
        if event.inaxes is self.ax_block:
            return 'block'
        return None

    def _press(self, event: Any) -> None:
        which = self._which(event)
        if which is None or event.button != 1 or event.xdata is None or self._toolbar_busy():
            return
        vertex = self._current[which].nearest(event.xdata, event.ydata)
        if vertex is None:
            return
        self._drag = (which, vertex)
        self._marker = event.inaxes.plot([event.xdata], [event.ydata], 'o', ms=14, mfc='none', mec='#ff8800',
                                         mew=2)[0]
        self.fig.canvas.draw_idle()

    def _motion(self, event: Any) -> None:
        if self._drag is None or self._marker is None or event.xdata is None or self._which(event) != self._drag[0]:
            return
        self._marker.set_data([event.xdata], [event.ydata])
        self.fig.canvas.draw_idle()

    def _release(self, event: Any) -> None:
        if self._drag is None:
            return
        which, vertex = self._drag
        self._drag = None
        if self._marker is not None:
            self._marker.remove()
            self._marker = None
        if event.xdata is None or self._which(event) != which:
            self.fig.canvas.draw_idle()                  # released outside the picture: no move
            return
        x, y = self._current[which].from_px(event.xdata, event.ydata)
        self.move_spider(which, vertex, x, -y)           # PyZX draws a spider at (row, -qubit)

    # -- pictures -----------------------------------------------------------

    def _program_picture(self, level: int) -> _Picture:
        if level not in self._program_pictures:
            from .drawing import draw_program, program_layout
            if self._layout is None:
                self._layout = program_layout(self.program)
            h = self._layout
            positions = {v: (h.row(v), -h.qubit(v)) for v in h.vertices()}
            positions.update({v: (r, -q) for v, (r, q) in self.program_moves.items() if v in positions})
            fig = draw_program(self.program, level, figsize=self._program_figsize, positions=self.program_moves)
            self._program_pictures[level] = _Picture(fig, positions)
        return self._program_pictures[level]

    def _block_picture(self) -> _Picture:
        from ..gpf import phases_modulo

        block = self.program.block(self.level, self.index)
        key = (block, self.plain)
        if key not in self._block_pictures:
            if self.plain:
                g = self.program.block_graph(self.level, self.index, classical=False).clone()
            else:                                 # outcomes modulo the detectors, as the examples draw them
                rb = self.program.reduced_block(self.level, self.index)
                g = phases_modulo(rb.graph, rb.detectors)          # a clone, same vertices
            _move(g, self.block_moves.get(key, {}))
            fig = _draw_pyzx(g, self._block_figsize)
            self._block_pictures[key] = _Picture(fig, {v: (g.row(v), -g.qubit(v)) for v in g.vertices()})
        return self._block_pictures[key]

    def _level_picture(self) -> _Picture:
        """The program as the level rewrites it, each block's part shaded and labelled as ``draw_program`` does."""
        from pyzx.utils import VertexType

        from ..gpf import phases_modulo
        from .drawing import block_colors

        level = self.level
        if level not in self._level_pictures:
            rl = self.program.reduced_level(level)
            g = phases_modulo(rl.graph, rl.detectors)              # a clone, same vertices
            _move(g, self.level_moves.get(level, {}))
            fig = _draw_pyzx(g, self._level_figsize)
            ax = fig.axes[0]
            ymax = ax.get_ylim()[1]
            spans: list[tuple[float, float] | None] = []
            for rb, vs, color in zip(rl.blocks, rl.vertices, block_colors(len(rl.blocks))):
                rows = [g.row(v) for v in vs if g.type(v) != VertexType.BOUNDARY]
                if not rows:                       # nothing left of the block but wires
                    spans.append(None)
                    continue
                x0, x1 = min(rows) - 0.5, max(rows) + 0.5
                ax.axvspan(x0, x1, facecolor=color, alpha=0.15, zorder=-1)
                ax.text(x0 + 0.1, ymax - 0.15, str(rb.block), fontsize=7, color=color, va='top')
                spans.append((x0, x1))
            pic = _Picture(fig, {v: (g.row(v), -g.qubit(v)) for v in g.vertices()})
            pic.spans = spans
            self._level_pictures[level] = pic
        return self._level_pictures[level]

    def _ranges(self, block: Any) -> list[tuple[int, int]]:
        self.program.graph()
        return [self.program.stage(s).ticks for s in block.stages if self.program.stage(s).ticks is not None]

    def redraw(self) -> None:
        self._draw_program()
        self._draw_level()
        self._draw_block()
        self.ax_text.clear()
        self.ax_text.set_axis_off()
        self.ax_text.text(0, 1, self.describe(), va='top', ha='left', family='monospace', fontsize=8,
                          transform=self.ax_text.transAxes)
        self.fig.canvas.draw_idle()

    def _draw_program(self) -> None:
        from matplotlib.patches import Rectangle

        p, level, index = self.program, self.level, self.index
        block = p.block(level, index)
        pic = self._current['program'] = self._program_picture(level)
        top, bottom = pic.rows
        self.ax_program.clear()
        self.ax_program.imshow(pic.image)
        self.ax_program.set_axis_off()
        for lo, hi in self._ranges(block):
            x0, x1 = pic.column(lo - 0.5), pic.column(hi + 0.5)
            self.ax_program.add_patch(Rectangle((x0, top), x1 - x0, bottom - top, fill=False, lw=2.2, ec='black'))
        for child in block.children:
            for lo, hi in self._ranges(child):
                x0, x1 = pic.column(lo - 0.5), pic.column(hi + 0.5)
                self.ax_program.add_patch(Rectangle((x0 + 3, top + 6), x1 - x0 - 6, bottom - top - 12,
                                                    fill=False, lw=1.2, ec='black', ls='--'))
        self.ax_program.set_title("level {} of {}: block {} of {} outlined{} (drag a spider to move it)".format(
            level, p.levels.num_levels - 1, index, p.levels.level_size(level),
            ", the blocks it was made from dashed" if block.children else ""), fontsize=9)

    def _draw_level(self) -> None:
        from matplotlib.patches import Rectangle

        pic = self._current['level'] = self._level_picture()
        rl = self.program.reduced_level(self.level)
        ax = self.ax_level
        ax.clear()
        ax.imshow(pic.image)
        ax.set_axis_off()
        span = pic.spans[self.index]
        if span is not None:
            top, bottom = pic.rows
            x0, x1 = pic.column(span[0]), pic.column(span[1])
            ax.add_patch(Rectangle((x0, top), x1 - x0, bottom - top, fill=False, lw=2.2, ec='black'))
        ax.set_title("level {}, every block after its rewrite: {} spiders{}".format(
            self.level, rl.spiders, "\noutcomes modulo its {} detectors".format(len(rl.detectors))
            if rl.detectors else ""), fontsize=9)

    def _draw_block(self) -> None:
        bpic = self._current['block'] = self._block_picture()
        self.ax_block.clear()
        self.ax_block.imshow(bpic.image)
        self.ax_block.set_axis_off()
        self.ax_block.set_title(self._block_title(), fontsize=9)

    def _block_title(self) -> str:
        block = self.program.block(self.level, self.index)
        if self.plain or not block.rewrite:
            what = "cut out of the master graph"
        else:
            n = len(self.program.reduced_block(self.level, self.index).detectors)
            what = "after rewrite " + block.rewrite + ("\noutcomes modulo its {} detectors".format(n) if n else "")
        return "{}: {}".format(block, what)

    def describe(self) -> str:
        p, level, index = self.program, self.level, self.index
        block = p.block(level, index)
        rb = p.reduced_block(level, index)
        lines = ["block {} '{}'{}{}".format(index, block, " ({})".format(block.role) if block.role else "",
                                           "   rewrite: {}".format(block.rewrite) if block.rewrite
                                           else "   no rewrite (grouping only)"),
                 "stages: " + ", ".join("{} {}".format(s, p.stage(s)) for s in block.stages)]
        if block.children:
            lines.append("made from: " + "; ".join("'{}'{}".format(c, " [{}]".format(c.rewrite) if c.rewrite else "")
                                                  for c in block.children))
        else:
            lines.append("a stage of the circuit (level 0)" if level == 0 else "not levelled up further")
        lines.append("spiders: {} glued -> {} after {}; detectors removed here: {}, in all below: {}{}".format(
            rb.spiders_before, rb.spiders_after, "its rewrite" if block.rewrite else "no rewrite",
            len(rb.new_detectors), len(rb.detectors), "; zero in every run" if rb.is_zero else ""))
        if rb.new_detectors:
            shown = ", ".join("{} = 0".format(d) for d in rb.new_detectors[:4])
            lines.append("  e.g. " + shown + (" ..." if len(rb.new_detectors) > 4 else ""))
        return "\n".join(lines)


def _move(g: Any, moves: dict[Any, tuple[float, float]]) -> None:
    for v, (row, qubit) in moves.items():
        if v in g.vertex_set():
            g.set_row(v, row)
            g.set_qubit(v, qubit)


def _draw_pyzx(g: Any, figsize: tuple[float, float]) -> Any:
    """PyZX's ``draw_matplotlib`` of ``g``, returning the figure (it would call ``plt.show()`` itself otherwise)."""
    from pyzx.drawing import draw_matplotlib
    from pyzx.utils import get_mode, settings

    mode = get_mode()
    settings.mode = "notebook"
    try:
        return draw_matplotlib(g, figsize=figsize)
    finally:
        settings.mode = mode


def view_levels(program: 'QProgram', level: int | None = None, index: int = 0, *,
                figsize: tuple[float, float] = (18, 8), program_figsize: tuple[float, float] = (12, 5),
                level_figsize: tuple[float, float] = (10, 5), block_figsize: tuple[float, float] = (6, 5)) -> Any:
    """Open the levels of ``program`` in a window (see the module docstring); returns the figure.

    ``level`` (default: the highest) and ``index`` choose the block shown first.
    The figure's ``levels_view`` attribute is the :class:`LevelsView`, whose
    ``set_level``, ``set_block``, ``set_plain``, ``move_spider`` and
    ``reset_positions`` act on it from code.
    """
    level = _first_level(program, level, index)
    view = LevelsView(program, level, index, figsize, program_figsize, block_figsize, level_figsize)
    view.fig.levels_view = view
    return view.fig


def _first_level(program: 'QProgram', level: int | None, index: int) -> int:
    n = program.levels.num_levels
    level = n - 1 if level is None else level
    if not 0 <= level < n:
        raise IndexError("level {} does not exist ({} levels)".format(level, n))
    if not 0 <= index < program.levels.level_size(level):
        raise IndexError("level {} has {} blocks".format(level, program.levels.level_size(level)))
    return level


class LevelsView3D(LevelsView):
    """The state and drawing of :func:`view_levels_3d` (``fig.levels_view``); its methods move it."""

    def __init__(self, program: 'QProgram', level: int, index: int, figsize: tuple[float, float],
                 block_figsize: tuple[float, float], level_figsize: tuple[float, float],
                 view: str | tuple[float, float], show_phases: bool) -> None:
        self._view = view
        self._show_phases = show_phases
        self._extent: Any = None                 # x and y ranges of the slabs
        self._slabs: list[Any] = []              # the artists of the level's slabs, replaced on every redraw
        self._block_view: tuple[float, float] | None = None
        self._level_drawn: int | None = None     # the level the middle picture shows
        self._level_extent: Any = None
        self._level_slabs: list[Any] = []
        super().__init__(program, level, index, figsize, (0, 0), block_figsize, level_figsize)

    def _make_axes(self) -> None:
        import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)
        from matplotlib.widgets import Button

        from .drawing import VIEWS

        self.ax_program = self.fig.add_axes([0.0, 0.40, 0.40, 0.53], projection='3d')
        self.ax_level = self.fig.add_axes([0.41, 0.40, 0.33, 0.53], projection='3d')
        self.ax_level.shareview(self.ax_program)   # the program and the level turn together
        self.ax_block = self.fig.add_axes([0.75, 0.40, 0.245, 0.53])
        self.ax_block3d = self.fig.add_axes([0.75, 0.40, 0.245, 0.53], projection='3d')
        self.ax_text = self.fig.add_axes([0.02, 0.11, 0.96, 0.23])
        self.view_buttons = []
        for i, name in enumerate(VIEWS):
            button = Button(self.fig.add_axes([0.03 + 0.09 * i, 0.355, 0.08, 0.035]), name)
            button.on_clicked(lambda _e, name=name: self.set_view(name))
            self.view_buttons.append(button)

    def set_view(self, view: str | tuple[float, float]) -> None:
        """Turn the 3D pictures to a named viewpoint (:data:`~zxqec.spacetime.VIEWS`) or an ``(elev, azim)`` pair."""
        from .drawing import VIEWS

        if isinstance(view, str) and view not in VIEWS:
            raise ValueError("unknown view {!r}; choose from {}".format(view, sorted(VIEWS)))
        elev, azim = VIEWS[view] if isinstance(view, str) else view
        self.ax_program.view_init(elev=elev, azim=azim, share=True)   # and the level, which shares its view
        if self.ax_block3d.get_visible():
            self.ax_block3d.view_init(elev=elev, azim=azim)
        self._block_view = (elev, azim)
        self.fig.canvas.draw_idle()

    def _which(self, event: Any) -> str | None:
        # only the 2D block picture has spiders to drag: on a 3D picture the mouse turns it
        return 'block' if event.inaxes is self.ax_block and self.ax_block.get_visible() else None

    def _draw_program(self) -> None:
        from .drawing import _block_slabs, _block_tick_ranges, _positions, _slab, _space_extent, draw_graph_3d

        p, level, index, ax = self.program, self.level, self.index, self.ax_program
        if self._extent is None:                 # the program is drawn once; only the slabs change
            g = p.quantum_graph()
            draw_graph_3d(g, p.topology, ax=ax, view=self._view, show_phases=self._show_phases, finish=False)
            self._extent = _space_extent(_positions(g, p.topology))
        for artist in self._slabs:
            artist.remove()
        (x0, x1), (y0, y1) = self._extent
        block = p.block(level, index)
        self._slabs = _block_slabs(ax, (x0, x1), (y0, y1), _block_tick_ranges(p, level))
        for lo, hi in self._ranges(block):
            self._slabs += _slab(ax, (x0, x1), (y0, y1), (lo - 0.5, hi + 0.5), 'black', face_alpha=0,
                                 linewidth=2.2, line_alpha=1.0)
        for child in block.children:
            for lo, hi in self._ranges(child):
                self._slabs += _slab(ax, (x0 + 0.15, x1 - 0.15), (y0 + 0.15, y1 - 0.15), (lo - 0.4, hi + 0.4),
                                     'black', face_alpha=0, linewidth=1.2, linestyle='--', line_alpha=1.0)
        ax.set_title("level {} of {}: block {} of {} outlined{} (drag to turn it)".format(
            level, p.levels.num_levels - 1, index, p.levels.level_size(level),
            ", the blocks it was made from dashed" if block.children else ""), fontsize=9)

    def _draw_level(self) -> None:
        from pyzx.utils import VertexType

        from ..gpf import phases_modulo
        from .drawing import _block_slabs, _block_tick_ranges, _slab, _space_extent, draw_graph_3d, level_layout_3d

        p, level, ax = self.program, self.level, self.ax_level
        if self._level_drawn != level:           # drawn once per level; only the outline changes between blocks
            g, pos, detectors = level_layout_3d(p, level)
            ax.clear()
            draw_graph_3d(phases_modulo(g, detectors), p.topology, positions=pos, ax=ax,
                          view=(self.ax_program.elev, self.ax_program.azim), show_phases=self._show_phases,
                          finish=False)
            self._level_extent = _space_extent(pos)
            self._level_slabs = _block_slabs(ax, *self._level_extent, _block_tick_ranges(p, level))
            self._level_outline: list[Any] = []
            self._level_drawn = level
            n = sum(1 for v in g.vertices() if g.type(v) != VertexType.BOUNDARY)
            ax.set_title("level {}, every block after its rewrite: {} spiders{}\n(a rewritten block's spiders "
                         "between the ends of its wires)".format(level, n, ", outcomes modulo {} detectors".format(
                             len(detectors)) if detectors else ""), fontsize=9)
        for artist in self._level_outline:
            artist.remove()
        (x0, x1), (y0, y1) = self._level_extent
        self._level_outline = []
        for lo, hi in self._ranges(p.block(level, self.index)):
            self._level_outline += _slab(ax, (x0, x1), (y0, y1), (lo - 0.5, hi + 0.5), 'black', face_alpha=0,
                                         linewidth=2.2, line_alpha=1.0)

    def _draw_block(self) -> None:
        from .drawing import draw_graph_3d

        block = self.program.block(self.level, self.index)
        cut = self.plain or not block.rewrite    # a diagram cut out of the program has a place in space-time
        self.ax_block.set_visible(not cut)
        self.ax_block3d.set_visible(cut)
        if not cut:
            super()._draw_block()                # a rewritten diagram has none: PyZX's drawing, as in 2D
            return
        ax = self.ax_block3d
        if ax.has_data():                        # keep the viewpoint the user turned it to
            self._block_view = (ax.elev, ax.azim)
        ax.clear()
        draw_graph_3d(self.program.block_graph(self.level, self.index, classical=False), self.program.topology,
                      ax=ax, view=self._block_view or self._view, show_phases=self._show_phases, finish=False)
        ax.set_title(self._block_title(), fontsize=9)


def view_levels_3d(program: 'QProgram', level: int | None = None, index: int = 0, *,
                   figsize: tuple[float, float] = (18, 9), level_figsize: tuple[float, float] = (10, 5),
                   block_figsize: tuple[float, float] = (6, 5), view: str | tuple[float, float] = 'default',
                   show_phases: bool = True) -> Any:
    """The levels window of :func:`view_levels` with the program in 3D (2D space, time upwards).

    On the left the program as :func:`~zxqec.spacetime.draw_program_3d` draws it,
    with the blocks of the selected level as slabs, the selected block outlined
    in bold and the blocks it was made from dashed; drag with the mouse to turn
    it, or use the buttons below it for the named viewpoints (``view`` is the
    first; :meth:`LevelsView3D.set_view` from code). The middle picture is the
    program as the level rewrites it in 3D (:func:`~zxqec.spacetime.draw_level_3d`),
    turned together with the program: blocks without a rewrite where they are,
    the spiders of a rewritten block laid out between the ends of its wires
    (:func:`~zxqec.spacetime.level_layout_3d`). On the right a block cut out
    of the program (*plain diagram*, or a block without a rewrite) is drawn in 3D
    too; a rewritten block has no place in space-time and is PyZX's drawing, as
    in :func:`view_levels`, with spiders to drag. The slider, block buttons,
    arrow keys, *spread out block* and *reset positions* work as there.
    """
    level = _first_level(program, level, index)
    view3d = LevelsView3D(program, level, index, figsize, block_figsize, level_figsize, view, show_phases)
    view3d.fig.levels_view = view3d
    return view3d.fig
