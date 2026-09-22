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

"""Interactive (JavaScript) views of space-time diagrams for Jupyter notebooks.

In a notebook ``pyzx.draw`` renders with ``zx_viewer.inline.js`` (d3, loaded
from a CDN): an SVG with draggable spiders and mouse-wheel zoom. This module
reuses that viewer with the diagram's *time* as the horizontal axis:

* :func:`draw_web` / :func:`draw_program_web` -- spiders at their timestep,
  delays as squiggly wires with ``Δ`` labels (as in
  ``pyzx.draw_matplotlib(show_time=True)``), the timestep of every spider shown
  above it and, for a program, the blocks of a level shaded behind the diagram;
  shading and squiggles follow dragging and zooming.

The 2+1D (3D) view stays with matplotlib (:func:`~zxqec.spacetime.drawing.draw_program_3d`);
the matplotlib 2D views remain the choice for scripts and static figures.
"""

from __future__ import annotations

import json
import math
import os
import random
import string
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pyzx.drawing import graph_json
from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.spacetime import get_delay, get_timestep
from pyzx.utils import VertexType, get_mode, settings

from .drawing import _block_tick_ranges, _time_layout, block_colors

if TYPE_CHECKING:
    from ..qprogram.program import QProgram

__all__ = ['draw_web', 'draw_program_web', 'graph_html', 'program_html']


def _graph_id() -> str:
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))


def _display(html: str) -> Any:
    if get_mode() == "notebook":
        from IPython.display import HTML, display
        display(HTML(html))
        return None
    return html


# ---------------------------------------------------------------------------
# 2D: d3 viewer with time on the horizontal axis and shaded blocks
# ---------------------------------------------------------------------------

def graph_html(g: BaseGraph[VT, ET], *, time_axis: bool = True,
               blocks: Sequence[tuple[str, Sequence[tuple[int, int]]]] | None = None,
               labels: bool = False, vdata: Sequence[str] = ('t',), squiggles: bool = True,
               scale: float | None = None, alpha: float = 0.18) -> str:
    """The HTML of :func:`draw_web`: PyZX's d3 viewer plus time layout, delays and blocks.

    With ``time_axis`` spiders are placed at their timestep (see
    :func:`~zxqec.spacetime.drawing.time_positions`). With ``squiggles`` a wire
    along one qubit whose earlier spider has a non-zero ``delay`` is drawn as a
    squiggle with one wiggle per tick and a ``Δ<delay>`` label, like
    ``pyzx.draw_matplotlib(show_time=True)``; the squiggles follow the spiders
    when they are dragged. As in ``draw_matplotlib``, a vertical edge in a
    column (tick) holding three or more vertices is bent by an amount growing
    with its length, so that an edge from qubit 0 to qubit 2 is not read as
    0-1 plus 1-2. ``blocks`` is a list of ``(label, tick_ranges)`` shaded
    behind the diagram.
    """
    h = _time_layout(g) if time_axis else g
    rows = [h.row(v) for v in h.vertices()]
    qubits = [h.qubit(v) for v in h.vertices()]
    minrow, maxrow = (min(rows), max(rows)) if rows else (0, 0)
    minqub, maxqub = (min(qubits), max(qubits)) if qubits else (0, 0)
    if scale is None:
        scale = min(50.0, max(20.0, 800 / (maxrow - minrow + 2)))
    width = (maxrow - minrow + 2) * scale
    height = (maxqub - minqub + 3) * scale
    node_size = max(2.0, 0.2 * scale)
    coords = {v: ((h.row(v) - minrow + 1) * scale, (h.qubit(v) - minqub + 2) * scale, 0)
              for v in h.vertices()}
    graphj = graph_json(h, coords, vdata=list(vdata))

    # wires to draw as squiggles: keyed like the viewer's link data
    squiggle: dict[str, int] = {}
    if squiggles:
        counts: dict[tuple[str, str], int] = {}
        for e in h.edges():
            s_, t_ = h.edge_s(e), h.edge_t(e)
            key = (str(s_), str(t_))
            index = counts.get(key, 0)
            counts[key] = index + 1
            if h.qubit(s_) != h.qubit(t_):
                continue
            earlier = s_ if h.row(s_) <= h.row(t_) else t_
            delay = get_delay(h, earlier)
            if delay and delay > 0 and h.type(earlier) != VertexType.BOUNDARY:
                squiggle["{}|{}|{}".format(str(s_), str(t_), index)] = int(delay)

    # column of every vertex (its tick), for PyZX's bend rule on vertical edges
    columns = {str(v): int(math.floor(h.row(v))) for v in h.vertices()}

    shading: list[dict[str, Any]] = []
    for index, (label, ticks) in enumerate(blocks or []):
        color = block_colors(len(blocks or []))[index]
        first = True
        for lo, hi in ticks:
            if time_axis:
                x0, x1 = (lo - 0.5 - minrow + 1) * scale, (hi + 0.5 - minrow + 1) * scale
            else:
                r = [g.row(v) for v in g.vertices() if g.type(v) != VertexType.BOUNDARY
                     and get_timestep(g, v) is not None and lo <= get_timestep(g, v) <= hi]  # type: ignore[operator]
                if not r:
                    continue
                x0, x1 = (min(r) - 0.5 - minrow + 1) * scale, (max(r) + 0.5 - minrow + 1) * scale
            shading.append({'x0': x0, 'x1': x1, 'color': color, 'label': label, 'index': index,
                            'show_label': first})
            first = False

    with open(os.path.join(settings.javascript_location, 'zx_viewer.inline.js')) as f:
        library_code = f.read()
    gid = _graph_id()
    return """<div style="overflow:auto; background-color: white" id="graph-output-{id}"></div>
<script type="module">
var d3;
if (d3 == null) {{ d3 = await import("https://esm.sh/d3@5"); }}
var _settings_colors = JSON.parse('{colors}');
{library_code}
showGraph('#graph-output-{id}', JSON.parse('{graph}'), {width}, {height}, {scale}, {node_size}, false, {labels}, '');
const canvas_{id} = d3.select('#graph-output-{id} svg > g');
const blocks_{id} = canvas_{id}.insert('g', ':first-child');
for (const b of {blocks}) {{
  blocks_{id}.append('rect').attr('x', b.x0).attr('y', 0).attr('width', b.x1 - b.x0).attr('height', {height})
      .attr('fill', b.color).attr('opacity', {alpha});
  if (b.show_label) {{
    blocks_{id}.append('text').attr('x', b.x0 + 4).attr('y', 14 + 13 * (b.index % 2)).attr('font-size', '11px')
        .attr('font-family', 'sans-serif').attr('fill', b.color).text(b.label);
  }}
}}
// delays: squiggly wires that follow the spiders when they are dragged
const squiggle_{id} = {squiggle};
const amp_{id} = {amp};
const delayLabels_{id} = canvas_{id}.append('g');
function squigglePath_{id}(d) {{
  const n = squiggle_{id}[d.source.name + '|' + d.target.name + '|' + d.index];
  if (!n) return null;
  const x1 = d.source.x, y1 = d.source.y, x2 = d.target.x, y2 = d.target.y;
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len, ny = dx / len, steps = Math.max(24, 12 * n);
  let p = `M ${{x1}} ${{y1}}`;
  for (let i = 1; i <= steps; i++) {{
    const f = i / steps, off = amp_{id} * Math.sin(f * n * 2 * Math.PI);
    p += ` L ${{x1 + f * dx + off * nx}} ${{y1 + f * dy + off * ny}}`;
  }}
  return p;
}}
// vertical edges in a column with three or more vertices are bent (PyZX's rule in
// draw_matplotlib), by an amount growing with their length, so that an edge from
// qubit 0 to qubit 2 is not read as 0-1 plus 1-2
const column_{id} = {columns};
const columnSize_{id} = {{}};
for (const name in column_{id}) {{ columnSize_{id}[column_{id}[name]] = (columnSize_{id}[column_{id}[name]] || 0) + 1; }}
function bentPath_{id}(d) {{
  if (d.num_parallel > 1) return null;
  const cs = column_{id}[d.source.name], ct = column_{id}[d.target.name];
  if (cs === undefined || cs !== ct || columnSize_{id}[cs] < 3) return null;
  const x1 = d.source.x, y1 = d.source.y, x2 = d.target.x, y2 = d.target.y;
  if (Math.abs(x2 - x1) > Math.abs(y2 - y1) || y1 === y2) return null;
  const bend = 0.25 * Math.abs(y2 - y1);
  return `M ${{x1}} ${{y1}} Q ${{(x1 + x2) / 2 + bend}} ${{(y1 + y2) / 2}}, ${{x2}} ${{y2}}`;
}}
function applySquiggles_{id}() {{
  const mids = [];
  d3.select('#graph-output-{id} svg g.link').selectAll('path').each(function(d) {{
    const p = squigglePath_{id}(d);
    if (p) {{
      d3.select(this).attr('d', p);
      mids.push({{x: (d.source.x + d.target.x) / 2, y: (d.source.y + d.target.y) / 2,
                  n: squiggle_{id}[d.source.name + '|' + d.target.name + '|' + d.index]}});
    }} else {{
      const b = bentPath_{id}(d);
      if (b) d3.select(this).attr('d', b);
    }}
  }});
  const t = delayLabels_{id}.selectAll('text').data(mids);
  t.enter().append('text').merge(t)
      .attr('x', m => m.x).attr('y', m => m.y - amp_{id} - 4).attr('text-anchor', 'middle')
      .attr('font-size', '9px').attr('font-family', 'monospace').attr('fill', '#8000a0')
      .attr('style', 'pointer-events: none; user-select: none;').text(m => '\\u0394' + m.n);
  t.exit().remove();
}}
applySquiggles_{id}();
const svg_{id} = document.querySelector('#graph-output-{id} svg');
svg_{id}.addEventListener('mousemove', () => requestAnimationFrame(applySquiggles_{id}));
window.addEventListener('mouseup', () => requestAnimationFrame(applySquiggles_{id}));
</script>""".format(id=gid, colors=json.dumps(settings.colors), library_code=library_code,
                    graph=graphj, width=width, height=height, scale=scale, node_size=node_size,
                    labels='true' if labels else 'false', blocks=json.dumps(shading), alpha=alpha,
                    squiggle=json.dumps(squiggle), amp=0.35 * node_size, columns=json.dumps(columns))


def draw_web(g: BaseGraph[VT, ET], **kwargs: Any) -> Any:
    """Interactive d3 view of a time-annotated graph (see :func:`graph_html`).

    Like ``pyzx.draw`` in a notebook (drag spiders, wheel to zoom), with the
    spiders at their timesteps, delays drawn as squiggly wires with ``Δ``
    labels, and the timestep above each spider (``vdata``; pass ``()`` for
    none). Displays in a notebook, returns the HTML otherwise.
    """
    return _display(graph_html(g, **kwargs))


def program_html(program: 'QProgram', level: int | None = None, *, classical: bool = False,
                 **kwargs: Any) -> str:
    """The HTML of :func:`draw_program_web` (also usable outside a notebook)."""
    g = program.graph() if classical else program.quantum_graph()
    blocks = _block_tick_ranges(program, level) if level is not None else None
    return graph_html(g, blocks=blocks, **kwargs)


def draw_program_web(program: 'QProgram', level: int | None = None, **kwargs: Any) -> Any:
    """Interactive d3 view of a program's master graph (spiders draggable, wheel to zoom).

    Spiders are placed at their timestep along the horizontal axis
    (``time_axis=False`` keeps PyZX's row layout); delays are squiggly wires
    with ``Δ`` labels; the blocks of ``level`` are shaded and labelled behind
    the diagram; ``vdata`` (default ``('t',)``, the timestep) names vertex data
    shown above each spider (pass ``()`` for none); ``labels`` shows vertex
    ids; ``classical=True`` also draws the classical-bit wires. Displays in a
    notebook, returns the HTML otherwise.
    """
    return _display(program_html(program, level, **kwargs))
