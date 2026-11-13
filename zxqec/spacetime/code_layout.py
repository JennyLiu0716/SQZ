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

"""A reduced diagram in a code's variables, drawn in 3D on the code's own layout.

The input is a diagram in generalized parity form whose outputs are the qubits of
a code, written in the variables of the code: one Z-spider per face (an X-type
check drawn as a face of the layout) and one per logical X. This is what
:func:`~zxqec.gpf.change_variables` gives with the rows "logical X, then the
faces"; :func:`in_face_variables` does it for a reduced diagram.
:func:`code_layout` puts every spider in a place, as the textbook picture of the
code:

* height 2, the plane of the code: the X-spider of each code qubit at the place
  of the qubit (its coordinates in the circuit file, or, with ``regular=True``,
  a place on an equilateral triangle: :func:`regular_triangle_coords`), and the
  spider of each face in the middle of its face;
* height 1: the spider of each logical X, outside the layout, next to the side
  it is wired to; the other Z-spiders (hidden variables, wired to no output), in
  a row outside the layout, on the side away from the logical spider; each
  gadget hub under the qubit whose column it reads (the face and logical
  spiders wired to that qubit: a T gate on that qubit), else under the middle
  of the face and logical spiders it reads;
* height 0: the one-legged spiders, under their neighbour. The constant phase
  of a logical spider, when it is not Clifford, is split off first (spider
  fusion read backwards): the state that the code holds, such as ``T^dag|+>``;
* height 3: the outputs, above their qubits.

Spiders that would land on the same place are spread on a small circle (a
gadget hub also when a spider of another height is there, so that it can be
seen from above). The gadgets of one qubit, two when the fault variables are
kept, go side by side along x, the one that reads the fewest hidden spiders on
the left. The logical spider stands 2.2 steps (between the qubits of its side)
out from the layout, so that the oblique view shows it clear. The result is a :class:`CodeLayout`: the placed diagram (a copy, the same map), a
:class:`~zxqec.spacetime.Topology` with the sides of the faces, and the
positions. :meth:`CodeLayout.draw` draws it with
:func:`~zxqec.spacetime.draw_graph_3d`, with the faces as coloured patches in the
plane of the code; :func:`draw_code_3d` does both in one call.

The faces are an argument. Two ways to get them:

* ``code.layout_checks(coords)['X']`` of the code
  (:meth:`~zxqec.codes.StabilizerCode.layout_checks`): the most compact
  generating set of the X stabilizers on the layout. It lists the whole group,
  so it works up to 20 X generators: it finds the faces of Stim's colour codes
  at d = 3, 5 and 7 (18 generators, about 10 s), and refuses d = 9 (30);
* the circuit file itself, when it measures the faces, as the ``MPP`` lines of
  the final check of d = 5 cultivation do.

The recipe, for d = 3 and d = 5 cultivation alike (run it from ``zxqec/examples``;
for d = 5 read ``d5_cultivation.stim``)::

    from zxqec.gpf import reduce_gpf
    from zxqec.qprogram import program_from_stim
    from zxqec.spacetime import code_layout, in_face_variables

    p, circ = program_from_stim(open('circuits/cultivation/d3_cultivation.stim').read())
    code = p.code_at(p.num_stages - 1).data()          # the code at the end
    faces = code.layout_checks(circ.coords)['X']       # its faces
    m = in_face_variables(reduce_gpf(p.quantum_graph()), code, faces)
    lay = code_layout(m, code.qubits, circ.coords, faces, regular=True)
    fig = lay.draw()                                   # or lay.draw('top')

``regular=True`` draws the triangle as in the textbook; the d = 3 file places
the Steane code lopsided. To keep the fault variables as they are, give
``error_formulas(p).reduced`` to :func:`in_face_variables` with
``simplify=False, modulo=False``: only the variables change, and one gadget per
fault stays under its qubit.
"""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
from pyzx.graph.base import BaseGraph
from pyzx.spacetime import set_timestep
from pyzx.symbolic import Poly
from pyzx.utils import VertexType, phase_to_s

from .topology import Topology

__all__ = ['CodeLayout', 'code_layout', 'draw_code_3d', 'in_face_variables', 'regular_triangle_coords', 'HEIGHTS',
           'FACE_COLOURS', 'VIEW', 'VIEW_HUBS']

#: The height of each kind of spider (see the module docstring).
HEIGHTS: dict[str, int] = {'state': 0, 'leaf': 0, 'logical': 1, 'hub': 1, 'hidden': 1, 'other': 1,
                           'face': 2, 'qubit': 2, 'output': 3}

#: The default viewpoint ``(elev, azim)``: a little higher than :data:`~zxqec.spacetime.VIEWS` ``'default'``,
#: so that the layout reads as in its textbook picture.
VIEW: tuple[float, float] = (35.0, -70.0)

#: The default viewpoint when the diagram has gadget hubs (fault variables kept, for example): lower than
#: :data:`VIEW`, so that the plane of the code, the hubs and their leaves lie apart on the screen.
VIEW_HUBS: tuple[float, float] = (28.0, -70.0)

#: The three colours of the faces, as in the textbook picture of the Steane code.
FACE_COLOURS: tuple[str, str, str] = ('#4C72B0', '#DD8452', '#55A868')


@dataclass
class CodeLayout:
    """A diagram placed on the layout of its code (see :func:`code_layout`)."""
    graph: BaseGraph                                 #: a copy of the diagram, a state split off; qubit = node, timestep = height
    topology: Topology                               #: the qubits, the sides of the faces and a node for every other place
    positions: dict[Any, tuple[float, float, float]]  #: ``(x, y, height)`` of every vertex of :attr:`graph`
    roles: dict[Any, str]                            #: what each vertex is: a key of :data:`HEIGHTS`
    names: dict[Any, str]                            #: ``L`` (or ``L1``, ...), ``S1``, ... for the faces, ``h1``, ... hidden
    qubits: list[Hashable]                           #: the label of each output, in order
    faces: list[tuple[Hashable, ...]]                #: the faces, as tuples of qubit labels
    qubit_spiders: dict[Any, int] = field(default_factory=dict)   #: the X-spider of each output -> the output's index
    face_spiders: list[list[Any]] = field(default_factory=list)   #: the spiders wired to exactly face ``i``
    colours: list[int] = field(default_factory=list)              #: the colour (0, 1, 2, ...) of face ``i``
    split: list[tuple[Any, Any]] = field(default_factory=list)    #: ``(logical spider, state split off it)``

    def face_middle(self, i: int) -> tuple[float, float]:
        """The middle of face ``i``: the mean of the places of its qubits."""
        pts = [self.positions[self._spider_of[q]][:2] for q in self.faces[i]]
        return (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))

    @property
    def _spider_of(self) -> dict[Hashable, Any]:
        return {self.qubits[j]: v for v, j in self.qubit_spiders.items()}

    def draw(self, view: str | tuple[float, float] | None = None, *, ax: Any = None,
             figsize: tuple[float, float] | None = None, names: bool = True, qubit_labels: bool | None = None,
             phases: str = 'auto', faces: bool = True, title: str | None = None, finish: bool = True,
             **kwargs: Any) -> Any:
        """Draw the placed diagram in 3D. Returns the figure.

        ``view`` is an ``(elev, azim)`` pair or a name of :data:`~zxqec.spacetime.VIEWS`
        (``'default'``, ``'top'``, ``'front'``, ``'side'``); by default :data:`VIEW`, or
        :data:`VIEW_HUBS` when the diagram has gadget hubs. ``names`` writes ``L``,
        ``S1``, ... and ``h1``, ... beside their spiders; ``qubit_labels`` writes the label
        of each code qubit beside its X-spider (by default in the view from the top, and
        in the others when there are at most 10 qubits). ``phases`` says how phases are
        written:

        * ``'auto'`` (the default): a short phase (at most 7 characters, such as ``7pi/4``)
          beside its spider, a longer one in a list under the picture, after the name of
          its spider (:meth:`phase_list`); a spider with a long phase and no name gets
          one (``g1``, ``g2``, ...), written beside it;
        * ``'all'``: every phase beside its spider;
        * ``'constant'``: only the number in each phase, beside its spider (the outcome
          and fault variables left out);
        * ``'none'``: no phases.

        A label goes on the side of its spider where it covers the fewest other labels,
        spiders and wires: the name of a face below its spider if that side is free,
        the label of a qubit away from the middle of the code. The wires of hidden
        spiders are drawn light. The list of long phases goes under the picture, and
        the figure gets taller to hold it, so that a plain ``fig.savefig(path)`` keeps
        every line (with ``ax`` given, the axes are lifted instead).

        ``faces`` fills each face in the plane of the code, in three colours. ``ax`` draws
        into an existing 3D axes. ``figsize`` is ``(8, 6.5)`` by default, ``(9, 7.5)``
        with gadget hubs. Other keyword arguments go to
        :func:`~zxqec.spacetime.draw_graph_3d`.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        from .drawing import VIEWS, _finish, draw_graph_3d

        hubs = any(r == 'hub' for r in self.roles.values())
        if view is None:
            view = VIEW_HUBS if hubs else VIEW
        if figsize is None:
            figsize = (9, 7.5) if hubs else (8, 6.5)
        if phases not in ('auto', 'all', 'constant', 'none'):
            raise ValueError("phases is 'auto', 'all', 'constant' or 'none', not {!r}".format(phases))
        made = ax is None
        if made:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        fig = ax.figure
        kwargs.setdefault('show_topology', False)
        kwargs.setdefault('space_ticks', False)
        draw_graph_3d(self.graph, self.topology, ax=ax, view=view, positions=self.positions,
                      show_phases=False, finish=False, **kwargs)
        hidden = {self.positions[v] for v, r in self.roles.items() if r == 'hidden'}
        if hidden:                                # the wires of hidden spiders light, so that they stay apart
            for line in ax.lines:
                xs, ys, zs = (list(c) for c in line.get_data_3d())
                if len(xs) == 2 and any((xs[i], ys[i], zs[i]) in hidden for i in range(2)):
                    line.set_color('#9a9a9a')
                    line.set_linewidth(0.6)
                    line.set_alpha(0.6)
        elev = VIEWS[view][0] if isinstance(view, str) else view[0]
        top = abs(elev - 90) < 1e-9
        plane = HEIGHTS['qubit']
        if faces:
            palette = FACE_COLOURS + tuple(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            for i in range(len(self.faces)):
                ring = self._ring(i)
                poly = Poly3DCollection([[(x, y, plane) for x, y in ring]], facecolor=palette[self.colours[i]],
                                        edgecolor='#555555', linewidths=0.8, alpha=0.22)
                ax.add_collection3d(poly)
        spacing = self._spacing()
        shown, listed, extra = self._phase_texts(phases)
        if qubit_labels is None:
            qubit_labels = top or len(self.qubits) <= 10
        drawn = ax.__dict__.setdefault('_zx_drawn', {'labels': [], 'points': [], 'segments': []})
        if faces:                                 # the sides of the faces, for the labels to keep off
            for i in range(len(self.faces)):
                ring = self._ring(i)
                drawn['segments'] += [((a[0], a[1], plane), (b[0], b[1], plane))
                                      for a, b in zip(ring, ring[1:] + ring[:1])]
        # every label as (text, anchor, the side it would like): names first, so the others make room for them
        labels: list[tuple[Any, tuple[float, float, float], Any]] = []
        for v in sorted(self.positions, key=lambda v: (self._label_order(v), self._sort_name(v), str(v))):
            x, y, z = self.positions[v]
            name = self.names.get(v, extra.get(v)) if names or v in extra else None
            if name is not None:
                big = v in self.names
                t = ax.text(x, y, z, name, fontsize=9 if big else 6, fontweight='bold' if big else 'normal',
                            color='#222222', zorder=10)
                want = 'down' if self.roles[v] == 'face' else ('out', self._outward(x, y))
                labels.append((t, (x, y, z), want))
        if qubit_labels:
            for v in sorted(self.qubit_spiders, key=lambda v: self.qubit_spiders[v]):
                x, y, z = self.positions[v]
                t = ax.text(x, y, z, str(self.qubits[self.qubit_spiders[v]]), fontsize=7, color='#666666', zorder=10)
                labels.append((t, (x, y, z), ('out', self._outward(x, y))))
        for v in sorted(shown, key=lambda v: (self._label_order(v), self._sort_name(v), str(v))):
            x, y, z = self.positions[v]
            t = ax.text(x, y, z, shown[v], fontsize=7, color='blue', zorder=10)
            labels.append((t, (x, y, z), 'down' if self.roles[v] in ('state', 'leaf') else 'lower right'))
        # the same scale on x and y, the heights a little flatter
        xs = [p[0] for p in self.positions.values()]
        ys = [p[1] for p in self.positions.values()]
        pad = 0.4 * spacing
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        xr, yr = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
        ax.set_box_aspect((xr, yr, 0.75 * max(xr, yr)), zoom=1.3 if top else 1.0)
        ax.set_zlabel('' if top else 'height')
        if top:
            ax.set_proj_type('ortho')             # the outputs right above their qubits
        else:
            ax.set_zticks(sorted({int(p[2]) for p in self.positions.values()}))
        if title is not None:
            ax.set_title(title, fontsize=9)
        if listed:
            self._list_below(ax, listed, grow=made)
        _place_labels_in(ax, labels)
        return _finish(fig) if finish else fig

    def phase_list(self) -> list[tuple[str, str]]:
        """The phases that ``draw(phases='auto')`` writes under the picture: ``(name, phase)`` pairs."""
        return self._phase_texts('auto')[1]

    def _phase_texts(self, phases: str) -> tuple[dict[Any, str], list[tuple[str, str]], dict[Any, str]]:
        """The phases written beside spiders, those in the list (with their names), and the names given for it."""
        shown: dict[Any, str] = {}
        listed: list[tuple[str, str]] = []
        extra: dict[Any, str] = {}
        if phases == 'none':
            return shown, listed, extra
        order = {'logical': 0, 'state': 1, 'face': 2, 'hidden': 3, 'qubit': 4, 'hub': 5, 'leaf': 6, 'other': 7}
        for v in sorted((v for v in self.positions if self.roles[v] != 'output'),
                        key=lambda v: (order[self.roles[v]], self._sort_name(v), str(v))):
            p = self.graph.phase(v)
            if phases == 'constant':
                p = _constant(p)
            s = _in_order(phase_to_s(p, self.graph.type(v)))
            if not s or s == '0':
                continue
            if phases != 'auto' or len(s) <= 7:
                shown[v] = s
                continue
            if v in self.names:
                name = self.names[v]
            elif self.roles[v] == 'qubit':
                name = 'qubit {}'.format(self.qubits[self.qubit_spiders[v]])
            else:
                name = extra[v] = 'g{}'.format(len(extra) + 1)
            listed.append((name, s))
        return shown, listed, extra

    def _label_order(self, v: Any) -> int:
        """Which labels are placed first: the faces, the logical spiders, the hidden ones, then the rest."""
        return {'face': 0, 'logical': 1, 'hidden': 2, 'state': 3}.get(self.roles[v], 4)

    @staticmethod
    def _list_below(ax: Any, listed: list[tuple[str, str]], grow: bool) -> None:
        """Write the long phases under the axes, and make room for them in the figure.

        With ``grow`` (a figure made by :meth:`draw`) the figure gets taller by the
        height of the list; else the bottom of the axes is lifted, so that a plain
        ``fig.savefig(path)`` keeps every line.
        """
        fig = ax.figure
        size = 7
        need = (len(listed) * size * 1.25 + 8) / 72        # inches: the lines and a margin
        w, h = (float(x) for x in fig.get_size_inches())
        pos = ax.get_position()
        free = 0.8 * pos.y0 * h                              # the white under the axes, which the list may use
        if grow and need > free:
            more = need - free
            fig.set_size_inches(w, h + more)
            ax.set_position([pos.x0, (pos.y0 * h + more) / (h + more), pos.width, pos.height * h / (h + more)])
            h += more
        elif not grow and pos.y0 * h < need:
            ax.set_position([pos.x0, need / h, pos.width, pos.y1 - need / h])
        pos = ax.get_position()                              # now at least `need` above the bottom of the figure
        ax.text2D(pos.x0, pos.y0, '\n'.join('{}: {}'.format(n, s) for n, s in listed), transform=fig.transFigure,
                  fontsize=size, color='blue', ha='left', va='top', family='monospace', linespacing=1.25)

    def _sort_name(self, v: Any) -> tuple[str, int]:
        """Names in their natural order: S2 before S10."""
        name = self.names.get(v, '')
        head = name.rstrip('0123456789')
        return (head, int(name[len(head):]) if name[len(head):] else 0)

    def draw_plane(self, height: int = HEIGHTS['qubit'], title: str | None = None,
                   figsize: tuple[float, float] = (5, 4.5), **kwargs: Any) -> Any:
        """The plane at ``height`` alone, flat, with PyZX's drawing (:func:`~zxqec.spacetime.draw_slice`).

        At height 2 this is the code without its state: the X-spiders of the qubits and
        the spiders of the faces. Returns the figure.
        """
        from .drawing import draw_slice

        return draw_slice(self.graph, self.topology, z=height, positions=self.positions, title=title,
                          figsize=figsize, **kwargs)

    # -- helpers ----------------------------------------------------------------------

    def _ring(self, i: int) -> list[tuple[float, float]]:
        """The places of the qubits of face ``i``, around its middle."""
        c = self.face_middle(i)
        pts = [self.positions[self._spider_of[q]][:2] for q in self.faces[i]]
        return sorted(pts, key=lambda p: math.atan2(p[1] - c[1], p[0] - c[0]))

    def _spacing(self) -> float:
        pts = np.array([self.positions[v][:2] for v in self.qubit_spiders], dtype=float)
        return _spacing(pts)

    def _outward(self, x: float, y: float) -> tuple[float, float]:
        """A unit vector from the middle of the code towards ``(x, y)``; down-left at the middle itself."""
        pts = np.array([self.positions[v][:2] for v in self.qubit_spiders], dtype=float)
        d = np.array([x, y]) - pts.mean(axis=0)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return (-0.7071, -0.7071)
        return (float(d[0] / n), float(d[1] / n))


def code_layout(g: BaseGraph, qubits: Sequence[Hashable], coords: Mapping[Hashable, Sequence[float]],
                faces: Sequence[Sequence[Hashable]], *, split_state: bool = True,
                y_down: bool = False, regular: bool = False) -> CodeLayout:
    """Place a diagram in the variables of a code on the code's layout, as the textbook picture.

    ``g`` is in generalized parity form; its outputs are the code qubits, and output
    ``j`` is the qubit ``qubits[j]`` (for a code read by :meth:`QProgram.code_at`,
    ``code.qubits``). ``coords[q]`` is the place ``(x, y)`` of qubit ``q``, as in the
    ``QUBIT_COORDS`` of a Stim file (``program_from_stim`` returns them); ``y_down``
    puts larger ``y`` lower. ``faces`` are the X checks drawn as faces, each a list of
    qubit labels (see the module docstring for two ways to get them).
    :func:`in_face_variables` writes a reduced diagram in these variables.

    Which places to use:

    * ``regular=True`` for a triangular colour code: the qubits go on an
      equilateral triangle (:func:`regular_triangle_coords`; ``coords`` then only
      give the order around each face). Use it for cultivation: the d = 3 file
      places the Steane code lopsided, and at d = 5 it looks right too;
    * ``regular=False`` (the default) for any other layout: the qubits stay at
      ``coords``, as on the chip.

    A Z-spider wired to the X-spiders of exactly the qubits of a face is that face's
    spider. A Z-spider wired to output X-spiders that are not a face is a logical
    spider, unless its qubits are a product of faces: then ``g`` is not in the faces'
    variables, and ``ValueError`` says to change the variables first. With
    ``split_state`` (the default), the constant part of a logical spider's phase is
    split off as a one-legged Z-spider when it is not a multiple of pi/2; the phase
    that depends on outcomes stays on the logical spider. This is spider fusion read
    backwards, so the placed diagram is the same map as ``g``; nothing else changes
    but the places (``qubit`` and ``timestep`` of each vertex). ``g`` is not changed.
    """
    outs = list(g.outputs())
    if len(qubits) != len(outs):
        raise ValueError("{} qubit labels for {} outputs".format(len(qubits), len(outs)))
    missing = [q for q in qubits if q not in coords]
    if missing:
        raise ValueError("no coordinates for qubits {}".format(missing))
    faces_t = [tuple(f) for f in faces]
    unknown = sorted({q for f in faces_t for q in f if q not in qubits}, key=str)
    if unknown:
        raise ValueError("faces name qubits that are not outputs: {}".format(unknown))
    index = {q: j for j, q in enumerate(qubits)}
    if regular:
        where = regular_triangle_coords(coords, faces_t, y_down=y_down)
        alone = [q for q in qubits if q not in where]
        if alone:
            raise ValueError("regular places need every qubit in a face; not in one: {}".format(alone))
        xy = np.array([where[q] for q in qubits])
    else:
        flip = -1.0 if y_down else 1.0
        xy = np.array([(float(coords[q][0]), flip * float(coords[q][1])) for q in qubits])

    h = _own_copy(g)                          # keeps the vertex numbers; g is not changed
    h_outs = list(h.outputs())
    qubit_spider: dict[Any, int] = {}
    for j, b in enumerate(h_outs):
        nbs = list(h.neighbors(b))
        if len(nbs) != 1:
            raise ValueError("output {} has {} neighbours, not one".format(j, len(nbs)))
        qubit_spider[nbs[0]] = j
    roles: dict[Any, str] = {b: 'output' for b in h_outs}
    roles.update({v: 'qubit' for v in qubit_spider})

    face_of = {frozenset(index[q] for q in f): i for i, f in enumerate(faces_t)}
    face_rows = np.zeros((len(faces_t), len(qubits)), dtype=np.uint8)
    for i, f in enumerate(faces_t):
        for q in f:
            face_rows[i, index[q]] = 1
    face_spiders: list[list[Any]] = [[] for _ in faces_t]
    wired: dict[Any, frozenset[int]] = {}
    for v in h.vertices():
        if v in roles:
            continue
        w = frozenset(qubit_spider[u] for u in h.neighbors(v) if u in qubit_spider)
        if h.type(v) == VertexType.Z and w:
            wired[v] = w
            if w in face_of:
                roles[v] = 'face'
                face_spiders[face_of[w]].append(v)
            elif _in_span(face_rows, w, len(qubits)):
                raise ValueError("a Z-spider is wired to qubits {}, a product of faces but not a face: write the "
                                 "diagram in the variables of the faces first (change_variables with the rows "
                                 "'logical X, then the faces')".format(sorted((qubits[j] for j in w), key=str)))
            else:
                roles[v] = 'logical'

    # split the non-Clifford constant off each logical spider
    split: list[tuple[Any, Any]] = []
    if split_state:
        for v in sorted((v for v, r in roles.items() if r == 'logical'), key=str):
            c = _constant(h.phase(v))
            if c is not None and (Fraction(c) * 2).denominator != 1:
                s = h.add_vertex(VertexType.Z, phase=c)
                h.add_edge((s, v))
                rest = h.phase(v) - c
                h.set_phase(v, 0 if isinstance(rest, Poly) and not rest.terms else rest)
                split.append((v, s))

    for v in h.vertices():
        if v in roles:
            continue
        nbs = list(h.neighbors(v))
        ty = h.type(v)
        if len(nbs) == 1 and ty in (VertexType.Z, VertexType.X) and nbs[0] not in qubit_spider:
            roles[v] = 'state' if roles.get(nbs[0]) == 'logical' else 'leaf'
        elif ty == VertexType.X:
            roles[v] = 'hub'
        elif ty == VertexType.Z:
            roles[v] = 'hidden'
        else:
            roles[v] = 'other'

    # places in the plane
    spacing = _spacing(xy)
    centre = xy.mean(axis=0)
    place: dict[Any, np.ndarray] = {}
    for v, j in qubit_spider.items():
        place[v] = xy[j]
    for b in h_outs:
        place[b] = place[next(iter(h.neighbors(b)))]
    for i, vs in enumerate(face_spiders):
        for v in vs:
            place[v] = xy[[index[q] for q in faces_t[i]]].mean(axis=0)
    logical = sorted((v for v, r in roles.items() if r == 'logical'), key=str)
    away = None
    for v in logical:
        place[v], u = _outside(xy, sorted(wired[v]), centre, spacing)
        away = -u if away is None else away
    hidden = sorted((v for v, r in roles.items() if r == 'hidden'), key=str)
    if hidden:
        u = away if away is not None else np.array([0.0, -1.0])
        perp = np.array([-u[1], u[0]])
        reach = float(max((xy - centre) @ u))
        base = centre + (reach + 1.5 * spacing) * u
        for k, v in enumerate(hidden):
            place[v] = base + (k - (len(hidden) - 1) / 2) * spacing * perp
    # gadget hubs: under the qubit whose column they read (the Z-spiders on the outputs wired to that qubit's
    # X-spider), else the middle of the face and logical spiders they read, else of all their placed neighbours
    column = {frozenset(u for u in h.neighbors(x) if roles.get(u) in ('face', 'logical')): j
              for x, j in qubit_spider.items()}
    rest = [v for v, r in roles.items() if r in ('hub', 'other')]
    for v in list(rest):
        reads = frozenset(u for u in h.neighbors(v) if roles.get(u) in ('face', 'logical'))
        if roles[v] == 'hub' and reads in column:
            place[v] = xy[column[reads]]
            rest.remove(v)
        elif reads:
            place[v] = np.mean([place[u] for u in reads], axis=0)
            rest.remove(v)
    while rest:                               # the rest, in rounds (for chains)
        done = []
        for v in rest:
            nbs = [u for u in h.neighbors(v) if u in place and roles[u] not in ('state', 'leaf')]
            if nbs:
                place[v] = np.mean([place[u] for u in nbs], axis=0)
                done.append(v)
        if not done:                          # nothing to follow: a row below the layout
            low = float(xy[:, 1].min()) - 2.5 * spacing
            for k, v in enumerate(rest):
                place[v] = np.array([float(xy[:, 0].min()) + k * spacing, low])
            break
        rest = [v for v in rest if v not in done]

    height = {v: HEIGHTS[roles[v]] for v in h.vertices() if roles[v] not in ('state', 'leaf')}
    fixed = [v for v in height if roles[v] in ('qubit', 'output')]
    _spread(place, height, [v for v in height if roles[v] in ('face', 'logical', 'hidden')], fixed,
            0.3 * spacing, by_height=True)
    # the gadgets of one qubit side by side along x (across the screen in the usual views), the one that reads
    # fewest hidden spiders on the left
    _spread(place, height, [v for v in height if roles[v] in ('hub', 'other')],
            [v for v in height if roles[v] not in ('hub', 'other')], 0.3 * spacing, by_height=False,
            start=math.pi, order=lambda v: (sum(roles.get(u) == 'hidden' for u in h.neighbors(v)), str(v)))
    for v, r in roles.items():                # one-legged spiders under their neighbour
        if r in ('state', 'leaf'):
            (u,) = list(h.neighbors(v))
            if u not in place:
                place[u], height[u] = centre, HEIGHTS['other']
            place[v] = place[u]
            height[v] = max(height[u] - 1, 0)
    positions = {v: (float(place[v][0]), float(place[v][1]), float(height[v])) for v in h.vertices()}

    # the topology: a node per place, the qubits first (their labels when these are numbers)
    topo = Topology(2)
    numbered = all(isinstance(q, (int, np.integer)) and not isinstance(q, bool) for q in qubits)
    node_of_place: dict[tuple[float, float], int] = {}
    for j, q in enumerate(qubits):
        n = int(q) if numbered else j        # type: ignore[call-overload]
        topo.add_node(n, tuple(xy[j]))
        node_of_place[_key(xy[j])] = n
    nxt = max(node_of_place.values(), default=-1) + 1
    for v in sorted(h.vertices(), key=str):
        at = _key(positions[v][:2])
        if at not in node_of_place:
            topo.add_node(nxt, positions[v][:2])
            node_of_place[at] = nxt
            nxt += 1
    for f in faces_t:
        c = xy[[index[q] for q in f]].mean(axis=0)
        ring = sorted(f, key=lambda q: math.atan2(xy[index[q]][1] - c[1], xy[index[q]][0] - c[0]))
        for a, b in zip(ring, ring[1:] + ring[:1]):
            na, nb = node_of_place[_key(xy[index[a]])], node_of_place[_key(xy[index[b]])]
            if na != nb and not topo.are_adjacent(na, nb):
                topo.add_edge(na, nb)
    for v in h.vertices():
        h.set_qubit(v, node_of_place[_key(positions[v][:2])])
        if roles[v] != 'output':
            set_timestep(h, v, int(positions[v][2]))

    names: dict[Any, str] = {}
    for i, vs in enumerate(face_spiders):
        for v in vs:
            names[v] = 'S{}'.format(i + 1)
    for k, v in enumerate(logical):
        names[v] = 'L' if len(logical) == 1 else 'L{}'.format(k + 1)
    for k, v in enumerate(hidden):
        names[v] = 'h{}'.format(k + 1)

    layout = CodeLayout(graph=h, topology=topo, positions=positions, roles=roles, names=names,
                        qubits=list(qubits), faces=faces_t, qubit_spiders=dict(qubit_spider),
                        face_spiders=face_spiders, colours=_colour_faces(faces_t), split=split)
    return layout


def regular_triangle_coords(coords: Mapping[Hashable, Sequence[float]], faces: Sequence[Sequence[Hashable]], *,
                            y_down: bool = False) -> dict[Hashable, tuple[float, float]]:
    """The qubits of a triangular colour code on an equilateral triangle, as in the textbook picture.

    ``coords`` are the places of the qubits in the circuit file. They are used only
    for the order of the qubits around each face and for which way round the
    triangle goes; ``faces`` are the faces, each a list of qubit labels. The new
    places are a Tutte (barycentric) embedding:

    * the three corners (the qubits in exactly one face) on the corners of an
      equilateral triangle, the corner highest in ``coords`` at the top;
    * the other qubits of the border evenly along the sides, neighbours 1 apart;
    * each inner qubit at the mean of its neighbours (the qubits it shares a side
      of a face with).

    Every face is then convex. The triangle turns the same way round as in
    ``coords``; ``y_down`` says that larger ``y`` is lower there. Returns a place
    for each qubit of a face. Raises ``ValueError`` when the faces are not a
    triangle: the sides that lie in one face only must make one cycle through
    exactly three corners.
    """
    faces_t = [tuple(f) for f in faces]
    flip = -1.0 if y_down else 1.0
    qs = sorted({q for f in faces_t for q in f}, key=str)
    missing = [q for q in qs if q not in coords]
    if missing:
        raise ValueError("no coordinates for qubits {}".format(missing))
    at = {q: np.array([float(coords[q][0]), flip * float(coords[q][1])]) for q in qs}
    # the sides of the faces, and how many faces each side is in
    sides: dict[frozenset[Hashable], int] = {}
    nbrs: dict[Hashable, set[Hashable]] = {q: set() for q in qs}
    for f in faces_t:
        c = np.mean([at[q] for q in f], axis=0)
        ring = sorted(f, key=lambda q: math.atan2(at[q][1] - c[1], at[q][0] - c[0]))
        for a, b in zip(ring, ring[1:] + ring[:1]):
            if a != b:
                sides[frozenset((a, b))] = sides.get(frozenset((a, b)), 0) + 1
                nbrs[a].add(b)
                nbrs[b].add(a)
    # the border: the sides in one face only, one cycle
    border: dict[Hashable, list[Hashable]] = {}
    for s, n in sides.items():
        if n == 1:
            a, b = tuple(s)
            border.setdefault(a, []).append(b)
            border.setdefault(b, []).append(a)
    if not border or any(len(v) != 2 for v in border.values()):
        raise ValueError("the faces are not a triangle: their border is not one cycle")
    start = min(border, key=str)
    cycle = [start, border[start][0]]
    while cycle[-1] != start:
        a, b = border[cycle[-1]]
        cycle.append(b if a == cycle[-2] else a)
    cycle.pop()
    if len(cycle) != len(border):
        raise ValueError("the faces are not a triangle: their border is more than one cycle")
    count = {q: sum(q in f for f in faces_t) for q in qs}
    corners = [q for q in cycle if count[q] == 1]
    if len(corners) != 3:
        raise ValueError("the faces are not a triangle: {} qubits are in one face only, not three".format(
            len(corners)))
    # counterclockwise, from the corner at the top
    area = sum(at[a][0] * at[b][1] - at[b][0] * at[a][1] for a, b in zip(cycle, cycle[1:] + cycle[:1]))
    if area < 0:
        cycle.reverse()
    top = max(corners, key=lambda q: (at[q][1], -at[q][0]))
    k = cycle.index(top)
    cycle = cycle[k:] + cycle[:k]
    cut = [cycle.index(q) for q in sorted(corners, key=cycle.index)] + [len(cycle)]
    steps = [cut[i + 1] - cut[i] for i in range(3)]
    size = float(max(steps))
    ends = [np.array([0.0, size * math.sqrt(3) / 2]), np.array([-size / 2, 0.0]), np.array([size / 2, 0.0])]
    place: dict[Hashable, np.ndarray] = {}
    for i in range(3):
        for j in range(steps[i]):
            place[cycle[cut[i] + j]] = ends[i] + (ends[(i + 1) % 3] - ends[i]) * j / steps[i]
    # the inner qubits: each at the mean of its neighbours (a linear system)
    inner = [q for q in qs if q not in place]
    if inner:
        index = {q: i for i, q in enumerate(inner)}
        a = np.zeros((len(inner), len(inner)))
        b = np.zeros((len(inner), 2))
        for q in inner:
            i = index[q]
            a[i, i] = len(nbrs[q])
            for u in nbrs[q]:
                if u in index:
                    a[i, index[u]] -= 1
                else:
                    b[i] += place[u]
        try:
            xy = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            raise ValueError("the faces are not a triangle: some inner qubits are not joined to the border")
        for q in inner:
            place[q] = xy[index[q]]
    return {q: (float(place[q][0]), float(place[q][1])) for q in qs}


def in_face_variables(reduced: Any, code: Any, faces: Sequence[Sequence[Hashable]], *,
                      simplify: bool = True, modulo: bool = True) -> BaseGraph:
    """A reduced diagram written in the variables of a code's faces: the input of :func:`code_layout`.

    ``reduced`` is what :func:`~zxqec.gpf.reduce_gpf` returns, or the ``reduced`` of
    :func:`~zxqec.qprogram.error_formulas`: a diagram (``reduced.graph``) and its
    detectors (``reduced.constraints``). Its outputs are the qubits of ``code``, in
    the order of ``code.qubits`` (``code = program.code_at(stage).data()``).
    ``faces`` are the X checks drawn as faces, each a list of qubit labels.

    The new variables are one per logical X of the code, then one per face
    (:func:`~zxqec.gpf.change_variables` with these rows). With ``simplify`` (the
    default), the gadgets are first simplified until nothing changes
    (:func:`~zxqec.gpf.simplify_gadgets`, in the new variables): at d = 5
    cultivation this sums the 22 gadgets of the T gates into one phase, 7pi/4, on
    the logical spider. With ``modulo`` (the default), every phase is then
    written modulo the detectors (:func:`~zxqec.gpf.phases_modulo`): an outcome
    that a detector fixes, such as an outcome that is always 0, drops out.

    For a diagram with fault variables, give ``simplify=False, modulo=False``:
    then only the variables change (:func:`~zxqec.gpf.change_variables`), every
    outcome and fault variable stays where it was, and each fault keeps its own
    gadget under its qubit.

    The result is the same map up to a non-zero scalar in each run, as
    :func:`~zxqec.gpf.reduce_gpf` keeps it. ``reduced`` is not changed.
    """
    from ..gpf import change_variables, phases_modulo, simplify_gadgets

    qubits = list(code.qubits)
    unknown = sorted({q for f in faces for q in f if q not in qubits}, key=str)
    if unknown:
        raise ValueError("faces name qubits that are not in the code: {}".format(unknown))
    rows = [[int(b) for b in lx.x] for lx in code.logical_x] + [[int(q in f) for q in qubits] for f in faces]
    if simplify:
        m = change_variables(reduced.graph, rows, fuse=False)
        while simplify_gadgets(m):
            pass
        m = change_variables(m, rows)
    else:
        m = change_variables(reduced.graph, rows)
    return _own_copy(phases_modulo(m, reduced.constraints) if modulo else m)


def draw_code_3d(g: BaseGraph, qubits: Sequence[Hashable], coords: Mapping[Hashable, Sequence[float]],
                 faces: Sequence[Sequence[Hashable]], view: str | tuple[float, float] | None = None, *,
                 split_state: bool = True, y_down: bool = False, regular: bool = False, **kwargs: Any) -> Any:
    """:func:`code_layout` and :meth:`CodeLayout.draw` in one call. Returns the figure.

    Keyword arguments go to :meth:`CodeLayout.draw` (``ax``, ``figsize``, ``names``,
    ``phases``, ``title``, ...).
    """
    return code_layout(g, qubits, coords, faces, split_state=split_state, y_down=y_down,
                       regular=regular).draw(view, **kwargs)


# -- helpers --------------------------------------------------------------------------

def _own_copy(g: BaseGraph) -> BaseGraph:
    """A clone of ``g`` whose vertices have their own data.

    PyZX's ``clone`` copies the table of vertex data, but not the data of each
    vertex: the clone and ``g`` share it, and writing a ``timestep`` into the clone
    writes it into ``g`` too. Here each vertex gets a copy of its own.
    """
    h = g.clone()
    for v in h.vertices():
        h.set_vdata_dict(v, dict(g.vdata_dict(v)))
    return h


def _in_order(text: str) -> str:
    """A parity such as ``oz8 + c[14] + 1 + c[4]`` written in order: ``c[4] + c[14] + oz8 + 1``.

    Only a sum of plain variable names and ``1`` is reordered; any other text is
    returned as it is.
    """
    parts = text.split(' + ')
    if len(parts) < 2 or not all(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*(\[\d+\])?|1', t) for t in parts):
        return text

    def key(t: str) -> tuple[Any, ...]:
        return (t == '1',) + tuple(int(x) if x.isdigit() else x for x in re.findall(r'\d+|\D+', t))
    return ' + '.join(sorted(parts, key=key))


def _constant(p: Any) -> Any:
    """The number in a phase: the phase itself, or the constant term of a polynomial (0 if none)."""
    if isinstance(p, Poly):
        return sum((c for c, t in p.terms if not t.vars), Fraction(0))
    return p


def _in_span(rows: np.ndarray, support: frozenset[int], n: int) -> bool:
    """Whether the 0/1 vector on ``support`` is a sum of ``rows`` (over GF(2))."""
    v = np.zeros(n, dtype=np.uint8)
    v[sorted(support)] = 1
    return _rank(np.vstack([rows, v])) == _rank(rows)


def _rank(a: np.ndarray) -> int:
    """The rank of a 0/1 matrix over GF(2)."""
    m = (np.array(a, dtype=np.uint8) % 2).copy()
    r = 0
    for c in range(m.shape[1] if m.ndim == 2 else 0):
        pivots = np.flatnonzero(m[r:, c]) + r
        if not len(pivots):
            continue
        m[[r, pivots[0]]] = m[[pivots[0], r]]
        for i in np.flatnonzero(m[:, c]):
            if i != r:
                m[i] ^= m[r]
        r += 1
        if r == m.shape[0]:
            break
    return r


def _spacing(pts: np.ndarray) -> float:
    """The usual distance between neighbouring qubits: the median distance to the nearest other qubit."""
    if len(pts) < 2:
        return 1.0
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    s = float(np.median(d.min(axis=1)))
    return s if s > 1e-9 else 1.0


def _outside(xy: np.ndarray, support: list[int], centre: np.ndarray,
             spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """A place outside the layout next to the qubits ``support``, and the unit vector towards it.

    When the qubits lie on a line (a side of a triangle), the place is beyond the
    middle of the line, along its normal, away from the middle of the layout;
    otherwise beyond the middle of the qubits, away from the middle of the layout.
    It is 2.2 steps beyond the farthest qubit that way, a step being the least
    distance between neighbouring qubits of ``support`` (``spacing`` if larger),
    so that it stands clear of the layout in the oblique view.
    """
    pts = xy[support]
    mid = pts.mean(axis=0)
    u = None
    if len(pts) >= 2:
        _, s, vt = np.linalg.svd(pts - mid)
        if len(s) < 2 or s[1] <= 0.35 * s[0]:
            u = vt[-1] if len(s) >= 2 else np.array([-vt[0][1], vt[0][0]])
            side = float(u @ (mid - centre))
            if side < 0 or (abs(side) < 1e-9 and u[0] > 0):
                u = -u
    if u is None:
        u = mid - centre
        if np.linalg.norm(u) < 1e-9 * spacing:
            u = np.array([-1.0, 0.0])
    u = u / np.linalg.norm(u)
    reach = float(max((xy - mid) @ u))
    step = spacing
    if len(pts) >= 2:
        line = np.sort(pts @ np.array([-u[1], u[0]]))
        gaps = np.diff(line)
        gaps = gaps[gaps > 1e-9]
        if len(gaps):
            step = max(spacing, float(gaps.min()))
    return mid + (reach + 2.2 * step) * u, u


def _spread(place: dict[Any, np.ndarray], height: dict[Any, int], movable: list[Any], fixed: list[Any],
            radius: float, by_height: bool, start: float = math.pi / 2, order: Any = str) -> None:
    """Move spiders that share a place onto a small circle around it.

    ``movable`` spiders in the same place (and, with ``by_height``, at the same height)
    are put on the circle, and so is a lone one whose place a ``fixed`` spider takes.
    Without ``by_height`` a place is taken at any height, so that spiders under one
    another can be told apart from above. A group goes round the circle from the
    angle ``start``, in the order ``order``; a lone spider goes to the lower right.
    """
    def key(v: Any) -> tuple[Any, ...]:
        return _key(place[v]) + ((height[v],) if by_height else ())

    taken = {key(v) for v in fixed}
    groups: dict[tuple[Any, ...], list[Any]] = {}
    for v in movable:
        groups.setdefault(key(v), []).append(v)
    for k, vs in groups.items():
        if len(vs) < 2 and k not in taken:
            continue
        c = place[vs[0]]
        first = -math.pi / 4 if len(vs) == 1 else start
        for i, v in enumerate(sorted(vs, key=order)):
            a = first + 2 * math.pi * i / len(vs)
            place[v] = c + radius * np.array([math.cos(a), math.sin(a)])


#: The sides a label can take, as screen angles: lower right first, as in the drawing module.
_SIDES = [math.radians(-45 + 45 * k) for k in range(8)]


def _place_labels_in(ax: Any, labels: list[tuple[Any, tuple[float, float, float], Any]], gap: float = 6) -> None:
    """Put each label on the side of its anchor where it covers least, the side it wants on a tie.

    ``labels`` are ``(text, anchor, want)``; ``want`` is ``'down'``, ``'lower right'``
    or ``('out', (dx, dy))``, a direction in the plane of the code, turned into a
    side on the screen as the axes are now seen. The choice itself is
    :func:`~zxqec.spacetime.drawing._place_labels`: it avoids the labels placed
    before, the spiders and the edges. ``gap`` is the distance from the anchor, in points.
    """
    from mpl_toolkits.mplot3d import proj3d

    from .drawing import _place_labels, _put_on_side

    proj = ax.get_proj()

    def screen(p: tuple[float, float, float]) -> np.ndarray:
        x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], proj)
        return np.asarray(ax.transData.transform((x, y)), dtype=float)

    drawn = ax.__dict__.setdefault('_zx_drawn', {'labels': [], 'points': [], 'segments': []})
    for text, anchor, want in labels:
        if want == 'down':
            angle = -math.pi / 2
        elif want == 'lower right':
            angle = -math.pi / 4
        else:
            dx, dy = want[1]
            d = screen((anchor[0] + dx, anchor[1] + dy, anchor[2])) - screen(anchor)
            angle = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-9 else -math.pi / 4
        side = min(_SIDES, key=lambda a: abs((a - angle + math.pi) % (2 * math.pi) - math.pi))
        _put_on_side(text, side, gap)
        drawn['labels'].append((text, anchor))
    _place_labels(ax, gap)


def _key(p: Any) -> tuple[float, float]:
    """A place, rounded, to compare places."""
    return (round(float(p[0]), 6), round(float(p[1]), 6))


def _colour_faces(faces: list[tuple[Hashable, ...]]) -> list[int]:
    """A colour per face, faces that share a qubit apart: three colours if that can be done (a colour code).

    Tries three colours by backtracking, the faces with most neighbours first;
    if three are not enough, colours greedily.
    """
    m = len(faces)
    near = [[j for j in range(m) if j != i and set(faces[i]) & set(faces[j])] for i in range(m)]
    order = sorted(range(m), key=lambda i: (-len(near[i]), i))
    colours = [-1] * m
    steps = [0]

    def fill(k: int) -> bool:
        if k == m:
            return True
        steps[0] += 1
        if steps[0] > 20000:
            return False
        i = order[k]
        for c in range(3):
            if all(colours[j] != c for j in near[i]):
                colours[i] = c
                if fill(k + 1):
                    return True
        colours[i] = -1
        return False

    if fill(0):
        return colours
    colours = [-1] * m
    for i in order:
        used = {colours[j] for j in near[i]}
        colours[i] = next(c for c in range(m + 1) if c not in used)
    return colours
