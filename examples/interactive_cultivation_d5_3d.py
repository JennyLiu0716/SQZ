"""Open the reduced d = 5 cultivation circuit on three planes in an interactive 3D window.

The diagram is the one of ``Cultivation.ipynb`` (Part 6, "The result in 3D"): the
d = 5 cultivation circuit ``circuits/cultivation/d5_cultivation.stim``, reduced by
``reduce_gpf`` and written in the variables of the faces that the authors' final
check (``circuits/cultivation/d5_final_check.stim``) measures. It is drawn with
``draw_graph_3d`` on three heights:

* 0: the state T-dag|+>, split off the logical spider (spider fusion read backwards);
* 1: the Z-layer: one spider above the middle of each face, and the logical spider
  outside the triangle, beyond the side it is wired to;
* 2: the 19 output X-spiders, at the coordinates the circuit file gives the qubits
  (the outputs end at height 3).

With ``--textbook`` the window shows instead T-dag|+> fed into the colour code built
from the faces and the side alone, which the notebook compares with the reduced
diagram by tensor contraction. The phases are written modulo the detectors; none
depends on an outcome. Run from a terminal (the window blocks until it is closed):

    python examples/interactive_cultivation_d5_3d.py                # oblique view
    python examples/interactive_cultivation_d5_3d.py top            # start from the bird's-eye view (also: front, side)
    python examples/interactive_cultivation_d5_3d.py --textbook     # the colour code built from the faces
    python examples/interactive_cultivation_d5_3d.py --save fig.png # write the figure to a file instead

Mouse controls are matplotlib's standard 3D navigation, as in interactive_cultivation_3d.py.
"""

import argparse
import math
import os
import sys
from fractions import Fraction

import matplotlib

# The native "MacOSX" backend can block in plt.show() without opening a window
# when Python is launched from some terminals; Tk is reliable wherever it is
# installed, so prefer it and fall back to matplotlib's default otherwise.
try:
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt

from pyzx.graph import Graph
from pyzx.spacetime import set_timestep
from pyzx.utils import VertexType
from zxqec.gpf import change_variables, phases_modulo, reduce_gpf, simplify_gadgets
from zxqec.qprogram import parse_stim, program_from_stim
from zxqec.spacetime import VIEWS, Topology, draw_graph_3d

HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUIT = os.path.join(HERE, "circuits", "cultivation", "d5_cultivation.stim")
FINAL_CHECK = os.path.join(HERE, "circuits", "cultivation", "d5_final_check.stim")
SIDE = [0, 3, 9, 15, 23]                     # one side of the triangle: the logical X of the code
LOGICAL = 200                                # the node of the logical spider and its state
FACE = 100                                   # face i sits on node FACE + i


def faces():
    """The faces, as the first MPP of the final check measures them (each also as a product of Z)."""
    with open(CIRCUIT) as f, open(FINAL_CHECK) as g:
        circ = parse_stim(f.read() + g.read())
    first_mpp = next(ins for ins in circ.instructions if ins.name == 'MPP')
    return [[int(f[1:]) for f in t.split('*')] for t in first_mpp.targets if t.startswith('X')]


def build():
    """The program, its diagram and reduction, the code at the end, the faces, and the reduced diagram in their variables."""
    with open(CIRCUIT) as f:
        p, circ = program_from_stim(f.read())
    g = p.quantum_graph()
    res = reduce_gpf(g)
    code = p.code_at(p.num_stages - 1).data()
    qubits = [int(q) for q in code.qubits]
    fs = faces()
    rows = [[int(q in support) for q in qubits] for support in [SIDE] + fs]
    m = change_variables(res.graph, rows, fuse=False)
    while simplify_gadgets(m):                # the gadgets, added up in these variables
        pass
    m = phases_modulo(change_variables(m, rows), res.constraints)   # fuses a gadget left on one spider
    return p, circ, g, res, code, fs, m


def build_topology(circ, qubits, fs) -> Topology:
    """The code qubits at their coordinates, the sides of the faces, a node in each face, and LOGICAL."""
    xy = {q: (circ.coords[q][1], -circ.coords[q][0]) for q in qubits}
    middle = lambda qs: tuple(sum(xy[q][k] for q in qs) / len(qs) for k in range(2))
    topo = Topology(2)
    for q in qubits:
        topo.add_node(q, xy[q])
    for i, f in enumerate(fs):
        c = middle(f)
        ring = sorted(f, key=lambda q: math.atan2(xy[q][1] - c[1], xy[q][0] - c[0]))     # around the face
        for a, b in zip(ring, ring[1:] + ring[:1]):
            if not topo.are_adjacent(a, b):
                topo.add_edge(a, b)
        topo.add_node(FACE + i, c)
    side, rest = middle(SIDE), middle([q for q in qubits if q not in SIDE])
    topo.add_node(LOGICAL, tuple(side[k] + 2.4 * (side[k] - rest[k]) / math.dist(side, rest) for k in range(2)))
    return topo


def textbook(qubits, fs, angle=Fraction(7, 4)):
    """The colour code built from the faces and SIDE alone, with the state of this angle under the logical spider."""
    t = Graph()
    xs, bs = [], []
    for _q in qubits:                         # outputs in the circuit's order
        xs.append(t.add_vertex(VertexType.X))
        bs.append(t.add_vertex(VertexType.BOUNDARY))
        t.add_edge((xs[-1], bs[-1]))
    for support in fs + [SIDE]:
        s = t.add_vertex(VertexType.Z)
        for q in support:
            t.add_edge((s, xs[qubits.index(q)]))
    t.add_edge((t.add_vertex(VertexType.Z, phase=angle), s))      # s is the spider of the side
    t.set_inputs(())
    t.set_outputs(tuple(bs))
    return t


def on_planes(d, qubits, fs):
    """A copy of ``d`` with every spider on a node of the layout and its height as timestep."""
    d = d.copy()
    xs = [next(iter(d.neighbors(b))) for b in d.outputs()]
    for q, b, x in zip(qubits, d.outputs(), xs):
        d.set_qubit(b, q)
        d.set_qubit(x, q)
        set_timestep(d, x, 2)
    for z in [v for v in d.vertices() if d.type(v) == VertexType.Z and any(w in xs for w in d.neighbors(v))]:
        wired = sorted(qubits[j] for j, x in enumerate(xs) if d.connected(z, x))
        set_timestep(d, z, 1)
        if wired != sorted(SIDE):
            d.set_qubit(z, FACE + [sorted(f) for f in fs].index(wired))
            continue
        d.set_qubit(z, LOGICAL)
        if d.phase(z) != 0:                   # split the state off the logical spider
            d.add_edge((d.add_vertex(VertexType.Z, phase=d.phase(z)), z))
            d.set_phase(z, 0)
        for state in [w for w in d.neighbors(z) if d.vertex_degree(w) == 1]:
            d.set_qubit(state, LOGICAL)
            set_timestep(d, state, 0)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("view", nargs="?", default="default", choices=sorted(VIEWS), help="starting viewpoint")
    ap.add_argument("--textbook", action="store_true",
                    help="show T-dag|+> in the colour code built from the faces instead of the reduced circuit")
    ap.add_argument("--save", metavar="FILE", help="save the figure to FILE instead of opening a window")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "with a GUI backend (e.g. TkAgg, QtAgg), or pass --save FILE")

    _p, circ, _g, _res, code, fs, m = build()
    qubits = [int(q) for q in code.qubits]
    xs = [next(iter(m.neighbors(b))) for b in m.outputs()]
    for z in [v for v in m.vertices() if m.type(v) == VertexType.Z]:
        print("  Z-spider with phase {:4}  wired to qubits {}".format(
              str(m.phase(z)), [q for q, x in zip(qubits, xs) if m.connected(z, x)]))
    d = textbook(qubits, fs) if args.textbook else m
    fig = draw_graph_3d(on_planes(d, qubits, fs), build_topology(circ, qubits, fs), figsize=(9, 8),
                        view=args.view, buttons=args.save is None)
    fig.axes[0].set_title("T-dag|+> into the colour code of the faces" if args.textbook else
                          "reduced d = 5 cultivation in the faces' variables: the state (0), Z-layer (1), "
                          "X-spiders (2)", fontsize=9)

    if args.save is not None:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print("saved", args.save)
        return
    ax = fig.axes[0]
    fig.canvas.mpl_connect(
        'motion_notify_event',
        lambda event: print("\relev = {:6.1f}   azim = {:7.1f}".format(ax.elev, ax.azim), end=""))
    print("backend: {} -- drag to rotate, scroll to zoom; close the window to exit".format(plt.get_backend()))
    plt.show()


if __name__ == '__main__':
    main()
