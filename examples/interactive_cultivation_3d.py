"""Open the reduced d = 3 cultivation circuit on three planes in an interactive 3D window.

The diagram is step 4 of ``Cultivation.ipynb`` (Part 4): the d = 3 cultivation
circuit ``circuits/cultivation/d3_cultivation.stim``, reduced by ``reduce_gpf``
and written in the variables of its code. It is drawn with ``draw_graph_3d`` on
three heights:

* 0: the non-Clifford single-qubit states. The 7pi/4 of the logical spider is
  split off as its own one-legged Z-spider (spider fusion read backwards): the
  state T-dag|+>;
* 1: the Z-layer of the normal form, each spider above the qubits it is wired to;
* 2: the seven output X-spiders, on the textbook triangle of the Steane code
  (the outputs end at height 3).

The layout is a ``Topology``, as for the fold-transversal Hadamard: the seven
qubits of the triangle with the sides of its faces, and a node in the middle of
each face. The circuit's code qubits are put on the triangle by the first of the
orderings that put the three spiders wired to four qubits on the three faces;
those spiders sit above the middle of their face, and the logical spider and its
state outside the triangle, next to the side {0, 1, 3} they are wired to. Run
from a terminal (the window blocks until it is closed):

    python examples/interactive_cultivation_3d.py                # oblique view, the run where every outcome is 0
    python examples/interactive_cultivation_3d.py top            # start from the bird's-eye view (also: front, side)
    python examples/interactive_cultivation_3d.py --outcomes     # keep the outcome variables in the phases
    python examples/interactive_cultivation_3d.py --save fig.png # write the figure to a file instead of opening a window

Mouse controls (matplotlib's standard 3D navigation):

* rotate: left-click and drag -- the current ``elev`` / ``azim`` are printed in
  the terminal while dragging; the buttons below the plot turn the figure
  straight to the named viewpoints (``default`` = oblique, ``top`` = bird's-eye,
  ``front``, ``side``);
* zoom: right-click and drag, or scroll (two-finger scroll on a trackpad);
* pan: middle-click and drag, or the pan tool in the toolbar; the toolbar's
  home button resets the view.
"""

import argparse
import itertools
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

from pyzx.spacetime import set_timestep
from pyzx.utils import VertexType
from zxqec.gpf import boolean_variables, change_variables, reduce_gpf, reduce_modulo
from zxqec.qprogram import program_from_stim
from zxqec.spacetime import VIEWS, Topology, draw_graph_3d

from teaching_figures import FACES, POS

HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUIT = os.path.join(HERE, "circuits", "cultivation", "d3_cultivation.stim")
SIZE = 2.2                                   # the triangle's scale in the plot
LOGICAL = 10                                 # the node of the logical spider and its state:
LOGICAL_XY = (-1.5, 0.5)                     # outside the triangle, left of the side {0, 1, 3}


def build_step4():
    """The circuit's diagram, its reduction, the code after the double-check, and step 4."""
    p, _ = program_from_stim(open(CIRCUIT).read())
    g = p.quantum_graph()
    res = reduce_gpf(g)
    code = p.code_at(p.num_stages - 1).data()
    rows = [list(code.logical_x[0].x)] + [list(r) for r in code.sx]
    return g, res, code, change_variables(res.graph, rows)


def build_topology() -> Topology:
    """The triangle: qubits 0..6 with the sides of the faces, node 7 + i in the middle of face i, and LOGICAL."""
    topo = Topology(2)
    for q in range(7):
        topo.add_node(q, (SIZE * POS[q][0], SIZE * POS[q][1]))
    for i, f in enumerate(FACES):
        for a, b in zip(f, f[1:] + f[:1]):
            if not topo.are_adjacent(a, b):
                topo.add_edge(a, b)
        topo.add_node(7 + i, tuple(sum(SIZE * POS[q][k] for q in f) / 4 for k in range(2)))
    topo.add_node(LOGICAL, LOGICAL_XY)
    return topo


def on_triangle(step4, code):
    """The orderings of the code qubits onto the triangle that put the weight-4 Z-spiders on the faces."""
    qubits = list(code.qubits)
    out_x = [next(iter(step4.neighbors(b))) for b in step4.outputs()]
    zs = [v for v in step4.vertices() if step4.type(v) == VertexType.Z]
    wired = [frozenset(j for j, w in enumerate(out_x) if step4.connected(z, w)) for z in zs]
    faces = {frozenset(f) for f in FACES}
    return [dict(zip(qubits, perm)) for perm in itertools.permutations(range(7))
            if {frozenset(perm[j] for j in w) for w in wired if len(w) == 4} == faces]


def on_planes(step4, code, on):
    """A copy of step 4 with every spider on a node of the triangle and its height as timestep."""
    qubits = list(code.qubits)
    faces = [frozenset(f) for f in FACES]
    h = step4.copy()
    xs = [next(iter(h.neighbors(b))) for b in h.outputs()]
    for j, (b, x) in enumerate(zip(h.outputs(), xs)):
        h.set_qubit(b, on[qubits[j]])
        h.set_qubit(x, on[qubits[j]])
        set_timestep(h, x, 2)
    for z in [v for v in h.vertices() if h.type(v) == VertexType.Z]:
        tri = frozenset(on[qubits[j]] for j, x in enumerate(xs) if h.connected(z, x))
        set_timestep(h, z, 1)
        if tri in faces:
            h.set_qubit(z, 7 + faces.index(tri))                      # above the middle of its face
        else:                                                         # the logical spider: outside the triangle
            h.set_qubit(z, LOGICAL)
            const = Fraction(7, 4)
            state = h.add_vertex(VertexType.Z, qubit=LOGICAL, phase=const)  # T-dag|+>, split off the spider
            h.add_edge((state, z))
            h.set_phase(z, h.phase(z) - const)
            set_timestep(h, state, 0)
    return h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("view", nargs="?", default="default", choices=sorted(VIEWS), help="starting viewpoint")
    ap.add_argument("--outcomes", action="store_true",
                    help="keep the outcome variables in the phases (default: the run where every outcome is 0)")
    ap.add_argument("--save", metavar="FILE", help="save the figure to FILE instead of opening a window")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "with a GUI backend (e.g. TkAgg, QtAgg), or pass --save FILE")

    g, res, code, step4 = build_step4()
    orderings = on_triangle(step4, code)
    on = orderings[0]
    print("{} orderings put the three weight-4 spiders on the three faces; using circuit qubit -> triangle qubit {}"
          .format(len(orderings), on))
    out_x = [next(iter(step4.neighbors(b))) for b in step4.outputs()]
    for z in sorted(v for v in step4.vertices() if step4.type(v) == VertexType.Z):
        wired = [q for j, q in enumerate(code.qubits) if step4.connected(z, out_x[j])]
        print("  Z-spider {:42s} circuit qubits {:16s} triangle qubits {}".format(
            str(reduce_modulo(step4.phase(z), res.constraints)), str(wired), sorted(on[q] for q in wired)))

    shown = on_planes(step4, code, on)
    if not args.outcomes:
        variables = set(boolean_variables(g)) | {str(v) for c in res.constraints for v in c.free_vars()}
        shown = shown.substitute_variables({n: 0 for n in variables})
    fig = draw_graph_3d(shown, build_topology(), figsize=(9, 8), view=args.view, buttons=args.save is None)
    fig.axes[0].set_title("reduced d = 3 cultivation (step 4): non-Clifford states (0), Z-layer (1), "
                          "X-spiders on the Steane triangle (2)", fontsize=9)

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
