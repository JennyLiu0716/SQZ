"""Open the 2+1D view of case study 1 (fold-transversal Hadamard) in an interactive window.

The protocol is the one of ``FoldTransversalHadamard.ipynb``: the 19-qubit
distance-3 surface code, an X-check round, a transversal Hadamard, a QEC round
of the dual code, the 90-degree rotation of the data grid and QEC rounds of the
original code with the decoding measurements. Run from a terminal (the window
blocks until it is closed):

    python examples/interactive_fold_h_3d.py                 # whole protocol, oblique view, one slab per stage
    python examples/interactive_fold_h_3d.py top             # start from the bird's-eye view (also: front, side)
    python examples/interactive_fold_h_3d.py --level 2       # slabs = the five phases of the protocol
    python examples/interactive_fold_h_3d.py --stage 4       # one stage only: the rotation
    python examples/interactive_fold_h_3d.py --block 2 2     # one block only: block 2 of level 2 (the QEC round of code 2)
    python examples/interactive_fold_h_3d.py --phases        # label the spiders with their phases (outcome variables)
    python examples/interactive_fold_h_3d.py --save fig.png  # write the figure to a file instead of opening a window

Mouse controls (matplotlib's standard 3D navigation):

* rotate: left-click and drag -- the current ``elev`` / ``azim`` are printed in
  the terminal while dragging; the buttons below the plot turn the figure
  straight to the named viewpoints (``default`` = oblique, ``top`` = bird's-eye,
  ``front``, ``side``);
* zoom: right-click and drag, or scroll (two-finger scroll on a trackpad);
* pan: middle-click and drag, or the pan tool in the toolbar; the toolbar's
  home button resets the view.

The spatial layout is the thesis's: the 3x3 data grid ``q[1..9]`` with the
X-check ancillas ``q[10..13]`` and Z-check ancillas ``q[14..17]`` between the
plaquettes, the logical input ``q[0]`` and the decode ancilla ``q[18]`` on the
left (see ``circuits/ft_h/README.md`` for the qubit roles).
"""

import argparse
import os
import sys

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

import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import HAD, Reset
from zxqec.qprogram import QProgram
from zxqec.spacetime import VIEWS, Topology, draw_graph_3d, draw_program_3d

HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUITS = os.path.join(HERE, "circuits", "ft_h")

N, BITS = 19, 35
X_ANC, Z_ANC = [10, 11, 12, 13], [14, 15, 16, 17]

# thesis layout of the 19 qubits: (x, y) of every physical wire
COORDS = {0: (0, 0), 1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (1, 2), 5: (2, 2), 6: (3, 2), 7: (1, 3), 8: (2, 3), 9: (3, 3),
          10: (2.5, 0.5), 11: (1.5, 1.5), 12: (2.5, 2.5), 13: (1.5, 3.5),
          14: (0, 1.5), 15: (2.5, 1.5), 16: (1.5, 2.5), 17: (3.5, 2.5), 18: (0, 4)}
# hardware edges: every check ancilla to the data qubits of its plaquette, plus the injection and decoding wires
EDGES = [(1, 11), (2, 11), (4, 11), (5, 11), (2, 10), (3, 10), (5, 12), (6, 12), (8, 12), (9, 12), (7, 13), (8, 13),
         (1, 14), (4, 14), (2, 15), (3, 15), (5, 15), (6, 15), (4, 16), (5, 16), (7, 16), (8, 16), (6, 17), (9, 17),
         (0, 1), (0, 4), (0, 7), (7, 18), (8, 18), (9, 18)]


def load(name: str) -> Circuit:
    return zx.Circuit.load(os.path.join(CIRCUITS, name + ".qasm"))


def with_prefix(circ: Circuit, gates: list) -> Circuit:
    """``gates`` (ancilla preparations) followed by ``circ``."""
    c = Circuit(N, bit_amount=BITS)
    for g in gates:
        c.add_gate(g)
    c.add_circuit(circ)
    return c


def build_topology() -> Topology:
    topo = Topology(2)
    for n, c in COORDS.items():
        topo.add_node(n, c)
    for a, b in EDGES:
        topo.add_edge(a, b)
    return topo


def build_program() -> QProgram:
    """The protocol as in the notebook, plus two levels grouping the stages into the protocol's phases."""
    p = QProgram(N, bits=BITS, topology=build_topology(),
                 initial_states={**{q: '0' for q in range(1, 10)}, 18: '0', **{a: '+' for a in X_ANC + Z_ANC}})
    p.add_circuit(load("ft_h"), name="encode + X checks")
    p.add_circuit(load("ft_h_2"), name="transversal H")
    p.add_circuit(with_prefix(load("ft_h_3"), [Reset(a) for a in X_ANC]), name="Z checks (code 2)")
    p.add_circuit(load("ft_h_3.2"), name="X checks (code 2)")
    p.add_permutation([0, 3, 6, 9, 2, 5, 8, 1, 4, 7] + list(range(10, 19)), name="rotate 90°")
    p.add_circuit(with_prefix(load("ft_h_5"), [Reset(a) for a in Z_ANC]), name="Z checks", physical=True)
    p.add_circuit(with_prefix(load("ft_h_5.2"), [g for a in X_ANC for g in (Reset(a), HAD(a))]), name="X checks", physical=True)
    p.add_circuit(with_prefix(load("ft_h_6"), [Reset(a) for a in Z_ANC]), name="Z checks + decode", physical=True)

    # level 1: the two check rounds of code 2 form one QEC round;
    # level 2: the check rounds of code 1 after the rotation and the decoding form one block
    p.level_up(0, 2, 3, name="QEC round (code 2)")
    p.level_up(1, 4, 6, name="QEC rounds (code 1) + decode")
    p.graph()                                  # build the master graph: schedules the stages (sets their ticks)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("view", nargs="?", default="default", choices=sorted(VIEWS), help="starting viewpoint")
    ap.add_argument("--level", type=int, default=0, help="level whose blocks are shaded (default 0: one slab per stage)")
    ap.add_argument("--stage", type=int, metavar="I", help="draw only the sub-diagram of stage I")
    ap.add_argument("--block", type=int, nargs=2, metavar=("LEVEL", "I"), help="draw only the sub-diagram of block I of LEVEL")
    ap.add_argument("--phases", action="store_true", help="label the spiders with their phases")
    ap.add_argument("--save", metavar="FILE", help="save the figure to FILE instead of opening a window")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "with a GUI backend (e.g. TkAgg, QtAgg), or pass --save FILE")

    p = build_program()
    for level in range(p.levels.num_levels):
        print("level {}:\n{}".format(level, p.describe_level(level)))

    kw = dict(figsize=(10, 9), view=args.view, show_phases=args.phases, buttons=args.save is None)
    if args.stage is not None:
        st = p.stage(args.stage)
        fig = draw_graph_3d(p.stage_graph(args.stage, classical=False), p.topology, blocks=[(str(st), [st.ticks])], **kw)
    elif args.block is not None:
        level, i = args.block
        b = p.block(level, i)
        ranges = [p.stage(s).ticks for s in b.stages]
        fig = draw_graph_3d(p.block_graph(level, i, classical=False), p.topology, blocks=[(str(b), ranges)], **kw)
    else:
        fig = draw_program_3d(p, level=args.level, **kw)

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
