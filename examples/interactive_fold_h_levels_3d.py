"""The fold-transversal Hadamard in the levels window, with the program in 3D.

The same program and levels as interactive_fold_h_levels.py (level 1 the round in
code 2 rewritten by ``'gpf'``, level 2 the round in code 1 as well, level 3 the
whole program), in ``zxqec.spacetime.view_levels_3d``: on the left the program on
the layout of its 19 qubits, the 3x3 grid flat and time upwards, with the blocks
of the level as slabs; in the middle the program as that level rewrites it, in 3D
too and turning with the program; on the right the selected block. Drag the 3D
pictures to turn them, or use the view buttons; the slider moves between levels,
the block buttons (or the left/right arrow keys) between blocks. The quarter turn
is where the wires of the grid cross over to their new places. Run from a terminal:

    python examples/interactive_fold_h_levels_3d.py               # the program
    python examples/interactive_fold_h_levels_3d.py top           # start from the bird's-eye view (also: front, side)
    python examples/interactive_fold_h_levels_3d.py --miswired    # with the thesis files' wiring error
    python examples/interactive_fold_h_levels_3d.py --save fig.png   # write the window to a file instead
"""

import argparse
import os
import sys

import matplotlib

try:                                              # as in interactive_levels.py: Tk opens reliably
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt

from zxqec.spacetime import VIEWS, view_levels_3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interactive_fold_h_levels as levels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("view", nargs="?", default="default", choices=sorted(VIEWS), help="starting viewpoint")
    ap.add_argument("--miswired", action="store_true", help="the thesis files' wiring error in the round after the rotation")
    ap.add_argument("--save", metavar="FILE", help="save the window to FILE instead of opening it")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "or use --save FILE")
    p, lvls = levels.build(args.miswired)
    for level in lvls:
        print(p.describe_reduced(level))
    fig = view_levels_3d(p, lvls[-1], 0, view=args.view, by_outcomes=True)
    if args.save:
        fig.savefig(args.save, dpi=110)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
