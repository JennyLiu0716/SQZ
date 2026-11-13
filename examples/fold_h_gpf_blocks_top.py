"""Every block of the fold-transversal Hadamard after its rewrite, seen from a little off the top, in two rows.

The same program and level as interactive_fold_h_gpf_blocks_3d.py: each stage is a
block of its own, rewritten by ``'gpf_checks'`` (blocks 2 and 5, the Z checks, by
``'x-z gpf_checks'``), except the transversal Hadamard and the quarter turn of the
grid. ``zxqec.spacetime.draw_blocks_3d`` draws each block as the right-hand picture
of that window does, without the grey panes and the axes: the ends of the wires
where they enter and leave the block, each check on its ancilla
(``by_outcomes=True``), the phases beside the spiders and the Hadamards as yellow
boxes. The view is a little off the top, with time flattened (``time_scale``), so
the grid reads as in a picture from above while the wires in and out show as
short stubs, each at least a quarter of its block's time (``min_stub``). The
stored qubit is drawn at (4, 2), beside the right-hand column it
is encoded onto, so its edges do not cross the grid, and Z ancilla 14 at (0.5, 1.5),
half a step left of the grid like the ancillas on the other sides. Run from a
terminal:

    python examples/fold_h_gpf_blocks_top.py                    # open the picture in a window
    python examples/fold_h_gpf_blocks_top.py --save blocks.png  # write it to a file instead
    python examples/fold_h_gpf_blocks_top.py --elev 90          # straight from the top
    python examples/fold_h_gpf_blocks_top.py --columns 8        # all eight blocks in one row
    python examples/fold_h_gpf_blocks_top.py --miswired         # with the original files' wiring error
    python examples/fold_h_gpf_blocks_top.py --rewrite "X checks=x-z gpf_checks"   # one stage in another form
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

from zxqec.spacetime import draw_blocks_3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interactive_fold_h_gpf_3d as gpf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--miswired", action="store_true", help="the original files' wiring error in the round after the rotation")
    ap.add_argument("--rewrite", action="append", default=[], metavar="STAGE=REWRITE",
                    help="rewrite one stage with its own rewrite, e.g. \"X checks=x-z gpf_checks\" (repeatable)")
    ap.add_argument("--elev", type=float, default=70, help="angle above the chip, 90 straight from the top (default 70)")
    ap.add_argument("--azim", type=float, default=-80, help="angle around the chip (default -80)")
    ap.add_argument("--time-scale", type=float, default=0.5,
                    help="height of a block's time against the chip's width (default 0.5)")
    ap.add_argument("--columns", type=int, default=4, help="blocks in a row (default 4: two rows that fit a screen)")
    ap.add_argument("--save", metavar="FILE", help="save the picture to FILE instead of opening it")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "or use --save FILE")
    p, level = gpf.build(args.miswired, rewrite='gpf_checks', per_stage=gpf.stage_rewrites(args.rewrite),
                         places={0: (4, 2), 14: (0.5, 1.5)})
    fig = draw_blocks_3d(p, level, view=(args.elev, args.azim), time_scale=args.time_scale, by_outcomes=True, crossing_wires=True,
                         min_stub=0.25, columns=args.columns, phases_beside=True, hadamard_boxes=True, finish=False)
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
