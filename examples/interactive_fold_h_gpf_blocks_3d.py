"""The fold-transversal Hadamard with every block but the Hadamard and the turn in GPF, each block in 3D.

The same program and level as interactive_fold_h_gpf_3d.py: each stage is a block
of its own, and every block except the transversal Hadamard and the quarter turn
of the grid is rewritten, here by ``'gpf_checks'``: the GPF, with the checks of a
round one per spider, each spider's phase the outcome of its check and its edges
to that check's qubits. Blocks 2 and 5 (the Z checks) are in X-Z form, by
``'x-z gpf_checks'``. The window is ``zxqec.spacetime.view_levels_3d`` with
``block_3d=True``: the selected block on the right is in 3D too, also after its
rewrite. The ends of a rewritten block's wires are where the wires enter and leave
the block; each check is where its outcome is measured, on its ancilla at the
time of the measurement (``by_outcomes=True``), so from above a round looks like
the protocol's plaquettes; every other spider is over the wire ends nearest to
it, at a time set by how far it is from the inputs and from the outputs
(``zxqec.spacetime.draw_block_3d``). Run from a terminal:

    python examples/interactive_fold_h_gpf_blocks_3d.py               # the program
    python examples/interactive_fold_h_gpf_blocks_3d.py top           # start from the bird's-eye view (also: front, side)
    python examples/interactive_fold_h_gpf_blocks_3d.py --miswired    # with the original files' wiring error
    python examples/interactive_fold_h_gpf_blocks_3d.py --rewrite "X checks=x-z gpf_checks"   # one stage in X-Z form
    python examples/interactive_fold_h_gpf_blocks_3d.py --save fig.png   # write the window to a file instead
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
import interactive_fold_h_gpf_3d as gpf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("view", nargs="?", default="default", choices=sorted(VIEWS), help="starting viewpoint")
    ap.add_argument("--miswired", action="store_true", help="the original files' wiring error in the round after the rotation")
    ap.add_argument("--rewrite", action="append", default=[], metavar="STAGE=REWRITE",
                    help="rewrite one stage with its own rewrite, e.g. \"X checks=x-z gpf_checks\" (repeatable)")
    ap.add_argument("--save", metavar="FILE", help="save the window to FILE instead of opening it")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "or use --save FILE")
    p, level = gpf.build(args.miswired, rewrite='gpf_checks', per_stage=gpf.stage_rewrites(args.rewrite))
    print(p.describe_reduced(level))
    fig = view_levels_3d(p, level, 0, view=args.view, block_3d=True, by_outcomes=True,
                         phases_beside=True, hadamard_boxes=True,
                         crossing_wires=True)
    if args.save:
        fig.savefig(args.save, dpi=110)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
