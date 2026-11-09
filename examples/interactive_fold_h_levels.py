"""The fold-transversal Hadamard on the surface code, a program that changes codes, in the interactive levels window.

The program is the thesis's (``build_program`` in interactive_fold_h_3d.py): the
input copied into the distance-3 surface code with a round of X checks, a
Hadamard on every data qubit (which turns the code into the one with X and Z
swapped), a round of checks of that code, a quarter turn of the grid (which
brings the code back to the surface code on the chip), a round of checks of the
surface code, and a last round that copies the stored qubit out. Three levels,
each rewritten by ``'gpf'``: level 1 the round in code 2, level 2 the round in
code 1 as well, level 3 the whole program, which becomes a Hadamard followed by
Paulis that the results determine. The window (``zxqec.spacetime.view_levels``)
shows the program with the blocks of a level shaded and the selected block
outlined, the blocks it was made from dashed; in the middle the program as that
level rewrites it; on the right the selected block after its rewrite. The slider
moves between levels, the buttons (or the left/right arrow keys) between blocks.
Run from a terminal:

    python examples/interactive_fold_h_levels.py               # the program
    python examples/interactive_fold_h_levels.py --miswired    # with the thesis files' wiring error
    python examples/interactive_fold_h_levels.py --save fig.png   # write the window to a file instead
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

from zxqec.spacetime import view_levels

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interactive_fold_h_3d as fth  # noqa: E402


def build(miswired: bool):
    """The program, levelled up three times with ``'gpf'``; returns it and its three levels."""
    p = fth.build_program(miswired=miswired)
    level1 = p.level_up(0, 2, 3, 'round in code 2', 'syndrome extraction', rewrite='gpf')    # stages 2-3
    level2 = p.level_up(level1, 4, 5, 'round in code 1', 'syndrome extraction', rewrite='gpf')  # stages 5-6
    level3 = p.level_up(level2, 0, p.levels.level_size(level2) - 1, 'whole program', 'logical gate', rewrite='gpf')
    return p, (level1, level2, level3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--miswired", action="store_true", help="the thesis files' wiring error in the round after the rotation")
    ap.add_argument("--save", metavar="FILE", help="save the window to FILE instead of opening it")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "or use --save FILE")
    p, levels = build(args.miswired)
    for level in levels:
        print(p.describe_reduced(level))
    fig = view_levels(p, levels[-1], 0)
    if args.save:
        fig.savefig(args.save, dpi=110)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
