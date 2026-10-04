"""Walk through the levels of a program in a window, with simplification during level-up.

The stages of a program are grouped into blocks, each with a rewrite, and the
blocks levelled up once more into one block rewritten by ``'gpf'``. The window
(``zxqec.spacetime.view_levels``) shows the program with the blocks of a level
shaded and the selected block outlined, the blocks it was made from dashed; in
the middle the program as that level rewrites it (every block replaced by its
rewritten diagram, ``reduced_level``); and on the right the selected block's
diagram after its rewrite (tick *plain diagram* for the diagram cut out of the
circuit). The slider moves between
levels, the buttons (or the left/right arrow keys) between blocks; going down a
level shows what a block was made of. Drag a spider to move it when labels
overlap; *spread out block* lays a crowded block out with PyZX's force-directed
layout, and *reset positions* undoes the moves. With ``--3d`` the program and
the level are drawn in 3D (``view_levels_3d``: space flat, time upwards, the
blocks as slabs, the spiders of a rewritten block laid out between the ends of
its wires); drag them to turn them, or use the view buttons. Run from a terminal:

    python examples/interactive_levels.py                  # the fold-transversal Hadamard
    python examples/interactive_levels.py --cultivation    # d = 3 magic-state cultivation
    python examples/interactive_levels.py --3d             # the program in 3D
    python examples/interactive_levels.py --save fig.png   # write the window to a file instead
"""

import argparse
import os
import sys

import matplotlib

try:                                              # as in interactive_fold_h_3d.py: Tk opens reliably
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt

from zxqec.qprogram import program_from_stim
from zxqec.spacetime import view_levels, view_levels_3d

HERE = os.path.dirname(os.path.abspath(__file__))

FOLD_GROUPS = [(0, 0, 'encode + round 1', 'encoder', 'gpf'), (1, 1, 'transversal H', 'logical gate'),
               (2, 3, 'round in code 2', 'syndrome extraction', 'gpf'), (4, 4, 'rotate', 'logical gate'),
               (5, 6, 'round in code 1', 'syndrome extraction', 'full_reduce'),
               (7, 7, 'checks + decode', 'decoder', 'gpf')]
CULTIVATION_GROUPS = [(0, 0, 'first checks', 'state preparation', 'gpf'), (1, 1, 'inject T', 'injection', 'gpf'),
                      (2, 2, 'checks', 'syndrome extraction', 'gpf'),
                      (3, 4, 'double-check', 'cultivation', 'gpf')]


def build(cultivation: bool):
    """The program, grouped into blocks with rewrites and levelled up into one block rewritten by 'gpf'."""
    if cultivation:
        with open(os.path.join(HERE, 'circuits', 'cultivation', 'd3_cultivation.stim')) as f:
            p = program_from_stim(f.read())[0]
        groups = CULTIVATION_GROUPS
    else:
        sys.path.insert(0, HERE)
        import interactive_fold_h_3d as fth
        p = fth.build_program()
        groups = FOLD_GROUPS
    level = p.group_stages(groups)
    p.level_up(level, 0, len(groups) - 1, name='whole program', rewrite='gpf')
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cultivation", action="store_true", help="d = 3 cultivation instead of the fold-transversal H")
    ap.add_argument("--3d", dest="three_d", action="store_true", help="draw the program in 3D")
    ap.add_argument("--save", metavar="FILE", help="save the window to FILE instead of opening it")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "or use --save FILE")
    p = build(args.cultivation)
    print(p.describe_reduced(p.levels.num_levels - 2))
    print(p.describe_reduced(p.levels.num_levels - 1))
    fig = view_levels_3d(p) if args.three_d else view_levels(p)
    if args.save:
        fig.savefig(args.save, dpi=110)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
