"""The fold-transversal Hadamard in the levels window, every block but the Hadamard and the turn in GPF.

The program is the thesis's (``build_program`` in interactive_fold_h_3d.py), with one
level on top of its stages: each stage is a block of its own, and every block is
rewritten by ``'gpf'`` except the transversal Hadamard and the quarter turn of the
grid (the permutation), which stay as they are; the Z checks of code 2 and the Z
checks after the turn (blocks 2 and 5) by ``'x-z gpf'``, the normal form with the
colours swapped. Each qubit is prepared in the first stage that uses it (the Z
ancillas in the X checks of code 2, qubit 18 in the last stage), so no identity
wire runs through the blocks before. The window is
``zxqec.spacetime.view_levels_3d``, as in interactive_fold_h_levels_3d.py: on the
left the program on the layout of its 19 qubits, the 3x3 grid flat and time
upwards, with the blocks of the level as slabs; in the middle the program as that
level rewrites it; on the right the selected block. Drag the 3D pictures to turn
them, or use the view buttons; the block buttons (or the left/right arrow keys)
move between blocks. Run from a terminal:

    python examples/interactive_fold_h_gpf_3d.py               # the program
    python examples/interactive_fold_h_gpf_3d.py top           # start from the bird's-eye view (also: front, side)
    python examples/interactive_fold_h_gpf_3d.py --miswired    # with the thesis files' wiring error
    python examples/interactive_fold_h_gpf_3d.py --rewrite "X checks=x-z gpf"   # one stage in X-Z form
    python examples/interactive_fold_h_gpf_3d.py --save fig.png   # write the window to a file instead
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
import interactive_fold_h_3d as fth  # noqa: E402

KEEP = ("transversal H", "rotate 90°")           # the stages left as they are
X_Z = ("Z checks (code 2)", "Z checks")          # blocks 2 and 5: in X-Z normal form


def build(miswired: bool, rewrite: str = 'gpf', per_stage: dict[str, str] | None = None,
          places: dict[int, tuple[float, float]] | None = None):
    """The program with one level: each stage a block, ``rewrite`` on all but ``KEEP``; returns it and the level.

    The stages in ``X_Z`` take the X-Z form of ``rewrite`` (``'x-z gpf'`` for ``'gpf'``). ``per_stage`` gives
    some stages, by name, a rewrite of their own (e.g. ``{'X checks': 'x-z gpf'}``). ``places`` draws some qubits
    elsewhere in the pictures (``build_program``).
    """
    p = fth.build_program(miswired=miswired, prepare_when_used=True,    # no idle wires before first use
                          places=places)
    names = [str(p.stage(i)) for i in range(p.num_stages)]
    per_stage = {**{name: 'x-z ' + rewrite for name in X_Z}, **(per_stage or {})}
    unknown = set(per_stage) - set(names)
    if unknown:
        raise ValueError("no stage named {}; the stages: {}".format(sorted(unknown), names))
    groups = [(i, i, name, None, per_stage.get(name, None if name in KEEP else rewrite))
              for i, name in enumerate(names)]
    level = p.group_stages(groups, one_level=True)
    return p, level


def stage_rewrites(pairs: list[str]) -> dict[str, str]:
    """``["X checks=x-z gpf", ...]`` from the command line as ``{'X checks': 'x-z gpf', ...}``."""
    out = {}
    for pair in pairs:
        stage, sep, rewrite = pair.partition("=")
        if not sep:
            raise SystemExit("--rewrite takes STAGE=REWRITE, got {!r}".format(pair))
        out[stage.strip()] = rewrite.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("view", nargs="?", default="default", choices=sorted(VIEWS), help="starting viewpoint")
    ap.add_argument("--miswired", action="store_true", help="the thesis files' wiring error in the round after the rotation")
    ap.add_argument("--rewrite", action="append", default=[], metavar="STAGE=REWRITE",
                    help="rewrite one stage with its own rewrite, e.g. \"X checks=x-z gpf\" (repeatable)")
    ap.add_argument("--save", metavar="FILE", help="save the window to FILE instead of opening it")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "or use --save FILE")
    p, level = build(args.miswired, per_stage=stage_rewrites(args.rewrite))
    print(p.describe_reduced(level))
    fig = view_levels_3d(p, level, 0, view=args.view, by_outcomes=True,
                         phases_beside=True, hadamard_boxes=True,
                         crossing_wires=True)
    if args.save:
        fig.savefig(args.save, dpi=110)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
