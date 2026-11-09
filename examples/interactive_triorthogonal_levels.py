"""Transversal T on the [[15,1,3]] code in the interactive levels window.

The program of TriorthogonalStepByStep.ipynb: the [[15,1,3]] encoder (the stored
qubit comes in on qubit 14), a layer of T gates on all fifteen data qubits (or,
with ``--seven``, on qubits 0-6 only), and a round of the four X checks measured
by helpers 15-18. Three levels, each rewritten by ``'gpf'``: level 1 the encoder,
level 2 the encoder and the T layer, where the fifteen T gates add up to one
gadget with phase 7pi/4 on the logical spider (the logical T^dag), level 3 the
whole program, with the four checks gone into detectors. The window
(``zxqec.spacetime.view_levels``) shows the program with the blocks of a level
shaded and the selected block outlined, the blocks it was made from dashed; in
the middle the program as that level rewrites it; on the right the selected
block after its rewrite. The slider moves between levels, the buttons (or the
left/right arrow keys) between blocks. Run from a terminal:

    python examples/interactive_triorthogonal_levels.py              # T on all fifteen qubits
    python examples/interactive_triorthogonal_levels.py --seven      # T on qubits 0-6 only
    python examples/interactive_triorthogonal_levels.py --save fig.png   # write the window to a file instead
"""

import argparse
import sys

import matplotlib

try:                                              # as in interactive_levels.py: Tk opens reliably
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt
from pyzx.circuit import Circuit

from zxqec.qprogram import QProgram
from zxqec.spacetime import view_levels

N = 19                                            # data qubits 0-14, helpers 15-18
HEAD = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[19];\ncreg c[4];\n'
# qubit j stands for the number j + 1 in binary; check b is X on the qubits whose number has binary digit b set.
# The encoder copies the input's X onto the weight-7 logical X (the numbers with an even number of ones) and
# builds each check from the qubit whose number is a power of two.
ENCODER = "".join("cx q[14], q[{}];\n".format(j) for j in range(14) if bin(j + 1).count('1') % 2 == 0)
for _b in range(4):
    _pivot = (1 << _b) - 1
    ENCODER += "h q[{}];\n".format(_pivot) + "".join("cx q[{}], q[{}];\n".format(_pivot, j) for j in range(15)
                                                     if (j + 1) >> _b & 1 and j != _pivot)
X_CHECKS = "".join("h q[{h}];\n".format(h=15 + b) + "".join("cx q[{}], q[{}];\n".format(15 + b, j) for j in range(15)
                                                            if (j + 1) >> b & 1)
                   + "h q[{h}];\nmeasure q[{h}] -> c[{b}];\n".format(h=15 + b, b=b) for b in range(4))


def build(seven: bool):
    """The program, levelled up three times with ``'gpf'``; returns it and its three levels."""
    p = QProgram(N, bits=4, initial_states={q: '0' for q in range(N) if q != 14})
    p.add_circuit(Circuit.from_qasm(HEAD + ENCODER), name='encoder')
    p.add_circuit(Circuit.from_qasm(HEAD + "".join("t q[{}];\n".format(j) for j in (range(7) if seven else range(15)))),
                  name='T layer')
    p.add_circuit(Circuit.from_qasm(HEAD + X_CHECKS), name='X checks')
    level1 = p.level_up(0, 0, 0, 'encoder', 'encoder', rewrite='gpf')
    level2 = p.level_up(level1, 0, 1, 'encoder + T layer', 'state preparation', rewrite='gpf')
    level3 = p.level_up(level2, 0, 1, 'whole program', 'state preparation', rewrite='gpf')
    return p, (level1, level2, level3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seven", action="store_true", help="T on qubits 0-6 only")
    ap.add_argument("--save", metavar="FILE", help="save the window to FILE instead of opening it")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "or use --save FILE")
    p, levels = build(args.seven)
    for level in levels:
        print(p.describe_reduced(level))
    fig = view_levels(p, levels[1], 0)
    if args.save:
        fig.savefig(args.save, dpi=110)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
