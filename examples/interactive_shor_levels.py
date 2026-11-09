"""Shor's code levelled up twice, a code inside a code, in the interactive levels window.

The program of ShorCodeConcatenated.ipynb: Shor's encoder, one error, a round of
error correction for the inner code (ZZ checks and X corrections in each block)
and one for the outer code (X checks on two blocks and a Z correction on a
block). Level 1 groups the encoder, the error and the inner round into one
block rewritten by ``'gpf'``; level 2 groups everything into one block
rewritten by ``'gpf'``. The window (``zxqec.spacetime.view_levels``) shows the
program with the blocks of a level shaded and the selected block outlined, the
blocks it was made from dashed; in the middle the program as that level
rewrites it; on the right the selected block after its rewrite. The slider
moves between levels, the buttons (or the left/right arrow keys) between
blocks. For Z4, level 1 is Shor's encoder with a pi on the check on qubits 3-8
(the error, still there as a sign), and level 2 is Shor's encoder. Run from a
terminal:

    python examples/interactive_shor_levels.py                  # the error Z4
    python examples/interactive_shor_levels.py --error X0X1     # any of X0..Z8, X0X1, X0X3, Z0Z1, Z0Z3, none
    python examples/interactive_shor_levels.py --save fig.png   # write the window to a file instead
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

HEADER = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];        // data 0-8 (blocks 0-2, 3-5, 6-8); helpers 9-14 inner, 15-16 outer
creg s0[2];        // inner results of block 0
creg s1[2];        // block 1
creg s2[2];        // block 2
creg o[2];         // outer results
"""
ENCODER = ("cx q[0], q[3]; cx q[0], q[6];\nh q[0]; h q[3]; h q[6];\n"
           + "".join("cx q[{0}], q[{1}]; cx q[{0}], q[{2}];\n".format(b, b + 1, b + 2) for b in (0, 3, 6)))
INNER_CHECKS = "".join("cx q[{}], q[{h}]; cx q[{}], q[{h}];\nmeasure q[{h}] -> s{}[{}];\n".format(a, b, k, i, h=9 + 2 * k + i)
                       for k, base in enumerate((0, 3, 6)) for i, (a, b) in enumerate(((base, base + 1), (base + 1, base + 2))))
INNER_CORRECTIONS = "".join("if(s{0}==1) x q[{1}];\nif(s{0}==3) x q[{2}];\nif(s{0}==2) x q[{3}];\n".format(k, b, b + 1, b + 2)
                            for k, b in enumerate((0, 3, 6)))
OUTER_CHECKS = "".join("h q[{h}];\n".format(h=h) + "".join("cx q[{}], q[{}];\n".format(h, d) for d in data)
                       + "h q[{h}];\nmeasure q[{h}] -> o[{i}];\n".format(h=h, i=i)
                       for i, (h, data) in enumerate(((15, range(0, 6)), (16, range(3, 9)))))
OUTER_CORRECTIONS = "if(o==1) z q[0];\nif(o==3) z q[3];\nif(o==2) z q[6];\n"
QASM = {'X': 'x q[{0}];\n', 'Y': 'z q[{0}]; x q[{0}];\n', 'Z': 'z q[{0}];\n'}
ERRORS = {l + str(q): QASM[l].format(q) for q in range(9) for l in 'XYZ'}
ERRORS.update({'X0X1': 'x q[0];\nx q[1];\n', 'X0X3': 'x q[0];\nx q[3];\n',
               'Z0Z1': 'z q[0];\nz q[1];\n', 'Z0Z3': 'z q[0];\nz q[3];\n', 'none': ''})


def build(error: str) -> tuple[QProgram, int, int]:
    """The program with ``error``, levelled up twice with ``'gpf'``: level 1 (encoder, error, inner round), level 2."""
    p = QProgram(17, bits=8, initial_states={q: '0' for q in range(1, 17)})
    for name, text in [('encoder', ENCODER), ('error', ERRORS[error]), ('inner checks', INNER_CHECKS),
                       ('inner corrections', INNER_CORRECTIONS), ('outer checks', OUTER_CHECKS),
                       ('outer corrections', OUTER_CORRECTIONS)]:
        p.add_circuit(Circuit.from_qasm(HEADER + text), name=name)
    level1 = p.level_up(0, 0, 3, 'encoder + error + inner round', rewrite='gpf')
    level2 = p.level_up(level1, 0, 2, 'encoder + error + both rounds', rewrite='gpf')
    return p, level1, level2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--error", default='Z4', choices=list(ERRORS), metavar="E",
                    help="the error: X0..Z8, X0X1, X0X3, Z0Z1, Z0Z3 or none (default Z4)")
    ap.add_argument("--save", metavar="FILE", help="save the window to FILE instead of opening it")
    args = ap.parse_args()
    if args.save is None and plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "or use --save FILE")
    p, level1, level2 = build(args.error)
    print(p.describe_reduced(level1))
    print(p.describe_reduced(level2))
    fig = view_levels(p, level1, 0)
    if args.save:
        fig.savefig(args.save, dpi=110)
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
