"""Magic state distillation, level by level.

The script version of ``MagicStateDistillationStepByStep.ipynb``. It turns the
transversal CCZ of the [[8,3,2]] code around. The distillation identity: the decoder E^dag applied to eight magic states (T and T^dag on
|+>) equals E^dag applied to |+>^8, followed by a CCZ. The decoding derivation rewrites the
right-hand side, one rewrite per level, into the three-legged H-box: the CCZ
magic state. The tool checks the identity, and checks every level of the derivation
against the circuit.

Run from a terminal. Each step prints what the tool finds and opens its
figures; close the window(s) to go on to the next step. The last step opens
the levels window (``zxqec.spacetime.view_levels``) on level 2:

    python examples/magic_state_distillation_step_by_step.py              # step by step, in windows
    python examples/magic_state_distillation_step_by_step.py --save DIR   # write the figures to DIR instead
"""

import argparse
import os
import sys
import warnings

import matplotlib

try:                                              # as in interactive_fold_h_3d.py: Tk opens reliably
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CCZ, CNOT, T, PostSelect

from zxqec.gpf import equal_for_all_assignments
from zxqec.qprogram import QProgram
from zxqec.spacetime import draw_program, view_levels

# as in TransversalCCZ.ipynb: qubit k is qubit [8, 7, 6, 4, 1, 2, 3, 5][k] of the reference numbering; x, y, z are qubits 1, 2, 3
WIRES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 0: 7}          # drawn in the reference order of wires
GHZ = [CNOT(0, 4), CNOT(0, 5), CNOT(0, 6), CNOT(0, 7)]
FROM_X = [CNOT(1, 4), CNOT(1, 6), CNOT(1, 7), CNOT(0, 1)]
FROM_Y = [CNOT(2, 4), CNOT(2, 5), CNOT(2, 7), CNOT(0, 2)]
FROM_Z = [CNOT(3, 4), CNOT(3, 5), CNOT(3, 6), CNOT(0, 3)]
# the magic states: T^dag on the reference qubits 1, 4, 6, 7, T on 2, 3, 5, 8
P_DAGGER = [T(1, adjoint=True), T(2, adjoint=True), T(3, adjoint=True), T(4, adjoint=True), T(5), T(6), T(7), T(0)]
# the decoder E^dag: the encoder's CNOTs in reverse order, then each GHZ qubit projected on the state it started in
DECODER = (FROM_Z[::-1] + FROM_Y[::-1] + FROM_X[::-1] + GHZ[::-1]
           + [PostSelect(0, '+')] + [PostSelect(q, '0') for q in (4, 5, 6, 7)])


def build():
    """Both sides of the distillation identity as programs, and the four levels of the decoding on the right-hand side."""
    magic = QProgram(8, mapping=WIRES, initial_states={q: '+' for q in range(8)})
    decoded = QProgram(8, mapping=WIRES, initial_states={q: '+' for q in range(8)})
    for program, stages in ((magic, ((P_DAGGER, 'magic states'), (DECODER, 'decoder'))),
                            (decoded, (([], '|+> states'), (DECODER, 'decoder'), ([CCZ(1, 2, 3)], 'CCZ')))):
        for gates, name in stages:
            c = Circuit(8)
            for gate in gates:
                c.add_gate(gate)
            program.add_circuit(c, name=name)
    # four level-ups of the right-hand side, each with one rewrite
    notation = decoded.group_stages([(0, 0, '|+> states', None, 'standard notation')])
    normal_form = decoded.level_up(notation, 1, 1, 'decoder', rewrite='x-z gpf')
    copied = decoded.level_up(normal_form, 0, 1, '|+> states and decoder', rewrite='copy and fuse')
    h_box = decoded.level_up(copied, 0, 1, 'all', rewrite='remove identities')
    return magic, decoded, (notation, normal_form, copied, h_box)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--save", metavar="DIR", help="write the figures to DIR instead of opening windows")
    args = ap.parse_args()
    if args.save:
        plt.switch_backend("Agg")
        warnings.filterwarnings("ignore", message=".*non-interactive.*")   # PyZX's plt.show() under Agg
        os.makedirs(args.save, exist_ok=True)
    elif plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "or use --save DIR")

    def show(name, *figures):
        if args.save:
            for k, fig in enumerate(figures):
                path = os.path.join(args.save, name + ('-%d' % (k + 1) if len(figures) > 1 else '') + '.png')
                fig.savefig(path, dpi=110, bbox_inches='tight')
                print("  saved", path)
        else:
            plt.show()

    magic, decoded, (notation, normal_form, copied, h_box) = build()

    print("== The distillation identity: the magic states decoded (left) and |+>^8 decoded, then a CCZ (right)")
    print("left-hand side:", magic.logical_action().verdict())
    print("both sides the same map:", equal_for_all_assignments(magic.quantum_graph(), decoded.quantum_graph()))
    show('eq-6.48', draw_program(magic, show_time=False, labels=False, figsize=(12, 5)),
         draw_program(decoded, show_time=False, labels=False, figsize=(12, 5)))

    print("\n== The decoding, level by level")
    for level in (notation, normal_form, copied, h_box):
        print(decoded.describe_level(level), end='\n\n')
    circuit_graph = decoded.quantum_graph()

    steps = ((notation, 0, (12, 5), "Level 1: the |+> states in standard notation"),
             (normal_form, 1, (8, 6), "Level 2: E^dag in normal form"),
             (copied, 0, (5, 4), "Level 3: the |+> states copy and fuse"),
             (h_box, 0, (4, 3), "Level 4: the three-legged H-box"))
    for k, (level, index, figsize, title) in enumerate(steps, start=1):
        print("==", title)
        print(decoded.reduced_block(level, index))
        reduced = decoded.reduced_level(level)
        print("the same map as the circuit:", equal_for_all_assignments(circuit_graph, reduced.graph))
        if level == h_box:
            print("the same map as the magic states decoded:",
                  equal_for_all_assignments(magic.quantum_graph(), reduced.graph))
        show('level-%d' % k, zx.draw_matplotlib(reduced.graph, figsize=figsize))

    print("\n== The levels window, on level 2")
    show('levels-window', view_levels(decoded, normal_form, 1))


if __name__ == "__main__":
    main()
