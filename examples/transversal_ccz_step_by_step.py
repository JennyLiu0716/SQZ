"""The transversal CCZ, level by level.

The script version of ``TransversalCCZStepByStep.ipynb``. It shows that T on four qubits of the [[8,3,2]] code and T^dag on
the other four is a logical CCZ. It writes the logical gate as L = E; P; E^dag
(the encoder, the layer P of T and T^dag gates, the decoder). The first step moves
the CNOTs on either side of P through it, one group at a time, so that each T
becomes a phase gadget on the parity the CNOTs computed. The second step removes the
gadgets that involve the encoder's random bit w (the spider-nest identities),
which leaves the CCZ. Each step is one level-up with one rewrite, checked
against the circuit.

Run from a terminal. Each step prints what the tool finds and opens its
figures; close the window(s) to go on to the next step. The last step opens
the levels window (``zxqec.spacetime.view_levels``) on level 2:

    python examples/transversal_ccz_step_by_step.py              # step by step, in windows
    python examples/transversal_ccz_step_by_step.py --save DIR   # write the figures to DIR instead
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

# qubit k here is qubit REF[k] of the reference numbering; x, y and z enter on qubits 1, 2 and 3
REF = [8, 7, 6, 4, 1, 2, 3, 5]
# the wires, as the reference drawing shows them, top to bottom: its qubits 7, 6, 4, 1, 2, 3, 5, 8
WIRES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 0: 7}
# the encoder: a GHZ state on the reference qubits 8, 1, 2, 3, 5, then four CNOTs from each of x, y, z
GHZ = [CNOT(0, 4), CNOT(0, 5), CNOT(0, 6), CNOT(0, 7)]
FROM_X = [CNOT(1, 4), CNOT(1, 6), CNOT(1, 7), CNOT(0, 1)]
FROM_Y = [CNOT(2, 4), CNOT(2, 5), CNOT(2, 7), CNOT(0, 2)]
FROM_Z = [CNOT(3, 4), CNOT(3, 5), CNOT(3, 6), CNOT(0, 3)]
# P: T on the reference qubits 1, 4, 6, 7, T^dag on 2, 3, 5, 8
P = [T(1), T(2), T(3), T(4), T(5, adjoint=True), T(6, adjoint=True), T(7, adjoint=True), T(0, adjoint=True)]


def build():
    """The circuit of L = E; P; E^dag in nine stages: the encoder, P, and the decoder."""
    circuit = QProgram(8, mapping=WIRES, initial_states={0: '+', 4: '0', 5: '0', 6: '0', 7: '0'})
    decode = GHZ[::-1] + [PostSelect(0, '+')] + [PostSelect(q, '0') for q in (4, 5, 6, 7)]
    for gates, name in ((GHZ, 'GHZ'), (FROM_X, 'from x'), (FROM_Y, 'from y'), (FROM_Z, 'from z'), (P, 'P'),
                        (FROM_Z[::-1], 'to z'), (FROM_Y[::-1], 'to y'), (FROM_X[::-1], 'to x'),
                        (decode, 'GHZ back')):
        c = Circuit(8)
        for gate in gates:
            c.add_gate(gate)
        circuit.add_circuit(c, name=name)
    return circuit


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

    circuit = build()
    print("== The circuit of L = E; P; E^dag")
    print(circuit.logical_action().verdict())
    show('circuit', draw_program(circuit, show_time=False, labels=False, figsize=(16, 6)))

    # grouped into the encoder, P and the decoder: questions 2 and 3, and for P the composite of the code's
    # ideal encoder, P and the ideal decoder (the first diagram of the derivation). This grouping comes first:
    # levelling up again from level 0 below starts a new chain of levels, which replaces it.
    print("\n== E, P and E^dag")
    parts = circuit.group_stages([(0, 3, 'E', 'encoder'), (4, 4, 'P', 'transversal gate'),
                                  (5, 8, 'E dagger', 'decoder')])
    print(str(circuit.summary(parts)).split('\n\n')[0])
    show('composite-of-P', zx.draw_matplotlib(circuit.logical_action(parts, 1).composite, figsize=(10, 6)))

    # the levels: each level-up groups some blocks of the level before and applies one rewrite
    print("\n== The levels")
    through_z = circuit.level_up(0, 3, 5, 'z', rewrite='phase gadgets')
    through_y = circuit.level_up(through_z, 2, 4, 'y', rewrite='phase gadgets')
    through_x = circuit.level_up(through_y, 1, 3, 'x', rewrite='phase gadgets')
    through_ghz = circuit.level_up(through_x, 0, 2, 'all', rewrite='phase gadgets')
    nest = circuit.level_up(through_ghz, 0, 0, 'all', rewrite='spider nest')
    for level in (through_z, through_y, through_x, through_ghz, nest):
        print(circuit.describe_level(level), end='\n\n')

    print("== Level 0: the circuit")
    circuit_graph = circuit.quantum_graph()
    show('level-0', zx.draw_matplotlib(circuit_graph, figsize=(16, 6)))

    ccz = Circuit(3)
    ccz.add_gate(CCZ(0, 1, 2))
    steps = ((through_z, 3, (16, 6), "Level 1: through the CNOTs from z"),
             (through_y, 2, (16, 6), "Level 2: through the CNOTs from y"),
             (through_x, 1, (16, 6), "Level 3: through the CNOTs from x"),
             (through_ghz, 0, (10, 5), "Level 4: through the GHZ state"),
             (nest, 0, (10, 5), "Level 5: the spider nest"))
    for k, (level, index, figsize, title) in enumerate(steps, start=1):
        print("==", title)
        print(circuit.reduced_block(level, index))
        reduced = circuit.reduced_level(level)
        print("the same map as the circuit:", equal_for_all_assignments(circuit_graph, reduced.graph))
        if level == nest:
            print("the same map as PyZX's CCZ gate:", equal_for_all_assignments(reduced.graph, ccz.to_graph()))
        show('level-%d' % k, zx.draw_matplotlib(reduced.graph, figsize=figsize))

    print("\n== The levels window, on level 2")
    show('levels-window', view_levels(circuit, through_y, 2))


if __name__ == "__main__":
    main()
