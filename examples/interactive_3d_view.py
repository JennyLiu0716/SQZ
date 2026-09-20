"""Open the 2+1D view of a program in an interactive window.

Run from a terminal (the window blocks until it is closed):

    python examples/interactive_3d_view.py            # oblique start view
    python examples/interactive_3d_view.py top        # start from the bird's-eye view
    python examples/interactive_3d_view.py front

Mouse controls (matplotlib's standard 3D navigation):

* rotate: left-click and drag -- the current ``elev`` / ``azim`` are printed in
  the terminal while dragging (``elev`` ~ 90 is the bird's-eye view, ~ 0 the
  front/side views); the buttons below the plot turn the figure straight to the
  named viewpoints (``default`` = oblique, ``top`` = bird's-eye, ``front``, ``side``);
* zoom: right-click and drag, or scroll (two-finger scroll on a trackpad);
* pan: middle-click and drag, or the pan tool in the toolbar; the toolbar's
  home button resets the view.

The program is the one from ``QProgramBasics.ipynb``: two parity-check rounds of
a five-qubit "plus" (four data qubits around an ancilla in the centre of a
3x3 grid) with a T layer in between, grouped into two blocks.
"""

import sys

import matplotlib

# The native "MacOSX" backend can block in plt.show() without opening a window
# when Python is launched from some terminals; Tk is reliable wherever it is
# installed, so prefer it and fall back to matplotlib's default otherwise.
try:
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt

from pyzx.circuit import Circuit
from pyzx.circuit.gates import HAD, CNOT, T, Measurement, Reset
from zxqec.qprogram import QProgram
from zxqec.spacetime import VIEWS, Topology, draw_program_3d


def build_program() -> QProgram:
    grid = Topology.grid(3, 3)
    mapping = {0: 1, 1: 3, 2: 5, 3: 7, 4: 4}          # logical 4 = ancilla, on the centre node
    star = Circuit(5)
    for gate in [Reset(4), HAD(4), CNOT(4, 0), CNOT(4, 1), CNOT(4, 2), CNOT(4, 3), HAD(4), Measurement(4)]:
        star.add_gate(gate)
    t_layer = Circuit(5)
    t_layer.add_gate(T(0))
    t_layer.add_gate(T(2))

    p = QProgram(5, topology=grid, mapping=mapping, initial_states={4: '0'}, gate_durations={"CNOT": 2})
    p.add_circuit(star, name="round 1")
    p.add_circuit(t_layer, name="T layer")
    p.add_circuit(star, name="round 2")
    p.level_up(0, 0, 1, name="round + T")
    return p


def main() -> None:
    view = sys.argv[1] if len(sys.argv) > 1 else 'default'
    if view not in VIEWS:
        sys.exit("unknown view {!r}; choose from {}".format(view, ", ".join(VIEWS)))
    if plt.get_backend().lower() == 'agg':
        sys.exit("matplotlib is using the non-interactive Agg backend; run this from a terminal "
                 "with a GUI backend (e.g. TkAgg, QtAgg)")

    p = build_program()
    fig = draw_program_3d(p, level=1, figsize=(9, 7), view=view, buttons=True)
    ax = fig.axes[0]
    fig.canvas.mpl_connect(
        'motion_notify_event',
        lambda event: print("\relev = {:6.1f}   azim = {:7.1f}".format(ax.elev, ax.azim), end=""))
    print("backend: {} -- drag to rotate, scroll to zoom; close the window to exit".format(plt.get_backend()))
    plt.show()


if __name__ == '__main__':
    main()
