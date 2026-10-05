# zxqec

Space-time ZX-diagrams for designing and verifying dynamic quantum error
correction (QEC) protocols. A standalone library built **on top of**
[PyZX](https://github.com/zxcalc/pyzx): it imports PyZX and never patches it,
so users can extend it without modifying PyZX, and every PyZX simplification,
extraction and tensor check keeps working on the diagrams it produces.

It grows out of the MPhil thesis *A ZX-Calculus Based Software Framework for
Designing and Verifying Dynamic Quantum Error Correction Protocols*
(Chenxi Liu, University of Cambridge, 2026, supervised by Lia Yeh).

## Roadmap

| Package | Purpose | Status |
|---|---|---|
| `zxqec.qprogram` | `QProgram`: build a protocol from circuit stages and permutations; hierarchical *level-up* of stages into blocks (syndrome rounds, logical gates); extract the sub-diagram of any stage or block | **done** |
| `zxqec.spacetime` | 2+1D space-time programs: hardware `Topology`, qubit ↔ node mapping, `induced_subdiagram`; interactive notebook view on a time axis with delays and block shading (d3, via PyZX's viewer), the 3D view (2D space + 1D time) with matplotlib, plus static matplotlib versions of the 2D views | **done** (interactive level navigation belongs to the GUI) |
| `zxqec.gpf` | Generalized parity form **with phases**: Pauli phases (concrete π or Boolean measurement outcomes) stay on the spiders through the Boolean bialgebra rule and the GF(2) row reduction; non-Pauli phases become gadgets; Hadamards are pushed to the boundary wires; the reduction returns the detectors (outcome parities that must vanish) and flags impossible outcome combinations | **done** (`isolate_paulis=True` turns every non-zero phase into a gadget) |
| `zxqec.gpf` (report) | `measurement_report` / `QProgram.measurement_report`: what happens to every measurement outcome of a program or of one block of a level -- part of the Pauli frame, fixed by a detector, tied to other outcomes, or gauge -- with the detectors, the frame, and `violated(outcomes)` / `frame_values(outcomes)` for checking an actual run | **done** |
| `zxqec.tableau` | Import and extract stabilizer tableaus to/from ZX-diagrams: syndrome measurement circuits, stabilizer states, unitary Clifford circuits | planned |
| `zxqec.topological` | Pauli-box hypergraph normal form (each weight-*w* stabilizer = one off-qubit phase-free X spider on *w* on-qubit phase-free Z spiders, cf. Def. 5.1 of [arXiv:2603.09580](https://arxiv.org/abs/2603.09580)); topological view rendering each hyperedge as a surface/volume coloured by Pauli type (X red, Z green, Y blue, mixed as gradients); toggle between ZX and topological view | planned |
| `zxqec.gui` | ZXLive embedding for drag-and-drop rewrites after level-up | planned |
| `examples/` | `QProgramBasics`, `FoldTransversalHadamard` (case study 1 with syndrome measurement: logical `H` + Pauli frame + detectors); magic state injection, distillation and cultivation | first two done |

The 1+1D time layer (per-spider `timestep`/`delay` vertex data, ASAP
scheduling, space-time cost metrics, `time_slice`) lives upstream in
`pyzx.spacetime`; this library builds the second spatial dimension and the
protocol-level abstractions on top of it.

## A first look

```python
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import HAD, CNOT, Measurement, Reset
from zxqec.qprogram import QProgram

rnd = Circuit(3)                                   # one syndrome-extraction round: ancilla = qubit 2
for g in [Reset(2), HAD(2), CNOT(2, 0), CNOT(2, 1), HAD(2), Measurement(2)]:
    rnd.add_gate(g)

p = QProgram(3, initial_states={2: '0'}, gate_durations={"CNOT": 2})
p.add_circuit(rnd, name="round 1")
p.add_permutation([1, 0, 2], name="swap")          # move logical qubit 0 to wire 1 and back
p.add_circuit(rnd, name="round 2")

g = p.graph()                                      # master space-time ZX-diagram (a plain pyzx graph)
zx.draw(g, show_time=True)                         # spiders on physical wires, timestep labels, squiggly delays
p.level_up(0, 0, 1, name="round + swap")           # group stages into a block ...
h = p.block_graph(1, 0)                            # ... and cut its sub-diagram out
zx.full_reduce(h)                                  # everything PyZX offers applies to it

from zxqec.spacetime import Topology, draw_web, draw_program_web, draw_program_3d
draw_web(h)                                        # notebook: interactive d3 view, spiders at their timestep, delays as squiggles
draw_program_web(p, level=1)                       # the same for the whole program, blocks of level 1 shaded
q = QProgram(3, topology=Topology.line(3), initial_states={2: '0'})
q.add_circuit(rnd)
draw_program_3d(q, level=0)                        # 2+1D: qubits at their topology positions, time upwards (matplotlib)
# draw_program / draw_program_views: static matplotlib versions of the 2D and multi-view figures
```

Every spider of `g` carries `timestep`, `delay` (from `pyzx.spacetime`), `stage`
and `logical_qubit` vertex data. Within a stage gates are scheduled as soon as
possible; stages run one after the other. A permutation stage only changes on
which physical wire a logical qubit sits (the program as a map on logical
qubits is unchanged); with a `Topology` it is checked to be a layer of disjoint
swaps along edges. Qubits measured in one stage may be reused after a `Reset`
in a later one.

## Verifying circuits: GPF with phases

```python
from zxqec.gpf import reduce_gpf, equal_for_all_assignments

circ = zx.qasm(open("syndrome_rounds.qasm").read())   # measurements -> symbolic outcomes c[i]
circ.initialize_qubits([...])
g = circ.to_graph()
res = reduce_gpf(g)          # minimal bipartite Z/X form, Pauli phases carried along
res.graph                    # the reduced diagram (pyzx graph); res.to_graph() restores boundary Hadamards
res.detectors                # e.g. [c[3] + c[0], ...]: parities of outcomes that are always 0
res.is_zero                  # True if the given outcomes cannot occur

rep = p.measurement_report()             # or p.measurement_report(level, index) for one block
print(rep)                               # per-outcome table: frame / fixed / tied / gauge
rep.violated({'c[9]': 1})                # the detectors that fire for the outcomes of a run
rep.frame_values({'c[0]': 1})            # the Pauli correction those outcomes call for
equal_for_all_assignments(res.to_graph(), g, constraints=res.constraints)   # brute-force check
```

Row addition in the split-matrix reduction is sound for a *free* row (no input,
output or gadget attached, Pauli phase) added to any other row, provided the
phases are added too; a free row that becomes empty is the scalar `1 + (-1)^p`,
which is where the detectors come from.

## Installation

`pyzx.spacetime` is not in a PyPI release yet, so the dependency points at the
`master` branch of the PyZX fork [JennyLiu0716/pyzx](https://github.com/JennyLiu0716/pyzx),
which is upstream PyZX plus the pending pull requests. A plain install pulls it in:

```bash
pip install -e ".[dev]"
```

For development next to a PyZX checkout:

```bash
pip install -e ../pyzx
pip install -e ".[dev]"
```

## Design rules

- Import `pyzx`; never monkey-patch or subclass its graph classes for storage.
  Extra per-spider information (coordinates, block ids, ...) is kept in
  vertex data (`g.vdata`), so any PyZX backend and rewrite keeps working.
- Anything general enough for PyZX is contributed upstream by pull request;
  QEC-specific machinery stays here.

## Tests

```bash
python -m pytest
```

## License

Apache-2.0, the same license as PyZX.

The optional `distance` extra installs Stim (Apache-2.0) and Mark Webster's `codedistance`, which is MIT-licensed: its PyPI metadata says "GNUv3", but the license file it ships, like its repository, is the MIT License.
