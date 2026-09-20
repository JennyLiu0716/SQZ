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
| `zxqec.qprogram` | `QProgram`: build a protocol from circuit stages and permutations; hierarchical *level-up* of gates into blocks (syndrome rounds, logical gates); extract the sub-diagram of a block | planned (port from thesis) |
| `zxqec.spacetime` | 2+1D space-time programs: hardware `Topology`, qubit ↔ node mapping, 3D view (2D space + 1D time) of ZX-diagrams | planned (port from thesis) |
| `zxqec.tableau` | Import and extract stabilizer tableaus to/from ZX-diagrams: syndrome measurement circuits, stabilizer states, unitary Clifford circuits | planned |
| `zxqec.topological` | Pauli-box hypergraph normal form (each weight-*w* stabilizer = one off-qubit phase-free X spider on *w* on-qubit phase-free Z spiders, cf. Def. 5.1 of [arXiv:2603.09580](https://arxiv.org/abs/2603.09580)); topological view rendering each hyperedge as a surface/volume coloured by Pauli type (X red, Z green, Y blue, mixed as gradients); toggle between ZX and topological view | planned |
| `zxqec.gui` | ZXLive embedding for drag-and-drop rewrites after level-up | planned |
| `examples/` | Magic state injection, distillation and cultivation | planned |

The 1+1D time layer (per-spider `timestep`/`delay` vertex data, ASAP
scheduling, space-time cost metrics, `time_slice`) lives upstream in
`pyzx.spacetime`; this library builds the second spatial dimension and the
protocol-level abstractions on top of it.

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
