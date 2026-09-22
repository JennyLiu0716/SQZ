# Fold-transversal logical Hadamard on the distance-3 surface code

Circuit stages of thesis case study 1 (§4.6), on 19 qubits:

| qubits | role |
|---|---|
| `q[0]` | logical input, injected into the code by fan-out and an X measurement |
| `q[1]..q[9]` | data qubits, a 3×3 grid (row-major: `q[1] q[2] q[3]` is the top row) |
| `q[10]..q[13]` | ancillas of the X-type checks (weight 2/4/4/2) |
| `q[14]..q[17]` | ancillas of the Z-type checks |
| `q[18]` | decode ancilla: receives the logical Z parity of `q[7] q[8] q[9]` |

| file | stage |
|---|---|
| `ft_h.qasm` | encode `q[0]` into the code and measure the X checks (code 1) |
| `ft_h_2.qasm` | transversal Hadamard on the data qubits |
| `ft_h_3.qasm` | Z checks on the `q[10..13]` positions (code 2) |
| `ft_h_3.2.qasm` | X checks on the `q[14..17]` positions (code 2) |
| *(rotation)* | rotate the data grid by 90°: `1→3→9→7→1`, `2→6→8→4→2`, `5` fixed |
| `ft_h_5.qasm` | Z checks (code 1 again, written in the rotated frame) |
| `ft_h_5.2.qasm` | X checks |
| `ft_h_6.qasm` | Z checks, then decode: copy Z_L onto `q[18]` and measure all data in the X basis |

The files do not re-prepare ancillas after measuring them (the `// reset` lines
are commented out); `FoldTransversalHadamard.ipynb` adds the `reset` (and `h`
for X checks) explicitly when it builds the program.

**Correction to the thesis files.** In `ft_h_5.qasm` the Z check with ancilla
`q[14]` read `cx q[1], q[14]; cx q[2], q[14]` (the operator $Z_1 Z_2$, which is
not a stabilizer of the code); it is `cx q[4], q[14]` here, matching the other
stages and the layout. The symbolic reduction in `FoldTransversalHadamard.ipynb`
found this: the wrong check gave a random outcome, and the anticommuting X check
$X_2 X_3$ of the next round became random too. Post-selection to all-zero
outcomes, as in the thesis, cannot detect such an error.
