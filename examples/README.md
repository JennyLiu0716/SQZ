# Examples

The notebooks draw the 2D views with PyZX's interactive JavaScript viewer (`zx.draw`, `draw_web`,
`draw_program_web`): open them in Jupyter to drag spiders and zoom (the viewer loads d3 from a CDN).
The 3D views use matplotlib (`draw_program_3d`). GitHub's notebook preview does not
run JavaScript, so the figures only show there if you re-execute with the matplotlib functions.

- [QProgramBasics.ipynb](QProgramBasics.ipynb) — building a `QProgram` from stages,
  the master space-time graph, the time-axis and 2+1D views, level-up and block
  sub-diagrams, permutations on a topology, and measurement rounds.
- [interactive_3d_view.py](interactive_3d_view.py) — the 2+1D view of the same
  program in an interactive window: `python examples/interactive_3d_view.py [top|front|side]`,
  then rotate with the mouse (left-drag), zoom (scroll / right-drag), pan (middle-drag),
  or click the `default` / `top` / `front` / `side` buttons to turn to those viewpoints.

- [FoldTransversalHadamard.ipynb](FoldTransversalHadamard.ipynb) — thesis case study 1
  re-verified with syndrome *measurement* instead of post-selection: the 19-qubit
  fold-transversal logical Hadamard on the distance-3 surface code as a `QProgram`,
  reduced with `zxqec.gpf` to a Hadamard plus an outcome-dependent Pauli frame and
  24 detectors; all-zero outcomes give exactly `H`. Circuits in `circuits/ft_h/` (with a
  wiring error of the thesis files corrected -- the symbolic reduction found it).
- [TInjectionWalkthrough.ipynb](TInjectionWalkthrough.ipynb) — **start here.** The running example
  of the paper, for a reader with no error-correction background: the ideas first (Pauli operators,
  codes, stabilizers, transversal gates, injection) with pictures, then the three questions answered
  from the gates alone: which code after every stage (`code_at`), what each block does to the logical
  qubits (`logical_action`, `summary`), and whether the whole injection is `T` in every run. Ends with
  four one-line changes to the program and what the tool says about each, including a check wired to
  the wrong qubit that keeps every run correct but leaves a distance-1 code. Figures come from
  [teaching_figures.py](teaching_figures.py).
- [Cultivation.ipynb](Cultivation.ipynb) — d = 3 magic-state cultivation, from the circuit of Wan,
  Zhong and Zapirain in Stim's format (`circuits/cultivation/`, Apache-2.0, with attribution). The tool
  finds the colour code after the injection, derives all 26 detectors the authors wrote by hand (two of
  them only once the layer of T gates is simplified as a whole; `check_detectors` reports each declared one as
  deterministic, not deterministic with two runs as evidence, or unknown), reads the double-check as a logical
  measurement in the basis {T†|+⟩, T†|−⟩}, and shows that every noiseless run prepares T†|+⟩ up to a
  Pauli correction. Part 4 draws the reduction step by step, as the thesis did (the circuit, the
  normal form with 15 T-type gadgets, the same gadgets in the code's variables as in block
  form, one T† gadget, and T†|+⟩ fed into the encoder of the colour code), and checks every picture against the circuit by tensor contraction in all 32 runs the
  detectors allow. The pictures are PyZX's own drawings (`zx.draw_matplotlib`) of the library's
  layout of a normal form (`layout_gpf`), with the phases written modulo the detectors
  (`phases_modulo`); the change of variables is `change_variables`. The last of them is also drawn in
  3D (`draw_graph_3d`) on the triangle of the Steane code, next to T†|+⟩ in the textbook Steane code.
  Every claim is then checked with an independent state-vector simulation. Part 6 asks the same
  questions of d = 5 cultivation with real T gates (42 qubits, from `haoliri0/SOFT`, Apache-2.0): the
  [[19,1,5]] code after the growth and after the d = 5 double-check, all 89 declared detectors and no
  others, T†|+⟩ prepared in every run, and the 53 T-type gadgets, written in the code's variables
  modulo the detectors, adding up to one T† gadget on the logical variable. The authors' own final check
  (`circuits/cultivation/d5_final_check.stim`, read after the protocol, with `MPP`) passes: its 18
  detectors and the observable are deterministic. As at d = 3, the result is
  drawn in 3D: in the variables of the faces that the final check measures, the reduced diagram is T†|+⟩ fed into the colour code of those faces, the same in every
  run, drawn on the file's own qubit coordinates next to that state built from the faces alone. Both are
  compared by tensor contraction, and so is the whole circuit, run by run, in 4 of its 1024 allowed runs.
  There is no simulation check at 42 qubits.
- [CultivationDoubleCheck.ipynb](CultivationDoubleCheck.ipynb) — only the double-check of that
  circuit, cut out of the program (`block_graph`) and reduced: T† gadgets on the seven inputs, T
  gadgets on the seven outputs, the block's six detectors, and a check that it measures
  ((X+Y)/√2)⊗7. The discard of a helper qubit's old state by its reset belongs before the block and is
  left out.
- [interactive_cultivation_3d.py](interactive_cultivation_3d.py) — step 4 of that reduction in an
  interactive window, drawn with `draw_graph_3d` on three planes: the non-Clifford state T†|+⟩, the
  Z-layer, and the seven output X-spiders on the textbook triangle of the Steane code (given as a
  `Topology`): `python examples/interactive_cultivation_3d.py [top|front|side]`, `--outcomes` to keep
  the outcome variables in the phases (default: the run where every outcome is 0), `--save FILE` to
  write the figure instead of opening a window.
- [interactive_cultivation_d5_3d.py](interactive_cultivation_d5_3d.py) — the same for d = 5 (Part 6 of
  `Cultivation.ipynb`): the reduced circuit in the variables of the faces that the authors' final
  check measures, on three planes over the 19 code qubits at the coordinates of the circuit file:
  `python examples/interactive_cultivation_d5_3d.py [top|front|side]`, `--textbook` for T†|+⟩ in the
  colour code built from the faces alone, `--save FILE` to write the figure. The paper's d = 5 figure
  calls its functions.
- [StabilizerCodes.ipynb](StabilizerCodes.ipynb) — the three questions for codes that are not CSS. The
  [[5,1,3]] code read off a ring of `CZ` gates (its stabilizers mix `X` and `Z`); its ideal encoder, built
  from the Choi state following Yeh et al., checked with numpy against its definition; a program that
  encodes, measures the checks and applies `H` then `S` on every qubit, which the tool finds to be the
  logical `SH`, transversal and keeping the code; `H` on every qubit, which leaves it; the Steane code
  after one `S` and its logical `H`; and what is not covered (a code that differs between runs).
- [TransversalGates.ipynb](TransversalGates.ipynb) — the role `transversal gate`: a logical gate whose
  gates never connect two qubits of one code block, directly or through a chain. The transversal `CNOT`
  of two Steane blocks passes; the same `CNOT` with a chain that undoes itself, and a `SWAP` inside the
  [[4,2,2]] code (a logical `CNOT`), do not; `H` on every qubit of the [[4,2,2]] code is transversal but
  its logical action, `SWAP (H x H)` (checked with numpy), is not one of the named gates.
- [ComparingPrograms.ipynb](ComparingPrograms.ipynb) — are two programs the same *logically*? A
  demo of `compare_programs`, for comparing a circuit as written with a compiled one: block by block,
  the same codes, the same logical operations up to a Pauli frame, and the same detectors. Four
  variants of the Steane T injection (Z checks measured first and `S` instead of `S†` in the
  correction are equivalent; no correction changes the logical action; a check wired to the wrong
  qubit changes the code), and the thesis's mis-wired fold-transversal H, whose codes and logical
  operations are all correct and which only the lost detectors (24 against 22) reveal, located in
  the round with the wrong check. Ends with what the comparison does not show.
- [CircuitDistance.ipynb](CircuitDistance.ipynb) — the circuit distance of Clifford stages
  (`circuit_distance`, with Stim and Mark Webster's `codedistance`; `pip install zxqec[distance]`).
  The fold-transversal H's rounds have circuit distance 2 although every code has distance 3, and
  `explain()` names the two hook errors; the mis-wired ones have 1. Then the Z distance of
  cultivation through its `T` gates (`without_gates`): 3 at d = 3 and 4 at d = 5, with the four
  faults, and an independent check with Stim alone on the published d = 5 file with `T → S`.
- [SteaneTInjection.ipynb](SteaneTInjection.ipynb) — a `T` gate injected into the [[7,1,3]] Steane
  code, from the physical circuit to the logical gate. **Written for a reader with no quantum error
  correction background**: it draws the code and its stabilizers, the injection gadget and the
  protocol's stages, and explains each result in plain language. Along the way: a verified encoder
  (its stabilizers and logical operators are derived, not asserted) and its transversal gates; the
  protocol as a 20-qubit `QProgram`; level-up into the logical operations with a measurement report
  per block, showing checks that are meaningless in a block alone and deterministic once it is
  merged; and the result, `T` when the logical readout is 0 and `S†T` when it is 1, always inside
  the code space.
- [TrackingOutcomeVariables.ipynb](TrackingOutcomeVariables.ipynb) — why a reduced diagram shows
  only 8 of the 34 measurement outcomes, although each occurs once in the circuit. A one-gate example
  where the same measurement both keeps and loses its variable, the character-sum rule behind it, the
  rewrite-by-rewrite trace of `c[5]` through PyZX's `full_reduce` and through `reduce_gpf` (both end
  by moving it into the diagram's scalar, as the factor `1 + e^{iπ(c5+c1)}` — a detector), a table
  classifying all 34 outcomes as frame / fixed / tied / free, and the nine final data readouts, where
  the code's X stabilizers, the logical X and four gauge directions are separated and checked by
  enumerating all 32 possible patterns. Uses [track_variable.py](track_variable.py), a tracer that
  wraps the rewrite appliers for the duration of a `with tracing(name)` block.
- [interactive_fold_h_3d.py](interactive_fold_h_3d.py) — the 2+1D view of that protocol
  in an interactive window, on the thesis layout of the 19 qubits:
  `python examples/interactive_fold_h_3d.py [top|front|side]` for the whole protocol
  (`--level 2` shades its five phases instead of the nine stages), `--stage I` or
  `--block LEVEL I` for one stage or block only, `--phases` to label the spiders with
  their phases, `--miswired` for the wiring error of the thesis's circuit files
  (`ComparingPrograms.ipynb`, Part 2), `--save FILE` to write the figure instead of opening a window.

Planned: magic state distillation; cultivation at larger distances.
