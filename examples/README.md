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
  them only once the layer of T gates is simplified as a whole), reads the double-check as a logical
  measurement in the basis {T†|+⟩, T†|−⟩}, and shows that every noiseless run prepares T†|+⟩ up to a
  Pauli correction. Part 4 draws the reduction step by step, as the thesis did (the circuit, the
  normal form with 15 T-type gadgets, the same gadgets in the code's variables as in block
  form, one T† gadget, and T†|+⟩ fed into the encoder of the colour code), and checks every picture against the circuit by tensor contraction in all 32 runs the
  detectors allow; the pictures come from [zx_steps.py](zx_steps.py). Every claim is then checked
  with an independent state-vector simulation.
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
  their phases, `--save FILE` to write the figure instead of opening a window.

Planned: magic state distillation; cultivation at larger distances.
