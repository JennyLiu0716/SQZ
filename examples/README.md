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

Planned: magic state injection, distillation and cultivation.
