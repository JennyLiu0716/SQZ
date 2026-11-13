# Examples

Every notebook starts with a short *In short* for readers without a background in quantum error
correction: the setting, the question, the answer, and how to read the pictures. The pictures are
static drawings made by library functions (`draw_program`, `draw_graph`, `draw_program_slices`,
`draw_relaxed`, `draw_program_3d`, PyZX's `draw_matplotlib`), so they all show on GitHub. The
interactive views are still there: PyZX's `zx.draw`, `draw_web` and `draw_program_web` in Jupyter
(drag spiders, zoom; the viewer loads d3 from a CDN), and the scripts below in a window.

- [QProgramBasics.ipynb](QProgramBasics.ipynb) — building a `QProgram` from stages,
  the master space-time graph, the time-axis and 2+1D views, level-up and block
  sub-diagrams, permutations on a topology, and measurement rounds.
- [interactive_3d_view.py](interactive_3d_view.py) — the 2+1D view of the same
  program in an interactive window: `python examples/interactive_3d_view.py [top|front|side]`,
  then rotate with the mouse (left-drag), zoom (scroll / right-drag), pan (middle-drag),
  or click the `default` / `top` / `front` / `side` buttons to turn to those viewpoints.

- [FoldTransversalHadamard.ipynb](FoldTransversalHadamard.ipynb) — a fold-transversal logical
  Hadamard verified with syndrome *measurement*, outcomes kept symbolic: the 19-qubit
  fold-transversal logical Hadamard on the distance-3 surface code as a `QProgram`,
  reduced with `zxqec.gpf` to a Hadamard plus an outcome-dependent Pauli frame and
  24 detectors; all-zero outcomes give exactly `H`. Drawn level by level: the physical circuit,
  each stage time step by time step on the layout of the qubits, and the logical Hadamard.
  Circuits in `circuits/ft_h/` (with a wiring error of an earlier version corrected -- the
  symbolic reduction found it).
- [FoldH.ipynb](FoldH.ipynb) — the same protocol at four levels, as in Section 4.1 of the
  SQZ paper: level 0 the eight stages (the physical gates, `summary(0)`), level 1 the protocol's
  six pieces with roles, level 2 the fold-transversal H as one block (a logical H when qubits are
  matched by their place on the chip), level 3 the whole program (H and a Pauli frame). Each stage
  in its normal form on the chip (`gpf_checks`, `draw_blocks_3d`, `save_panels`), and the wiring
  error put back: levels 1 and 2 flag it, level 0 shows the [[9,1,1]] code after stage 5, level 3
  still says H. Makes every figure of that section (saved to `figures/`) and prints every number.
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
  Pauli correction. Part 4 draws the reduction step by step (the circuit, the
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
  compared by tensor contraction, and so is the whole circuit, run by run, in 4 of its 1024 allowed runs
  (all 1024: `cultivation_d5_all_runs.py`, about nine hours).
  There is no simulation check at 42 qubits.
- [CultivationDoubleCheck.ipynb](CultivationDoubleCheck.ipynb) — only the double-check of that
  circuit, cut out of the program (`block_graph`) and reduced: T† gadgets on the seven inputs, T
  gadgets on the seven outputs, the block's six detectors, and a check that it measures
  ((X+Y)/√2)⊗7. The discard of a helper qubit's old state by its reset belongs before the block and is
  left out.
- [CultivationErrorFormulas.ipynb](CultivationErrorFormulas.ipynb) — does d = 3 cultivation keep its fault
  distance 3 with real T gates? Stim cannot run T (`circuit_distance` refuses it; with S in place of T it finds 3),
  and trying the 1.7 million pairs of faults one at a time would take days. `error_formulas` puts a switch for
  every possible error into the ZX-diagram and simplifies that one diagram once: an X error on a T gate becomes
  the sign of one phase gadget (`phase_polynomial` shows the 15), and each pattern of faults is decided from short
  formulas. The answer is 2: 48 pairs, each with an X error on a T gate of the double-check; with S in place of T
  all 48 are caught or harmless. The checks come from the diagram (`check_detectors`: the same 26 as the file),
  rates come from random faults without simulation, and every verdict is checked the slow way. A small two-round
  example shows the idea first.
- [CultivationFaultsD3.ipynb](CultivationFaultsD3.ipynb) — the same question with every Pauli fault after every
  gate (X, Y or Z after a one-qubit gate, any of the 15 two-qubit Paulis after a CNOT), in one reduced diagram,
  and a solver (`search_lowest_wrong`) that finds the fewest faults that are wrong: 2 with real T gates (3 with S in
  place of T), with a witness and why it is wrong. `all_wrong=True` lists the 12 wrong signatures; trying every pair
  finds 48 wrong pairs, with exactly these signatures. The reduced diagram is drawn in 3D on the Steane triangle
  (`draw_code_3d`).
- [CultivationFaultsD5.ipynb](CultivationFaultsD5.ipynb) — the same at d = 5 (42 qubits, 888 gates, 53 T gates):
  all 8109 single faults and all 33 million pairs tried the fast way, none wrong; the solver finds a wrong set of 3
  faults, and `classify` and `classify_directly` both call it wrong. So the fault distance is 3 with real T gates
  (4 with S in place of T). The machine's load is printed before and after the long steps.
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
- [cultivation_d5_all_runs.py](cultivation_d5_all_runs.py) — the whole d = 5 circuit contracted in
  every one of the 1024 runs its detectors allow, against T†|+⟩ in the colour code of the faces
  (Part 6 of `Cultivation.ipynb` does 4); about nine hours on a laptop, `--start`/`--end` to split
  the runs, `--log FILE`. All 1024 are equal.
- [DecompilationFlow.ipynb](DecompilationFlow.ipynb) — L. Yeh's decompilation flow chart (Section 5 of the paper),
  box by box: a Clifford gate read as its tableau and isometry AP normal form (fold-transversal H, with the Pauli
  frame as a formula in the results); a block that measures read from the AP form of its Choi state; transversal
  T on the [[15,1,3]] code split into a Clifford part and one phase effect (`clifford_part`); and a test that the
  chart's way and the matrix way give the same answers on every block of four programs (`with_route`, `named_by`).
- [StabilizerCodes.ipynb](StabilizerCodes.ipynb) — the three questions for codes that are not CSS. The
  [[5,1,3]] code read off a ring of `CZ` gates (its stabilizers mix `X` and `Z`); its ideal encoder, built
  from the Choi state following Yeh et al., checked with numpy against its definition, and drawn
  after PyZX's spider rules as the ring graph code (five qubits in a ring of CZs, the input copied
  onto all five), and in the isometry AP normal form of Yeh et al. (`iap_normal_form`: a Clifford
  on the inputs, a CSS core, a diagonal Clifford on the outputs); a program that
  encodes, measures the checks and applies `H` then `S` on every qubit, which the tool finds to be the
  logical `SH`, transversal and keeping the code; `H` on every qubit, which leaves it; the Steane code
  after one `S` and its logical `H`; and what is not covered (a code that differs between runs).
- [TransversalGates.ipynb](TransversalGates.ipynb) — the role `transversal gate`: a logical gate whose
  gates never connect two qubits of one code block, directly or through a chain. The transversal `CNOT`
  of two Steane blocks passes; the same `CNOT` with a chain that undoes itself, and a `SWAP` inside the
  [[4,2,2]] code (a logical `CNOT`), do not; `H` on every qubit of the [[4,2,2]] code is transversal but
  its logical action, `SWAP (H x H)` (checked with numpy), is not one of the named gates.
- [CodeSwitching.ipynb](CodeSwitching.ipynb) — the role `code switching`: the Steane code switched to the
  [[15,1,3]] Reed-Muller code by gauge fixing. Eight new qubits in the cube state, three weight-4 Z gauge
  operators measured, and X on a face of the triangle for each outcome 1. The tool finds the two codes (drawn by
  their shapes with `check_diagram`: a triangle and a cube, then a tetrahedron), the identity in every run, and
  circuit distance 2 when the cube state is prepared without checks (one fault in preparing it, one in a gauge
  measurement), 3 once four of its faces are measured first.
- [GateTeleportation.ipynb](GateTeleportation.ipynb) — a logical CX between the two logical qubits of one block
  of the 2×2 toric code [[8,2,2]], by gate teleportation through a second block (L. Yeh's thesis, Eq. (6.42)):
  teleport c into the ancilla, swap c and t there (a reordering of the qubits, `add_permutation`), a
  transversal CX, swap back, teleport c back. First the diagram of Eq. (6.42) on four logical wires, rewritten
  step by step (`to_gpf_steps`) into a CX with the ancilla's t in |0⟩; then the 17-qubit circuit, whose normal
  form is the toric code's encoder after that CX. The tool follows c into the ancilla and back, finds the whole
  program to be a CX from c to t in every run, and circuit distance 1 when c is measured once (one flipped
  result), 2 when it is measured twice (`circuit_distance(..., several_logicals=True)`).
- [GateTeleportationStepByStep.ipynb](GateTeleportationStepByStep.ipynb) — Eq. (6.42) of L. Yeh's thesis
  level by level: the circuit of its first diagram on four logical wires, levelled up with one rewrite per level
  (`'thesis notation'`, `'copy and fuse'`, `'remove identities'`, each made of PyZX's single rules), so that the
  levels' pictures (`reduced_level`) are the five diagrams of the equation, down to a CX from c to t; each checked
  against the circuit in every run, and the levels window (`view_levels`).
- [TransversalCCZ.ipynb](TransversalCCZ.ipynb) — the [[8,3,2]] code (eight qubits on the corners of a cube,
  three logical qubits) and its transversal CCZ, as in L. Yeh's thesis, Section 6.5: the encoder of Eq. (6.27),
  then T on four corners and T† on the other four. The tool finds the code (its checks drawn as a cube), the
  encoder as the identity and the T layer as a CCZ in every run, transversal; the whole circuit reduces to seven
  phase gadgets on x, y, z (phase polynomial π·xyz) followed by the encoder, and E; P; E† to the same CCZ,
  equal to PyZX's CCZ gate.
- [TransversalCCZStepByStep.ipynb](TransversalCCZStepByStep.ipynb) — Eqs. (6.45) and (6.46) of the thesis level
  by level: the circuit of E; P; E†, then one level for each group of CNOTs moved through P (`'phase gadgets'`)
  and one for the spider-nest identity (`'spider nest'`), down to the seven gadgets of the CCZ; each level
  checked against the circuit, and the levels window.
- [MagicStateDistillation.ipynb](MagicStateDistillation.ipynb) — eight T states to one CCZ state with the
  [[8,3,2]] code (L. Yeh's thesis, Eqs. (6.47)-(6.49)): the encoded |+++⟩, the T layer, decoding and five checks.
  The tool finds the protocol to prepare CCZ|+++⟩ in every run with its checks always 0, reduces it to the
  CCZ state, and finds the thesis's form (the magic states decoded directly) the same where its results are 0
  but steered by them otherwise. With independent Z errors on the magic states (`pauli_errors`, all 256
  patterns): every single error caught, every pair missed, output error 28p² to leading order.
- [MagicStateDistillationStepByStep.ipynb](MagicStateDistillationStepByStep.ipynb) — Eq. (6.48) checked as two
  programs, and Eq. (6.49) level by level, one rewrite per level (`'thesis notation'`, `'x-z gpf'`,
  `'copy and fuse'`, `'remove identities'`): the |+⟩ states copied through the decoder, down to the
  three-legged H-box.
- [FindingCorrections.ipynb](FindingCorrections.ipynb) — the tool finds the Pauli corrections a protocol needs
  (`find_corrections`, L. Yeh's "universal gadget fixing": X ones off the Z-X normal form, Z ones off the X-Z
  normal form, then every run checked to be the run with every outcome 0). Teleportation; the gate
  teleportation of [GateTeleportation.ipynb](GateTeleportation.ipynb) with its corrections taken out; a CNOT
  by lattice surgery between three distance-2 surface-code patches (the byproducts, plus an X that repairs a
  check the split re-measured); and a T injection, whose correction is not a Pauli.
- [FindingCorrectionsStepByStep.ipynb](FindingCorrectionsStepByStep.ipynb) — the recipe as levels, one rewrite
  per level (`'gpf'`, `'unfuse outcomes'`, `'x-z gpf'`, `'unfuse outcomes'`), on teleportation and on the
  lattice surgery: the outcome phases left on the outputs are the corrections; each level checked against the
  circuit.
- [ShorCode.ipynb](ShorCode.ipynb) — Shor's nine-qubit code, after the three-qubit bit-flip code of L. Yeh's
  thesis, Section 4.2. The bit-flip code catches every single X but no single Z (Eq. (4.12), every pattern tried
  with `pauli_errors`). Shor's code: the [[9,1,3]] code (equal to PyZX's CSS code from its checks), one round of
  its eight checks the identity in every run with every result 0, the whole circuit reduced to the code's
  encoder (an H in the tool's names of the logical operators: Shor's |0⟩ is the tool's |+⟩); every error on one
  or two qubits noticed; circuit distance 3 with the X checks alternating between blocks, 1 in order (one fault
  on a helper spreads to X on a whole block).
- [ShorCodeConcatenated.ipynb](ShorCodeConcatenated.ipynb) — Shor's code levelled up twice, a code inside a
  code, following the paper's section on it figure for figure: the two codes (read right after the encoder,
  with the tool's names for the logical operators checked against the encoder), the program (one round of
  error correction for each code), the two levels, where each of four errors is fixed (with the results of
  the run), Table 1 for all 31 errors (printed, grouped as in the paper: an X is fixed at level 1, a Z passes
  level 1 as a Z on its block and is fixed at level 2), and the two levels with `rewrite='gpf'` in the levels
  view: Shor's encoder with Z4 left as a sign on one check, then Shor's encoder.
- [ShorCodeStepByStep.ipynb](ShorCodeStepByStep.ipynb) — Eqs. (4.5), (4.7)-(4.10) and (4.12) of the thesis level by
  level, one rewrite per level (`'copy and fuse'`, `'copy results'`, `'spider_simp'`, `'remove identities'`):
  the encoder, the parity check and what it measures, checks pushed into the encoder, a phase flip that is a
  logical Z; then Shor's encoder as the phase-flip code on top of three bit-flip codes, where a phase flip stops
  at its block and an X check catches it.
- [LatticeSurgery.ipynb](LatticeSurgery.ipynb) — a CNOT by lattice surgery on distance-3 surface-code patches
  (Kissinger, *Phase-free ZX diagrams are CSS codes*, Section 7.1), on a 42-qubit chip: a 6×3 control patch split
  in two by one XX check, its lower half merged with a 3×3 target by the two X checks across the seam. The codes
  after every stage; read as one code, the split control still holds one logical qubit (Z_L ⊗ Z_L is a
  stabilizer), so the split is the identity and the merge the CNOT; read as separate patches
  (`StabilizerCode.separated`), the split is a Z-split and the merge a measurement of XX. Levelled up, split and
  merge are one block, a CNOT in every run; the whole program a CNOT up to a Pauli frame (the byproducts). Drawn
  on the chip in space and time.
- [LatticeSurgeryStepByStep.ipynb](LatticeSurgeryStepByStep.ipynb) — Kissinger's Section 7.1 derivations on 3×3
  patches as level-ups, drawn on the chip from above as he draws them: the Z-split of a 6×3 patch (the outcome
  moves into the checks next to the cut, the logical picture is a Z-copy) and the X-merge of two 3×3 patches (kπ on an input, the logical picture a red merge after a green
  kπ); then L. Yeh's recipe on the split as two more level-ups (the X-Z normal form puts the split's outcome
  on the outputs, one on each side of the cut: the corrections), ending with the dagger of the run with every
  outcome 0, after which one outcome phase is left.
- [WhichCodeStepByStep.ipynb](WhichCodeStepByStep.ipynb) — how `code_at` / `code_of` reads a code, one step at a
  time with every matrix printed, on one round of repetition-code checks with its correction (the paper's running
  example): the reduced diagram drawn with the names of its spiders (`code_layers`), the matrices L and M, the X
  stabilizers with outcomes as signs, the Z stabilizers by Gaussian elimination on L over M, k, the generators,
  the logical operators chosen from the stabilizers alone, the distance, and the code after the correction. Each
  stabilizer is checked with its sign on the circuit in every run (`stabilizes`).
- [TInjectionStepByStep.ipynb](TInjectionStepByStep.ipynb) — Question 3 on a T injection, following the
  paper's appendix section for section: the program worked out by hand (T when the result is 0, T^dag when
  it is 1), the codes, steps 1-4 of `logical_action`, the search for a Pauli relating the two classes (none:
  m1 m0^-1 = S^dag, so not deterministic), and the same with `if(c==1) s q[0];` (T in every run).
- [TeleportationStepByStep.ipynb](TeleportationStepByStep.ipynb) — a block with several classes of runs:
  teleportation without its corrections through the four steps of `logical_action`. Step 2 leaves a wire with
  Z^c[0] and X^c[1] and no detectors; step 3 finds four classes, each the identity followed by a different
  Pauli; with the corrections put back there is one class, the identity.
- [TwoRoundsStepByStep.ipynb](TwoRoundsStepByStep.ipynb) — the paper's running example through Questions 1, 2
  and 3: two rounds of repetition-code checks with the error Z0Z1 between them. The code after every stage;
  the second round as the block; the four steps of `logical_action` with every diagram and printout (encoder
  and decoder, composite, the stages of the reduction, detectors, the one class of runs: G = I with the Pauli
  correction Z, the check that the block keeps the code); the runs the detectors allow; and all 36 one- and
  two-qubit errors.
- [StabilizerCodesStepByStep.ipynb](StabilizerCodesStepByStep.ipynb) — a code that is not CSS, step by step:
  the repetition code of the paper with an S on qubit 0, then a T on the parity of the three qubits. The
  generators of the encoder's Choi state, its reduced AP form, the isometry AP normal form E = U_O . C . U_I
  of Yeh et al. (Algorithm 3: U_I empty, the core the repetition code, U_O the S), and the logical action of
  the diagonal block through the normal forms: U_O cancels, the block is checked on the CSS core, one phase
  gadget of pi/4 on the logical bit, a logical T, the same as through the ideal encoders.
- [ComparingPrograms.ipynb](ComparingPrograms.ipynb) — are two programs the same *logically*? A
  demo of `compare_programs`, for comparing a circuit as written with a compiled one: block by block,
  the same codes, the same logical operations up to a Pauli frame, and the same detectors. Four
  variants of the Steane T injection (Z checks measured first and `S` instead of `S†` in the
  correction are equivalent; no correction changes the logical action; a check wired to the wrong
  qubit changes the code), and a mis-wired fold-transversal H, whose codes and logical
  operations are all correct and which only the lost detectors (24 against 22) reveal, located in
  the round with the wrong check. It starts with how PyZX writes a measurement and a fed-forward
  gate (a spider whose phase is the outcome), draws both programs and the stages where they differ,
  shows the ZX diagram after each step of the reduction inside the comparison (`to_gpf_steps`), and
  ends with how long it takes, and why, and what the comparison does not show.
- [LevelUpWithRewrites.ipynb](LevelUpWithRewrites.ipynb) — simplification during level-up: blocks that
  carry a rewrite (`'gpf'`, `'full_reduce'`, ...), the rewritten diagram built from the children's
  rewritten diagrams (`reduced_block`; a whole level, `reduced_level`), the fold-transversal Hadamard and d = 3 cultivation reduced
  block by block with the same result as at once, and the levels window (`view_levels`, and
  `view_levels_3d` with the program in 3D).
- [interactive_levels.py](interactive_levels.py) — the levels window: a slider over the levels, the
  selected block outlined in the program with the blocks it was made from dashed, the program as the
  level rewrites it (every block replaced by its rewritten diagram), and the block's diagram after
  its rewrite; drag a spider to pull overlapping labels apart, or *spread out block* for PyZX's
  force-directed layout; `--3d` draws the program and the level in 3D (`view_levels_3d`; the spiders
  of a rewritten block laid out between the ends of its wires), to be turned with the mouse
  (`python examples/interactive_levels.py [--cultivation] [--3d] [--save FILE]`).
- [interactive_shor_levels.py](interactive_shor_levels.py) — the same window on Shor's code levelled up
  twice ([ShorCodeConcatenated.ipynb](ShorCodeConcatenated.ipynb)): level 1 the encoder, the error and the
  inner round rewritten by `'gpf'`, level 2 everything; for Z4, Shor's encoder with a pi on one check,
  then Shor's encoder (`python examples/interactive_shor_levels.py [--error E] [--save FILE]`, E one of
  X0..Z8, X0X1, X0X3, Z0Z1, Z0Z3, none).
- [DecomposingTerms.ipynb](DecomposingTerms.ipynb) — a block as a sum of terms, one for each value of its
  hidden spiders (`decompose_terms`, L. Yeh's decomposition), and what each term does to each qubit and
  to X, Y, Z errors. On the d = 3 cultivation double-check the terms depend on how the hidden bit is
  labelled: as reduced they link all code qubits; with `'fewest links'` they are "nothing" and T X T†
  on every qubit (Y with S for T). The check for distance: every term maps each Pauli error to the same
  Pauli, up to a sign; it fails with real T and passes with S, while single-qubit Clifford terms are not
  enough. Also the fold-transversal H's blocks and the d = 5 double-checks.
- [RelaxedLayouts.ipynb](RelaxedLayouts.ipynb) — diagrams laid out where a physical energy is lowest,
  with no coordinates given (`relax_layout`, `draw_relaxed`, `draw_relaxed_3d`; Kamada-Kawai springs or a
  force simulation, with networkx): the first checks and the double-check of d = 3 cultivation as
  reduced, and the whole fold-transversal Hadamard; and codes by their shape, for debugging and for
  recognising a code (`check_diagram`, each check shaded): the Steane code's three faces, the same
  whatever the qubits' order, the [[8,3,2]] code as a cube, the surface code as a grid of plaquettes, and
  the [[15,1,3]] code as a tetrahedron cut into four cells (`pip install zxqec[layout]`).
- [DistanceChecking.ipynb](DistanceChecking.ipynb) — L. Yeh's distance-checking pipeline (`check_distance`):
  the stages cut by gate type, the circuit distance of every Clifford run, and the non-Clifford stages
  grouped and checked term by term. On d = 3 cultivation the double-check fails the check, as its fault
  distance with real T (2) says.
- [CircuitDistance.ipynb](CircuitDistance.ipynb) — the circuit distance of Clifford stages
  (`circuit_distance`, with Stim and Mark Webster's `codedistance`; `pip install zxqec[distance]`).
  The fold-transversal H's rounds have circuit distance 2 although every code has distance 3, and
  `explain()` names the two hook errors; the mis-wired ones have 1, and with the CNOTs of two check
  rounds in N order (`build_program(cnot_order='N')`, same noiseless program) 3. Then the Z distance of
  cultivation through its `T` gates (`z_distance_through_t`, which checks that removing them is
  exact, and refuses two small programs where it is not): 3 at d = 3 and 4 at d = 5, with the four
  faults; why the X distance found by removing `T` is not the real one; an independent check
  with Stim alone on the published d = 5 file with `T → S`; and Gidney et al.'s own d = 5 circuit
  (`circuits/cultivation/d5_cultivation_gidney_s.stim`), the same CNOTs and noise, with fault distance 4 too:
  the distance d = 5 cultivation is designed to have before its escape stage.
- [SteaneTInjection.ipynb](SteaneTInjection.ipynb) — a `T` gate injected into the [[7,1,3]] Steane
  code, from the physical circuit to the logical gate. **Written for a reader with no quantum error
  correction background**: it draws the code and its stabilizers, the injection gadget and the
  protocol's stages, and explains each result in plain language. Along the way: a verified encoder
  (its stabilizers and logical operators are derived, not asserted) and its transversal gates; the
  protocol as a 20-qubit `QProgram`; level-up into the logical operations with a measurement report
  per block, showing checks that are meaningless in a block alone and deterministic once it is
  merged; and the result, `T` when the logical readout is 0 and `S†T` when it is 1, always inside
  the code space. The injection is first shown on two qubits in ZX, as in PyZX's
  `demos/BooleanRewrites.ipynb` (the outcome cancels under push-Pauli and fusion); with the
  correction added (`ParityControlled`), the levelled-up 20-qubit protocol reduces to the same
  one-gadget diagram, `T` in every run.
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
  in an interactive window, on the layout of the 19 qubits:
  `python examples/interactive_fold_h_3d.py [top|front|side]` for the whole protocol
  (`--level 2` shades its five phases instead of the nine stages), `--stage I` or
  `--block LEVEL I` for one stage or block only, `--phases` to label the spiders with
  their phases, `--miswired` for the wiring error of an earlier version of the circuit files
  (`ComparingPrograms.ipynb`, Part 3), `--save FILE` to write the figure instead of opening a window.
  `build_program(z_error=Q)` adds one Z on the data qubit at grid position Q right after the Z checks
  that `--miswired` changes (used in `paper/sqz_fold_h`).
- [interactive_fold_h_levels.py](interactive_fold_h_levels.py) — the same protocol in the levels window
  (`view_levels`), three levels rewritten by `'gpf'`: the round in code 2, then the round in code 1,
  then the whole program, a Hadamard followed by Paulis the results determine
  (`python examples/interactive_fold_h_levels.py [--miswired] [--save FILE]`).
- [interactive_fold_h_levels_3d.py](interactive_fold_h_levels_3d.py) — the same three levels in the 3D levels
  window (`view_levels_3d`): the program on the layout of its 19 qubits, time upwards, the blocks as slabs, and
  the level rewritten in 3D, turning together
  (`python examples/interactive_fold_h_levels_3d.py [default|top|front|side] [--miswired] [--save FILE]`).
- [TriorthogonalStepByStep.ipynb](TriorthogonalStepByStep.ipynb) — transversal T on the [[15,1,3]] code,
  following the paper's subsection on triorthogonal codes: the code after the T layer (the same code; with T on
  seven qubits a [[15,4,1]] code), the thesis's matrix M of which gadget reads which spider and its three counts
  (Definition 4.4.1), the phase polynomial that holds the same counts and leaves 7pi/4 on the logical bit
  (T^dag), `summary`, `compare_programs` against "T^dag, then encode" (equivalent; with T on seven qubits not, 1
  detector instead of 4), the detectors after the T layer, and the distances (`z_distance_through_t`: 3).
- [interactive_triorthogonal_levels.py](interactive_triorthogonal_levels.py) — that program in the levels
  window, three levels rewritten by `'gpf'`: the encoder; the encoder and the T layer (one gadget, 7pi/4); the
  whole program (`python examples/interactive_triorthogonal_levels.py [--seven] [--save FILE]`).
- [FoldTransversalHadamardStepByStep.ipynb](FoldTransversalHadamardStepByStep.ipynb) — the fold-transversal
  Hadamard as a program that changes codes, following the paper's section on it figure for figure: the surface
  code drawn on the grid (`StabilizerCode.draw_layout`), the program, the code after every stage (the code with X
  and Z swapped after the Hadamards, and the surface code again after the quarter turn, compared by position),
  `summary` for the six blocks, the whole program (a Hadamard followed by Paulis the results determine, 24
  detectors), the thesis files' check on the wrong qubit (one problem flagged, 22 detectors, a [[9,1,1]] code
  right after it, and one Z error that the correct program catches and the mis-wired one does not), and the
  levels window.

Planned: magic state distillation; cultivation at larger distances.
