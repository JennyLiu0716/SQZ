# d = 3 magic-state cultivation

`d3_cultivation.stim` is the d = 3 cultivation circuit (injection, a round of checks, and the
double-check) copied without change from
[kh428/accel-cutting-magic-state](https://github.com/kh428/accel-cutting-magic-state)
(`circuit_source_injection_T` in `d_3_circuit_definitions.py`, Apache License 2.0), the companion code of

> K. H. Wan, Z. Zhong, A. Zapirain, *Simulating magic state cultivation with few Clifford terms*,
> Quantum 10, 2134 (2026), arXiv:2509.08658,

which implements the protocol of

> C. Gidney, N. Shutty, C. Jones, *Magic state cultivation: growing T states as cheap as CNOT gates*,
> arXiv:2409.17595.

The file is in Stim's text format with `T` and `T_DAG` added. Its `DETECTOR` lines are the detectors
the authors wrote by hand; zxqec does not use them to build the program, only to compare against the
detectors it derives (`zxqec.qprogram.program_from_stim`). The second circuit of the original file,
a noiseless projection used to check the output in simulation, is not part of the protocol and is not
included.
