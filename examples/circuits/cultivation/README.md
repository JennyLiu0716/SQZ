# Magic-state cultivation

## d = 3

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

## d = 5

`d5_cultivation.stim` is the d = 5 protocol with real `T` gates: injection, the d = 3 double-check,
growth to d = 5 with three rounds of checks, and the d = 5 double-check. It is lines 1-465 of
`legacy/softv1/magic_state_cultivation/circuits/circuit_d5_p0.001.stim` in
[haoliri0/SOFT](https://github.com/haoliri0/SOFT) (Apache License 2.0), unchanged. The file's last 25
lines are in `d5_final_check.stim`, also unchanged: a noiseless check of the output that the authors
wrote for simulation, not part of the protocol. It measures the 18 stabilizers of the code as products of
Paulis (`MPP`), with a `DETECTOR` for each, then applies a layer of `T` and `T_DAG` gates and measures
the product of Y on the 19 code qubits, the end of `OBSERVABLE_INCLUDE(0)`. Its `rec[-k]` refer to the
protocol's measurements, so it is read after it; the two files together are the published file, line
for line.

It follows Gidney, Shutty and Jones with their *unitary* injection, so its first stages differ from
the d = 3 file above, which uses a different injection. Gidney et al.'s own circuits (Zenodo record
10.5281/zenodo.13777072) use `S` and `S_DAG` in place of `T` and `T_DAG`, so that Stim can simulate
them; this one has the real gates. Its noise lines (`DEPOLARIZE1`, `X_ERROR`, ...) and the flip
probabilities of noisy measurements are kept as published; `program_from_stim` skips them. Its
`DETECTOR` lines are used, as for d = 3, only for comparison.
