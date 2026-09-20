# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Repository skeleton: package layout (`qprogram`, `spacetime`, `tableau`, `topological`, `gui`), packaging, tests.
- `zxqec.qprogram.QProgram`: programs as sequences of circuit stages and permutations on a hardware layout; master space-time graph with `stage` / `logical_qubit` vertex data on top of `pyzx.spacetime`; stage and block sub-diagrams; measured qubits reborn by `Reset`; per-stage renaming of PyZX's generated symbolic variables.
- `zxqec.qprogram.LevelManager` / `Block`: level-up of consecutive stages and merging of blocks separated by stages on disjoint qubits (port of the thesis `GateLevelManager`).
- `zxqec.spacetime.Topology` (port of the thesis class, with `grid` / `line` constructors) and `induced_subdiagram`, which cuts any set of spiders out of a program graph as a valid ZX-diagram, keeping idle wires and attributing every edge to its later endpoint.
