# zxqec - space-time ZX-diagrams for dynamic quantum error correction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protocol-level abstraction over PyZX circuits.

* :class:`QProgram` -- a protocol as a sequence of circuit stages and qubit
  permutations on a hardware layout, producing one master space-time
  ZX-diagram on demand and the sub-diagram of any stage or block;
* :class:`LevelManager` / :class:`Block` -- hierarchical *level-up* of
  consecutive stages into blocks (a syndrome-extraction round, a logical gate)
  and merging of blocks separated only by stages on other qubits; a level-up can
  also rewrite the block it makes (``rewrite='gpf'``, ``'full_reduce'``, ...;
  :meth:`QProgram.reduced_block`, :meth:`QProgram.reduced_level` for a whole level,
  :mod:`~zxqec.qprogram.rewrites`);
* :class:`ParityControlled` -- a correction applied when the XOR of some
  measurement outcomes is 1 (feed-forward), kept symbolic in the diagram;
* :func:`program_from_stim` -- a program from a circuit in Stim's text format;
* :func:`compare_programs` -- two programs compared block by block: the same
  codes, the same logical operations up to a Pauli frame, the same detectors;
* :func:`check_detectors` -- declared detectors (e.g. Stim's ``DETECTOR`` lines)
  checked against the ones the tool derives: deterministic with their value,
  not deterministic with two runs as evidence, or unknown;
* :func:`circuit_distance` -- the circuit distance of Clifford stages, with Stim
  and ``codedistance`` (optional dependencies); :func:`z_distance_through_t` --
  the Z distance of stages with ``T`` gates, after checking that removing them is exact;
* :func:`check_distance` -- L. Yeh's distance-checking pipeline: stages cut by gate
  type (:func:`split_by_gate_type`), circuit distances of the Clifford runs, and the
  non-Clifford stages grouped and checked term by term (:func:`~zxqec.gpf.decompose_terms`).
"""

from .compare import BlockComparison, ProgramComparison, compare_programs
from .declared import DeclaredDetector, DetectorCheck, check_detectors
from .distance import (CircuitDistance, Fault, NotCliffordError, NotPhaseFreeError, TRemovalError, circuit_distance,
                       without_gates, z_distance_through_t)
from .distance_check import DistanceCheck, GroupCheck, SplitProgram, check_distance, split_by_gate_type
from .feedforward import ParityControlled
from .levels import Block, LevelManager
from .program import QProgram, Stage, same_layer
from .rewrites import REWRITES, ReducedBlock, ReducedLevel, register_rewrite
from .stim_text import StimCircuit, parse_stim, program_from_stim

__all__ = ['QProgram', 'Stage', 'same_layer', 'Block', 'LevelManager', 'ParityControlled',
           'StimCircuit', 'parse_stim', 'program_from_stim',
           'BlockComparison', 'ProgramComparison', 'compare_programs',
           'DeclaredDetector', 'DetectorCheck', 'check_detectors',
           'CircuitDistance', 'Fault', 'NotCliffordError', 'NotPhaseFreeError', 'TRemovalError', 'circuit_distance',
           'without_gates', 'z_distance_through_t', 'REWRITES', 'ReducedBlock', 'ReducedLevel', 'register_rewrite',
           'check_distance', 'split_by_gate_type', 'SplitProgram', 'GroupCheck', 'DistanceCheck']
