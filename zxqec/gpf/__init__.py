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

"""Generalized parity form (GPF) with phases: extraction and reduction.

* :func:`to_gpf` -- Algorithm 2 of the thesis generalised to Pauli (incl.
  Boolean-symbolic) phases on the spiders; non-Pauli phases become gadgets;
* :func:`reduce_gpf` -- Algorithm 3: split-matrix row reduction over GF(2) of
  both layers with the phases carried along, yielding a minimal GPF, the
  detectors (Boolean parities of outcomes that must vanish) and a zero flag;
* :func:`equal_for_all_assignments` -- tensor equality for every assignment of
  the Boolean variables, the verification used throughout the tests (only the
  runs the detectors allow, :func:`consistent_assignments`, when they are linear);
* :func:`change_variables` -- a state in new summation variables, with given
  rows for its Z-layer (for example a code's logical operator and generators);
* :func:`layout_gpf` and :func:`phases_modulo` -- a GPF diagram laid out in
  layers, with its phases written modulo the detectors, for PyZX's drawing
  functions (``zx.draw``, ``zx.draw_matplotlib``);
* :func:`measurement_report` -- what happens to every measurement outcome of a
  diagram (frame / fixed / tied / gauge), its detectors, and which of them fire
  for the outcomes of a given run.
* :func:`decompose_terms` -- a reduced diagram as a sum of terms, one for each
  value of its hidden spiders, and what each term does to each qubit and to
  X, Y, Z errors.
"""

from .gadgets import phase_polynomial, simplify_gadgets
from .hadamard import (BOUNDARY_HADAMARD_KEY, has_boundary_hadamard, push_hadamards_to_boundary,
                       restore_boundary_hadamards)
from .matrix import SplitMatrix, apply_matrix, change_variables, reduce_rows, split_matrix
from .normal_form import GPF_ROLE_KEY, LEAF, is_gpf, is_leaf, is_stem, layout_gpf, to_gpf
from .reduce import (GPFResult, boolean_variables, consistent_assignments, constraint_system, detector_rank,
                     equal_for_all_assignments, evaluate_phase, phases_modulo, reduce_gpf, reduce_modulo,
                     same_map, sample_consistent_assignment, unsatisfiable)
from .report import Measurement, MeasurementReport, measurement_report
from .terms import Decomposition, Part, Piece, Term, decompose_terms

__all__ = [
    'to_gpf', 'is_gpf', 'is_leaf', 'is_stem', 'layout_gpf', 'GPF_ROLE_KEY', 'LEAF',
    'SplitMatrix', 'split_matrix', 'reduce_rows', 'apply_matrix', 'change_variables',
    'reduce_gpf', 'GPFResult', 'equal_for_all_assignments', 'boolean_variables', 'evaluate_phase',
    'sample_consistent_assignment', 'consistent_assignments', 'same_map', 'constraint_system',
    'detector_rank', 'unsatisfiable', 'reduce_modulo', 'phases_modulo',
    'measurement_report', 'MeasurementReport', 'Measurement',
    'push_hadamards_to_boundary', 'restore_boundary_hadamards', 'has_boundary_hadamard',
    'BOUNDARY_HADAMARD_KEY', 'simplify_gadgets', 'phase_polynomial',
    'decompose_terms', 'Decomposition', 'Part', 'Term', 'Piece',
]
