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
  the Boolean variables, the verification used throughout the tests;
* :func:`measurement_report` -- what happens to every measurement outcome of a
  diagram (frame / fixed / tied / gauge), its detectors, and which of them fire
  for the outcomes of a given run.
"""

from .hadamard import (BOUNDARY_HADAMARD_KEY, has_boundary_hadamard, push_hadamards_to_boundary,
                       restore_boundary_hadamards)
from .matrix import SplitMatrix, apply_matrix, reduce_rows, split_matrix
from .normal_form import GPF_ROLE_KEY, LEAF, is_gpf, is_leaf, is_stem, layout_gpf, to_gpf
from .reduce import (GPFResult, boolean_variables, equal_for_all_assignments, evaluate_phase, reduce_gpf,
                     sample_consistent_assignment)
from .report import Measurement, MeasurementReport, measurement_report

__all__ = [
    'to_gpf', 'is_gpf', 'is_leaf', 'is_stem', 'layout_gpf', 'GPF_ROLE_KEY', 'LEAF',
    'SplitMatrix', 'split_matrix', 'reduce_rows', 'apply_matrix',
    'reduce_gpf', 'GPFResult', 'equal_for_all_assignments', 'boolean_variables', 'evaluate_phase',
    'sample_consistent_assignment',
    'measurement_report', 'MeasurementReport', 'Measurement',
    'push_hadamards_to_boundary', 'restore_boundary_hadamards', 'has_boundary_hadamard',
    'BOUNDARY_HADAMARD_KEY',
]
