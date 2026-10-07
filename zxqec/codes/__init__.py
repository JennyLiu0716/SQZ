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

"""Codes and logical operations: questions 1 and 2 of the paper.

* :func:`code_of` -- the stabilizer code a Clifford isometry encodes into, read
  off its generalized parity form, with outcome-dependent signs and canonical
  logical operators (:class:`StabilizerCode`);
* :class:`PauliString` -- a Pauli operator with a possibly symbolic sign;
* :func:`iap_normal_form` -- a Clifford isometry (such as a code's ideal
  encoder) in isometry affine-with-phases normal form (:class:`IAPNormalForm`),
  computed from :func:`reduced_ap_form`, the reduced AP form of its Choi state
  (:class:`APForm`).
* :func:`check_diagram` -- the checks of a CSS code as a ZX-diagram, to see its
  shape with :func:`~zxqec.spacetime.draw_relaxed`; :func:`surface_code_checks`.
"""

from .checks import check_diagram, surface_code_checks
from .code import NotACodeError, OutcomeDependentCodeError, StabilizerCode, code_of
from .iapnf import APForm, IAPNormalForm, iap_normal_form, reduced_ap_form
from .logical import Branch, LogicalAction, glue, logical_action
from .pauli import PauliString, phase_sum

__all__ = ['StabilizerCode', 'NotACodeError', 'OutcomeDependentCodeError', 'code_of', 'PauliString', 'phase_sum',
           'logical_action', 'LogicalAction', 'Branch', 'glue', 'iap_normal_form', 'IAPNormalForm',
           'reduced_ap_form', 'APForm', 'check_diagram', 'surface_code_checks']
