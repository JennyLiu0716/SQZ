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

"""2+1D space-time programs: two spatial dimensions plus time.

Builds on :mod:`pyzx.spacetime` (per-spider ``timestep`` / ``delay`` vertex
data, ASAP scheduling, ``time_slice``):

* :class:`Topology` -- hardware connectivity with 1D/2D/3D node coordinates;
* :mod:`~zxqec.spacetime.subdiagram` -- ``stage`` / ``logical_qubit`` vertex
  data and :func:`induced_subdiagram`, which cuts any set of spiders out of a
  program graph as a valid ZX-diagram;
* (planned) the 3D view, 2D space + 1D time, of a space-time ZX-diagram.
"""

from .subdiagram import (
    LOGICAL_QUBIT_KEY, STAGE_KEY, get_logical_qubit, get_stage, induced_subdiagram,
    set_logical_qubit, set_stage, shift_timesteps,
)
from .topology import Topology

__all__ = [
    'Topology',
    'STAGE_KEY', 'LOGICAL_QUBIT_KEY',
    'get_stage', 'set_stage', 'get_logical_qubit', 'set_logical_qubit',
    'shift_timesteps', 'induced_subdiagram',
]
