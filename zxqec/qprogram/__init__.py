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
  and merging of blocks separated only by stages on other qubits.
"""

from .levels import Block, LevelManager
from .program import QProgram, Stage

__all__ = ['QProgram', 'Stage', 'Block', 'LevelManager']
