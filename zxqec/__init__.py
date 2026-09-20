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

"""zxqec: space-time ZX-diagrams for designing and verifying dynamic QEC protocols.

Built on top of PyZX. The 1+1D time layer is provided by ``pyzx.spacetime``;
this package adds the second spatial dimension, the protocol-level ``QProgram``
abstraction, stabilizer-tableau conversion, a topological view and a GUI.

Subpackages are imported explicitly (``import zxqec.qprogram``) so that optional
dependencies such as matplotlib or ZXLive are only needed where they are used.

"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
