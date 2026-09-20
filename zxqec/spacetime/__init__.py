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

Builds on ``pyzx.spacetime`` (per-spider ``timestep`` / ``delay`` vertex data,
ASAP scheduling, ``time_slice``). Planned contents:

* ``Topology`` -- hardware connectivity with 1D/2D/3D node coordinates and a
  qubit-to-node mapping, used to validate permutation layers;
* spatial coordinates of spiders stored as vertex data;
* a 3D view (2D space + 1D time) of a space-time ZX-diagram, with block
  shading for the levels of a ``QProgram``.

"""
