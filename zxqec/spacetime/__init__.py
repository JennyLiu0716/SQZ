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
* :mod:`~zxqec.spacetime.drawing` -- :func:`draw_program` (2D, time axis,
  block shading) and :func:`draw_program_3d` / :func:`draw_graph_3d` (2D space
  + 1D time) with matplotlib, for scripts and static figures; :func:`draw_level_3d`
  for the program as a level rewrites it;
* :mod:`~zxqec.spacetime.web` -- the same views as interactive JavaScript
  widgets for notebooks (:func:`draw_web`, :func:`draw_program_web`),
  built on PyZX's d3 and three.js viewers;
* a window for walking through the levels of a program, block by block, with
  each block's diagram before and after its rewrite (:func:`view_levels`), and
  the same with the program in 3D (:func:`view_levels_3d`).
"""

from .subdiagram import (
    LOGICAL_QUBIT_KEY, STAGE_KEY, get_logical_qubit, get_stage, induced_subdiagram,
    set_logical_qubit, set_stage, shift_timesteps,
)
from .drawing import (VIEWS, draw_graph_3d, draw_level_3d, draw_program, draw_program_3d, draw_program_views,
                      level_layout_3d)
from .topology import Topology
from .web import draw_program_web, draw_web
from .levels_view import LevelsView, LevelsView3D, view_levels, view_levels_3d

__all__ = [
    'Topology',
    'STAGE_KEY', 'LOGICAL_QUBIT_KEY',
    'get_stage', 'set_stage', 'get_logical_qubit', 'set_logical_qubit',
    'shift_timesteps', 'induced_subdiagram',
    'draw_program', 'draw_program_3d', 'draw_graph_3d', 'draw_program_views', 'VIEWS',
    'draw_level_3d', 'level_layout_3d',
    'draw_web', 'draw_program_web',
    'view_levels', 'LevelsView', 'view_levels_3d', 'LevelsView3D',
]
