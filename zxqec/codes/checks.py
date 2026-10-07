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

"""The checks of a CSS code as a ZX-diagram, to see the code's shape.

A parity-check matrix is hard to read, and it changes when the qubits are
numbered differently. The diagram of the checks does not: :func:`check_diagram`
gives one spider per check joined to one spider per qubit, each qubit with an
output, which is the code state in parity normal form (PyZX's
``generate_css_encoder_graph`` without logical operators). Laid out by
:func:`~zxqec.spacetime.relax_layout` (``draw_relaxed`` and ``draw_relaxed_3d``,
with ``shade`` to colour each check), only the connections decide the picture,
so the faces of the Steane code, the plaquettes of the surface code and the cells
of the tetrahedral ``[[15,1,3]]`` code appear whatever the column order, and a
wrong entry shows as a misshapen face.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyzx.graph.base import BaseGraph

__all__ = ['check_diagram', 'surface_code_checks']


def check_diagram(checks: Any, kind: str = 'X') -> BaseGraph:
    """The checks as a phase-free ZX-diagram: a spider per check, joined to a spider on each of its qubits.

    ``checks`` is a 0/1 matrix (one row per check, one column per qubit) or a
    CSS :class:`~zxqec.codes.StabilizerCode`, whose generators of type ``kind``
    are taken (signs are left out). ``kind='X'``: X-checks, the code state
    ``|0...0>_L`` in Z-X parity form, a Z-spider per check and an X-spider per
    qubit; ``kind='Z'``: Z-checks, the state ``|+...+>_L`` in X-Z form, the
    colours swapped. Each qubit spider has one output.
    """
    from pyzx.css import generate_css_encoder_graph
    from pyzx.linalg import Mat2

    from .code import StabilizerCode
    if kind not in ('X', 'Z'):
        raise ValueError("kind must be 'X' or 'Z', got {!r}".format(kind))
    if isinstance(checks, StabilizerCode):
        if not checks.is_css:
            raise ValueError("check_diagram needs a CSS code")
        pure = [p for p in checks.stabilizers if not any(p.z if kind == 'X' else p.x)]
        rows = np.array([p.x if kind == 'X' else p.z for p in pure], dtype=np.uint8).reshape(len(pure), checks.n)
    else:
        rows = np.asarray(checks, dtype=np.uint8) % 2
    if rows.ndim != 2 or not rows.shape[0]:
        raise ValueError("expected a non-empty matrix of checks")
    g, _ = generate_css_encoder_graph(Mat2([[int(b) for b in r] for r in rows]), None,
                                      'Z-X' if kind == 'X' else 'X-Z')
    g.auto_detect_io()
    return g


def surface_code_checks(d: int) -> tuple[np.ndarray, np.ndarray]:
    """The X- and Z-checks of the rotated surface code of distance ``d``.

    Qubit ``r * d + c`` sits at row ``r``, column ``c`` of a ``d x d`` grid. The
    checks are the faces between four qubits, X and Z in a checkerboard, and the
    faces of two qubits on the boundary: X-checks on the top and bottom edges,
    Z-checks on the left and right. One logical qubit; for odd ``d``,
    ``(d^2 - 1) / 2`` checks of each kind.
    """
    if d < 2:
        raise ValueError("d must be at least 2")
    rows: dict[str, list[list[int]]] = {'X': [], 'Z': []}
    for r in range(-1, d):
        for c in range(-1, d):
            cells = [(r + a, c + b) for a in (0, 1) for b in (0, 1) if 0 <= r + a < d and 0 <= c + b < d]
            kind = 'X' if (r + c) % 2 == 0 else 'Z'
            inside = len(cells) == 4
            edge = len(cells) == 2 and ((kind == 'X' and r in (-1, d - 1)) or (kind == 'Z' and c in (-1, d - 1)))
            if inside or edge:
                row = [0] * (d * d)
                for a, b in cells:
                    row[a * d + b] = 1
                rows[kind].append(row)
    return np.array(rows['X'], dtype=np.uint8), np.array(rows['Z'], dtype=np.uint8)
