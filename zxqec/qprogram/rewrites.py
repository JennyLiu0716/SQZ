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

"""Simplification during level-up: a block can carry a rewrite, applied when it is levelled up.

A level-up groups blocks; by default that is all it does, and every analysis cuts
the block's diagram out of the master graph. A block can also name a *rewrite*
(``QProgram.level_up(..., rewrite='gpf')``). :meth:`QProgram.reduced_block` then
builds the block's diagram from its children: each child's diagram, rewritten if
the child has a rewrite of its own, glued in order along the qubit wires, and the
block's own rewrite applied to the result. So small blocks are simplified first and
the simplified pieces combined, level by level. A block without rewrites anywhere
below it comes out as its cut diagram. :meth:`QProgram.reduced_level` glues all
the blocks of a level in the same way: the whole program with every block of that
level replaced by its rewritten diagram (a :class:`ReducedLevel`).

The rewrites known by name (:data:`REWRITES`):

* ``'gpf'`` -- :func:`~zxqec.gpf.reduce_gpf`, gadget layer simplified; the
  detectors it removes are kept with the result;
* ``'gpf-no-gadgets'`` -- the same with the gadgets held fixed;
* ``'full_reduce'``, ``'clifford_simp'``, ``'spider_simp'`` -- PyZX's simplifiers;
* ``'teleport_reduce'`` -- PyZX's phase teleportation, which keeps the shape of
  the diagram and only moves phases;
* ``'thesis notation'``, ``'copy and fuse'``, ``'remove identities'`` -- the steps
  of a derivation by hand (:mod:`zxqec.gpf.rules`): measurements as effects and
  swaps as crossing wires; states copied and one-legged spiders fused; identities
  removed and cancelling phases fused. Levelling up one part of a circuit after
  another with them gives the diagrams of Eq. (6.42) of L. Yeh's thesis.

:func:`register_rewrite` adds more: a function from a diagram to a diagram, to a
``GPFResult``, or to ``(diagram, detectors)``. Every rewrite must keep the linear
map, up to a non-zero scalar, on the outcomes that satisfy the detectors it
reports. The rewritten diagram is a view: the three questions are still answered
from the cut diagram.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pyzx.graph.base import BaseGraph
from pyzx.utils import VertexType

from .levels import Block

__all__ = ['REWRITES', 'register_rewrite', 'apply_rewrite', 'ReducedBlock', 'ReducedLevel', 'check_rewrite']


@dataclass
class _Rewritten:
    graph: BaseGraph
    detectors: list[Any] = field(default_factory=list)
    is_zero: bool = False


def _gpf(gadgets: bool) -> Callable[[BaseGraph], Any]:
    def run(g: BaseGraph) -> Any:
        from ..gpf import GPF_ROLE_KEY, reduce_gpf
        for v in g.vertices():               # markings of an earlier reduction (a rewritten child) no longer
            if g.vdata(v, GPF_ROLE_KEY, None) is not None:      # hold once it is glued to its neighbours
                g.set_vdata(v, GPF_ROLE_KEY, None)
        return reduce_gpf(g, gadgets=gadgets)
    return run


def _pyzx(name: str) -> Callable[[BaseGraph], Any]:
    def run(g: BaseGraph) -> Any:
        import pyzx.simplify as simplify
        h = g.copy()
        if name == 'teleport_reduce':
            return simplify.teleport_reduce(h)
        f = getattr(simplify, name)
        if name in ('full_reduce', 'clifford_simp'):
            f(h, quiet=True)
        else:
            f(h)
        return h
    return run


def _rules(name: str) -> Callable[[BaseGraph], Any]:
    def run(g: BaseGraph) -> Any:
        from ..gpf import rules
        return getattr(rules, name)(g.copy())
    return run


#: name -> function from a diagram to a diagram, a ``GPFResult`` or ``(diagram, detectors)``
REWRITES: dict[str, Callable[[BaseGraph], Any]] = {
    'gpf': _gpf(True),
    'gpf-no-gadgets': _gpf(False),
    'full_reduce': _pyzx('full_reduce'),
    'clifford_simp': _pyzx('clifford_simp'),
    'spider_simp': _pyzx('spider_simp'),
    'teleport_reduce': _pyzx('teleport_reduce'),
    'thesis notation': _rules('thesis_notation'),
    'copy and fuse': _rules('copy_and_fuse'),
    'remove identities': _rules('remove_identities'),
}


def register_rewrite(name: str, function: Callable[[BaseGraph], Any]) -> None:
    """Make ``function`` available as ``rewrite=name`` (it must keep the map up to a scalar)."""
    if not callable(function):
        raise TypeError("a rewrite must be a function of a diagram")
    REWRITES[name] = function


def check_rewrite(name: str | None) -> None:
    if name is not None and name not in REWRITES:
        raise ValueError("unknown rewrite {!r}; known: {} (register_rewrite adds more)".format(
            name, ', '.join(sorted(REWRITES))))


def apply_rewrite(name: str, g: BaseGraph) -> _Rewritten:
    """``g`` rewritten by the rewrite ``name``, with the detectors the rewrite removed."""
    check_rewrite(name)
    out = REWRITES[name](g.copy())
    if hasattr(out, 'to_graph') and hasattr(out, 'constraints'):      # a GPFResult
        return _Rewritten(out.to_graph(), [c for c in out.constraints if hasattr(c, 'free_vars')
                                           and c.free_vars()], bool(out.is_zero))
    if isinstance(out, tuple):
        graph, detectors = out
        return _Rewritten(graph, list(detectors))
    return _Rewritten(out)


def _spiders(g: BaseGraph) -> int:
    return sum(1 for v in g.vertices() if g.type(v) != VertexType.BOUNDARY)


@dataclass
class ReducedBlock:
    """A block's diagram after the rewrites of its level-ups (see the module docstring).

    Its phases are exact but need not be written as reducing the whole block at
    once would write them: a parity of outcomes may carry detector parities
    (always 0) picked up from the children. ``phases_modulo(graph, detectors)``
    (:mod:`zxqec.gpf`) writes them in the earliest outcomes possible, the same
    for any two diagrams equal in every run the detectors allow; the levels
    window draws that.
    """
    block: Block
    graph: BaseGraph                       #: the diagram, rewritten if the block has a rewrite
    children: tuple['ReducedBlock', ...]   #: the children it was glued from (empty for a stage)
    detectors: tuple[Any, ...]             #: every detector removed on the way, by this block or below
    new_detectors: tuple[Any, ...]         #: the ones this block's rewrite removed
    spiders_before: int                    #: spiders of the glued diagram, before this block's rewrite
    spiders_after: int
    seconds: float                         #: time of this block's gluing and rewrite (children not included)
    is_zero: bool = False                  #: the rewrite found the diagram to be zero in every run

    @property
    def rewrite(self) -> str | None:
        return self.block.rewrite

    def describe(self) -> str:
        what = "rewrite {}".format(self.rewrite) if self.rewrite else "no rewrite"
        line = "{} ({}): {} spiders -> {} after {}".format(
            self.block, ', '.join("stage {}".format(s) for s in self.block.stages) if self.block.is_leaf
            else "{} children".format(len(self.children)), self.spiders_before, self.spiders_after, what)
        if self.new_detectors:
            line += "; {} detector(s) removed here".format(len(self.new_detectors))
        if self.is_zero:
            line += "; zero in every run"
        return line

    def __str__(self) -> str:
        return self.describe()


def reduce_block(block: Block, cut: Callable[[Block], BaseGraph], glue: Callable[[BaseGraph, BaseGraph], None],
                 cache: dict[Block, ReducedBlock]) -> ReducedBlock:
    """The :class:`ReducedBlock` of ``block``: ``cut`` gives a leaf's diagram, ``glue`` appends a diagram."""
    if block in cache:
        return cache[block]
    children = tuple(reduce_block(c, cut, glue, cache) for c in block.children)
    t0 = time.perf_counter()
    if block.is_leaf:
        base = cut(block)
    else:
        base = children[0].graph.copy()
        for c in children[1:]:
            glue(base, c.graph)
        _order_boundaries(base)
    below = tuple(d for c in children for d in c.detectors)
    if block.rewrite is None:
        out = ReducedBlock(block, base, children, below, (), _spiders(base), _spiders(base),
                           time.perf_counter() - t0, any(c.is_zero for c in children))
    else:
        r = apply_rewrite(block.rewrite, base)
        out = ReducedBlock(block, r.graph, children, below + tuple(r.detectors), tuple(r.detectors),
                           _spiders(base), _spiders(r.graph), time.perf_counter() - t0,
                           r.is_zero or any(c.is_zero for c in children))
    cache[block] = out
    return out


@dataclass
class ReducedLevel:
    """The whole program as a level rewrites it: every block of the level after the rewrites of its
    level-ups (:class:`ReducedBlock`), glued in order along the qubit wires."""
    level: int
    blocks: tuple[ReducedBlock, ...]
    graph: BaseGraph
    vertices: tuple[frozenset, ...]        #: the vertices of ``graph`` that come from each block, in block order
    detectors: tuple[Any, ...]             #: every detector the level's blocks removed, by them or below

    @property
    def spiders(self) -> int:
        return _spiders(self.graph)

    def describe(self) -> str:
        return "level {}: {} blocks glued, {} spiders; {} detector(s) removed".format(
            self.level, len(self.blocks), self.spiders, len(self.detectors))

    def __str__(self) -> str:
        return self.describe()


def reduce_level(level: int, blocks: list[ReducedBlock],
                 glue: Callable[[BaseGraph, BaseGraph], None]) -> ReducedLevel:
    """The :class:`ReducedLevel` of the blocks of ``level``: their diagrams glued in order by ``glue``."""
    graph = blocks[0].graph.copy()
    groups = [set(graph.vertices())]
    for rb in blocks[1:]:
        before = set(graph.vertices())
        glue(graph, rb.graph)
        groups.append(set(graph.vertices()) - before)
    _order_boundaries(graph)
    alive = set(graph.vertices())          # gluing removes the boundaries it plugs
    return ReducedLevel(level, tuple(blocks), graph, tuple(frozenset(g & alive) for g in groups),
                        tuple(d for rb in blocks for d in rb.detectors))


def _order_boundaries(g: BaseGraph) -> None:
    """Inputs and outputs ordered by logical qubit, as in a cut diagram."""
    from ..spacetime.subdiagram import get_logical_qubit

    def key(b: Any) -> tuple:
        q = get_logical_qubit(g, b)
        return (q is None, q if q is not None else g.qubit(b), g.row(b))
    g.set_inputs(tuple(sorted(g.inputs(), key=key)))
    g.set_outputs(tuple(sorted(g.outputs(), key=key)))
