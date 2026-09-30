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

"""Hierarchical grouping ("level-up") of the stages of a program into blocks.

Level 0 has one :class:`Block` per stage. :meth:`LevelManager.level_up` wraps a
range of consecutive blocks of a level into one composite block, producing the
next level; :meth:`LevelManager.merge_blocks` does the same for two blocks that
are not adjacent, provided everything in between acts on disjoint qubits, so
the later block can be slid next to the earlier one without changing the
program. A block is described by the set of stage indices it covers; the
program turns that into a circuit or a sub-diagram.

Ported from the thesis ``GateLevelManager``, with stages instead of single
gates as the atomic unit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

__all__ = ['Block', 'LevelManager', 'ROLES']

#: the stage types of fault-tolerant protocols that :meth:`QProgram.summary` knows how to check
ROLES = ('encoder', 'decoder', 'state preparation', 'syndrome extraction', 'logical gate',
         'transversal gate', 'logical measurement', 'injection', 'correction', 'distillation',
         'cultivation')


@dataclass(frozen=True)
class Block:
    """A block of a level: the stages it covers and, if composite, its children."""
    stages: tuple[int, ...]
    children: tuple['Block', ...] = ()
    name: str | None = None
    role: str | None = None          #: what the block is for, e.g. 'syndrome extraction' (see ROLES)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a block must cover at least one stage")
        object.__setattr__(self, 'stages', tuple(sorted(set(self.stages))))

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def first(self) -> int:
        return self.stages[0]

    @property
    def last(self) -> int:
        return self.stages[-1]

    @property
    def contiguous(self) -> bool:
        """Whether the block covers a range of stages without gaps."""
        return self.last - self.first + 1 == len(self.stages)

    def leaves(self) -> list['Block']:
        if self.is_leaf:
            return [self]
        out: list[Block] = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def __str__(self) -> str:
        label = self.name or ("stage {}".format(self.first) if len(self.stages) == 1
                              else "stages {}".format(",".join(map(str, self.stages))))
        return label


class LevelManager:
    """The levels of blocks over ``num_stages`` stages.

    Level 0 is one leaf block per stage. Higher levels are created by
    :meth:`level_up` / :meth:`merge_blocks`; creating a new level above ``k``
    discards every level above ``k`` that existed before. Appending a stage
    adds a new leaf block at the end of *every* level.
    """

    def __init__(self, num_stages: int = 0) -> None:
        self._levels: list[list[Block]] = [[Block((i,)) for i in range(num_stages)]]

    # -- structure ----------------------------------------------------------

    @property
    def num_levels(self) -> int:
        return len(self._levels)

    @property
    def num_stages(self) -> int:
        return len(self._levels[0])

    def level_size(self, level: int) -> int:
        return len(self._level(level))

    def blocks(self, level: int) -> list[Block]:
        """The blocks of ``level`` (a copy of the list)."""
        return list(self._level(level))

    def block(self, level: int, index: int) -> Block:
        blocks = self._level(level)
        if not 0 <= index < len(blocks):
            raise IndexError("level {} has {} blocks, no block {}".format(level, len(blocks), index))
        return blocks[index]

    def append_stage(self, name: str | None = None, role: str | None = None) -> int:
        """Register a new stage; returns its index."""
        i = self.num_stages
        for blocks in self._levels:
            blocks.append(Block((i,), name=name, role=role))
        return i

    def _level(self, level: int) -> list[Block]:
        if not 0 <= level < len(self._levels):
            raise IndexError("level {} does not exist ({} levels)".format(level, len(self._levels)))
        return self._levels[level]

    def _push_level(self, level: int, blocks: list[Block]) -> int:
        del self._levels[level + 1:]
        self._levels.append(blocks)
        return level + 1

    # -- grouping -----------------------------------------------------------

    def level_up(self, level: int, start: int, end: int, name: str | None = None,
                 role: str | None = None) -> int:
        """Group blocks ``start..end`` (inclusive) of ``level`` into one block.

        ``role`` records what the block is for (one of :data:`ROLES`, or any
        string); :meth:`QProgram.summary` checks the block against it.

        Returns the index of the new level, whose blocks are those of ``level``
        with the range replaced by the composite block (at position ``start``).
        """
        blocks = self._level(level)
        if not 0 <= start <= end < len(blocks):
            raise IndexError("invalid range [{}, {}] for level {} with {} blocks".format(
                start, end, level, len(blocks)))
        children = tuple(blocks[start:end + 1])
        composite = Block(tuple(s for c in children for s in c.stages), children, name, role)
        return self._push_level(level, blocks[:start] + [composite] + blocks[end + 1:])

    def merge_blocks(self, level: int, i: int, j: int,
                     block_qubits: Callable[[Block], Iterable[int]],
                     name: str | None = None, role: str | None = None) -> int | None:
        """Merge blocks ``i`` and ``j`` of ``level`` (not necessarily adjacent).

        The later block is moved next to the earlier one; this is only sound
        if every block in between acts on qubits disjoint from the later
        block's, which is checked with ``block_qubits`` (the physical qubits a
        block acts on). Returns the new level index, or ``None`` (and changes
        nothing) if the condition fails. Adjacent blocks always merge.
        """
        blocks = self._level(level)
        if i == j:
            raise ValueError("cannot merge a block with itself")
        lo, hi = sorted((i, j))
        if not 0 <= lo or hi >= len(blocks):
            raise IndexError("block index out of range for level {}".format(level))
        later = set(block_qubits(blocks[hi]))
        for k in range(lo + 1, hi):
            if later & set(block_qubits(blocks[k])):
                return None
        composite = Block(blocks[lo].stages + blocks[hi].stages, (blocks[lo], blocks[hi]), name, role)
        new_blocks = blocks[:lo] + [composite] + blocks[lo + 1:hi] + blocks[hi + 1:]
        return self._push_level(level, new_blocks)

    # -- display ------------------------------------------------------------

    def describe(self, level: int, stage_name: Callable[[int], str] | None = None) -> str:
        """A textual listing of the blocks of ``level``."""
        lines = ["Level {} ({} blocks):".format(level, self.level_size(level))]
        for k, b in enumerate(self._level(level)):
            if stage_name is not None:
                content = ", ".join(stage_name(s) for s in b.stages)
            else:
                content = ", ".join("stage {}".format(s) for s in b.stages)
            label = " '{}'".format(b.name) if b.name else ""
            role = " ({})".format(b.role) if b.role else ""
            lines.append("  Block {}{}{}: [{}]".format(k, label, role, content))
        return "\n".join(lines)
