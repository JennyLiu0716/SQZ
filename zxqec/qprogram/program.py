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

"""``QProgram``: a quantum program as a sequence of stages on a hardware layout.

A program is built stage by stage. A stage is either a PyZX :class:`Circuit`
(a syndrome-extraction round, a logical gate, a single gate, ...) or a
permutation of the qubits over the hardware topology. The stages are
scheduled one after the other: within a stage the gates are placed as soon as
possible (:mod:`pyzx.spacetime`), and every stage starts on the tick after the
previous one finished. :meth:`QProgram.graph` returns the resulting master
space-time ZX-diagram, in which every spider carries ``timestep``, ``delay``,
``stage`` and ``logical_qubit`` vertex data and sits on the *physical* wire of
its qubit.

Stages can be grouped hierarchically into blocks ("level-up", see
:mod:`zxqec.qprogram.levels`), and the sub-diagram of any stage or block can
be extracted as a valid ZX-diagram for further reasoning with PyZX.

This is a rework of the thesis ``QProgram`` on top of unmodified PyZX: no
``GraphST`` subclass, no ``Permutation`` edge type; a permutation is a change
of the logical-to-physical mapping, drawn as identity spiders on the new wires.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pyzx.circuit import Circuit
from pyzx.circuit.gates import HAD, Reset
from pyzx.circuit.graphparser import circuit_to_graph
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph
from pyzx.spacetime import set_delay, set_timestep, spacetime_metrics, time_extent, used_qubits
from pyzx.symbolic import new_var
from pyzx.utils import EdgeType, VertexType

from ..spacetime.subdiagram import (
    get_logical_qubit, get_stage, induced_subdiagram,
    set_logical_qubit, set_stage, shift_timesteps,
)
from ..spacetime.topology import Topology
from .levels import Block, LevelManager

__all__ = ['Stage', 'QProgram']

# Symbolic variables that PyZX generates per circuit_to_graph call (measurement
# outcomes ``m1, m2, ...`` and reset randomness ``_r0, ...``).  They restart at
# every call, so they are renamed per stage to keep them distinct.
_AUTO_VAR = re.compile(r'^(m\d+|_r\d+)$')


@dataclass
class Stage:
    """One stage of a :class:`QProgram`.

    ``mapping[k]`` is the physical wire logical qubit ``k`` occupies while the
    stage runs; for a permutation stage that is the mapping *after* it. ``ticks``
    is the ``(first, last)`` tick the stage occupies, set when the master graph
    is built (``None`` for a stage that takes no time).
    """
    index: int
    kind: str                       # 'circuit' or 'permutation'
    circuit: Circuit                # empty for a permutation stage
    mapping: tuple[int, ...]
    perm: tuple[int, ...] | None = None
    name: str | None = None
    visible: bool = True
    ticks: tuple[int, int] | None = None

    def __str__(self) -> str:
        if self.name:
            return self.name
        if self.kind == 'permutation':
            return "perm{}".format(list(self.perm or ()))
        return "circuit({} gates)".format(len(self.circuit.gates))


class QProgram:
    """A quantum program as a sequence of stages on a hardware layout.

    Args:
        qubits: number of (logical, i.e. circuit) qubits.
        bits: number of classical bits every stage circuit uses.
        topology: optional hardware layout; when given, the mapping must send
            qubits to its nodes and permutations can be checked against it.
        mapping: initial logical-to-physical map ``{k: node}``; identity by default.
        gate_durations: per gate class/name durations forwarded to
            :func:`pyzx.circuit.graphparser.circuit_to_graph`; ``{}`` (default)
            gives a pure dependency layering.
        initial_states: ``{k: '0' | '+'}`` states to prepare on inputs before the
            first stage; other inputs stay open.
    """

    def __init__(self, qubits: int, *, bits: int = 0,
                 topology: Topology | None = None,
                 mapping: Mapping[int, int] | None = None,
                 gate_durations: Mapping[Any, int] | None = None,
                 initial_states: Mapping[int, str] | None = None) -> None:
        if qubits < 1:
            raise ValueError("a program needs at least one qubit")
        self.qubits = qubits
        self.bits = bits
        self.topology = topology
        self.gate_durations: dict[Any, int] = dict(gate_durations) if gate_durations else {}
        self._mapping: list[int] = self._check_mapping(mapping)
        self._initial_states: dict[int, str] = {}
        for k, s in (initial_states or {}).items():
            if s not in ('0', '+'):
                raise ValueError("initial state of qubit {} must be '0' or '+', got {!r}".format(k, s))
            if not 0 <= k < qubits:
                raise ValueError("qubit {} out of range".format(k))
            self._initial_states[k] = s
        self._stages: list[Stage] = []
        self.levels = LevelManager()
        self._graph: BaseGraph | None = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_circuit(cls, circ: Circuit, per_gate: bool = False, **kwargs: Any) -> 'QProgram':
        """A program with ``circ`` as its only stage, or one stage per gate.

        ``per_gate=True`` reproduces the thesis model in which every gate is
        its own timestep; keyword arguments go to the constructor.
        """
        p = cls(circ.qubits, bits=circ.bits, **kwargs)
        if per_gate:
            for gate in circ.gates:
                c = Circuit(circ.qubits, bit_amount=circ.bits)
                c.add_gate(gate.copy())
                p.add_circuit(c, name=str(gate))
        else:
            p.add_circuit(circ)
        return p

    def add_circuit(self, circ: Circuit, name: str | None = None) -> int:
        """Append a circuit stage; returns its stage index."""
        if circ.qubits != self.qubits:
            raise ValueError("stage has {} qubits, program has {}".format(circ.qubits, self.qubits))
        if circ.bits != self.bits:
            raise ValueError("stage has {} bits, program has {}".format(circ.bits, self.bits))
        st = Stage(len(self._stages), 'circuit', circ.copy(), tuple(self._mapping), name=name)
        return self._append(st)

    def add_permutation(self, perm: Sequence[int] | Sequence[Sequence[int]], *,
                        cycles: bool = False, index_base: int = 0,
                        check_topology: bool = False, visible: bool = True,
                        name: str | None = None) -> int:
        """Append a permutation stage; returns its stage index.

        ``perm`` is either one-line notation (``perm[k]`` is the logical qubit
        whose wire logical qubit ``k`` moves to) or, with ``cycles=True``, a
        list of cycles. ``index_base=1`` accepts 1-based indices. With
        ``check_topology=True`` the permutation must consist of disjoint swaps
        along edges of the topology (under the current mapping), i.e. be
        realisable without routing. ``visible=False`` makes the stage take no
        time and draw as plain crossing wires instead of identity spiders.
        """
        one_line = self._to_one_line(perm, cycles, index_base)
        old = list(self._mapping)
        if check_topology:
            self._check_permutation_topology(one_line, old)
        new = [old[one_line[k]] for k in range(self.qubits)]
        st = Stage(len(self._stages), 'permutation', Circuit(self.qubits, bit_amount=self.bits),
                   tuple(new), perm=tuple(one_line), name=name, visible=visible)
        self._mapping = new
        return self._append(st)

    def _append(self, st: Stage) -> int:
        self._stages.append(st)
        self.levels.append_stage(st.name)
        self._graph = None
        return st.index

    # -- mapping / topology -------------------------------------------------

    @property
    def mapping(self) -> tuple[int, ...]:
        """Current physical wire of every logical qubit (after the last stage)."""
        return tuple(self._mapping)

    def mapping_at(self, stage: int) -> tuple[int, ...]:
        return self._stages[stage].mapping

    def _check_mapping(self, mapping: Mapping[int, int] | None) -> list[int]:
        m = [mapping.get(k, k) for k in range(self.qubits)] if mapping else list(range(self.qubits))
        if len(set(m)) != self.qubits:
            raise ValueError("mapping must send distinct qubits to distinct wires")
        if self.topology is not None:
            missing = [p for p in m if p not in self.topology]
            if missing:
                raise ValueError("wires {} are not nodes of the topology".format(missing))
        return m

    def _to_one_line(self, perm: Sequence[Any], cycles: bool, index_base: int) -> list[int]:
        n = self.qubits
        if cycles:
            one_line = list(range(n))
            for cyc in perm:
                cyc = [int(c) - index_base for c in cyc]
                for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                    one_line[a] = b
        else:
            one_line = [int(c) - index_base for c in perm]
        if sorted(one_line) != list(range(n)):
            raise ValueError("{} is not a permutation of 0..{}".format(one_line, n - 1))
        return one_line

    def _check_permutation_topology(self, one_line: list[int], old: list[int]) -> None:
        if self.topology is None:
            raise ValueError("check_topology requires a topology")
        for k, j in enumerate(one_line):
            if one_line[j] != k:
                raise ValueError("permutation {} is not a product of disjoint swaps".format(one_line))
            if j != k and not self.topology.are_adjacent(old[k], old[j]):
                raise ValueError("swap ({}, {}) is not along an edge of the topology "
                                 "(physical wires {}, {})".format(k, j, old[k], old[j]))

    # -- stages -------------------------------------------------------------

    @property
    def stages(self) -> tuple[Stage, ...]:
        return tuple(self._stages)

    @property
    def num_stages(self) -> int:
        return len(self._stages)

    def stage(self, index: int) -> Stage:
        return self._stages[index]

    def physical_qubits(self, stage: int) -> set[int]:
        """Physical wires the stage acts on."""
        st = self._stages[stage]
        if st.kind == 'permutation':
            before = self._stages[stage - 1].mapping if stage > 0 else st.mapping
            assert st.perm is not None
            return {before[k] for k in range(self.qubits) if st.perm[k] != k} | \
                   {st.mapping[k] for k in range(self.qubits) if st.perm[k] != k}
        return {st.mapping[k] for g in st.circuit.gates for k in used_qubits(g)}

    def stage_circuit(self, index: int) -> Circuit:
        """The circuit of a stage on the logical qubits (identity for a permutation)."""
        return self._stages[index].circuit.copy()

    # -- levels -------------------------------------------------------------

    def level_up(self, level: int, start: int, end: int, name: str | None = None) -> int:
        """Group blocks ``start..end`` of ``level`` into one; returns the new level."""
        return self.levels.level_up(level, start, end, name)

    def merge_blocks(self, level: int, i: int, j: int, name: str | None = None) -> int | None:
        """Merge two blocks of ``level`` if everything between them acts on other qubits."""
        return self.levels.merge_blocks(level, i, j, self._block_qubits, name)

    def _block_qubits(self, block: Block) -> set[int]:
        out: set[int] = set()
        for s in block.stages:
            out |= self.physical_qubits(s)
        return out

    def block(self, level: int, index: int) -> Block:
        return self.levels.block(level, index)

    def describe_level(self, level: int) -> str:
        return self.levels.describe(level, lambda s: str(self._stages[s]))

    def block_circuit(self, level: int, index: int) -> Circuit:
        """The circuit of a block: its stage circuits concatenated (on logical qubits)."""
        c = Circuit(self.qubits, bit_amount=self.bits)
        for s in self.levels.block(level, index).stages:
            c.add_circuit(self._stages[s].circuit)
        return c

    # -- graphs -------------------------------------------------------------

    def graph(self) -> BaseGraph:
        """The master space-time ZX-diagram of the whole program (cached)."""
        if self._graph is None:
            self._graph = self._build()
        return self._graph

    def stage_graph(self, index: int) -> BaseGraph:
        """The sub-diagram of a stage, cut out of the master graph."""
        return self._subdiagram({index})

    def block_graph(self, level: int, index: int) -> BaseGraph:
        """The sub-diagram of a block of a level, cut out of the master graph."""
        return self._subdiagram(set(self.levels.block(level, index).stages))

    def time_extent(self) -> tuple[int, int] | None:
        return time_extent(self.graph())

    def metrics(self) -> dict[str, int]:
        """Space-time cost metrics of the master graph (see :func:`pyzx.spacetime.spacetime_metrics`)."""
        return spacetime_metrics(self.graph())

    def draw(self, **kwargs: Any) -> Any:
        """Draw the master graph with :func:`pyzx.draw` (``show_time=True`` shows the schedule)."""
        import pyzx
        return pyzx.draw(self.graph(), **kwargs)

    def _subdiagram(self, stages: set[int]) -> BaseGraph:
        g = self.graph()
        ins = set(g.inputs())
        n = self.num_stages
        lo, hi = min(stages), max(stages)

        def key(v: Any) -> float:
            # Stages strictly between the block's first and last that are not
            # part of it (a merged block) act on other qubits and are ordered
            # after the block, as if slid out of the way.
            s = get_stage(g, v)
            if s is None:
                return -1 if v in ins else n
            if lo < s < hi and s not in stages:
                return hi + 0.5
            return s

        def keep(v: Any) -> bool:
            return g.type(v) != VertexType.BOUNDARY and get_stage(g, v) in stages

        return induced_subdiagram(g, keep, key)

    # -- building the master graph ------------------------------------------

    def _build(self) -> BaseGraph:
        master: BaseGraph | None = None
        live = set(range(self.qubits))       # logical wires that currently have an output
        next_tick = 0
        for st in self._stages:
            sg = self._stage_graph(st, live, next_tick)
            ext = time_extent(sg)
            if ext is None:
                st.ticks = None
            else:
                st.ticks = ext
                next_tick = ext[1] + 1
            if master is None:
                master = sg
            else:
                self._glue(master, sg)
            live = set()
            for v in master.outputs():
                k = get_logical_qubit(master, v)
                if k is not None and k < self.qubits:
                    live.add(k)
        if master is None:
            master = self._identity_graph()
        return master

    def _identity_graph(self) -> BaseGraph:
        g = Graph()
        ins, outs = [], []
        for k in range(self.qubits):
            p = self._mapping[k]
            i = g.add_vertex(VertexType.BOUNDARY, p, 0)
            o = g.add_vertex(VertexType.BOUNDARY, p, 1)
            set_logical_qubit(g, i, k)
            set_logical_qubit(g, o, k)
            g.add_edge((i, o))
            ins.append(i)
            outs.append(o)
        g.set_inputs(tuple(ins))
        g.set_outputs(tuple(outs))
        return g

    def _classical_wire(self, bit: int) -> int:
        """Physical coordinate of classical bit wire ``bit`` (drawn below the qubits)."""
        top = max(self._mapping) if self.topology is None else max(self.topology.nodes)
        return top + 1 + bit

    def _stage_graph(self, st: Stage, live: set[int], start_tick: int) -> BaseGraph:
        """The graph of one stage: physical coordinates, tagged and shifted in time."""
        if st.kind == 'permutation':
            g = self._permutation_graph(st, live)
        else:
            g = self._circuit_stage_graph(st, live)
            self._to_physical(g, st)
        for v in g.vertices():
            if g.type(v) != VertexType.BOUNDARY:
                set_stage(g, v, st.index)
        shift_timesteps(g, start_tick)
        self._rename_auto_vars(g, st.index)
        return g

    def _to_physical(self, g: BaseGraph, st: Stage) -> None:
        """Record the logical qubit of every vertex and move it to its physical wire.

        ``circuit_to_graph`` puts vertices at qubit coordinate ``k`` (logical
        qubit ``k``; multi-qubit gates may use ``k + 0.5``) and classical bit
        ``b`` at ``qubits + b``.
        """
        n = self.qubits
        for v in g.vertices():
            q = g.qubit(v)
            k = int(q // 1)
            frac = q - k
            set_logical_qubit(g, v, k)
            if k < n:
                g.set_qubit(v, st.mapping[k] + frac)
            else:
                g.set_qubit(v, self._classical_wire(k - n) + frac)

    def _circuit_stage_graph(self, st: Stage, live: set[int]) -> BaseGraph:
        n = self.qubits
        c = st.circuit.copy()
        init = [False] * n
        strip: list[int] = []
        first_gate: dict[int, Any] = {}
        for gate in c.gates:
            for k in used_qubits(gate):
                first_gate.setdefault(k, gate)
        if st.index == 0:
            for k, state in self._initial_states.items():
                init[k] = True
                if state == '+':
                    c.gates.insert(0, HAD(k))
                elif isinstance(first_gate.get(k), Reset):
                    c.gates.remove(first_gate[k])   # resetting a fresh |0> is a no-op
        for k in range(n):
            if k in live or st.index == 0 and k in self._initial_states:
                continue
            # the wire was measured in an earlier stage
            if k not in first_gate:
                strip.append(k)
            elif isinstance(first_gate[k], Reset):
                init[k] = True
                c.gates.remove(first_gate[k])       # the |0> preparation replaces the reset
            else:
                raise ValueError("qubit {} was measured before stage {} and is reused by {} "
                                 "without a Reset".format(k, st.index, first_gate[k]))
        g = circuit_to_graph(c, initialize_qubits=init if any(init) else None,
                             gate_durations=self.gate_durations)
        # drop the wires of dead qubits the stage does not touch
        if strip:
            ins = list(g.inputs())
            outs = list(g.outputs())
            for k in strip:
                i = next(v for v in ins if g.qubit(v) == k)
                o = next(iter(g.neighbors(i)))
                assert g.type(o) == VertexType.BOUNDARY and o in outs
                ins.remove(i)
                outs.remove(o)
                g.remove_vertices([i, o])
            g.set_inputs(tuple(ins))
            g.set_outputs(tuple(outs))
        return g

    def _permutation_graph(self, st: Stage, live: set[int]) -> BaseGraph:
        """Wires from the old physical positions of the live qubits to the new ones.

        With ``visible=True`` each wire carries a phase-free Z spider on its new
        position (an identity that occupies one tick); otherwise the boundaries
        are wired directly. Classical bit wires pass straight through.
        """
        assert st.perm is not None
        old = self._stages[st.index - 1].mapping if st.index > 0 else st.mapping
        g = Graph()
        ins, outs = [], []
        wires = [(k, old[k], st.mapping[k]) for k in range(self.qubits) if k in live]
        wires += [(self.qubits + b, self._classical_wire(b), self._classical_wire(b))
                  for b in range(self.bits)]
        for k, p_old, p_new in wires:
            i = g.add_vertex(VertexType.BOUNDARY, p_old, 0)
            if st.visible and k < self.qubits:
                m = g.add_vertex(VertexType.Z, p_new, 1)
                set_timestep(g, m, 0)
                set_delay(g, m, 0)
                set_logical_qubit(g, m, k)
                g.add_edge((i, m))
                o = g.add_vertex(VertexType.BOUNDARY, p_new, 2)
                g.add_edge((m, o))
            else:
                o = g.add_vertex(VertexType.BOUNDARY, p_new, 1)
                g.add_edge((i, o))
            set_logical_qubit(g, i, k)
            set_logical_qubit(g, o, k)
            ins.append(i)
            outs.append(o)
        g.set_inputs(tuple(ins))
        g.set_outputs(tuple(outs))
        return g

    @staticmethod
    def _rename_auto_vars(g: BaseGraph, stage: int) -> None:
        names = set()
        for v in g.vertices():
            ph = g.phase(v)
            if hasattr(ph, 'free_vars'):
                names |= {var.name for var in ph.free_vars()}
        fresh = {nm: "s{}_{}".format(stage, nm.lstrip('_')) for nm in names if _AUTO_VAR.match(nm)}
        if fresh:
            g.substitute_variables({nm: new_var(new, True, g.var_registry) for nm, new in fresh.items()},
                                   in_place=True)
            for nm in fresh:
                g.var_registry.types.pop(nm, None)
            for v in g.vertices():
                sym = g.vdata(v, 'result_symbol', None)
                if sym in fresh:
                    g.set_vdata(v, 'result_symbol', fresh[sym])

    def _glue(self, master: BaseGraph, sg: BaseGraph) -> None:
        """Append stage graph ``sg`` to ``master``, plugging wires by logical qubit."""
        m_out = {get_logical_qubit(master, v): v for v in master.outputs()}
        s_in = {get_logical_qubit(sg, v): v for v in sg.inputs()}
        plugged = {v for k, v in s_in.items() if k in m_out}

        maxr = max((master.row(v) for v in master.vertices()
                    if master.type(v) != VertexType.BOUNDARY), default=0)
        minr = min((sg.row(v) for v in sg.vertices()
                    if sg.type(v) != VertexType.BOUNDARY), default=0)
        offset = maxr - minr + 1

        master.scalar.mult_with_scalar(sg.scalar)
        for name in sg.var_registry.vars():
            master.var_registry.set_type(name, sg.var_registry.get_type(name))

        vtab: dict[Any, Any] = {}
        for v in sg.vertices():
            if v in plugged:
                continue
            w = master.add_vertex(sg.type(v), phase=sg.phase(v), qubit=sg.qubit(v),
                                  row=offset + sg.row(v), ground=sg.is_ground(v))
            master.set_vdata_dict(w, sg.vdata_dict(v))
            vtab[v] = w
        for e in sg.edges():
            s, t = sg.edge_st(e)
            if s in plugged or t in plugged:
                continue
            master.add_edge((vtab[s], vtab[t]), sg.edge_type(e))

        for k, i in s_in.items():
            if k not in m_out:
                continue
            o = m_out[k]
            no = next(iter(master.neighbors(o)))
            ni = next(iter(sg.neighbors(i)))
            et = (EdgeType.HADAMARD
                  if master.edge_type(master.edge(no, o)) != sg.edge_type(sg.edge(i, ni))
                  else EdgeType.SIMPLE)
            master.remove_vertex(o)
            master.add_edge((no, vtab[ni]), et)

        inputs = list(master.inputs()) + [vtab[v] for k, v in s_in.items() if k not in m_out]
        outputs = [v for v in master.outputs() if get_logical_qubit(master, v) not in s_in]
        outputs += [vtab[v] for v in sg.outputs()]
        master.set_inputs(tuple(inputs))
        master.set_outputs(tuple(outputs))
        master.rebind_variables_to_registry()
