"""Tests for zxqec.qprogram: stages, master graph, levels, permutations, measurements."""
import pytest
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement, Reset, S, T
from pyzx.utils import VertexType
from pyzx.spacetime import get_delay, get_timestep

from zxqec.qprogram import Block, LevelManager, QProgram
from zxqec.spacetime import Topology, get_logical_qubit, get_stage


def circ(*gates, n=3):
    c = Circuit(n)
    for g in gates:
        c.add_gate(g)
    return c


STAGES = [
    circ(HAD(0), HAD(1), HAD(2), CNOT(0, 1)),
    circ(T(2), CNOT(1, 2)),
    circ(HAD(0), S(1)),
    circ(HAD(1), CNOT(2, 0)),
]


def concat(*cs):
    c = cs[0].copy()
    for d in cs[1:]:
        c.add_circuit(d)
    return c


def program(**kw):
    p = QProgram(3, **kw)
    for i, c in enumerate(STAGES):
        p.add_circuit(c, name="s{}".format(i))
    return p


# -- building ---------------------------------------------------------------

def test_master_graph_is_the_whole_circuit():
    p = program(gate_durations={"CNOT": 2})
    assert zx.compare_tensors(p.graph(), concat(*STAGES).to_graph(), preserve_scalar=False)


def test_stages_are_sequential_and_tagged():
    p = program(gate_durations={"CNOT": 2})
    g = p.graph()
    ticks = [st.ticks for st in p.stages]
    assert ticks == [(0, 2), (3, 5), (6, 6), (7, 8)]           # every stage starts after the previous one
    for v in g.vertices():
        if g.type(v) == VertexType.BOUNDARY:
            continue
        s = get_stage(g, v)
        lo, hi = p.stage(s).ticks
        assert lo <= get_timestep(g, v) <= hi
        assert get_logical_qubit(g, v) in (0, 1, 2)
    assert p.time_extent() == (0, 8)
    assert p.metrics()["time_depth"] == 9


def test_stage_graphs_equal_their_circuits():
    p = program(gate_durations={"CNOT": 2})
    for i, c in enumerate(STAGES):
        assert zx.compare_tensors(p.stage_graph(i), c.to_graph(), preserve_scalar=False)


def test_from_circuit_per_gate_reproduces_thesis_model():
    c = circ(HAD(0), CNOT(0, 1), T(1))
    p = QProgram.from_circuit(c, per_gate=True)
    p.graph()
    assert p.num_stages == 3 and [st.ticks for st in p.stages] == [(0, 0), (1, 1), (2, 2)]
    assert zx.compare_tensors(p.graph(), c.to_graph(), preserve_scalar=False)
    assert QProgram.from_circuit(c).num_stages == 1


def test_validation():
    p = QProgram(3)
    with pytest.raises(ValueError):
        p.add_circuit(Circuit(2))
    with pytest.raises(ValueError):
        QProgram(3, mapping={0: 1, 1: 1})
    with pytest.raises(ValueError):
        QProgram(3, initial_states={0: '1'})
    with pytest.raises(ValueError):
        QProgram(2, topology=Topology.line(2), mapping={0: 5})
    assert QProgram(2).graph().num_vertices() == 4           # empty program = identity wires


# -- levels -------------------------------------------------------------------

def test_level_up_and_block_graphs():
    p = program()
    lvl = p.level_up(0, 1, 2, name="middle")
    assert lvl == 1 and p.levels.num_levels == 2 and p.levels.level_size(1) == 3
    b = p.block(1, 1)
    assert b.stages == (1, 2) and b.name == "middle" and not b.is_leaf and b.contiguous
    assert zx.compare_tensors(p.block_graph(1, 1), p.block_circuit(1, 1).to_graph(), preserve_scalar=False)
    assert zx.compare_tensors(p.block_graph(1, 1), concat(STAGES[1], STAGES[2]).to_graph(), preserve_scalar=False)
    assert "middle" in p.describe_level(1)
    # levelling up again from level 1 replaces anything above
    p.level_up(1, 0, 1)
    assert p.levels.num_levels == 3 and p.block(2, 0).stages == (0, 1, 2)
    p.level_up(0, 0, 3)
    assert p.levels.num_levels == 2 and p.block(1, 0).stages == (0, 1, 2, 3)


def test_merge_blocks_requires_disjoint_intermediates():
    p = program()
    assert p.merge_blocks(0, 0, 3) is None                    # stages 1,2 touch qubits of stage 3
    q = QProgram(3)
    q.add_circuit(circ(HAD(0))); q.add_circuit(circ(T(2))); q.add_circuit(circ(CNOT(0, 1)))
    lvl = q.merge_blocks(0, 0, 2)
    assert lvl == 1 and q.block(1, 0).stages == (0, 2) and not q.block(1, 0).contiguous
    assert zx.compare_tensors(q.block_graph(1, 0), circ(HAD(0), CNOT(0, 1)).to_graph(), preserve_scalar=False)
    assert zx.compare_tensors(q.block_graph(1, 1), circ(T(2)).to_graph(), preserve_scalar=False)


def test_level_manager_standalone():
    m = LevelManager(3)
    assert m.num_levels == 1 and m.num_stages == 3
    m.level_up(0, 0, 1)
    m.append_stage("x")
    assert m.level_size(0) == 4 and m.level_size(1) == 3     # new stage appended to every level
    with pytest.raises(IndexError):
        m.level_up(0, 2, 9)
    with pytest.raises(ValueError):
        Block(())


# -- permutations and topology ----------------------------------------------

def test_permutation_changes_layout_not_semantics():
    p = QProgram(3)
    p.add_circuit(STAGES[0]); p.add_permutation([2, 1, 0]); p.add_circuit(STAGES[1])
    assert p.mapping == (2, 1, 0) and p.stage(1).kind == 'permutation'
    g = p.graph()
    assert p.stage(1).ticks is not None                        # a visible permutation takes a tick
    assert zx.compare_tensors(g, concat(STAGES[0], STAGES[1]).to_graph(), preserve_scalar=False)
    # stage 2's spiders (T on logical 2, CNOT(1,2)) now sit on physical wires 0 and 1
    assert {g.qubit(v) for v in g.vertices() if get_stage(g, v) == 2} == {0, 1}
    assert zx.compare_tensors(p.stage_graph(2), STAGES[1].to_graph(), preserve_scalar=False)
    assert zx.compare_tensors(p.stage_graph(1), Circuit(3).to_graph(), preserve_scalar=False)
    q = QProgram(3)
    q.add_circuit(STAGES[0]); q.add_permutation([[0, 2]], cycles=True, visible=False); q.add_circuit(STAGES[1])
    assert q.mapping == (2, 1, 0)
    assert zx.compare_tensors(q.graph(), concat(STAGES[0], STAGES[1]).to_graph(), preserve_scalar=False)
    assert q.stage(1).ticks is None                            # an invisible one takes no time
    with pytest.raises(ValueError):
        q.add_permutation([0, 0, 1])


def test_permutation_topology_check():
    p = QProgram(3, topology=Topology.line(3))
    p.add_circuit(STAGES[0])
    with pytest.raises(ValueError):
        p.add_permutation([2, 1, 0], check_topology=True)        # 0 and 2 are not adjacent
    with pytest.raises(ValueError):
        p.add_permutation([1, 2, 0], check_topology=True)        # a 3-cycle is not disjoint swaps
    p.add_permutation([1, 0, 2], check_topology=True)
    assert p.mapping == (1, 0, 2)
    # the check uses the *current* mapping: logical 1 now sits on wire 0 and
    # logical 2 on wire 2, which are not adjacent
    with pytest.raises(ValueError):
        p.add_permutation([0, 2, 1], check_topology=True)
    p.add_permutation([2, 1, 0], check_topology=True)            # logical 0 (wire 1) <-> logical 2 (wire 2)
    assert p.mapping == (2, 0, 1)


# -- measurements across stages --------------------------------------------

def test_measured_wire_is_reborn_by_reset():
    rnd = circ(Reset(2), HAD(2), CNOT(2, 0), CNOT(2, 1), HAD(2), Measurement(2))
    p = QProgram(3, initial_states={2: '0'})
    p.add_circuit(rnd, name="round 1"); p.add_circuit(rnd, name="round 2")
    g = p.graph()
    assert [get_logical_qubit(g, v) for v in g.inputs()] == [0, 1]     # ancilla prepared, not open
    assert [get_logical_qubit(g, v) for v in g.outputs()] == [0, 1]    # ancilla measured at the end
    assert sorted(g.var_registry.types) == ['s0_m2', 's1_m2']  # outcomes distinct per stage; no reset randomness
    assert not any(g.vdata(v, 'outcome_type', None) == 'reset_discard' for v in g.vertices())
    sub = p.stage_graph(1)
    assert [get_logical_qubit(sub, v) for v in sub.inputs()] == [0, 1]
    assert [get_logical_qubit(sub, v) for v in sub.outputs()] == [0, 1]


def test_reuse_without_reset_is_rejected():
    p = QProgram(3)
    p.add_circuit(circ(Measurement(2))); p.add_circuit(circ(HAD(2)))
    with pytest.raises(ValueError):
        p.graph()
    q = QProgram(3)
    q.add_circuit(circ(Measurement(2))); q.add_circuit(circ(HAD(0)))   # dead wire untouched: fine
    assert [get_logical_qubit(q.graph(), v) for v in q.graph().outputs()] == [0, 1]


def test_initial_plus_state():
    p = QProgram(1, initial_states={0: '+'})
    p.add_circuit(circ(HAD(0), n=1))
    g = p.graph()
    assert g.inputs() == ()
    zero = Circuit(1); zero.add_gate(HAD(0)); zero.add_gate(HAD(0))
    zero.initialize_qubits([True])
    assert zx.compare_tensors(g, zero.to_graph(), preserve_scalar=False)


def test_physical_frame_circuits_and_quantum_graph():
    # after swapping wires 0 and 1, a circuit written on physical positions acts on the swapped logical qubits
    p = QProgram(3)
    p.add_permutation([1, 0, 2])
    p.add_circuit(circ(T(0)), physical=True)          # position 0 now holds logical qubit 1
    assert zx.compare_tensors(p.graph(), circ(T(1)).to_graph(), preserve_scalar=False)
    q = QProgram(2, topology=Topology.line(3), mapping={0: 2, 1: 1})
    with pytest.raises(ValueError):
        q.add_circuit(Circuit(2), physical=True)      # wires are not 0..n-1
    # classical wires are dropped by quantum_graph
    c = Circuit(2, bit_amount=1); c.add_gate(HAD(0)); c.add_gate(Measurement(0, 0))
    r = QProgram(2, bits=1); r.add_circuit(c)
    full, quantum = r.graph(), r.quantum_graph()
    assert len(full.inputs()) == 3 and len(quantum.inputs()) == 2 and len(quantum.outputs()) == 1
    # ... and from stage / block sub-diagrams on request
    c2 = Circuit(2, bit_amount=1); c2.add_gate(T(1)); r.add_circuit(c2)
    assert len(r.stage_graph(0).inputs()) == 3 and len(r.stage_graph(0, classical=False).inputs()) == 2
    lvl = r.level_up(0, 0, 1)
    assert len(r.block_graph(lvl, 0).inputs()) == 3 and len(r.block_graph(lvl, 0, classical=False).inputs()) == 2
    assert r.block_graph(lvl, 0, classical=False).num_vertices() == r.quantum_graph().num_vertices()
