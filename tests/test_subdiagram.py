"""Tests for zxqec.spacetime.induced_subdiagram and the vertex-data helpers."""
from fractions import Fraction

import pytest
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import HAD, T
from pyzx.utils import EdgeType, VertexType
from pyzx.spacetime import get_timestep

from zxqec.spacetime import (
    get_logical_qubit, get_stage, induced_subdiagram, set_logical_qubit, set_stage,
    shift_timesteps,
)


def _graph():
    # single qubit: T at tick 0, H at tick 1; tag ticks as "stage" for the test
    c = Circuit(1); c.add_gate(T(0)); c.add_gate(HAD(0))
    g = c.to_graph(gate_durations={})
    for v in g.vertices():
        set_logical_qubit(g, v, 0)
        if g.type(v) != VertexType.BOUNDARY:
            set_stage(g, v, get_timestep(g, v))
    ins = set(g.inputs())
    key = lambda v: get_stage(g, v) if get_stage(g, v) is not None else (-1 if v in ins else 99)
    return g, key


def test_vdata_helpers():
    g, _ = _graph()
    v = next(v for v in g.vertices() if g.type(v) != VertexType.BOUNDARY)
    assert get_logical_qubit(g, v) == 0 and get_stage(g, v) in (0, 1)
    before = {v: get_timestep(g, v) for v in g.vertices()}
    shift_timesteps(g, 5)
    assert all(get_timestep(g, v) == before[v] + 5 for v in g.vertices())


def test_following_hadamard_stays_outside():
    # The edge between the T spider and the H spider was created by the H gate,
    # so the sub-diagram of the T spider alone must not carry a Hadamard.
    g, key = _graph()
    sub = induced_subdiagram(g, lambda v: get_stage(g, v) == 0, key)
    t_only = Circuit(1); t_only.add_gate(T(0))
    assert zx.compare_tensors(sub, t_only.to_graph(), preserve_scalar=False)
    assert all(sub.edge_type(e) == EdgeType.SIMPLE for e in sub.edges())
    # ... and the H spider's sub-diagram does keep its Hadamard.
    sub_h = induced_subdiagram(g, lambda v: get_stage(g, v) == 1, key)
    h_only = Circuit(1); h_only.add_gate(HAD(0))
    assert zx.compare_tensors(sub_h, h_only.to_graph(), preserve_scalar=False)


def test_boundaries_and_vdata():
    g, key = _graph()
    sub = induced_subdiagram(g, lambda v: get_stage(g, v) == 0, key)
    assert len(sub.inputs()) == 1 and len(sub.outputs()) == 1
    assert get_logical_qubit(sub, sub.inputs()[0]) == 0
    v = next(v for v in sub.vertices() if sub.type(v) != VertexType.BOUNDARY)
    assert sub.phase(v) == Fraction(1, 4) and get_stage(sub, v) == 0


def test_idle_wire_becomes_identity():
    # two qubits, only qubit 0 has a gate; selecting it must keep qubit 1 as a wire
    c = Circuit(2); c.add_gate(T(0))
    g = c.to_graph(gate_durations={})
    for v in g.vertices():
        set_logical_qubit(g, v, int(g.qubit(v)))
        if g.type(v) != VertexType.BOUNDARY:
            set_stage(g, v, 0)
    ins = set(g.inputs())
    key = lambda v: 0 if get_stage(g, v) is not None else (-1 if v in ins else 1)
    sub = induced_subdiagram(g, lambda v: get_stage(g, v) == 0, key)
    assert len(sub.inputs()) == 2 and len(sub.outputs()) == 2
    assert zx.compare_tensors(sub, c.to_graph(), preserve_scalar=False)


def test_empty_selection_rejected():
    g, key = _graph()
    with pytest.raises(ValueError):
        induced_subdiagram(g, lambda v: False, key)
