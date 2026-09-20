"""Tests for zxqec.spacetime.Topology."""
import math

import pytest

from zxqec.spacetime import Topology


def test_grid_structure():
    t = Topology.grid(2, 3)
    assert len(t) == 6 and t.dimension == 2
    assert t.coordinates(4) == (1.0, 1.0)          # node r*cols + c at (c, r)
    assert t.neighbors(0) == [1, 3]
    assert t.neighbors(4) == [1, 3, 5]
    assert t.are_adjacent(1, 4) and not t.are_adjacent(0, 4)
    assert t.degree(4) == 3
    assert t.is_connected()
    assert len(t.edge_list()) == 7


def test_line_and_distance():
    t = Topology.line(4)
    assert t.dimension == 1 and t.edge_list() == [(0, 1), (1, 2), (2, 3)]
    assert t.distance(0, 3) == 3.0
    t3 = Topology(3)
    t3.add_node(0, (0, 0, 0)); t3.add_node(1, (1, 2, 2))
    assert math.isclose(t3.distance(0, 1), 3.0)


def test_validation():
    t = Topology(2)
    t.add_node(0, (0, 0))
    with pytest.raises(ValueError):
        t.add_node(0, (1, 1))                       # duplicate
    with pytest.raises(ValueError):
        t.add_node(1, (1, 1, 1))                    # wrong dimension
    with pytest.raises(ValueError):
        t.add_edge(0, 5)                            # unknown node
    with pytest.raises(ValueError):
        t.add_edge(0, 0)                            # self-loop
    with pytest.raises(ValueError):
        Topology(4)


def test_remove_and_connectivity():
    t = Topology.line(3)
    t.remove_edge(0, 1)
    assert not t.is_connected()
    t.remove_node(1)
    assert 1 not in t and t.edge_list() == []
    assert Topology(1).is_connected()               # empty


def test_roundtrip_dict():
    t = Topology.grid(2, 2)
    u = Topology.from_dict(t.to_dict())
    assert u.dimension == 2 and u.nodes == t.nodes and u.edges == t.edges


def test_draw_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    assert Topology.grid(2, 2).draw() is not None
    assert Topology.line(3).draw() is not None
    t3 = Topology(3); t3.add_node(0, (0, 0, 0)); t3.add_node(1, (1, 1, 1)); t3.add_edge(0, 1)
    assert t3.draw() is not None
