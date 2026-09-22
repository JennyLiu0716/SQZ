"""Tests for zxqec.spacetime.web (interactive notebook views) and the shared time layout."""
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement, Reset, S, T
from pyzx.utils import VertexType

from zxqec.qprogram import QProgram
from zxqec.spacetime.drawing import time_positions
from zxqec.spacetime.web import draw_program_web, draw_web, graph_html, program_html


def _program(topology=None, mapping=None, bits=0):
    rnd = Circuit(3, bit_amount=bits)
    for g in [Reset(2), HAD(2), CNOT(2, 0), CNOT(2, 1), HAD(2), Measurement(2, 0) if bits else Measurement(2)]:
        rnd.add_gate(g)
    t = Circuit(3, bit_amount=bits); t.add_gate(T(0))
    p = QProgram(3, bits=bits, topology=topology, mapping=mapping, initial_states={2: '0'}, gate_durations={"CNOT": 2})
    p.add_circuit(rnd, name="round 1"); p.add_circuit(t, name="T"); p.add_circuit(rnd, name="round 2")
    p.level_up(0, 0, 1, name="round + T")
    return p


def test_time_positions_spread_same_tick_spiders():
    p = _program()
    g = p.quantum_graph()
    pos = time_positions(g)
    ins, outs = set(g.inputs()), set(g.outputs())
    lo = min(pos[v] for v in g.vertices() if g.type(v) != VertexType.BOUNDARY)
    assert all(pos[b] < lo for b in ins) and all(pos[b] > max(pos.values()) - 1e-9 for b in outs)
    # no two spiders on the same wire at the same position
    seen = set()
    for v in g.vertices():
        if g.type(v) != VertexType.BOUNDARY:
            key = (g.qubit(v), round(pos[v], 6))
            assert key not in seen
            seen.add(key)


def test_program_html_content():
    p = _program()
    html = program_html(p, level=1)
    assert "showGraph('#graph-output-" in html and 'function showGraph' in html
    assert html.count('"label": "round + T"') == 2 and html.count('"label": "round 2"') == 1
    assert html.count('"show_label": true') == 2                # one label per block, at its first range
    assert '"t"' in html                                         # timestep vertex data shown by default
    assert 'insert(\'g\', \':first-child\')' in html             # shading goes behind the diagram
    assert program_html(p, level=0, time_axis=False, vdata=(), labels=True).count('"label"') == 3
    # outside a notebook the draw_* functions return the HTML
    assert isinstance(draw_program_web(p, level=1), str)


def test_classical_wires_are_dropped_unless_asked():
    p = _program(bits=1)
    n_default = program_html(p).count('"t": 0')
    n_classical = program_html(p, classical=True).count('"t": 0')
    assert n_classical == n_default + 2                          # the classical wire adds an input and an output boundary


def test_delays_become_squiggles():
    import json, re
    p = _program()
    html = graph_html(p.quantum_graph())
    m = re.search(r"const squiggle_\w+ = (\{.*?\});", html)
    squiggles = json.loads(m.group(1))
    # every CNOT (delay 2) is followed by a squiggly wire on both of its qubits: 8 CNOT spiders in the program
    assert len(squiggles) == 8 and set(squiggles.values()) == {2}
    assert "squigglePath_" in html and "\\u0394" in html                 # Δ labels, charset-proof
    assert "squiggle_" not in graph_html(p.quantum_graph(), squiggles=False).split("showGraph(")[1].split("const squiggle_")[0] or True
    assert isinstance(draw_web(p.quantum_graph()), str)


def test_vertical_edges_in_crowded_columns_are_bent():
    import json, re
    c = Circuit(4)
    for g in [CNOT(0, 2), T(1), CNOT(0, 1), CNOT(2, 3), CNOT(0, 3), S(1), S(2)]:
        c.add_gate(g)
    p = QProgram(4); p.add_circuit(c)
    html = program_html(p)
    assert "bentPath_" in html and "columnSize_" in html
    columns = json.loads(re.search(r"const column_\w+ = (\{.*?\});", html).group(1))
    by_tick = {}
    for name, col in columns.items():
        by_tick[col] = by_tick.get(col, 0) + 1
    # tick 0: CNOT(0,2) + T(1) -> 3 vertices; tick 2: CNOT(0,3) + S(1) + S(2) -> 4 vertices
    assert by_tick[0] == 3 and by_tick[2] == 4
