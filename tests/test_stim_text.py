"""Reading circuits in Stim's text format."""
import pytest

from zxqec.qprogram import parse_stim, program_from_stim

SMALL = """
QUBIT_COORDS(0, 0) 0
R 0 1
RX 2
TICK
CX 2 0 2 1
T_DAG 0
M 0 1
MX 2
DETECTOR(1, 0) rec[-3] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
R 0
H 0
M 0
"""


def test_parse_counts_measurements_and_translates_records():
    c = parse_stim(SMALL)
    assert c.qubits == 3 and c.measurements == 4
    assert c.detectors == [['c[0]', 'c[1]']]           # rec[-3], rec[-2] after three measurements
    assert c.observables == {0: ['c[2]']}
    assert c.coords[0] == (0.0, 0.0)


def test_program_is_cut_where_a_measured_qubit_is_reset():
    p, c = program_from_stim(SMALL)
    assert p.num_stages == 2                             # the R 0 after M 0 starts a new stage
    assert p.quantum_graph() is not None


def test_unsupported_instruction_is_rejected():
    with pytest.raises(ValueError):
        parse_stim("MPP X0*X1")
