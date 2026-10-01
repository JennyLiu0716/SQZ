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
        parse_stim("SQRT_X 0")
    with pytest.raises(ValueError):
        parse_stim("MPP X0*X0")                         # a qubit twice in one product


@pytest.mark.parametrize('text, value', [
    ("RX 0 1\nMPP X0*X1\n", 0), ("RX 0\nS 0\nMPP Y0\n", 0), ("RX 0\nS_DAG 0\nMPP Y0\n", 1),
    ("RX 0\nS 0\nMPP !Y0\n", 1), ("R 0\nX 0\nMPP Z0\n", 1)])
def test_a_product_of_paulis_on_its_eigenstate_has_the_eigenvalue_as_result(text, value):
    from zxqec.qprogram import check_detectors
    p, circ = program_from_stim(text)
    assert circ.measurements == 1
    assert [(d.status, d.value) for d in check_detectors(p, [['c[0]']]).declared] == [('deterministic', value)]


def test_products_of_paulis_on_helper_qubits():
    # on |00>, YY and XX are random but opposite (XX YY = -ZZ), and ZZ stays +1; each product has its own helper
    from zxqec.qprogram import check_detectors
    p, circ = program_from_stim("R 0 1\nMPP Y0*Y1 X0*X1 Z0*Z1\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-1]\n")
    assert circ.qubits == 2 and circ.measurements == 3 and p.num_stages == 1
    assert p.stage_circuit(0).qubits == 2 + 3                 # the circuit's two qubits and three helpers
    assert len(p.quantum_graph().outputs()) == 2                # the helpers are measured: not outputs
    chk = check_detectors(p, circ.detectors)
    assert [(d.status, d.value) for d in chk.declared] == [('deterministic', 1), ('deterministic', 0)]
    assert [(d.status, d.value) for d in check_detectors(p, [['c[0]']]).declared][0][0] == 'not deterministic'


def test_noise_is_skipped():
    noisy = "RX 0\nDEPOLARIZE1(0.001) 0\nZ_ERROR(0.002) 0\nCX 0 1\nDEPOLARIZE2(0.001) 0 1\nMX(0.001) 0\nM(0.001) 1\n"
    clean = "RX 0\nCX 0 1\nMX 0\nM 1\n"
    a, _ = program_from_stim(noisy)
    b, _ = program_from_stim(clean)
    assert [str(g) for g in a.stage_circuit(0).gates] == [str(g) for g in b.stage_circuit(0).gates]


def test_paulis_fed_forward_from_a_measurement():
    # an X on qubit 1 when c[0] = 1 makes c[1] equal c[0]; so does a Z before an X-basis readout
    from zxqec.gpf import reduce_gpf
    for text in ("H 0\nM 0\nCX rec[-1] 1\nM 1\n", "RX 1\nH 0\nM 0\nCZ rec[-1] 1\nMX 1\n",
                 "RX 1\nH 0\nM 0\nCZ 1 rec[-1]\nMX 1\n"):
        p, circ = program_from_stim(text)
        assert circ.measurements == 2
        assert any(type(g).__name__ == 'ParityControlled' for s in range(p.num_stages) for g in p.stage_circuit(s).gates)
        detectors = [sorted(v.name for v in d.free_vars()) for d in reduce_gpf(p.quantum_graph()).detectors]
        assert ['c[0]', 'c[1]'] in detectors


def test_measurement_records_only_as_controls():
    with pytest.raises(ValueError):
        parse_stim("M 0\nCX 1 rec[-1]\n")
    with pytest.raises(ValueError):
        parse_stim("M 0\nH rec[-1]\n")
