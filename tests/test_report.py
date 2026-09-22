"""The measurement report: classification, detectors and checking a run."""
import pytest
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement, NOT, Reset, T

from zxqec.gpf import Measurement as M, measurement_report
from zxqec.qprogram import QProgram


def parity_round(n=3, ancilla=2, bits=(0,)):
    """One Z_0 Z_1 parity check with an ancilla, writing its outcome to bit ``bits[0]``."""
    c = Circuit(n, bit_amount=4)
    for g in [Reset(ancilla), HAD(ancilla), CNOT(ancilla, 0), CNOT(ancilla, 1), HAD(ancilla),
              Measurement(ancilla, bits[0])]:
        c.add_gate(g)
    return c


def test_single_measurement_fixed_and_random():
    # |0> measured in the Z basis: deterministic; |+> measured: random and gauge
    for prep, kind, value in [([], 'fixed', 0), ([NOT(0)], 'fixed', 1), ([HAD(0)], 'gauge', None)]:
        c = Circuit(1, bit_amount=1)
        for g in prep:
            c.add_gate(g)
        c.add_gate(Measurement(0, 0))
        c.initialize_qubits([True])
        rep = measurement_report(c.to_graph())
        (m,) = rep.measurements.values()
        assert isinstance(m, M) and m.kind == kind and m.value == value
        assert len(rep.detectors) == (0 if kind == 'gauge' else 1)


def test_repeated_check_is_tied_and_detects():
    # two rounds of the same parity check on |+ + 0>: first random, second must agree
    p = QProgram(3, bits=4, initial_states={2: '0'})
    p.add_circuit(parity_round(bits=(0,)), name="round 1")
    p.add_circuit(parity_round(bits=(1,)), name="round 2")
    rep = p.measurement_report()
    kinds = {n: m.kind for n, m in rep.measurements.items()}
    # the data wires are open, so the first outcome steers the map (it says which eigenspace the
    # input was projected into); the second one must repeat it
    assert kinds == {'c[0]': 'frame', 'c[1]': 'tied'}
    assert len(rep.detectors) == 1
    assert {str(v) for v in rep.detectors[0].free_vars()} == {'c[0]', 'c[1]'}
    # checking a run
    assert rep.is_possible({'c[0]': 1, 'c[1]': 1}) and rep.is_possible({'c[0]': 0, 'c[1]': 0})
    assert not rep.is_possible({'c[0]': 1, 'c[1]': 0})
    assert len(rep.violated({'c[0]': 1, 'c[1]': 0})) == 1
    assert rep.dimensions == {'outcomes': 2, 'forced': 1, 'steering': 1, 'gauge': 0}


def test_block_report_in_context_and_alone():
    p = QProgram(3, bits=4, initial_states={2: '0'})
    p.add_circuit(parity_round(bits=(0,)), name="round 1")
    p.add_circuit(parity_round(bits=(1,)), name="round 2")
    lvl = p.level_up(0, 0, 1, name="two rounds")
    whole = p.measurement_report(lvl, 0)                        # the block is the whole program here
    assert sorted(whole.measurements) == ['c[0]', 'c[1]'] and len(whole.internal_detectors()) == 1
    second = p.measurement_report(0, 1)                         # only round 2, but in context
    assert sorted(second.measurements) == ['c[1]']
    assert second.measurements['c[1]'].kind == 'tied'           # tied to round 1, which is outside
    assert len(second.detectors) == 1 and second.internal_detectors() == []
    alone = p.measurement_report(0, 1, context=False)           # round 2 on an arbitrary input state
    assert alone.measurements['c[1]'].kind == 'frame' and alone.detectors == []
    with pytest.raises(ValueError):
        p.measurement_report(0)


def test_non_clifford_spectator_keeps_the_detector():
    # a T gate on a qubit outside the check's support: the circuit is not Clifford any more
    # (stim could not do this), but the check still repeats, so the detector survives
    p = QProgram(4, bits=4, initial_states={2: '0'})
    p.add_circuit(parity_round(n=4, bits=(0,)), name="round 1")
    t = Circuit(4, bit_amount=4); t.add_gate(T(3))
    p.add_circuit(t, name="T on a spectator")
    p.add_circuit(parity_round(n=4, bits=(1,)), name="round 2")
    rep = p.measurement_report()
    assert len(rep.detectors) == 1
    assert {str(v) for v in rep.detectors[0].free_vars()} == {'c[0]', 'c[1]'}
    assert rep.measurements['c[1]'].kind == 'tied'


def test_non_clifford_inside_the_check_breaks_the_detector():
    # the same T, now on a qubit the check acts on: X_0 X_1 is no longer repeatable and the
    # reduction reports no detector -- the outcomes become independent random bits
    p = QProgram(3, bits=4, initial_states={2: '0'})
    p.add_circuit(parity_round(bits=(0,)), name="round 1")
    t = Circuit(3, bit_amount=4); t.add_gate(T(0))
    p.add_circuit(t, name="T inside the check")
    p.add_circuit(parity_round(bits=(1,)), name="round 2")
    rep = p.measurement_report()
    assert rep.detectors == []
    assert {m.kind for m in rep.measurements.values()} == {'frame'}


def test_report_table_and_summary_render():
    p = QProgram(3, bits=4, initial_states={2: '0'})
    p.add_circuit(parity_round(bits=(0,)), name="round 1")
    rep = p.measurement_report()
    text = str(rep)
    assert 'c[0]' in text and 'measurements in' in text and 'outcome space' in text
    (m,) = rep.of_kind(rep.measurements['c[0]'].kind)
    assert m.stage == 0 and m.qubit == 2 and m.timestep is not None
