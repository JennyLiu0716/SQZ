"""Protocols shared by several tests: the Steane-code T-gate injection."""

from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement, T

from zxqec.qprogram import ParityControlled, QProgram

#: the three weight-4 checks of the Steane code (qubit indices within a block)
STEANE_CHECKS = [(0, 3, 4, 5), (1, 3, 4, 6), (2, 4, 5, 6)]
IN_QUBIT = 3                                 #: the qubit of a block that carries the input
DATA, MAGIC, ANC = list(range(7)), list(range(7, 14)), list(range(14, 20))
N, BITS = 20, 13
READOUT = ['c[{}]'.format(6 + q) for q in range(7)]   #: the transversal readout of the magic block


def steane_encoder(block):
    """Encoder of the Steane code: the input on ``block[3]``, every other qubit in |0>."""
    gates = [CNOT(block[IN_QUBIT], block[t]) for t in (5, 6)]
    gates += [HAD(block[s]) for s in (0, 1, 2)]
    for s, support in zip((0, 1, 2), STEANE_CHECKS):
        gates += [CNOT(block[s], block[t]) for t in support if t != s]
    return gates


def _stage(gates):
    c = Circuit(N, bit_amount=BITS)
    for g in gates:
        c.add_gate(g)
    return c


def steane_t_injection(correct=True):
    """T on a Steane-encoded qubit by injecting an encoded magic state.

    Stages: encode the data (its input is open), prepare T|+> on the magic block
    and encode it, one round of checks on the magic block, a transversal CNOT
    from data to magic, a transversal readout of the magic block, and (with
    ``correct``) the correction S_L = (S^dag)^7 when the readout parity is 1.
    """
    p = QProgram(N, bits=BITS, initial_states={q: '0' for q in DATA + MAGIC + ANC if q != IN_QUBIT})
    p.add_circuit(_stage(steane_encoder(DATA)), name='encode data')
    p.add_circuit(_stage([HAD(MAGIC[IN_QUBIT]), T(MAGIC[IN_QUBIT])]), name='prepare T|+>')
    p.add_circuit(_stage(steane_encoder(MAGIC)), name='encode magic')
    checks = []
    for k, support in enumerate(STEANE_CHECKS):
        a = ANC[k]
        checks += [HAD(a)] + [CNOT(a, MAGIC[q]) for q in support] + [HAD(a), Measurement(a, k)]
    for k, support in enumerate(STEANE_CHECKS):
        a = ANC[3 + k]
        checks += [CNOT(MAGIC[q], a) for q in support] + [Measurement(a, 3 + k)]
    p.add_circuit(_stage(checks), name='check magic')
    p.add_circuit(_stage([CNOT(DATA[q], MAGIC[q]) for q in range(7)]), name='transversal CNOT')
    p.add_circuit(_stage([Measurement(MAGIC[q], 6 + q) for q in range(7)]), name='read out magic')
    if correct:
        p.add_circuit(_stage([ParityControlled('S^dag', DATA[q], READOUT) for q in range(7)]),
                      name='correct')
    return p
