"""Lattice surgery on distance-3 surface-code patches: rectangular patches, splits and merges, and their names.

The programs follow Kissinger, "Phase-free ZX diagrams are CSS codes", Section 7.1: a Z-split cuts a patch
in two by measuring the X checks of the new boundary, an X-merge joins two patches by measuring the X checks
across the seam, and a Z-split followed by an X-merge is a CNOT.
"""
import numpy as np
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, Measurement, Reset

from zxqec.codes import logical_action, surface_code_checks
from zxqec.qprogram import QProgram


def _faces(rows, cols, data, offset=(0, 0), parity=0):
    """The X checks of a ``rows x cols`` patch at ``offset``, as lists of qubits (``data`` maps cells to qubits)."""
    hx = surface_code_checks(rows, cols, parity=parity)[0]
    return [[data[(i // cols + offset[0], i % cols + offset[1])] for i in np.flatnonzero(row)] for row in hx]


def _x_check(ancilla, qubits, bit, reset=False):
    return ([Reset(ancilla)] if reset else []) + [HAD(ancilla)] + [CNOT(ancilla, q) for q in qubits] + \
        [HAD(ancilla), Measurement(ancilla, bit)]


def _encode(inp, column, checks, ancillas, bit):
    """Encode ``inp`` into a patch: fan it out onto the logical X (a column), measure it in the X basis, then
    measure the patch's X checks (the data qubits start in |0>)."""
    gates = [CNOT(inp, q) for q in column] + [HAD(inp), Measurement(inp, bit)]
    for k, (f, a) in enumerate(zip(checks, ancillas)):
        gates += _x_check(a, f, bit + 1 + k)
    return gates


def _program(n, bits, opens, stages):
    p = QProgram(n, bits=bits, initial_states={q: '0' for q in range(n) if q not in opens})
    for name, gates in stages:
        c = Circuit(n, bit_amount=bits)
        for g in gates:
            c.add_gate(g)
        p.add_circuit(c, name=name)
    return p


def test_rectangular_and_flipped_surface_code_checks():
    hx, hz = surface_code_checks(3)
    hx2, hz2 = surface_code_checks(3, 3, parity=0)
    assert np.array_equal(hx, hx2) and np.array_equal(hz, hz2)
    for rows, cols, parity in ((6, 3, 0), (3, 6, 0), (3, 3, 1)):
        hx, hz = surface_code_checks(rows, cols, parity=parity)
        assert hx.shape[1] == hz.shape[1] == rows * cols
        assert len(hx) + len(hz) == rows * cols - 1                 # one logical qubit
        assert not ((hx @ hz.T) % 2).any()                          # the checks commute
    # the other checkerboard: different X checks, the same number of each kind
    flipped_x, flipped_z = surface_code_checks(3, parity=1)
    assert {tuple(r) for r in flipped_x} != {tuple(r) for r in surface_code_checks(3)[0]}
    assert len(flipped_x) == len(flipped_z) == 4


def _split_program():
    """A 6x3 patch (rows 0-5) encoded from qubit 18, then cut between rows 2 and 3 (one XX measurement)."""
    data = {(r, c): r * 3 + c for r in range(6) for c in range(3)}
    checks = _faces(6, 3, data)
    seam = next(i for i, f in enumerate(checks) if sorted(f) == [data[(2, 0)], data[(2, 1)], data[(3, 0)], data[(3, 1)]])
    ancillas = list(range(19, 19 + len(checks)))
    n, bits = 19 + len(checks), 2 + len(checks)
    split = _x_check(ancillas[seam], [data[(2, 0)], data[(2, 1)]], bits - 1, reset=True)
    return _program(n, bits, (18,), [('encode', _encode(18, [data[(r, 0)] for r in range(6)], checks, ancillas, 0)),
                                     ('Z-split', split)])


def test_a_split_keeps_one_logical_qubit_and_is_a_z_split_between_the_two_halves():
    p = _split_program()
    before, after = p.code_at(0).data(), p.code_at(1).data()
    assert before.parameters == (18, 1, 3) and after.parameters == (18, 1, 3)
    # the two halves still share the product of their logical Z
    assert after.contains('ZZZ' + 'I' * 6 + 'ZZZ' + 'I' * 6)
    halves = after.separated([list(range(9)), list(range(9, 18))])
    assert [b.parameters for b in halves.blocks()] == [(9, 1, 3), (9, 1, 3)]
    level = p.group_stages([(0, 0, 'encode', 'encoder'), (1, 1, 'Z-split', None)])
    act = logical_action(p.block_graph(level, 1, classical=False), before, halves)
    assert act.operations() == {'Z-split'} and act.is_deterministic()


def _merge_program():
    """Two 3x3 patches side by side (columns 0-2 and 3-5), each encoded from an input, then merged."""
    data = {(r, c): r * 3 + c for r in range(3) for c in range(3)}
    data.update({(r, c): 9 + r * 3 + (c - 3) for r in range(3) for c in range(3, 6)})
    a, b = _faces(3, 3, data), _faces(3, 3, data, offset=(0, 3), parity=1)
    merged = _faces(3, 6, data)
    seam = [f for f in merged if any(q < 9 for q in f) and any(q >= 9 for q in f)]
    anc = list(range(20, 20 + len(a) + len(b) + len(seam)))
    n, bits = 20 + len(anc), 2 + len(anc)
    encode = (_encode(18, [data[(r, 0)] for r in range(3)], a, anc[:len(a)], 0)
              + _encode(19, [data[(r, 3)] for r in range(3)], b, anc[len(a):len(a) + len(b)], 1 + len(a)))
    merge = [g for k, f in enumerate(seam) for g in _x_check(anc[len(a) + len(b) + k], f, 2 + len(a) + len(b) + k)]
    return _program(n, bits, (18, 19), [('encode', encode), ('X-merge', merge)])


def test_a_merge_is_a_measurement_of_xx_named_by_its_outcome():
    p = _merge_program()
    assert [blk.parameters for blk in p.code_at(0).blocks()] == [(9, 1, 3), (9, 1, 3)]
    assert p.code_at(1).data().parameters == (18, 1, 3)
    level = p.group_stages([(0, 0, 'encode', 'encoder'), (1, 1, 'X-merge', 'logical measurement')])
    act = p.logical_action(level, 1)
    assert act.operations() == {'X-merge, XX = +1', 'X-merge, XX = -1'}
    assert not act.is_deterministic()
