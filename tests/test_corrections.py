"""Finding Pauli corrections from the Z-X and X-Z normal forms
(examples/FindingCorrections.ipynb and examples/FindingCorrectionsStepByStep.ipynb)."""
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, HAD, T, Measurement
from pyzx.utils import VertexType

from zxqec.gpf import equal_for_all_assignments, phases_modulo
from zxqec.qprogram import QProgram, find_corrections

from .test_teleportation import _teleport_cx

A, C, T_ = [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]


def _program(n, bits, initial, stages):
    p = QProgram(n, bits=bits, initial_states=initial)
    for name, gates in stages:
        c = Circuit(n, bit_amount=bits)
        for gate in gates:
            c.add_gate(gate)
        p.add_circuit(c, name=name)
    return p


def _teleport():
    return _program(3, 2, {1: '0', 2: '0'},
                    [('teleport', [HAD(1), CNOT(1, 2), CNOT(0, 1), HAD(0), Measurement(0, 0), Measurement(1, 1)])])


def _surgery():
    """A CNOT from C to T by lattice surgery through A, on distance-2 surface-code patches (ZZ merge, XX merge, split)."""
    def encode(p):
        return [CNOT(p[1], p[3]), HAD(p[0]), CNOT(p[0], p[1]), CNOT(p[0], p[2]), CNOT(p[0], p[3])]
    return _program(17, 9, {**{q: '0' for q in range(17) if q not in (5, 9)}, 1: '+'}, [
        ('encode C and T', encode(C) + encode(T_)),
        ('ancilla |+>', encode(A)),
        ('merge C, A: Z Z', [CNOT(q, 12) for q in (2, 3, 4, 5)] + [Measurement(12, 0)]),
        ('merge A, T: X X', [g for h, qs, bit in ((13, (1, 8), 1), (14, (3, 10), 2))
                             for g in [HAD(h)] + [CNOT(h, q) for q in qs] + [HAD(h), Measurement(h, bit)]]),
        ('split A, T', [g for h, qs, bit in ((15, (1, 3), 3), (16, (8, 10), 4))
                        for g in [CNOT(q, h) for q in qs] + [Measurement(h, bit)]]),
        ('measure A', [Measurement(q, 5 + k) for k, q in enumerate(A)])])


def _corrected(p, fix):
    c = Circuit(p.qubits, bit_amount=p.bits)
    for gate in fix.gates():
        c.add_gate(gate)
    p.add_circuit(c, name='corrections')
    return p.logical_action()


def test_teleportation_x_from_the_z_x_form_and_z_from_the_x_z_form():
    p = _teleport()
    fix = find_corrections(p)
    assert (fix.x, fix.z, fix.complete) == ({2: ('c[1]',)}, {2: ('c[0]',)}, True)
    act = _corrected(p, fix)
    assert act.is_deterministic() and {str(b.operation) for b in act.branches()} == {'I'}


def test_gate_teleportation_without_its_x_corrections():
    # the program keeps its Z corrections; the X-Z form still shows Z on 0, 2, 4, 6, a check of the toric code:
    # with the X corrections alone every run is already the same, and that is what is returned
    p = _teleport_cx(corrected=False)
    fix = find_corrections(p)
    assert (fix.x, fix.z, fix.complete) == ({1: ('c[0]',), 3: ('c[0]',)}, {}, True)


def test_lattice_surgery_cnot():
    p = _surgery()
    assert p.logical_action().verdict().startswith('deterministic up to a Pauli frame')
    fix = find_corrections(p)
    frame = ('c[0]', 'c[3]', 'c[5]', 'c[6]')
    assert fix.x == {8: ('c[3]',), 9: frame, 11: frame}
    assert fix.z == {4: ('c[1]', 'c[2]'), 5: ('c[1]', 'c[2]')} and fix.complete
    act = _corrected(p, fix)
    assert act.is_deterministic() and {str(b.operation) for b in act.branches()} == {'CNOT(0->1)'}


def test_t_injection_needs_an_s_not_a_pauli():
    p = _program(2, 1, {1: '0'}, [('inject T', [HAD(1), T(1), CNOT(0, 1), Measurement(1, 0)])])
    fix = find_corrections(p)
    assert (fix.x, fix.z, fix.complete) == ({}, {}, False)
    assert fix.describe().startswith('no Pauli correction found')


def _outcome_leaves(g):
    """(qubit of the output, colour, phase) for each one-legged spider next to an output's spider."""
    from zxqec.spacetime.subdiagram import get_logical_qubit
    out = set()
    for b in g.outputs():
        (s,) = list(g.neighbors(b))
        for w in g.neighbors(s):
            if g.type(w) != VertexType.BOUNDARY and g.vertex_degree(w) == 1:
                out.add((get_logical_qubit(g, b), g.type(w), str(g.phase(w))))
    return out


def test_the_recipe_level_by_level_as_in_the_step_by_step_notebook():
    for p, last, zx_leaves, xz_leaves in (
            (_teleport(), 0, {(2, VertexType.X, 'c[1]')}, {(2, VertexType.Z, 'c[0]')}),
            (_surgery(), 5, {(8, VertexType.X, 'c[3]'), (9, VertexType.X, 'c[0] + c[3] + c[5] + c[6]'),
                             (11, VertexType.X, 'c[0] + c[3] + c[5] + c[6]')},
             {(4, VertexType.Z, 'c[1] + c[2]'), (7, VertexType.Z, 'c[1] + c[2]')})):
        circuit = p.quantum_graph()
        zx_form = p.level_up(0, 0, last, 'all', rewrite='gpf')
        zx_apart = p.level_up(zx_form, 0, 0, 'all', rewrite='unfuse outcomes')
        xz_form = p.level_up(zx_apart, 0, 0, 'all', rewrite='x-z gpf')
        xz_apart = p.level_up(xz_form, 0, 0, 'all', rewrite='unfuse outcomes')
        for level in (zx_form, zx_apart, xz_form, xz_apart):
            r = p.reduced_level(level)
            assert equal_for_all_assignments(circuit, r.graph, constraints=r.detectors)
        for level, leaves in ((zx_apart, zx_leaves), (xz_apart, xz_leaves)):
            r = p.reduced_level(level)
            assert _outcome_leaves(phases_modulo(r.graph, r.detectors)) == leaves
