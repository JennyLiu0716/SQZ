"""The decompilation flow as the default way of naming a logical gate (zxqec/codes/flow.py): a Clifford class of runs
is named from the stabilizers of its Choi state, with the same answers as the matrix route."""
import random
from fractions import Fraction

import numpy as np
import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, S
from pyzx.graph.scalar import Scalar
from pyzx.utils import VertexType

from zxqec.codes import APForm, IAPNormalForm, NotCliffordError, flow
from zxqec.codes.iapnf import _Pauli
from zxqec.gpf import reduce_gpf

from . import test_case_study_fold_h as F
from .programs import steane_t_injection
from .test_ccz import FROM_X, FROM_Y, FROM_Z, GHZ, P, _program as _ccz
from .test_code_switching import _switch
from .test_shor import _shor
from .test_teleportation import _teleport_cx


def _random_map(rng):
    """A random Clifford map: a circuit with some inputs and outputs plugged by Z or X states and effects."""
    n = rng.randint(1, 4)
    c = Circuit(n)
    for _ in range(rng.randint(1, 4 * n)):
        r = rng.random()
        if r < .35 or n == 1:
            q = rng.randrange(n)
            c.add_gate(HAD(q) if rng.random() < .5 else S(q))
        else:
            a, b = rng.sample(range(n), 2)
            c.add_gate(CNOT(a, b) if r < .7 else CZ(a, b))
    g = c.to_graph()
    ins, outs = list(g.inputs()), list(g.outputs())
    keep_in = [b for b in ins if rng.random() < .7]
    keep_out = [b for b in outs if rng.random() < .7]
    for b in ins + outs:
        if b not in keep_in and b not in keep_out:
            g.set_type(b, VertexType.Z if rng.random() < .5 else VertexType.X)
            g.set_phase(b, Fraction(rng.choice([0, 1])))
    g.set_inputs(tuple(keep_in))
    g.set_outputs(tuple(keep_out))
    return g, len(keep_in), len(keep_out)


def test_choi_stabilizers_match_the_matrix_on_every_shape_of_map():
    # isometries, projectors, states, effects, splits and merges: the Choi reading (outputs first, inputs bent)
    # gives the stabilizers and signs of the matrix's Choi vector
    rng = random.Random(1)
    checked = 0
    for _ in range(80):
        g, ki, ko = _random_map(rng)
        if ki + ko == 0:
            continue
        res = reduce_gpf(g.copy())
        if any(str(c) == '1' for c in res.constraints):
            continue                               # the zero map
        h = res.to_graph()
        h.scalar = Scalar()
        m = np.asarray(h.to_matrix(preserve_scalar=False)).reshape(2 ** ko, 2 ** ki)
        want = flow.StabMap(ki, ko, [_Pauli.hermitian(x, z, s)
                                     for x, z, s in flow._stabilizers_of_matrix(m, ki + ko)])
        got = flow.stab_map_of(h, ki, ko)
        assert got.key == want.key and np.array_equal(got.signs, want.signs)
        checked += 1
    assert checked > 60


def _answers(act):
    return [(b.operation, b.correction, b.unitary, b.same_as_reference, b.conditions) for b in act.branches()]


@pytest.mark.parametrize('build', [_shor, F.fold_transversal_h, _switch, _teleport_cx, steane_t_injection,
                                   lambda: _ccz([GHZ, FROM_X, FROM_Y, FROM_Z, P])])
def test_flow_and_matrix_routes_give_the_same_answers_for_every_block(build):
    p = build()
    actions = [p.logical_action()] + [p.logical_action(0, i) for i in range(p.num_stages)]
    for act in actions:
        assert _answers(act) == _answers(act.with_route('matrix'))


@pytest.mark.parametrize('build', [_shor, F.fold_transversal_h, _switch, _teleport_cx])
def test_clifford_programs_are_named_without_a_matrix(build):
    p = build()
    for act in [p.logical_action()] + [p.logical_action(0, i) for i in range(p.num_stages)]:
        act.branches()
        assert act.named_by['matrix'] == 0 and act.named_by['flow'] >= 1


def test_paths_and_normal_forms_follow_the_flow_chart():
    act = _shor().logical_action()
    assert act.path().startswith('Clifford isometry')
    assert isinstance(act.normal_form(), IAPNormalForm)
    t = steane_t_injection()
    whole = t.logical_action()
    assert whole.path().startswith('not Clifford')
    with pytest.raises(NotCliffordError):
        whole.normal_form()
    measuring = [t.logical_action(0, i) for i in range(t.num_stages)]
    projectors = [a for a in measuring if a.path().startswith('Clifford, not an isometry')]
    assert projectors and all(isinstance(a.normal_form(), APForm) for a in projectors)
    for a in projectors:                         # named through the Choi state, not a matrix
        a.branches()
        assert a.named_by['matrix'] == 0


def test_clifford_part_plugs_back_into_the_same_diagram():
    from pyzx.circuit.gates import T
    from zxqec.codes import clifford_plus_phases
    from zxqec.gpf import same_map
    c = Circuit(3)
    for g in (HAD(0), T(0), CNOT(0, 1), T(1), S(2), CZ(1, 2), T(2), HAD(1)):
        c.add_gate(g)
    cp = clifford_plus_phases(c.to_graph())
    assert len(cp.phases) == 3 and cp.is_isometry
    assert same_map(c.to_graph(), cp.to_graph())
