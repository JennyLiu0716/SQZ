"""zxqec.codes.iap_normal_form: Clifford isometries in isometry affine-with-phases normal form (Figure 4.12).

Every normal form is checked against the isometry it came from by tensor contraction
(PyZX's, in every run the signs allow), and the same map presented in different ways
must give the same normal form.
"""
import random

import numpy as np
import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, S
from pyzx.utils import VertexType

from zxqec.codes import IAPNormalForm, NotACodeError, code_of, iap_normal_form
from zxqec.codes.gf2 import rref
from zxqec.gpf import boolean_variables, equal_for_all_assignments

from .programs import steane_t_injection
from .test_noncss import _random_program, _wheel_program


def _isometry(gates, n, k):
    """The circuit ``gates`` on ``n`` qubits, inputs on qubits ``0..k-1``, the others starting in ``|0>``."""
    c = Circuit(n)
    for g in gates:
        c.add_gate(g)
    g = c.to_graph()
    for b in list(g.inputs())[k:]:
        g.set_type(b, VertexType.X)
    g.set_inputs(tuple(list(g.inputs())[:k]))
    return g


def _random_gates(rng, n, depth):
    gates = []
    for _ in range(depth):
        r = rng.random()
        if n == 1 or r < 0.3:
            q = rng.randrange(n)
            gates.append(HAD(q) if rng.random() < 0.5 else S(q))
        else:
            a, b = rng.sample(range(n), 2)
            gates.append(CNOT(a, b) if r < 0.65 else CZ(a, b))
    return gates


def _assert_shape(nf):
    """The layers have the shape of Figure 4.12, with the conventions that make the form unique."""
    a = set(nf.hadamards)
    assert set(nf.s_gates) <= a and all(p in a and q in a for p, q in nf.cz_in)
    assert all(p in a and q not in a and q > p for p, q in nf.xcz_in)
    if len(nf.check_rows):
        r, piv, _ = rref(nf.check_rows)
        assert (r == nf.check_rows).all()                          # Z-checks in reduced row echelon form
        assert not nf.logical_rows[:, piv].any()                   # logical rows reduced modulo them
        assert all(q not in piv for q in nf.out_phases)            # U_O on the free columns only
        assert all(p not in piv and q not in piv for p, q in nf.cz_out)


def test_random_isometries_are_reproduced():
    rng = random.Random(3)
    for _ in range(80):
        n = rng.randrange(1, 6)
        k = rng.randrange(0, n + 1)
        g = _isometry(_random_gates(rng, n, rng.randrange(0, 25)), n, k)
        nf = iap_normal_form(g)
        _assert_shape(nf)
        assert zx.compare_tensors(nf.to_graph(), g, preserve_scalar=False)


def test_the_form_is_unique():
    # the same isometry, written with gates that act on the |0> helpers first (they leave |0> alone) or
    # in a different order of commuting gates, has the same normal form
    rng = random.Random(5)
    for _ in range(60):
        n = rng.randrange(2, 6)
        k = rng.randrange(0, n)
        gates = _random_gates(rng, n, rng.randrange(1, 20))
        nf = iap_normal_form(_isometry(gates, n, k))
        helpers = list(range(k, n))
        for _ in range(3):
            pre = []
            for _ in range(rng.randrange(1, 5)):
                if len(helpers) >= 2 and rng.random() < 0.6:
                    a, b = rng.sample(helpers, 2)
                    pre.append(CNOT(a, b) if rng.random() < 0.5 else CZ(a, b))
                else:
                    pre.append(S(rng.choice(helpers)))
            assert iap_normal_form(_isometry(pre + gates, n, k)) == nf


def test_codes_equal_their_ideal_encoders():
    # CSS and non-CSS codes, signs that are measurement outcomes included; a code and the diagram of
    # its ideal encoder give the same normal form
    codes = [_wheel_program().code_at(0).data(), steane_t_injection().code_at(0).data(),
             steane_t_injection().code_at(3)]
    prng = random.Random(12)
    while len(codes) < 40:
        p = _random_program(prng, prng.randint(3, 5), prng.randint(0, 2), prng.randint(1, 3), (3, 10))
        try:
            codes.append(code_of(p.prefix_graph(p.num_stages - 1)))
        except NotACodeError:
            continue
    assert sum(not c.is_css for c in codes) >= 10
    symbolic = 0
    for code in codes:
        nf = iap_normal_form(code)
        _assert_shape(nf)
        enc, g = code.encoder(), nf.to_graph()
        names = sorted(set(boolean_variables(enc)) | set(boolean_variables(g)))
        assert equal_for_all_assignments(enc, g, names)
        if names:
            symbolic += 1
        else:
            assert iap_normal_form(enc) == nf
    assert symbolic >= 3


def test_the_five_qubit_code():
    # the ring graph code: no U_I, the input copied in the X basis onto all five qubits, and a ring of CZs
    # with a pi on every qubit (the pi gives the logical X its sign +)
    nf = iap_normal_form(_wheel_program().code_at(0).data())
    assert (nf.k, nf.n) == (1, 5)
    assert nf.hadamards == nf.s_gates == nf.cz_in == nf.xcz_in == ()
    assert nf.logical_rows.tolist() == [[1, 1, 1, 1, 1]] and len(nf.check_rows) == 0
    assert set(nf.cz_out) == {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
    assert all(nf.out_phases[q] == 1 for q in range(5))
    assert 'U_O' in nf.describe()


def test_the_drawing_names_the_three_layers():
    import matplotlib
    matplotlib.use('Agg')
    rng = random.Random(7)
    nf = iap_normal_form(_isometry(_random_gates(rng, 4, 20), 4, 2))
    assert isinstance(nf, IAPNormalForm)
    fig = nf.draw()
    texts = {t.get_text() for t in fig.axes[0].texts}
    assert {'U_I', 'X-Z GPF', 'U_O'} <= texts
    import matplotlib.pyplot as plt
    plt.close('all')


def test_u_i_has_one_choice_per_coset():
    # Hadamards on A, S and CZ within A, X-CZ from a in A to j > a outside A: each choice gives a different set
    # of Paulis that U_I maps to Z-type ones (what the core cannot absorb), and every set is reached
    import itertools
    import math
    from zxqec.codes.gf2 import nullspace
    from zxqec.codes.iapnf import _conj, _Pauli
    for k in (1, 2, 3):
        seen, count = set(), 0
        for size in range(k + 1):
            for a_set in itertools.combinations(range(k), size):
                rest = [q for q in range(k) if q not in a_set]
                pairs = [(a, b) for a in a_set for b in a_set if a <= b]
                xpairs = [(a, j) for a in a_set for j in rest if j > a]
                for gbits in itertools.product([0, 1], repeat=len(pairs)):
                    for xbits in itertools.product([0, 1], repeat=len(xpairs)):
                        gates = ([('S', a) for (a, b), v in zip(pairs, gbits) if v and a == b]
                                 + [('CZ', a, b) for (a, b), v in zip(pairs, gbits) if v and a != b]
                                 + [('H', a) for a in a_set]
                                 + [('XCZ', a, j) for (a, j), v in zip(xpairs, xbits) if v])
                        images = [_conj(_Pauli.single(k, i, t), gates) for t in 'XZ' for i in range(k)]
                        lam = nullspace(np.array([p.x for p in images], np.uint8).T)
                        seen.add(rref(lam)[0].tobytes())
                        count += 1
        assert count == len(seen) == math.prod(2 ** i + 1 for i in range(1, k + 1))
