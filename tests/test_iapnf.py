"""zxqec.codes.iap_normal_form: Clifford isometries in isometry affine-with-phases normal form (Yeh et al., Thm 4.2).

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

from zxqec.codes import IAPNormalForm, NotACodeError, code_of, iap_normal_form, reduced_ap_form
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
    """The layers have the shape of Theorem 4.2, with the conventions Algorithm 3 produces."""
    a = set(nf.hadamards)
    assert set(nf.in_phases) <= a and all(p in a and q in a for p, q in nf.cz_in)
    assert all(p not in a and q in a and p < q for p, q in nf.xcz_in)    # an X-CZ joins an earlier pivot input
    assert all(str(nf.logical_signs[q]) == '0' for q in a)
    if len(nf.check_rows):
        r, piv, _ = rref(nf.check_rows)
        assert (r == nf.check_rows).all()                          # Z-checks in reduced row echelon form
        assert not nf.logical_rows[:, piv].any()                   # logical rows reduced modulo them
        assert all(q not in piv for q in nf.out_phases)            # U_O on the free columns only
        assert all(p not in piv and q not in piv for p, q in nf.cz_out)


def test_the_reduced_ap_form_of_the_choi_state():
    # step 1 of Algorithm 3: B in reduced row echelon form, inputs first, the phases and CZ gates only on the free
    # variables, and its diagram with the inputs on the left is the isometry itself; the cases read off it
    rng = random.Random(8)
    for _ in range(60):
        n = rng.randrange(1, 6)
        k = rng.randrange(0, n + 1)
        g = _isometry(_random_gates(rng, n, rng.randrange(0, 25)), n, k)
        ap = reduced_ap_form(g)
        assert zx.compare_tensors(ap.to_graph(inputs=k), g, preserve_scalar=False)
        if len(ap.rows):
            r, piv, _ = rref(ap.rows)
            assert (r == ap.rows).all() and tuple(piv) == ap.pivots
        assert set(ap.phases) <= set(ap.free) and all(a in ap.free and b in ap.free for a, b in ap.cz)
        nf = iap_normal_form(g)
        assert nf.hadamards == tuple(q for q in ap.free if q < k)
        rows = zip(ap.rows, ap.pivots)
        assert set(nf.xcz_in) == {(p, j) for row, p in rows for j in nf.hadamards if p < k and row[j]}


def test_an_input_in_the_rows_of_two_pivots():
    # Case 3 for an input in the rows of two earlier pivot inputs: one X-CZ gate with each of them
    from fractions import Fraction
    nf = IAPNormalForm(k=3, n=4, qubits=(0, 1, 2, 3), hadamards=(2,), in_phases={2: Fraction(1, 2)}, cz_in=(),
                       xcz_in=((0, 2), (1, 2)),
                       logical_rows=np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 1]], np.uint8),
                       logical_signs=(1, 0, 0), check_rows=np.array([[1, 0, 0, 1]], np.uint8), check_signs=(0,),
                       out_phases={1: Fraction(1, 2)}, cz_out=((2, 3),))
    g = nf.to_graph()
    ap = reduced_ap_form(g)
    assert ap.pivots == (0, 1, 3) and ap.free == (2, 4, 5, 6)
    assert [int(r[2]) for r in ap.rows] == [1, 1, 0]                # input 2 is in the rows of inputs 0 and 1
    assert iap_normal_form(g) == nf
    assert zx.compare_tensors(ap.to_graph(inputs=3), g, preserve_scalar=False)


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
    assert nf.hadamards == nf.cz_in == nf.xcz_in == () and nf.in_phases == {}
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
    assert {'$U_I$', 'X-Z GPF', '$U_O$'} <= texts
    import matplotlib.pyplot as plt
    plt.close('all')


def _u_i_choices(k):
    """Every U_I of Algorithm 3 up to its Paulis: H on A, S and CZ within A, X-CZ from i outside A to j > i in A."""
    import itertools
    for size in range(k + 1):
        for a_set in itertools.combinations(range(k), size):
            rest = [q for q in range(k) if q not in a_set]
            pairs = [(a, b) for a in a_set for b in a_set if a <= b]
            xpairs = [(i, j) for i in rest for j in a_set if i < j]
            for gbits in itertools.product([0, 1], repeat=len(pairs)):
                for xbits in itertools.product([0, 1], repeat=len(xpairs)):
                    yield (a_set, tuple(a for (a, b), v in zip(pairs, gbits) if v and a == b),
                           tuple((a, b) for (a, b), v in zip(pairs, gbits) if v and a != b),
                           tuple(p for p, v in zip(xpairs, xbits) if v))


def test_u_i_has_one_choice_per_coset():
    # each U_I (up to its Paulis) maps a different set of Paulis to Z-type ones (what the core cannot absorb), and
    # every such set, a Lagrangian subspace, is reached: prod (2^i + 1) of them on k qubits
    import math
    from zxqec.codes.gf2 import nullspace

    def image(x, z, a_set, s_gates, cz, xcz):
        # U P U^dagger on the symplectic vector (x | z), signs ignored, gates in the order of U_I
        x, z = x.copy(), z.copy()
        for q in s_gates:
            z[q] ^= x[q]
        for a, b in cz:
            z[a] ^= x[b]
            z[b] ^= x[a]
        for q in a_set:
            x[q], z[q] = z[q], x[q]
        for a, b in xcz:
            x[a] ^= z[b]
            x[b] ^= z[a]
        return x
    for k in (1, 2, 3):
        seen, count = set(), 0
        for choice in _u_i_choices(k):
            basis = [(np.eye(k, dtype=np.uint8)[i], np.zeros(k, np.uint8)) for i in range(k)]
            basis += [(np.zeros(k, np.uint8), np.eye(k, dtype=np.uint8)[i]) for i in range(k)]
            lam = nullspace(np.array([image(x, z, *choice) for x, z in basis], np.uint8).T)
            seen.add(rref(lam)[0].tobytes())
            count += 1
        assert count == len(seen) == math.prod(2 ** i + 1 for i in range(1, k + 1))


def _all_forms(n, k):
    """Every normal form from k inputs to n outputs (all choices of the layers, with Algorithm 3's conventions)."""
    import itertools
    from fractions import Fraction
    vectors = lambda m: [np.array([(v >> (m - 1 - i)) & 1 for i in range(m)], np.uint8) for v in range(1, 2 ** m)]
    checks_of = {}
    for j in range(n - k + 1):
        found = {}
        for rows in itertools.combinations(vectors(n), j):
            r, piv, _ = rref(np.array(rows, np.uint8).reshape(j, n))
            if len(piv) == j:
                found[r.tobytes()] = r
        checks_of[j] = list(found.values())
    for j, all_checks in checks_of.items():
        for checks in all_checks:
            piv = rref(checks)[1] if j else []
            free = [c for c in range(n) if c not in piv]
            for lrows in itertools.permutations(vectors(len(free)), k):
                logical = np.zeros((k, n), np.uint8)
                for i, row in enumerate(lrows):
                    logical[i, free] = row
                if k + j and len(rref(np.concatenate([logical, checks], 0))[1]) != k + j:
                    continue
                pairs = [(a, b) for a in free for b in free if a < b]
                for a_set, s_gates, cz, xcz in _u_i_choices(k):
                    rest = [q for q in range(k) if q not in a_set]
                    for signs in itertools.product([0, 1], repeat=len(rest) + j):
                        lsigns = [0] * k
                        for q, v in zip(rest, signs):
                            lsigns[q] = v
                        for zs in itertools.product([0, 1], repeat=len(a_set)):
                            in_phases = {a: Fraction(int(a in s_gates) + 2 * z, 2) for a, z in zip(a_set, zs)}
                            for phases in itertools.product(range(4), repeat=len(free)):
                                for czs in itertools.product([0, 1], repeat=len(pairs)):
                                    yield IAPNormalForm(
                                        k=k, n=n, qubits=tuple(range(n)), hadamards=a_set,
                                        in_phases={a: v for a, v in in_phases.items() if v}, cz_in=cz, xcz_in=xcz,
                                        logical_rows=logical, logical_signs=tuple(lsigns), check_rows=checks,
                                        check_signs=tuple(signs[len(rest):]),
                                        out_phases={f: Fraction(p, 2) for f, p in zip(free, phases) if p},
                                        cz_out=tuple(p for p, v in zip(pairs, czs) if v))


def test_every_small_normal_form_comes_back_and_is_a_different_map():
    # 6, 24, 60 and 720 normal forms: the numbers of 1-qubit stabilizer states, 1-qubit Cliffords (up to a
    # phase), 2-qubit stabilizer states and isometries from 1 qubit to 2
    for (n, k), expected in {(1, 0): 6, (1, 1): 24, (2, 0): 60, (2, 1): 720}.items():
        maps = set()
        count = 0
        for nf in _all_forms(n, k):
            g = nf.to_graph()
            assert iap_normal_form(g) == nf
            m = np.asarray(g.to_matrix(preserve_scalar=False), complex)
            m = m / m.flat[np.argmax(np.abs(m) > 1e-9)]
            maps.add((np.round(m.real, 6) + 0.0).tobytes() + (np.round(m.imag, 6) + 0.0).tobytes())
            count += 1
        assert count == len(maps) == expected


def test_other_presentations_of_the_same_isometry():
    # re-synthesised by PyZX, simplified by full_reduce (no longer a circuit), or reduced by reduce_gpf
    from zxqec.gpf import reduce_gpf
    rng = random.Random(9)
    for _ in range(12):
        n = rng.randrange(4, 8)
        k = rng.randrange(0, 4)
        gates = _random_gates(rng, n, rng.randrange(10, 50))
        g = _isometry(gates, n, k)
        nf = iap_normal_form(g)
        c = Circuit(n)
        for gate in gates:
            c.add_gate(gate)
        h = c.to_graph()
        zx.full_reduce(h)
        assert iap_normal_form(_isometry(zx.extract_circuit(h).to_basic_gates().gates, n, k)) == nf
        simplified = g.copy()
        zx.full_reduce(simplified)
        assert iap_normal_form(simplified) == nf
        assert iap_normal_form(reduce_gpf(g).to_graph()) == nf


def test_named_codes():
    # [[8,3,2]], the [[15,1,3]] Reed-Muller code and the 2x2 toric code, as CSS encoders read by code_of, and
    # the same after single-qubit Cliffords on every qubit (codes that are not CSS)
    from zxqec.codes.gf2 import nullspace

    def encoder(hx, hz, n):
        hx, hz = np.array(hx, np.uint8).reshape(-1, n), np.array(hz, np.uint8).reshape(-1, n)
        span, lx = [list(r) for r in rref(hx)[0]], []
        for v in nullspace(hz):
            if len(rref(np.array(span + [list(v)], np.uint8))[1]) > len(span):
                span.append(list(v))
                lx.append(v)
        from pyzx.graph import Graph
        g = Graph()
        xs = [g.add_vertex(VertexType.X, j, 2) for j in range(n)]
        outs = [g.add_vertex(VertexType.BOUNDARY, j, 3) for j in range(n)]
        for x, o in zip(xs, outs):
            g.add_edge((x, o))
        ins = []
        for i, row in enumerate(list(lx) + list(hx)):
            z = g.add_vertex(VertexType.Z, i, 1)
            if i < len(lx):
                b = g.add_vertex(VertexType.BOUNDARY, i, 0)
                g.add_edge((b, z))
                ins.append(b)
            for j in np.flatnonzero(row):
                g.add_edge((z, xs[j]))
        g.set_inputs(tuple(ins))
        g.set_outputs(tuple(outs))
        return g

    cols = range(1, 16)
    g1 = [[(c >> b) & 1 for c in cols] for b in range(4)]
    g2 = [[(c >> a) & (c >> b) & 1 for c in cols] for a in range(4) for b in range(a + 1, 4)]
    faces = [[1 if ((v >> ax) & 1) == val else 0 for v in range(8)] for ax in range(3) for val in (0, 1)]
    edge = lambda x, y, d: 2 * ((y % 2) * 2 + (x % 2)) + d
    star = [[int(q in (edge(x, y, 0), edge(x, y, 1), edge(x - 1, y, 0), edge(x, y - 1, 1))) for q in range(8)]
            for x in range(2) for y in range(2)]
    plaq = [[int(q in (edge(x, y, 0), edge(x, y, 1), edge(x + 1, y, 1), edge(x, y + 1, 0))) for q in range(8)]
            for x in range(2) for y in range(2)]
    diagrams = [encoder([[1] * 8], faces, 8), encoder(g1, g1 + g2, 15), encoder(star, plaq, 8)]
    rng = random.Random(4)
    from fractions import Fraction
    from pyzx.utils import EdgeType
    for g in list(diagrams):
        h = g.copy()
        for b in list(h.outputs()):
            (w,) = list(h.neighbors(b))
            h.remove_edge(h.edge(w, b))
            v = h.add_vertex(VertexType.Z, h.qubit(b), 2.5, Fraction(rng.randrange(4), 2))
            h.add_edge((w, v), EdgeType.HADAMARD if rng.random() < 0.5 else EdgeType.SIMPLE)
            h.add_edge((v, b), EdgeType.HADAMARD if rng.random() < 0.5 else EdgeType.SIMPLE)
        diagrams.append(h)
    kinds = set()
    for g in diagrams:
        code = code_of(g)
        kinds.add(code.is_css)
        nf = iap_normal_form(code)
        _assert_shape(nf)
        assert zx.compare_tensors(nf.to_graph(), code.encoder(), preserve_scalar=False)
    assert kinds == {True, False}


# -- logical_action through the normal forms (thesis, Section 4.5) -------------------------------------------

def _same_up_to_scalar(a, b, constraints=()):
    from pyzx.graph.scalar import Scalar
    a, b = a.copy(), b.copy()
    a.scalar, b.scalar = Scalar(), Scalar()
    names = sorted(set(boolean_variables(a)) | set(boolean_variables(b))
                   | {str(v) for c in constraints for v in c.free_vars()})
    return equal_for_all_assignments(a, b, names, list(constraints))


def _rm15_after(diagonal):
    """The [[15,1,3]] code, encoded by a CNOT circuit, then the gates ``diagonal``; then T on every qubit."""
    from pyzx.circuit.gates import T
    from zxqec.qprogram import QProgram
    n = 15
    hx = np.array([[(c >> b) & 1 for c in range(1, 16)] for b in range(4)], np.uint8)
    rx, piv, _ = rref(hx)
    lx = np.ones(n, np.uint8)
    for row, p in zip(rx, piv):
        if lx[p]:
            lx ^= row
    j = next(c for c in range(n) if lx[c] and c not in piv)
    gates = [CNOT(j, c) for c in range(n) if lx[c] and c != j]
    for row, p in zip(rx, piv):
        gates += [HAD(p)] + [CNOT(p, c) for c in range(n) if row[c] and c != p]
    prog = QProgram(n, initial_states={q: '0' for q in range(n) if q != j})
    for name, gs in (('encode', gates + list(diagonal)), ('transversal T', [T(q) for q in range(n)])):
        c = Circuit(n)
        for gate in gs:
            c.add_gate(gate)
        prog.add_circuit(c, name=name)
    return prog


def test_a_diagonal_block_is_checked_on_the_css_core():
    # [[15,1,3]] after a random diagonal Clifford is not CSS; transversal T commutes with that layer, so the two
    # U_O cancel and the block is the transversal T of the CSS core: the logical T^dag
    from zxqec.codes import logical_action
    rng = random.Random(3)
    diagonal = ([S(q) for q in range(15) if rng.random() < 0.4]
                + [CZ(a, b) for a in range(15) for b in range(a + 1, 15) if rng.random() < 0.15])
    p = _rm15_after(diagonal)
    code = p.code_at(0)
    assert not code.is_css
    act = p.logical_action(0, 1)
    assert act.via == 'iapnf' and act.u_o_cancelled is True
    assert act.verdict() == 'deterministic: every run implements T^dag'
    assert act.on_core().verdict() == act.verdict()
    old = logical_action(p.block_graph(0, 1, classical=False), code, code, via='encoder')
    assert old.verdict() == act.verdict() and _same_up_to_scalar(act.graph, old.graph)
    assert 'U_O cancelled' in act.describe()


def test_a_block_that_is_not_diagonal_keeps_u_o():
    from zxqec.codes import logical_action
    from .test_noncss import _on_every_qubit
    five = _wheel_program().code_at(0).data()
    block = _on_every_qubit(five, [HAD, S])
    act = logical_action(block, five, five)
    assert act.via == 'iapnf' and act.u_o_cancelled is False and act.operations() == {'SH'}
    assert _same_up_to_scalar(act.graph, logical_action(block, five, five, via='encoder').graph)


def test_both_routes_give_the_same_logical_action_on_random_codes():
    # codes read off random programs (not CSS, signs from outcomes included), then a random block of
    # diagonal gates (S, CZ, T, Z) or of any Clifford gates; the composite through the normal forms is the
    # same map as through the ideal encoders, in every run the detectors allow
    from pyzx.circuit.gates import T, Z
    from zxqec.codes import logical_action
    from zxqec.qprogram import QProgram
    from zxqec.gpf import reduce_gpf
    prng = random.Random(31)
    checked = cancelled = kept = 0
    while checked < 30:
        p = _random_program(prng, prng.randint(3, 5), prng.randint(1, 2), prng.randint(1, 2), (3, 10))
        try:
            code = code_of(p.prefix_graph(p.num_stages - 1))
        except NotACodeError:
            continue
        if code.is_css:
            continue
        n = p.qubits
        diagonal = prng.random() < 0.6
        gates = []
        for _ in range(prng.randint(1, 6)):
            a, b = prng.sample(range(n), 2)
            gates.append(prng.choice([S(a), T(a), Z(a), CZ(a, b)]) if diagonal
                         else prng.choice([HAD(a), S(a), CNOT(a, b), CZ(a, b)]))
        c = Circuit(n, bit_amount=6)
        for gate in gates:
            c.add_gate(gate)
        p.add_circuit(c, name='block')
        s = p.num_stages - 1
        try:
            act = p.logical_action(0, s)
        except NotACodeError:
            continue                          # the block leaves no code to read
        old = logical_action(p.block_graph(0, s, classical=False), act.code_in, act.code_out,
                             context=reduce_gpf(p.quantum_graph()).detectors, via='encoder')
        assert act.via == 'iapnf'
        assert _same_up_to_scalar(act.graph, old.graph, list(act.detectors) + list(old.detectors) + act.context)
        cancelled += bool(act.u_o_cancelled)
        kept += not act.u_o_cancelled
        checked += 1
    assert cancelled >= 2 and kept >= 5                 # a random diagonal block usually changes the code


def test_the_diagonal_test():
    from pyzx.circuit.gates import NOT, T
    from zxqec.codes.logical import _is_diagonal
    from zxqec.spacetime.subdiagram import set_logical_qubit

    def block(gates, n=3):
        c = Circuit(n)
        for gate in gates:
            c.add_gate(gate)
        g = c.to_graph()
        for i, b in enumerate(g.inputs()):
            set_logical_qubit(g, b, i)
        for i, b in enumerate(g.outputs()):
            set_logical_qubit(g, b, i)
        return g
    assert _is_diagonal(block([T(0), S(1), CZ(0, 2), T(2)]))
    assert _is_diagonal(block([CNOT(0, 1), T(1), CNOT(0, 1)]))        # a phase gadget
    assert _is_diagonal(block([]))
    for gates in ([HAD(0)], [CNOT(0, 1)], [NOT(2)], [HAD(0), T(0), HAD(0)]):
        assert not _is_diagonal(block(gates))


def test_pyzx_css_encoder_and_an_unlabelled_block():
    # PyZX's own CSS encoder (pyzx.css) for the [[15,1,3]] code, a layer of S and CZ gates after it (not CSS),
    # and transversal T as a plain circuit diagram, its wires matched by position
    from pyzx.circuit.gates import T
    from pyzx.css import generate_css_encoder_graph
    from pyzx.linalg import Mat2
    from zxqec.codes import logical_action
    sx = Mat2([[(c >> b) & 1 for c in range(1, 16)] for b in range(4)])
    enc, _ = generate_css_encoder_graph(sx, Mat2([[1] * 15]), 'Z-X')
    enc.auto_detect_io()
    layer = Circuit(15)
    for gate in [S(0), S(4), S(9), CZ(1, 2), CZ(2, 7), CZ(3, 11), CZ(5, 14), CZ(8, 12)]:
        layer.add_gate(gate)
    code = code_of(enc + layer.to_graph())
    assert code.parameters == (15, 1, 3) and not code.is_css
    t = Circuit(15)
    for q in range(15):
        t.add_gate(T(q))
    act = logical_action(t.to_graph(), code, code)
    assert act.u_o_cancelled is True and act.operations() == {'T^dag'} == act.on_core().operations()
