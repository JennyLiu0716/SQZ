"""Termination of the two loops of the reduction (appendix A of the paper).

The skeleton reduction of ``to_gpf``: every bialgebra step lowers the number of
Z-spiders that are neither leaves nor stems and touch no input (the lemma after
Yeh et al., Lemma B.1). This is checked here in the real loop, by recording the
count each time it looks for the next pair.

The alternation of ``reduce_gpf`` has no termination proof; these tests only
record that it settles on random circuits, and how many rounds it needs.
"""
import random

import pytest
from pyzx.circuit import Circuit
from pyzx.circuit.gates import CNOT, CZ, HAD, S, T, Measurement, Reset
from pyzx.circuit.graphparser import circuit_to_graph
from pyzx.utils import EdgeType, VertexType

import zxqec.gpf.normal_form as NF
import zxqec.gpf.reduce as R
from zxqec.gpf import reduce_gpf


def _random_circuits(seed, count):
    """Circuits of up to 8 qubits and 60 gates with H, S, T, CZ, CNOT and mid-circuit measurements."""
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(3, 8)
        c = Circuit(n, bit_amount=6)
        m = 0
        for _ in range(rng.randint(10, 60)):
            r = rng.random()
            a, b = rng.sample(range(n), 2)
            if r < 0.2:
                c.add_gate(HAD(a))
            elif r < 0.3:
                c.add_gate(S(a))
            elif r < 0.35:
                c.add_gate(T(a))
            elif r < 0.45:
                c.add_gate(CZ(a, b))
            elif r < 0.9:
                c.add_gate(CNOT(a, b))
            elif m < 6:
                c.add_gate(Measurement(a, result_symbol='c[%d]' % m))
                c.add_gate(Reset(a))
                m += 1
        yield circuit_to_graph(c, initialize_qubits=[rng.random() < 0.5 for _ in range(n)])


def test_every_skeleton_step_lowers_the_count(monkeypatch):
    counts = []
    find = NF._find_bialgebra_pair

    def recording(g):
        if not counts[-1]:
            # the proof's assumption: every edge between two spiders is plain when the loop starts
            for e in g.edges():
                u, v = g.edge_st(e)
                if VertexType.BOUNDARY not in (g.type(u), g.type(v)):
                    assert g.edge_type(e) == EdgeType.SIMPLE
        counts[-1].append(NF.skeleton_count(g))
        return find(g)
    monkeypatch.setattr(NF, '_find_bialgebra_pair', recording)
    steps = 0
    for g in _random_circuits(7, 120):
        counts.append([])
        NF.to_gpf(g)
        seq = counts[-1]
        assert all(b < a for a, b in zip(seq, seq[1:]))      # one step between consecutive looks
        assert len(seq) - 1 <= seq[0]
        steps += len(seq) - 1
    assert steps == 1234                                      # the number appendix A quotes


def _round_counter(monkeypatch, sizes=None):
    """Counts the rounds of reduce_gpf's alternation (one call of _remove_isolated_pairs per round);
    with ``sizes``, also records the number of vertices after each round's row reduction."""
    count = [0]
    pairs = R._remove_isolated_pairs

    def counting(h):
        count[0] += 1
        if sizes is not None:
            sizes[-1].append(h.num_vertices())
        return pairs(h)
    monkeypatch.setattr(R, '_remove_isolated_pairs', counting)
    return count


def test_the_alternation_settles_on_random_circuits(monkeypatch):
    sizes = []
    count = _round_counter(monkeypatch, sizes)
    rounds = []
    for g in _random_circuits(7, 120):
        count[0] = 0
        sizes.append([])
        reduce_gpf(g)
        rounds.append(count[0])
    assert max(rounds) == 14                                  # the number appendix A quotes
    # the loop stops at the first round that changes nothing, so every other round changed the
    # diagram; some of them did so without lowering the number of vertices, which is why a count
    # of spiders cannot prove termination
    middle = [(s[i - 1], s[i]) for s in sizes for i in range(1, len(s) - 1)]
    assert (sum(1 for before, after in middle if after >= before), len(middle)) == (161, 390)


def test_every_reduction_of_the_case_studies_settles_in_a_few_rounds(monkeypatch):
    import os
    import zxqec.codes.code as CC
    import zxqec.codes.logical as L
    from . import test_case_study_fold_h as F
    from .programs import steane_t_injection
    from .test_codes import FOLD_GROUPS
    from .test_compare import STEANE
    from zxqec.qprogram import program_from_stim
    count = _round_counter(monkeypatch)
    per = []

    def recording(*args, **kwargs):
        count[0] = 0
        out = reduce_gpf(*args, **kwargs)
        per.append(count[0])
        return out
    monkeypatch.setattr(CC, 'reduce_gpf', recording)
    monkeypatch.setattr(L, 'reduce_gpf', recording)
    path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'cultivation', 'd3_cultivation.stim')
    cultivation = [(0, 0, 'first checks', 'state preparation'), (1, 1, 'inject T', 'injection'),
                   (2, 2, 'checks', 'syndrome extraction'), (3, 4, 'double-check', 'cultivation')]
    found = []
    for p, groups in ((F.fold_transversal_h(), FOLD_GROUPS), (steane_t_injection(), STEANE),
                      (program_from_stim(open(path).read())[0], cultivation)):
        per.clear()
        for s in range(p.num_stages):
            p.code_at(s)                                      # question 1 at every cut
        p.summary(p.group_stages(groups))                     # questions 2 and 3 for every block
        p.logical_action()                                    # and for the whole program
        found.append(max(per))
    assert found == [2, 2, 5]                                 # the numbers appendix A quotes


def test_every_reduction_of_d5_cultivation_settles_within_seven_rounds(monkeypatch):
    # the same count on d = 5 cultivation (codes at every cut, the five blocks of test_cultivation, the whole
    # program); its gadget layer is simplified over more rounds than at d = 3
    import os
    import zxqec.codes.code as CC
    import zxqec.codes.logical as L
    from zxqec.qprogram import program_from_stim
    from .test_cultivation import GROUPS_D5
    count = _round_counter(monkeypatch)
    per = []

    def recording(*args, **kwargs):
        count[0] = 0
        out = reduce_gpf(*args, **kwargs)
        per.append(count[0])
        return out
    monkeypatch.setattr(CC, 'reduce_gpf', recording)
    monkeypatch.setattr(L, 'reduce_gpf', recording)
    path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'cultivation', 'd5_cultivation.stim')
    p = program_from_stim(open(path).read())[0]
    for s in range(p.num_stages):
        p.code_at(s)
    p.summary(p.group_stages(GROUPS_D5))
    p.logical_action()
    assert max(per) == 7                                      # the number appendix A and Section 5 quote


def test_a_skeleton_count_that_cannot_be_met_raises():
    # the bound is a hard limit: a smaller one than the lemma's is refused loudly, not silently cut
    g = next(g for g in _random_circuits(7, 50) if _needs_steps(g))
    with pytest.raises(RuntimeError):
        NF.to_gpf(g, max_steps=0)


def _needs_steps(g):
    seen = []
    find = NF._find_bialgebra_pair

    def recording(h):
        seen.append(1)
        return find(h)
    NF._find_bialgebra_pair = recording
    try:
        NF.to_gpf(g)
    finally:
        NF._find_bialgebra_pair = find
    return len(seen) > 1


# -- the row reduction settles after one changing round, when to_gpf's post-condition holds ---------

def _internal_edges(h):
    """Edges between a Z-spider not on an input and an X-spider not on an output, gadgets aside."""
    ins, outs = set(h.inputs()), set(h.outputs())
    bad = 0
    for e in h.edges():
        u, v = h.edge_st(e)
        if VertexType.BOUNDARY in (h.type(u), h.type(v)) or NF.is_leaf(h, u) or NF.is_leaf(h, v):
            continue
        z, x = (u, v) if h.type(u) == VertexType.Z else (v, u)
        if NF.is_stem(h, z) or NF.is_stem(h, x):
            continue
        bad += not (NF._touches(h, z, ins) or NF._touches(h, x, outs))
    return bad


def _alternate(h):
    """The row-reduction rounds of reduce_gpf with the gadget passes off, on a diagram already in GPF:
    ``(changed, spiders deleted, internal edges afterwards)`` per round."""
    out = []
    while True:
        n0 = h.num_vertices()
        changed = False
        for side in (VertexType.Z, VertexType.X):
            m = R.split_matrix(h, side)
            before = ([row[:] for row in m.entries], [str(p) for p in m.phases])
            R.reduce_rows(m)
            if m.entries != before[0] or [str(p) for p in m.phases] != before[1]:
                changed = True
            R.apply_matrix(h, m)
        R._remove_isolated_pairs(h)
        out.append((changed, n0 - h.num_vertices(), _internal_edges(h)))
        if not changed or len(out) > 50:
            return out


def _case_study_graphs():
    import os
    from . import test_case_study_fold_h as F
    from .programs import steane_t_injection
    from zxqec.qprogram import program_from_stim
    path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'circuits', 'cultivation', 'd3_cultivation.stim')
    return [F.fold_transversal_h().quantum_graph(), steane_t_injection().quantum_graph(),
            program_from_stim(open(path).read())[0].quantum_graph()]


def test_to_gpf_leaves_no_edge_between_spiders_off_the_boundary():
    # the skeleton loop stops only when no such edge is left, and padding and splitting add edges at
    # the boundary only: the hypothesis of the two-round lemma
    for g in list(_random_circuits(7, 120)) + _case_study_graphs():
        assert _internal_edges(NF.to_gpf(g)) == 0


def test_without_gadget_passes_the_alternation_takes_at_most_two_rounds(monkeypatch):
    count = _round_counter(monkeypatch)
    for g in list(_random_circuits(7, 120)) + _case_study_graphs():
        count[0] = 0
        reduce_gpf(g, gadgets=False)
        assert count[0] <= 2


def _random_gpf(rng, gadget_rate, boundary_property):
    """A random diagram in GPF: bipartite Z-X, inputs on Z-spiders, outputs on X-spiders, Pauli phases,
    some gadgets; with ``boundary_property`` every Z-X edge has its Z on an input or its X on an output."""
    from fractions import Fraction
    from pyzx.graph import Graph
    from pyzx.symbolic import new_var
    g = Graph()
    kinds = ['Z'] * rng.randint(2, 9) + ['X'] * rng.randint(2, 9)
    rng.shuffle(kinds)
    zs, xs = [], []
    for k in kinds:
        (zs if k == 'Z' else xs).append(g.add_vertex(VertexType.Z if k == 'Z' else VertexType.X, 0, 1))
    ins, outs = [], []
    for z in zs:
        if rng.random() < 0.35:
            b = g.add_vertex(VertexType.BOUNDARY, len(ins), 0)
            g.add_edge((b, z))
            ins.append(b)
    for x in xs:
        if rng.random() < 0.35:
            b = g.add_vertex(VertexType.BOUNDARY, len(outs), 3)
            g.add_edge((x, b))
            outs.append(b)
    zin = {n for b in ins for n in g.neighbors(b)}
    xout = {n for b in outs for n in g.neighbors(b)}
    p = rng.uniform(0.2, 0.7)
    for z in zs:
        for x in xs:
            if (not boundary_property or z in zin or x in xout) and rng.random() < p:
                g.add_edge((z, x))
    for v in zs + xs:
        r = rng.random()
        if r < 0.25:
            g.set_phase(v, Fraction(1))
        elif r < 0.45:
            g.set_phase(v, new_var('c[%d]' % rng.randint(0, 3), True, g.var_registry))
    for hubs, leaf_ty in ((xs, VertexType.Z), (zs, VertexType.X)):
        for h in hubs:
            if h not in zin | xout and rng.random() < gadget_rate:
                leaf = g.add_vertex(leaf_ty, 0, 4)
                g.set_phase(leaf, Fraction(1, 4))
                g.set_vdata(leaf, NF.GPF_ROLE_KEY, NF.LEAF)
                g.add_edge((h, leaf))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def test_two_rounds_on_random_diagrams_with_the_property():
    # the lemma, on diagrams built directly: with no internal edge, the second round changes nothing,
    # and no round creates an internal edge
    for gadget_rate in (0.0, 0.2):
        rng = random.Random(2026)
        tried = 0
        while tried < 1500:
            g = _random_gpf(rng, gadget_rate, boundary_property=True)
            if not NF.is_gpf(g):
                continue
            tried += 1
            rounds = _alternate(g.copy())
            assert len(rounds) <= 2 and not rounds[-1][0]
            assert all(internal == 0 for _changed, _deleted, internal in rounds)


def _load_counterexamples():
    import json
    import os
    from fractions import Fraction
    from pyzx.graph import Graph
    from pyzx.symbolic import new_var
    path = os.path.join(os.path.dirname(__file__), 'data', 'alternation_counterexamples.json')
    out = []
    for d in json.load(open(path))['diagrams']:
        g = Graph()
        new = {}
        for v in d['vertices']:                               # in the saved order, which fixes the pivots
            ty = {'Z': VertexType.Z, 'X': VertexType.X, 'B': VertexType.BOUNDARY}[v['type']]
            w = g.add_vertex(ty, 0, 0)
            if 'var' in v['phase']:
                g.set_phase(w, new_var(v['phase']['var'], True, g.var_registry))
            elif v['phase']['const'] != '0':
                g.set_phase(w, Fraction(v['phase']['const']))
            if v['leaf']:
                g.set_vdata(w, NF.GPF_ROLE_KEY, NF.LEAF)
            new[v['id']] = w
        for a, b in d['edges']:
            g.add_edge((new[a], new[b]))
        g.set_inputs(tuple(new[b] for b in d['inputs']))
        g.set_outputs(tuple(new[b] for b in d['outputs']))
        out.append((g, d))
    return out


def test_the_saved_counterexamples_need_more_rounds_without_the_property():
    # tests/data/alternation_counterexamples.json: diagrams with an internal edge, which to_gpf never
    # leaves, on which the alternation needs three or four rounds; so the lemma needs its hypothesis
    loaded = _load_counterexamples()
    assert len(loaded) == 16 and sum(1 for _g, d in loaded if d['rounds'] == 4) == 6
    for g, d in loaded:
        assert NF.is_gpf(g) and _internal_edges(g) > 0
        rounds = _alternate(g.copy())
        assert len(rounds) == d['rounds'] > 2
        assert any(ch and deleted <= 0 for ch, deleted, _i in rounds[1:]) == d['later_round_changes_without_deleting']


def test_the_gadget_passes_can_break_the_property():
    # which is why the lemma covers the alternation between gadget passes but not the whole loop
    from collections import Counter
    seen = Counter()
    for g in list(_random_circuits(7, 120)) + _case_study_graphs():
        h = R.to_gpf(g)
        R._scalar_constraints(h)
        for _ in range(60):
            changed = _alternate_once(h)
            how = None
            if not changed and R.simplify_gadgets(h):
                how = 'gadgets'
            if not changed and how is None:
                if R._expose_free_bits(h) is not None:
                    how = 'expose'
                elif R._resolve_checks(h) is not None:
                    how = 'resolve'
            if how is None and not changed:
                break
            if how is not None:
                seen[(how, _internal_edges(h) == 0)] += 1
    assert seen == {('gadgets', False): 100, ('gadgets', True): 134, ('expose', True): 28,
                    ('expose', False): 4, ('resolve', True): 16, ('resolve', False): 1}


def _alternate_once(h):
    changed = False
    for side in (VertexType.Z, VertexType.X):
        m = R.split_matrix(h, side)
        before = ([row[:] for row in m.entries], [str(p) for p in m.phases])
        R.reduce_rows(m)
        if m.entries != before[0] or [str(p) for p in m.phases] != before[1]:
            changed = True
        R.apply_matrix(h, m)
    R._remove_isolated_pairs(h)
    return changed
