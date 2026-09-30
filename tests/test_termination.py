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
