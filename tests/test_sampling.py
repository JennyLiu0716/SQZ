"""The derived detectors against sampled runs of the case studies (state-vector simulation, no ZX).

Sound: every sampled run satisfies every derived detector on the results.
Complete: the sampled outcome strings span an affine space of dimension
(results) - (independent detectors), so no parity outside the detectors' span
is deterministic. The inputs are in random states, so that no parity is
deterministic by accident of the input.
"""
import numpy as np

from . import test_case_study_fold_h as F
from .program_sampling import affine_dimension, detectors_on_results, sample_run
from .programs import steane_t_injection
from .statevector import run
from .test_compare import _miswired_fold_h
from zxqec.qprogram import program_from_stim

CULTIVATION = 'examples/circuits/cultivation/d3_cultivation.stim'


def _check(results, dets, samples):
    strings = [np.array([s[r] for r in results], dtype=np.uint8) for s in samples]
    violated = sum(int((d[:-1] @ x + d[-1]) % 2) for x in strings for d in dets)
    return violated, affine_dimension(strings)


def test_t_injection_detectors_are_sound_and_complete():
    p = steane_t_injection()
    results, dets = detectors_on_results(p)
    rng = np.random.default_rng(1)
    violated, dim = _check(results, dets, [sample_run(p, rng) for _ in range(16)])
    assert (len(results), len(dets)) == (13, 9)
    assert violated == 0 and dim == 13 - 9


def test_fold_transversal_h_detectors_are_sound_and_complete():
    p = F.fold_transversal_h()
    results, dets = detectors_on_results(p)
    rng = np.random.default_rng(1)
    violated, dim = _check(results, dets, [sample_run(p, rng) for _ in range(24)])
    assert (len(results), len(dets)) == (34, 24)
    assert violated == 0 and dim == 34 - 24


def test_cultivation_detectors_are_sound_and_complete():
    import os
    path = os.path.join(os.path.dirname(__file__), '..', CULTIVATION)
    p, circ = program_from_stim(open(path).read())
    results, dets = detectors_on_results(p)
    rng = np.random.default_rng(3)
    samples = [{'c[{}]'.format(i): o for i, o in enumerate(run(circ, rng)[1])} for _ in range(20)]
    violated, dim = _check(results, dets, samples)
    assert (len(results), len(dets)) == (31, 26)
    assert violated == 0 and dim == 31 - 26


def test_the_check_tells_the_mis_wired_circuit_apart():
    # the mis-wired fold-transversal H satisfies its own 22 detectors and fills 34 - 22 dimensions,
    # and its runs break detectors of the correct circuit
    bad = _miswired_fold_h()
    results, dets = detectors_on_results(bad)
    rng = np.random.default_rng(2)
    samples = [sample_run(bad, rng) for _ in range(24)]
    violated, dim = _check(results, dets, samples)
    assert len(dets) == 22 and violated == 0 and dim == 34 - 22
    good_results, good_dets = detectors_on_results(F.fold_transversal_h())
    assert _check(good_results, good_dets, samples)[0] > 0
