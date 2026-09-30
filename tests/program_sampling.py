"""Sampling a QProgram with a state-vector simulator, independent of ZX (tests only).

Used to check the derived detectors against the circuit itself: every sampled
run must satisfy them (soundness), and the sampled outcome strings must span an
affine space of dimension ``(number of results) - (number of independent
detectors)`` (completeness). The second is a proof, not a statistic: a
deterministic parity outside the detectors' span would keep every sample in a
smaller space.
"""
from fractions import Fraction

import numpy as np

from zxqec.gpf import boolean_variables, reduce_gpf
from zxqec.qprogram.declared import _RESET, _row, _rref
from .statevector import H, X, apply1, apply_cx, measure


def _phase(angle):
    return np.diag([1, np.exp(1j * np.pi * float(angle))]).astype(complex)


def _rotation(axis, angle):
    u = _phase(angle)
    return u if axis == 'Z' else H @ u @ H


def _apply_cz(state, a, b):
    s = state.copy()
    idx = [slice(None)] * state.ndim
    idx[a], idx[b] = 1, 1
    s[tuple(idx)] *= -1
    return s


def _random_qubit(rng):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    return v / np.linalg.norm(v)


def sample_run(program, rng):
    """One run of ``program``: prepared qubits in their states, inputs in random states; returns the outcomes."""
    n = program.qubits
    state = np.array([1], dtype=complex)
    for q in range(n):
        s = program._initial_states.get(q)
        v = (np.array([1, 0], complex) if s == '0' else np.array([1, 1], complex) / np.sqrt(2) if s == '+'
             else _random_qubit(rng))
        state = np.kron(state, v)
    state = state.reshape((2,) * n)
    out = {}
    for s in range(program.num_stages):
        if program.stage(s).kind == 'permutation':          # relabels positions; the identity on qubits
            continue
        for g in program.stage_circuit(s).gates:
            name = type(g).__name__
            if name == 'CNOT':
                state = apply_cx(state, g.control, g.target)
            elif name == 'CZ':
                state = _apply_cz(state, g.control, g.target)
            elif name == 'HAD':
                state = apply1(state, H, g.target)
            elif name == 'NOT':
                state = apply1(state, X, g.target)
            elif name in ('ZPhase', 'S', 'T', 'Z'):
                angle = Fraction(g.phase) * (-1 if getattr(g, 'adjoint', False) else 1)
                state = apply1(state, _phase(angle), g.target)
            elif name == 'XPhase':
                state = apply1(state, _rotation('X', Fraction(g.phase)), g.target)
            elif name == 'ParityControlled':
                if sum(out[b] for b in g.bits) % 2:
                    state = apply1(state, _rotation(g.axis, g.angle), g.target)
            elif name == 'Measurement':
                state, o = measure(state, g.target, rng)
                out[g.result_symbol or 'c[{}]'.format(g.result_bit)] = o
            elif name == 'Reset':
                state, o = measure(state, g.target, rng)
                if o:
                    state = apply1(state, X, g.target)
            else:
                raise ValueError("the sampler does not know the gate {}".format(name))
    return out


def detectors_on_results(program):
    """The derived detectors that involve measurement results only: ``(names, rows [lin | const])``."""
    g = program.quantum_graph()
    res = reduce_gpf(g)
    names = list(boolean_variables(g))
    for c in res.detectors:
        for v in c.free_vars():
            if v.name not in names:
                names.append(v.name)
    resets = [x for x in names if _RESET.fullmatch(x)]
    results = [x for x in names if not _RESET.fullmatch(x)]
    order = resets + results
    col = {x: i for i, x in enumerate(order)}
    rows = np.array([_row(c, col) for c in res.constraints], dtype=np.uint8).reshape(-1, len(order) + 1)
    red, pivots = _rref(rows)
    keep = [r for r, p in zip(red, pivots) if p >= len(resets)]
    return results, np.array([np.concatenate([r[len(resets):-1], r[-1:]]) for r in keep],
                             dtype=np.uint8).reshape(-1, len(results) + 1)


def affine_dimension(strings):
    """The dimension of the affine span of 0/1 vectors."""
    if len(strings) < 2:
        return 0
    base = strings[0]
    diffs = np.array([(s ^ base) for s in strings[1:]], dtype=np.uint8)
    red, _ = _rref(np.concatenate([diffs, np.zeros((len(diffs), 1), np.uint8)], axis=1))
    return len(red)
