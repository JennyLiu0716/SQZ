"""A small state-vector simulator for Stim-format circuits with T / T_DAG, independent of ZX (tests only)."""
import numpy as np

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
T = np.diag([1, np.exp(1j * np.pi / 4)])
PAULI = {'I': np.eye(2, dtype=complex), 'X': X, 'Y': Y, 'Z': Z}

def apply1(state, u, q):
    return np.moveaxis(np.tensordot(u, state, axes=([1], [q])), 0, q)

def apply_cx(state, c, t):
    s = state.copy()
    idx1 = [slice(None)] * state.ndim; idx1[c] = 1
    sub = s[tuple(idx1)]                              # the half with control = 1 (axis c removed)
    tt = t if t < c else t - 1
    s[tuple(idx1)] = np.flip(sub, axis=tt)
    return s

def measure(state, q, rng):
    p1 = float(np.sum(np.abs(np.take(state, 1, axis=q)) ** 2))
    out = int(rng.random() < p1)
    proj = [slice(None)] * state.ndim; proj[q] = 1 - out
    state = state.copy(); state[tuple(proj)] = 0
    return state / np.linalg.norm(state), out

def run(circ, rng):
    """One sampled run of a parsed StimCircuit; returns (final state, list of outcomes)."""
    n = circ.qubits
    state = np.zeros((2,) * n, dtype=complex); state[(0,) * n] = 1
    outs = []
    for ins in circ.instructions:
        qs = [int(t) for t in ins.targets]
        if ins.name in ('R', 'RX'):
            for q in qs:
                state, o = measure(state, q, rng)
                if o:
                    state = apply1(state, X, q)
                if ins.name == 'RX':
                    state = apply1(state, H, q)
        elif ins.name in ('M', 'MX'):
            for q in qs:
                if ins.name == 'MX':
                    state = apply1(state, H, q)
                state, o = measure(state, q, rng)
                outs.append(o)
        elif ins.name == 'H':
            for q in qs: state = apply1(state, H, q)
        elif ins.name in ('T', 'T_DAG'):
            u = T if ins.name == 'T' else T.conj()
            for q in qs: state = apply1(state, u, q)
        elif ins.name in ('CX', 'CNOT'):
            for c, t in zip(qs[::2], qs[1::2]): state = apply_cx(state, c, t)
        else:
            raise ValueError(ins.name)
    return state, outs

def expectation(state, pauli, qubits):
    """<state| P |state> for the Pauli string ``pauli`` on ``qubits`` (a sign '+'/'-' prefix allowed)."""
    sign = -1 if pauli.startswith('-') else 1
    letters = pauli.lstrip('+-')
    s = state
    for c, q in zip(letters, qubits):
        if c != 'I':
            s = apply1(s, PAULI[c], q)
    return sign * complex(np.vdot(state, s))
