# zxqec - space-time ZX-diagrams for dynamic quantum error correction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A reduced diagram as a sum of terms, one for each value of its hidden spiders, and what each term does.

After :func:`~zxqec.gpf.reduce_gpf`, a diagram is in generalized parity form: every
input has its own Z-spider and every output its own X-spider. The other spiders
are *hidden*: Z-spiders not on an input are bits summed over, X-spiders not on an
output are parity checks. Splitting each hidden Z-spider into ``|0...0> + |1...1>``
and each hidden X-spider into ``|+...+> + |-...->`` writes the diagram as a sum of
``2^(n_i + n_o)`` terms (L. Yeh's decomposition). The diagram falls apart into
connected *parts*, and the terms of the whole are the products of the terms of the
parts, so each part is split on its own; a part with more than ``max_terms`` terms
is reported but not split. Each term is reduced again and cut into *pieces*: its
connected parts, each split further into a product over qubits where its matrix is
one (a term can stay connected after its reduction and still be a product). So a
term reads as single-qubit operators where it is one, and names the qubits it links
where it is not.

The terms depend on how the hidden bits are labelled: replacing a hidden bit ``z``
by ``z + (parity of some inputs)`` changes the terms but not their sum. Both are
offered: ``labelling='as reduced'`` keeps the labels :func:`reduce_gpf` chose, and
``labelling='fewest links'`` adds inputs to the hidden bits so that each input's
Z-spider reaches as few outputs as possible. On the d = 3 cultivation double-check
the first gives two terms that link all seven qubits, the second ``I`` and
``T X T^dag`` on every qubit.

For each single-qubit piece the report gives its name, when it is one of a short
list, and what it does to an X, Y or Z error on that qubit (``U E U^-1``). For every
piece proportional to a unitary, linking qubits or not, it also gives the Pauli
string on the outputs that an X, Y or Z error on each input becomes, up to a sign.

The check reported for distance is that, in every part, every term maps each Pauli
error to the same Pauli, up to a sign (:attr:`Decomposition.same_error_action`).
Products of single-qubit Clifford gates are not enough: the d = 3 double-check
splits into ``I`` and ``T X T^dag`` on every qubit and still loses fault distance
with ``T`` (Chan et al., arXiv:2609.17706), since ``T X T^dag`` turns an X error into
a Y error while ``I`` keeps it; with ``S`` in place of ``T`` the terms are ``I`` and
``Y``, which keep every Pauli up to a sign, and the distance is kept. The check is
not a proof of distance, and it is made without ``split_gadgets``: split at its hub,
a phase gadget is a sum of Pauli strings, so a bare ``T`` splits into ``I`` and ``Z``
and would pass. Whether the terms are single-qubit is reported as
information, and only for the non-Clifford parts: a Clifford part generally has
terms that link qubits, as one CNOT does.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.utils import VertexType

from .normal_form import GPF_ROLE_KEY, is_leaf
from .reduce import GPFResult, evaluate_phase, reduce_gpf

__all__ = ["decompose_terms", "Decomposition", "Part", "Term", "Piece", "LABELLINGS"]

LABELLINGS = ('as reduced', 'fewest links')

_P = {'X': np.array([[0, 1], [1, 0]], dtype=complex), 'Y': np.array([[0, -1j], [1j, 0]]),
      'Z': np.diag([1, -1]).astype(complex)}
_T = np.diag([1, np.exp(1j * np.pi / 4)])
_S = np.diag([1, 1j])
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_NAMED = {
    'I': np.eye(2), 'X': _P['X'], 'Y': _P['Y'], 'Z': _P['Z'], 'H': _H, 'S': _S, 'S^dag': _S.conj().T,
    'T': _T, 'T^dag': _T.conj().T, 'T X T^dag': _T @ _P['X'] @ _T.conj().T, 'T^dag X T': _T.conj().T @ _P['X'] @ _T,
}
_STATES = {'0': np.array([1, 0]), '1': np.array([0, 1]), '+': np.array([1, 1]) / np.sqrt(2),
           '-': np.array([1, -1]) / np.sqrt(2), '+i': np.array([1, 1j]) / np.sqrt(2),
           '-i': np.array([1, -1j]) / np.sqrt(2)}
_STATES.update({'T+': _T @ _STATES['+'], 'T^dag+': _T.conj().T @ _STATES['+']})


def _proportional(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.ravel(a), np.ravel(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return bool(na > 1e-12 and nb > 1e-12 and abs(abs(np.vdot(a, b)) - na * nb) < 1e-9 * na * nb)


def _name(m: np.ndarray) -> str | None:
    """The name of a 2x2 matrix, a state (2x1) or an effect (1x2) from a short list, up to a scalar."""
    if m.shape == (2, 2):
        return next((k for k, v in _NAMED.items() if _proportional(m, v)), None)
    if m.shape == (2, 1):
        return next(('|{}>'.format(k) for k, s in _STATES.items() if _proportional(m, s)), None)
    if m.shape == (1, 2):
        return next(('<{}|'.format(k) for k, s in _STATES.items() if _proportional(m, s.conj())), None)
    return None


def _error_action(m: np.ndarray) -> dict[str, str] | None:
    """For a 2x2 matrix proportional to a unitary: each of X, Y, Z -> the Pauli m E m^-1 is proportional to."""
    if m.shape != (2, 2) or not np.abs(m).max() > 0:
        return None
    m = m / np.abs(m).max()                            # a piece carries the term's scalar, which can be tiny
    mm = m @ m.conj().T
    if abs(mm[0, 0]) < 1e-12 or not np.allclose(mm, mm[0, 0] * np.eye(2), atol=1e-9 * abs(mm[0, 0])):
        return None
    inv = np.linalg.inv(m)
    return {e: next((q for q in 'XYZ' if _proportional(m @ _P[e] @ inv, _P[q])), 'not a Pauli') for e in 'XYZ'}


_PAULI4 = np.stack([np.eye(2), _P['X'], _P['Y'], _P['Z']])


def _pauli_string(m: np.ndarray, k: int) -> str | None:
    """The Pauli string (one of I, X, Y, Z per factor, first factor first) a 2^k x 2^k matrix is proportional to."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    out, inn, idx = letters[:k], letters[k:2 * k], 'ABCDEFGHIJKLMNOP'[:k]
    operands: list[Any] = [m.reshape((2,) * (2 * k))]
    subs = [out + inn]
    for j in range(k):                                 # tr(P m) for every Pauli string P at once
        operands.append(_PAULI4)
        subs.append(idx[j] + inn[j] + out[j])
    c = np.einsum(','.join(subs) + '->' + idx, *operands, optimize=True)
    big = np.argwhere(np.abs(c) > 1e-9 * np.abs(c).max())
    return ''.join('IXYZ'[i] for i in big[0]) if len(big) == 1 else None


def _pauli_action(m: np.ndarray | None, inputs: tuple[Any, ...],
                  outputs: tuple[Any, ...]) -> dict[tuple[Any, str], str] | None:
    """For a matrix proportional to a unitary: (input qubit, E) -> the Pauli string ``m E m^-1`` is proportional
    to, written on the output qubits (``'I'`` if none, ``'not a Pauli'`` if it is not one); None otherwise."""
    if m is None or not inputs or len(inputs) != len(outputs) or not np.abs(m).max() > 0:
        return None
    m = m / np.abs(m).max()                            # a piece carries the term's scalar, which can be tiny
    mm = m @ m.conj().T
    if abs(mm[0, 0]) < 1e-12 or not np.allclose(mm, mm[0, 0] * np.eye(len(mm)), atol=1e-9 * abs(mm[0, 0])):
        return None
    inv = np.linalg.inv(m)
    action = {}
    for i, q in enumerate(inputs):
        for e in 'XYZ':
            error = np.eye(1)
            for j in range(len(inputs)):
                error = np.kron(error, _P[e] if j == i else np.eye(2))
            s = _pauli_string(m @ error @ inv, len(outputs))
            action[(q, e)] = 'not a Pauli' if s is None else (
                ' '.join('{}{}'.format(a, r) for a, r in zip(s, outputs) if a != 'I') or 'I')
    return action


def _sorted(qubits: Any) -> list[Any]:
    return sorted(qubits, key=lambda q: (not isinstance(q, (int, float)), q if isinstance(q, (int, float)) else str(q)))


@dataclass
class Piece:
    """A connected piece of a term: the qubits it reads and writes, and its matrix (outputs x inputs)."""
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    matrix: np.ndarray | None              #: None when the piece has more than ``max_legs`` legs
    name: str | None                       #: e.g. 'T X T^dag' or '<+|'; None if not in the short list
    errors: dict[str, str] | None          #: what a single-qubit unitary piece does to X, Y, Z errors
    paulis: dict[tuple[Any, str], str] | None = None   #: (input, E) -> the Pauli string E becomes (unitary pieces)

    @property
    def single_qubit(self) -> bool:
        return len(self.inputs) <= 1 and len(self.outputs) <= 1

    def describe(self) -> str:
        where = ', '.join(str(q) for q in _sorted(set(self.inputs) | set(self.outputs)))
        if not self.single_qubit:
            return 'links qubits {} (inputs {}, outputs {})'.format(where, list(self.inputs), list(self.outputs))
        what = self.name or ('a {}x{} matrix'.format(*self.matrix.shape) if self.matrix is not None else '?')
        if self.inputs and self.outputs and self.inputs != self.outputs:
            what += ' from qubit {} to qubit {}'.format(self.inputs[0], self.outputs[0])
        errors = '' if self.errors is None else ' ({})'.format(', '.join('{}->{}'.format(e, q)
                                                                           for e, q in self.errors.items()))
        return 'qubit {}: {}{}'.format(where, what, errors)


@dataclass
class Term:
    """One term of a part: the values of its hidden spiders, and its pieces (none if the term is zero)."""
    values: tuple[int, ...]
    zero: bool
    pieces: list[Piece] = field(default_factory=list)
    graph: BaseGraph | None = None         #: the term, reduced (up to a non-zero scalar)
    diagram: BaseGraph | None = None       #: the term before reduction; the terms of a part add up to the part

    @property
    def single_qubit(self) -> bool:
        return all(p.single_qubit for p in self.pieces)

    def error_action(self) -> dict[Any, dict[str, str] | None]:
        """Qubit -> what the term does to X, Y, Z errors there (single-qubit unitary pieces only)."""
        return {p.inputs[0]: p.errors for p in self.pieces if len(p.inputs) == 1 and len(p.outputs) == 1}

    def pauli_action(self) -> dict[tuple[Any, str], str | None]:
        """(input qubit, E) -> the Pauli string an E error there becomes, up to a sign (``'not a Pauli'`` if it
        is not one); None where the piece is not proportional to a unitary or was not contracted. Pieces with no
        output (an input that ends in an effect, such as a measured qubit) or no input (a state) are left out: an
        error there does not reach an output."""
        return {(q, e): None if p.paulis is None else p.paulis[(q, e)]
                for p in self.pieces if p.inputs and p.outputs for q in p.inputs for e in 'XYZ'}

    def describe(self) -> list[str]:
        return [p.describe() for p in sorted(self.pieces, key=lambda p: _sorted(p.inputs + p.outputs)[:1])]


@dataclass
class Part:
    """A connected part of the reduced diagram, with its hidden spiders and its terms."""
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    hidden: list[str]                      #: e.g. 'Z-spider z1 (+ inputs 14)'
    terms: list[Term] | None               #: None if the part has more than ``max_terms`` terms
    diagram: BaseGraph | None = None       #: the part of the reduced diagram

    @property
    def nonzero(self) -> list[Term]:
        return [t for t in self.terms or [] if not t.zero]

    @property
    def single_qubit(self) -> bool | None:
        return None if self.terms is None else all(t.single_qubit for t in self.nonzero)

    @property
    def clifford(self) -> bool | None:
        """Every phase of the part (in the run) is a multiple of pi/2; None if not known."""
        if self.diagram is None:
            return None
        for v in self.diagram.vertices():
            try:
                phase = Fraction(self.diagram.phase(v))
            except (TypeError, ValueError):
                return None
            if (2 * phase).denominator != 1:
                return False
        return True

    @property
    def same_error_action(self) -> bool | None:
        """Every term turns each X, Y, Z error on each input into the same Pauli string, up to a sign.

        False if, for some error on some input, two terms give different Paulis (an
        error that ends in an effect in one term and reaches an output in another
        counts as different), or a term gives something that is not a Pauli. Otherwise
        None if the part was not split, or a term has a piece with inputs and outputs
        that is not proportional to a unitary (it maps no Pauli to a Pauli; with
        ``labelling='as reduced'`` the terms are often such projections) or was not
        contracted (more than ``max_legs`` legs). Inputs that end in an effect in every
        term are left out (:meth:`Term.pauli_action`). Meaningful only without
        ``split_gadgets``, which writes every gadget as Pauli strings (see the module
        docstring).
        """
        if self.terms is None:
            return None
        actions = [t.pauli_action() for t in self.nonzero]
        unknown = False
        for key in set().union(*actions) if actions else ():
            images = [a.get(key, 'ends in an effect') for a in actions]
            if 'not a Pauli' in images:
                return False
            if None in images:
                unknown = True
            elif len(set(images)) > 1:
                return False
        return None if unknown else True


@dataclass
class Decomposition:
    """The result of :func:`decompose_terms`: the parts of the reduced diagram and their terms."""
    labelling: str
    parts: list[Part]

    @property
    def complete(self) -> bool:
        """Every part was split."""
        return all(p.terms is not None for p in self.parts)

    @property
    def single_qubit(self) -> bool | None:
        """Every non-zero term of every part is a product of single-qubit pieces (None if a part was not split)."""
        values = [p.single_qubit for p in self.parts]
        return False if False in values else (None if None in values else True)

    @property
    def non_clifford_single_qubit(self) -> bool | None:
        """Every non-zero term of every non-Clifford part is a product of single-qubit pieces (None if there is no
        non-Clifford part, or one was not split). The terms of a Clifford part generally link qubits."""
        values = [p.single_qubit for p in self.parts if p.clifford is False]
        if not values:
            return None
        return False if False in values else (None if None in values else True)

    @property
    def same_error_action(self) -> bool | None:
        """In every part, every term turns each X, Y, Z error into the same Pauli, up to a sign: the check for
        distance (see the module docstring). False if some part fails; else None if some part cannot tell."""
        values = [p.same_error_action for p in self.parts]
        return False if False in values else (None if None in values else True)

    def table(self, max_terms_shown: int = 8) -> str:
        n_hidden = sum(len(p.hidden) for p in self.parts)
        lines = ['labelling {!r}: {} part(s), {} hidden spider(s)'.format(self.labelling, len(self.parts), n_hidden)]
        for i, p in enumerate(self.parts, 1):
            head = 'part {} ({}inputs {}, outputs {})'.format(
                i, {True: 'Clifford; ', False: 'non-Clifford; '}.get(p.clifford, ''), list(p.inputs), list(p.outputs))
            if not p.hidden:
                body = p.terms[0].describe() if p.terms else []
                lines.append(head + ': no hidden spiders' + (': ' + body[0] if len(body) == 1 else ''))
                if len(body) > 1:
                    lines += ['    ' + b for b in body]
                continue
            if p.terms is None:
                lines.append(head + ': {} hidden spiders, {} terms, not split'.format(len(p.hidden), 2 ** len(p.hidden)))
                continue
            lines.append(head + ': hidden {}; {} terms, {} non-zero'.format('; '.join(p.hidden), len(p.terms),
                                                                         len(p.nonzero)))
            for t in p.terms[:max_terms_shown]:
                lines.append('  term {}:{}'.format(''.join(map(str, t.values)), ' zero' if t.zero else ''))
                lines += ['      ' + d for d in t.describe()]
            if len(p.terms) > max_terms_shown:
                lines.append('  ... {} more terms'.format(len(p.terms) - max_terms_shown))
        lines.append('in every part, every term maps each Pauli error to the same Pauli (up to a sign): {}'.format(
            self.same_error_action))
        lines.append('every term of the non-Clifford parts single-qubit: {} (every term single-qubit: {})'.format(
            self.non_clifford_single_qubit, self.single_qubit))
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.table()


def _qubit(g: BaseGraph, b: Any, index: int) -> Any:
    from ..spacetime.subdiagram import get_logical_qubit
    q = get_logical_qubit(g, b)
    return index if q is None else q


def _components(g: BaseGraph) -> list[set]:
    left, out = set(g.vertices()), []
    while left:
        stack, comp = [left.pop()], set()
        while stack:
            v = stack.pop()
            if v not in comp:
                comp.add(v)
                stack.extend(g.neighbors(v))
        left -= comp
        out.append(comp)
    return out


def _restricted(g: BaseGraph, comp: set) -> BaseGraph:
    """The part of ``g`` on the vertices ``comp``, with its own inputs and outputs (vertex numbers kept)."""
    sub = g.clone()
    sub.remove_vertices([v for v in list(sub.vertices()) if v not in comp])
    sub.set_inputs(tuple(b for b in g.inputs() if b in comp))
    sub.set_outputs(tuple(b for b in g.outputs() if b in comp))
    return sub


def _factors(m: np.ndarray, ins: tuple[Any, ...],
             outs: tuple[Any, ...]) -> list[tuple[tuple[Any, ...], tuple[Any, ...], np.ndarray]]:
    """A piece's matrix as a product over qubits where it is one: ``[(inputs, outputs, matrix)]``.

    A qubit's legs (its output and its input, either may be missing) are split off
    when the operator has Schmidt rank 1 between them and the other legs; what does
    not split stays one piece. A term can stay connected after its reduction and
    still be a product (a phase gadget split at its hub gives ``I x I`` and ``Z x Z``).
    """
    labels = list(dict.fromkeys(list(outs) + list(ins)))
    if len(labels) <= 1:
        return [(ins, outs, m)]
    k_out = len(outs)
    legs = {q: ([('out', outs.index(q))] if q in outs else []) + ([('in', ins.index(q))] if q in ins else [])
            for q in labels}
    axis = {('out', i): i for i in range(k_out)}
    axis.update({('in', i): k_out + i for i in range(len(ins))})
    t = m.reshape((2,) * (k_out + len(ins))).transpose([axis[leg] for q in labels for leg in legs[q]])
    t = t.reshape([2 ** len(legs[q]) for q in labels])
    remaining, out = list(labels), []
    for q in labels:
        if len(remaining) == 1:
            break
        j = remaining.index(q)
        u, sv, vh = np.linalg.svd(np.moveaxis(t, j, 0).reshape(t.shape[j], -1), full_matrices=False)
        if len(sv) > 1 and sv[1] > 1e-9 * sv[0]:
            continue                                   # q is linked to the others
        shape = [t.shape[i] for i in range(len(remaining)) if i != j]
        out.append((q, u[:, 0] * sv[0]))
        t = vh[0].reshape(shape)
        remaining.pop(j)
    if len(remaining) == 1:
        out.append((remaining[0], t.reshape(-1)))
        remaining = []
    pieces = []
    for q, vec in out:
        has_out, has_in = q in outs, q in ins
        pieces.append(((q,) if has_in else (), (q,) if has_out else (),
                       vec.reshape(2 if has_out else 1, 2 if has_in else 1)))
    if remaining:                                      # the linked rest, back to outputs x inputs
        r_out = tuple(q for q in outs if q in remaining)
        r_in = tuple(q for q in ins if q in remaining)
        order = [leg for q in remaining for leg in legs[q]]
        t = t.reshape((2,) * len(order))
        t = t.transpose([order.index(('out', outs.index(q))) for q in r_out] +
                        [order.index(('in', ins.index(q))) for q in r_in])
        pieces.append((r_in, r_out, t.reshape(2 ** len(r_out), 2 ** len(r_in))))
    return pieces


def _pieces(g: BaseGraph, qin: dict, qout: dict, max_legs: int) -> list[Piece]:
    out = []
    for comp in _components(g):
        cin = [b for b in g.inputs() if b in comp]
        cout = [b for b in g.outputs() if b in comp]
        if not cin and not cout:
            continue                                   # a scalar
        ins, outs = tuple(qin[b] for b in cin), tuple(qout[b] for b in cout)
        if len(cin) + len(cout) > max_legs:
            out.append(Piece(ins, outs, None, None, None))
            continue
        m = zx.tensor_to_matrix(zx.tensorfy(_restricted(g, comp), preserve_scalar=True), len(cin), len(cout))
        for f_in, f_out, f in _factors(m, ins, outs):
            errors = _error_action(f) if len(f_in) == 1 and len(f_out) == 1 else None
            out.append(Piece(f_in, f_out, f, _name(f), errors, _pauli_action(f, f_in, f_out)))
    return out


def _split_part(g: BaseGraph, labelling: str, split_gadgets: bool, max_terms: int, max_legs: int,
                qin: dict, qout: dict) -> Part:
    in_spider = {b: next(iter(g.neighbors(b))) for b in g.inputs()}
    out_spider = {b: next(iter(g.neighbors(b))) for b in g.outputs()}
    leaves = {v for v in g.vertices() if is_leaf(g, v)}
    hubs = {next(iter(g.neighbors(l))) for l in leaves}
    ins, outs = set(in_spider.values()), set(out_spider.values())
    hidden_z = sorted(v for v in g.vertices() if g.type(v) == VertexType.Z and v not in ins and v not in leaves)
    hidden_x = sorted(v for v in g.vertices() if g.type(v) == VertexType.X and v not in outs and v not in hubs)
    gadget_hubs = sorted(hubs) if split_gadgets else []
    split = hidden_z + hidden_x + gadget_hubs
    part_in, part_out = tuple(qin[b] for b in g.inputs()), tuple(qout[b] for b in g.outputs())

    # the labelling: hidden bit z is replaced by z + (parity of the inputs in offset[z])
    offset: dict[Any, list[Any]] = {z: [] for z in hidden_z}
    if labelling == 'fewest links' and hidden_z:
        rows = {z: {w for w in g.neighbors(z) if w in outs} for z in hidden_z}
        for b, v in in_spider.items():
            best = None
            for c in itertools.product((0, 1), repeat=len(hidden_z)):
                r = {w for w in g.neighbors(v) if w in outs}
                for z, bit in zip(hidden_z, c):
                    if bit:
                        r ^= rows[z]
                key = (len(r), sum(c), c)
                if best is None or key < best[0]:
                    best = (key, c)
            for z, bit in zip(hidden_z, best[1]):
                if bit:
                    offset[z].append(b)
    hidden = ['Z-spider z{}{}'.format(k + 1, ' (+ inputs {})'.format(', '.join(str(qin[b]) for b in offset[z]))
                                      if offset[z] else '') for k, z in enumerate(hidden_z)]
    hidden += ['X-spider x{}'.format(k + 1) for k in range(len(hidden_x))]
    hidden += ['gadget hub {}'.format(k + 1) for k in range(len(gadget_hubs))]
    if 2 ** len(split) > max_terms:
        return Part(part_in, part_out, hidden, None, g)

    terms = []
    for values in itertools.product((0, 1), repeat=len(split)):
        t = g.clone()
        for v, val in zip(split, values):
            other = VertexType.X if t.type(v) == VertexType.Z else VertexType.Z
            u = t.add_vertex(other, phase=val, row=t.row(v) + 0.5, qubit=t.qubit(v))
            t.add_edge((u, v))
            for b in offset.get(v, []):               # z + inputs = val: the same parity check reads the inputs
                t.add_edge((u, in_spider[b]))
        exact = t.clone()
        for v in t.vertices():                         # the markings of the first reduction do not hold here
            if t.vdata(v, GPF_ROLE_KEY, None) is not None:
                t.set_vdata(v, GPF_ROLE_KEY, None)
        tres = reduce_gpf(t)
        if tres.is_zero:
            terms.append(Term(values, True, diagram=exact))
            continue
        tg = tres.to_graph()
        tq_in = {b: _qubit(tg, b, part_in[i]) for i, b in enumerate(tg.inputs())}      # boundaries keep their order
        tq_out = {b: _qubit(tg, b, part_out[i]) for i, b in enumerate(tg.outputs())}
        terms.append(Term(values, False, _pieces(tg, tq_in, tq_out, max_legs), tg, exact))
    return Part(part_in, part_out, hidden, terms, g)


def decompose_terms(g: BaseGraph | GPFResult, run: dict[str, int] | None = None, labelling: str = 'as reduced',
                    split_gadgets: bool = False, max_terms: int = 1024, max_legs: int = 8) -> Decomposition:
    """The terms of ``g`` (a diagram, or the result of :func:`reduce_gpf`) in one run (see the module docstring).

    ``run`` gives a value to every measurement result left in the reduced diagram
    (one of :func:`~zxqec.gpf.consistent_assignments`); a run that violates a
    detector is refused. ``labelling`` is ``'as reduced'`` or ``'fewest links'``.
    With ``split_gadgets``, the hubs of the phase gadgets are split too, which writes
    every gadget as a sum of Pauli strings. A part with more than ``max_terms`` terms
    is not split; a piece with more than ``max_legs`` inputs and outputs is not
    contracted (its matrix and name are left out).
    """
    if labelling not in LABELLINGS:
        raise ValueError("labelling must be one of {}".format(LABELLINGS))
    res = g if isinstance(g, GPFResult) else reduce_gpf(g)
    if res.is_zero:
        raise ValueError("the diagram is zero in every run")
    run = dict(run or {})
    for c in res.constraints:
        missing = {v.name for v in c.free_vars()} - set(run) if hasattr(c, 'free_vars') else set()
        if missing:
            raise ValueError("give a value to {} (a run)".format(sorted(missing)))
        if evaluate_phase(c, run):
            raise ValueError("the run violates the detector {} = 0".format(c))
    h = res.to_graph().substitute_variables(run) if run else res.to_graph()
    if any(hasattr(h.phase(v), 'free_vars') and h.phase(v).free_vars() for v in h.vertices()):
        raise ValueError("measurement results are left in the reduced diagram; give them values in run")
    qin = {b: _qubit(h, b, i) for i, b in enumerate(h.inputs())}
    qout = {b: _qubit(h, b, i) for i, b in enumerate(h.outputs())}
    parts = []
    for comp in sorted(_components(h), key=lambda c: _sorted([qin.get(b, qout.get(b)) for b in c
                                                              if b in qin or b in qout])[:1]):
        if not any(b in comp for b in qin) and not any(b in comp for b in qout):
            continue                                   # a scalar
        parts.append(_split_part(_restricted(h, comp), labelling, split_gadgets, max_terms, max_legs, qin, qout))
    return Decomposition(labelling, parts)
