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

"""What a block does to the encoded information (question 2), and whether it
always does it (question 3).

Given the code ``C_in`` the qubits are in before a block ``B`` and the code
``C_out`` after it, the *logical action* of ``B`` is

    D_out ; B ; E_in

where ``E_in`` is the ideal encoder of ``C_in`` and ``D_out`` the ideal decoder
of ``C_out`` (the adjoint of its ideal encoder), both built from the codes'
canonical logical operators (:meth:`StabilizerCode.encoder`). The composite is
reduced with :func:`~zxqec.gpf.reduce_gpf`; it acts on ``k_in`` logical qubits
and keeps the measurement outcomes as Boolean variables.

The reduced diagram depends on the outcomes only through a few Boolean
parities (its Pauli frame and gadget signs). :meth:`LogicalAction.branches`
picks one consistent outcome assignment for every combination of those
parities, so looking at those branches covers every run of the program.
:meth:`LogicalAction.identify` names the operation in each branch, up to a
Pauli correction, and :meth:`LogicalAction.verdict` answers question 3: the
protocol is *deterministic* if every branch implements the same operation up
to a Pauli correction that the outcomes determine.

The composite is reduced up to a non-zero scalar, so it cannot tell a logical
operation from one that also sends part of the code elsewhere: ``T`` on 7 of
the 15 qubits of the ``[[15,1,3]]`` code gives ``0.75 T^dag`` (up to a phase)
between the ideal encoders, and ``T^dag`` up to scalar, although a run stays
in the code with probability 9/16 only. :meth:`LogicalAction.stays_in_code`
therefore checks the condition under which the logical action is the
operation on the logical qubits: the block maps the code before it into the
code after it. A block that fails is reported as leaving the code, not as a
gate.

**Codes that are not CSS: through the isometry AP normal form.** Section 4.5
of the thesis reduces any stabilizer code to the CSS case: the ideal encoder
is ``E = U_O . C . U_I`` (:func:`~zxqec.codes.iap_normal_form`), a diagonal
Clifford ``U_O`` after a phase-free CSS core ``C`` after a logical Clifford
``U_I``. When a code is not CSS, the composite is built from these layers,

    U_I^out^dag ; C_out^dag ; U_O^out^dag ; B ; U_O^in ; C_in ; U_I^in ,

and when the block is diagonal (a layer of ``T`` gates, say) and both codes
have the same ``U_O``, the two ``U_O`` cancel, since diagonal operators
commute: the block is checked on the CSS core, exactly as for a CSS code, and
``U_I`` only relabels the logical qubits. :meth:`LogicalAction.on_core` gives
the action on the cores alone, before that relabelling. For CSS codes the
ideal encoder is already its core, and the composite is built from it as
before.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
from pyzx.graph.base import BaseGraph
from pyzx.graph.scalar import Scalar
from pyzx.utils import EdgeType

from ..gpf import GPFResult, boolean_variables, measurement_report, reduce_gpf
from ..gpf.reduce import sample_consistent_assignment
from ..spacetime.subdiagram import get_logical_qubit, set_logical_qubit
from .code import StabilizerCode

__all__ = ['glue', 'logical_action', 'LogicalAction', 'Branch', 'TooManyRunsError']


class TooManyRunsError(NotImplementedError):
    """The reduced diagram depends on too many parities of the outcomes to try every combination."""


# -- composing diagrams by wire label --------------------------------------------------

def glue(first: BaseGraph, second: BaseGraph) -> BaseGraph:
    """``first`` followed by ``second``, plugging outputs into inputs with the same wire label.

    Labels are the ``logical_qubit`` vertex data of the boundaries (the program's
    qubit index). Every output of ``first`` must meet an input of ``second``.
    """
    g = first.copy()
    fo = {get_logical_qubit(g, v): v for v in g.outputs()}
    si = {get_logical_qubit(second, v): v for v in second.inputs()}
    if set(fo) != set(si):
        raise ValueError("cannot glue: outputs {} do not match inputs {}".format(
            sorted(fo, key=str), sorted(si, key=str)))
    plugged = set(si.values())
    maxr = max((g.row(v) for v in g.vertices()), default=0)
    minr = min((second.row(v) for v in second.vertices()), default=0)
    offset = maxr - minr + 1
    g.scalar.mult_with_scalar(second.scalar)
    for name in second.var_registry.vars():
        g.var_registry.set_type(name, second.var_registry.get_type(name))
    vtab: dict[Any, Any] = {}
    for v in second.vertices():
        if v in plugged:
            continue
        w = g.add_vertex(second.type(v), phase=second.phase(v), qubit=second.qubit(v),
                         row=offset + second.row(v))
        g.set_vdata_dict(w, dict(second.vdata_dict(v)))
        vtab[v] = w
    for e in second.edges():
        s, t = second.edge_st(e)
        if s in plugged or t in plugged:
            continue
        g.add_edge((vtab[s], vtab[t]), second.edge_type(e))
    for k, i in si.items():
        o = fo[k]
        (no,) = list(g.neighbors(o))
        (ni,) = list(second.neighbors(i))
        et = (EdgeType.HADAMARD
              if g.edge_type(g.edge(no, o)) != second.edge_type(second.edge(i, ni))
              else EdgeType.SIMPLE)
        g.remove_vertex(o)
        g.add_edge((no, vtab[ni]), et)
    g.set_outputs(tuple(vtab[v] for v in second.outputs()))
    g.rebind_variables_to_registry()
    return g


def _labelled_encoder(code: StabilizerCode) -> BaseGraph:
    e = code.encoder()
    for b, q in zip(e.outputs(), code.qubits):
        set_logical_qubit(e, b, q)
    return e


def _labelled_decoder(code: StabilizerCode) -> BaseGraph:
    d = code.encoder().adjoint()
    for b, q in zip(d.inputs(), code.qubits):
        set_logical_qubit(d, b, q)
    return d


# -- the logical action --------------------------------------------------------------------

def logical_action(block: BaseGraph, code_in: StabilizerCode, code_out: StabilizerCode,
                   context: Sequence[Any] = (), via: str = 'auto') -> 'LogicalAction':
    """The logical action of ``block`` from ``code_in`` to ``code_out`` (see the module docstring).

    The block's input wires must be the qubits of ``code_in`` and its output
    wires those of ``code_out``, matched by label; a block whose wires carry no
    labels (a diagram built by hand) is matched by position. ``context`` holds detectors
    of the program around the block, for example those of the whole program.
    Those that mention none of the block's own outcomes restrict the classes
    of runs to what the program can produce: an outcome measured before the
    block that the program fixes, which may appear in a sign of ``code_in``,
    is read as its value. Those that mention an outcome of the block are left
    out, so that the block is still analysed as a map of its own outcomes (a
    check whose outcome the program predicts is still a measurement).

    ``via`` chooses the encoders: ``'encoder'`` the ideal encoders of
    :meth:`StabilizerCode.encoder`, ``'iapnf'`` their isometry AP normal forms
    (see the module docstring), ``'auto'`` (default) the normal forms when a code
    is not CSS. Both give the same map; the normal forms cancel ``U_O`` when the
    block is diagonal.
    """
    if via not in ('auto', 'encoder', 'iapnf'):
        raise ValueError("via must be 'auto', 'encoder' or 'iapnf', not {!r}".format(via))
    block = _labelled_by_position(block, code_in, code_out)
    own = set(boolean_variables(block))
    outside = [d for d in context if hasattr(d, 'free_vars') and not ({v.name for v in d.free_vars()} & own)]
    if via == 'encoder' or (via == 'auto' and code_in.is_css and code_out.is_css):
        composite = glue(glue(_labelled_encoder(code_in), block), _labelled_decoder(code_out))
        return LogicalAction(reduce_gpf(composite), code_in, code_out, composite, block=block, context=outside)
    from .iapnf import iap_normal_form
    nf_in, nf_out = iap_normal_form(code_in), iap_normal_form(code_out)
    cancel = _same_u_o(nf_in, nf_out) and _is_diagonal(block)
    composite = _through_normal_forms(block, nf_in, nf_out, u_i=True, u_o=not cancel)
    return LogicalAction(reduce_gpf(composite), code_in, code_out, composite, block=block, context=outside,
                         via='iapnf', normal_forms=(nf_in, nf_out), u_o_cancelled=cancel)


def _labelled_by_position(block: BaseGraph, code_in: StabilizerCode, code_out: StabilizerCode) -> BaseGraph:
    """``block`` itself if its wires are labelled; otherwise a copy labelled by position with the codes' qubits."""
    wires = list(block.inputs()) + list(block.outputs())
    if any(get_logical_qubit(block, b) is not None for b in wires):
        return block
    if len(block.inputs()) != code_in.n or len(block.outputs()) != code_out.n:
        raise ValueError("an unlabelled block needs as many inputs as the code before it has qubits ({}) "
                         "and as many outputs as the code after it ({})".format(code_in.n, code_out.n))
    block = block.copy()
    for b, q in zip(block.inputs(), code_in.qubits):
        set_logical_qubit(block, b, q)
    for b, q in zip(block.outputs(), code_out.qubits):
        set_logical_qubit(block, b, q)
    return block


def _through_normal_forms(block: BaseGraph, nf_in: Any, nf_out: Any, u_i: bool, u_o: bool) -> BaseGraph:
    """``block`` between the layers of two isometry AP normal forms (``U_I`` and ``U_O`` optional)."""
    enc = nf_in.to_graph(u_i=u_i, u_o=u_o, labelled=True)
    dec = nf_out.to_graph(u_i=u_i, u_o=u_o).adjoint()
    for b, q in zip(dec.inputs(), nf_out.qubits):
        set_logical_qubit(dec, b, q)
    return glue(glue(enc, block), dec)


def _same_u_o(nf_in: Any, nf_out: Any) -> bool:
    """Whether two normal forms have the same ``U_O`` on the same labelled qubits."""
    def u_o(nf: Any) -> tuple[Any, Any]:
        lab = nf.qubits
        return ({lab[q]: str(v) for q, v in nf.out_phases.items()},
                {frozenset((lab[a], lab[b])) for a, b in nf.cz_out})
    return set(nf_in.qubits) == set(nf_out.qubits) and u_o(nf_in) == u_o(nf_out)


def _is_diagonal(block: BaseGraph) -> bool:
    """A sufficient test that ``block`` is diagonal in the computational basis (so it commutes with ``U_O``).

    Its inputs and outputs are the same labelled qubits, and its reduced
    diagram is the identity skeleton with Z-phases: each input on a Z-spider
    joined by a plain edge to the output X-spider of the same qubit, which has
    no phase, and otherwise only Z-phase gadgets on those Z-spiders.
    """
    from ..gpf import is_leaf, is_stem
    from pyzx.utils import VertexType
    ins = [get_logical_qubit(block, b) for b in block.inputs()]
    outs = [get_logical_qubit(block, b) for b in block.outputs()]
    if None in ins or len(set(ins)) != len(ins) or set(ins) != set(outs):
        return False
    res = reduce_gpf(block)
    if res.is_zero:
        return False
    g = res.to_graph()
    zero = lambda p: str(p) == '0'
    z_of: dict[Any, Any] = {}
    for b, q in zip(g.inputs(), ins):
        (w,) = list(g.neighbors(b))
        if g.type(w) != VertexType.Z or g.edge_type(g.edge(b, w)) != EdgeType.SIMPLE:
            return False
        z_of[w] = q
    x_of: dict[Any, Any] = {}
    for b, q in zip(g.outputs(), outs):
        (w,) = list(g.neighbors(b))
        if (g.type(w) != VertexType.X or g.edge_type(g.edge(b, w)) != EdgeType.SIMPLE
                or not zero(g.phase(w))):
            return False
        x_of[w] = q
    bounds = set(g.inputs()) | set(g.outputs())
    for v in g.vertices():
        if v in bounds or v in z_of or v in x_of:
            continue
        if is_leaf(g, v):
            if g.type(v) != VertexType.Z:
                return False
        elif is_stem(g, v) and g.type(v) == VertexType.X:
            if any(u not in z_of and not is_leaf(g, u) for u in g.neighbors(v)):
                return False
        else:
            return False
    for w, q in x_of.items():
        inner = [u for u in g.neighbors(w) if u not in bounds]
        if len(inner) != 1 or z_of.get(inner[0]) != q or g.edge_type(g.edge(w, inner[0])) != EdgeType.SIMPLE:
            return False
    for z in z_of:
        for u in g.neighbors(z):
            if not (u in bounds or u in x_of or (is_stem(g, u) and g.type(u) == VertexType.X)):
                return False
            if g.edge_type(g.edge(z, u)) != EdgeType.SIMPLE:
                return False
    return True


@dataclass
class Branch:
    """One class of runs: an outcome assignment and what the block does in it."""
    outcomes: dict[str, int]
    operation: str | None          #: name of the operation up to a Pauli, or None if not recognised
    correction: str | None         #: the Pauli P with (block) = P . (operation); '' for none
    unitary: bool                  #: whether the branch is a unitary (up to scale)
    conditions: list[tuple[str, int]] = field(default_factory=list)  #: (parity, value) defining the class
    same_as_reference: bool = True  #: equal to the first class of runs up to a Pauli frame

    def when(self) -> str:
        """The condition on the outcomes that selects this class of runs."""
        if not self.conditions:
            return "in every run"
        return "when " + " and ".join("{} = {}".format(p, v) for p, v in self.conditions)


@dataclass
class LogicalAction:
    """The reduced logical action of a block; see :func:`logical_action`."""
    result: GPFResult
    code_in: StabilizerCode | None
    code_out: StabilizerCode | None
    composite: BaseGraph | None = field(default=None, repr=False)
    _branches: list[Branch] | None = field(default=None, repr=False)
    block: BaseGraph | None = field(default=None, repr=False)
    _stays: tuple[bool | None] | None = field(default=None, repr=False)
    context: list[Any] = field(default_factory=list, repr=False)  #: detectors on outcomes outside the block
    via: str = 'encoder'                       #: 'encoder' or 'iapnf': how the composite was built
    normal_forms: tuple[Any, Any] | None = field(default=None, repr=False)   #: (in, out), with 'iapnf'
    u_o_cancelled: bool | None = None          #: with 'iapnf': the two U_O cancelled (diagonal block)
    _core: Any = field(default=None, repr=False)

    def on_core(self) -> 'LogicalAction':
        """The block's action on the CSS cores alone: the composite without the ``U_I`` layers.

        Only with ``via='iapnf'``. The logical action is this one with ``U_I^in``
        before it and ``U_I^out^dagger`` after it; when ``U_O`` cancelled, it is
        the check of the block on a CSS code, as in the thesis (Section 4.5).
        """
        if self.normal_forms is None or self.block is None:
            raise ValueError("on_core needs a logical action computed via the isometry AP normal form")
        if self._core is None:
            nf_in, nf_out = self.normal_forms
            g = _through_normal_forms(self.block, nf_in, nf_out, u_i=False, u_o=not self.u_o_cancelled)
            self._core = LogicalAction(reduce_gpf(g), None, None, g, context=list(self.context))
        return self._core

    def stays_in_code(self) -> bool | None:
        """Whether the block maps the code before it into the code after it (``None``: not checked).

        Only then is the logical action the operation the block performs on the
        logical qubits (the proposition on the logical action in the paper). The
        encoder of the code before, followed by the block, is bent into a state
        (inputs turned into outputs: its Choi state), and that state's
        stabilizers common to all runs are read with :func:`~zxqec.codes.code_of`
        (a conditional logical gate changes the others between runs). For a Pauli ``s``
        and a map ``D``, ``s D`` is proportional to ``D`` exactly when ``s D = +-D``,
        so the block stays in the code exactly when every stabilizer of the code
        after it, with the identity on the bent inputs, is among them up to sign.
        Runs in which such a stabilizer has the other sign put the image
        orthogonal to the code; there the logical action is the zero map, which
        :meth:`branches` already shows. ``None`` if the codes or the block are not
        known, or the Choi state's code cannot be read.
        """
        if self._stays is None:
            self._stays = (self._check_stays(),)
        return self._stays[0]

    def _check_stays(self) -> bool | None:
        from .code import NotACodeError, code_of
        from .pauli import PauliString
        if self.block is None or self.code_in is None or self.code_out is None:
            return None
        first = glue(_labelled_encoder(self.code_in), self.block)
        outs = list(first.outputs())
        where = {get_logical_qubit(first, b): j for j, b in enumerate(outs)}
        choi = first.copy()
        choi.set_outputs(tuple(choi.outputs()) + tuple(choi.inputs()))
        choi.set_inputs(())
        try:
            image = code_of(choi, common_to_all_runs=True)
        except (NotACodeError, NotImplementedError):
            return None
        n = len(choi.outputs())
        for s in self.code_out.stabilizers:
            x, z = [0] * n, [0] * n
            for q, a, b in zip(self.code_out.qubits, s.x, s.z):
                x[where[q]], z[where[q]] = a, b
            if not image.contains(PauliString.from_arrays(x, z)):
                return False
        return True

    @property
    def graph(self) -> BaseGraph:
        return self.result.to_graph()

    @property
    def k_in(self) -> int:
        return len(self.result.graph.inputs())

    @property
    def k_out(self) -> int:
        return len(self.result.graph.outputs())

    @property
    def detectors(self) -> list[Any]:
        return self.result.detectors

    def report(self) -> Any:
        """The measurement report of the logical action (frame, gadget signs, fates)."""
        return measurement_report(self.composite if self.composite is not None else self.graph,
                                  result=self.result)

    def matrix(self, outcomes: dict[str, int]) -> np.ndarray:
        """The ``2^k_out x 2^k_in`` matrix of one branch, scaled to unit largest singular value."""
        g = self.graph
        names = boolean_variables(g)
        a = {n: outcomes.get(n, 0) for n in names}
        h = g.substitute_variables(a) if names else g.copy()
        h.scalar = Scalar()               # scaled below anyway; a symbolic scalar phase cannot be evaluated
        m = np.asarray(h.to_matrix(preserve_scalar=False)).reshape(2 ** self.k_out, 2 ** self.k_in)
        s = np.linalg.norm(m, 2)
        return m / s if s > 1e-12 else m

    def branches(self, limit: int = 10) -> list[Branch]:
        """One consistent run for every combination of the diagram's Boolean phases.

        The reduced diagram depends on the outcomes only through the parities
        written on its spiders, so these runs show everything the block can do.
        Runs that violate a detector, of the block or of its context, are left
        out. ``limit`` caps the number of distinct parities (2^limit runs); a
        parity written in another order, or plus 1, is the same parity.
        """
        if self._branches is not None:
            return self._branches
        g = self.result.graph
        constraints = list(self.detectors) + list(self.context)
        fixed = _fixed_outcomes(constraints)
        polys: dict[Any, Any] = {}
        for v in g.vertices():
            p = g.phase(v)
            if hasattr(p, 'free_vars') and p.free_vars():
                p = _substitute(p, fixed)          # outcomes the detectors pin down read as constants
                if hasattr(p, 'free_vars') and p.free_vars():
                    polys.setdefault(_parity_key(p), p)
        names = sorted(set(boolean_variables(g)) | {str(v) for c in constraints for v in c.free_vars()})
        plist = list(polys.values())
        if len(plist) > limit:
            raise TooManyRunsError(
                "the reduced diagram depends on {} distinct parities of the outcomes; trying all 2^{} "
                "combinations is above the limit of 2^{}".format(len(plist), len(plist), limit))
        runs = []
        for values in itertools.product((0, 1), repeat=len(plist)):
            targets = [p + 1 if val else p for p, val in zip(plist, values)]
            try:
                a = sample_consistent_assignment(constraints + targets, names, _rng(values))
            except ValueError:
                continue
            runs.append((values, a, self.matrix(a)))
        out = []
        if runs:
            ref = runs[0][2]
            ref_op, ref_out, ref_in = self._name(ref)
            effect_inputs = [i for ins, outs, _w in self.parts() if not outs for i in ins]
            for values, a, m in runs:
                rel = _relating_pauli(m, ref, self.k_in, self.k_out, effect_inputs)
                if rel is not None:
                    op, pout, pin = ref_op, _times(rel[0], ref_out), _times(rel[1], ref_in)
                else:
                    op, pout, pin = self._name(m)
                whole = op is not None and ' (x) ' not in op
                out.append(Branch(a, op, self._render(pout, pin, whole), _is_unitary(m, self.k_in, self.k_out),
                                  [(str(p), v) for p, v in zip(plist, values)], rel is not None))
        self._branches = out
        return out

    def _render(self, pout: str, pin: str, whole: bool = False) -> str:
        """A Pauli correction as text: ``'X'`` for one part, ``'X on qubits 7-13'`` per part otherwise.

        ``whole``: the operation was named as one map, so the correction is on its outputs as a whole.
        """
        if whole and set(pin or 'I') <= {'I'}:
            return '' if set(pout or 'I') <= {'I'} else pout
        parts = self.parts()
        texts = []
        for ins, outs, where in parts:
            if outs:
                letters = "".join(pout[i] for i in outs) if pout else ""
                if letters and set(letters) != {'I'}:
                    texts.append(letters if len(parts) == 1 else "{} on {}".format(letters, where))
            else:
                letters = "".join(pin[i] for i in ins) if pin else ""
                if letters and set(letters) != {'I'}:
                    texts.append("outcome flipped by " + (letters if len(parts) == 1 else "{} on {}".format(letters, where)))
        return ", ".join(texts)

    def parts(self) -> list[tuple[list[int], list[int], str]]:
        """How the logical qubits split over the code blocks: ``(inputs, outputs, qubits)`` per part.

        Logical qubits are numbered in the order of the codes' logical operators.
        A part pairs the logical qubits of a code block before the block with those
        of the code block on the same physical qubits after it; a part with no
        inputs is prepared by the block, one with no outputs is measured by it.
        """
        def groups(code: StabilizerCode | None) -> list[tuple[frozenset, int]]:
            if code is None:
                return []
            blocks = [frozenset(b.qubits) for b in code.blocks()]
            out = []
            for i, (x, z) in enumerate(zip(code.logical_x, code.logical_z)):
                sup = {code.qubits[j] for j in set(x.support) | set(z.support)}
                blk = next((b for b in blocks if sup <= b), frozenset(sup))
                out.append((blk, i))
            return out
        gin, gout = groups(self.code_in), groups(self.code_out)
        # disjoint unions of overlapping blocks; a block that overlaps several groups joins them
        # all (transitively), in the place of the first
        keys: list[frozenset | None] = []
        for blk, _ in gin + gout:
            merged, first = blk, None
            changed = True
            while changed:
                changed = False
                for i, k in enumerate(keys):
                    if k is not None and k & merged:
                        merged, keys[i] = merged | k, None
                        first = i if first is None else min(first, i)
                        changed = True
            if first is None:
                keys.append(merged)
            else:
                keys[first] = merged
        joined: list[frozenset] = [k for k in keys if k is not None]
        parts = []
        for k in joined:
            ins = [i for blk, i in gin if blk & k]
            outs = [i for blk, i in gout if blk & k]
            parts.append((ins, outs, _qubit_range(k)))
        return parts

    def _name(self, m: np.ndarray) -> tuple[str | None, str, str]:
        """Name of the operation and the smallest Paulis ``(on outputs, on inputs)`` that go with it."""
        op, pout, pin = _identify(m, self.k_in, self.k_out)
        if op is not None:
            return op, pout, pin
        parts = self.parts()
        if len(parts) < 2:
            return None, '', ''
        factors = _factorise(m, self.k_in, self.k_out, [(i, o) for i, o, _ in parts])
        if factors is None:
            return None, '', ''
        names = []
        full_out, full_in = ['I'] * self.k_out, ['I'] * self.k_in
        for (ins, outs, where), f in zip(parts, factors):
            fo, fout, fin = _identify(f, len(ins), len(outs))
            if fo is None:
                return None, '', ''
            names.append("{} on {}".format(fo, where))
            for i, c in zip(outs, fout):
                full_out[i] = c
            for i, c in zip(ins, fin):
                full_in[i] = c
        return " (x) ".join(names), "".join(full_out), "".join(full_in)

    def operations(self) -> set[str | None]:
        return {b.operation for b in self.branches()}

    def is_deterministic(self) -> bool:
        """Every run does the same operation, up to a Pauli correction fixed by the outcomes.

        Decided on the matrices, not on names: every class of runs equals the first
        one up to a Pauli on the outputs (or, for a measured qubit, before it).
        A block that leaves the code (:meth:`stays_in_code`) is not deterministic:
        it performs no logical operation at all.
        """
        if self.stays_in_code() is False:
            return False
        br = self.branches()
        return bool(br) and all(b.same_as_reference for b in br)

    def verdict(self) -> str:
        """A one-paragraph answer to question 3."""
        br = self.branches()
        if self.result.is_zero or not br:
            return "never succeeds: every run is the zero map (the detectors contradict each other)"
        ops = sorted({str(b.operation) if b.operation else 'a map outside the list of named operations'
                      for b in br})
        if self.stays_in_code() is False:
            return ("leaves the code: the block maps part of the code before it outside the code after it, "
                    "so it is not a logical operation (between the ideal encoders it looks like {}, "
                    "but shrunk)".format(" / ".join(ops)))
        if self.is_deterministic():
            corr = sorted({b.correction or 'none' for b in br})
            if corr == ['none']:
                return "deterministic: every run implements {}".format(ops[0])
            return ("deterministic up to a Pauli frame: every run implements {} followed by a Pauli "
                    "correction ({}) that the outcomes determine".format(ops[0], ", ".join(corr)))
        basis = self.measured_basis()
        if basis is not None:
            return ("a logical measurement in the basis {{{}}}: {}".format(
                ", ".join(basis), "; ".join("{} {}".format((b.operation or '')[len('projects onto '):], b.when())
                                            for b in br)))
        parts = ["{} {}".format(b.operation or 'an unrecognised map', b.when()) for b in br]
        return "not deterministic: the operation depends on the outcomes ({})".format("; ".join(parts))

    def measured_basis(self) -> tuple[str, str] | None:
        """If the classes of runs are the two projections onto a named basis, that basis.

        This is a logical measurement that keeps the qubit, of an observable that
        need not be a Pauli (the double-check of magic-state cultivation measures
        one whose eigenstates are ``T^dag|+>`` and ``T^dag|->``).
        """
        br = self.branches()
        names = {b.operation for b in br}
        if len(br) != 2 or any(b.correction for b in br) or None in names:
            return None
        for pair in _BASES:
            if names == {'projects onto ' + pair[0], 'projects onto ' + pair[1]}:
                return pair
        return None

    def describe(self) -> str:
        lines = ["logical action: {} -> {} logical qubit(s)".format(self.k_in, self.k_out)]
        if self.via == 'iapnf' and self.normal_forms is not None:
            nf_in, nf_out = self.normal_forms
            ui = lambda nf: 'empty' if not nf.hadamards else 'phases {}, CZ {}, H on {}, X-CZ {}'.format(
                {q: str(v) for q, v in sorted(nf.in_phases.items())}, list(nf.cz_in), list(nf.hadamards),
                list(nf.xcz_in))
            lines.append("  through the isometry AP normal forms: {}; U_I before: {}; U_I after: {}".format(
                "U_O cancelled (the block is diagonal), so the block is checked on the CSS core"
                if self.u_o_cancelled else "U_O kept", ui(nf_in), ui(nf_out)))
        for b in self.branches():
            lines.append("  {}: {}{}".format(
                b.when(), b.operation or 'not recognised',
                " with correction {}".format(b.correction) if b.correction else ''))
        lines.append("  " + self.verdict())
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


# -- naming small operations -------------------------------------------------------------------

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]])
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1, -1])
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_S = np.diag([1, 1j])
_T = np.diag([1, np.exp(1j * np.pi / 4)])
_PAULI = {'I': _I, 'X': _X, 'Y': _Y, 'Z': _Z}

_ONE = {'I': _I, 'H': _H, 'S': _S, 'S^dag': _S.conj(), 'T': _T, 'T^dag': _T.conj(),
        'HS': _H @ _S, 'SH': _S @ _H, 'HSH': _H @ _S @ _H}
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
_TWO = {'CNOT(0->1)': _CNOT, 'CNOT(1->0)': np.kron(_H, _H) @ _CNOT @ np.kron(_H, _H),
        'CZ': np.diag([1, 1, 1, -1]), 'SWAP': np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])}
_STATES = {'|0>': np.array([1, 0]), '|+>': np.array([1, 1]) / np.sqrt(2),
           '|+i>': np.array([1, 1j]) / np.sqrt(2),
           'T|+>': np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2),
           'T^dag|+>': np.array([1, np.exp(-1j * np.pi / 4)]) / np.sqrt(2)}


def _proportional(a: np.ndarray, b: np.ndarray) -> bool:
    k = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    if abs(b[k]) < 1e-9 or abs(a[k]) < 1e-9:
        return False
    return np.allclose(a, a[k] / b[k] * b, atol=1e-7)


def _paulis(k: int) -> dict[str, np.ndarray]:
    """All Pauli strings on ``k`` qubits, lightest first."""
    out = {}
    for letters in sorted(itertools.product('IXYZ', repeat=k), key=lambda t: (sum(c != 'I' for c in t), t)):
        m = np.array([[1]])
        for c in letters:
            m = np.kron(m, _PAULI[c])
        out[''.join(letters)] = m
    return out


#: states with their Z-flipped partners, for naming projections: each pair is a basis
_BASES = [('|0>', '|1>'), ('|+>', '|->'), ('|+i>', '|-i>'), ('T|+>', 'T|->'), ('T^dag|+>', 'T^dag|->')]


def _basis_states() -> dict[str, np.ndarray]:
    out = dict(_STATES)
    out['|1>'] = np.array([0, 1])
    for name in ('|+>', '|+i>', 'T|+>', 'T^dag|+>'):
        out[name.replace('+', '-')] = _Z @ _STATES[name]
    return out


def _library(k_in: int, k_out: int) -> dict[str, np.ndarray]:
    if k_in == k_out == 1:
        lib = dict(_ONE)
        for name, v in _basis_states().items():       # a logical measurement that keeps the qubit
            lib['projects onto ' + name] = np.outer(v, v.conj())
        return lib
    if k_in == k_out == 2:
        lib = dict(_TWO)
        for (a, ma), (b, mb) in itertools.product(_ONE.items(), repeat=2):
            lib['{} (x) {}'.format(a, b) if (a, b) != ('I', 'I') else 'I'] = np.kron(ma, mb)
        return lib
    if k_in == 0 and k_out == 1:
        return {'prepares ' + n: v.reshape(2, 1) for n, v in _STATES.items()}
    if k_in == 1 and k_out == 0:
        return {'Z measurement': np.array([[1, 0]]), 'X measurement': np.array([[1, 1]]) / np.sqrt(2),
                'Y measurement': np.array([[1, -1j]]) / np.sqrt(2)}
    return {}


def _identify(m: np.ndarray, k_in: int, k_out: int) -> tuple[str | None, str, str]:
    """``(name, Pauli on outputs, Pauli on inputs)`` with ``m`` = Pauli . named . Pauli, smallest Pauli first."""
    if k_in == k_out == 0:
        return ('a non-zero scalar' if abs(m).max() > 1e-9 else 'zero'), '', ''
    lib = _library(k_in, k_out)
    # the smallest correction wins: X.T|+> is T^dag|+>, and is named so, not "T|+> with correction X"
    if k_out == 0:                             # an effect: the Pauli acts before it
        for pn, p in _paulis(k_in).items():
            for name, g in lib.items():
                if _proportional(m, g @ p):
                    return name, '', pn
        return None, '', ''
    for pn, p in _paulis(k_out).items():
        for name, g in lib.items():
            if _proportional(m, p @ g):
                return name, pn, 'I' * k_in
    return None, '', ''


def _relating_pauli(m: np.ndarray, ref: np.ndarray, k_in: int, k_out: int,
                    effect_inputs: list[int]) -> tuple[str, str] | None:
    """Paulis ``(on outputs, on the inputs of measured parts)`` with ``m`` proportional to ``Q . ref . Q'``."""
    outs = _paulis(k_out) if k_out else {'': np.array([[1]])}
    ins: dict[str, np.ndarray] = {}
    for letters in itertools.product('IXYZ', repeat=len(effect_inputs)):
        full = ['I'] * k_in
        for i, c in zip(effect_inputs, letters):
            full[i] = c
        key = ''.join(full)
        mat = np.array([[1]])
        for c in full:
            mat = np.kron(mat, _PAULI[c])
        ins[key] = mat
    identity_in = 'I' * k_in
    if identity_in not in ins:
        ins[identity_in] = np.eye(2 ** k_in)
    # a frame on the outputs first; a flipped outcome of a measured part only if there is none
    for qo, po in outs.items():
        if _proportional(m, po @ ref):
            return qo, identity_in
    for qo, po in outs.items():
        for qi, pi in ins.items():
            if _proportional(m, po @ ref @ pi):
                return qo, qi
    return None


def _times(a: str, b: str) -> str:
    """The product of two Pauli strings, up to phase (either may be empty for the identity)."""
    if not a:
        return b
    if not b:
        return a
    bits = {'I': (0, 0), 'X': (1, 0), 'Z': (0, 1), 'Y': (1, 1)}
    name = {v: k for k, v in bits.items()}
    return ''.join(name[(bits[x][0] ^ bits[y][0], bits[x][1] ^ bits[y][1])] for x, y in zip(a, b))


def _factorise(m: np.ndarray, k_in: int, k_out: int,
               parts: list[tuple[list[int], list[int]]]) -> list[np.ndarray] | None:
    """Split ``m`` into one factor per part (a tensor product), or ``None`` if it does not split."""
    t = m.reshape([2] * k_out + [2] * k_in)
    axes_left = list(range(k_out + k_in))           # axes of t, in order: outputs, then inputs
    factors = []
    for n_done, (ins, outs) in enumerate(parts):
        mine = [o for o in outs] + [k_out + i for i in ins]
        pos = [axes_left.index(a) for a in mine]
        rest = [j for j in range(len(axes_left)) if j not in pos]
        a = np.transpose(t, pos + rest).reshape(2 ** len(pos), -1)
        u, sv, vh = np.linalg.svd(a, full_matrices=False)
        if len(sv) > 1 and sv[1] > 1e-8 * sv[0]:
            return None
        f = (u[:, 0] * sv[0]).reshape([2] * len(pos))
        f = f.reshape(2 ** len(outs), 2 ** len(ins))
        factors.append(f / max(np.linalg.norm(f, 2), 1e-300))
        axes_left = [axes_left[j] for j in rest]
        t = vh[0].reshape([2] * len(rest)) if rest else vh[0]
    return factors


def _qubit_range(qs: frozenset) -> str:
    q = sorted(qs, key=lambda x: (str(type(x)), x))
    if all(isinstance(x, int) for x in q) and q and q == list(range(q[0], q[-1] + 1)) and len(q) > 2:
        return "qubits {}-{}".format(q[0], q[-1])
    return "qubit{} {}".format("s" if len(q) > 1 else "", ",".join(map(str, q)))


def _is_unitary(m: np.ndarray, k_in: int, k_out: int) -> bool:
    if k_in != k_out:
        return False
    return np.allclose(m @ m.conj().T, np.eye(m.shape[0]) * (m @ m.conj().T)[0, 0], atol=1e-7)


def _vector(g: BaseGraph, k_in: int, k_out: int) -> np.ndarray:
    t = g.to_tensor(preserve_scalar=False)
    return np.asarray(t).reshape(2 ** k_out, 2 ** k_in)


def _fixed_outcomes(detectors: list[Any]) -> dict[str, int]:
    """Outcomes whose value the detectors force (a detector that mentions only them)."""
    from ..gpf import constraint_system
    names, rows = constraint_system(detectors)
    out = {}
    for row in rows:
        vs = [names[i] for i, b in enumerate(row[:-1]) if b]
        if len(vs) == 1:
            out[vs[0]] = row[-1]
    return out


def _parity_key(p: Any) -> Any:
    """The outcomes of a parity ``p`` (its constant ignored), or its text if ``p`` is not a parity."""
    names: set[str] = set()
    for coeff, term in p.terms:
        if not term.vars:
            continue
        if len(term.vars) != 1 or term.vars[0][1] != 1 or Fraction(coeff).denominator != 1:
            return str(p)
        if int(coeff) % 2:
            names ^= {term.vars[0][0].name}
    return frozenset(names)


def _substitute(p: Any, values: dict[str, int]) -> Any:
    names = {v.name for v in p.free_vars()} & set(values)
    if not names:
        return p
    subs = {v: Fraction(values[v.name]) for v in p.free_vars() if v.name in names}
    return p.substitute(subs)


class _Zeros:
    """Stands in for a random generator: free outcomes are set to 0, giving the simplest run."""
    @staticmethod
    def randint(a: int, b: int) -> int:
        return 0


def _rng(values: tuple[int, ...]) -> Any:
    return _Zeros()


def _short(a: dict[str, int]) -> str:
    ones = sorted(k for k, v in a.items() if v)
    return "{" + ", ".join(ones) + "} = 1" if ones else "all outcomes 0"
