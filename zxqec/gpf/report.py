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

"""What happens to every measurement of a diagram or a block of a program.

:func:`measurement_report` reduces a diagram with :func:`~zxqec.gpf.reduce_gpf`
and turns the result into a table: for each measurement outcome, where it was
measured and which of four things happens to it in the error-free setting.

* ``frame``  -- the reduced diagram still depends on it: it is part of the
  outcome-dependent Pauli correction, so a decoder has to act on it;
* ``fixed``  -- the detectors force its value (the operator's eigenvalue was
  already fixed by the state preparation);
* ``tied``   -- the detectors equate it to other outcomes (the operator was
  measured before, or its value follows from other measurements); its own
  value is still random;
* ``gauge``  -- no detector mentions it and the map does not depend on it: a
  random bit no surviving qubit can reflect.

The report also carries the detectors themselves, the Pauli frame, and the
dimensions of the three subspaces of the outcome space
(``forced + steering + gauge = number of outcomes``). Given the outcomes of an
actual (error-free or noisy) run, :meth:`MeasurementReport.violated` returns
the detectors that fired and :meth:`MeasurementReport.frame_values` the Pauli
correction to apply -- the same information a decoder gets from a detector
error model, but derived diagrammatically and without assuming the circuit is
Clifford: non-Pauli phases are kept as gadgets and simply never appear in a
detector.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pyzx.graph.base import BaseGraph, ET, VT
from pyzx.symbolic import Poly
from pyzx.utils import VertexType

from .normal_form import is_leaf
from .reduce import GPFResult, boolean_variables, evaluate_phase, reduce_gpf

__all__ = ['Measurement', 'MeasurementReport', 'measurement_report']

_VAR = re.compile(r'[A-Za-z_][A-Za-z_0-9]*(?:\[\d+\])?')

FRAME, FIXED, TIED, GAUGE = 'frame', 'fixed', 'tied', 'gauge'


def _vars_of(p: Any) -> list[str]:
    return sorted({str(v) for v in p.free_vars()}) if isinstance(p, Poly) else []


@dataclass
class Measurement:
    """One measurement outcome and what happens to it (see the module docstring)."""
    name: str                                  #: the outcome variable, e.g. ``'c[5]'``
    kind: str                                  #: ``frame`` / ``fixed`` / ``tied`` / ``gauge``
    stage: int | None = None                   #: stage of the program that measured it
    qubit: int | None = None                   #: logical qubit it was measured on
    position: int | None = None                #: physical wire it sat on (they differ after a permutation)
    timestep: int | None = None                #: when, in the program's schedule
    value: int | None = None                   #: for ``fixed``: the value it always takes
    relations: list[Any] = field(default_factory=list)   #: detectors mentioning it

    def explain(self, report: 'MeasurementReport | None' = None) -> str:
        """Why this outcome has the fate it has, in terms of the protocol's own measurements."""
        def other(name: str) -> str:
            m = report.measurements.get(name) if report is not None else None
            if m is None or m.stage is None:
                return name
            return "{} (stage {}, qubit {}, t={})".format(name, m.stage, m.qubit, m.timestep)

        if self.kind == FIXED:
            return "always {}: nothing in the protocol can change it".format(self.value)
        if self.kind == TIED:
            rel = min(self.relations, key=lambda r: len(str(r)))
            partners = [v for v in _vars_of(rel) if v != self.name]
            return "its parity with {} is deterministic ({} = 0)".format(
                ", ".join(other(v) for v in partners), rel)
        if self.kind == FRAME:
            if report is None:
                return "the reduced diagram depends on it"
            parts = []
            for kind, ph in report.frame:
                if self.name in _vars_of(ph):
                    rest = [v for v in _vars_of(ph) if v != self.name]
                    parts.append("{} correction{}".format(kind, ", with " + ", ".join(rest) if rest else ""))
            for _kind, leaf, ph in report.gadget_signs:
                if self.name in _vars_of(ph):
                    parts.append("negates the phase-{} gadget (a conditional rotation, not a Pauli)".format(leaf))
            return "the map depends on it: " + "; ".join(parts)
        return "no detector mentions it and the map does not depend on it"

    def __str__(self) -> str:
        at = "" if self.stage is None else " (stage {}, qubit {})".format(self.stage, self.qubit)
        if self.kind == FIXED:
            what = "always {}".format(self.value)
        elif self.kind == TIED:
            what = "tied by {} = 0".format(min((str(r) for r in self.relations), key=len))
        elif self.kind == FRAME:
            what = "in the Pauli frame"
        else:
            what = "gauge: random, affects nothing"
        return "{}{}: {}".format(self.name, at, what)


@dataclass
class MeasurementReport:
    """The measurements of a diagram, its detectors and its Pauli frame."""
    measurements: dict[str, Measurement]
    detectors: list[Any]                       #: Boolean polynomials that are 0 in every run
    frame: list[tuple[str, Any]]               #: ``('Z'|'X', phase)``: the Pauli frame
    result: GPFResult                          #: the reduction this came from
    gadget_signs: list[tuple[str, Any, Any]] = field(default_factory=list)
    #: ``(colour, leaf phase, Boolean phase)`` for each gadget hub carrying outcome variables:
    #: the outcomes do not apply a Pauli there, they flip the sign of the gadget's phase
    dimensions: dict[str, int] = field(default_factory=dict)
    scope: str = 'the whole diagram'           #: what these measurements are part of

    # -- querying ----------------------------------------------------------

    def restrict(self, names: Iterable[str], scope: str = 'a block') -> 'MeasurementReport':
        """The same analysis, listing only these outcomes (see :meth:`QProgram.measurement_report`).

        The classification, the detectors and the frame stay those of the full
        diagram -- that is the point: a block's measurements are deterministic
        only relative to what the rest of the program prepared. Detectors that
        mention none of ``names`` are dropped.
        """
        keep = [n for n in self.measurements if n in set(names)]
        dets = [d for d in self.detectors if any(n in _vars_of(d) for n in keep)]
        return MeasurementReport(measurements={n: self.measurements[n] for n in keep},
                                 detectors=dets, frame=list(self.frame),
                                 gadget_signs=list(self.gadget_signs), result=self.result,
                                 dimensions=dict(self.dimensions), scope=scope)

    def internal_detectors(self) -> list[Any]:
        """Detectors all of whose outcomes are in this report (checkable without context)."""
        return [d for d in self.detectors if set(_vars_of(d)) <= set(self.measurements)]

    def of_kind(self, kind: str) -> list[Measurement]:
        """All measurements of one kind, in measurement order."""
        return [m for m in self.measurements.values() if m.kind == kind]

    def violated(self, outcomes: Mapping[str, int]) -> list[Any]:
        """The detectors that fire for these outcomes (unset outcomes count as 0)."""
        return [d for d in self.detectors if evaluate_phase(d, dict(outcomes))]

    def is_possible(self, outcomes: Mapping[str, int]) -> bool:
        """Whether this outcome combination can occur in the error-free setting."""
        return not self.result.is_zero and not self.violated(outcomes)

    def frame_values(self, outcomes: Mapping[str, int]) -> list[tuple[str, int]]:
        """The Pauli correction for these outcomes: ``[('Z', 0|1), ('X', 0|1), ...]``.

        These are the phases of the reduced diagram read in the **input frame**,
        as the diagram is drawn: a ``('Z', 1)`` entry is a Z on the spider next
        to the input, not a correction to apply to an output. A decoder acting
        after the map has to push them through it -- for a diagram ending in a
        Hadamard, as in the fold-transversal case study, Z and X swap roles.
        Gadget signs are not included; see :meth:`gadget_sign_values`.
        """
        return [(kind, evaluate_phase(phase, dict(outcomes))) for kind, phase in self.frame]

    def gadget_sign_values(self, outcomes: Mapping[str, int]) -> list[tuple[str, Any, int]]:
        """For each gadget hub carrying outcomes: ``(colour, leaf phase, 0|1)``.

        A value of 1 means the gadget's phase is negated on this branch, i.e. the
        protocol applied the *adjoint* of that rotation -- for a ``pi/4`` leaf,
        ``T`` becomes ``T^dagger``, which a conditional ``S`` (not a Pauli)
        corrects.
        """
        return [(kind, leaf, evaluate_phase(phase, dict(outcomes)))
                for kind, leaf, phase in self.gadget_signs]

    # -- display -----------------------------------------------------------

    def table(self, why: bool = True) -> str:
        """The measurements as a fixed-width table, one line each.

        ``why=False`` drops the explanation column and keeps only the fate.
        """
        rows = [("outcome", "stage", "qubit", "pos", "t", "fate", "why")]
        for m in self.measurements.values():
            rows.append((m.name, _str(m.stage), _str(m.qubit), _str(m.position), _str(m.timestep),
                         m.kind, m.explain(self)))
        last = 6 if why else 5                       # the column printed without padding
        widths = [max(len(r[i]) for r in rows) for i in range(last)]
        return "\n".join(("  ".join(r[i].ljust(widths[i]) for i in range(last)) + "  " + r[last]).rstrip()
                         for r in rows)

    def summary(self) -> str:
        d = self.dimensions
        frame = ", ".join("{}({})".format(k, p) for k, p in self.frame) or "none"
        head = "{} measurements in {}; Pauli frame: {}".format(len(self.measurements), self.scope, frame)
        if self.gadget_signs:
            head += "\n  gadget signs (not Pauli corrections): " + ", ".join(
                "phase {} negated when {} = 1".format(leaf, phase) for _k, leaf, phase in self.gadget_signs)
        counts = "; ".join("{} {}".format(len(self.of_kind(k)), k) for k in (FRAME, FIXED, TIED, GAUGE)
                           if self.of_kind(k))
        dims = ("{} detectors ({} of them internal); outcome space of the diagram: "
                "{} forced + {} steering the map + {} gauge = {}"
                .format(len(self.detectors), len(self.internal_detectors()), d.get('forced', 0),
                        d.get('steering', 0), d.get('gauge', 0), d.get('outcomes', 0)))
        return head + "\n" + counts + "\n" + dims

    def __str__(self) -> str:
        return self.summary() + "\n" + self.table()


def _str(x: Any) -> str:
    return "" if x is None else str(x)


# -- GF(2) helpers -------------------------------------------------------------

def _row(names: Sequence[str], vs: Iterable[str]) -> list[int]:
    idx = {n: i for i, n in enumerate(names)}
    r = [0] * len(names)
    for v in vs:
        if v in idx:
            r[idx[v]] ^= 1
    return r


def _rank(rows: Sequence[Sequence[int]], width: int) -> int:
    m = [list(r) for r in rows]
    rank = 0
    for col in range(width):
        piv = next((i for i in range(rank, len(m)) if m[i][col]), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        for i in range(len(m)):
            if i != rank and m[i][col]:
                m[i] = [a ^ b for a, b in zip(m[i], m[rank])]
        rank += 1
    return rank


# -- building the report -------------------------------------------------------

def _measured_at(g: BaseGraph[VT, ET]) -> dict[str, tuple[Any, ...]]:
    """``{outcome name: (stage, logical qubit, physical wire, timestep)}`` from the leaves' vertex data."""
    from pyzx.spacetime import get_timestep

    from ..spacetime import get_logical_qubit, get_stage
    out: dict[str, tuple[Any, ...]] = {}
    for v in g.vertices():
        name = g.vdata(v, 'result_symbol', None)
        if name is None:
            continue
        hosts = [w for w in g.neighbors(v) if g.type(w) != VertexType.BOUNDARY] or [v]
        host = hosts[0]
        wire = g.qubit(host)
        out[str(name)] = (get_stage(g, host), get_logical_qubit(g, host),
                          int(wire) if wire is not None else None, get_timestep(g, host))
    return out


def measurement_report(g: BaseGraph[VT, ET], result: GPFResult | None = None,
                       **kwargs: Any) -> MeasurementReport:
    """Reduce ``g`` and report what happens to each of its measurement outcomes.

    ``result`` reuses a reduction computed earlier; any other keyword arguments
    are passed to :func:`~zxqec.gpf.reduce_gpf`.
    """
    res = result if result is not None else reduce_gpf(g, **kwargs)
    names = boolean_variables(g)
    where = _measured_at(g)

    detectors = list(res.detectors)
    det_rows = [_row(names, _vars_of(d)) for d in detectors]
    forced = _rank(det_rows, len(names))

    frame: list[tuple[str, Any]] = []
    gadget_signs: list[tuple[str, Any, Any]] = []
    for v in sorted(res.graph.vertices(), key=lambda w: (res.graph.row(w), w)):
        # in row order, so the frame reads from the input to the output, which is the
        # order in which the corrections are applied
        if res.graph.type(v) == VertexType.BOUNDARY:
            continue
        phase = res.graph.phase(v)
        if not _vars_of(phase):
            continue
        colour = 'Z' if res.graph.type(v) == VertexType.Z else 'X'
        leaves = [w for w in res.graph.neighbors(v) if is_leaf(res.graph, w)]
        if leaves:
            # a hub: the outcomes flip the sign of the gadget's phase, which is a
            # conditional rotation (for a pi/4 leaf, T versus T-dagger), not a Pauli
            gadget_signs.append((colour, res.graph.phase(leaves[0]), phase))
        else:
            frame.append((colour, phase))
    frame_vars = {n for _, p in frame for n in _vars_of(p)} | {n for _k, _l, p in gadget_signs for n in _vars_of(p)}
    steering_rows = [_row(names, _vars_of(p)) for _, p in frame]
    steering_rows += [_row(names, _vars_of(p)) for _k, _l, p in gadget_signs]
    steering = _rank(det_rows + steering_rows, len(names)) - forced

    def order(name: str) -> tuple[Any, ...]:
        stage, _q, _pos, tick = where.get(name, (None, None, None, None))
        digits = re.findall(r'\d+', name)
        return (tick if tick is not None else 1 << 30, stage if stage is not None else 1 << 30,
                int(digits[-1]) if digits else 0, name)

    measurements: dict[str, Measurement] = {}
    for name in sorted(names, key=order):
        stage, qubit, pos, tick = where.get(name, (None, None, None, None))
        relations = [d for d in detectors if name in _vars_of(d)]
        if name in frame_vars:
            kind, value = FRAME, None
        elif _rank(det_rows + [_row(names, [name])], len(names)) == forced:
            # the singleton is implied by the detectors: the value is forced
            kind, value = FIXED, _forced_value(name, detectors, names)
        elif relations:
            kind, value = TIED, None
        else:
            kind, value = GAUGE, None
        measurements[name] = Measurement(name=name, kind=kind, stage=stage, qubit=qubit,
                                         position=pos, timestep=tick, value=value, relations=relations)

    dims = {'outcomes': len(names), 'forced': forced, 'steering': steering,
            'gauge': len(names) - forced - steering}
    return MeasurementReport(measurements=measurements, detectors=detectors, frame=frame,
                             gadget_signs=gadget_signs, result=res, dimensions=dims)


def _forced_value(name: str, detectors: Sequence[Any], names: Sequence[str]) -> int:
    """The value a fixed outcome always takes: solve the detectors for it over GF(2)."""
    rows = [_row(names, _vars_of(d)) + [_constant(d)] for d in detectors]
    width = len(names)
    target = names.index(name)
    # eliminate every other column, leaving rows of the form  x_target = b
    m = [list(r) for r in rows]
    rank = 0
    order = [c for c in range(width) if c != target] + [target]
    pivot_of: dict[int, int] = {}
    for col in order:
        piv = next((i for i in range(rank, len(m)) if m[i][col]), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        for i in range(len(m)):
            if i != rank and m[i][col]:
                m[i] = [a ^ b for a, b in zip(m[i], m[rank])]
        pivot_of[col] = rank
        rank += 1
    r = pivot_of.get(target)
    if r is None or any(m[r][c] for c in range(width) if c != target):
        return 0
    return m[r][width]


def _constant(p: Any) -> int:
    """The constant term of a Boolean polynomial (1 if the detector reads ``... = 1``)."""
    if isinstance(p, Poly):
        for coeff, term in p.terms:
            if not term.vars:
                return int(coeff) % 2
        return 0
    return int(p) % 2
