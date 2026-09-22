"""Follow one Boolean outcome variable through the rewrites of PyZX's ``full_reduce`` and of ``zxqec.gpf``.

    from track_variable import tracing, where
    with tracing('c[5]') as t:
        zx.full_reduce(h)           # or reduce_gpf(g)
    print(t)                        # every rewrite that moved c[5], with where it was before and after
    where(h, 'c[5]')                # {'spiders': {id: (type, phase, degree)}, 'scalar': [terms of g.scalar]}

The tracer wraps the rewrite appliers of PyZX (fusion, identity removal, pivot, local complementation,
gadget rules, copy, Hopf), the Boolean bialgebra call and the row reduction of ``zxqec.gpf``, and
``remove_isolated_vertices``. After every single application it compares where the variable sits --
on which spiders (type, phase, degree) and in which factors of ``g.scalar`` -- and records an event
whenever that changed. Nothing in PyZX or zxqec is modified on disk; the wrappers are installed for
the duration of the ``with`` block only. Used by ``TrackingOutcomeVariables.ipynb``.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from pyzx import simplify as S
from pyzx.graph.graph_s import GraphS
from pyzx.symbolic import Poly
from pyzx.utils import VertexType
import pyzx.rewrite_rules.pivot_rule as PR
import zxqec.gpf.normal_form as NF
import zxqec.gpf.reduce as RD
import zxqec.gpf.matrix as MX


def _vars(p: Any) -> set[str]:
    return {str(v) for v in p.free_vars()} if isinstance(p, Poly) else set()


def where(g, name: str) -> dict[str, Any]:
    """Where variable ``name`` sits: on spiders (id -> (type, phase, degree)) and in the scalar."""
    spiders = {v: (VertexType(g.type(v)).name, str(g.phase(v)), g.vertex_degree(v))
               for v in g.vertices() if name in _vars(g.phase(v))}
    scalar = [str(p) for p in g.scalar.phasenodes if name in _vars(p)]
    if name in _vars(g.scalar.phase):
        scalar.append("phase " + str(g.scalar.phase))
    return {"spiders": spiders, "scalar": scalar}


def describe(state: dict) -> str:
    s = ", ".join(f"spider {v}: {t}({p}), degree {d}" for v, (t, p, d) in sorted(state["spiders"].items()))
    if state["scalar"]:
        s += (";  " if s else "") + "in g.scalar: " + ", ".join(f"1 + e^(i pi ({p}))" for p in state["scalar"])
    return s or "nowhere"


@dataclass
class Event:
    rule: str
    args: tuple
    before: dict
    after: dict | None = None
    depth: int = 0                          # > 0: happened inside the rewrite logged just before it

    def __str__(self) -> str:
        pad = "      " + "    " * self.depth
        head = ("within it: " if self.depth else "") + f"{self.rule}{self.args}"
        return f"{head}\n{pad}before: {describe(self.before)}\n{pad}after : {describe(self.after or {})}"


@dataclass
class Trace:
    name: str
    events: list[Event] = field(default_factory=list)
    rewrites: int = 0                       # rewrites applied in total (with or without the variable)
    depth: int = 0

    def __str__(self) -> str:
        out = [f"{sum(1 for e in self.events if e.depth == 0)} of {self.rewrites} rewrites touched {self.name}:"]
        k = 0
        for e in self.events:
            if e.depth == 0:
                k += 1
                out.append(f"  {k}. {e}")
            else:
                out.append("      " + "    " * (e.depth - 1) + str(e))
        return "\n".join(out)


@contextlib.contextmanager
def tracing(name: str):
    """Install the wrappers; yields the Trace that fills up while PyZX / zxqec rewrite graphs."""
    trace = Trace(name)

    def wrap(label, func, argfmt=None):
        def wrapped(g, *args, **kw):
            shown = argfmt(args) if argfmt else tuple(a for a in args if not isinstance(a, (list, set, dict)))
            event = Event(label, shown, where(g, name), depth=trace.depth)
            trace.events.append(event)          # appended on entry, so nested rewrites follow their parent
            trace.depth += 1
            try:
                result = func(g, *args, **kw)
            finally:
                trace.depth -= 1
            event.after = where(g, name)
            trace.rewrites += 1
            if event.before == event.after:
                trace.events.remove(event)
            return result
        return wrapped

    saved = []
    # PyZX's automatic rewrites: wrap the per-match applier
    for label, obj in [("fuse", S.fuse_simp), ("remove_self_loop", S.remove_self_loop_simp), ("remove_id", S.id_simp),
                       ("pivot", S.pivot_simp), ("pivot_gadget", S.pivot_gadget_simp), ("pivot_boundary", S.pivot_boundary_simp),
                       ("lcomp", S.lcomp_simp), ("copy", S.copy_simp), ("hopf", S.hopf_simp)]:
        saved.append((obj, "applier", obj.applier)); obj.applier = wrap(label, obj.applier)
    for label, obj in [("gadget_merge", S.gadget_simp), ("supplementarity", S.supplementarity_simp)]:
        saved.append((obj, "simp_applier", obj.simp_applier)); obj.simp_applier = wrap(label, obj.simp_applier)
    # pivot_gadget_simp / pivot_boundary_simp apply all their matches in one batch through this function
    saved.append((PR, "pivot_NOT_REWORKED", PR.pivot_NOT_REWORKED))
    PR.pivot_NOT_REWORKED = wrap("pivot_batch", PR.pivot_NOT_REWORKED, lambda args: tuple((m[0], m[1]) for m in args[0]))
    # PyZX's scalar bookkeeping for isolated spiders
    saved.append((GraphS, "remove_isolated_vertices", GraphS.remove_isolated_vertices))
    GraphS.remove_isolated_vertices = wrap("remove_isolated_vertices", GraphS.remove_isolated_vertices)
    # zxqec's calls
    saved.append((NF, "bialgebra", NF.bialgebra)); NF.bialgebra = wrap("boolean_bialgebra", NF.bialgebra)
    saved.append((RD, "apply_matrix", RD.apply_matrix)); RD.apply_matrix = wrap("row_reduction", RD.apply_matrix)
    saved.append((RD, "_remove_isolated_pairs", RD._remove_isolated_pairs)); RD._remove_isolated_pairs = wrap("remove_isolated_pairs", RD._remove_isolated_pairs)
    try:
        yield trace
    finally:
        for obj, attr, orig in saved:
            setattr(obj, attr, orig)
