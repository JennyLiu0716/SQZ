"""The reduction of a program drawn step by step (used by Cultivation.ipynb, Part 4).

* :func:`draw_layers` draws a diagram in generalized parity form as the thesis
  did: the Z-layer, the X-layer with the outputs, and the phase gadgets.
* :func:`encoder_form` changes the summation variables of a reduced state so
  that its Z-layer rows are a code's logical operator and generators: the
  picture then reads "a logical state, fed into the encoder of the code".
* :func:`encoded_state` builds that expected diagram directly from a code.
* :func:`allowed_runs` lists every assignment of the outcomes that the
  detectors allow, so that each picture can be checked against the circuit.

Kept out of the notebook so that its cells show the steps, not matplotlib.
"""
from fractions import Fraction
import itertools

import matplotlib.pyplot as plt
import numpy as np
from pyzx.graph import Graph
from pyzx.symbolic import new_var
from pyzx.utils import EdgeType, VertexType

from zxqec.codes.gf2 import as_matrix, nullspace, rank, solve
from zxqec.gpf import constraint_system, has_boundary_hadamard, is_leaf

Z_COLOUR, X_COLOUR, H_COLOUR = '#ccffcc', '#ff8888', '#ffff00'


# -- the runs to check against ---------------------------------------------------------------------

def allowed_runs(constraints, names):
    """Every assignment of ``names`` (0/1) that satisfies the constraints (linear ones).

    The constraints are solved over GF(2); the free variables range over all
    values and the others follow.
    """
    cnames, rows = constraint_system(constraints)
    rows = [r for r in rows if any(r[:-1])]
    pivots = [r.index(1) for r in rows]
    free = [n for n in names if n not in cnames or cnames.index(n) not in pivots]
    out = []
    for values in itertools.product((0, 1), repeat=len(free)):
        a = dict(zip(free, values))
        for r, piv in zip(rows, pivots):
            a[cnames[piv]] = (r[-1] + sum(r[j] * a.get(cnames[j], 0) for j in range(len(cnames)) if j != piv)) % 2
        out.append({n: a.get(n, 0) for n in names})
    return out


# -- phases as short labels -------------------------------------------------------------------------

def _parts(phase):
    """A phase as (constant in units of pi, sorted outcome names), for a constant plus a parity."""
    if not hasattr(phase, 'terms'):
        return Fraction(phase) % 2, []
    const, names = Fraction(0), []
    for coeff, term in phase.terms:
        if not term.vars:
            const += Fraction(coeff)
        elif int(coeff) % 2:
            names.append(term.vars[0][0].name)
    return const % 2, sorted(set(n for n in names if names.count(n) % 2), key=_outcome_key)


def _outcome_key(name):
    digits = ''.join(ch for ch in name if ch.isdigit())
    return (name.rstrip('0123456789[]'), int(digits) if digits else 0)


def reduce_modulo(phase, constraints):
    """``(const, names)`` of ``phase`` rewritten modulo the (linear) constraints.

    The constraints are brought to reduced echelon form with the *latest*
    outcomes as pivots, and those are eliminated from the phase, so what is left
    is written in the earliest outcomes it can be. Two phases that agree in
    every allowed run come out equal, so two equal pictures carry equal labels.
    """
    const, names = _parts(phase)
    cnames, rows = constraint_system(constraints)
    order = sorted(cnames, key=_outcome_key, reverse=True)
    m = [[r[cnames.index(nm)] for nm in order] + [r[-1]] for r in rows if any(r[:-1])]
    pivots = []
    rank_ = 0
    for col in range(len(order)):
        piv = next((i for i in range(rank_, len(m)) if m[i][col]), None)
        if piv is None:
            continue
        m[rank_], m[piv] = m[piv], m[rank_]
        for i in range(len(m)):
            if i != rank_ and m[i][col]:
                m[i] = [a ^ b for a, b in zip(m[i], m[rank_])]
        pivots.append(col)
        rank_ += 1
    v = [1 if nm in names else 0 for nm in order]
    k = 0
    for r, col in zip(m, pivots):
        if v[col]:
            v = [a ^ b for a, b in zip(v, r[:-1])]
            k ^= r[-1]
    extra = [nm for nm in names if nm not in cnames]
    return (const + k) % 2, sorted([nm for nm, b in zip(order, v) if b] + extra, key=_outcome_key)


def phase_label(phase, constraints=None):
    const, names = reduce_modulo(phase, constraints) if constraints is not None else _parts(phase)
    parts = []
    if const:
        num, den = const.numerator, const.denominator
        parts.append(('π' if num == 1 else '{}π'.format(num)) + ('' if den == 1 else '/{}'.format(den)))
    if names:
        parts.append('π(' + '+'.join(n.replace('[', '').replace(']', '') for n in names) + ')')
    return ' + '.join(parts)


# -- drawing -------------------------------------------------------------------------------------------

def draw_layers(g, ax=None, title='', labels=True, constraints=None, qubit_names=None, width=5.0,
                znames=None):
    """Draw a diagram in generalized parity form: Z-layer, X-layer with outputs, gadgets.

    Z-spiders are ordered by the mean height of their X-neighbours, which keeps
    edges short, and named ``z1, z2, ...`` from the top; ``znames`` (a dict from
    Z-spider to name) fixes both the names and the order instead. ``labels``
    writes the phases (outcome parities reduced modulo ``constraints`` if
    given); non-Pauli leaf phases are always written. Returns ``(ax, names)``
    with ``names`` mapping each Z-spider to its name.
    """
    outs = list(g.outputs())
    out_x = {}
    for j, b in enumerate(outs):
        (w,) = list(g.neighbors(b))
        out_x[w] = j
    leaves = [v for v in g.vertices() if is_leaf(g, v)]
    hubs = sorted({next(iter(g.neighbors(v))) for v in leaves})
    xs_inner = sorted(v for v in g.vertices() if g.type(v) == VertexType.X and v not in out_x
                      and v not in hubs and not is_leaf(g, v))
    zs = [v for v in g.vertices() if g.type(v) == VertexType.Z and not is_leaf(g, v)]
    n_rows = len(outs) + len(xs_inner) + len(hubs)
    y = {}
    for w, j in out_x.items():
        y[w] = -j
    for i, v in enumerate(xs_inner + hubs):
        y[v] = -(len(outs) + 0.5 + i)
    def bary(v):
        ys = [y[w] for w in g.neighbors(v) if w in y]
        return sum(ys) / len(ys) if ys else 0.0
    if znames is not None:
        zs = [v for v in znames if v in zs]
    else:
        zs.sort(key=bary, reverse=True)
    top, bottom = 0.0, -(n_rows - 1)
    for i, v in enumerate(zs):
        y[v] = top - i * (top - bottom) / max(len(zs) - 1, 1) if len(zs) > 1 else bary(v)
    znames = dict(znames) if znames is not None else {v: 'z{}'.format(i + 1) for i, v in enumerate(zs)}
    x = {}
    for v in zs:
        x[v] = 0.0
    for v in list(out_x) + xs_inner + hubs:
        x[v] = 2.0
    for j, b in enumerate(outs):
        x[b], y[b] = 3.0, -j
    for v in leaves:
        (hub,) = list(g.neighbors(v))
        x[v], y[v] = 2.7, y[hub]
    if ax is None:
        _fig, ax = plt.subplots(figsize=(width, 0.32 * n_rows + 1.0))
    for e in g.edges():
        s, t = g.edge_st(e)
        if s in x and t in x:
            ax.plot([x[s], x[t]], [y[s], y[t]], color='#555555', lw=0.6, zorder=1,
                    ls='--' if g.edge_type(e) == EdgeType.HADAMARD else '-')
    for v in x:
        ty = g.type(v)
        if ty == VertexType.BOUNDARY:
            ax.plot(x[v], y[v], 'o', color='black', ms=2.5, zorder=3)
            if qubit_names is not None and v in outs:
                ax.text(x[v] + 0.1, y[v], str(qubit_names[outs.index(v)]), fontsize=7, va='center')
            if has_boundary_hadamard(g, v):
                ax.add_patch(plt.Rectangle((x[v] - 0.55, y[v] - 0.12), 0.24, 0.24, facecolor=H_COLOUR,
                                           edgecolor='black', lw=0.6, zorder=3))
            continue
        colour = Z_COLOUR if ty == VertexType.Z else X_COLOUR
        ax.add_patch(plt.Circle((x[v], y[v]), 0.13, facecolor=colour, edgecolor='black', lw=0.7, zorder=3))
        if v in znames:
            ax.text(x[v], y[v] - 0.3, znames[v], fontsize=6, ha='center', va='center', color='#333333')
        text = phase_label(g.phase(v), constraints)
        if text and (labels or is_leaf(g, v)):
            dx, ha = (0.2, 'left') if is_leaf(g, v) else ((-0.2, 'right') if ty == VertexType.Z else (0.0, 'center'))
            dy = 0.0 if ha != 'center' else 0.22
            ax.text(x[v] + dx, y[v] + dy, text, fontsize=7 if len(text) < 20 else 6, ha=ha, va='center',
                    color='#1f3fbf', zorder=4)
    ax.set_xlim(-2.4 if labels else -0.6, 3.6)
    ax.set_ylim(min(y.values()) - 0.6, 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9)
    return ax, znames


def layer_rows(g, constraints=None):
    """The Z-layer of a reduced state as sorted ``(phase label, outputs it is joined to)`` pairs.

    Two diagrams with the same result have the same spiders, edges and labels
    (the output X-spiders are listed separately by :func:`output_phases`).
    """
    outs = list(g.outputs())
    ox = [next(iter(g.neighbors(b))) for b in outs]
    zs = [v for v in g.vertices() if g.type(v) == VertexType.Z and not is_leaf(g, v)]
    return sorted((phase_label(g.phase(z), constraints), tuple(j for j, w in enumerate(ox) if g.connected(z, w)))
                  for z in zs)


def output_phases(g, constraints=None):
    """The phase labels of the output X-spiders, in output order."""
    return [phase_label(g.phase(next(iter(g.neighbors(b)))), constraints) for b in g.outputs()]

# -- the last two steps: the shape of an encoder, and the diagram we expect ----------------------------

def _layers(g):
    """(Z-spiders, output X-spider per output, gadgets as (hub, leaves)) of a reduced state."""
    outs = list(g.outputs())
    out_x = [next(iter(g.neighbors(b))) for b in outs]
    leaves = [v for v in g.vertices() if is_leaf(g, v)]
    gadgets = {}
    for v in leaves:
        gadgets.setdefault(next(iter(g.neighbors(v))), []).append(v)
    zs = sorted(v for v in g.vertices() if g.type(v) == VertexType.Z and not is_leaf(g, v))
    return zs, out_x, gadgets


def encoder_form(g, rows, fuse=True):
    """``g`` with new summation variables, so that its Z-layer rows are ``rows``.

    ``g`` is a state (no inputs) in generalized parity form whose Z-spiders each
    carry a Pauli phase and connect only to the output X-spiders and to gadget
    hubs; ``rows`` are 0/1 vectors over the outputs spanning the same space as
    the Z-layer rows. With ``A`` the old rows and ``T A = rows``, the old bits are
    ``y = y' T``: Pauli phases become ``T s``, a gadget that read the parity
    ``l . y`` reads ``(T l) . y'``, and the output X-spiders do not change. The
    map is the same (only the order of summation changes).

    If the Z-layer has more spiders than ``rows`` (its rows are dependent), the
    remaining new variables are *hidden*: their rows are zero on the outputs
    (a basis of the combinations ``c`` with ``c A = 0``), and they come last.
    With ``fuse``, a gadget left reading one spider, with a constant hub phase
    ``k``, is fused into that spider as the phase ``(-1)^k a``.
    """
    zs, out_x, gadgets = _layers(g)
    n = len(out_x)
    col = {w: j for j, w in enumerate(out_x)}
    old = as_matrix([[1 if g.connected(z, w) else 0 for w in out_x] for z in zs], n)
    new = as_matrix([list(r) for r in rows], n)
    t = []
    for r in new:
        c = solve(old.T.copy(), r)
        if c is None:
            raise ValueError("row {} is not in the span of the Z-layer".format(list(r)))
        t.append(c)
    hidden = nullspace(old.T.copy()) if old.size else np.zeros((0, len(zs)), np.uint8)
    t = as_matrix(t + [list(c) for c in hidden], len(zs))
    new = as_matrix([list(r) for r in new] + [[0] * n for _ in hidden], n)
    if t.shape[0] != len(zs):
        raise ValueError("{} rows and {} hidden variables for a Z-layer of {} spiders".format(
            len(rows), len(hidden), len(zs)))
    if rank(t) < len(zs):
        raise ValueError("the rows are not a basis of the Z-layer's span")
    h = Graph()
    reg = h.var_registry
    def copy_phase(ph):
        const, names = _parts(ph)
        out = const
        for nm in names:
            out = out + new_var(nm, True, reg)
        return out
    xs = []
    for j, w in enumerate(out_x):
        xv = h.add_vertex(VertexType.X, j, 2, copy_phase(g.phase(w)))
        b = h.add_vertex(VertexType.BOUNDARY, j, 3)
        h.add_edge((xv, b))
        old_b = list(g.outputs())[j]
        if has_boundary_hadamard(g, old_b):
            h.set_vdata(b, 'hadamard', True)
        xs.append(xv)
    new_z = []
    for i, r in enumerate(new):
        phase = 0
        for k, z in enumerate(zs):
            if t[i][k]:
                phase = phase + copy_phase(g.phase(z))
        zv = h.add_vertex(VertexType.Z, i, 1, phase)
        for j in np.flatnonzero(r):
            h.add_edge((zv, xs[j]))
        new_z.append(zv)
    for hub, leaves in gadgets.items():
        reads = [1 if g.connected(hub, z) else 0 for z in zs]
        new_reads = [int(sum(t[i][k] * reads[k] for k in range(len(zs))) % 2) for i in range(len(new_z))]
        (leaf,) = leaves                           # a phase gadget has one leaf
        leaf_const, leaf_names = _parts(g.phase(leaf))
        hub_const, hub_names = _parts(g.phase(hub))
        targets = [new_z[i] for i, b in enumerate(new_reads) if b]
        if fuse and len(targets) == 1 and not hub_names:
            z = targets[0]                        # (a + c)(y xor k) = (-1)^k (a + c) y + constant
            phase = h.phase(z) + ((leaf_const if not hub_const else -leaf_const) % 2)
            for nm in leaf_names:
                phase = phase + new_var(nm, True, reg)
            h.set_phase(z, phase)
            continue
        hv = h.add_vertex(VertexType.X, len(new_z), 2, copy_phase(g.phase(hub)))
        lv = h.add_vertex(VertexType.Z, len(new_z), 2.7, copy_phase(g.phase(leaf)))
        h.set_vdata(lv, 'gpf', 'leaf')
        h.add_edge((hv, lv))
        for z in targets:
            h.add_edge((hv, z))
    h.set_inputs(())
    h.set_outputs(tuple(v for v in h.vertices() if h.type(v) == VertexType.BOUNDARY))
    return h


def encoded_state(code, angle, correction=()):
    """The ideal encoder of ``code`` applied to ``Z^p Z(angle)|+>``, ``p`` the parity of ``correction``.

    ``code.encoder()`` has one Z-spider on its input (one logical qubit); the
    state ``Z(a)|+>`` is a one-legged Z-spider with phase ``a``, and a logical
    ``Z^p`` adds ``pi p``; both fuse into the input spider.
    """
    enc = code.encoder()
    (b,) = list(enc.inputs())
    (z,) = list(enc.neighbors(b))
    enc.remove_vertex(b)
    enc.set_inputs(())
    phase = Fraction(angle)
    for nm in correction:
        phase = phase + new_var(nm, True, enc.var_registry)
    enc.set_phase(z, enc.phase(z) + phase)
    return enc
