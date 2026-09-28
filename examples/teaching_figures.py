"""Pictures for the walkthrough notebook (TInjectionWalkthrough.ipynb).

Kept out of the notebook so that its cells show the analysis, not matplotlib.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

# The Steane code drawn as a triangle: seven qubits, three four-qubit faces.
POS = {0: (0.0, 1.0), 1: (-0.87, -0.5), 2: (0.87, -0.5),
       3: (-0.43, 0.25), 5: (0.43, 0.25), 6: (0.0, -0.5), 4: (0.0, 0.0)}
FACES = [(0, 3, 4, 5), (1, 3, 4, 6), (2, 5, 4, 6)]
COLORS = ['#4C72B0', '#DD8452', '#55A868']
RED = '#C44E52'


def draw_code(ax, highlight=(), title="", label_faces=True, letters=None, offset=0):
    """The triangle; ``highlight`` qubits are filled, ``letters`` maps a qubit to a Pauli letter."""
    for face, colour in zip(FACES, COLORS):
        ax.add_patch(Polygon([POS[q] for q in face], closed=True, facecolor=colour,
                             edgecolor=colour, alpha=0.18, lw=1.4))
    if label_faces:
        for face, colour, name in zip(FACES, COLORS, ("face 1", "face 2", "face 3")):
            cx = sum(POS[q][0] for q in face) / 4
            cy = sum(POS[q][1] for q in face) / 4
            ax.text(cx, cy + 0.12, name, color=colour, fontsize=8, ha='center')
    for q, (x, y) in POS.items():
        on = q in highlight
        ax.add_patch(plt.Circle((x, y), 0.12, facecolor=RED if on else 'white',
                                edgecolor='black', zorder=3, lw=1.1))
        label = letters.get(q, str(q + offset)) if letters else str(q + offset)
        ax.text(x, y, label, ha='center', va='center', zorder=4, fontsize=8,
                color='white' if on else 'black', fontweight='bold' if letters else 'normal')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.8, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9)


def steane_code_figure():
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    draw_code(axes[0], title="seven qubits, three faces")
    draw_code(axes[1], highlight=(0, 3, 4, 5), label_faces=False,
              title="a stabilizer: X on face 1\n(or Z on face 1)")
    draw_code(axes[2], highlight=(0, 1, 2, 3, 4, 5, 6), label_faces=False,
              title="the logical X: X on all seven\n(also X on 3, 5, 6)")
    fig.tight_layout()
    return fig


def paulis_figure(paulis, qubits, title, offset=0, per_row=6):
    """One small triangle per Pauli string, with its letters written on the qubits."""
    n = len(paulis)
    rows = (n + per_row - 1) // per_row
    fig, axes = plt.subplots(rows, min(n, per_row), figsize=(1.55 * min(n, per_row), 1.75 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, p in zip(axes.flat, paulis):
        letters = {q - offset: c for q, c in zip(qubits, p.letters) if c != 'I'}
        draw_code(ax, highlight=tuple(letters), label_faces=False, letters=letters, offset=offset,
                  title=str(p))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def t_gadget_figure():
    fig, ax = plt.subplots(figsize=(6.8, 1.9))
    ax.plot([0.4, 6.4], [1.0, 1.0], color='black', lw=1.1)
    ax.plot([0.4, 3.4], [0.2, 0.2], color='black', lw=1.1)
    ax.text(0.25, 1.0, r"data $|\psi\rangle$", ha='right', va='center', fontsize=11)
    ax.text(0.25, 0.2, r"magic $T|+\rangle$", ha='right', va='center', fontsize=11)
    ax.plot([2.2, 2.2], [0.2, 1.0], color='black', lw=1.1)
    ax.add_patch(plt.Circle((2.2, 1.0), 0.06, color='black', zorder=3))
    ax.add_patch(plt.Circle((2.2, 0.2), 0.15, facecolor='white', edgecolor='black', zorder=3))
    ax.plot([2.05, 2.35], [0.2, 0.2], color='black', zorder=4)
    ax.plot([2.2, 2.2], [0.05, 0.35], color='black', zorder=4)
    ax.add_patch(plt.Rectangle((3.4, 0.02), 0.46, 0.36, facecolor='white', edgecolor='black', zorder=3))
    ax.text(3.63, 0.2, "$Z$", ha='center', va='center', fontsize=9, zorder=4)
    ax.plot([3.86, 4.75], [0.2, 0.2], color='grey', lw=0.9, ls=':')
    ax.plot([4.75, 4.75], [0.2, 0.85], color='grey', lw=0.9, ls=':')
    ax.text(4.3, -0.02, "outcome $m$", fontsize=8, color='grey', ha='center')
    ax.add_patch(plt.Rectangle((4.52, 0.85), 0.46, 0.3, facecolor='white', edgecolor='black', zorder=3))
    ax.text(4.75, 1.0, "$S$", ha='center', va='center', fontsize=9, zorder=4)
    ax.text(5.35, 1.22, "only if $m = 1$", fontsize=8, color='grey', ha='center')
    ax.text(6.55, 1.0, r"$T|\psi\rangle$", ha='left', va='center', fontsize=11)
    ax.set_xlim(-1.4, 7.4)
    ax.set_ylim(-0.2, 1.45)
    ax.axis('off')
    return fig


def protocol_figure():
    """The seven stages on the three groups of qubits."""
    fig, ax = plt.subplots(figsize=(9.0, 2.4))
    rows = {"data block\nqubits 0-6": 2, "magic block\nqubits 7-13": 1, "helper qubits\n14-19": 0}
    for label, y in rows.items():
        ax.plot([0, 10.4], [y, y], color='lightgrey', lw=9, solid_capstyle='butt', zorder=0)
        ax.text(-0.15, y, label, ha='right', va='center', fontsize=8)
    boxes = [("0 encode\nthe data", 0.2, 1.4, [2], '#4C72B0'),
             ("1 prepare\n$T|+\\rangle$", 1.9, 1.0, [1], '#DD8452'),
             ("2 encode\nthe magic", 3.1, 1.2, [1], '#DD8452'),
             ("3 check the\nmagic block", 4.6, 1.3, [1, 0], '#55A868'),
             ("4 transversal\nCNOT", 6.2, 1.3, [2, 1], '#8172B3'),
             ("5 read out\nthe magic", 7.7, 1.2, [1], RED),
             ("6 correct\n(if odd)", 9.1, 1.1, [2], '#937860')]
    for name, x, w, ys, colour in boxes:
        lo, hi = min(ys) - 0.3, max(ys) + 0.3
        ax.add_patch(plt.Rectangle((x, lo), w, hi - lo, facecolor=colour, alpha=0.3,
                                   edgecolor=colour, lw=1.3, zorder=2))
        ax.text(x + w / 2, (lo + hi) / 2, name, ha='center', va='center', fontsize=7.5, zorder=3)
    ax.annotate("", xy=(9.5, 1.68), xytext=(8.6, 1.32),
                arrowprops=dict(arrowstyle="->", color='grey', ls=':'))
    ax.text(7.75, 1.52, "parity of the\n7 outcomes", fontsize=7, color='grey', ha='center')
    ax.set_xlim(-2.2, 10.6)
    ax.set_ylim(-0.6, 2.6)
    ax.axis('off')
    return fig


def sandwich_figure():
    """Question 2: the block between the ideal encoder and decoder of its codes."""
    fig, ax = plt.subplots(figsize=(8.6, 2.1))
    boxes = [(0.6, "ideal encoder\nof the code before", '#4C72B0'),
             (3.4, "the block\n(physical gates and\nmeasurements)", '#8172B3'),
             (6.2, "ideal decoder\nof the code after", '#4C72B0')]
    for x, text, colour in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.1), 1.9, 1.3, boxstyle="round,pad=0.05",
                                    facecolor=colour, alpha=0.2, edgecolor=colour, lw=1.3))
        ax.text(x + 0.95, 0.75, text, ha='center', va='center', fontsize=8.5)
    for x0, x1, n, lab in [(-0.3, 0.6, 1, "k logical\nqubits"), (2.5, 3.4, 4, "n physical"),
                           (5.3, 6.2, 4, "n physical"), (8.1, 9.0, 1, "k logical\nqubits")]:
        for i in range(n):
            y = 0.75 + (i - (n - 1) / 2) * 0.22
            ax.plot([x0, x1], [y, y], color='black', lw=1)
        ax.text((x0 + x1) / 2, -0.3, lab, ha='center', fontsize=7.5, color='grey')
    ax.text(4.35, 1.65, "reduce the whole thing:  what is left is what the block does to the encoded qubits",
            ha='center', fontsize=8.5)
    ax.set_xlim(-0.6, 9.3)
    ax.set_ylim(-0.6, 1.9)
    ax.axis('off')
    return fig


def levels_figure(program, levels, names=None):
    """Stages at the bottom, then the blocks of each level in ``levels`` above them."""
    rows = [('stages', [((s,), str(program.stage(s))) for s in range(program.num_stages)])]
    for lv in levels:
        rows.append(('level {}'.format(lv) if not names else names[levels.index(lv)],
                     [(b.stages, (b.name or str(b)) + ("\n[" + b.role.replace(' ', '\n', 1) + "]"
                                                        if b.role and len(b.stages) == 1 and len(b.role) > 12
                                                        else "\n[" + b.role + "]" if b.role else ""))
                      for b in program.levels.blocks(lv)]))
    n = program.num_stages
    fig, ax = plt.subplots(figsize=(1.5 * n + 1.5, 1.15 * len(rows) + 0.4))
    palette = ['#e6e6e6', '#cfe0f3', '#f7dcc4', '#d6ecd2']
    for r, (label, blocks) in enumerate(rows):
        y = r * 1.0
        ax.text(-0.25, y + 0.35, label, ha='right', va='center', fontsize=8.5)
        for stages, text in blocks:
            x0, x1 = min(stages), max(stages) + 1
            ax.add_patch(FancyBboxPatch((x0 + 0.05, y), x1 - x0 - 0.1, 0.8, boxstyle="round,pad=0.02",
                                        facecolor=palette[r % len(palette)], edgecolor='grey', lw=0.8))
            ax.text((x0 + x1) / 2, y + 0.4, text, ha='center', va='center', fontsize=7)
    ax.set_xlim(-1.6, n + 0.1)
    ax.set_ylim(-0.2, len(rows) - 0.1)
    ax.axis('off')
    return fig
