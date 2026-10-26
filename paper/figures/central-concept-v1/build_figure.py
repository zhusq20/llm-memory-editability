"""Concept sketch only: schematic shapes are not measured representations.

Sources: paper/preliminaries-revised.tex; docs/roadmap.md;
docs/reports/layer-evidence-chain-20261021.md; docs/results.md.
This script reads no model weights and runs no experiments.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent
BLUE = "#0077BB"
TEAL = "#008877"
PURPLE = "#7654A3"
ORANGE = "#D96525"
INK = "#243447"
MUTED = "#667789"
LIGHT = "#D4DEE7"
PALE = "#F6F9FC"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "mathtext.fontset": "dejavusans", "svg.fonttype": "none",
    "pdf.fonttype": 42, "savefig.dpi": 200,
})
fig = plt.figure(figsize=(17, 11), facecolor="white")
fig.text(.035, .957, "From a learned answer to reusable knowledge", fontsize=25,
         weight="bold", color=INK)
fig.text(.035, .922, "For reuse: reach the learned parameters, activate a compatible address, and pass on a usable state.",
         fontsize=14, color=MUTED)


def panel(bounds, label, title):
    ax = fig.add_axes(bounds)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.4, .5), 99.2, 99, boxstyle="round,pad=0,rounding_size=2",
                              fc="white", ec=LIGHT, lw=1.2))
    ax.text(4, 93, label, fontsize=15, weight="bold", color=BLUE)
    ax.text(10, 93, title, fontsize=15, weight="bold", color=INK)
    return ax


def txt(ax, x, y, s, size=12, color=INK, ha="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center", **kw)


def box(ax, x, y, w, h, s="", fc=PALE, ec=LIGHT, color=INK, size=12, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.25,rounding_size=1.5",
                              fc=fc, ec=ec, lw=lw))
    if s:
        txt(ax, x + w / 2, y + h / 2, s, size, color)


def arrow(ax, start, end, color=INK, lw=1.6, style="-", rad=0, scale=13):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=scale,
                               lw=lw, color=color, linestyle=style,
                               connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def node(ax, x, y, label, color=BLUE, size=16, radius=5):
    txt(ax, x, y, label, size, color,
        bbox={"boxstyle": f"circle,pad={radius / 20}", "fc": "white", "ec": color, "lw": 1.5})


def plus(ax, x, y):
    txt(ax, x, y, "+", 17, INK, bbox={"boxstyle": "circle,pad=.05", "fc": "white", "ec": LIGHT})


a = panel([.025, .425, .355, .455], "a", "Where reading and writing meet")
txt(a, 53, 83, "One illustrative pre-norm layer", 11, MUTED)

# A faithful residual block, with different memory sources in its two branches.
txt(a, 53, 75, r"$h_{\ell+1,t}$", 15)
arrow(a, (53, 66), (53, 72))
plus(a, 53, 63)
box(a, 33, 47, 40, 11, r"MLP:  $B_\ell\,\phi_\ell(x)$", "#EDF5FA", BLUE)
arrow(a, (53, 58), (53, 60))
txt(a, 53, 43, r"$x=\mathrm{LN}(u)$", 11, MUTED)
arrow(a, (53, 37), (53, 40))
plus(a, 53, 34)
box(a, 31, 17, 44, 11, r"Attention$(\mathrm{LN}(h))$", "#EEF8F5", TEAL, size=11)
arrow(a, (53, 28), (53, 31))
arrow(a, (53, 11), (53, 17))
txt(a, 53, 8, r"$h_{\ell,t}$", 14)

# Residual additions are retained; lateral arrows encode causal reads only.
a.plot([53, 84, 84, 57], [12, 12, 34, 34], color=MUTED, lw=1.1)
a.plot([53, 84, 84, 57], [38, 38, 63, 63], color=MUTED, lw=1.1)
box(a, 4, 17, 23, 15, "Context KV\n" + r"$s\leq t$", "#EEF8F5", TEAL, TEAL, 11)
arrow(a, (27, 24), (31, 24), TEAL)
txt(a, 15.5, 12, "input-dependent", 9, TEAL)
txt(a, 17, 56, "Persistent\nparameters", 11, BLUE)
arrow(a, (25, 53), (33, 53), BLUE)
txt(a, 91, 53, r"$-\nabla L$", 13, ORANGE)
arrow(a, (88, 52), (74, 52), ORANGE, style="--")
txt(a, 89, 45, "write", 10, ORANGE)

b = panel([.395, .425, .58, .455], "b", "Same fact, different retrieval states")
# The semantic graph is deliberately separated from the activation diagram.
node(b, 12, 78, r"$e_1$")
node(b, 30, 78, r"$e_2$")
node(b, 48, 78, r"$e_3$")
arrow(b, (15, 78), (27, 78), BLUE)
arrow(b, (33, 78), (45, 78), BLUE)
txt(b, 21, 84, r"$r_1$", 12, BLUE)
txt(b, 39, 84, r"$r_2$", 12, BLUE)
txt(b, 70, 79, "Both queries ask for " + r"$e_3$" + ".\nOnly one explicitly contains " + r"$e_2$" + ".", 11, MUTED)
b.plot([5, 95], [69, 69], color=LIGHT, lw=1)

box(b, 4, 51, 28, 11, r"$q_2=(e_2,r_2)$", "#EDF5FA", BLUE, BLUE, 15)
box(b, 4, 30, 28, 11, r"$q_{12}=(e_1,r_1,r_2)$", "#F4F0F9", PURPLE, PURPLE, 15)
txt(b, 18, 47, "entity supplied in input", 10, BLUE)
txt(b, 18, 25, "compositional route: internal handoff", 10, PURPLE)
arrow(b, (32, 56), (39, 56), BLUE)
arrow(b, (32, 35), (39, 35), PURPLE, style="--")

# Activation strips are symbolic, not an invented embedding plot.
for row, col, label in [(56, BLUE, r"$\phi_j(x(q_2))$"), (35, PURPLE, r"$\phi_j(x(q_{12}))$")]:
    vals = [.12, .6, .95, .22, .72, .13] if row == 56 else [.16, .43, .77, .35, .86, .11]
    for k, v in enumerate(vals):
        b.add_patch(Rectangle((40 + 2.7 * k, row - 3.5), 2.2, 7, fc=col, ec=col, alpha=.12 + .8*v))
    txt(b, 48, row + 6.4, label, 11, col)
txt(b, 48, 24.5, "compatible address?", 10, PURPLE)
arrow(b, (57, 56), (66, 50), BLUE)
arrow(b, (57, 35), (66, 44), PURPLE, style="--")
box(b, 66, 34, 14, 26, r"$B_j$", "#EDF5FA", BLUE, BLUE, 25)
box(b, 66.6, 34.6, 12.8, 5.5, r"$+\Delta B_j$", "#FFF3E9", ORANGE, ORANGE, 11)
txt(b, 73, 64, "same parameters", 10, BLUE)
arrow(b, (80, 47), (86, 47), BLUE)
box(b, 86, 35, 10, 24, "Later\nlayers", size=10)
arrow(b, (91, 35), (91, 30), BLUE)
txt(b, 91, 27, r"$\longrightarrow e_3$", 14, BLUE)

box(b, 4, 5, 92, 13, fc=PALE)
txt(b, 50, 13.5, r"$K_j=\phi_j(x(q_2))^\top\phi_j(x(q_{12}))$", 14)
txt(b, 50, 8, "Address overlap controls the direct MLP response; later computation determines the answer.", 10, MUTED)

c = panel([.025, .105, .465, .295], "c", "Updating knowledge changes its uses")
txt(c, 26, 79, "Edit the first fact", 13, ORANGE, weight="bold")
txt(c, 75, 79, "Edit the second fact", 13, ORANGE, weight="bold")
c.plot([50, 50], [24, 84], color=LIGHT, lw=1)

node(c, 8, 52, r"$e_1$", size=15)
node(c, 24, 63, r"$e_2$", color=MUTED, size=15)
node(c, 41, 63, r"$e_3$", color=MUTED, size=15)
node(c, 24, 39, r"$e'_2$", color=ORANGE, size=15)
txt(c, 41, 39, r"$r_2(e'_2)$", 14, ORANGE)
arrow(c, (11, 54), (21, 62), LIGHT)
arrow(c, (27, 63), (38, 63), LIGHT)
arrow(c, (11, 49), (21, 40), ORANGE)
arrow(c, (27, 39), (35, 39), ORANGE)
txt(c, 16, 38, r"$r_1$", 10, ORANGE)
txt(c, 32, 46, r"$r_2$", 10, ORANGE)
txt(c, 26, 23, "Supply a new intermediate entity", 11, MUTED)

node(c, 57, 51, r"$e_1$", size=15)
node(c, 72, 51, r"$e_2$", size=15)
node(c, 90, 51, r"$e'_3$", color=ORANGE, size=15)
arrow(c, (60, 51), (69, 51), BLUE)
arrow(c, (75, 51), (87, 51), ORANGE)
txt(c, 64.5, 61, r"$r_1$", 11, BLUE)
txt(c, 81, 61, r"$r_2$", 11, ORANGE)
txt(c, 75, 33, r"$q_2\to e'_3\quad q_{12}\to e'_3$", 14, ORANGE)
txt(c, 75, 23, "Reuse the same updated association", 11, MUTED)
txt(c, 50, 8, "Desired behavior: update dependent answers; preserve unrelated answers.", 11, MUTED)

d = panel([.505, .105, .47, .295], "d", "Two routes to reusable knowledge")
txt(d, 27, 79, "Share an entity address", 13, TEAL, weight="bold")
txt(d, 76, 79, "Re-enter a compatible computation", 12, PURPLE, weight="bold")
d.plot([50, 50], [24, 84], color=LIGHT, lw=1)

box(d, 6, 36, 20, 24, "Entity\naddress", "#EEF8F5", TEAL, TEAL, 12)
for yy, label in [(65, "Recall"), (48, "Compose"), (31, "New use")]:
    arrow(d, (26, 48), (32, yy), TEAL)
    box(d, 32, yy - 5, 14, 10, label, size=10)
txt(d, 26, 19, "Diverse expressions / shared-use training", 10, MUTED)

box(d, 55, 46, 17, 15, "Output\nstate", "#F4F0F9", PURPLE, PURPLE, 11)
arrow(d, (72, 53), (77, 53), PURPLE)
box(d, 77, 46, 18, 15, "Usable\ninput", "#EEF8F5", TEAL, TEAL, 11)
txt(d, 75, 67, "e.g. vocabulary decode + re-encode", 10, MUTED)
box(d, 63, 24, 24, 12, "Learned layers", "#EDF5FA", BLUE, BLUE, 11)
arrow(d, (86, 46), (82, 36), PURPLE)
arrow(d, (65, 36), (61, 46), PURPLE)
txt(d, 75, 17, "CoT / repeated shared blocks", 10, MUTED)
txt(d, 50, 8, "Architecture controls both access to the parameters and the form of the handoff.", 11, INK)

fig.text(.035, .066,
         r"Writing:  $\Delta B_\ell=-\eta\sum_s\delta_{\ell,s}\phi_\ell(x_{\ell,s})^\top$"
         r"     Reading:  $\Delta m_\ell(x)=\Delta B_\ell\phi_\ell(x)$",
         fontsize=15, color=INK)
fig.text(.035, .031,
         "CONCEPT SKETCH  ·  Activation strips and graph layouts are schematic. Dashed purple path denotes required reuse, not a measured circuit.\n"
         "The readout identity holds for a down-projection update at fixed features. Layer identities are not fixed knowledge/reasoning regions.",
         fontsize=10, color=MUTED)

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"central-concept.{ext}", facecolor="white")
plt.close(fig)
print(OUT / "central-concept.png")
