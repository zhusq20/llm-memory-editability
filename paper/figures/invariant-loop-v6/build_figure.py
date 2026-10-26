"""Vertical token-state diagram for the implemented Invariant Loop.

Uses matplotlib only. No model, experiment, or training code is imported.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STEM = "invariant-loop-vertical"
INK = "#223849"
BLUE = "#0876B5"
TEAL = "#008C7D"
GOLD = "#BC862D"
GRAY = "#607381"


class Figure:
    def __init__(self):
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "mathtext.fontset": "dejavusans",
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
            }
        )
        self.fig = plt.figure(figsize=(7.2, 9.0), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, 7.2), ylim=(9.0, 0))
        self.ax.axis("off")

    def text(self, x, y, value, size=12, color=INK, ha="center", **kwargs):
        return self.ax.text(
            x,
            y,
            value,
            ha=ha,
            va="center",
            fontsize=size,
            color=color,
            zorder=6,
            **kwargs,
        )

    def arrow(self, points, color=GRAY, width=1.3, head=True):
        for a, b in zip(points[:-2], points[1:-1], strict=True):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=width, zorder=1)
        self.ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>" if head else "-",
                mutation_scale=10,
                linewidth=width,
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )

    def state(self, x, y, kind, aggregate=False):
        styles = {
            "carry": ("#D9EBF8", BLUE),
            "output": ("#F9DF98", GOLD),
            "candidate": ("#CDECE2", TEAL),
        }
        fill, edge = styles[kind]
        if aggregate:
            self.ax.add_patch(
                Circle((x, y), 0.205, facecolor="white", edgecolor=TEAL, linewidth=1.45, zorder=3)
            )
        self.ax.add_patch(
            Circle((x, y), 0.145, facecolor=fill, edgecolor=edge, linewidth=1.3, zorder=4)
        )

    def feedback(self, x, y):
        self.ax.add_patch(
            FancyBboxPatch(
                (x - 0.105, y - 0.105),
                0.21,
                0.21,
                boxstyle="round,pad=0,rounding_size=0.025",
                facecolor="#8DD8C3",
                edgecolor=TEAL,
                linewidth=1.2,
                zorder=4,
            )
        )

    def layer(self, y, index):
        self.ax.add_patch(
            FancyBboxPatch(
                (1.1, y),
                3.4,
                0.62,
                boxstyle="round,pad=0,rounding_size=0.075",
                facecolor="#F3F7FB",
                edgecolor=BLUE,
                linewidth=1.35,
                zorder=3,
            )
        )
        self.text(2.8, y + 0.31, r"Transformer $F_\theta$", 14)
        self.text(5.2, y + 0.31, index, 12, color=GRAY, ha="left")

    def save(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        frame = self.fig.bbox
        for item in self.ax.texts:
            box = item.get_window_extent(renderer)
            assert frame.x0 <= box.x0 <= box.x1 <= frame.x1, item.get_text()
            assert frame.y0 <= box.y0 <= box.y1 <= frame.y1, item.get_text()
        for extension in ("png", "pdf", "svg"):
            self.fig.savefig(HERE / f"{STEM}.{extension}", dpi=300)
        self.fig.savefig(HERE / f"{STEM}-preview.png", dpi=170)
        plt.close(self.fig)


def main():
    c = Figure()
    c.text(3.6, 0.37, "Invariant Loop", 23, weight="bold")
    xs = (1.7, 2.8, 3.9)
    for i, x in enumerate(xs, start=1):
        c.text(x, 0.83, rf"$x_{i}$", 12, color=GRAY)

    # Only the inter-loop computation is highlighted.
    c.ax.add_patch(
        FancyBboxPatch(
            (1.08, 2.96),
            3.44,
            3.17,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor="#F2FAF7",
            edgecolor="none",
            zorder=0,
        )
    )
    c.layer(2.02, r"Loop $t$")
    c.layer(6.55, r"Loop $t+1$")

    # Every column is one token position, with the same local update.
    for x in xs:
        c.state(x, 1.25, "carry")
        c.arrow([(x, 1.395), (x, 2.02)], color=BLUE)
        c.arrow([(x, 2.64), (x, 3.055)], color=GOLD)
        c.state(x, 3.2, "output")

        # H -> c, together with H -> soft feedback -> c.
        c.arrow([(x, 3.345), (x, 4.355)], color=GOLD)
        qx = x + 0.38
        c.arrow([(x + 0.105, 3.305), (qx, 3.715)], color=TEAL)
        c.feedback(qx, 3.82)
        c.arrow([(qx, 3.925), (x + 0.09, 4.385)], color=TEAL)
        c.state(x, 4.5, "candidate")

        c.arrow([(x, 4.645), (x, 5.545)], color=TEAL, width=1.55)
        c.state(x, 5.75, "carry", aggregate=True)
        c.arrow([(x, 5.955), (x, 6.55)], color=BLUE)
        c.arrow([(x, 7.17), (x, 7.655)], color=GOLD)
        c.state(x, 7.8, "output")
        c.arrow([(x, 7.945), (x, 8.28)], color=GOLD)
        c.text(x, 8.5, r"$\vdots$", 18, color=GRAY)

    # One representative carry is drawn in full; aggregation is per token.
    c.arrow(
        [(4.045, 1.25), (4.86, 1.25), (4.86, 5.75), (4.105, 5.75)],
        color=BLUE,
        width=1.45,
    )
    c.text(4.26, 5.08, r"$\gamma_t$", 12, color=TEAL)
    c.text(5.2, 3.82, "Soft feedback", 12, color=TEAL, ha="left")
    c.text(5.2, 5.02, "State aggregation", 12, color=TEAL, ha="left")
    c.text(5.2, 5.43, r"5 statistics $\to\gamma_t$", 10.5, color=GRAY, ha="left")

    for y, label in (
        (1.25, r"$h_{t-1}$"),
        (3.2, r"$H_t$"),
        (4.5, r"$c_t$"),
        (5.75, r"$h_t$"),
        (7.8, r"$H_{t+1}$"),
    ):
        c.text(0.64, y, label, 13)
    c.text(3.6, 8.84, "Shared weights. Same update at each token.", 10.5, color=GRAY)
    c.save()

    sources = [
        "src/llm_memory_editability/loop_native_handoff.py",
        "src/llm_memory_editability/self_decode_alignment.py",
        "configs/loop-native-handoff-development-v1.json",
        "paper/figures/invariant-loop-v6/build_figure.py",
    ]
    manifest = {
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        "reference": "DiscoLoop v2, Figure 1, https://arxiv.org/abs/2607.00341v2",
        "diagram": {
            "orientation": "vertical",
            "language": "English",
            "token_positions_shown": 3,
            "shared_transformer_calls": 2,
            "full_carry_paths_shown": 1,
            "caption_only": [
                "input reinjection",
                "normalization",
                "auxiliary recurrent state",
                "weighting equations",
                "initialization",
            ],
        },
        "checks": {"text_inside_canvas": "passed", "model_executed": False},
        "figure_table_trace": [
            {
                "artifact_id": STEM,
                "source_data": {"file": sources[0]},
                "transformation": {"script": "build_figure.py"},
                "caption_claim": "Shared Transformer calls are connected by per-token soft "
                "feedback and aggregation of the previous state and candidate.",
                "supported_manuscript_claims": [],
                "limitations": [
                    "Only one old-state carry path is fully drawn.",
                    "Exact equations and auxiliary state are specified in the caption.",
                ],
            }
        ],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Rendered vertical English diagram with three token columns and two shared layers.")


if __name__ == "__main__":
    main()
