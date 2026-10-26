"""Vertical token-state diagram for the implemented Invariant Loop.

Uses matplotlib only. No model, experiment, or training code is imported.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STEM = "invariant-loop-vertical"
INK = "#223849"
BLUE = "#0876B5"
TEAL = "#008C7D"
GOLD = "#BC862D"
GRAY = "#607381"
TOKEN_XS = (1.8, 3.4, 5.0)
CARRY_OFFSET = 0.72


class Figure:
    def __init__(self):
        self.token_alpha = 1.0
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "mathtext.fontset": "dejavusans",
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
            }
        )
        self.fig = plt.figure(figsize=(8.2, 9.0), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, 8.2), ylim=(9.0, 0))
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
            self.ax.plot(
                [a[0], b[0]], [a[1], b[1]], color=color, lw=width, alpha=self.token_alpha, zorder=1
            )
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
                alpha=self.token_alpha,
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
                Circle(
                    (x, y),
                    0.205,
                    facecolor="white",
                    edgecolor=TEAL,
                    linewidth=1.45,
                    alpha=self.token_alpha,
                    zorder=3,
                )
            )
        self.ax.add_patch(
            Circle(
                (x, y),
                0.145,
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.3,
                alpha=self.token_alpha,
                zorder=4,
            )
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
                alpha=self.token_alpha,
                zorder=4,
            )
        )

    def layer(self, y, index):
        # A single layer band with three token ports. White channels belong to
        # the external carry, not to the Transformer's computation.
        self.ax.add_patch(
            FancyBboxPatch(
                (1.15, y),
                4.9,
                0.62,
                boxstyle="round,pad=0,rounding_size=0.075",
                facecolor="#E8EFF6",
                edgecolor="none",
                zorder=0.4,
            )
        )
        for x in TOKEN_XS:
            self.ax.add_patch(
                Rectangle(
                    (x + CARRY_OFFSET - 0.10, y + 0.06),
                    0.20,
                    0.585,
                    facecolor="white",
                    edgecolor="none",
                    zorder=0.5,
                )
            )
            self.ax.add_patch(
                FancyBboxPatch(
                    (x - 0.22, y + 0.17),
                    0.44,
                    0.28,
                    boxstyle="round,pad=0,rounding_size=0.045",
                    facecolor="#D9E5F2",
                    edgecolor=BLUE,
                    linewidth=1.25,
                    alpha=1.0 if x == TOKEN_XS[-1] else 0.28,
                    zorder=3,
                )
            )
        self.text(
            6.4,
            y + 0.31,
            "Shared Transformer\n" + r"$F_\theta$",
            11,
            color=INK,
            ha="left",
            linespacing=1.25,
        )
        self.text(0.62, y + 0.31, index, 10.5, color=GRAY)

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
    c.text(4.1, 0.37, "Invariant Loop", 23, weight="bold")
    xs = TOKEN_XS
    for i, x in enumerate(xs, start=1):
        c.text(x, 0.83, rf"$x_{i}$", 12, color=GRAY)

    # Only the inter-loop computation is highlighted.
    c.ax.add_patch(
        FancyBboxPatch(
            (1.15, 2.96),
            4.9,
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
        c.token_alpha = 1.0 if x == xs[-1] else 0.28
        c.state(x, 1.25, "carry")
        c.arrow([(x, 1.395), (x, 2.19)], color=BLUE)
        c.arrow([(x, 2.47), (x, 3.055)], color=GOLD)
        c.state(x, 3.2, "output")

        # H -> c, together with H -> soft feedback -> c.
        c.arrow([(x, 3.345), (x, 4.355)], color=GOLD)
        fx = x + 0.38
        c.arrow([(x + 0.105, 3.305), (fx, 3.715)], color=TEAL)
        c.feedback(fx, 3.82)
        c.arrow([(fx, 3.925), (x + 0.09, 4.385)], color=TEAL)
        if x == xs[-1]:
            c.text(x + 0.54, 3.82, r"$f_t$", 11, color=TEAL)
        c.state(x, 4.5, "candidate")

        c.arrow([(x, 4.645), (x, 5.545)], color=TEAL, width=1.55)
        c.state(x, 5.75, "carry", aggregate=True)
        c.arrow([(x, 5.955), (x, 6.72)], color=BLUE)
        c.arrow([(x, 7.0), (x, 7.655)], color=GOLD)
        c.state(x, 7.8, "output")
        c.arrow([(x, 7.945), (x, 8.28)], color=GOLD)
        c.text(x, 8.5, r"$\vdots$", 18, color=GRAY)

    # Identical local carry lanes: no token crosses another token's wiring.
    carries = [
        [(x + 0.145, 1.25), (x + CARRY_OFFSET, 1.25), (x + CARRY_OFFSET, 5.75), (x + 0.205, 5.75)]
        for x in xs
    ]
    for x, points in zip(xs, carries, strict=True):
        c.token_alpha = 1.0 if x == xs[-1] else 0.28
        c.arrow(points, color=BLUE, width=1.45)
    c.token_alpha = 1.0
    c.text(5.36, 5.08, r"$\gamma_t$", 12, color=TEAL)
    c.text(6.4, 3.82, "Soft feedback", 12, color=TEAL, ha="left")
    c.text(6.4, 5.02, "State aggregation", 12, color=TEAL, ha="left")
    # The compact module summarizes the learned score and scalar normalization.
    c.ax.add_patch(
        FancyBboxPatch(
            (6.4, 5.23),
            0.72,
            0.4,
            boxstyle="round,pad=0,rounding_size=0.055",
            facecolor="#EDF7F3",
            edgecolor=TEAL,
            linewidth=1.2,
            zorder=3,
        )
    )
    c.text(6.76, 5.43, "MLP", 11, color=TEAL)
    c.arrow([(7.12, 5.43), (7.55, 5.43)], color=TEAL, width=1.1)
    c.text(7.77, 5.43, r"$\gamma_t$", 13, color=TEAL)
    c.text(6.4, 6.02, "Common Q/K/V input", 11, color=BLUE, ha="left")
    c.text(6.4, 6.34, r"$\mathrm{LN}_1(h_t+e)$", 11, color=GRAY, ha="left")
    c.arrow([(6.2, 6.34), (5.05, 6.34)], color=GRAY, width=0.8, head=False)

    for y, label in (
        (1.25, r"$h_{t-1}$"),
        (3.2, r"$H_t$"),
        (4.5, r"$c_t$"),
        (5.75, r"$h_t$"),
        (7.8, r"$H_{t+1}$"),
    ):
        c.text(0.12, y, label, 13, ha="left")
    c.text(4.1, 8.84, "One token highlighted. Same update at every token.", 10.5, color=GRAY)
    c.save()

    sources = [
        "src/llm_memory_editability/loop_native_handoff.py",
        "src/llm_memory_editability/self_decode_alignment.py",
        "configs/loop-native-handoff-development-v1.json",
        "paper/figures/invariant-loop-v11/build_figure.py",
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
            "highlighted_token": 3,
            "soft_feedback_symbol": "f_t",
            "next_loop_qkv_common_input": "LN_1(h_t + e)",
            "gamma_display": "compact MLP-to-gamma abstraction",
            "shared_transformer_calls": 2,
            "full_carry_paths_shown": len(carries),
            "carry_paths": [
                {"token": i, "from": f"h_(t-1,{i})", "to": f"h_(t,{i})", "points": p}
                for i, p in enumerate(carries, start=1)
            ],
            "wire_bridges": [],
            "carry_routing": "one straight local channel beside each token",
            "layer_representation": "one joint Transformer band with per-token ports",
            "token_lanes": [[x - 0.22, x + CARRY_OFFSET + 0.10] for x in xs],
            "caption_only": [
                "feedback and candidate RMS normalization",
                "auxiliary recurrent state",
                "weighting equations",
                "score clipping, exponentiation, and historical-mass normalization",
                "initialization",
            ],
        },
        "checks": {"text_inside_canvas": "passed", "model_executed": False},
        "figure_table_trace": [
            {
                "artifact_id": STEM,
                "source_data": {"file": sources[0]},
                "transformation": {"script": "build_figure.py"},
                "caption_claim": "Soft feedback f_t and MLP-based state aggregation are "
                "highlighted for one token; after input reinjection and LayerNorm, the "
                "aggregate supplies the common next-loop Q/K/V input.",
                "supported_manuscript_claims": [],
                "limitations": [
                    "Exact equations and auxiliary state are specified in the caption.",
                ],
            }
        ],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Rendered vertical English diagram with three token columns and two shared layers.")


if __name__ == "__main__":
    main()
