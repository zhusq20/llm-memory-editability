"""Draw a separate, implementation-based enlargement of Invariant Loop gamma.

Only matplotlib is required; no model or experiment code is executed.
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
STEM = "invariant-loop-gamma-detail"
INK = "#223849"
GRAY = "#607381"
BLUE = "#0876B5"
TEAL = "#008C7D"
ORANGE = "#B87928"


class Detail:
    def __init__(self):
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "mathtext.fontset": "dejavusans",
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
            }
        )
        self.fig = plt.figure(figsize=(6.9, 8.65), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, 6.9), ylim=(8.65, 0))
        self.ax.axis("off")

    def text(self, x, y, value, size=12, color=INK, **kwargs):
        return self.ax.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            ha=kwargs.pop("ha", "center"),
            va="center",
            zorder=5,
            **kwargs,
        )

    def box(self, x, y, w, h, label="", fill="#EDF7F3", edge=TEAL, size=12):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.075",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.25,
                zorder=2,
            )
        )
        if label:
            self.text(x + w / 2, y + h / 2, label, size)

    def arrow(self, points, color=TEAL, dashed=False, head=True, width=1.35):
        for a, b in zip(points[:-2], points[1:-1], strict=True):
            self.ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=color,
                lw=width,
                ls="--" if dashed else "-",
                zorder=1,
            )
        self.ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>" if head else "-",
                color=color,
                linewidth=width,
                linestyle="--" if dashed else "-",
                mutation_scale=11,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )

    def state(self, x, y, label, candidate=False, aggregate=False):
        if aggregate:
            self.ax.add_patch(
                Circle((x, y), 0.34, facecolor="white", edgecolor=TEAL, linewidth=1.5, zorder=3)
            )
        self.ax.add_patch(
            Circle(
                (x, y),
                0.27,
                facecolor="#CDECE2" if candidate else "#D9EBF8",
                edgecolor=TEAL if candidate else BLUE,
                linewidth=1.4,
                zorder=4,
            )
        )
        self.text(x, y, label, 15)

    def save(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for item in self.ax.texts:
            box = item.get_window_extent(renderer)
            assert self.fig.bbox.contains(box.x0, box.y0), item.get_text()
            assert self.fig.bbox.contains(box.x1, box.y1), item.get_text()
        for extension in ("png", "svg", "pdf"):
            self.fig.savefig(HERE / f"{STEM}.{extension}", dpi=300)
        self.fig.savefig(HERE / f"{STEM}-preview.png", dpi=170)
        plt.close(self.fig)


def main():
    c = Detail()
    c.text(3.45, 0.36, "State aggregation", 22, weight="bold")
    c.text(3.45, 0.82, r"One token · update weight $\gamma_t$", 11, color=GRAY)

    # Probability and state statistics remain distinct source groups.
    c.text(1.76, 1.32, r"$p_t,\ p_{t-1}$", 15, color=BLUE)
    c.text(5.1, 1.32, r"$c_t,\ h_{t-1}$", 15, color=BLUE)
    c.arrow([(1.76, 1.52), (1.76, 1.78)])
    c.arrow([(5.1, 1.52), (5.1, 1.78)])
    c.box(0.38, 1.78, 2.8, 1.76)
    c.box(3.72, 1.78, 2.8, 1.76)
    for y, label in (
        (2.13, "Normalized entropy"),
        (2.66, "Max probability"),
        (3.19, "Distribution overlap"),
    ):
        c.text(1.78, y, label, 11.5)
    c.text(5.12, 2.37, "Relative change", 11.5)
    c.text(5.12, 2.98, "Cosine similarity", 11.5)

    c.arrow([(1.78, 3.54), (1.78, 3.82), (3.45, 3.82)], head=False)
    c.arrow([(5.12, 3.54), (5.12, 3.82), (3.45, 3.82)], head=False)
    c.arrow([(3.45, 3.82), (3.45, 4.08)])
    c.box(2.15, 4.08, 2.6, 0.72, "Shared MLP\n" + r"$5\to16\to1$", size=12)
    c.arrow([(3.45, 4.8), (3.45, 5.13)])
    c.text(4.04, 4.98, r"$s_t$", 13, color=TEAL)
    c.box(2.0, 5.13, 2.9, 0.58, r"$w_t=\exp(\mathrm{clip}(s_t,-8,8))$", fill="#F5FAF8", size=12)
    c.arrow([(3.45, 5.71), (3.45, 6.06)])

    # Retained mass is an independent input, not produced by the scorer.
    c.box(0.38, 5.42, 1.35, 1.0, fill="#FFF2DF", edge=ORANGE)
    c.text(1.055, 5.68, "Retained", 10.5, color=ORANGE)
    c.text(1.055, 5.94, "weight", 10.5, color=ORANGE)
    c.text(1.055, 6.21, r"$\beta D_{t-1}$", 13, color=ORANGE)
    c.arrow([(1.73, 6.2), (2.0, 6.2)], color=ORANGE)
    c.box(2.0, 6.06, 2.9, 0.88, r"$\gamma_t=\frac{w_t}{\beta D_{t-1}+w_t}$", size=18)

    # The scalar controls both arms of the convex state update.
    c.arrow([(3.45, 6.94), (3.45, 7.51)], dashed=True)
    c.state(1.055, 7.85, r"$h_{t-1}$")
    c.state(5.85, 7.85, r"$c_t$", candidate=True)
    c.state(3.45, 7.85, r"$h_t$", aggregate=True)
    c.arrow([(1.325, 7.85), (3.11, 7.85)], color=BLUE)
    c.arrow([(5.58, 7.85), (3.79, 7.85)], color=TEAL)
    c.text(2.16, 7.52, r"$1-\gamma_t$", 13, color=BLUE)
    c.text(4.67, 7.52, r"$\gamma_t$", 13, color=TEAL)
    c.text(3.45, 8.43, "Same rule at every token.", 10.5, color=GRAY)
    c.save()

    sources = [
        "src/llm_memory_editability/loop_native_handoff.py",
        "configs/loop-native-handoff-development-v1.json",
        "paper/figures/invariant-loop-v10/build_detail.py",
    ]
    manifest = {
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        "overview": "../invariant-loop-v9/invariant-loop-vertical.pdf",
        "diagram": {
            "language": "English",
            "scope": "one token, one loop boundary",
            "features": ["entropy", "max_probability", "agreement", "relative_change", "cosine"],
            "retained_weight": "beta * D_(t-1)",
            "candidate_weight": "exp(clamp(MLP(phi),-8,8))",
            "acceptance": "candidate_weight / (retained_weight + candidate_weight)",
            "state_update": "h_t = (1-gamma_t)*h_(t-1) + gamma_t*c_t",
        },
        "checks": {"text_inside_canvas": "passed", "model_executed": False},
        "figure_table_trace": [
            {
                "artifact_id": STEM,
                "source_data": {"file": sources[0]},
                "transformation": {"script": "build_detail.py"},
                "caption_claim": "Five-statistic scoring and retained mass determine the "
                "candidate's fraction in a convex hidden-state update.",
                "supported_manuscript_claims": [],
                "limitations": ["Architecture detail only; no empirical correctness claim."],
            }
        ],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Rendered the independent gamma detail figure.")


if __name__ == "__main__":
    main()
