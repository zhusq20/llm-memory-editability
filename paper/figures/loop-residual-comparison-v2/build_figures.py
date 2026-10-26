"""Three independent vertical figures: Vanilla Loop, Highway Networks, and Invariant Loop.

Only matplotlib is imported. Model sources are read for provenance, never run.
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
INK, BLUE, TEAL, GOLD, GRAY = "#223849", "#0876B5", "#008C7D", "#BC862D", "#607381"
XS = (1.8, 3.6, 5.4)
CARRY_OFFSET = -0.65
CANVAS = (9.4, 10.2)
INPUT_Y, OUTPUT_Y, CANDIDATE_Y, MIX_Y = 1.3, 3.25, 4.65, 6.4
LAYER_YS = (2.05, 7.38)
CONTROLLER = (6.95, 4.96, 1.8, 0.8)


class Diagram:
    def __init__(self, kind):
        self.kind = kind
        self.alpha = 1.0
        self.paths = []
        self.carries = []
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "mathtext.fontset": "dejavusans",
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
            }
        )
        self.fig = plt.figure(figsize=CANVAS, facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, CANVAS[0]), ylim=(CANVAS[1], 0))
        self.ax.axis("off")

    def text(self, x, y, value, size=12, color=INK, ha="center", **kwargs):
        return self.ax.text(
            x, y, value, fontsize=size, color=color, ha=ha, va="center", zorder=8, **kwargs
        )

    def box(self, x, y, width, height, fill, edge="none", lw=1.2, zorder=3):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=fill,
                edgecolor=edge,
                linewidth=lw,
                alpha=self.alpha,
                zorder=zorder,
            )
        )

    def arrow(self, points, color=GRAY, width=1.35, dashed=False, head=True, role=None):
        self.paths.append({"role": role, "points": points})
        for a, b in zip(points[:-2], points[1:-1], strict=True):
            self.ax.plot(
                (a[0], b[0]),
                (a[1], b[1]),
                color=color,
                linewidth=width,
                linestyle="--" if dashed else "-",
                alpha=self.alpha,
                zorder=2,
            )
        self.ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>" if head else "-",
                mutation_scale=10,
                shrinkA=0,
                shrinkB=0,
                linewidth=width,
                linestyle="--" if dashed else "-",
                color=color,
                alpha=self.alpha,
                zorder=2,
            )
        )

    def state(self, x, y, kind="carry", aggregate=False):
        fill, edge = {
            "carry": ("#D9EBF8", BLUE),
            "output": ("#F9DF98", GOLD),
            "candidate": ("#CDECE2", TEAL),
        }[kind]
        if aggregate:
            self.ax.add_patch(
                Circle(
                    (x, y),
                    0.205,
                    facecolor="white",
                    edgecolor=TEAL,
                    lw=1.45,
                    alpha=self.alpha,
                    zorder=4,
                )
            )
        self.ax.add_patch(
            Circle(
                (x, y), 0.145, facecolor=fill, edgecolor=edge, lw=1.3, alpha=self.alpha, zorder=5
            )
        )

    def layer(self, y, loop):
        self.box(0.9, y, 5.3, 0.65, "#E8EFF6", zorder=0.5)
        for x in XS:
            if self.kind != "vanilla-loop":
                self.ax.add_patch(
                    Rectangle(
                        (x + CARRY_OFFSET - 0.09, y + 0.04),
                        0.18,
                        0.63,
                        facecolor="white",
                        edgecolor="none",
                        zorder=0.6,
                    )
                )
            self.alpha = 1.0 if x == XS[-1] else 0.28
            self.box(x - 0.22, y + 0.185, 0.44, 0.28, "#D9E5F2", BLUE)
        self.alpha = 1.0
        name = "Transform" if self.kind == "highway" else "Shared Transformer"
        symbol = (
            (r"$H_\ell$" if y == LAYER_YS[0] else r"$H_{\ell+1}$")
            if self.kind == "highway"
            else r"$F_\theta$"
        )
        self.text(6.65, y + 0.325, name + "\n" + symbol, 11.5, ha="left", linespacing=1.25)
        self.text(0.43, y + 0.325, loop, 11, color=GRAY)

    def carry(self, x, token):
        points = [
            (x - 0.145, INPUT_Y),
            (x + CARRY_OFFSET, INPUT_Y),
            (x + CARRY_OFFSET, MIX_Y),
            (x - 0.205, MIX_Y),
        ]
        self.carries.append({"token": token, "points": points})
        self.arrow(points, BLUE, width=1.5, role=f"carry-{token}")

    def controller(self):
        x, y, width, height = CONTROLLER
        self.box(x, y, width, height, "#EDF7F3", TEAL)
        self.ax.plot((x, x + width), (y + 0.4, y + 0.4), color=TEAL, lw=0.7, zorder=4)
        self.text(
            x + width / 2, y + 0.2, "Linear" if self.kind == "highway" else "MLP", 11.5, color=TEAL
        )
        self.text(
            x + width / 2,
            y + 0.6,
            "Sigmoid" if self.kind == "highway" else "Mass norm",
            11.5,
            color=TEAL,
        )
        self.arrow(
            [(7.85, y + height), (7.85, MIX_Y), (XS[-1] + 0.205, MIX_Y)],
            TEAL,
            dashed=True,
            role="gate-to-aggregate",
        )
        label = r"Vector $g_\ell$" if self.kind == "highway" else r"Scalar $\gamma_t$"
        self.text(6.95, 6.15, label, 12, color=TEAL)

    def save(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        frame = self.fig.bbox
        for item in self.ax.texts:
            bounds = item.get_window_extent(renderer)
            assert frame.contains(bounds.x0, bounds.y0), item.get_text()
            assert frame.contains(bounds.x1, bounds.y1), item.get_text()
        for extension in ("png", "pdf", "svg"):
            self.fig.savefig(HERE / f"{self.kind}.{extension}", dpi=300)
        self.fig.savefig(HERE / f"{self.kind}-preview.png", dpi=150)
        plt.close(self.fig)


def render(kind):
    c = Diagram(kind)
    c.text(
        4.7,
        0.4,
        {
            "vanilla-loop": "Vanilla Loop",
            "highway": "Highway Networks",
            "invariant-loop": "Invariant Loop",
        }[kind],
        23,
        weight="bold",
    )
    for token, x in enumerate(XS, start=1):
        c.text(x, 0.87, rf"$i={token}$" if kind == "highway" else rf"$x_{token}$", 12, color=GRAY)
    if kind != "vanilla-loop":
        c.box(0.9, 2.99, 5.3, 3.78, "#F2FAF7", zorder=0)
    c.layer(LAYER_YS[0], r"Layer $\ell$" if kind == "highway" else r"Loop $t$")
    c.layer(LAYER_YS[1], r"Layer $\ell+1$" if kind == "highway" else r"Loop $t+1$")

    for token, x in enumerate(XS, start=1):
        c.alpha = 1.0 if x == XS[-1] else 0.28
        c.state(x, INPUT_Y)
        c.arrow([(x, INPUT_Y + 0.145), (x, LAYER_YS[0] + 0.185)], BLUE)
        if kind == "highway":
            c.arrow([(x, LAYER_YS[0] + 0.465), (x, CANDIDATE_Y - 0.145)], GOLD)
            c.state(x, CANDIDATE_Y, "candidate")
        else:
            c.arrow([(x, LAYER_YS[0] + 0.465), (x, OUTPUT_Y - 0.145)], GOLD)
            c.state(x, OUTPUT_Y, "output")
            if kind == "invariant-loop":
                c.arrow([(x, OUTPUT_Y + 0.145), (x, CANDIDATE_Y - 0.145)], GOLD)
                fx, fy = x + 0.40, 3.95
                c.arrow([(x + 0.105, OUTPUT_Y + 0.105), (fx, fy - 0.105)], TEAL)
                c.box(fx - 0.105, fy - 0.105, 0.21, 0.21, "#8DD8C3", TEAL)
                c.arrow([(fx, fy + 0.105), (x + 0.09, CANDIDATE_Y - 0.115)], TEAL)
                c.state(x, CANDIDATE_Y, "candidate")
                if x == XS[-1]:
                    c.text(fx + 0.18, fy, r"$f_t$", 12, color=TEAL, ha="left")

        if kind == "vanilla-loop":
            c.arrow([(x, OUTPUT_Y + 0.145), (x, LAYER_YS[1] + 0.185)], BLUE)
        else:
            c.arrow([(x, CANDIDATE_Y + 0.145), (x, MIX_Y - 0.205)], TEAL, width=1.6)
            c.state(x, MIX_Y, aggregate=True)
            c.carry(x, token)
            c.arrow([(x, MIX_Y + 0.205), (x, LAYER_YS[1] + 0.185)], BLUE)
        c.arrow([(x, LAYER_YS[1] + 0.465), (x, 8.555)], GOLD)
        c.state(x, 8.7, "output")
        c.arrow([(x, 8.845), (x, 9.13)], GOLD)
        c.text(x, 9.31, r"$\vdots$", 18, color=GRAY)

    c.alpha = 1.0
    if kind == "highway":
        # Highway's transform gate reads the layer input directly.
        c.arrow(
            [(XS[-1] + 0.145, INPUT_Y), (6.3, INPUT_Y), (6.3, 4.55), (7.85, 4.55), (7.85, 4.96)],
            TEAL,
            width=1.15,
            role="layer-input-to-gate",
        )
        c.text(7.85, 4.0, "Layer input", 12, color=TEAL)
        c.text(7.85, 4.25, r"$h_{\ell-1}$", 12, color=TEAL)
        c.controller()
        labels = (
            (INPUT_Y, r"$h_{\ell-1}$"),
            (CANDIDATE_Y, r"$u_\ell$"),
            (MIX_Y, r"$h_\ell$"),
            (8.7, r"$u_{\ell+1}$"),
        )
        weight, kept = r"$g_\ell$", r"$1-g_\ell$"
        formula = r"$h_\ell=(1-g_\ell)\odot h_{\ell-1}+g_\ell\odot u_\ell$"
    elif kind == "invariant-loop":
        c.text(7.85, 4.0, r"5 statistics $\phi_t$", 12, color=TEAL)
        c.text(7.85, 4.25, r"from $c_t,\ h_{t-1}$", 11.5, color=GRAY)
        c.text(7.85, 4.49, r"$p_t,\ p_{t-1}$", 11.5, color=GRAY)
        c.arrow([(7.85, 4.72), (7.85, 4.96)], TEAL, width=1.15, role="statistics-to-gate")
        c.controller()
        c.text(6.30, 5.08, "Retained mass", 11, color=BLUE)
        c.box(5.85, 5.35, 0.90, 0.42, "#F0F6FC", BLUE, lw=1.0)
        c.text(6.30, 5.56, r"$\beta D_{t-1}$", 11.5, color=BLUE)
        c.arrow(
            [(6.75, 5.56), (6.95, 5.56)],
            BLUE,
            dashed=True,
            width=1.0,
            role="retained-mass-to-normalization",
        )
        labels = (
            (INPUT_Y, r"$h_{t-1}$"),
            (OUTPUT_Y, r"$H_t$"),
            (CANDIDATE_Y, r"$c_t$"),
            (MIX_Y, r"$h_t$"),
            (8.7, r"$H_{t+1}$"),
        )
        weight, kept = r"$\gamma_t$", r"$1-\gamma_t$"
        formula = r"$h_t=(1-\gamma_t)h_{t-1}+\gamma_t c_t$"
    else:
        c.text(6.65, 6.15, "Direct transfer", 12, color=BLUE, ha="left")
        labels = ((INPUT_Y, r"$h_{t-1}$"), (OUTPUT_Y, r"$H_t=h_t$"), (8.7, r"$H_{t+1}$"))
        formula = r"$h_t=H_t$"

    if kind != "vanilla-loop":
        c.text(XS[-1] + 0.17, 5.98, weight, 12, color=TEAL, ha="left")
        c.text(XS[-1] + CARRY_OFFSET - 0.16, 5.98, kept, 12, color=BLUE, ha="right")
    for y, label in labels:
        c.text(0.10, y, label, 13, ha="left")
    is_ours = kind == "invariant-loop"
    c.text(
        6.65,
        6.96,
        "Common Q/K/V input"
        if is_ours
        else ("Next-layer input" if kind == "highway" else "Next-loop input"),
        11.5,
        color=BLUE,
        ha="left",
    )
    c.text(
        6.65,
        7.2,
        r"$\mathrm{LN}_1(h_t+e)$" if is_ours else (r"$h_\ell$" if kind == "highway" else r"$h_t$"),
        11.5,
        color=GRAY,
        ha="left",
    )
    c.arrow([(6.4, 7.2), (5.45, 7.2)], GRAY, width=0.8, head=False)
    c.text(
        4.7,
        9.57,
        "One position highlighted. Same rule at every position."
        if kind == "highway"
        else "One token highlighted. Same rule at every token.",
        11,
        color=GRAY,
    )
    c.text(4.7, 9.94, formula, 13, color=INK)
    c.save()
    return c


def write_manifest(diagrams):
    source_names = [
        "paper/figures/loop-residual-comparison-v2/build_figures.py",
        "paper/figures/loop-residual-comparison-v2/references.json",
        "src/llm_memory_editability/loop_native_handoff.py",
        "src/llm_memory_editability/self_decode_alignment.py",
        "configs/loop-invariant-poisson-terminal-v1.json",
    ]
    traces = []
    for c in diagrams:
        # Local token lanes are disjoint. The three carries cannot cross.
        for left, right in zip(c.carries[:-1], c.carries[1:], strict=True):
            assert max(p[0] for p in left["points"]) < min(p[0] for p in right["points"])
        traces.append({"artifact_id": c.kind, "carry_paths": c.carries, "paths": c.paths})
    manifest = {
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names
        },
        "diagram": {
            "language": "English",
            "orientation": "vertical",
            "canvas_inches": CANVAS,
            "token_xs": XS,
            "highlighted_token": 3,
            "carry_routing": "three disjoint local channels on the left",
            "transform_calls": 2,
            "weight_sharing": {"vanilla-loop": True, "highway": False, "invariant-loop": True},
            "one_figure_per_file": True,
            "highway_gate": "vector sigmoid(W_T,ell^T h_(ell-1)+b_T,ell), direct layer input",
            "invariant_gate": "scalar normalized weight from five statistics and retained mass",
            "invariant_normalization": "w=exp(clip(MLP(phi),-8,8)); gamma=w/(beta*D_previous+w)",
            "omitted_auxiliary_state": ["N", "D carry edge", "previous p carry edge"],
            "mass_node": "beta*D_previous, the scalar historical mass entering the normalizer",
        },
        "figure_table_trace": traces,
        "checks": {
            "text_inside_canvas": "passed",
            "carry_lanes_disjoint": "passed",
            "model_executed": False,
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    diagrams = [render(kind) for kind in ("vanilla-loop", "highway", "invariant-loop")]
    write_manifest(diagrams)
    print("Rendered three separate vertical figures; all three carry lanes are disjoint.")


if __name__ == "__main__":
    main()
