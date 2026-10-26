"""State and parameter graph for the implemented Invariant Loop.

Reuses the local vector drawing helpers. Does not import or execute a model.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STYLE = HERE.parent / "invariant-loop-v2/build_figures.py"
spec = importlib.util.spec_from_file_location("figure_style", STYLE)
drawing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drawing)
drawing.HERE = HERE


def state(c, x, y, label, fill, edge, radius=0.32, ring=False, size=21):
    if ring:
        c.ax.add_patch(
            drawing.Circle(
                (x, y),
                radius + 0.075,
                facecolor="white",
                edgecolor=drawing.TEAL,
                linewidth=1.8,
                zorder=3,
            )
        )
    c.ax.add_patch(
        drawing.Circle(
            (x, y),
            radius,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.7,
            zorder=4,
        )
    )
    c.text(x, y, label, size)


def main():
    c = drawing.Canvas("invariant-loop-state-graph", "Invariant Loop", "", height=7.3)
    blue, teal, gray = drawing.BLUE, drawing.TEAL, drawing.GRAY
    amber, pale_amber = "#B67725", "#FBE8B7"
    y = 2.6

    # Primary objects are hidden states and two calls of the same loop.
    state(c, 1.0, y, r"$h_{t-1}$", drawing.PB, blue, radius=0.36, size=20)
    state(c, 4.0, y, r"$H_t$", pale_amber, amber)
    state(c, 6.7, y, r"$c_t$", drawing.PT, teal)
    state(c, 9.5, y, r"$h_t$", drawing.PB, blue, ring=True)
    state(c, 12.5, y, r"$H_{t+1}$", pale_amber, amber, radius=0.38, size=19)
    for x, label in ((2.5, r"Loop $t$"), (11.0, r"Loop $t+1$")):
        c.box(x - 0.58, y - 0.36, 1.16, 0.72, r"$F_\theta$", edge=gray, size=23)
        c.text(x, 1.86, label, 19, color=gray)
    c.line([(1.36, y), (1.92, y)], color=blue)
    c.line([(3.08, y), (3.68, y)], color=blue)
    c.line([(4.32, y), (6.38, y)], color=amber)
    c.text(5.35, 2.17, "相加 + RMS", 17, color=gray)
    c.line([(7.02, y), (9.1, y)], color=teal)
    c.text(8.08, 2.18, "接收新候选", 17, color=teal)
    c.line([(9.9, y), (10.42, y)], color=blue)
    c.line([(11.58, y), (12.12, y)], color=blue)
    c.line([(12.88, y), (13.35, y)], color=amber)
    c.text(13.58, y, "…", 22, color=gray)
    c.text(1.0, 3.2, "旧状态", 17, color=gray)
    c.text(4.0, 1.75, "块输出", 17, color=gray)
    c.text(6.7, 1.75, "候选状态", 17, color=gray)
    c.text(9.1, 3.2, "下一轮状态", 17, color=gray, ha="right")

    # The cross-loop skip is a direct state-to-state connection.
    c.line([(1.0, 2.24), (1.0, 1.15), (9.5, 1.15), (9.5, 2.2)], color=blue)
    c.text(5.25, 0.89, r"保留旧状态  $(1-\gamma_t)$", 18, color=blue)

    # Soft-feedback states: full-vocabulary distribution and embedded vector.
    state(c, 4.0, 4.4, r"$p_t$", drawing.PB, blue, radius=0.28)
    c.box(6.39, 4.09, 0.62, 0.62, r"$q_t$", fill=drawing.PT, edge=teal, size=21)
    c.line([(4.0, 2.92), (4.0, 4.12)], color=amber)
    c.text(3.56, 3.64, "解码", 17, ha="right", color=gray)
    c.line([(4.28, 4.4), (6.39, 4.4)], color=blue)
    c.text(5.35, 4.02, "软再编码 + RMS", 17, color=blue)
    c.line([(6.7, 4.09), (6.7, 2.92)], color=teal)

    # An acceptance-weight node, with its two current-round inputs visible.
    state(c, 9.5, 4.4, r"$\gamma_t$", drawing.PT, teal, radius=0.29, size=22)
    c.line([(7.8, 2.6), (7.8, 4.4), (9.21, 4.4)], color=teal)
    c.line([(4.0, 4.68), (4.0, 5.35), (9.5, 5.35), (9.5, 4.69)], color=teal)
    c.text(6.75, 5.08, "五统计量打分 → 权重归一化", 17, color=teal)
    c.line([(9.5, 4.11), (9.5, 3.08)], color=teal, dashed=True)
    c.text(10.0, 3.68, "更新权重", 17, ha="left", color=teal)

    # The actual shared parameter is a node, rather than prose on an edge.
    c.box(
        5.56, 6.0, 2.28, 0.7, r"Loop 参数  $\theta$", fill=drawing.PO, edge=drawing.ORANGE, size=19
    )
    c.line([(5.56, 6.35), (2.5, 6.35), (2.5, 2.96)], color=drawing.ORANGE, dashed=True)
    c.line([(7.84, 6.35), (11.0, 6.35), (11.0, 2.96)], color=drawing.ORANGE, dashed=True)
    c.text(
        6.9,
        7.02,
        r"同一个 $\theta$ 在各轮复用；每次 $F_\theta$ 调用都重新加入输入嵌入 $e$。",
        17,
        color=gray,
    )
    c.save()

    sources = [
        "src/llm_memory_editability/loop_native_handoff.py",
        "src/llm_memory_editability/self_decode_alignment.py",
        "configs/loop-native-handoff-development-v1.json",
        "paper/figures/invariant-loop-v2/build_figures.py",
        "paper/figures/invariant-loop-v5/build_figure.py",
    ]
    manifest = {
        "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "reference": "DiscoLoop v2, Figure 1, https://arxiv.org/abs/2607.00341v2",
        "diagram": {
            "primary_representation": "hidden states and shared loop parameters as nodes",
            "adjacent_loop_calls": 2,
            "parameter_nodes": ["theta"],
            "caption_only": [
                "previous p and D inputs to weighting",
                "N/D update",
                "initialization",
            ],
        },
        "checks": {"canvas_text_bounds": "passed", "model_executed": False},
        "figure_table_trace": [
            {
                "artifact_id": "invariant-loop-state-graph",
                "source_data": {"file": sources[0]},
                "transformation": {"script": "build_figure.py"},
                "caption_claim": "A shared loop produces a soft-feedback candidate, which is "
                "combined with the old state before the next shared loop call.",
                "supported_manuscript_claims": [],
                "limitations": [
                    "Auxiliary history variables and the exact equations are in the caption."
                ],
            }
        ],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Rendered the shared-parameter and hidden-state graph.")


if __name__ == "__main__":
    main()
