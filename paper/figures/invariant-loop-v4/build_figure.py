"""Draw adjacent iterations as five state nodes, without model execution."""

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


def node(canvas, x, y, color, merged=False):
    if merged:
        canvas.ax.add_patch(
            drawing.Circle(
                (x, y), 0.39, facecolor="white", edgecolor=drawing.TEAL, linewidth=2.2, zorder=3
            )
        )
    canvas.ax.add_patch(
        drawing.Circle((x, y), 0.29, facecolor=color, edgecolor=color, linewidth=1.5, zorder=4)
    )


def main():
    c = drawing.Canvas("invariant-loop-nodes", "Invariant Loop", "", height=6.15)
    blue, orange, teal, gray = drawing.BLUE, "#D58B32", drawing.TEAL, drawing.GRAY
    c.text(1.1, 1.4, "旧状态", 21)
    c.text(6.8, 1.4, "合并后的状态", 21)
    c.text(12.5, 1.4, "合并后的状态", 21)
    for x, merged in ((1.1, False), (6.8, True), (12.5, True)):
        node(c, x, 2.15, blue, merged)
    for x in (3.95, 9.65):
        node(c, x, 4.2, orange)
        c.text(x, 4.83, "新候选", 21, color=orange)

    # Each triangle is the same update: retain the old state, compute a candidate,
    # and merge both into a single state that is reused by the next iteration.
    for offset in (0.0, 5.7):
        c.line([(1.4 + offset, 2.15), (6.4 + offset, 2.15)], color=blue, width=2.3)
        c.text(3.95 + offset, 1.86, "保留旧状态", 18, color=blue)
        c.line([(1.28 + offset, 2.4), (3.7 + offset, 4.03)], color=gray, width=1.9)
        c.line([(4.2 + offset, 4.03), (6.52 + offset, 2.43)], color=orange, width=2.3)
        c.text(5.8 + offset, 3.48, "加权合并", 18, color=teal)

    c.text(1.83, 3.73, "Transformer\n＋软词表反馈", 17, color=gray)
    c.text(7.53, 3.73, "同一 Transformer\n＋软词表反馈", 17, color=gray)
    c.text(3.95, 5.32, "这一轮", 18, color=gray)
    c.text(9.65, 5.32, "下一轮", 18, color=gray)
    c.text(6.9, 5.85, "合并比例由五统计量打分与历史累计权重共同确定。", 17, color=teal)
    c.save()

    sources = [
        "src/llm_memory_editability/loop_native_handoff.py",
        "paper/figures/invariant-loop-v2/build_figures.py",
        "paper/figures/invariant-loop-v4/build_figure.py",
    ]
    manifest = {
        "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "diagram": {
            "aggregate_nodes": 3,
            "candidate_nodes": 2,
            "adjacent_rounds": 2,
            "boxes": 0,
            "arrow_width_meaning": "Visual grouping only; no measured weights.",
            "caption_only": ["input reinjection", "N/D recursion", "p carry", "initialization"],
        },
        "checks": {"canvas_text_bounds": "passed", "model_executed": False},
        "figure_table_trace": [
            {
                "artifact_id": "invariant-loop-nodes",
                "source_data": {"file": sources[0]},
                "transformation": {"script": "build_figure.py"},
                "caption_claim": "Each round combines its old aggregate and new candidate; "
                "the result is the next round's old state.",
                "supported_manuscript_claims": [],
                "limitations": [
                    "Conceptual state flow; auxiliary recurrent variables are in the caption."
                ],
            }
        ],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Rendered the five-node, two-round architecture diagram.")


if __name__ == "__main__":
    main()
