"""Compact implementation-based architecture figure; no model is imported."""

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


def main():
    c = drawing.Canvas("invariant-loop", "Invariant Loop", "", height=5.8)
    blue, teal, gray = drawing.BLUE, drawing.TEAL, drawing.GRAY

    # Two consecutive calls make the cross-loop state path explicit.
    c.text(0.74, 2.0, r"$h_{t-1}$", 21)
    c.text(0.74, 2.65, "旧状态", 17, color=gray)
    c.line([(0.74, 2.3), (1.49, 2.3)])
    c.plus(1.65, 2.3)
    c.text(1.65, 1.68, "$e$", 20, color=drawing.ORANGE)
    c.line([(1.65, 1.88), (1.65, 2.14)], color=drawing.ORANGE)
    c.line([(1.81, 2.3), (2.15, 2.3)])
    c.box(2.15, 1.9, 2.2, 0.8, r"Transformer $F_\theta$", edge=gray, size=18)
    c.line([(4.35, 2.3), (5.45, 2.3)])
    c.text(4.86, 1.95, r"$H_t$", 21)
    c.box(5.45, 1.9, 2.1, 0.8, "相加并归一化", fill=drawing.PB, edge=blue, size=18)
    c.line([(7.55, 2.3), (8.9, 2.3)])
    c.text(8.0, 1.94, r"$c_t$", 21)
    c.text(8.0, 2.66, "新候选", 17, color=gray)
    c.box(8.9, 1.9, 1.6, 0.8, "加权更新", fill=drawing.PT, edge=teal, size=18)
    c.line([(10.5, 2.3), (11.39, 2.3)])
    c.text(10.96, 1.94, r"$h_t$", 21)
    c.text(10.96, 2.66, "新状态", 17, color=gray)
    c.plus(11.55, 2.3)
    c.text(11.55, 1.68, "$e$", 20, color=drawing.ORANGE)
    c.line([(11.55, 1.88), (11.55, 2.14)], color=drawing.ORANGE)
    c.line([(11.71, 2.3), (12.0, 2.3)])
    c.box(12.0, 1.9, 1.2, 0.8, r"$F_\theta$", edge=gray, size=23)
    c.text(12.6, 1.48, "下一轮", 17, color=gray)
    c.line([(13.2, 2.3), (13.55, 2.3)])
    c.text(13.65, 2.3, "…", 19)

    # Retained old state visibly bypasses the entire candidate-generation path.
    c.line([(0.74, 2.3), (0.2, 2.3), (0.2, 1.1), (9.7, 1.1), (9.7, 1.9)], color=teal)
    c.text(6.9, 0.88, "保留旧状态", 17, color=teal)

    # The soft vocabulary channel is a single existing-method abstraction.
    c.line([(4.86, 2.3), (4.86, 3.9), (5.15, 3.9)])
    c.box(5.15, 3.5, 2.4, 0.8, "软解码 → 词嵌入", fill=drawing.PB, edge=blue, size=17)
    c.line([(6.5, 3.5), (6.5, 2.7)])
    c.text(6.8, 3.08, r"$q_t$", 20, color=blue)

    # Collapse feature extraction, score, and mass normalization to one module.
    c.line([(7.55, 3.9), (7.95, 3.9), (7.95, 4.4), (8.65, 4.4)], color=teal)
    c.text(8.15, 4.03, r"$p_t$", 18, color=teal)
    c.line([(8.55, 2.3), (8.55, 3.3), (8.95, 3.3), (8.95, 3.78)], color=teal)
    c.box(
        8.65,
        3.78,
        2.8,
        1.25,
        "当前 / 上一轮\n五统计量打分\n+ 权重归一化",
        fill=drawing.PT,
        edge=teal,
        size=17,
    )
    c.line([(10.45, 3.78), (10.45, 3.12), (9.7, 3.12), (9.7, 2.7)], color=teal, dashed=True)
    c.text(10.86, 3.28, r"$\gamma_t$", 21, color=teal)

    c.text(
        0.55,
        5.41,
        r"$F_\theta$：同一组参数    $e$：输入嵌入    终轮从 $h_R$ 读出答案",
        17,
        ha="left",
        color=gray,
    )
    c.save()
    sources = [
        "src/llm_memory_editability/loop_native_handoff.py",
        "configs/loop-native-handoff-development-v1.json",
        "paper/figures/invariant-loop-v2/build_figures.py",
        "paper/figures/invariant-loop-v3/build_figure.py",
    ]
    manifest = {
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        "reference": "https://arxiv.org/abs/2607.00341v2",
        "abstraction": {
            "feedback": "Decode, probability-weighted embedding, and feedback RMS.",
            "score": "Five statistics, MLP, score clipping, exp, and mass normalization.",
            "update": "Exact equivalent convex combination of old state and candidate.",
            "caption_only": ["N/D recurrence", "auxiliary p and D carry", "initialization"],
        },
        "checks": {"canvas_text_bounds": "passed", "model_executed": False},
        "figure_table_trace": [
            {
                "artifact_id": "invariant-loop",
                "source_data": {"file": sources[0]},
                "transformation": {"script": "build_figure.py"},
                "caption_claim": "The old state and soft-feedback candidate are combined "
                "using statistic-based weights between shared Transformer calls.",
                "supported_manuscript_claims": [],
                "limitations": [
                    "Auxiliary recurrent state and exact equations are in the caption."
                ],
            }
        ],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Compact figure rendered as SVG, PDF and PNG.")


if __name__ == "__main__":
    main()
