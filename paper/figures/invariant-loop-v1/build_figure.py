"""Draw the implemented Invariant Loop as a paper architecture diagram.

Rebuild both languages without loading a model or touching experiment artifacts:
    uv run --no-project --with 'matplotlib>=3.9,<4' python \
        paper/figures/invariant-loop-v1/build_figure.py

The coordinates describe a diagram, not measured model representations. All
scientific connections come from LoopNativeHandoffGPT and its inherited _round.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = ROOT / "configs/loop-native-handoff-development-v1.json"
SOURCES = (
    "src/llm_memory_editability/loop_native_handoff.py",
    "src/llm_memory_editability/self_decode_alignment.py",
    "src/llm_memory_editability/supervised_alignment.py",
    "src/llm_memory_editability/grokking_reproduction.py",
    "scripts/run_loop_native_handoff.py",
    "configs/loop-native-handoff-development-v1.json",
)

INK = "#253747"
MUTED = "#526474"
LINE = "#394D5C"
BLUE = "#0077BB"
TEAL = "#008578"
ORANGE = "#BE632D"
PALE_BLUE = "#EAF4FA"
PALE_TEAL = "#E8F5F1"
PALE_ORANGE = "#FBEEDF"
PALE_YELLOW = "#FAF6DA"
PALE_GRAY = "#F5F7F9"
LIGHT = "#CFD8DF"

LABELS = {
    "en": {
        "title": "Invariant Loop",
        "subtitle": (
            "Shared causal Transformer, vocabulary feedback, and invariant history weighting"
        ),
        "a": "(a) Recurrent architecture",
        "b": "(b) Vocabulary feedback and candidate",
        "c": "(c) Invariant handoff",
        "tokens": "Input tokens",
        "embedding": "Token embedding",
        "position": "Learned\nposition",
        "emb_drop": "Embedding dropout",
        "initialize": "Initialize carry",
        "round": "ONE ROUND",
        "block": "Shared GPT-2 block",
        "attention": "Masked multi-head\nself-attention",
        "attn_drop": "Residual dropout",
        "ffn": "FFN + dropout\nGELU_new",
        "candidate": "Vocabulary feedback\nCandidate RMS",
        "handoff": "Invariant handoff\nHistory aggregation",
        "head": r"$\mathrm{LN}_f$ + tied Linear $E$",
        "logits": "Answer / EOS logits",
        "carry": "Carry state; repeat R rounds",
        "decode_ln": "LayerNorm",
        "decode_linear": "Tied Linear",
        "softmax": "Softmax (full vocab)",
        "reencode": "Soft re-encoding",
        "rms": "RMS norm",
        "candidate_rms": "Candidate RMS norm",
        "feature_title": "Five scalar features",
        "entropy": "Entropy",
        "max": "Peak probability",
        "agreement": "Decode agreement",
        "change": "Relative change",
        "cosine": "State cosine",
        "mlp": "Shared score MLP",
        "weight": "Positive weight",
        "aggregate": "Normalized history",
        "old": "Candidate + previous history",
        "beta": "learned scalar",
        "legend_content": "Hidden states / vocabulary distributions",
        "legend_control": "Scalar features / weighting",
        "footer": (
            "Shared block: d = 768, 12 heads. All positions share modules. "
            "Dashed arrows: scalar control."
        ),
    },
    "zh": {
        "title": "Invariant Loop 架构",
        "subtitle": "共享因果 Transformer、词表反馈与不变统计量驱动的历史聚合",
        "a": "(a) 整体循环架构",
        "b": "(b) 词表反馈与候选生成",
        "c": "(c) 不变统计量交接",
        "tokens": "输入 token",
        "embedding": "词嵌入",
        "position": "可学习\n位置嵌入",
        "emb_drop": "Embedding dropout",
        "initialize": "初始化携带状态",
        "round": "一轮计算",
        "block": "共享 GPT-2 块",
        "attention": "带因果掩码的\n多头自注意力",
        "attn_drop": "残差 dropout",
        "ffn": "前馈网络\nGELU_new + dropout",
        "candidate": "词表解码再编码\n+ 候选 RMS 归一化",
        "handoff": "不变统计量交接\n归一化历史聚合",
        "head": r"$\mathrm{LN}_f$ + 绑定词表头 $E$",
        "logits": "答案 / EOS logits",
        "carry": "携带状态；共享参数循环 R 轮",
        "decode_ln": "LayerNorm",
        "decode_linear": "绑定词表头",
        "softmax": "全词表 Softmax",
        "reencode": "软再编码",
        "rms": "RMS 归一化",
        "candidate_rms": "候选 RMS 归一化",
        "feature_title": "五个标量统计量",
        "entropy": "预测熵",
        "max": "最大概率",
        "agreement": "相邻解码一致度",
        "change": "相对状态变化",
        "cosine": "状态余弦",
        "mlp": "共享打分 MLP",
        "weight": "正权重",
        "aggregate": "归一化历史聚合",
        "old": "本轮候选 + 上轮历史",
        "beta": "可学习标量",
        "legend_content": "隐藏状态 / 词表分布",
        "legend_control": "标量统计量 / 权重控制",
        "footer": "共享主干：1 个块、宽度 768、12 个头。虚线表示标量控制；所有位置使用同一套模块。",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_contract() -> dict:
    """Read config and source structure; never import torch or experiment runners."""
    config = json.loads(CONFIG.read_text())
    module = ast.parse((ROOT / SOURCES[0]).read_text())
    features = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FEATURES" for t in node.targets)
    )
    cls = next(
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "LoopNativeHandoffGPT"
    )
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert {"aggregate_round", "initial_state", "features", "score", "run"} <= methods
    assert features == ("entropy", "max_probability", "agreement", "relative_change", "cosine")
    assert config["arms"]["invariant"]["handoff"] == "invariant"
    assert config["training"]["unique_layers"] == 1
    assert config["model"]["n_embd"] == 768
    assert config["model"]["n_head"] == 12
    assert config["handoff"]["score_hidden"] == 16
    assert config["handoff"]["score_clip"] == 8.0
    return {
        "features_in_code_order": features,
        "unique_blocks": config["training"]["unique_layers"],
        "width": config["model"]["n_embd"],
        "heads": config["model"]["n_head"],
        "score_hidden": config["handoff"]["score_hidden"],
        "score_clip": config["handoff"]["score_clip"],
        "initial_beta": config["handoff"]["initial_beta"],
        "dropout": {k: config["model"][k] for k in ("embd_pdrop", "attn_pdrop", "resid_pdrop")},
        "source_sha256": {name: digest(ROOT / name) for name in SOURCES},
    }


class Diagram:
    def __init__(self, language: str):
        self.language = language
        self.labels = LABELS[language]
        if language == "zh":
            font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
            if font.exists():
                font_manager.fontManager.addfont(str(font))
            family = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
        else:
            family = ["DejaVu Sans"]
        plt.rcParams.update(
            {
                "font.family": family,
                "font.size": 17,
                "mathtext.fontset": "dejavusans",
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "axes.unicode_minus": False,
            }
        )
        self.fig = plt.figure(figsize=(13.8, 11.4), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1), xlim=(0, 18), ylim=(0, 14.87))
        self.ax.axis("off")
        self.box_texts = []

    def text(self, x, y, value, size=17, color=INK, ha="center", weight="normal", **kwargs):
        return self.ax.text(
            x,
            y,
            value,
            ha=ha,
            va="center",
            fontsize=size,
            color=color,
            weight=weight,
            linespacing=1.1,
            zorder=8,
            **kwargs,
        )

    def box(self, x, y, width, height, label="", fill="white", color=LINE, size=17):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=1.3,
            edgecolor=color,
            facecolor=fill,
            zorder=3,
        )
        self.ax.add_patch(patch)
        if label:
            item = self.text(x + width / 2, y + height / 2, label, size=size)
            self.box_texts.append((patch, item))
        return patch

    def arrow(self, points, color=LINE, dashed=False, width=1.5, head=True):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        if head:
            patch = FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=width,
                color=color,
                zorder=5,
                linestyle=(0, (4, 3)) if dashed else "-",
            )
        else:
            patch = PathPatch(
                path,
                fill=False,
                edgecolor=color,
                linewidth=width,
                zorder=4,
                linestyle=(0, (4, 3)) if dashed else "-",
            )
        self.ax.add_patch(patch)

    def dot(self, x, y, color=LINE):
        self.ax.add_patch(Circle((x, y), 0.04, color=color, zorder=7))

    def plus(self, x, y):
        self.ax.add_patch(Circle((x, y), 0.17, facecolor="white", edgecolor=LINE, lw=1.3, zorder=6))
        self.text(x, y, "+", size=19)

    def frame(self, x, y, width, height, fill="white", color=LIGHT):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=0.16",
                facecolor=fill,
                edgecolor=color,
                linewidth=1.2,
                zorder=0,
            )
        )

    def architecture(self):
        labels = self.labels
        x = 2.65
        self.text(0.45, 13.77, labels["a"], 20, ha="left", weight="bold")
        self.text(x, 0.98, labels["tokens"])
        self.arrow([(x, 1.17), (x, 1.42)])
        self.box(1.15, 1.42, 3.0, 0.52, labels["embedding"] + r" $E$", PALE_ORANGE)
        self.arrow([(x, 1.94), (x, 2.13)])
        self.plus(x, 2.3)
        self.box(4.25, 1.42, 1.15, 0.78, labels["position"], PALE_ORANGE, size=16.2)
        self.arrow([(4.82, 2.2), (4.82, 2.3), (2.82, 2.3)])
        self.arrow([(x, 2.47), (x, 2.7)])
        self.box(1.15, 2.7, 3.0, 0.44, labels["emb_drop"], PALE_GRAY, size=16.2)
        self.arrow([(x, 3.14), (x, 3.47)])
        self.text(3.0, 3.3, r"$e$", 19)
        self.dot(x, 3.3)
        self.arrow([(x, 3.3), (0.75, 3.3), (0.75, 4.48), (2.48, 4.48)])
        self.box(1.15, 3.47, 3.0, 0.55, labels["initialize"], PALE_GRAY)
        self.arrow([(x, 4.02), (x, 4.31)])
        self.frame(0.48, 4.15, 4.6, 7.84, PALE_GRAY)
        self.plus(x, 4.48)
        self.text(1.33, 4.67, r"$e$", 17)
        self.text(3.5, 4.23, r"$h_{t-1}$", 17)
        self.arrow([(x, 4.65), (x, 5.04)])
        self.frame(0.98, 4.88, 3.65, 5.12, "white", BLUE)
        self.text(0.78, 7.5, labels["block"] + r" $F_\theta$", 16.4, color=BLUE, rotation=90)
        self.box(1.32, 5.04, 2.66, 0.42, r"LayerNorm$_1$", PALE_YELLOW)
        self.arrow([(x, 5.46), (x, 5.72)])
        self.box(1.12, 5.72, 3.06, 0.79, labels["attention"], PALE_ORANGE, size=16.3)
        self.arrow([(x, 6.51), (x, 6.72)])
        self.box(1.32, 6.72, 2.66, 0.4, labels["attn_drop"], PALE_GRAY, size=16.2)
        self.arrow([(x, 7.12), (x, 7.29)])
        self.plus(x, 7.46)
        self.dot(x, 4.78)
        self.arrow([(x, 4.78), (4.3, 4.78), (4.3, 7.46), (2.82, 7.46)])
        self.arrow([(x, 7.63), (x, 7.84)])
        self.box(1.32, 7.84, 2.66, 0.42, r"LayerNorm$_2$", PALE_YELLOW)
        self.arrow([(x, 8.26), (x, 8.48)])
        self.box(1.12, 8.48, 3.06, 0.7, labels["ffn"], PALE_BLUE, size=16.3)
        self.plus(x, 9.7)
        self.arrow([(x, 9.18), (x, 9.53)])
        self.dot(x, 7.72)
        self.arrow([(x, 7.72), (4.3, 7.72), (4.3, 9.7), (2.82, 9.7)])
        self.arrow([(x, 9.87), (x, 10.12)])
        self.text(4.75, 9.97, r"$H_t$", 18)
        self.box(0.85, 10.12, 3.6, 0.75, labels["candidate"], PALE_BLUE, BLUE, 16.4)
        self.text(4.6, 10.45, "(b)", 16.2, color=BLUE)
        self.arrow([(x, 10.87), (x, 11.05)])
        self.text(4.75, 10.9, r"$c_t$", 18)
        self.box(0.85, 11.05, 3.6, 0.73, labels["handoff"], PALE_TEAL, TEAL, 16.4)
        self.text(4.6, 11.33, "(c)", 16.2, color=TEAL)
        self.arrow([(x, 11.78), (x, 12.38)])
        self.dot(x, 12.09)
        self.arrow([(x, 12.09), (5.48, 12.09), (5.48, 4.48), (2.82, 4.48)], BLUE, width=1.8)
        self.text(5.23, 8.05, labels["carry"], 16.2, color=BLUE, rotation=90)
        self.text(3.25, 12.15, r"$h_R$", 18)
        self.box(0.92, 12.38, 3.46, 0.48, labels["head"], PALE_YELLOW, size=16.2)
        self.arrow([(x, 12.86), (x, 13.12)])
        self.text(x, 13.33, labels["logits"], 17)

    def feedback(self):
        labels = self.labels
        self.text(6.13, 13.77, labels["b"], 20, ha="left", weight="bold")
        self.frame(6.03, 8.5, 11.54, 4.9, "white", LIGHT)
        left, right = 8.38, 14.42
        self.text(left, 9.12, r"$H_t$", 21)
        self.dot(left, 9.45)
        self.arrow([(left, 9.35), (left, 9.75)], BLUE)
        self.box(6.98, 9.75, 2.8, 0.52, labels["decode_ln"] + r" $\mathrm{LN}_f$", PALE_YELLOW)
        self.arrow([(left, 10.27), (left, 10.61)], BLUE)
        self.box(6.98, 10.61, 2.8, 0.58, labels["decode_linear"] + r" $E$", PALE_ORANGE)
        self.arrow([(left, 11.19), (left, 11.52)], BLUE)
        self.box(6.58, 11.52, 3.6, 0.6, labels["softmax"], PALE_BLUE, size=16.2)
        self.arrow([(left, 12.12), (left, 12.69), (right, 12.69), (right, 12.16)], BLUE)
        self.text(11.38, 12.97, r"$p_t=\mathrm{softmax}(E\,\mathrm{LN}_f(H_t))$", 18, color=BLUE)
        self.box(12.47, 11.54, 3.9, 0.62, labels["reencode"] + r" $E^\top p_t$", PALE_ORANGE)
        self.arrow([(right, 11.54), (right, 11.19)], BLUE)
        self.box(12.77, 10.61, 3.3, 0.58, labels["rms"], PALE_YELLOW)
        self.text(15.05, 10.4, r"$q_t$", 20, color=BLUE)
        self.arrow([(right, 10.61), (right, 10.27)], BLUE)
        self.plus(right, 10.1)
        self.arrow([(left, 9.45), (11.35, 9.45), (11.35, 10.1), (14.25, 10.1)], BLUE)
        self.text(11.22, 9.13, r"$H_t$", 18, color=BLUE)
        self.arrow([(right, 9.93), (right, 9.65)], BLUE)
        self.box(12.37, 9.1, 4.1, 0.55, labels["candidate_rms"], PALE_YELLOW)
        self.arrow([(right, 9.1), (right, 8.79)], BLUE)
        self.text(14.42, 8.67, r"$c_t=\mathrm{RMS}(H_t+q_t)$", 18, color=BLUE)

    def handoff(self):
        labels = self.labels
        self.text(6.13, 8.0, labels["c"], 20, ha="left", weight="bold")
        self.frame(6.03, 0.98, 11.54, 6.68, "white", LIGHT)
        self.text(9.14, 7.24, r"$c_t,\ h_{t-1},\ p_t,\ p_{t-1}$", 19, color=BLUE)
        self.arrow([(9.14, 7.0), (9.14, 6.73)], BLUE)
        self.box(6.43, 3.2, 5.42, 3.53, fill=PALE_TEAL, color=TEAL)
        self.text(9.14, 6.4, labels["feature_title"] + r" $\phi_t$", 18, weight="bold", color=TEAL)
        rows = (
            ("entropy", r"$\mathcal{H}(p_t)/\log |V|$"),
            ("max", r"$\max_v p_t(v)$"),
            ("agreement", r"$p_t^\top p_{t-1}$"),
            ("change", r"$\frac{\|c_t-h_{t-1}\|}{\|h_{t-1}\|+\epsilon}$"),
            ("cosine", r"$\cos(c_t,h_{t-1})$"),
        )
        for i, (key, formula) in enumerate(rows):
            y = 5.88 - 0.55 * i
            self.text(6.66, y, labels[key], 16.2, ha="left")
            self.text(10.35, y, formula, 17)
        self.arrow([(11.85, 5.7), (12.16, 5.7), (12.16, 6.29), (12.65, 6.29)], TEAL, True)
        self.box(12.65, 5.83, 4.5, 0.91, fill=PALE_TEAL, color=TEAL)
        self.text(14.9, 6.48, labels["mlp"], 18, weight="bold", color=TEAL)
        self.text(14.9, 6.08, r"$5\ \to\ 16\ (\mathrm{GELU})\ \to\ 1$", 18)
        self.arrow([(14.9, 5.83), (14.9, 5.46)], TEAL, True)
        self.text(15.45, 5.65, r"$s_t$", 19, color=TEAL)
        self.box(12.65, 4.64, 4.5, 0.82, fill=PALE_TEAL, color=TEAL)
        self.text(14.9, 5.23, labels["weight"], 16.4, color=TEAL)
        self.text(14.9, 4.87, r"$w_t=\exp(\mathrm{clip}(s_t,-8,8))$", 17)
        self.arrow([(14.9, 4.64), (14.9, 4.32)], TEAL, True)
        self.box(12.65, 1.72, 4.5, 2.6, fill=PALE_TEAL, color=TEAL)
        self.text(14.9, 4.0, labels["aggregate"], 18, weight="bold", color=TEAL)
        self.text(14.9, 3.48, r"$N_t=\beta N_{t-1}+w_t c_t$", 19)
        self.text(14.9, 2.97, r"$D_t=\beta D_{t-1}+w_t$", 19)
        self.text(14.9, 2.48, r"$h_t=N_t/D_t$", 20)
        self.text(14.9, 1.99, r"$\beta=\sigma(b)$" + "  " + labels["beta"], 16.2)
        self.box(6.43, 1.98, 5.42, 0.97, fill=PALE_BLUE, color=BLUE)
        self.text(9.14, 2.68, labels["old"], 17, color=BLUE)
        self.text(9.14, 2.27, r"$c_t,\ N_{t-1},\ D_{t-1}$", 20, color=BLUE)
        self.arrow([(11.85, 2.46), (12.65, 2.46)], BLUE)
        self.arrow([(14.9, 1.72), (14.9, 1.41)], BLUE)
        self.text(14.9, 1.18, r"$S_t=(h_t,N_t,D_t,p_t)$", 18, color=BLUE)
        self.text(9.14, 1.26, r"$S_0=(e,e,1,p_0)$", 18, color=BLUE)

    def draw(self) -> dict:
        labels = self.labels
        self.text(0.45, 14.47, labels["title"], 27, ha="left", weight="bold")
        self.text(17.54, 14.42, labels["subtitle"], 16.2, ha="right", color=MUTED)
        self.architecture()
        self.feedback()
        self.handoff()
        self.arrow([(0.5, 0.54), (1.03, 0.54)], BLUE)
        self.text(1.15, 0.54, labels["legend_content"], 16.2, ha="left", color=MUTED)
        self.arrow([(8.45, 0.54), (8.98, 0.54)], TEAL, True)
        self.text(9.1, 0.54, labels["legend_control"], 16.2, ha="left", color=MUTED)
        self.text(0.5, 0.22, labels["footer"], 16.2, ha="left", color=MUTED)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        outside, overflow = [], []
        for item in self.ax.texts:
            b = item.get_window_extent(renderer)
            if b.x0 < 0 or b.y0 < 0 or b.x1 > self.fig.bbox.width or b.y1 > self.fig.bbox.height:
                outside.append(item.get_text())
        for patch, item in self.box_texts:
            pb, tb = patch.get_window_extent(renderer), item.get_window_extent(renderer)
            if tb.x0 < pb.x0 - 2 or tb.y0 < pb.y0 - 2 or tb.x1 > pb.x1 + 2 or tb.y1 > pb.y1 + 2:
                overflow.append(
                    {
                        "text": item.get_text(),
                        "text_size": (tb.width, tb.height),
                        "box_size": (pb.width, pb.height),
                    }
                )
        if outside or overflow:
            raise ValueError({"outside_canvas": outside, "outside_module": overflow})
        stem = HERE / ("invariant-loop" if self.language == "en" else "invariant-loop-zh")
        for ext in ("svg", "pdf", "png"):
            kwargs = {"dpi": 300} if ext == "png" else {}
            self.fig.savefig(stem.with_suffix("." + ext), facecolor="white", **kwargs)
        plt.close(self.fig)
        return {
            "language": self.language,
            "files": [
                str(stem.with_suffix("." + ext).relative_to(ROOT)) for ext in ("pdf", "svg", "png")
            ],
            "text_within_canvas": True,
            "module_labels_within_boxes": True,
            "minimum_font_points_at_6_9_inch_width": 8.1,
            "raster_dpi": 300,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "zh", "all"), default="all")
    args = parser.parse_args()
    contract = source_contract()
    languages = ("en", "zh") if args.language == "all" else (args.language,)
    outputs = [Diagram(language).draw() for language in languages]
    manifest = {
        "figure": "Invariant Loop architecture",
        "version": "2026-10-24-v1",
        "scope": "LoopNativeHandoffGPT(handoff='invariant'); inference and one recurrent round",
        "new_model_execution": False,
        "contract": contract,
        "builder_sha256": digest(Path(__file__)),
        "outputs": outputs,
        "figure_table_trace": [
            {
                "artifact_id": "invariant-loop-architecture",
                "source_data": {
                    "files": list(SOURCES),
                    "kind": "implementation and frozen recipe config",
                },
                "transformation": {
                    "script": str(Path(__file__).relative_to(ROOT)),
                    "sha256": digest(Path(__file__)),
                },
                "caption_claim": (
                    "The diagram depicts the implemented shared pre-LN causal block, "
                    "tied vocabulary feedback, five-feature scorer, and exact N/D streaming carry."
                ),
                "supported_manuscript_claims": [],
                "limitations": [
                    "Architecture only; arrows encode no measured strength or performance.",
                    (
                        "Invariant describes score inputs under matched vocabulary relabeling / "
                        "state rotation, not distribution invariance of the whole model."
                    ),
                    (
                        "Training segment sampling, segment-boundary detach, and parameter EMA "
                        "are described in README rather than drawn as inference modules."
                    ),
                ],
            }
        ],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for output in outputs:
        print("\n".join(output["files"]))


if __name__ == "__main__":
    main()
