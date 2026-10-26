"""Render two separate architecture figures from the verified implementation.

Only matplotlib is required. This script never imports the model or runs training.
Run from any directory with a writable MPLCONFIGDIR.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INK = "#243746"
GRAY = "#627482"
BLUE = "#0077BB"
TEAL = "#008577"
ORANGE = "#B86128"
LIGHT = "#D4DEE5"
BG = "#F3F6F8"
PB = "#EAF4FA"
PT = "#E6F4EF"
PO = "#FFF2E5"


class Canvas:
    def __init__(self, name, title, subtitle, height=11.4):
        font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        plt.rcParams.update(
            {
                "font.family": "Noto Sans CJK JP",
                "mathtext.fontset": "dejavusans",
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
                "font.size": 17,
            }
        )
        self.name = name
        self.fig = plt.figure(figsize=(13.8, height), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, 13.8), ylim=(height, 0))
        self.ax.axis("off")
        self.text(0.45, 0.45, title, 28, ha="left", weight="bold")
        self.text(0.45, 0.99, subtitle, 17, ha="left", color=GRAY)

    def text(self, x, y, text, size=18, color=INK, ha="center", **kwargs):
        return self.ax.text(
            x,
            y,
            text,
            fontsize=size,
            color=color,
            ha=ha,
            va="center",
            linespacing=1.4,
            zorder=5,
            **kwargs,
        )

    def box(self, x, y, w, h, text="", fill=BG, edge=LIGHT, size=18, zorder=2):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.015,rounding_size=0.09",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.6,
                zorder=zorder,
            )
        )
        if text:
            self.text(x + w / 2, y + h / 2, text, size)

    def line(self, pts, color=BLUE, dashed=False, arrow=True, width=1.9):
        for a, b in zip(pts[:-2], pts[1:-1], strict=True):
            self.ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=color,
                lw=width,
                ls="--" if dashed else "-",
                zorder=1,
            )
        a, b = pts[-2:]
        self.ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>" if arrow else "-",
                mutation_scale=15,
                linewidth=width,
                color=color,
                linestyle="--" if dashed else "-",
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )

    def plus(self, x, y):
        self.ax.add_patch(Circle((x, y), 0.16, facecolor="white", edgecolor=BLUE, lw=1.8, zorder=4))
        self.text(x, y - 0.005, "+", 22)

    def save(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        frame = self.fig.bbox
        clipped = []
        for text in self.ax.texts:
            ext = text.get_window_extent(renderer)
            if not (
                frame.x0 <= ext.x0
                and ext.x1 <= frame.x1
                and frame.y0 <= ext.y0
                and ext.y1 <= frame.y1
            ):
                clipped.append(text.get_text())
        if clipped:
            raise ValueError(f"Text outside canvas: {clipped}")
        for ext in ("svg", "pdf", "png"):
            self.fig.savefig(HERE / f"{self.name}.{ext}", dpi=300)
        self.fig.savefig(HERE / f"{self.name}-preview.png", dpi=130)
        plt.close(self.fig)


def architecture():
    c = Canvas(
        "01-architecture",
        "Invariant Loop｜架构总览",
        "Transformer 抽象为共享块；展开词表反馈与状态更新。箭头按计算顺序阅读。",
    )
    # One round: main hidden-state path, top to bottom.
    c.text(3.05, 1.65, r"上一轮状态  $h_{t-1}$", 20)
    c.line([(3.05, 1.98), (3.05, 2.39)])
    c.box(0.45, 2.14, 1.75, 0.82, "输入表示  $e$\n词嵌入 + 位置", fill=PO, size=17)
    c.line([(2.2, 2.55), (2.88, 2.55)])
    c.plus(3.05, 2.55)
    c.line([(3.05, 2.71), (3.05, 3.12)])
    c.box(
        1.5, 3.12, 3.1, 0.96, "共享 Transformer  $F_\u03b8$\n1 个因果 GPT-2 块", fill=BG, edge=GRAY
    )
    c.line([(3.05, 4.08), (3.05, 5.73)])
    c.text(3.47, 4.55, r"$H_t$", 23)
    c.text(1.5, 4.55, "本轮块输出", 17, ha="right", color=GRAY)

    # DiscoLoop-style soft vocabulary feedback.
    c.box(5.2, 3.12, 7.95, 2.65, fill="#FAFCFD", edge=LIGHT, zorder=0)
    c.text(5.45, 3.44, "词表反馈：把预测分布转回 embedding", 19, ha="left", color=BLUE)
    c.line([(3.05, 4.38), (5.55, 4.38)])
    c.box(
        5.55,
        3.88,
        2.85,
        0.99,
        "共享输出头 + Softmax\n" + r"$p_t=\mathrm{softmax}(E\,\mathrm{LN}_f(H_t))$",
        fill=PB,
        edge=BLUE,
        size=16.5,
    )
    c.line([(8.4, 4.38), (9.15, 4.38)])
    c.text(8.76, 4.06, r"$p_t$", 20, color=BLUE)
    c.box(
        9.15,
        3.88,
        3.55,
        0.99,
        "概率加权的词嵌入\n" + r"$q_t=\mathrm{RMS}(E^\top p_t)$",
        fill=PB,
        edge=BLUE,
        size=18,
    )
    c.text(6.99, 5.32, "全词表软反馈", 17, color=GRAY)
    c.line([(10.92, 4.87), (10.92, 5.89), (3.22, 5.89)])
    c.text(5.15, 5.65, r"词表反馈  $q_t$", 18, color=BLUE)
    c.plus(3.05, 5.89)
    c.line([(3.05, 6.05), (3.05, 6.4)])
    c.box(
        1.5,
        6.4,
        3.1,
        0.83,
        "候选归一化\n" + r"$c_t=\mathrm{RMS}(H_t+q_t)$",
        fill=PB,
        edge=BLUE,
        size=18,
    )
    c.line([(3.05, 7.23), (3.05, 8.46)])
    c.text(2.72, 7.8, r"$c_t$", 22, ha="right")

    # Our scorer and the inherited normalized recurrent aggregation.
    c.line([(4.6, 6.81), (5.65, 6.81), (5.65, 7.54), (6.15, 7.54)], color=TEAL)
    c.line([(8.78, 4.38), (8.78, 5.75)], color=TEAL, dashed=True, arrow=False)
    c.line([(8.78, 6.02), (8.78, 6.91)], color=TEAL, dashed=True)
    c.text(9.08, 6.36, r"$p_t$", 20, ha="left", color=TEAL)
    c.box(
        6.15,
        6.91,
        4.7,
        1.25,
        "五统计量打分（详见图 2）\n熵、峰值、分布内积、相对变化、余弦\n"
        + r"$5\ \to\ 16\ \to\ 1\quad\Longrightarrow\quad w_t>0$",
        fill=PT,
        edge=TEAL,
        size=17,
    )
    c.text(11.05, 7.24, "上一轮保存", 17, ha="left", color=GRAY)
    c.text(11.05, 7.69, r"$h_{t-1},\ p_{t-1}$", 19, ha="left", color=TEAL)
    c.line([(11.1, 7.54), (10.85, 7.54)], color=TEAL, dashed=True)
    c.line([(8.5, 8.16), (8.5, 8.67)], color=TEAL, dashed=True)
    c.text(9.1, 8.43, r"$w_t$", 18, color=TEAL)
    c.box(
        1.05,
        8.46,
        4.0,
        1.1,
        "历史与本轮候选加权平均\n" + r"$h_t=(1-\gamma_t)h_{t-1}+\gamma_t c_t$",
        fill=PT,
        edge=TEAL,
        size=18,
    )
    c.box(
        6.15,
        8.67,
        4.7,
        1.04,
        r"接收比例  $\gamma_t=\frac{w_t}{\beta D_{t-1}+w_t}$",
        fill=PT,
        edge=TEAL,
        size=21,
    )
    c.line([(6.15, 9.15), (5.05, 9.15)], color=TEAL, dashed=True)
    c.text(12.0, 8.95, "历史累计权重\n" + r"$D_{t-1}$", 17, color=GRAY)
    c.line([(11.45, 9.23), (11.45, 9.54), (10.85, 9.54)], color=TEAL, dashed=True)
    c.line([(3.05, 9.56), (3.05, 10.02)])
    c.text(3.05, 10.25, r"更新后状态  $h_t$", 21)
    c.line([(3.05, 9.82), (0.22, 9.82), (0.22, 1.65), (1.48, 1.65)], width=2.2)
    c.text(0.53, 6.0, "送入下一轮", 17, rotation=90, color=BLUE)
    c.line([(4.48, 10.25), (6.15, 10.25)])
    c.text(5.35, 9.99, r"$t=R$", 17, color=GRAY)
    c.box(
        6.15,
        10.0,
        6.99,
        0.55,
        r"共享输出头 $E\,\mathrm{LN}_f(h_R)$  →  答案 / EOS logits",
        fill=PO,
        size=17,
    )
    c.text(
        0.45,
        11.0,
        r"蓝色实线：表示流。绿色虚线：权重计算。完整跨轮状态 $(h,N,D,p)$ 在图 2 展开。",
        17,
        ha="left",
        color=GRAY,
    )
    c.save()


def handoff():
    c = Canvas(
        "02-state-handoff",
        "Invariant Loop｜跨轮状态与权重计算",
        "同一个 token 位置贯穿相邻轮次；每一轮都调用同一套参数。",
        height=12.5,
    )
    # An explicitly unrolled carry, with names instead of an ambiguous feedback arrow.
    c.box(0.45, 1.5, 3.05, 2.15, fill=PB, edge=BLUE)
    c.text(1.975, 1.83, r"第 $t$ 轮结束", 21, weight="bold")
    rows = [
        (2.32, r"$h_t$", "更新后状态"),
        (2.83, r"$N_t,D_t$", "加权和、累计权重"),
        (3.34, r"$p_t$", "本轮块输出的预测分布"),
    ]
    destinations = [
        "输入 Transformer；参与打分与聚合",
        "继续更新加权和与累计权重",
        "与下一轮预测分布计算内积",
    ]
    for (y, symbol, label), dest in zip(rows, destinations, strict=True):
        c.text(0.68, y, symbol, 20, ha="left", color=BLUE)
        c.text(4.05, y - 0.18, label, 17, ha="left", color=GRAY)
        c.line([(3.5, y + 0.08), (7.5, y + 0.08)])
        c.text(7.75, y + 0.08, dest, 17, ha="left")
    c.text(9.9, 1.83, r"第 $t+1$ 轮使用", 21, weight="bold")
    c.text(
        0.5,
        4.04,
        r"$h_t=N_t/D_t$。代码保留四元组；$p_t$ 来自块输出 $H_t$，不从聚合状态 $h_t$ 重新计算。",
        17,
        ha="left",
        color=GRAY,
    )

    # Five source-bound, position-wise scalar statistics.
    c.box(0.45, 4.55, 7.05, 3.63, fill=PT, edge=TEAL)
    c.text(0.73, 4.91, "本轮打分只读取五个统计量", 21, ha="left", color=TEAL, weight="bold")
    features = [
        ("预测熵", r"$-\sum_v p_t(v)\log p_t(v)\,/\,\log |V|$"),
        ("最大预测概率", r"$\max_v\ p_t(v)$"),
        ("与上一轮的分布内积", r"$p_t^\top p_{t-1}$"),
        ("候选相对旧状态的变化", r"$\|c_t-h_{t-1}\|\,/\,(\|h_{t-1}\|+\epsilon)$"),
        ("候选与旧状态的余弦", r"$\cos(c_t,h_{t-1})$"),
    ]
    for i, (label, formula) in enumerate(features):
        y = 5.48 + 0.51 * i
        c.text(0.73, y, label, 17, ha="left")
        c.text(4.0, y, formula, 17, ha="left")
    c.box(
        8.1,
        4.55,
        5.22,
        1.12,
        "共享打分网络\n" + r"$s_t=\mathrm{MLP}_{5\to16\to1}(\phi_t)$  ·  GELU",
        fill=PT,
        edge=TEAL,
        size=19,
    )
    c.line([(7.5, 5.13), (8.1, 5.13)], color=TEAL, dashed=True)
    c.line([(10.71, 5.67), (10.71, 6.04)], color=TEAL, dashed=True)
    c.box(
        8.1,
        6.04,
        5.22,
        0.81,
        r"本轮权重  $w_t=\exp(\mathrm{clip}(s_t,-8,8))$",
        fill=PT,
        edge=TEAL,
        size=18,
    )
    c.line(
        [(8.1, 6.45), (7.8, 6.45), (7.8, 8.42), (8.8, 8.42), (8.8, 8.68)], color=TEAL, dashed=True
    )
    c.box(
        8.25,
        7.21,
        5.07,
        0.97,
        "另一个共享参数：历史衰减\n" + r"$\beta=\sigma(b)$，可学习标量，初值 0.55",
        fill=PO,
        edge=ORANGE,
        size=17,
    )

    # Exact implementation update, with semantically named terms.
    c.box(0.45, 8.68, 12.87, 1.66, fill=BG, edge=GRAY)
    c.text(0.75, 9.01, "代码中的状态更新", 20, ha="left", weight="bold")
    c.text(2.75, 9.67, r"$N_t=\beta N_{t-1}+w_t c_t$", 23)
    c.text(7.12, 9.67, r"$D_t=\beta D_{t-1}+w_t$", 23)
    c.text(11.2, 9.67, r"$h_t=N_t/D_t$", 23)
    c.line([(10.71, 8.18), (10.71, 8.68)], color=TEAL, dashed=True)
    c.line([(6.88, 10.34), (6.88, 10.69)])
    c.box(
        0.45,
        10.69,
        12.87,
        0.73,
        r"保存 $(h_t,N_t,D_t,p_t)$  →  下一轮；终轮仅从 $h_R$ 读出答案",
        fill=PB,
        edge=BLUE,
        size=21,
    )
    c.text(
        0.5,
        11.87,
        r"初始化：$h_0=N_0=e,\ D_0=1,\ p_0=\mathrm{softmax}(E\,\mathrm{LN}_f(e))$"
        r"；首轮块输入为 $2e$。",
        17,
        ha="left",
        color=GRAY,
    )
    c.text(
        0.5,
        12.25,
        "统计量与聚合逐 token 计算；跨 token 的信息交换由共享 Transformer 的因果注意力完成。",
        17,
        ha="left",
        color=GRAY,
    )
    c.save()


def main():
    architecture()
    handoff()
    sources = [
        "src/llm_memory_editability/loop_native_handoff.py",
        "src/llm_memory_editability/self_decode_alignment.py",
        "src/llm_memory_editability/supervised_alignment.py",
        "configs/loop-native-handoff-development-v1.json",
    ]
    manifest = {
        "implementation": 'LoopNativeHandoffGPT(handoff="invariant")',
        "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "papers": ["https://arxiv.org/abs/2607.00341v2", "https://arxiv.org/abs/2610.11570v2"],
        "figure_table_trace": [
            {
                "artifact_id": name,
                "source_data": {"file": sources[0]},
                "transformation": {"script": "build_figures.py"},
                "caption_claim": claim,
                "supported_manuscript_claims": [],
                "limitations": [
                    "Architecture diagram; does not establish performance "
                    "or whole-model invariance."
                ],
            }
            for name, claim in [
                (
                    "01-architecture",
                    "Soft vocabulary feedback and statistic-based weighting "
                    "update the recurrent state.",
                ),
                (
                    "02-state-handoff",
                    "The implementation carries h, N, D "
                    "and the previous block-output distribution.",
                ),
            ]
        ],
        "checks": {
            "canvas_text_bounds": "passed",
            "model_executed": False,
            "training_updates": 0,
            "png_dpi": 300,
            "minimum_font_pt_at_6_9_in_width": 8.25,
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Rendered two independent figures: PNG, SVG, PDF and previews.")


if __name__ == "__main__":
    main()
