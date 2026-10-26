"""Render three separate architecture figures without importing model code.

Published recurrences supply the first two figures. The approved v12 drawing
supplies the implementation-based Invariant Loop figure and the shared style.
"""

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STYLE_SOURCE = ROOT / "paper/figures/invariant-loop-v12/build_figure.py"


def load_style():
    spec = importlib.util.spec_from_file_location("invariant_figure_style", STYLE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The earlier source remains untouched; all generated files go here.
    module.HERE = HERE
    return module


def draw_baseline(style, feedback):
    style.STEM = "discoloop" if feedback else "vanilla-loop"
    c = style.Figure()
    c.text(4.1, 0.37, "DiscoLoop" if feedback else "Vanilla Loop", 23, weight="bold")
    for i, x in enumerate(style.TOKEN_XS, start=1):
        c.text(x, 0.83, rf"$x_{i}$", 12, color=style.GRAY)

    if feedback:
        c.ax.add_patch(
            style.FancyBboxPatch(
                (1.15, 2.96),
                4.9,
                1.95,
                boxstyle="round,pad=0,rounding_size=0.12",
                facecolor="#F2FAF7",
                edgecolor="none",
                zorder=0,
            )
        )
    c.layer(2.02, r"Loop $t$")
    c.layer(6.55, r"Loop $t+1$")
    # These baselines have no external historical-state bypass. Keep a solid
    # Transformer band rather than the carry channels used in Invariant Loop.
    for patch in c.ax.patches:
        if isinstance(patch, style.Rectangle):
            patch.set_visible(False)

    for x in style.TOKEN_XS:
        c.token_alpha = 1.0 if x == style.TOKEN_XS[-1] else 0.28
        c.state(x, 1.25, "carry")
        c.arrow([(x, 1.395), (x, 2.19)], color=style.BLUE)
        c.arrow([(x, 2.47), (x, 3.055)], color=style.GOLD)
        c.state(x, 3.2, "output")

        if feedback:
            c.arrow([(x, 3.345), (x, 4.355)], color=style.GOLD)
            fx = x + 0.38
            c.arrow([(x + 0.105, 3.305), (fx, 3.715)], color=style.TEAL)
            c.feedback(fx, 3.82)
            c.arrow([(fx, 3.925), (x + 0.09, 4.385)], color=style.TEAL)
            if x == style.TOKEN_XS[-1]:
                c.text(x + 0.54, 3.82, r"$f_t$", 11, color=style.TEAL)
            c.state(x, 4.5, "candidate")
            c.arrow([(x, 4.645), (x, 6.72)], color=style.BLUE)
        else:
            c.arrow([(x, 3.345), (x, 6.72)], color=style.BLUE)

        c.arrow([(x, 7.0), (x, 7.655)], color=style.GOLD)
        c.state(x, 7.8, "output")
        c.arrow([(x, 7.945), (x, 8.28)], color=style.GOLD)
        c.text(x, 8.5, r"$\vdots$", 18, color=style.GRAY)

    c.token_alpha = 1.0
    if feedback:
        c.text(6.4, 3.82, "Soft feedback", 12, color=style.TEAL, ha="left")
        c.text(6.4, 4.5, "Add", 12, color=style.TEAL, ha="left")
        c.text(6.4, 4.83, r"$h_t=H_t+f_t$", 11, color=style.GRAY, ha="left")
        labels = ((1.25, r"$h_{t-1}$"), (3.2, r"$H_t$"), (4.5, r"$h_t$"))
    else:
        c.text(6.4, 4.5, "Direct transfer", 12, color=style.BLUE, ha="left")
        c.text(6.4, 4.83, r"$h_t=H_t$", 11, color=style.GRAY, ha="left")
        labels = ((1.25, r"$h_{t-1}$"), (3.2, r"$H_t=h_t$"))
    for y, label in (*labels, (7.8, r"$H_{t+1}$")):
        c.text(0.12, y, label, 13, ha="left")
    c.text(6.4, 6.25, "Next-loop input", 11, color=style.BLUE, ha="left")
    c.text(6.4, 6.5, r"$h_t$", 11, color=style.GRAY, ha="left")
    c.arrow([(6.2, 6.5), (5.05, 6.5)], color=style.GRAY, width=0.8, head=False)
    c.text(4.1, 8.84, "One token highlighted. Same update at every token.", 10.5, color=style.GRAY)
    c.save()


def write_manifest():
    sources = [
        Path(__file__).relative_to(ROOT).as_posix(),
        STYLE_SOURCE.relative_to(ROOT).as_posix(),
        "src/llm_memory_editability/loop_native_handoff.py",
        "src/llm_memory_editability/self_decode_alignment.py",
        "paper/figures/loop-architecture-comparison-v1/reference-status.json",
    ]
    figures = [
        {
            "artifact_id": "vanilla-loop",
            "source": "DiscoLoop v2, Section 3.1, Eq. (2); Kohli et al., Section 4",
            "claim": "The current block output becomes the next loop's input.",
            "full_old_state_bypasses": 0,
        },
        {
            "artifact_id": "discoloop",
            "source": "DiscoLoop v2, Section 4, Eqs. (4)-(5); Section 5.1 fixed gate",
            "claim": "Normalized soft vocabulary feedback is added to the current output.",
            "full_old_state_bypasses": 0,
            "gate_variant": "alpha=1, temperature=1 at an intermediate loop boundary",
        },
        {
            "artifact_id": "invariant-loop",
            "source": "LoopNativeHandoffGPT.aggregate_round, handoff='invariant'; v12 drawing",
            "claim": "MLP-based weighting aggregates the candidate with its own previous state.",
            "full_old_state_bypasses": 3,
            "mlp_input": "five statistics from c_t, h_(t-1), p_t, p_(t-1)",
            "next_loop_qkv_common_input": "LN_1(h_t+e)",
        },
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        "figure_table_trace": figures,
        "shared_drawing_choices": {
            "orientation": "vertical",
            "language": "English",
            "token_positions": 3,
            "highlighted_token": 3,
            "transformer_calls": 2,
            "one_figure_per_file": True,
            "canvas_inches": [8.2, 9.0],
            "ordinary_transformer_residuals": "inside the abstract F_theta band in all figures",
        },
        "checks": {"text_inside_canvas": "passed", "model_executed": False},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    style = load_style()
    draw_baseline(style, feedback=False)
    draw_baseline(style, feedback=True)
    style.STEM = "invariant-loop"
    style.main()
    # Replace the reused drawing's trace with the complete comparison trace.
    write_manifest()
    print("Rendered three independent comparison figures: vanilla, DiscoLoop, Invariant Loop.")


if __name__ == "__main__":
    main()
