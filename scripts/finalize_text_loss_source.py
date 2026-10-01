"""Audit reciprocal parameter swaps and assemble completed experiment evidence."""

import csv
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.text_pretrain import sha

ROOT = Path("docs/development-artifacts/text-loss-source-v1")


def main():
    lock = json.loads((ROOT / "weight-analysis-execution-lock.json").read_text())
    for f, expected in lock["sources"].items():
        assert sha(f) == expected
    cfg = json.loads(Path("configs/text-loss-source-followup-v1.json").read_text())
    cells = []
    checks = 0
    for parent in sorted({v["parent"] for v in cfg["runs"].values()}):
        base = ROOT / "followup/weight-analysis" / Path(parent).name
        d = json.loads((base / "summary.json").read_text())
        raw = np.load(base / "predictions.npz")
        world = np.load(Path(parent) / "world.npz")
        assert list(d["sources"].values()) == [
            lock["sources"]["scripts/analyze_text_loss_weights.py"]
        ]
        names = {v["arm"]: name for name, v in cfg["runs"].items() if v["parent"] == parent}
        for arm in ("A", "N"):
            p = Path(cfg["output_root"]) / names[arm] / f"weights-{cfg['steps']:06d}.pt"
            assert d["checkpoints"][arm] == sha(p)
        grouped = [k for names in d["groups"].values() for k in names]
        assert len(grouped) == len(set(grouped))
        for row in d["records"]:
            prefix = row["recipient"] + "_" + row["group"] + "_"
            values = {}
            for split, score in row["metrics"].items():
                prefix_s = prefix + split + "_"
                target = raw[prefix_s + "target"]
                np.testing.assert_array_equal(target, world[split][:, -1])
                correct = (raw[prefix_s + "answer"] == target) & (
                    raw[prefix_s + "stops"] == [5, 1]
                ).all(-1)
                assert score["accuracy"] == float(correct.mean())
                assert (
                    abs(score["answer_probability"] - raw[prefix_s + "probability"].mean()) < 1e-12
                )
                values[split] = float(correct.mean())
                checks += len(target)
            cells.append(
                dict(
                    world=d["world"],
                    initialization=d["initialization"],
                    recipient=row["recipient"],
                    donor=row["donor"],
                    group=row["group"],
                    **values,
                )
            )
    summary = []
    for recipient in ("A", "N"):
        for group in ("self", "token", "attention", "mlp", "norm_position", "all"):
            selected = [r for r in cells if r["recipient"] == recipient and r["group"] == group]
            assert len(selected) == 6
            summary.append(
                dict(
                    recipient=recipient,
                    group=group,
                    **{
                        k: float(np.mean([r[k] for r in selected]))
                        for k in ("atomic", "background_train", "background_test", "target_test")
                    },
                )
            )
    with (ROOT / "followup/report/weight-cells.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    write_json(
        ROOT / "followup/report/weight-summary.json",
        dict(
            status="passed",
            pairs=6,
            parameter_state_conditions=72,
            prediction_rows_checked=checks,
            summary=summary,
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    groups = ["self", "token", "attention", "mlp", "norm_position", "all"]
    labels = ["Baseline", "Token/readout", "Attention", "MLP", "Norm/position", "All"]
    for ax, recipient in zip(axes, ["A", "N"], strict=True):
        vals = [
            next(
                r["target_test"] for r in summary if r["recipient"] == recipient and r["group"] == g
            )
            * 100
            for g in groups
        ]
        atomic = [
            next(r["atomic"] for r in summary if r["recipient"] == recipient and r["group"] == g)
            * 100
            for g in groups
        ]
        ax.bar(range(6), vals, color="#2563eb", label="Target composition")
        ax.plot(range(6), atomic, color="#0f9d76", marker="o", label="Atomic facts")
        ax.set(
            title=f"{recipient} receives {'N' if recipient == 'A' else 'A'} parameters",
            xticks=range(6),
            xticklabels=labels,
            ylabel="Full answer accuracy (%)",
            ylim=(0, 103),
        )
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(ROOT / "followup/report/weights.png", dpi=180)
    fig.savefig(ROOT / "followup/report/weights.pdf")
    plt.close(fig)
    reports = [
        json.loads((ROOT / stage / "report/summary.json").read_text())
        for stage in ("development", "followup")
    ]
    sources = [
        *Path("scripts").glob("*text_loss*.py"),
        Path("src/llm_memory_editability/text_loss_source.py"),
        Path("tests/test_text_loss_source.py"),
    ]
    runs = []
    for stage in ("development", "followup"):
        c = json.loads(Path(f"configs/text-loss-source-{stage}-v1.json").read_text())
        for name in c["runs"]:
            p = Path(c["output_root"]) / name / "complete.json"
            assert p.exists()
            runs.append(dict(name=name, stage=stage, complete_sha256=sha(p)))
    manifest = dict(
        status="complete",
        training_runs=len(runs),
        development=4,
        frozen_followup=24,
        source_worlds=3,
        initializations=2,
        steps_each=4096,
        total_steps=4096 * len(runs),
        learning_nodes=8 * len(runs),
        endpoint_reloads=sum(r["audit"]["runs"] for r in reports),
        prediction_rows_audited=sum(r["audit"]["raw_prediction_rows"] for r in reports),
        weight_swap_model_states=72,
        weight_swap_prediction_rows_audited=checks,
        related_tests=11,
        budget={k: sum(r["budget"][k] for r in reports) for k in reports[0]["budget"]},
        training_seconds=sum(r["training_seconds"] for r in reports),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            device="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            gpu=2,
        ),
        sources={str(p): sha(p) for p in sources},
        runs=runs,
        notes=[
            "Followup uses previously observed worlds, not new-world confirmation.",
            "No source or arm excluded for outcome. All endpoints kept.",
            "Training time sums concurrent process timers, not dedicated GPU wall time.",
            "Freeze-path error fixed before training; four missing-lock starts made zero updates.",
            "Weight-analysis formatting fixed before followup; both source locks retained.",
        ],
    )
    write_json(ROOT / "completion-manifest.json", manifest)
    print(
        json.dumps(dict(weight_summary=summary, checks=checks, total=manifest["budget"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
