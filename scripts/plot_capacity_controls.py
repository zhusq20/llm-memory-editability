"""Plot audited capacity-control endpoints; no selection of best checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root):
    data = json.loads((root / "summary.json").read_text())
    names = ["recall-calibration-long"] + [f"recall-load-h{h}" for h in (1024, 4096, 16384)]
    by_name = {r["name"]: r for r in data["runs"]}
    if not all(n in by_name for n in names):
        raise ValueError("The four original load checkpoints must all be audited")
    rows = [by_name[n] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, keys, title in (
        (
            axes[0],
            [
                ("atomic", "Atomic recall", "#1d4ed8"),
                ("II", "Native composition (II)", "#dc2626"),
                ("serial_II", "External self-bridge (II)", "#059669"),
            ],
            "Same weights, different recall protocols",
        ),
        (
            axes[1],
            [
                ("II", "Native composition (II)", "#dc2626"),
                ("serial_II", "External self-bridge (II)", "#059669"),
                ("wrong_bridge_II", "Wrong bridge (II)", "#6b7280"),
            ],
            "Fixed target queries + increasing background",
        ),
    ):
        panel = (
            rows
            if ax is axes[0]
            else [by_name["recall-calibration-long"]]
            + [by_name[f"recall-background-h{h}"] for h in (1024, 4096, 16384)]
        )
        for key, label, color in keys:
            ax.plot(
                [r["density"] for r in panel],
                [r[key] * 100 for r in panel],
                marker="o",
                label=label,
                color=color,
                linewidth=2,
            )
        ax.set(
            xscale="log",
            ylim=(-3, 105),
            xlabel="Required knowledge bits / parameter",
            ylabel="Exact-match accuracy (%)",
            title=title,
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "One development world; fixed endpoints; external decomposition adds computation",
        fontsize=11,
    )
    fig.savefig(root / "recall-comparison.png", dpi=180)
    fig.savefig(root / "recall-comparison.pdf")
    plt.close(fig)
    (root / "plot-provenance.json").write_text(
        json.dumps(
            {
                "source": str(root / "summary.json"),
                "source_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "audited": data["audited"],
                "selected": names,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    plot(parser.parse_args().root)
