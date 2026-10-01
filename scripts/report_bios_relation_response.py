"""Paired descriptive reporting, with no case-level significance claims."""

import csv
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest
from llm_memory_editability.bios_relation_response import CELLS, factorial_effects

ART = ROOT / "docs/development-artifacts/relation-response-v1"
OUT = ROOT / "results/bios-relation-response-v1"


def export(name, rows):
    with (ART / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cases, precases, receipts = [], [], {}
    for w, s in itertools.product((0, 1), repeat=2):
        directory = OUT / f"world-{w}-seed-{s}"
        analysis = directory / "analysis.json"
        prediction = directory / "prediction.json"
        data = json.loads(analysis.read_text())
        assert data["all_endpoint_generation_verified"]
        cases.extend(data["cases"])
        precases.extend(json.loads(prediction.read_text())["cases"])
        receipts[str(analysis.relative_to(ROOT))] = digest(analysis)
        receipts[str(prediction.relative_to(ROOT))] = digest(prediction)
        for fname in ("complete.json", "launch.json"):
            p = directory / "new" / fname
            receipts[str(p.relative_to(ROOT))] = digest(p)
    assert len(cases) == 96 and len(precases) == 24
    groups = defaultdict(dict)
    case_rows, curves, controls, intervals = [], [], [], []
    for c in cases:
        meta, final = c["metadata"], c["final"]
        keys = {k: meta[k] for k in ("world", "seed", "chain", "group", "person", "cell")}
        sets = final["sets"]
        margins = c["trajectory"][-1]["margins_ab"]
        row = dict(
            **keys,
            source=c["source"],
            accepted=c["accepted"],
            E=int(final["e_joint"]),
            D=sets["D_focal"]["correct"],
            D_heldout=sets["D_heldout"]["correct"],
            D_heldout_n=sets["D_heldout"]["n"],
            local_broken=sets["local"]["broken"],
            local_known=sets["local"]["old_known"],
            U_broken=sets["U"]["broken"],
            U_known=sets["U"]["old_known"],
            autonomous=int(c["two_step"]["autonomous"]),
            membership=int(c["two_step"]["membership_correct"]),
            predicts_actual=int(
                c["two_step"]["direct_value"] == meta["actual_target"]
                and c["two_step"]["direct_eos"]
            ),
            prediction=c["two_step"]["direct_value"],
            ended=c["two_step"]["direct_eos"],
            root_margin=margins[0],
            actual_margin=margins[1],
            D_margin=margins[2],
            root_response=c["final_geometry"]["root_response"],
            actual_response=c["final_geometry"]["actual_response"],
            unexplained_fraction=c["final_geometry"]["unexplained_D_gradient_fraction"],
            first_step_replay_difference=c["first_step_replay_max_difference"],
        )
        case_rows.append(row)
        group = tuple(meta[k] for k in ("world", "seed", "chain", "group", "person"))
        assert meta["cell"] not in groups[group]
        groups[group][meta["cell"]] = row
        for node in c["trajectory"]:
            curves.append(
                dict(
                    **keys,
                    step=node["step"],
                    root_margin=node["margins_ab"][0],
                    actual_margin=node["margins_ab"][1],
                    D_margin=node["margins_ab"][2],
                    **node["geometry"],
                )
            )
            if "interval" in node:
                v = node["interval"]
                intervals.append(
                    dict(
                        **keys,
                        step=node["step"],
                        next_step=v["next_step"],
                        predicted_D=v["predicted_ab"][2],
                        observed_D=v["observed_ab"][2],
                        delta_norm=v["delta_norm"],
                    )
                )
        for control in c["final_calibration"]:
            controls.append(
                dict(
                    **keys,
                    phase="post",
                    **{
                        k: v for k, v in control.items() if k not in ("predicted_ab", "observed_ab")
                    },
                    predicted_D=control["predicted_ab"][2],
                    observed_D=control["observed_ab"][2],
                )
            )
    pre_rows, prediction_rows = [], []
    for c in precases:
        meta = c["metadata"]
        keys = {k: meta[k] for k in ("world", "seed", "chain", "group", "person")}
        pre_rows.append(dict(**keys, **c["geometry"]))
        for control in c["calibration"]:
            controls.append(
                dict(
                    **keys,
                    cell="parent",
                    phase="pre",
                    **{
                        k: v for k, v in control.items() if k not in ("predicted_ab", "observed_ab")
                    },
                    predicted_D=control["predicted_ab"][2],
                    observed_D=control["observed_ab"][2],
                )
            )
        for first in c["first_step"]:
            prediction_rows.append(
                dict(
                    **keys,
                    cell=first["cell"],
                    prospective=first["cell"] in ("ba", "bb"),
                    predicted_D=first["predicted_ab"][2],
                    observed_D=first["observed_ab"][2],
                )
            )
    pair_rows = []
    for key, four in sorted(groups.items()):
        assert set(four) == set(CLS := CELLS)
        e = factorial_effects({cell: four[cell]["D_margin"] for cell in CLS})
        source_root = factorial_effects({cell: four[cell]["root_margin"] for cell in CLS})
        source_actual = factorial_effects({cell: four[cell]["actual_margin"] for cell in CLS})
        matrix = np.array(
            [
                [source_root["root"], source_root["actual"]],
                [source_actual["root"], source_actual["actual"]],
            ]
        )
        calibrated = np.array([e["root"], e["actual"]]) @ np.linalg.pinv(matrix, rcond=1e-10)
        pattern = (
            "root"
            if all(four[cell]["D"] for cell in CLS)
            else "actual"
            if all(four[cell]["predicts_actual"] for cell in CLS)
            else "mixed"
        )
        pair_rows.append(
            dict(
                **dict(zip(("world", "seed", "chain", "group", "person"), key, strict=True)),
                pattern=pattern,
                all_E=all(four[cell]["E"] for cell in CLS),
                all_local=all(four[cell]["local_broken"] == 0 for cell in CLS),
                all_U=all(four[cell]["U_broken"] == 0 for cell in CLS),
                **e,
                root_source_effect=source_root["root"],
                root_source_cross=source_root["actual"],
                actual_source_effect=source_actual["actual"],
                actual_source_cross=source_actual["root"],
                source_matrix_condition=float(np.linalg.cond(matrix)),
                calibrated_root=float(calibrated[0]),
                calibrated_actual=float(calibrated[1]),
            )
        )
    endpoints = []
    for cell in CELLS:
        rs = [r for r in case_rows if r["cell"] == cell]
        endpoints.append(
            dict(
                cell=cell,
                n=len(rs),
                E=sum(r["E"] for r in rs),
                D=sum(r["D"] for r in rs),
                D_heldout=sum(r["D_heldout"] for r in rs),
                D_heldout_n=sum(r["D_heldout_n"] for r in rs),
                E_local=sum(r["E"] and r["local_broken"] == 0 for r in rs),
                E_local_D=sum(r["E"] and r["local_broken"] == 0 and r["D"] for r in rs),
                autonomous=sum(r["autonomous"] for r in rs),
                actual_answer=sum(r["predicts_actual"] for r in rs),
                U_broken=sum(r["U_broken"] for r in rs),
                U_known=sum(r["U_known"] for r in rs),
            )
        )
    world_rows = []
    for w, s in itertools.product((0, 1), repeat=2):
        pairs = [r for r in pair_rows if r["world"] == w and r["seed"] == s]
        world_rows.append(
            dict(
                world=w,
                seed=s,
                n=len(pairs),
                root_patterns=sum(r["pattern"] == "root" for r in pairs),
                actual_patterns=sum(r["pattern"] == "actual" for r in pairs),
                root_effect=float(np.mean([r["root"] for r in pairs])),
                actual_effect=float(np.mean([r["actual"] for r in pairs])),
            )
        )
    nonzero_controls = [
        r for r in controls if abs(r["predicted_D"]) > 1e-6 and abs(r["observed_D"]) > 1e-6
    ]
    prospect = [r for r in prediction_rows if r["prospective"]]
    summary = dict(
        created=datetime.now(timezone.utc).isoformat(),
        n_cases=len(cases),
        n_pairs=len(pair_rows),
        new_edits=48,
        reused_edits=48,
        new_training=0,
        endpoints=endpoints,
        patterns=dict(Counter(r["pattern"] for r in pair_rows)),
        world_seed=world_rows,
        all_accepted_1024=all(r["accepted"] == 1024 for r in case_rows),
        pre=dict(
            root_response_mean=float(np.mean([r["root_response"] for r in pre_rows])),
            actual_response_mean=float(np.mean([r["actual_response"] for r in pre_rows])),
            actual_positive=sum(r["actual_response"] > 0 for r in pre_rows),
            actual_exceeds_root=sum(r["actual_response"] > r["root_response"] for r in pre_rows),
            unexplained_fraction_median=float(
                np.median([r["unexplained_D_gradient_fraction"] for r in pre_rows])
            ),
        ),
        source_controls=dict(
            n=len(controls),
            nonzero=len(nonzero_controls),
            sign_matches=sum(r["predicted_D"] * r["observed_D"] > 0 for r in nonzero_controls),
            other_contrast_drift_max=max(r["other_ac_bc_drift"] for r in controls),
            other_output_kl_max=max(r["other_output_kl"] for r in controls),
        ),
        prospective_first_step=dict(
            n=len(prospect),
            sign_matches=sum(r["predicted_D"] * r["observed_D"] > 0 for r in prospect),
            max_actual_replay_difference=max(
                r["first_step_replay_difference"] for r in case_rows if r["source"] == "new"
            ),
        ),
        finite_effects=dict(
            root_mean=float(np.mean([r["root"] for r in pair_rows])),
            actual_mean=float(np.mean([r["actual"] for r in pair_rows])),
            actual_exceeds_root=sum(r["actual"] > r["root"] for r in pair_rows),
            calibrated_root_mean=float(np.mean([r["calibrated_root"] for r in pair_rows])),
            calibrated_actual_mean=float(np.mean([r["calibrated_actual"] for r in pair_rows])),
        ),
        scope="Two previously studied development worlds; no independent significance inference.",
        verified_sources=receipts,
    )
    for name, rows in (
        ("cases.csv", case_rows),
        ("pairs.csv", pair_rows),
        ("curves.csv", curves),
        ("controls.csv", controls),
        ("pre-response.csv", pre_rows),
        ("first-step-predictions.csv", prediction_rows),
        ("intervals.csv", intervals),
        ("endpoints.csv", endpoints),
        ("world-seed.csv", world_rows),
    ):
        export(name, rows)
    write_json(ART / "summary.json", summary)
    plot(endpoints, pair_rows, controls)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def plot(endpoints, pairs, controls):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    matrix = np.array([r["D"] / r["n"] for r in endpoints]).reshape(2, 2)
    axes[0].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for i, row in enumerate(matrix):
        for j, x in enumerate(row):
            axes[0].text(
                j,
                i,
                f"{x:.1%}\n{endpoints[2 * i + j]['D']}/24",
                ha="center",
                va="center",
                color="white" if x > 0.5 else "black",
            )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Personal a", "Personal b"],
        yticks=[0, 1],
        yticklabels=["Root a", "Root b"],
        title="Composite accuracy: four factual states",
    )
    for world, color in ((0, "tab:blue"), (1, "tab:orange")):
        rows = [r for r in pairs if r["world"] == world]
        axes[1].scatter(
            [r["root"] for r in rows],
            [r["actual"] for r in rows],
            color=color,
            label=f"World {world}",
        )
    limit = max(abs(r[k]) for r in pairs for k in ("root", "actual")) * 1.1
    axes[1].plot([-limit, limit], [-limit, limit], "k--", lw=1)
    axes[1].set(
        xlabel="Root effect on composite margin",
        ylabel="Personal effect on composite margin",
        title="Paired finite-update response",
    )
    axes[1].legend()
    for name, color in (("root", "tab:blue"), ("actual", "tab:orange")):
        rows = [r for r in controls if r["control"] == name]
        axes[2].scatter(
            [r["predicted_D"] for r in rows],
            [r["observed_D"] for r in rows],
            s=8,
            alpha=0.45,
            color=color,
            label=name,
        )
    lim = max(abs(r[k]) for r in controls for k in ("predicted_D", "observed_D")) * 1.1
    axes[2].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    axes[2].set(
        xlabel="Jacobian prediction",
        ylabel="Observed composite margin change",
        title="Source-isolated small-step intervention",
    )
    axes[2].legend()
    for ext in ("png", "pdf"):
        fig.savefig(ART / f"four-cell-response.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
