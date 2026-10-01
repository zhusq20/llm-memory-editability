"""Independent artifact audit and descriptive paired development summaries."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest


def rows_for(root):
    for path in sorted(Path(root).glob("**/metrics.json")):
        records = json.loads(path.read_text())
        for index, record in enumerate(records):
            yield path, index, record


def solver_terminated_early(record, requested_steps=1024):
    # Frozen v1 increments the idle rejection counter while other batch members
    # finish. A completed trajectory can therefore have a stale raw failure flag.
    return bool(record["failed"] and record["accepted"] < requested_steps)


def validate_batch(path):
    directory = path.parent
    receipt = json.loads((directory / "complete.json").read_text())
    for name, sha in receipt["files"].items():
        assert digest(directory / name) == sha, (directory, name)
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
    steps = np.load(directory / "steps.npz")["values"]
    records = json.loads(path.read_text())
    assert steps.shape[1] == len(records) == state["weights"].shape[0]
    assert np.isfinite(steps).all() and torch.isfinite(state["weights"]).all()
    for j, r in enumerate(records):
        assert int(steps[-1, j, 7]) == r["accepted"] == int(state["age"][j])
        assert int(steps[:, j, 6].sum()) == r["accepted"]
        sets = r["task"]["sets"]
        names = ["E", "R", "V", "U"] + (["D"] if "D" in sets else [])
        ids = sum((sets[k] for k in names), [])
        assert len(ids) == len(set(ids)) == len(r["task"]["labels"])
        old = np.array(r["task"]["old_labels"])
        new = np.array(r["task"]["labels"])
        assert np.array_equal(old[sets["R"]], new[sets["R"]])
        assert np.array_equal(old[sets["U"]], new[sets["U"]])
        assert np.all(old[sets["E"], 0] != new[sets["E"], 0])
        if "-hard-" in r["arm"]:
            budget = r["config"]["level"]
            tol = 1e-10 if r["arm"].startswith("repr") and budget == 0 else 1e-8 + 1e-5 * budget
            assert np.max(steps[:, j, 3]) <= budget + tol + 1e-12, (
                path,
                j,
                np.max(steps[:, j, 3]),
                budget,
            )
    return len(records)


def summarize(root, destination, expected=None):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    detailed = []
    curves = []
    count = 0
    paths = set()
    groups = defaultdict(list)
    for path, index, r in rows_for(root):
        if path not in paths:
            count += validate_batch(path)
            paths.add(path)
        meta = r["task"]["metadata"]
        timeline = r["timeline"]
        last = timeline[-1]
        sets = last["sets"]
        selected = min(timeline, key=lambda m: (m["sets"]["V"]["broken"], m["sets"]["E"]["nll"]))
        stable = next(
            (
                timeline[i]["accepted"]
                for i in range(1, len(timeline))
                if timeline[i]["joint"]
                and timeline[i - 1]["joint"]
                and timeline[i]["accepted"] > timeline[i - 1]["accepted"]
            ),
            None,
        )
        df = sets.get("D_focal")
        dc = sets.get("D_heldout")
        row = dict(
            path=str(path.relative_to(ROOT)),
            index=index,
            arm=r["arm"],
            world=meta.get("world", -1),
            seed=meta["seed"],
            kind=meta["kind"],
            person=meta.get("person"),
            train_arm=meta.get("train_arm", "original"),
            accepted=r["accepted"],
            solver_failed=solver_terminated_early(r),
            raw_failure_flag=r["failed"],
            E_joint=int(last["e_joint"]),
            joint=int(last["joint"]),
            E_NLL=sets["E"]["nll"],
            R_broken=sets["R"]["broken"],
            V_broken=sets["V"]["broken"],
            U_broken=sets["U"]["broken"],
            U_known=sets["U"]["old_known"],
            local_broken=sets["local"]["broken"],
            local_correct=sets["local"]["correct"],
            local_n=sets["local"]["n"],
            stable_success_step=stable,
            selected_step=selected["accepted"],
            selected_joint=int(selected["joint"]),
            D_focal_correct=df["correct"] if df else None,
            D_heldout_correct=dc["correct"] if dc else None,
            D_heldout_n=dc["n"] if dc else None,
            E_local_D_joint=int(
                last["e_joint"]
                and sets["local"]["correct"] == sets["local"]["n"]
                and df["correct"] == df["n"]
            )
            if df
            else None,
        )
        detailed.append(row)
        groups[(row["train_arm"], r["arm"], meta["kind"])].append(row)
        for m in timeline:
            curves.append(
                dict(
                    arm=r["arm"],
                    kind=meta["kind"],
                    train_arm=row["train_arm"],
                    world=row["world"],
                    seed=row["seed"],
                    person=row["person"],
                    attempt=m["attempt"],
                    accepted=m["accepted"],
                    E_joint=int(m["e_joint"]),
                    joint=int(m["joint"]),
                    U_broken=m["sets"]["U"]["broken"],
                    E_NLL=m["sets"]["E"]["nll"],
                )
            )
    if expected is not None:
        assert count == expected, (count, expected)
    for name, rows in [("cases.csv", detailed), ("curves.csv", curves)]:
        with (destination / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = []
    for (train_arm, arm, kind), rows in sorted(groups.items()):
        row = dict(
            train_arm=train_arm,
            arm=arm,
            kind=kind,
            n=len(rows),
            E_joint=sum(x["E_joint"] for x in rows),
            joint=sum(x["joint"] for x in rows),
            selected_joint=sum(x["selected_joint"] for x in rows),
            U_broken=sum(x["U_broken"] for x in rows),
            U_known=sum(x["U_known"] for x in rows),
            local_broken=sum(x["local_broken"] for x in rows),
            solver_failed=sum(x["solver_failed"] for x in rows),
            raw_failure_flags=sum(x["raw_failure_flag"] for x in rows),
            accepted_min=min(x["accepted"] for x in rows),
            accepted_max=max(x["accepted"] for x in rows),
        )
        if rows[0]["D_focal_correct"] is not None:
            row.update(
                D_focal=sum(x["D_focal_correct"] for x in rows),
                D_heldout_correct=sum(x["D_heldout_correct"] for x in rows),
                D_heldout_n=sum(x["D_heldout_n"] for x in rows),
                E_local_D_joint=sum(x["E_local_D_joint"] for x in rows),
            )
        summary.append(row)
    write_json(destination / "summary.json", summary)
    write_json(
        destination / "audit.json",
        dict(
            batches=len(paths),
            trajectories=count,
            all_receipts_valid=True,
            info_disjoint=True,
            parent_relative_hard_constraints_verified=True,
            steps_and_states_agree=True,
            expected_trajectories=expected,
            interpretation="Queries are nested within parent/world; not independent replicates.",
            status_correction=(
                "Raw v1 failure counters also advance for completed, idle batch members. "
                "Early solver failure requires accepted < 1024; raw flags retained in cases.csv."
            ),
        ),
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("minimal", "cross", "formation"))
    args = parser.parse_args()
    if args.stage == "minimal":
        summarize(
            ROOT / "results/bios-direction-v1/main",
            ROOT / "docs/development-artifacts/direction-v1",
            144,
        )
    elif args.stage == "cross":
        summarize(
            ROOT / "results/bios-direction-cross-v1/main",
            ROOT / "docs/development-artifacts/direction-cross-v1",
            192,
        )
    else:
        summarize(
            ROOT / "results/bios-direction-formation-v1",
            ROOT / "docs/development-artifacts/direction-formation-v1",
            1344,
        )


if __name__ == "__main__":
    main()
