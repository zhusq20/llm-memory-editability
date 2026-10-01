"""Assemble development-only completion evidence; never read confirmation worlds."""

import hashlib
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/bios-mechanism-dev-v1"
OUTPUT = ROOT / "docs/development-artifacts/mechanism-v1/development-completion-audit.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    evidence, errors, source_checks = {}, [], {}

    def read(path):
        path = Path(path)
        contents = path.read_bytes()
        evidence[str(path.relative_to(ROOT))] = hashlib.sha256(contents).hexdigest()
        return json.loads(contents)

    stage_specs = [
        (
            "P0_original_rescoring",
            "p0/audit.json",
            0,
            0,
            {
                "models": 48,
                "learning_checkpoints": 288,
                "edit_cases": 384,
                "edit_checkpoints": 1536,
            },
        ),
        (
            "P0_autonomous_two_step",
            "p0/summary/two-step-audit.json",
            0,
            0,
            {"models": 48, "states": 528, "chain_files": 1056},
        ),
        (
            "P1_counterfactual_and_balance",
            "p1/audit.json",
            0,
            192,
            {
                "complete_models": 24,
                "complete_new_edits": 192,
                "complete_reference_edits": 96,
                "prediction_checkpoints_recomputed": 1152,
                "weights_and_optimizer_states_verified": 192,
            },
        ),
        (
            "P2_continuation",
            "p2/audit.json",
            36,
            144,
            {
                "learning_runs": 36,
                "learning_checkpoints": 144,
                "edit_cases": 144,
                "edit_checkpoints": 576,
            },
        ),
        (
            "P3_high_exception_learning",
            "p3-summary/audit.json",
            12,
            0,
            {"complete_pairs": 12, "learning_pairs": 72, "two_step_pairs": 24},
        ),
        (
            "P3_context_learning",
            "p3-context-summary/audit.json",
            12,
            0,
            {"complete_pairs": 12, "learning_pairs": 72, "two_step_pairs": 24},
        ),
        (
            "P3_matched_E39_v2",
            "p3-shortcut-edit-v2/summary/audit.json",
            0,
            96,
            {"models": 24, "cases": 96, "checkpoints": 384},
        ),
        (
            "H5_frozen_tied_input_output",
            "h5-readout-control/summary/audit.json",
            0,
            48,
            {
                "models": 24,
                "new_cases": 48,
                "reference_cases": 96,
                "all_checkpoints": 576,
                "frozen_readout_checks": 48,
            },
        ),
        (
            "P4_localization_discovery",
            "p4/discovery/world-0-audit.json",
            0,
            0,
            {"worlds_read": [0]},
        ),
        (
            "P4_localization_replication",
            "p4/replication/world-1-audit.json",
            0,
            0,
            {"worlds_read": [1], "available_model_chain_blocks": 24},
        ),
        (
            "P4_necessity_recovery_v2",
            "p4/necessity-v2-summary/audit.json",
            0,
            0,
            {
                "complete_models": 24,
                "model_chain_blocks": 48,
                "producer_seals_verified": True,
                "raw_value_and_EOS_recomputed": True,
            },
        ),
    ]
    stages = []
    for name, relative, learning, editing, expected in stage_specs:
        audit = read(RAW / relative)
        complete = (
            audit.get("complete") is True or audit.get("state", audit.get("status")) == "complete"
        )
        for field in ("errors", "missing", "missing_models", "incomplete_organization_blocks"):
            if audit.get(field):
                errors.append({"stage": name, "field": field, "value": audit[field]})
        for key, value in expected.items():
            if audit.get(key) != value:
                errors.append(
                    {"stage": name, "field": key, "expected": value, "observed": audit.get(key)}
                )
        if not complete:
            errors.append({"stage": name, "error": "Scientific stage audit incomplete"})
        stages.append(
            {
                "stage": name,
                "complete": complete,
                "new_learning": learning,
                "new_edits": editing,
                "audit": str((RAW / relative).relative_to(ROOT)),
                "verified_counts_and_flags": expected,
            }
        )
    weights_path = OUTPUT.parent / "p2/weight-audit.json"
    weights = read(weights_path)
    if not weights["complete"] or weights["checkpoint_files_verified"] != 468:
        errors.append({"stage": "P2_weights", "error": "Independent weight audit incomplete"})
    read(weights_path.with_name("weight-checksums.json"))
    read(weights_path.with_name("weight-run-checks.json"))

    def check_source(path, expected, origin):
        path = Path(path)
        if not path.is_file() or digest(path) != expected:
            errors.append(
                {"source": str(path), "origin": str(origin), "error": "Source lock mismatch"}
            )
        else:
            source_checks[str(path.relative_to(ROOT))] = expected

    # Recheck all run-level source maps, not merely one representative model.
    contracts = []
    for folder in (
        "p1",
        "p2",
        "p3",
        "p3-context",
        "p3-shortcut-edit-v2",
        "h5-readout-control",
        "p4/necessity-v2",
    ):
        root = (RAW / folder).resolve()
        for directory, _, names in os.walk(root, followlinks=True):
            contracts.extend(
                Path(directory) / name
                for name in names
                if name in ("launch-contract.json", "contract.json", "config.json")
            )
    for path in sorted(set(contracts)):
        value = json.loads(path.read_text())
        for name, checksum in value.get("sources", {}).items():
            if name == "runner":
                target = ROOT / "scripts/analyze_bios_mechanism_causal_v2.py"
            else:
                options = [
                    ROOT / name,
                    ROOT / "src/llm_memory_editability" / name,
                    ROOT / "scripts" / name,
                ]
                target = next((p for p in options if p.is_file()), options[0])
            check_source(target, checksum, path)
    explicit_sources = [
        ("p0/audit.json", "analysis_sources"),
        ("p3-context/launch-gate.json", "sources_sha256"),
        ("p3-context/launch-gate.json", "audits_sha256"),
    ]
    for relative, field in explicit_sources:
        payload = read(RAW / relative)
        for name, checksum in payload[field].items():
            check_source(ROOT / name, checksum, RAW / relative)
    runner_contracts = [
        ("p2", "run_bios_cross_continue.py", None),
        ("p3-context", "run_bios_context_control.py", "bios-context-control-v1.json"),
        ("p3-shortcut-edit-v2", "run_bios_shortcut_edit_v2.py", "bios-shortcut-edit-v1.json"),
        ("h5-readout-control", "run_bios_readout_control.py", "bios-readout-control-v1.json"),
    ]
    for folder, runner, config in runner_contracts:
        payload = read(RAW / folder / "launch-contract.json")
        check_source(ROOT / "scripts" / runner, payload["runner_sha256"], folder)
        if config:
            check_source(ROOT / "configs" / config, payload["config_sha256"], folder)
    read(RAW / "p4/candidate-lock.json")

    archives = []
    for path in sorted(RAW.glob("**/*.tar.json")):
        index = read(path)
        archive = Path(index["archive"])
        entries = index["entries"]
        consistent = (
            index.get("complete") is True
            and index.get("every_member_verified") is True
            and len(entries) == index["files"]
            and sum(row["bytes"] for row in entries.values()) == index["bytes"]
        )
        headers = {}
        with tarfile.open(archive, "r:") as stream:
            for member in stream:
                if member.isfile():
                    if member.name in headers:
                        raise ValueError(f"Duplicate archived member: {member.name}")
                    headers[member.name] = member.size
        if headers != {name: row["bytes"] for name, row in entries.items()}:
            consistent = False
        if not consistent:
            errors.append(
                {"archive": str(archive), "error": "Archive index/header verification failed"}
            )
        archives.append(
            {
                "index": str(path.relative_to(ROOT)),
                "archive": str(archive.relative_to(ROOT)),
                "index_complete": index.get("complete"),
                "stage_complete_declared": index.get("stage_complete"),
                "every_member_payload_previously_verified": index["every_member_verified"],
                "headers_and_index_rechecked_now": consistent,
                "files": index["files"],
                "payload_bytes": index["bytes"],
                "tar_bytes": archive.stat().st_size,
                "tar_mtime_ns": archive.stat().st_mtime_ns,
            }
        )
    if len(archives) != 9:
        errors.append(
            {
                "error": "Archive inventory differs from nine expected index files",
                "observed": len(archives),
            }
        )
    totals = {
        "new_learning": sum(s["new_learning"] for s in stages),
        "new_edits": sum(s["new_edits"] for s in stages),
    }
    if totals != {"new_learning": 60, "new_edits": 480}:
        errors.append({"error": "Scientific new-training count mismatch", "observed": totals})
    output = {
        "protocol": "v2.8-development-completion",
        "complete": not errors,
        "worlds": [0, 1],
        "valid_scientific_totals": totals,
        "stages": stages,
        "archives": archives,
        "p2_new_weight_audit": {
            k: weights[k]
            for k in (
                "complete",
                "learning_runs",
                "learning_snapshots",
                "learning_resumes",
                "edit_cases",
                "checkpoint_files_verified",
                "checkpoint_bytes_read_and_hashed",
            )
        },
        "source_locks": {
            "run_contracts_inspected": len(set(contracts)),
            "unique_current_sources_checked": len(source_checks),
            "sha256": source_checks,
        },
        "retained_failed_batches": [
            {
                "stage": "P4 necessity v1",
                "counts": "18 complete / 3 failed / 3 unstarted",
                "reason": "BF16 subset/full-batch near-tie archive mismatch",
                "resolution": "Independent v2 numerical policy; all 24 rerun; "
                "v1 excluded from final effects",
            },
            {
                "stage": "E39 v1",
                "counts": "6 parents / 24 edits completed before guard failures",
                "reason": "Hard-coded vocabulary guard rejected world1 before training",
                "resolution": "Independent v2 world-derived guard; all 24 parents/96 edits rerun; "
                "v1 excluded",
            },
        ],
        "durable_storage": {
            "P0": "Raw predictions and source checkpoints on shared storage",
            "P1": "20 models directly shared; four local tail models in verified tar; "
            "all 192 final model/resume states audited",
            "P2": "All468 new weight files directly shared, independently hashed and state-audited",
            "local_stages": "All successful local raw stages have verified durable tar indexes; "
            "failed v1 batches also retained",
        },
        "conditional_branches": {
            "Q_editor": "Not triggered: frozen candidate failed selective necessity/recovery",
            "natural_language": "Not triggered; cannot count as completed",
            "v2.9_confirmation": "Separate prospective scope; no new-world data read by this audit",
        },
        "method_limits": [
            "This completion audit checks stage receipts, current source locks and archive "
            "headers/indexes; it does not rerun model inference.",
            "Archive payload SHA verification is inherited from explicit every_member_verified "
            "indexes; this pass does not reread all tar payloads.",
            "P2 now has one-pass SHA/CPU state checks for every new checkpoint; "
            "intermediate checkpoint predictions were previously audited, not rerun.",
            "Historical reference uses and retained partial v1 reruns are excluded "
            "from valid new-learning/edit totals.",
            "Completion of the executed development matrix does not establish the rejected "
            "internal mechanism or imply natural-language completion.",
        ],
        "evidence_sha256": evidence,
        "errors": errors,
        "remaining_required_development_gaps": errors,
        "audit_script_sha256": digest(__file__),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "complete": not errors,
                "totals": totals,
                "archives": len(archives),
                "source_files": len(source_checks),
                "errors": errors,
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if errors:
        raise RuntimeError("Development completion evidence has unresolved gaps")


if __name__ == "__main__":
    main()
