"""Read-correction calibration and paired independent second-hop edit readouts.

No base training or automatic reader adaptation is performed. Failed reading
calibration is retained and blocks evaluation rather than being relabeled an
editing/combination failure. Every case starts from the identical frozen parent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from .edit_scope_development import evaluate_queries, prediction_metrics
from .grok_depth import utc, write_json
from .grokking_reproduction import model_digest
from .path_overlap import World, sha256
from .supervised_alignment import SupervisedAlignmentGPT
from .ttcd_editing import (
    TTCDReader,
    build_case_bank,
    correction_packet,
    load_fast,
    save_fast,
    tensor_digest,
)

CODE_FILES = (
    "src/llm_memory_editability/ttcd_editing.py",
    "src/llm_memory_editability/ttcd_experiment.py",
    "src/llm_memory_editability/grokking_reproduction.py",
    "src/llm_memory_editability/self_decode_alignment.py",
    "src/llm_memory_editability/supervised_alignment.py",
    "src/llm_memory_editability/edit_scope_development.py",
    "src/llm_memory_editability/path_overlap.py",
    "src/llm_memory_editability/grok_depth.py",
    "scripts/run_ttcd_second_hop.py",
    "src/llm_memory_editability/ttcd_scratch.py",
    "scripts/prepare_ttcd_scratch.py",
)


class Session(torch.nn.Module):
    def __init__(self, reader, fast):
        super().__init__()
        self.reader, self.fast = reader, fast

    def forward(self, tokens, positions=None):
        return self.reader(tokens, positions, fast=self.fast)


def load_inputs(config, name, device, *, require_training_audit=True):
    if (
        config["writer"]["steps"] < 1
        or config["atomic_ce"]["steps"] < 1
        or len(set(config["methods"])) != len(config["methods"])
        or not set(config["methods"]).issubset({"ttcd", "atomic_ce", "sham"})
    ):
        raise ValueError("Invalid writer budget or duplicate/unknown methods")
    source = config["models"][name]
    required = [Path(config["data_dir"]) / n for n in ("world.npz", "panels.npz", "complete.json")]
    required += [Path(config["model_config"]), Path(source["checkpoint"])]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Restore audited inputs; no replacement training is started: " + ", ".join(missing)
        )
    world = World(config["data_dir"])
    for stem in ("world", "panels"):
        if sha256(Path(config["data_dir"]) / f"{stem}.npz") != world.metadata[f"{stem}_sha256"]:
            raise ValueError(f"Archived {stem} hash mismatch")
    checkpoint = torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    if "scratch" in config:
        from .ttcd_scratch import training_identity, validate_scratch_inputs

        validate_scratch_inputs(config)
        if checkpoint.get("training_identity") != training_identity(config):
            raise ValueError("Scratch training provenance differs from current code/config/inputs")
        if require_training_audit:
            audit_path = Path(source["checkpoint"]).parent / "audit.json"
            if not audit_path.exists():
                raise ValueError("Independent training reload audit is required before editing")
            receipt = json.loads(audit_path.read_text())
            if (
                not receipt["passed"]
                or receipt["parent_sha256"] != sha256(source["checkpoint"])
                or receipt["training_identity"] != training_identity(config)
            ):
                raise ValueError("Training audit does not bind this parent and contract")
    if checkpoint["step"] != source["step"]:
        raise ValueError("Wrong parent checkpoint step")
    if "sha256" in source and sha256(source["checkpoint"]) != source["sha256"]:
        raise ValueError("Parent checkpoint hash differs from pinned source")
    flags = checkpoint.get("architecture", {"discoloop": False, "first_round_kv": False})
    if flags.get("discoloop", False) != source["discoloop"] or flags.get("first_round_kv", False):
        raise ValueError("Checkpoint architecture does not match the configured parent")
    spec = checkpoint["spec"]
    if spec["phi"] != config["phi"]:
        raise ValueError("Training support phi mismatch")
    hf = GPT2Config.from_json_file(config["model_config"])
    hf.vocab_size, hf.n_layer = world.metadata["vocab_size"], spec["unique_layers"]
    hf.use_cache = False
    parent = SupervisedAlignmentGPT(hf, repeats=spec["repeats"], **flags)
    parent.load_state_dict(checkpoint["model"], strict=True)
    if "model_sha256" in checkpoint and model_digest(parent) != checkpoint["model_sha256"]:
        raise ValueError("Checkpoint's internal model hash does not match its tensors")
    reader = TTCDReader(parent.to(device)).eval()
    return reader, world


def bank_for(config, world):
    return build_case_bank(world, phi=config["phi"], **config["cases"])


def fingerprint(config, name):
    root = Path(__file__).resolve().parents[2]
    paths = [Path(config["models"][name]["checkpoint"]), Path(config["model_config"])]
    paths += [Path(config["data_dir"]) / n for n in ("world.npz", "panels.npz", "complete.json")]
    if "scratch" in config:
        paths += [
            Path(config["data_dir"]) / "training-split.json",
            Path(config["models"][name]["checkpoint"]).parent / "audit.json",
        ]
    paths += [root / p for p in CODE_FILES]
    return {
        "config": json.loads(json.dumps(config)),
        "model": name,
        "files": {str(p): sha256(p) for p in paths},
    }


def packet_for(config, world, case, row=None):
    atom = world.atoms[case["atom"]]
    query = atom[:2] if row is None else row[:-1]
    return correction_packet(
        atom,
        case["new_tail"],
        query,
        world.metadata,
        gap=config["writer"]["gap"],
        teacher_window=config["writer"]["teacher_window"],
        student_window=config["writer"]["student_window"],
    )


@torch.no_grad()
def open_query(reader, fast, packet, window, old, new, metadata, device):
    x = torch.tensor([packet], dtype=torch.long, device=device)
    positions = torch.tensor([[len(packet) - 1]], device=device)
    logits = reader(x, positions, fast=fast, window=window)[0, 0].float()
    answer = int(logits.argmax())
    second = torch.cat((x, x.new_tensor([[answer]])), dim=1)
    eos = int(reader(second, positions + 1, fast=fast, window=window)[0, 0].argmax())
    eo = metadata["entity_offset"]
    return {
        "tokens": packet,
        "answer": answer,
        "stop": eos,
        "old_answers": old,
        "new_answers": new,
        "new_answer_correct": answer == new + eo,
        "new_correct": answer == new + eo and eos == metadata["end_marker"],
        "new_minus_old": float(logits[new + eo] - logits[old + eo]),
    }


def mean(values):
    return float(np.mean(values)) if len(values) else None


def retention(before, after, metadata):
    """Original-correct coverage and retention; do not equate endpoint accuracy with retention."""
    original = (
        np.asarray(before["answer"])
        == np.asarray(before["old_answers"]) + metadata["entity_offset"]
    )
    original &= np.asarray(before["stop"]) == metadata["end_marker"]
    kept = (
        np.asarray(after["answer"]) == np.asarray(after["old_answers"]) + metadata["entity_offset"]
    )
    kept &= np.asarray(after["stop"]) == metadata["end_marker"]
    return {
        "total": len(original),
        "original_correct": int(original.sum()),
        "coverage": mean(original),
        "retained": mean(kept[original]),
    }


def calibrate(config, name, out, device):
    reader, world = load_inputs(config, name, device)
    identity = fingerprint(config, name)
    bank = bank_for(config, world)
    write_json(out / "case-bank.json", bank)
    records = []
    fast = reader.empty_fast()
    for case in bank:
        if case["split"] != "calibration":
            continue
        atom = world.atoms[case["atom"]]
        packet = packet_for(config, world, case)
        record = {"case": case}
        for key in ("teacher", "student"):
            record[key] = open_query(
                reader,
                fast,
                packet,
                config["writer"][f"{key}_window"],
                int(atom[2]),
                case["new_tail"],
                world.metadata,
                device,
            )
        # A native-position baseline, not a second capacity training stage.
        record["closed_before"] = evaluate_queries(
            Session(reader, fast).eval(),
            atom[None],
            world.metadata,
            torch.device(device),
            new_answers=[case["new_tail"]],
            precision="fp32",
        )["answer"].tolist()
        records.append(record)
        write_json(out / "calibration-progress.json", {"cases": len(records), "utc": utc()})
    groups = {
        group: {
            "n": sum(r["case"]["group"] == group for r in records),
            "teacher_new_answer_accuracy": mean(
                [r["teacher"]["new_answer_correct"] for r in records if r["case"]["group"] == group]
            ),
            "teacher_new_joint_accuracy": mean(
                [r["teacher"]["new_correct"] for r in records if r["case"]["group"] == group]
            ),
            "student_new_answer_accuracy": mean(
                [r["student"]["new_answer_correct"] for r in records if r["case"]["group"] == group]
            ),
        }
        for group in ("id", "ood")
    }
    passed = all(
        g["n"] >= config["reading_gate"]["minimum_cases_per_group"]
        and g["teacher_new_answer_accuracy"] >= config["reading_gate"]["minimum_accuracy"]
        for g in groups.values()
    )
    result = {
        "identity": identity,
        "groups": groups,
        "records": records,
        "passed": passed,
        "status": "reading_passed" if passed else "reading_needs_attention",
        "meaning": "Full long/short forwards diagnose record reading. TTCD local views "
        "share incoming residual/QKV; these are not claimed identical to isolated forwards.",
    }
    write_json(out / "calibration.json", result)
    return result


def write_atomic_ce(reader, world, case, steps, learning_rate):
    """Same physical down matrices; atomic answer/EOS labels only, no replay."""
    fast = {k: v.requires_grad_() for k, v in reader.empty_fast().items()}
    optimizer = torch.optim.Adam(list(fast.values()), lr=learning_rate)
    atom = world.atoms[case["atom"]]
    x = torch.tensor(
        [[atom[0] + world.eo, atom[1] + world.ro, case["new_tail"] + world.eo]],
        device=next(reader.parameters()).device,
    )
    labels = torch.tensor(
        [[case["new_tail"] + world.eo, world.metadata["end_marker"]]], device=x.device
    )
    positions = torch.tensor([[1, 2]], device=x.device)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(reader(x, positions, fast=fast).flatten(0, 1), labels.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite atomic CE")
        loss.backward()
        optimizer.step()
    return {k: v.detach() for k, v in fast.items()}, {"steps": steps, "loss": float(loss.detach())}


@torch.no_grad()
def read_case(reader, fast, world, case, device):
    session = Session(reader, fast).eval()
    atom = world.atoms[case["atom"]]
    pools = {
        "E": (atom[None], [case["new_tail"]]),
        "D": (world.chains[case["D"]], [case["new_tail"]] * len(case["D"])),
        "U_atoms": (world.atoms[case["U_atoms"]], None),
        "U_chains": (world.chains[case["U_chains"]], None),
        "same_answer_atoms": (world.atoms[case["same_answer_atoms"]], None),
        "prerequisite_atoms": (world.atoms[world.first[case["D"]]], None),
    }
    predictions, metrics = {}, {}
    for key, (rows, targets) in pools.items():
        result = evaluate_queries(
            session,
            rows,
            world.metadata,
            torch.device(device),
            new_answers=targets,
            batch_size=64,
            precision="fp32",
        )
        metrics[key] = prediction_metrics(result, world.metadata)
        predictions[key] = {k: v.tolist() for k, v in result.items()}
    return {"metrics": metrics, "predictions": predictions}


def evaluate(config, name, out, device, log=None):
    receipt = json.loads((out / "calibration.json").read_text())
    if not receipt["passed"] or receipt["identity"] != fingerprint(config, name):
        raise ValueError(
            "Passing reading calibration with identical code/config/inputs is required"
        )
    reader, world = load_inputs(config, name, device)
    original = tensor_digest(reader.backbone.state_dict())
    bank = bank_for(config, world)
    if bank != json.loads((out / "case-bank.json").read_text()):
        raise ValueError("Case bank changed since calibration")
    records = []
    for case in bank:
        if case["split"] != "evaluation":
            continue
        before = read_case(reader, reader.empty_fast(), world, case, device)
        for method in config["methods"]:
            started = time.monotonic()
            compute_start = dict(reader.compute)
            fast, history = reader.empty_fast(), []
            if method == "ttcd":
                packet = torch.tensor([packet_for(config, world, case)], device=device)
                for _ in range(config["writer"]["steps"]):
                    fast, info = reader.write(
                        packet,
                        fast,
                        teacher_window=config["writer"]["teacher_window"],
                        student_window=config["writer"]["student_window"],
                        learning_rate=config["writer"]["learning_rate"],
                    )
                    history.append(info)
            elif method == "atomic_ce":
                fast, info = write_atomic_ce(reader, world, case, **config["atomic_ce"])
                history.append(info)
            elif method != "sham":
                raise ValueError(f"Unknown method {method}")
            writer_compute = {k: reader.compute[k] - compute_start[k] for k in reader.compute}
            directory = out / "cases" / case["case_id"] / method
            directory.mkdir(parents=True, exist_ok=False)
            fast_path = directory / "fast.pt"
            digest = save_fast(fast_path, reader, fast, {"case": case, "method": method})
            after = read_case(reader, fast, world, case, device)
            # Same edited state, all four context/query cells. Open-context
            # references are NEVER fed back into the saved edit state.
            atom = world.atoms[case["atom"]]
            open_refs = {}
            for key, row in (("atomic", atom), ("two_hop", world.chains[case["D"][0]])):
                open_refs[key] = open_query(
                    reader,
                    fast,
                    packet_for(config, world, case, row),
                    config["writer"]["teacher_window"],
                    int(row[-1]),
                    case["new_tail"],
                    world.metadata,
                    device,
                )
            before_d, after_d = before["predictions"]["D"], after["predictions"]["D"]
            eligible = (
                np.asarray(before_d["answer"]) == np.asarray(before_d["old_answers"]) + world.eo
            )
            eligible &= np.asarray(before_d["stop"]) == world.metadata["end_marker"]
            prerequisites = before["predictions"]["prerequisite_atoms"]
            eligible &= (
                np.asarray(prerequisites["answer"])
                == np.asarray(prerequisites["old_answers"]) + world.eo
            )
            eligible &= np.asarray(prerequisites["stop"]) == world.metadata["end_marker"]
            if before["metrics"]["E"]["old_accuracy"] != 1.0:
                eligible[:] = False
            atom_success = after["metrics"]["E"]["new_accuracy"] == 1.0
            success = np.asarray(after_d["answer"]) == case["new_tail"] + world.eo
            success &= np.asarray(after_d["stop"]) == world.metadata["end_marker"]
            result = {
                "case": case,
                "method": method,
                "before": before,
                "after": after,
                "open_context": open_refs,
                "write_history": history,
                "fast_digest": digest,
                "writer_compute": writer_compute,
                "total_method_compute": {
                    k: reader.compute[k] - compute_start[k] for k in reader.compute
                },
                "wall_seconds": time.monotonic() - started,
                "conditional": {
                    "n": int(eligible.sum()) if atom_success else 0,
                    "total": len(eligible),
                    "new_accuracy": mean(success[eligible]) if atom_success else None,
                },
                "retention": {
                    pool: retention(
                        before["predictions"][pool], after["predictions"][pool], world.metadata
                    )
                    for pool in ("U_atoms", "U_chains", "same_answer_atoms", "prerequisite_atoms")
                },
                "exposure": {
                    "updates": config["writer"]["steps"]
                    if method == "ttcd"
                    else config["atomic_ce"]["steps"]
                    if method == "atomic_ce"
                    else 0,
                    "correction_packet_tokens": len(packet_for(config, world, case))
                    if method == "ttcd"
                    else 0,
                    "base_parameter_updates": 0,
                    "writer_input_tokens": (
                        len(packet_for(config, world, case)) * config["writer"]["steps"]
                        if method == "ttcd"
                        else 3 * config["atomic_ce"]["steps"]
                        if method == "atomic_ce"
                        else 0
                    ),
                },
            }
            write_json(directory / "readout.json", result)
            records.append(result)
            write_json(out / "evaluation-progress.json", {"completed": len(records), "utc": utc()})
            if log:
                log(len(records), result)
        if tensor_digest(reader.backbone.state_dict()) != original:
            raise AssertionError("Editing mutated the frozen backbone")
    if fingerprint(config, name) != receipt["identity"]:
        raise ValueError("Source/input changed during execution")
    summary = {
        "identity": receipt["identity"],
        "records": len(records),
        "primary": "Case-equal mean independent D answer+EOS accuracy, all fixed cases",
        "groups": {
            f"{method}/{group}": {
                "cases": sum(
                    r["method"] == method and r["case"]["group"] == group for r in records
                ),
                "E": mean(
                    [
                        r["after"]["metrics"]["E"]["new_accuracy"]
                        for r in records
                        if r["method"] == method and r["case"]["group"] == group
                    ]
                ),
                "D": mean(
                    [
                        r["after"]["metrics"]["D"]["new_accuracy"]
                        for r in records
                        if r["method"] == method and r["case"]["group"] == group
                    ]
                ),
                "U_chains": mean(
                    [
                        r["after"]["metrics"]["U_chains"]["old_accuracy"]
                        for r in records
                        if r["method"] == method and r["case"]["group"] == group
                    ]
                ),
            }
            for method in config["methods"]
            for group in ("id", "ood")
        },
        "audit_state": "pending_fresh_process_reload",
        "parameters": {
            "base_total": sum(p.numel() for p in reader.parameters()),
            "base_trainable": sum(p.numel() for p in reader.parameters() if p.requires_grad),
            "session_fast_state_elements": sum(a.numel() for a in reader.empty_fast().values()),
        },
        "compute": reader.compute,
    }
    write_json(out / "summary.json", summary)
    return summary


def audit(config, name, out, device):
    reader, world = load_inputs(config, name, device)
    receipt = json.loads((out / "calibration.json").read_text())
    if fingerprint(config, name) != receipt["identity"]:
        raise ValueError("Audit inputs/code differ from execution")
    expected_cases = {
        c["case_id"]: c for c in bank_for(config, world) if c["split"] == "evaluation"
    }
    expected = len(expected_cases) * len(config["methods"])
    paths = sorted((out / "cases").glob("*/*/readout.json"))
    if len(paths) != expected:
        raise ValueError("Incomplete execution matrix; cannot pass audit")
    checked = 0
    seen = set()
    for path in paths:
        recorded = json.loads(path.read_text())
        key = (recorded["case"]["case_id"], recorded["method"])
        if (
            expected_cases.get(key[0]) != recorded["case"]
            or key[1] not in config["methods"]
            or key in seen
        ):
            raise ValueError("Audit matrix contains an altered or duplicate case/method")
        seen.add(key)
        fast, provenance = load_fast(path.parent / "fast.pt", reader)
        if provenance != {"case": recorded["case"], "method": recorded["method"]}:
            raise ValueError("Fast-state provenance mismatch")
        if tensor_digest(fast) != recorded["fast_digest"]:
            raise ValueError("Readout fast-state hash mismatch")
        replay = read_case(reader, fast, world, recorded["case"], device)
        baseline = read_case(reader, reader.empty_fast(), world, recorded["case"], device)
        for pool in replay["predictions"]:
            for key in ("answer", "stop", "rows", "new_answers", "old_answers"):
                if replay["predictions"][pool][key] != recorded["after"]["predictions"][pool][key]:
                    raise AssertionError(f"Independent replay mismatch: {path}/{pool}/{key}")
                if (
                    baseline["predictions"][pool][key]
                    != recorded["before"]["predictions"][pool][key]
                ):
                    raise AssertionError(f"Independent baseline mismatch: {path}/{pool}/{key}")
            for key in ("old_rank", "new_rank", "new_minus_old", "old_margin", "new_margin"):
                np.testing.assert_allclose(
                    replay["predictions"][pool][key],
                    recorded["after"]["predictions"][pool][key],
                    rtol=1e-5,
                    atol=1e-5,
                )
            for key, value in replay["metrics"][pool].items():
                wanted = recorded["after"]["metrics"][pool][key]
                if value is None:
                    if wanted is not None:
                        raise AssertionError("Empty-pool metrics changed")
                else:
                    np.testing.assert_allclose(value, wanted, rtol=1e-5, atol=1e-5)
        atom = world.atoms[recorded["case"]["atom"]]
        for key, row in (("atomic", atom), ("two_hop", world.chains[recorded["case"]["D"][0]])):
            ref = open_query(
                reader,
                fast,
                packet_for(config, world, recorded["case"], row),
                config["writer"]["teacher_window"],
                int(row[-1]),
                recorded["case"]["new_tail"],
                world.metadata,
                device,
            )
            for field in ("tokens", "answer", "stop", "new_correct", "new_answer_correct"):
                if ref[field] != recorded["open_context"][key][field]:
                    raise AssertionError("Independent open-context reference mismatch")
        checked += 1
    result = {
        "passed": True,
        "cases_methods": checked,
        "identity": receipt["identity"],
        "utc": utc(),
    }
    write_json(out / "audit.json", result)
    return result
