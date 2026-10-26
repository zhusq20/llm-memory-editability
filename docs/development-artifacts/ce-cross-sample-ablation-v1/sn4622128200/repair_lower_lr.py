"""Resume the failed scheduler arm, retaining all scientific source and outputs."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import time
from datetime import datetime

spec = importlib.util.spec_from_file_location(
    "continuation", Path(__file__).with_name("continue_four.py")
)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
ARM = "discoloop-lower-lr"
GPU = "GPU-9c6ca0d7-11bb-e940-c845-efea3cc49c23"
IMAGE = "sha256:27b3ab677fb92aaf4e4cd953e3a87e953c4b164a0d4d4c4785dc9779851c11d2"


def validate_probe(actual, image, gpu):
    assert actual["Image"] == image
    assert actual["Config"]["Labels"]["lm.batch"] == r.BATCH
    assert actual["State"]["Status"] == "exited"
    assert actual["State"]["ExitCode"] == 0
    requests = actual["HostConfig"]["DeviceRequests"]
    assert len(requests) == 1 and requests[0]["DeviceIDs"] == [gpu["uuid"]]


def reuse_probe(arm, image, gpu):
    assert arm == ARM and gpu["uuid"] == GPU
    name = f"ce-ablation-sn8200-v2-{arm}-cuda-check"
    actual = json.loads(r.docker("inspect", name).stdout)[0]
    validate_probe(actual, image, gpu)
    logs = r.docker("logs", name)
    receipts = [
        json.loads(line)
        for line in logs.stdout.decode().splitlines()
        if line.startswith("{")
    ]
    assert any(
        x.get("passed") is True and x.get("optimizer_updates") == 0 for x in receipts
    )
    r.save(
        r.STATE / "lower-lr-repair" / "reused-probe.json",
        {
            "passed": True,
            "container": name,
            "id": actual["Id"],
            "gpu": gpu,
            "actual_image": image,
            "optimizer_updates": 0,
        },
    )


def main():
    repair = r.STATE / "lower-lr-repair"
    repair.mkdir(mode=0o700, exist_ok=True)
    with (repair / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (repair / "process.json").exists(), (
            "Explicit recovery only; no automatic resubmission"
        )
        assert r.sha(r.ROOT / "source-v1/source-manifest.json") == r.MANIFEST
        assert (
            r.sha(r.ROOT / "cross-server-handoff/frozen-source.tar")
            == r.read(r.ROOT / "cross-server-handoff/manifest.json")[
                "source_tar_sha256"
            ]
        )
        assert r.existing(ARM, "gate")
        # The previous scheduler exited before collecting this completed container.
        arm_root = r.ROOT / ARM
        if not (arm_root / "exit-audit-gate.json").exists():
            launch = r.read(arm_root / "launch-audit-gate.json")
            actual = json.loads(r.docker("inspect", launch["name"]).stdout)[0]
            assert (
                actual["Id"] == launch["id"] and launch["source_sha256"] == r.MANIFEST
            )
            validate_probe(actual, IMAGE, {"uuid": GPU})
            audit = r.read(arm_root / "output/audit-gate/result.json")
            assert (
                audit["passed"]
                and audit["gate_valid"]
                and audit["replayed_states"] == 64
            )
            assert audit["gate_result_sha256"] == r.sha(
                arm_root / "output/gate/result.json"
            )
            assert audit["scientific_optimizer_updates"] == 0
            logs = r.docker("logs", launch["name"])
            (arm_root / "audit-gate.log").write_bytes(logs.stdout + logs.stderr)
            r.save(arm_root / "exit-audit-gate.json", actual["State"])
            duration = datetime.fromisoformat(
                actual["State"]["FinishedAt"]
            ) - datetime.fromisoformat(actual["State"]["StartedAt"])
            r.save(
                arm_root / "duration-audit-gate.json",
                {
                    "elapsed_seconds": duration.total_seconds(),
                    "recovered_from_docker_timestamps": True,
                },
            )
            r.save(
                repair / "adopted-audit-gate.json",
                {"id": actual["Id"], "passed": True, "replayed_again": False},
            )
        assert r.existing(ARM, "audit-gate")
        assert not r.existing(ARM, "science")
        previous = r.read(r.STATE / ARM / "status.json")
        assert previous["state"] == "needs_attention"
        r.save(repair / "original-failure.json", previous)
        r.save(
            repair / "process.json",
            {
                "pid": os.getpid(),
                "started_epoch": time.time(),
                "arm": ARM,
                "source_sha256": r.MANIFEST,
                "repair_sha256": r.sha(__file__),
                "remaining_scientific_updates": 2560,
                "automatic_retry": False,
                "completed_stages_reused": ["gate", "audit-gate"],
                "other_controller_preserved": r.read(
                    r.STATE / "controller-process.json"
                ),
            },
        )
        r.zero_probe = reuse_probe
        r.worker(ARM, IMAGE, {GPU})
        r.save(
            repair / "status.json", {"state": "arm_complete_waiting_for_batch_report"}
        )
        # The original controller still owns the other live arm. Wait for it to exit.
        deadline = time.monotonic() + 43200
        with (
            (r.STATE / "execution.lock").open("a") as execution,
            (r.ROOT / "controller.lock").open("a") as batch,
        ):
            for handle in (execution, batch):
                while True:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "Original controller still holds lock; preserved"
                            )
                        time.sleep(10)
            for arm in (*r.ARMS, "loop-lower-lr", "loop-no-bound"):
                receipt = r.read(r.ROOT / arm / "completion-manifest.json")
                assert (
                    receipt["passed"]
                    and receipt["independent_audit_passed"]
                    and receipt["full_cloud_json_verified"]
                )
            old_state = r.STATE / "controller-state.json"
            if old_state.exists():
                r.save(
                    repair / "original-controller-final-state.json", r.read(old_state)
                )
            r.stage(
                "report",
                "report",
                IMAGE,
                command=[
                    "python",
                    "scripts/report_ce_cross_sample_ablation.py",
                    "--plan",
                    f"configs/{r.BATCH}.json",
                    "--batch",
                    f"/persist/{r.BATCH}",
                    "--out",
                    f"/persist/{r.BATCH}/report/output",
                ],
            )
            cloud = r.read(r.ROOT / "report/output/cloud-verification.json")
            assert cloud["passed"] and cloud["canonical_full_json_exact"]
            complete = {
                "state": "complete",
                "passed": True,
                "scientific_updates": 15360,
                "independent_reload": True,
                "cloud_verified": True,
                "report": str(r.ROOT / "report/output/result.json"),
                "recovery": str(repair),
                "source_sha256": r.MANIFEST,
            }
            r.save(repair / "status.json", complete)
            r.save(old_state, complete)


if __name__ == "__main__":
    main()
