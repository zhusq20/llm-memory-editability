"""Contracts preventing duplicate training, premature admission, and pilot bypass."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gated_controller", ROOT / "scripts/launch_gated_depth.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class ControllerContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.config = json.loads(
            (ROOT / "configs/gated-loop-depth-extrapolation-v1.json").read_text()
        )

    def test_matrix_requires_three_seeds_and_two_lanes(self):
        launcher.validate(self.config)
        self.config["seeds"] = [11]
        with self.assertRaisesRegex(ValueError, "three paired"):
            launcher.validate(self.config)

    def old_queue(self):
        runtime = self.config["runtime"]
        runtime["previous_queue_root"] = str(self.base)
        runtime["previous_queue_lanes"] = {"1": ["first", "second"]}
        runtime["reserved_predecessors"] = []
        launcher.write(
            self.base / "handoff/lane-1.json",
            {"state": "finished", "outcomes": {"first": 0, "second": 0}},
        )
        for job in ("first", "second"):
            launcher.write(
                self.base / "handoff" / f"{job}-status.json",
                {
                    "state": "complete",
                    "independent_audit_passed": True,
                    "cloud_verified": True,
                },
            )
            for stage in ("train", "audit", "cloud-audit"):
                launcher.write(self.base / "execution" / job / stage / "exit.json", {"code": 0})
        return runtime

    def test_old_lane_reserved_through_all_audits(self):
        runtime = self.old_queue()
        self.assertEqual(launcher.previous_queue_blockers(runtime), {})
        (self.base / "execution/second/cloud-audit/exit.json").unlink()
        self.assertIn(1, launcher.previous_queue_blockers(runtime))

    def test_idle_gap_cannot_steal_old_scheduled_lane(self):
        runtime = self.old_queue()
        launcher.write(
            self.base / "handoff/lane-1.json",
            {"state": "running", "job": "second", "outcomes": {"first": 0}},
        )
        self.assertIn(1, launcher.previous_queue_blockers(runtime))

    def test_failed_old_lane_is_not_assumed_released(self):
        runtime = self.old_queue()
        launcher.write(self.base / "execution/second/audit/exit.json", {"code": 1})
        self.assertIn(1, launcher.previous_queue_blockers(runtime))

    def test_failed_reserved_predecessor_keeps_its_gpu_reservations(self):
        runtime = self.old_queue()
        root = self.base / "other-prior"
        runtime["reserved_predecessors"] = [
            {"name": "norm-handoff", "root": str(root), "gpus": [4, 1, 2]}
        ]
        launcher.write(root / "controller-state.json", {"state": "needs_attention"})
        self.assertEqual(set(launcher.previous_queue_blockers(runtime)), {1, 2, 4})
        launcher.write(root / "controller-state.json", {"state": "complete"})
        self.assertEqual(set(launcher.previous_queue_blockers(runtime)), {1, 2, 4})
        launcher.write(root / "completion-manifest.json", {"passed": True})
        self.assertEqual(launcher.previous_queue_blockers(runtime), {})

    def test_active_only_predecessor_does_not_block_dead_failed_queue(self):
        runtime = self.old_queue()
        root = self.base / "other-prior"
        runtime["reserved_predecessors"] = [
            {
                "name": "norm-handoff",
                "root": str(root),
                "gpus": [4, 1, 2],
                "policy": "active_controller_only",
            }
        ]
        launcher.write(root / "controller-state.json", {"state": "needs_attention"})
        with patch.object(launcher, "predecessor_controller_active", return_value=False):
            self.assertEqual(launcher.previous_queue_blockers(runtime), {})
        with patch.object(launcher, "predecessor_controller_active", return_value=True):
            self.assertEqual(set(launcher.previous_queue_blockers(runtime)), {1, 2, 4})

    def test_gpu_admission_is_exclusive_and_prefers_one_two(self):
        runtime = self.config["runtime"]
        row = {"uuid": "fake", "used": 0, "free": 90000, "utilization": 0}
        rows = {0: dict(row), 1: dict(row), 2: dict(row)}
        self.assertEqual(launcher.idle_candidates(runtime, rows, {}), [1, 2, 0])
        self.assertEqual(launcher.idle_candidates(runtime, rows, {1: ["old queue"]}, [2]), [0])
        rows[0]["used"] = 800
        self.assertEqual(launcher.idle_candidates(runtime, rows, {1: ["old queue"]}, [2]), [])

    def test_failed_claim_never_reruns(self):
        controller = launcher.Controller(self.config, self.base, self.base / "unused")
        launcher.write(self.base / "execution/job/train/exit.json", {"code": 137})
        with patch.object(launcher, "docker") as docker:
            with self.assertRaisesRegex(RuntimeError, "without retry"):
                controller.execute("job", "train", 0)
            docker.assert_not_called()

    def test_success_exit_requires_independent_audit_artifact(self):
        controller = launcher.Controller(self.config, self.base, self.base / "unused")
        launcher.write(self.base / "execution/job/audit/exit.json", {"code": 0})
        launcher.write(self.base / "train/N-Full-seed11/audit.json", {"passed": False})
        with self.assertRaisesRegex(ValueError, "did not pass"):
            controller.execute("job", "audit", 0, arm="N-Full", seed=11)

    def test_completed_stage_is_reused_without_container_call(self):
        controller = launcher.Controller(self.config, self.base, self.base / "unused")
        launcher.write(self.base / "execution/job/train/exit.json", {"code": 0})
        launcher.write(
            self.base / "train/N-Full-seed11/complete.json",
            {"passed": True, "steps": 100000, "arm": "N-Full", "seed": 11, "phase": "train"},
        )
        with patch.object(launcher, "docker") as docker:
            controller.execute("job", "train", 0, arm="N-Full", seed=11)
            docker.assert_not_called()

    def test_no_go_blocks_all_formal_training(self):
        controller = launcher.Controller(self.config, self.base, self.base / "unused")
        launcher.write(
            self.base / "pilot-gate.json",
            {"passed": False, "reason": "short task not learned"},
        )
        with (
            patch.object(controller, "execute"),
            patch.object(controller, "phase") as phase,
        ):
            controller.run()
        self.assertEqual([call.args[0] for call in phase.call_args_list], ["smoke", "pilot"])
        self.assertEqual(controller.state["state"], "pilot_no_go")

    def test_go_admits_exactly_six_by_three_matrix(self):
        controller = launcher.Controller(self.config, self.base, self.base / "unused")
        launcher.write(self.base / "pilot-gate.json", {"passed": True})
        with (
            patch.object(controller, "execute"),
            patch.object(controller, "phase") as phase,
        ):
            controller.run()
        self.assertEqual(
            [call.args[0] for call in phase.call_args_list], ["smoke", "pilot", "train"]
        )
        self.assertEqual(phase.call_args_list[-1].args[1], [11, 22, 33])
        self.assertEqual(len(launcher.ARMS) * len(phase.call_args_list[-1].args[1]), 18)

    def test_secret_transport_is_private_and_not_a_host_mount(self):
        token = self.base / "private-token"
        token.write_text("unit-test-dummy-value")
        token.chmod(0o600)
        with tarfile.open(fileobj=io.BytesIO(launcher.secret_archive(token))) as archive:
            info = archive.getmember("run/secrets/wandb")
            self.assertEqual(info.mode, 0o400)
            self.assertEqual(archive.extractfile(info).read(), b"unit-test-dummy-value")
        launcher.write(self.base / "freeze.json", {"dummy": True})
        name, command = launcher.container_command(
            self.config,
            self.base,
            "job",
            "train",
            0,
            {"uuid": "test-gpu"},
            "N-Full",
            11,
        )
        self.assertNotIn("unit-test-dummy-value", " ".join(command))
        self.assertFalse(any("source=" + str(token) in part for part in command))
        self.assertEqual(command[:4], ["docker", "--context", "lm-memory", "create"])
        self.assertIn("device=test-gpu", command)
        self.assertNotIn("--rm", command)
        self.assertTrue(name.startswith(launcher.BATCH))

    def test_secret_refuses_public_file(self):
        token = self.base / "public-token"
        token.write_text("dummy")
        token.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "private"):
            launcher.secret_archive(token)

    def test_cpu_contracts_do_not_request_cuda_or_credentials(self):
        launcher.write(self.base / "freeze.json", {"dummy": True})
        _, command = launcher.container_command(self.config, self.base, "setup", "contracts", 0)
        self.assertNotIn("--gpus", command)
        self.assertFalse(any("WANDB_API_KEY" in value for value in command))


if __name__ == "__main__":
    unittest.main()
