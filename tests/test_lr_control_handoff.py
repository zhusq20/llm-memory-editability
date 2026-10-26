"""Prevent duplicate scientific launches during the four-arm transfer."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/continue_lr_control_handoff.py"
SPEC = importlib.util.spec_from_file_location("handoff", SCRIPT)
HANDOFF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HANDOFF)


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.batch = Path(self.temp.name)
        for name in (
            "source/model.py",
            "input/weights",
            "baseline-reuse.json",
            "decay-contract.json",
        ):
            path = self.batch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original")
        self.sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.image = "sha256:" + "a" * 64
        manifest = {
            "files": {p: self.sha(self.batch / p) for p in ("source/model.py", "input/weights")},
            "baseline_reuse_sha256": self.sha(self.batch / "baseline-reuse.json"),
            "decay_contract_sha256": self.sha(self.batch / "decay-contract.json"),
        }
        (self.batch / "source-manifest.json").write_text(json.dumps(manifest))
        self.module = SimpleNamespace(
            read=lambda p: json.loads(Path(p).read_text()),
            sha=self.sha,
            docker=lambda *a: SimpleNamespace(stdout=json.dumps([{"Id": self.image}])),
        )

    def test_runtime_change_still_rejects_tampered_scientific_input(self):
        (self.batch / "input/weights").write_text("different weights")
        with self.assertRaisesRegex(ValueError, "Frozen bytes changed"):
            HANDOFF.verify_runtime(self.module, self.batch, self.image)

    def test_runtime_requires_exact_image_identity(self):
        with self.assertRaisesRegex(ValueError, "full image ID"):
            HANDOFF.verify_runtime(self.module, self.batch, "mutable-tag")

    def test_valid_runtime_preserves_manifest_bytes(self):
        before = (self.batch / "source-manifest.json").read_bytes()
        HANDOFF.verify_runtime(self.module, self.batch, self.image)
        self.assertEqual(before, (self.batch / "source-manifest.json").read_bytes())


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.batch = Path(self.temp.name)
        source = self.batch / "source"
        (source / "scripts").mkdir(parents=True)
        (source / "configs").mkdir()
        (source / "scripts/launch_lr_control.py").write_text(
            "import json\nfrom pathlib import Path\n"
            "STAGES=('cuda-contracts','preflight','smoke','smoke-audit','train','audit','cloud-audit')\n"
            "def read(p): return json.loads(Path(p).read_text())\n"
            "def validate(c): pass\n"
            "def docker(*a, **kw): raise AssertionError('Docker must not be called')\n"
            "def command(*a): raise AssertionError('Must not launch')\n"
        )
        (source / "configs/loop-discoloop-lr-control-v1.json").write_text(
            json.dumps({"runtime": {"cpu_sets": {}}})
        )
        self.script = Path(__file__).resolve().parents[1] / "scripts/continue_lr_control_handoff.py"

    def run_plan(self, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--batch",
                str(self.batch),
                "--job",
                "loop-lr5e-4",
                "--gpu",
                "0",
                "--cpus",
                "0,1,2,3",
                "--wandb-token-file",
                str(self.batch / "token"),
                *extra,
            ],
            text=True,
            capture_output=True,
        )

    def test_plan_skips_successful_stages_without_mutating_batch(self):
        stage = self.batch / "execution/loop-lr5e-4/smoke"
        stage.mkdir(parents=True)
        (stage / "exit.json").write_text('{"code": 0}')
        before = sorted(str(p) for p in self.batch.rglob("*"))
        result = self.run_plan()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("smoke", json.loads(result.stdout)["remaining_stages"])
        self.assertEqual(before, sorted(str(p) for p in self.batch.rglob("*")))

    def test_failed_stage_is_not_retried(self):
        stage = self.batch / "execution/loop-lr5e-4/smoke"
        stage.mkdir(parents=True)
        (stage / "exit.json").write_text('{"code": 1}')
        result = self.run_plan()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no automatic retry", result.stderr)

    def test_transfer_rejects_existing_scientific_claim(self):
        (self.batch / "handoff").mkdir()
        (self.batch / "handoff/paused-queue.json").write_text("{}")
        (self.batch / "execution/loop-lr5e-4/train").mkdir(parents=True)
        token = self.batch / "token"
        token.write_text("test-only")
        token.chmod(0o600)
        result = self.run_plan("--execute")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refuse duplicate", result.stderr)


if __name__ == "__main__":
    unittest.main()
