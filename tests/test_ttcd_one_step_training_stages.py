"""Canonical output mapping, zero-update recovery and narrow Docker-top races."""

import importlib.util
import json
import subprocess
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "one_step_stage_repair", ROOT / "scripts/resume_ttcd_one_step_training_stages.py"
)
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Unavailable(RuntimeError):
    pass


def controller_stub():
    def command(_plan, arm, stage, *_args, **_kwargs):
        name = f"{repair.BATCH}-{arm}-{stage}"
        return name, ["docker", "--name", name, "python", stage, "--out", "/output"]

    return SimpleNamespace(
        gpu_cohort=lambda *_: [],
        build_command=command,
        resources=SimpleNamespace(ResourceUnavailable=Unavailable, utc_now=lambda: "now"),
        STAGES=(
            "prepare",
            "preflight",
            "train",
            "train-audit",
            "train-cloud-audit",
            "eval-preflight",
            "calibrate",
            "evaluate",
            "audit",
            "cloud-audit",
        ),
        write_json=write,
    )


def install(controller):
    repair.install_stage_repairs(controller, SimpleNamespace(install=lambda *_: None), {})


def test_training_out_matches_config_canonical_path_and_uses_new_container_name(tmp_path):
    controller = controller_stub()
    install(controller)
    for stage in ("preflight", "train", "train-audit", "train-cloud-audit"):
        name, argv = controller.build_command({}, "loop", stage, tmp_path, Path("/token"), {})
        out = PurePosixPath(argv[argv.index("--out") + 1])
        task = json.loads((ROOT / f"configs/{repair.BATCH}-loop.json").read_text())
        configured = (
            PurePosixPath("/frozen") / task["training_config"]["models"]["stream"]["checkpoint"]
        )
        assert out / "training/parent.pt" == configured
        assert name.endswith("-output-v2") is (stage == "train")
    _, argv = controller.build_command({}, "loop", "evaluate", tmp_path, Path("/token"), {})
    assert argv[argv.index("--out") + 1] == "/output"
    canonical, alias = tmp_path / "canonical", tmp_path / "output-alias"
    canonical.mkdir()
    alias.mkdir()
    (canonical / "parent.pt").write_bytes(b"same underlying checkpoint")
    (alias / "parent.pt").hardlink_to(canonical / "parent.pt")
    assert (alias / "parent.pt").samefile(canonical / "parent.pt")
    assert (alias / "parent.pt").resolve() != (canonical / "parent.pt").resolve()


@pytest.mark.parametrize(
    "stderr", ["Error: container is not running", "Error: No such container", "No such process"]
)
def test_known_docker_top_exit_race_becomes_resource_wait(stderr):
    controller = controller_stub()

    def failed(*_):
        raise subprocess.CalledProcessError(1, ["docker", "top", "container"], stderr=stderr)

    controller.gpu_cohort = failed
    install(controller)
    with pytest.raises(Unavailable, match="exited during Docker top"):
        controller.gpu_cohort()


def test_other_docker_errors_remain_fatal():
    controller = controller_stub()

    def failed(*_):
        raise subprocess.CalledProcessError(
            1, ["docker", "top", "container"], stderr="permission denied"
        )

    controller.gpu_cohort = failed
    install(controller)
    with pytest.raises(subprocess.CalledProcessError):
        controller.gpu_cohort()


def test_resume_skips_passed_prepare_and_cuda_preflight(tmp_path):
    controller = controller_stub()
    called = []
    controller.run_stage = lambda _p, _a, stage, *_: called.append(stage)
    install(controller)
    (tmp_path / "loop").mkdir()
    result = controller.run_arm({}, "loop", tmp_path, Path("/token"))
    assert called[0] == "train"
    assert "prepare" not in called and "preflight" not in called
    assert result["completed_stages"] == list(controller.STAGES)


@pytest.fixture
def failed(tmp_path):
    batch = tmp_path / repair.BATCH
    write(batch / "source-manifest.json", {"unchanged": True})
    plan = batch / "transformer/source/configs/one-step-training.json"
    write(plan, {"same": "science"})
    write(
        batch / "controller-process.json",
        {
            "pid": repair.OLD_PID,
            "source_manifest_sha256": repair.sha256(batch / "source-manifest.json"),
            "config_sha256": repair.sha256(plan),
        },
    )
    write(batch / "controller-state.json", {"state": "needs_attention", "outcomes": {}})
    for arm in repair.ARMS:
        write(
            batch / arm / "controller-state.json",
            {
                "state": "needs_attention",
                "completed_stages": ["prepare", "preflight"],
                "traceback": "CalledProcessError docker top"
                if arm == "loop"
                else repair.OUTPUT_ERROR,
            },
        )
        for stage, output in (("prepare", "evaluation"), ("preflight", "training-run")):
            write(batch / arm / f"stages/{stage}/exit.json", {"returncode": 0})
            write(
                batch / arm / output / f"result-{stage}.json",
                {
                    "passed": True,
                    "writer_steps": 1,
                    "all_four_losses_exact": True,
                    "all_parameter_gradients_exact": True,
                    "all_fast_states_exact": True,
                    "base_parameter_updates": 0,
                    "optimizer_updates": 0,
                },
            )
        write(batch / arm / "stages/train/claim.json", {"stage": "train"})
        if arm != "loop":
            write(batch / arm / "stages/train/exit.json", {"returncode": 1})
            write(
                batch / arm / "training-run/failure-train.json", {"traceback": repair.OUTPUT_ERROR}
            )
            write(batch / arm / "training-run/wandb-train.json", {"id": f"failed-{arm}"})
            (batch / arm / "training-run/training").mkdir()
    controller = SimpleNamespace(
        validate_plan=lambda _: None,
        transport=SimpleNamespace(verify_snapshot=lambda *_: None),
        resources=SimpleNamespace(
            inspect_container=lambda name: (
                None if "-loop-" in name else {"State": {"Running": False, "ExitCode": 1}}
            ),
            running_containers=lambda _: [],
        ),
    )
    return batch, controller


def test_recovery_requires_dead_controller_and_zero_scientific_updates(failed):
    batch, controller = failed
    assert repair.inspect_failure(batch, controller, lambda _: False)["base_parameter_updates"] == 0
    with pytest.raises(RuntimeError, match="must be stopped"):
        repair.inspect_failure(batch, controller, lambda _: True)
    write(batch / "loop/training-run/metrics.jsonl", {"step": 1})
    with pytest.raises(ValueError, match="scientific training metric"):
        repair.inspect_failure(batch, controller, lambda _: False)


def test_archive_preserves_passed_receipts_and_failed_wandb_ids_without_overwrite(failed):
    batch, _ = failed
    success = batch / "transformer/training-run/result-preflight.json"
    before = success.read_bytes()
    archive = repair.archive_failed_launches(batch)
    assert success.read_bytes() == before
    assert (batch / "transformer/stages/preflight/exit.json").exists()
    assert read_id(archive / "discoloop/training-run/wandb-train.json") == "failed-discoloop"
    assert not (batch / "discoloop/training-run/failure-train.json").exists()
    with pytest.raises(FileExistsError):
        repair.archive_failed_launches(batch)


def read_id(path):
    return json.loads(path.read_text())["id"]


def test_nonempty_training_directory_refused_before_archive_mutation(failed):
    batch, _ = failed
    (batch / "discoloop/training-run/training/parent.pt").write_bytes(b"existing model")
    with pytest.raises(ValueError, match="nonempty training"):
        repair.archive_failed_launches(batch)
    assert not (batch / repair.REPAIR / "failed-attempts").exists()
    assert (batch / "transformer/stages/train/claim.json").exists()
