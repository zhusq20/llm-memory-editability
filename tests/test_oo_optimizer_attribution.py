"""Protect optimizer attribution, historical reuse, and the GPU queue boundary."""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "oo_optimizer_controller", ROOT / "scripts/execute_oo_optimizer_attribution.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture_config():
    return module.read(ROOT / "configs/oo-optimizer-attribution-v1.json")


def test_complete_paired_factorial_reuses_only_the_baseline():
    config = fixture_config()
    module.validate_matrix(config, module.read(config["historical_config"]))
    assert sum(s["steps"] for s in config["runs"]) == 1_800_000
    assert len(config["baseline_curve_carriers"]) == 2
    carriers = [s for s in config["runs"] if s["name"] in config["baseline_curve_carriers"]]
    assert {s["initialization"] for s in carriers} == {811201, 811202}


@pytest.mark.parametrize(
    "key,value",
    [
        ("steps", 1500000),
        ("sampling_seed", 1),
        ("dropout_seed", 1),
        ("microbatch_size", 128),
        ("repeats", 1),
        ("learning_rate", 5e-5),
    ],
)
def test_rejects_unpaired_or_duplicate_conditions(key, value):
    config = fixture_config()
    config["runs"][0][key] = value
    with pytest.raises(AssertionError):
        module.validate_matrix(config, module.read(config["historical_config"]))


def test_historical_training_dependency_closure_and_audits():
    config = fixture_config()
    result = module.verify_inputs(config, ROOT)
    assert result["passed"]
    assert set(result["trainer_dependency_sha256"]) == {
        "src/llm_memory_editability/" + name + ".py"
        for name in [
            "realworld_composition",
            "realworld_composition_data",
            "grokking_reproduction",
            "grok_depth",
            "bios_model",
        ]
    }


def test_queue_needs_every_predecessor_audit(tmp_path):
    config = {"wait_for_results_root": str(tmp_path), "wait_for_runs": ["a", "b"]}
    state = {"state": "running", "completed": ["a"], "active": {"2": "b"}, "failed": []}
    module.write(tmp_path / "controller-state.json", state)
    assert module.predecessor_ready(config)[0] is False
    state.update(state="complete", completed=["a", "b"], active={})
    module.write(tmp_path / "controller-state.json", state)
    module.write(tmp_path / "runs/a/audit.json", {"passed": True})
    with pytest.raises(FileNotFoundError):
        module.predecessor_ready(config)
    module.write(tmp_path / "runs/b/audit.json", {"passed": False})
    with pytest.raises(AssertionError):
        module.predecessor_ready(config)
    module.write(tmp_path / "runs/b/audit.json", {"passed": True})
    assert module.predecessor_ready(config)[0] is True
    state.update(state="finished_with_failures", failed=["b"])
    module.write(tmp_path / "controller-state.json", state)
    with pytest.raises(RuntimeError):
        module.predecessor_ready(config)


def test_container_stays_in_lm_context_with_a_single_gpu_and_output(tmp_path):
    config = copy.deepcopy(fixture_config())
    config.update(repository=str(ROOT), source_root=str(tmp_path / "source"))
    out = tmp_path / "runs" / config["runs"][0]["name"]
    _, command = module.container_command(
        config, tmp_path / "frozen.json", config["runs"][0], out, 4, 1
    )
    assert command[:4] == ["docker", "--context", "lm-memory", "run"]
    assert command[command.index("--gpus") + 1] == "device=4"
    assert str(tmp_path / "source/scripts/execute_oo_optimizer_attribution.py") in command
    assert f"type=bind,src={ROOT},dst={ROOT},readonly" in command
    assert f"type=bind,src={out},dst={out}" in command
    assert "--read-only" in command and "--network=none" in command


def test_report_pairs_both_factors_and_rejects_exposure_drift(tmp_path):
    config = fixture_config()
    config["results_root"] = str(tmp_path / "results")
    for baseline in config["reused_baselines"]:
        baseline["path"] = str(tmp_path / baseline["spec"]["name"])
    entries = [(b["spec"], Path(b["path"])) for b in config["reused_baselines"]]
    entries += [(s, Path(config["results_root"]) / "runs" / s["name"]) for s in config["runs"]]
    for spec, out in entries:
        count = (
            100
            + 20 * (spec["learning_rate"] == 1e-4)
            + 30 * (spec["weight_decay"] == 0.3)
            + 7 * (spec["learning_rate"] == 1e-4 and spec["weight_decay"] == 0.3)
        )
        module.write(out / "audit.json", {"passed": True})
        module.write(out / "run.json", {"initial_model_sha256": str(spec["initialization"])})
        module.write(
            out / "endpoint-predictions.json",
            {"test_all": [{"role": "OO", "alias_em": int(i < count)} for i in range(1283)]},
        )
        module.write(out / "endpoint.json", {"test_oo": {"alias_em": count / 1283}})
        module.write(
            out / "learning.json",
            [
                {
                    k: 512
                    for k in [
                        "examples",
                        "supervised_tokens",
                        "executed_input_tokens",
                        "effective_input_tokens",
                        "estimated_matmul_training_flops",
                    ]
                }
            ],
        )
        for step in config["evaluation_nodes"]:
            np.savez(out / f"exposure-{step:07d}.npz", counts=np.array([step, step]))
        for name in ["full-curve", "historical-baseline-curve"]:
            module.write(out / name / "complete.json", {"passed": True})
            module.write(out / name / "curve.json", [])
    module.report(config)
    result = module.read(Path(config["results_root"]) / "report.json")
    assert len(result["cells"]) == 8
    assert result["mean_contrasts"]["interaction_pp"] == pytest.approx(700 / 1283)
    assert result["mean_contrasts"]["joint_recipe_minus_baseline_pp"] == pytest.approx(5700 / 1283)
    out = Path(config["results_root"]) / "runs" / config["runs"][0]["name"]
    np.savez(out / "exposure-0300000.npz", counts=np.array([300001, 300000]))
    with pytest.raises(AssertionError):
        module.report(config)
