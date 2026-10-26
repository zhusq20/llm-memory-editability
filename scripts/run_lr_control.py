#!/usr/bin/env python3
"""Run one frozen LR-control stage through the unchanged native trainer."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from run_discoloop_restore import CurveTracker  # noqa: E402

from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.lr_control import (  # noqa: E402
    BASE_SPEC,
    INITIAL_SHA,
    PANELS_SHA,
    PARAMETERS,
    WORLD_SHA,
    decay_contract,
    initialization_config,
    job_config,
    rate_at,
    validate,
)
from llm_memory_editability.path_overlap import sha256  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def load_native(config, *, smoke=False):
    spec = importlib.util.spec_from_file_location(
        "lr_control_native", ROOT / "scripts/run_supervised_alignment.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    original_initialize = native.initialize

    def initialize(candidate, device, arm):
        model, metadata, provenance = original_initialize(
            initialization_config(candidate), device, arm
        )
        if provenance["initial_model_sha256"] != INITIAL_SHA:
            raise ValueError("Exact original initialization differs")
        if metadata["world_sha256"] != WORLD_SHA or metadata["panels_sha256"] != PANELS_SHA:
            raise ValueError("Original world/panels differ")
        if sum(p.numel() for p in model.parameters()) != PARAMETERS:
            raise ValueError("Original model shape differs")
        provenance["lr_control"] = {
            "source_peak_lr": BASE_SPEC["lr"],
            "source_weight_decay": BASE_SPEC["weight_decay"],
            "training_peak_lr": candidate["spec"]["lr"],
            "training_weight_decay": candidate["spec"]["weight_decay"],
            "initialization_only_uses_original_spec": True,
        }
        return model, metadata, provenance

    native.initialize = initialize
    original_graph = native.AlignmentStep
    original_values = native.graph_values

    class ObservedStep(original_graph):
        """Measure one actual update per heartbeat outside CUDA capture; no RNG."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observed_calls = 0
            self.update_measurement = {}

        def __call__(self, indices):
            self.observed_calls += 1
            interval = 8 if smoke else 1000
            if self.observed_calls % interval:
                return super().__call__(indices)
            with torch.no_grad():
                parameters = tuple(self.model.parameters())
                before = [p.detach().clone() for p in parameters]
            result = super().__call__(indices)
            with torch.no_grad():
                delta2 = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
                norm2 = delta2.clone()
                count = 0
                for old, current in zip(before, parameters, strict=True):
                    delta2 += (current - old).square().sum(dtype=torch.float64)
                    norm2 += old.square().sum(dtype=torch.float64)
                    count += old.numel()
                self.update_measurement = {
                    "measured_update_step": self.observed_calls,
                    "parameter_update_l2": float(delta2.sqrt()),
                    "parameter_update_rms": float((delta2 / count).sqrt()),
                    "parameter_norm_l2_before": float(norm2.sqrt()),
                    "relative_parameter_update_l2": float((delta2 / norm2).sqrt()),
                }
            return result

    native.AlignmentStep = ObservedStep

    def values(graph, coefficient):
        return {**original_values(graph, coefficient), **graph.update_measurement}

    native.graph_values = values
    return native


def preflight(config, native):
    name = config["job_name"]
    model, metadata, provenance = native.initialize(config, "cuda:0", config["arms"][name])
    if not provenance["constructor_matches_source_weights"]:
        raise ValueError("Original initialization must also reconstruct from its seed")
    world = dict(native.np.load(Path(config["input_root"]) / "data/world.npz"))
    table, selected = native.make_table(world, metadata, config["spec"], "cuda:0")
    stream = native.MergedBatchStream(len(table[0]) - 1, config["spec"]["stream_seed"])
    indices = torch.as_tensor(stream.batch(config["spec"]["batch_size"])[0], device="cuda:0")
    diagnostic = native.gradient_diagnostic(
        model, table, indices, config["arms"][name], torch.device("cuda:0")
    )
    receipt = decay_contract(config["spec"]["lr"])
    steps = torch.arange(1, 300001, dtype=torch.float64, device="cuda:0")
    warmup = ((steps - 1) / BASE_SPEC["warmup"]).clamp(max=1)
    ref_lr = (warmup * BASE_SPEC["lr"]).float()
    new_lr = (warmup * config["spec"]["lr"]).float()
    ref_product = ref_lr.double() * BASE_SPEC["weight_decay"]
    new_product = new_lr.double() * config["spec"]["weight_decay"]
    factor_difference = ((1 - ref_product).float() - (1 - new_product).float()).abs()
    receipt["fp32"] = {
        "maximum_product_error": float((ref_product - new_product).abs().max()),
        "maximum_multiplier_error": float(factor_difference.max()),
        "different_multipliers": int(torch.count_nonzero(factor_difference)),
    }
    if receipt["fp32"]["maximum_product_error"] > 2e-12:
        raise AssertionError("Quantized-LR decay differs beyond FP32 rounding")
    if receipt["fp32"]["maximum_multiplier_error"] > torch.finfo(torch.float32).eps:
        raise AssertionError("FP32 multiplicative decay differs by more than one ULP")
    optimizer = native.optimizer_for(
        model, torch.tensor(0.0, device="cuda:0"), config["spec"]["weight_decay"]
    )
    if [group["weight_decay"] for group in optimizer.param_groups] != [
        config["spec"]["weight_decay"],
        0.0,
    ]:
        raise AssertionError("Original decay/no-decay groups changed")
    probe = torch.randn(64, 64, device="cuda:0", requires_grad=True)
    probe.square().matmul(probe).mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(probe.grad).all():
        raise FloatingPointError("Actual CUDA forward/backward failed")
    return {
        "passed": True,
        "scientific_updates": 0,
        "job": name,
        "provenance": provenance,
        "composition_examples": len(selected),
        "gradient_diagnostic": diagnostic,
        "decay": receipt,
        "actual_cuda_forward_backward_synchronized": True,
    }


class Tracker(CurveTracker):
    def __init__(self, run, path, config):
        super().__init__(run, path)
        self.config = config

    def drain(self):
        if self.run is None or not (self.path / "learning.json").exists():
            return
        for record in read(self.path / "learning.json"):
            step = record["step"]
            if step in self.logged:
                continue
            values = {"training/step": step}
            for name, panel in record["metrics"].items():
                values.update({f"{name}/{key}": value for key, value in panel.items()})
            values.update(
                {f"cost/{k}": v for k, v in record.items() if isinstance(v, (float, int))}
            )
            values["optimizer/lr"] = rate_at(self.config["spec"]["lr"], max(1, step))
            values["optimizer/weight_decay"] = self.config["spec"]["weight_decay"]
            self.run.log(values, step=step)
            self.logged.add(step)


def cloud_audit(config, out):
    import wandb

    api, findings = wandb.Api(timeout=60), []
    for stage in ("train", "audit"):
        tracking = read(out / f"wandb-{stage}.json")
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        path = out / f"result-{stage}.json"
        expected = sha256(path)
        if (
            run.state != "finished"
            or run.config.get("batch") != config["batch"]
            or run.config.get("job_name") != config["job_name"]
            or run.summary.get("result_sha256") != expected
        ):
            raise ValueError(f"Cloud state/config/result hash differs: {stage}")
        if stage == "train" and run.summary.get("terminal_step") != 300000:
            raise ValueError("Cloud endpoint differs")
        folder = out / "cloud-verification" / stage
        folder.mkdir(parents=True, exist_ok=True)
        downloaded = run.file(path.name).download(root=str(folder), replace=True)
        if sha256(downloaded.name) != expected:
            raise ValueError("Canonical complete cloud JSON differs")
        findings.append({**tracking, "stage": stage, "result_sha256": expected})
    return {"passed": True, "runs": findings, "canonical_json_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("preflight", "smoke", "smoke-audit", "train", "audit", "cloud-audit")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    base = read(args.config)
    validate(base)
    config = job_config(base, args.job)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = out / f"result-{args.stage}.json"
        failure_path = out / f"failure-{args.stage}.json"
        if result_path.exists() or failure_path.exists():
            raise FileExistsError("Preserve completed/failed stages; no automatic rerun")
        run = None
        try:
            if args.stage == "cloud-audit":
                result = cloud_audit(config, out)
            else:
                torch.set_num_threads(4)
                torch.set_num_interop_threads(1)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.cuda.set_device(0)
                write_json(
                    out / f"environment-{args.stage}.json",
                    {
                        "utc": utc(),
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0),
                        "pid": os.getpid(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)),
                        "config_sha256": sha256(args.config),
                        "tf32": True,
                    },
                )
                if args.stage in ("train", "audit"):
                    import wandb

                    run = wandb.init(
                        entity="zhusq20",
                        project="llm-memory-editability",
                        name=f"{config['batch']}/{args.job}/{args.stage}",
                        group=config["batch"],
                        job_type=args.stage,
                        config=config,
                        mode="online",
                        dir=str(out),
                        save_code=False,
                        settings=wandb.Settings(disable_git=True),
                    )
                    write_json(out / f"wandb-{args.stage}.json", {"id": run.id, "url": run.url})
                    print(json.dumps({"wandb": run.url}), flush=True)
                native = load_native(config, smoke=args.stage in ("smoke", "smoke-audit"))
                science = out / ("smoke" if args.stage in ("smoke", "smoke-audit") else "training")
                if args.stage == "preflight":
                    result = preflight(config, native)
                elif args.stage in ("train", "smoke"):
                    with Tracker(run, science, config):
                        native.run(config, args.job, science, args.stage == "smoke", args.config)
                    result = read(science / "complete.json")
                else:
                    native.audit(config, args.job, science, args.config)
                    result = read(science / "audit.json")
                    if args.stage == "audit":
                        architecture = config["jobs"][args.job]["architecture"]
                        with native.np.load(science / "exposure.npz") as current:
                            with native.np.load(
                                Path("/baselines") / architecture / "exposure.npz"
                            ) as baseline:
                                matched = all(
                                    native.np.array_equal(current[key], baseline[key])
                                    for key in ("counts", "selected")
                                )
                        if not matched:
                            raise ValueError("Sample exposures or selected compositions differ")
                        result["exposure_matches_reused_1e4_control"] = True
            write_json(result_path, result)
            if run is not None:
                run.summary.update(
                    {
                        "result_sha256": sha256(result_path),
                        "terminal_step": result.get("step"),
                        "passed": result.get("passed", result.get("state") == "complete"),
                        "scientific_updates": result.get("step") if args.stage == "train" else 0,
                    }
                )
                run.save(str(result_path), base_path=str(out), policy="now")
                run.finish()
                run = None
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "job": args.job,
                        "result": str(result_path),
                        "passed": result.get("passed"),
                    }
                ),
                flush=True,
            )
            if result.get("passed") is False:
                raise RuntimeError("Stage validation failed")
        except BaseException as error:
            write_json(
                failure_path,
                {"error": repr(error), "traceback": traceback.format_exc(), "utc": utc()},
            )
            if run is not None:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    main()
