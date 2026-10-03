"""Preserve and fully evaluate the pre-existing 32k addressability node."""

import json
import os
import shutil
import time

import diagnose_realworld_addressability as experiment
import torch

common = experiment.common
ROOT = common.PROJECT / "results/realworld-loop-addressability-midpoint-v1"


def main():
    device = common.configure(7)
    for arch in ["standard8", "loop4x2"]:
        for condition in ["shared", "unique"]:
            source = experiment.ROOT / "development" / f"{arch}-{condition}"
            while json.loads((source / "status.json").read_text())["step"] <= 32000:
                time.sleep(5)
            out = ROOT / "development" / source.name
            out.mkdir(parents=True, exist_ok=True)
            checkpoint = out / "checkpoint-0032000.pt"
            digest = common.sha256(source / "latest.pt")
            shutil.copy2(source / "latest.pt", checkpoint)
            assert common.sha256(checkpoint) == common.sha256(source / "latest.pt") == digest
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert saved["step"] == 32000
            data, manifest = experiment.load(condition)
            model = common.construct(manifest["model"], experiment.spec_for(arch), device)
            model.load_state_dict(saved["model"])
            del saved
            before = common.model_digest(model)
            codec = experiment.Codec()
            _, control = common.evaluate_selected(model, data, [], codec, device, False)
            assert control == json.loads((source / "predictions-0032000.json").read_text())
            metrics, predictions = common.evaluate_selected(model, data, [], codec, device, True)
            assert common.model_digest(model) == before
            common.write_json(
                out / "run.json",
                {
                    "spec": {
                        "arch": arch,
                        "condition": condition,
                        "source_step": 32000,
                        "steps": 0,
                    },
                    "pid": os.getpid(),
                    "source": str(source),
                    "checkpoint_sha256": digest,
                    "tracking_group": "realworld-loop-diagnosis-v1",
                    "job_type": "midpoint-full-eval",
                },
            )
            common.write_json(out / "learning.json", [{"step": 32000, "metrics": metrics}])
            common.write_json(out / "endpoint.json", metrics)
            common.write_json(out / "endpoint-predictions.json", predictions)
            common.write_json(
                out / "complete.json",
                {
                    "passed": True,
                    "panel_exact_reload_control": True,
                    "model_unchanged": True,
                    "utc": common.utc(),
                },
            )
            print(
                json.dumps({"run": source.name, "OO": metrics["test_oo"]["alias_em"]}), flush=True
            )
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
