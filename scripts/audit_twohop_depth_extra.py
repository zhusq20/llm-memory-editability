#!/usr/bin/env python3
"""Post-hoc precision and matched-batch checks; never changes trained models."""

import argparse

import numpy as np
import torch

from llm_memory_editability.twohop_depth import (
    ART,
    DATA,
    ENTITY,
    RESULTS,
    amp,
    build_model,
    digest,
    evaluate,
    patched_logits,
    read,
    states_and_logits,
    write,
)


@torch.no_grad()
def main(device):
    audit = read(ART / "audit.json")
    assert audit["complete"] and audit["runs"] == 28 and audit["nodes"] == 196
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    lock = read(ART / "lock.json")
    cfg = lock["config"]
    checked, batch_differences, fp32_checks = 0, [], []
    for job in lock["jobs"]:
        w = dict(np.load(DATA / f"world-{job['world']}.npz"))
        c, r = w["cases"], cfg["relations"]
        index = c[:, 6] * r * r + c[:, 2] * r + c[:, 3]
        q = torch.as_tensor(w["composite_x"][c[:, 0]], device=device)
        alt = q.clone()
        alt[:, 1] = torch.as_tensor(ENTITY + c[:, 6], device=device)
        model = build_model(job["arch"], cfg).to(device).eval()
        root = RESULTS / job["name"]
        for step in cfg["nodes"]:
            ckpt = torch.load(root / f"model-{step}.pt", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            d = np.load(root / f"diagnostics-{step}.npz")
            donor, _ = states_and_logits(model, alt[:, :3], device)
            z = patched_logits(model, q, 0, [1, 2], donor[0][:, [1, 2]], device)
            with amp(device):
                ordinary = model(alt)[:, -1].float()
                clean = model(q)[:, -1].argmax(-1).cpu().numpy()
            assert torch.equal(z, ordinary), (job["name"], step)
            direct = ordinary.argmax(-1).cpu().numpy()
            np.testing.assert_array_equal(d["prefix_0_pred"], direct)
            np.testing.assert_array_equal(d["source_0_pred"], direct)
            np.testing.assert_array_equal(d["clean_pred"], clean)
            p = np.load(root / f"predictions-{step}.npz")
            all_pred = np.empty(len(w["train_mask"]), dtype=np.int64)
            all_pred[w["train_mask"]] = p["train_pred"]
            all_pred[~w["train_mask"]] = p["test_pred"]
            batch_differences.append(
                {
                    "run": job["name"],
                    "step": step,
                    "counterfactual_batch64_vs512_disagreements": int(
                        np.sum(direct != all_pred[index])
                    ),
                }
            )
            checked += 1
        metrics = read(root / f"node-{cfg['steps']}.json")["metrics"]
        if metrics["atomic_exact"] < 1 or metrics["train_exact"] < 1:
            cpu_model = build_model(job["arch"], cfg).eval()
            cpu_model.load_state_dict({k: v.cpu() for k, v in ckpt["model"].items()})
            cpu_metrics, _ = evaluate(cpu_model, w, cfg, "cpu")
            fp32_checks.append(
                {"run": job["name"], "bf16_metrics": metrics, "cpu_fp32_metrics": cpu_metrics}
            )
            del cpu_model
        del model, ckpt
    write(
        ART / "supplemental-audit.json",
        {
            "status": "post-hoc verification, not a new training condition",
            "script_sha256": digest(__file__),
            "nodes": checked,
            "all_matched_batch_prefix_logits_exact": True,
            "all_matched_batch_clean_predictions_exact": True,
            "batch_disagreements": batch_differences,
            "declining_endpoint_fp32_checks": fp32_checks,
        },
    )
    print(f"Verified {checked} nodes and {len(fp32_checks)} declining endpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:5")
    main(parser.parse_args().device)
