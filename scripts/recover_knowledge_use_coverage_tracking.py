"""Verify already-uploaded node results without model loading or training."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import run_knowledge_use_coverage as runner


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def recover(batch, credential, output):
    token = credential.read_text().strip()
    authorization = base64.b64encode(f"api:{token}".encode()).decode()
    query = """query Run($entity:String!,$project:String!,$run:String!) {
      project(name:$project,entityName:$entity) {
        run(name:$run) { name state config summaryMetrics }
      }
    }"""
    receipts = {}
    for arm in "ABC":
        run = batch / "runs" / f"development-pd-{arm}-addition"
        node = run / "node-200192"
        tracking = read(batch / "tracking-wandb" / run.name / "state.json")
        request = urllib.request.Request(
            "https://api.wandb.ai/graphql",
            data=json.dumps(
                {
                    "query": query,
                    "variables": {
                        "entity": tracking["entity"],
                        "project": tracking["project"],
                        "run": tracking["run_id"],
                    },
                }
            ).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Basic {authorization}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.load(response)
        if value.get("errors"):
            raise RuntimeError("Cloud query failed; no recovery receipt written")
        remote = value["data"]["project"]["run"]
        summary = json.loads(remote["summaryMetrics"])
        metadata, complete = read(run / "run.json"), read(run / "complete.json")
        assert remote["state"] == "finished" and remote["name"] == tracking["run_id"]
        assert json.loads(remote["config"])["spec"]["value"] == metadata["spec"]
        assert summary["scientific_final_step"] == complete["step"]
        assert summary["exposure/supervised_tokens"] == complete["supervised_tokens"]
        payload = {
            "audit": read(node / "audit.json"),
            "evaluation": read(node / "evaluation/summary.json"),
        }
        assert payload["audit"]["passed"] and payload["evaluation"]["complete"]
        assert payload["audit"]["weights_sha256"] == digest(node / "resume.pt")
        assert runner.node_summary_equal(summary, f"nodes/{node.name}", payload)
        snapshot = output / f"{arm}-verified-remote.json"
        write_new(snapshot, remote)
        receipt = {
            "passed": True,
            "url": tracking["url"],
            "state": remote["state"],
            "run_id": tracking["run_id"],
            "config_equal": True,
            "node_summary_equal": True,
            "recovery": "CPU readback of previously uploaded flattened keys; no new model work",
            "comparison": "dotted scalar keys; float tolerance 1e-14; counts and identities exact",
            "remote_snapshot_sha256": digest(snapshot),
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        write_new(node / "cloud-verification.json", receipt)
        receipts[arm] = receipt
        print(json.dumps({"arm": arm, "cloud_verified": True}), flush=True)
    write_new(
        output / "complete.json",
        {"complete": True, "new_model_forwards": 0, "new_updates": 0, "arms": receipts},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--credential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        recover(args.batch, args.credential, args.output)
    except Exception:
        print(
            "Cloud recovery failed; preserve completed receipts and original evaluations",
            file=sys.stderr,
        )
        raise
