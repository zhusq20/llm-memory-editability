"""Read back existing W&B endpoints without changing runs or training."""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def verify(batch, credential, output):
    token = credential.read_text().strip()
    authorization = base64.b64encode(f"api:{token}".encode()).decode()
    query = """query Run($entity:String!,$project:String!,$run:String!) {
      project(name:$project,entityName:$entity) {
        run(name:$run) { name state config summaryMetrics }
      }
    }"""
    result = {"utc": datetime.now(timezone.utc).isoformat(), "runs": {}}
    for record in sorted((batch / "tracking").glob("*.json")):
        tracking = json.loads(record.read_text())
        payload = json.dumps(
            {
                "query": query,
                "variables": {
                    "entity": "zsqzz",
                    "project": "llm-memory-editability",
                    "run": tracking["id"],
                },
            }
        ).encode()
        request = urllib.request.Request(
            "https://api.wandb.ai/graphql",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Basic {authorization}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.load(response)
        if value.get("errors"):
            raise RuntimeError("W&B returned GraphQL errors; preserve prior evidence")
        run = value["data"]["project"]["run"]
        if run is None:
            raise RuntimeError(f"Missing cloud run: {tracking['id']}")
        summary = json.loads(run["summaryMetrics"] or "{}")
        config = json.loads(run["config"] or "{}")
        result["runs"][record.stem] = {
            **tracking,
            "state": run["state"],
            "summary": summary,
            "config": {k: v for k, v in config.items() if k != "_wandb"},
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "states": {name: value["state"] for name, value in result["runs"].items()},
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--credential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.batch, args.credential, args.output)
