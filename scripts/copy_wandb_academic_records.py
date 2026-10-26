"""Copy W&B records to the authorized academic team; never mutate the source."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import socket
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "zhusq20/llm-memory-editability"
DESTINATION = "zsqzz/llm-memory-editability"
TERMINAL = {"finished", "failed", "crashed", "killed"}
ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def history_digest(rows):
    # W&B omits explicit null fields when reindexing sparse history. They denote
    # missing measurements, not zero. Preserve raw snapshots; compare non-null values.
    return digest([{k: v for k, v in row.items() if v is not None} for row in rows])


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def history_chunks(rows, max_bytes=2_000_000):
    """Preserve sparse original steps, timestamps and values; offset counts rows."""
    offset, lines, size = 0, [], 0
    for row in rows:
        line = canonical(row)
        line_bytes = len(line.encode())
        if lines and size + line_bytes > max_bytes:
            yield offset, lines
            offset += len(lines)
            lines, size = [], 0
        lines.append(line)
        size += line_bytes
    if lines:
        yield offset, lines


class Cloud:
    def __init__(self, key):
        import requests
        import wandb

        self.api = wandb.Api(api_key=key, timeout=45)
        self.session = requests.Session()
        self.session.auth = ("api", key)
        self.session.headers["User-Agent"] = "lm-memory-academic-record-copy/1"

    def post(self, url, payload):
        for attempt in range(6):
            response = self.session.post(url, json=payload, timeout=60)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(30, 2 ** (attempt + 1)))
                continue
            result = response.json()
            if result.get("errors"):
                raise RuntimeError(str(result["errors"])[:1000])
            response.raise_for_status()
            return result
        raise RuntimeError("W&B retry budget exceeded")

    def upsert(self, metadata, state):
        # This mutation and file_stream protocol match the installed W&B SDK.
        entity, project = DESTINATION.split("/")
        mutation = """mutation CopyRun($entity: String!, $project: String!,
          $name: String!, $display: String!, $state: String, $config: JSONString!,
          $summary: JSONString!, $notes: String!, $group: String, $job: String,
          $tags: [String!]) {
          upsertBucket(input: {entityName: $entity, modelName: $project, name: $name,
            displayName: $display, state: $state, config: $config,
            summaryMetrics: $summary, notes: $notes, groupName: $group,
            jobType: $job, tags: $tags}) { bucket { name } }
        }"""
        notes = (
            f"Copied from: {metadata['url']}\n"
            f"Original created_at: {metadata['created_at']}\n"
            f"Original state: {metadata['state']}\n"
            "Historical record copy; no new training. Config, summary and full metric "
            "history copied; original run files/artifacts remain at the source.\n---\n"
            + (metadata["notes"] or "")
        )
        return self.post(
            "https://api.wandb.ai/graphql",
            {
                "query": mutation,
                "variables": {
                    "entity": entity,
                    "project": project,
                    "name": metadata["id"],
                    "display": metadata["name"],
                    "state": state,
                    "config": canonical(metadata["raw_config"]),
                    "summary": canonical(metadata["summary"]),
                    "notes": notes,
                    "group": metadata["group"],
                    "job": metadata["job_type"],
                    "tags": sorted(
                        set(
                            metadata["tags"]
                            + ["copied-from-zhusq20", f"source-state-{metadata['state']}"]
                        )
                    ),
                },
            },
        )


def verify(cloud, metadata, rows):
    target = cloud.api.run(f"{DESTINATION}/{metadata['id']}")
    copied = list(target.scan_history(page_size=1000))
    # Compare every original summary value, not server bookkeeping additions.
    summary = dict(target.summary)
    checks = {
        "config": canonical(target.config) == canonical(metadata["config"]),
        "summary": all(
            k in summary and canonical(summary[k]) == canonical(v)
            for k, v in metadata["summary"].items()
            if k != "_wandb"
        ),
        "history": history_digest(copied) == history_digest(rows),
        "state": target.state == ("finished" if metadata["state"] == "finished" else "failed"),
        "source_state": f"source-state-{metadata['state']}" in target.tags,
        "name": target.name == metadata["name"],
        "group": target.group == metadata["group"],
        "job_type": target.job_type == metadata["job_type"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source_rows": len(rows),
        "target_rows": len(copied),
        "source_history_sha256": digest(rows),
        "target_history_sha256": digest(copied),
        "source_nonnull_history_sha256": history_digest(rows),
        "target_nonnull_history_sha256": history_digest(copied),
        "history_comparison": "Exact non-null fields; absent and explicit null both mean missing",
    }


def copy_one(run_id, output, key):
    folder = output / "runs" / run_id
    receipt_path = folder / "receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()).get("state") == "verified":
        return json.loads(receipt_path.read_text())
    if receipt_path.exists():
        save(folder / "attempts" / f"{time.time_ns()}.json", json.loads(receipt_path.read_text()))
    result = {
        "run_id": run_id,
        "source": f"{SOURCE}/{run_id}",
        "target": f"{DESTINATION}/{run_id}",
        "updated_utc": now(),
    }
    try:
        cloud = Cloud(key)
        run = cloud.api.run(f"{SOURCE}/{run_id}")
        if run.state not in TERMINAL:
            result.update(state="waiting_for_source", source_state=run.state)
            save(receipt_path, result)
            return result
        marker = f"Copied from: {run.url}"
        matches = list(cloud.api.runs(DESTINATION, filters={"name": run_id}, per_page=1))
        if matches and marker not in (matches[0].notes or ""):
            raise RuntimeError("Destination ID belongs to an unrelated run; refusing overwrite")
        raw_config = run.json_config
        if isinstance(raw_config, str):
            raw_config = json.loads(raw_config)
        metadata = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "url": run.url,
            "config": run.config,
            "raw_config": raw_config,
            "summary": dict(run.summary),
            "group": run.group,
            "job_type": run.job_type,
            "notes": run.notes,
            "tags": run.tags,
        }
        rows = list(run.scan_history(page_size=1000))
        save(folder / "source-metadata.json", metadata)
        save(folder / "source-history.json", rows)
        result.update(state="copying", source_state=run.state, history_rows=len(rows))
        save(receipt_path, result)
        cloud.upsert(metadata, "running")
        endpoint = f"https://api.wandb.ai/files/{DESTINATION}/{run_id}/file_stream"
        for offset, lines in history_chunks(rows):
            cloud.post(
                endpoint, {"files": {"wandb-history.jsonl": {"offset": offset, "content": lines}}}
            )
        cloud.post(endpoint, {"complete": True, "exitcode": 0 if run.state == "finished" else 1})
        # W&B only permits running/pending in upsert. Completion is set by
        # file_stream; original crashed/killed states remain explicit in tags/notes.
        cloud.upsert(metadata, None)
        for attempt in range(5):
            cloud.api.flush()
            check = verify(cloud, metadata, rows)
            if check["passed"]:
                break
            time.sleep(min(30, 3 * (attempt + 1)))
        result.update(
            state="verified" if check["passed"] else "verification_failed", verification=check
        )
    except Exception as exc:
        result.update(state="error", error=str(exc).replace(key, "[REDACTED]")[:1500])
    result["updated_utc"] = now()
    save(receipt_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/wandb-academic-copy-20261019"
    )
    parser.add_argument("--run-id", help="Copy and verify one pilot before bulk submission")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--watch-hours", type=float, default=48)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "execution.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    key = args.token_file.read_text().strip()
    os.environ["WANDB_CONFIG_DIR"] = str(args.output / "sdk-config")
    cloud = Cloud(key)
    # Creates only the explicitly authorized destination project, idempotently.
    cloud.api.create_project("llm-memory-editability", entity="zsqzz")
    if args.run_id:
        result = copy_one(args.run_id, args.output, key)
        print(json.dumps(result), flush=True)
        return 0 if result["state"] == "verified" else 1
    save(
        args.output / "process.json",
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_utc": now(),
            "source": SOURCE,
            "destination": DESTINATION,
        },
    )
    deadline = time.monotonic() + args.watch_hours * 3600
    first_pass = True
    while True:
        inventory = [
            {"id": r.id, "name": r.name, "state": r.state, "created_at": r.created_at}
            for r in cloud.api.runs(SOURCE, order="-created_at", per_page=100)
        ]
        save(args.output / "inventory.json", inventory)
        results = []

        def status(phase, inventory=inventory, results=results):
            save(
                args.output / "status.json",
                {
                    "phase": phase,
                    "updated_utc": now(),
                    "source": SOURCE,
                    "destination": DESTINATION,
                    "source_runs": len(inventory),
                    "processed_this_pass": len(results),
                    "counts": dict(Counter(r["state"] for r in results)),
                    "scientific_updates": 0,
                    "scope": (
                        "config, summary, complete metric history, "
                        "names/groups/tags/state; source links retained"
                    ),
                    "files_artifacts_copied": False,
                    "watch_hours": args.watch_hours,
                },
            )

        status("copying")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(copy_one, r["id"], args.output, key): r["id"] for r in inventory}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                status("copying")
                print(
                    json.dumps({k: result.get(k) for k in ["run_id", "state", "error"]}), flush=True
                )
        pending = any(r["state"] == "waiting_for_source" for r in results)
        errors = any(r["state"] in {"error", "verification_failed"} for r in results)
        if not pending and not errors:
            status("complete")
            return 0
        if time.monotonic() >= deadline:
            status("attention_required")
            return 1
        status("waiting_for_source" if pending and not errors else "retry_pending")
        # The detached controller waits; foreground checks remain responsive.
        time.sleep(300 if first_pass else 600)
        first_pass = False
        cloud.api.flush()


if __name__ == "__main__":
    raise SystemExit(main())
