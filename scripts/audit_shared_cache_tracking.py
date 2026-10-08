"""Read-only verification of final W&B runs against local audited artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path

import wandb

ROOT = Path("/ossfs/workspace/llm-memory-editability/results/shared-cache-branch-v1")


def read(path):
    return json.loads(path.read_text())


def main():
    state = read(ROOT / "controller-state.json")
    assert state["state"] == "complete" and not state["failed"]
    assert read(ROOT / "tracking-completion.json")["passed"]
    api = wandb.Api(timeout=30)
    records = []
    for name in state["completed"]:
        local = ROOT / "runs" / name
        tracking = read(ROOT / "tracking-wandb" / name / "state.json")
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['run_id']}")
        last = read(local / "learning.json")[-1]["step"]
        assert read(local / "audit.json")["passed"]
        assert run.name == name and run.group == "shared-cache-branch-v1"
        assert run.state == "finished", (name, run.state)
        assert run.summary.get("_step") == last, (name, run.summary.get("_step"), last)
        assert run.summary.get("independently_reloaded") is True
        assert tracking["last_logged_step"] == last
        records.append(dict(name=name, id=run.id, state=run.state, step=last, url=run.url))
    result = dict(
        passed=True,
        registered_runs=len(records),
        runs=records,
        utc=datetime.now(timezone.utc).isoformat(),
    )
    (ROOT / "cloud-final-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "runs": len(records)}))


if __name__ == "__main__":
    main()
