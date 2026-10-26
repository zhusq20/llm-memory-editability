#!/usr/bin/env python3
"""Run already authorized depth jobs on the user's released GPU4.

Keep the original scheduler and frozen scientific files intact. Its GPU4 lane
remains blocked by the historical Created reservations; atomic per-job claims
prevent its other lanes from duplicating the transferred jobs.
"""

import json
import os

import migrate_gate_extension_gpu6 as receiver

GPU = 4
UUID = "GPU-b4302c0e-77f9-f270-ae1b-b95aba3f3c7f"
PLACEHOLDERS = (
    (
        "default",
        "2aa7819426aa6e97c99b7ab18fdb1bfb1e82f0705675a42db3f0db82f40f0766",
        "/medium-k14-allmetrics-batch1-20261013-unused",
    ),
    (
        "default",
        "3c6ee821f04871a4d22ff3cb7c745927f459484cfa81828172ac580b38235175",
        "/medium-taylor-k14-start15-shape01-500-to1500-20261009",
    ),
    (
        "lm-memory",
        "cce540699bf932746933fdd6377d3e52b4d65c2723ef8a554a40d3721044b8f8",
        "/alphaedit-discoloop-novel-v1-development",
    ),
    (
        "lm-memory",
        "2acd7eef72894465d9c7210270797a7fccf123cc077217bbaf0c1a02a71ad4eb",
        "/ttcd-successful-base-v1-repair-verification-v2",
    ),
)


def inactive(container, identifier, name):
    state = container.get("State", {})
    return (
        container.get("Id") == identifier
        and container.get("Name") == name
        and state.get("Status") == "created"
        and state.get("StartedAt") == "0001-01-01T00:00:00Z"
        and state.get("Pid") == 0
        and not state.get("Running")
        and container.get("HostConfig", {}).get("RestartPolicy", {}).get("Name") == "no"
    )


def filter_snapshot(common, rows, blocked):
    # Inspect exact identities on every admission; never suppress processes,
    # running containers, changed identities or any other reservation.
    blocked = {gpu: list(names) for gpu, names in blocked.items()}
    for context, identifier, name in PLACEHOLDERS:
        container = json.loads(common.docker("inspect", identifier, context=context).stdout)[0]
        if inactive(container, identifier, name):
            blocked[GPU] = [entry for entry in blocked.get(GPU, []) if entry != name]
    return rows, blocked


def select_jobs(plan, identifiers):
    expected = [f"depth-1--{arm}" for arm in ("paper", "fixed-update", "token-update")]
    if identifiers != expected:
        raise ValueError("Only the three original depth-1 jobs may be transferred")
    lookup = {job["id"]: job for job in plan["jobs"]}
    jobs = [lookup[identifier] for identifier in identifiers]
    if any(j["family"] != "depth" or j["start_step"] != 0 or j["steps"] != 300000 for j in jobs):
        raise ValueError("Frozen depth budget changed")
    return jobs


if __name__ == "__main__":
    os.umask(0o022)
    receiver.GPU = GPU
    receiver.UUID = UUID
    receiver.IMAGE = "sha256:b7933a57d6a4702700b6b2578fe0cc697f91b2080ef5e18d8fb916443281eacd"
    receiver.CPUS = [84, 85]
    receiver.LEASE_PREFIX = "lm-memory-gate-extension-handoff"
    receiver.SNAPSHOT_FILTER = filter_snapshot
    receiver.select_jobs = select_jobs
    receiver.main()
