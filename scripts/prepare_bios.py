"""Fetch pinned original bioS materials, generate development worlds and audit them."""

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from llm_memory_editability.bios_data import (
    N_QUERIES,
    REVISION,
    array_hash,
    curriculum,
    make_world,
    paired_edit,
    write_json,
)


def prepare(args):
    source = Path(args.source)
    expected = json.loads(Path("configs/bios-source-manifest.json").read_text())
    base = f"https://raw.githubusercontent.com/zhuzeyuan/PhysicsLM4/{REVISION}/"

    def fetch(entry):
        path = source / entry["path"]
        content = (
            path.read_bytes()
            if path.exists()
            else urllib.request.urlopen(base + entry["path"], timeout=60).read()
        )
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("Official source checksum mismatch: " + entry["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, expected["files"]))
    write_json(source / "manifest.json", expected)
    for seed in args.worlds:
        world = make_world(seed, source)
        out = Path(args.output) / f"world-{seed}"
        world.save(out)
        edits = []
        for k in (1, 4, 16):
            for support in (0, 1):
                pair = paired_edit(world, support, k)
                np.savez_compressed(
                    out / f"edit-k{k}-support{support}.npz",
                    **{key: value for key, value in pair.items() if isinstance(value, np.ndarray)},
                )
                edits.append(
                    {
                        "k": k,
                        "support": support,
                        "E": len(pair["E"]),
                        "D": len(pair["D"]),
                        "coherent_sha256": array_hash(pair["coherent"]),
                        "exception_sha256": array_hash(pair["exception"]),
                    }
                )
        schedules = {order: curriculum(world, order, args.steps) for order in ("SA", "AS")}
        counts = {
            order: np.bincount(s.ravel(), minlength=N_QUERIES) for order, s in schedules.items()
        }
        assert np.array_equal(counts["SA"], counts["AS"])
        assert np.array_equal(schedules["SA"][5280:], schedules["AS"][5280:])
        write_json(
            out / "development-contract.json",
            {
                "steps": args.steps,
                "edits": edits,
                "equal_per_query_exposure": True,
                "exposure_sha256": array_hash(counts["SA"]),
                "schedule_sha256": {k: array_hash(v) for k, v in schedules.items()},
                "natural_language_status": "pending; not certified by symbolic audit",
            },
        )
        print(json.dumps({"world": seed, "status": "symbolic_contract_passed", "output": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/bios-source")
    parser.add_argument("--output", default="data/bios-work-v1")
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=13280)
    prepare(parser.parse_args())
