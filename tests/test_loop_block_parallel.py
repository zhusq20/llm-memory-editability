"""Reject adopting a container from another experiment, image, GPU or config."""

import copy
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/execute_loop_block_parallel.py"
    spec = importlib.util.spec_from_file_location("parallel_loop", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize("changed", [None, "batch", "run", "image", "gpu", "config"])
def test_adoption_requires_exact_ownership(changed):
    m = module()
    config = dict(batch="loop-block-depth-v1", runtime=dict(image="sha256:correct"))
    info = dict(
        Config=dict(
            Labels=dict(batch=config["batch"], run="l4-r2"), Cmd=["--config", str(m.FROZEN)]
        ),
        Image="sha256:correct",
        HostConfig=dict(DeviceRequests=[dict(DeviceIDs=["0"])]),
    )
    altered = copy.deepcopy(info)
    if changed in ["batch", "run"]:
        altered["Config"]["Labels"][changed] = "other"
    elif changed == "image":
        altered["Image"] = "other"
    elif changed == "gpu":
        altered["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["5"]
    elif changed == "config":
        altered["Config"]["Cmd"] = ["--config", "other.json"]
    if changed:
        with pytest.raises(AssertionError):
            m.ownership(altered, config, 0, "l4-r2")
    else:
        m.ownership(altered, config, 0, "l4-r2")
