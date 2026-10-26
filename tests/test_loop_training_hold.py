"""A user hold must stop queue construction before any launch or credential access."""

import importlib

import pytest


@pytest.mark.parametrize("module", ["launch_loop_transfer", "launch_loop_robustness"])
@pytest.mark.parametrize("contents", ["{}", "incomplete hold record"])
def test_user_hold_blocks_even_without_queue_configuration(tmp_path, module, contents):
    launcher = importlib.import_module(module)
    (tmp_path / "training-hold.json").write_text(contents)
    with pytest.raises(RuntimeError, match="held by the user"):
        launcher.Controller(tmp_path, tmp_path / "nonexistent-token")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["training-hold.json"]
