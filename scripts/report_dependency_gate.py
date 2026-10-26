#!/usr/bin/env python3
"""Validate all eight completed dependency runs and report paired observations.

Reads JSON and saved predictions only. Never loads weights, evaluates a model,
selects a checkpoint, trains, or contacts a tracking service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ARMS = ("loop", "only-u", "disco", "fixed-q-plus-u")
TASKS = ("listops", "arithmetic")
TESTS = ("test_iid", "test_depth")
ALL_PANELS = ("train_sample", "valid_iid", "valid_depth", *TESTS)
FEATURES = ("depth", "tree_depth", "length", "branch_imbalance", "computed_branch_imbalance")
OVER = {"listops": 24, "arithmetic": 20}
NAMES = {"loop": "普通 Loop", "only-u": "仅 u", "disco": "固定 q", "fixed-q-plus-u": "固定 q＋u"}
POOLS = {"test_iid": "IID", "test_depth": "更深依赖"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text())


def _rate(correct):
    return float(np.asarray(correct, dtype=bool).mean()) if len(correct) else None


def _load_prediction(path, metrics):
    with np.load(path, allow_pickle=False) as saved:
        required = {"ids", "labels", "answer", "stop", "nll", "join_count", *FEATURES}
        _require(required <= set(saved.files), f"Missing prediction columns: {path}")
        values = {key: np.asarray(saved[key]).copy() for key in required}
    ids = values["ids"].astype("U")
    count = len(ids)
    _require(
        ids.ndim == 1 and count > 0 and np.all(ids != "") and len(np.unique(ids)) == count,
        f"Nonunique/empty example IDs: {path}",
    )
    _require(
        values["labels"].shape == values["nll"].shape == (count, 2),
        f"Answer/EOS shape mismatch: {path}",
    )
    _require(np.isfinite(values["nll"]).all(), f"Nonfinite saved NLL: {path}")
    for key in ("answer", "stop", "join_count", *FEATURES):
        _require(
            values[key].shape == (count,) and np.issubdtype(values[key].dtype, np.integer),
            f"Invalid integer prediction/metadata column {key}: {path}",
        )
    _require(np.issubdtype(values["labels"].dtype, np.integer), f"Invalid labels: {path}")
    _require(np.all(values["join_count"] >= 0), f"Negative join counts: {path}")
    order = np.argsort(ids)
    values = {key: value[order] for key, value in values.items()}
    values["ids"] = ids[order]
    correct = values["answer"] == values["labels"][:, 0]
    exact = correct & (values["stop"] == values["labels"][:, 1])
    _require(metrics.get("n") == count, f"Grid/prediction denominator mismatch: {path}")
    for key, actual in (("answer_accuracy", _rate(correct)), ("accuracy", _rate(exact))):
        _require(
            isinstance(metrics.get(key), (int, float))
            and np.isclose(metrics[key], actual, atol=1e-12, rtol=0),
            f"Grid/prediction {key} mismatch: {path}",
        )
    if "correct" in metrics:
        _require(metrics["correct"] == int(correct.sum()), f"Grid correct count differs: {path}")
    groups = {
        f"by_{feature}": {
            str(value): values[feature] == value for value in np.unique(values[feature])
        }
        for feature in FEATURES
    }
    groups["by_join_presence"] = {"0": values["join_count"] == 0, "ge1": values["join_count"] >= 1}
    for key, masks in groups.items():
        _require(
            set(metrics.get(key, {})) == set(masks), f"Incomplete group coverage {key}: {path}"
        )
        for label, mask in masks.items():
            row = metrics[key][label]
            _require(
                row.get("n") == int(mask.sum()), f"Group denominator differs {key}/{label}: {path}"
            )
            for score, actual in (
                ("answer_accuracy", _rate(correct[mask])),
                ("accuracy", _rate(exact[mask])),
            ):
                _require(
                    row.get(score) is None
                    if actual is None
                    else row.get(score) is not None
                    and np.isclose(row[score], actual, atol=1e-12, rtol=0),
                    f"Group score differs {key}/{label}/{score}: {path}",
                )
    values["correct"], values["exact"] = correct, exact
    return values


def _match(first, second, label):
    for key in ("ids", "labels", "join_count", *FEATURES):
        _require(
            np.array_equal(first[key], second[key]), f"Unpaired examples or truth: {label}/{key}"
        )


def _transitions(before, after):
    before, after = np.asarray(before, bool), np.asarray(after, bool)
    initial, final = int(before.sum()), int(after.sum())
    kept, lost, gained = (
        int((before & after).sum()),
        int((before & ~after).sum()),
        int((~before & after).sum()),
    )
    return {
        "denominator": len(before),
        "original_correct": initial,
        "final_correct": final,
        "kept": kept,
        "lost": lost,
        "gained": gained,
        "both_wrong": int((~before & ~after).sum()),
        "retention_denominator": initial,
        "retention_rate": kept / initial if initial else None,
        "loss_rate_given_original_correct": lost / initial if initial else None,
        "gain_denominator": len(before) - initial,
        "gain_rate_given_original_wrong": gained / (len(before) - initial)
        if initial < len(before)
        else None,
        "original_accuracy": initial / len(before),
        "final_accuracy": final / len(before),
        "delta_count": final - initial,
        "delta_accuracy": (final - initial) / len(before),
    }


def _pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def _pp(value):
    return f"{100 * value:+.2f}"


def _markdown(result):
    lines = [
        "八臂固定终点评测均已完成，以下结果来自保存的逐题预测；未重载模型或追加训练。",
        "主比较固定使用训练轮数，未按测试结果选择检查点或推理轮数。答案为全词表预测，"
        "“答案＋EOS”要求继续生成正确结束符。",
    ]
    for task in TASKS:
        value = result["tasks"][task]
        lines += [
            "",
            f"**{task}：{value['steps']:,} 步，训练 {value['training_repeats']} 轮**",
            "",
            "| 架构 | IID 答案 | IID 答案＋EOS | 更深依赖答案 | 更深依赖答案＋EOS |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            a, b = (value["primary"][pool][arm] for pool in TESTS)
            lines.append(
                f"| {NAMES[arm]} | {_pct(a['answer_accuracy'])} | {_pct(a['answer_eos_accuracy'])} "
                f"| {_pct(b['answer_accuracy'])} | {_pct(b['answer_eos_accuracy'])} |"
            )
        majority = value["training_majority_baseline"]
        lines += [
            "",
            f"全部分母：IID {majority['test_iid']['n']:,} 题，更深依赖 "
            f"{majority['test_depth']['n']:,} 题。固定训练多数类的答案基线分别为 "
            f"{_pct(majority['test_iid']['answer_accuracy'])}、"
            f"{_pct(majority['test_depth']['answer_accuracy'])}；类别由完整训练集预先确定。",
            "",
            "| 配对比较 | 池 | 答案差（百分点） | 答案＋EOS差（百分点） | 新增答对／丢失答对 |",
            "|---|---|---:|---:|---:|",
        ]
        for pair, pools in value["paired"].items():
            label = "仅 u − 普通 Loop" if pair == "only-u-minus-loop" else "固定 q＋u − 固定 q"
            for pool, scores in pools.items():
                answer, exact = scores["answer"], scores["answer_eos"]
                lines.append(
                    f"| {label} | {POOLS[pool]} | {_pp(answer['delta_accuracy'])} "
                    f"| {_pp(exact['delta_accuracy'])} | {answer['gained']}／{answer['lost']} |"
                )
        lines += [
            "",
            f"从训练 {value['training_repeats']} 轮增加到预定 "
            f"{value['extra_repeats']} 轮，同题答案变化如下。",
            "",
            "| 架构 | 池 | 原答对仍答对／原答对数 | 丢失 | 新增 | 准确率差（百分点） |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            for pool in TESTS:
                score = value["extra_rounds"][arm][pool]["answer"]
                lines.append(
                    f"| {NAMES[arm]} | {POOLS[pool]} | "
                    f"{score['kept']}／{score['original_correct']} "
                    f"| {score['lost']} | {score['gained']} | {_pp(score['delta_accuracy'])} |"
                )
    lines += [
        "",
        "完整依赖深度、句长、分支合并和分支深度差分组，以及答案＋EOS的跨轮转移，"
        "保存在 comparison.json；每组保留原始分母和来源哈希。",
        "这些是每任务一份数据和一个初始化下的观察。深度分组不自动等同纯逻辑深度效应；"
        "增加轮数的结果也不证明门的通用稳定性或自然语言能力。",
        "",
    ]
    return "\n".join(lines)


def _write_new_or_identical(path, content):
    path = Path(path)
    if path.exists():
        _require(
            path.read_text() == content, f"Refusing to replace a different existing report: {path}"
        )
        return
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_report(config, output):
    """Validate the entire matrix before publishing comparison.json and report.md.

    ``config`` may be its file path or a decoded mapping. Passing the path also
    verifies every saved root-config SHA against its actual bytes. ``output`` is
    the execution root containing <task>/runs/<arm>, not a single task folder.
    Missing or inconsistent inputs raise; no partial-arm summary is produced.
    """
    output = Path(output)
    config_bytes_verified = isinstance(config, (str, Path))
    if isinstance(config, (str, Path)):
        expected_config_hash, config = _sha(config), _json(config)
    else:
        expected_config_hash = config.get("_config_sha256")
    _require(
        set(config["tasks"]) == set(TASKS) and set(config["arms"]) == set(ARMS),
        "The complete two-task/four-arm matrix is required",
    )
    sources, grids, kept_predictions, config_hashes = {}, {}, {}, set()
    for task in TASKS:
        settings = config["tasks"][task]
        _require(
            isinstance(settings.get("input_origin", {}).get("train_majority_token"), int),
            f"Missing predeclared full-training majority token: {task}",
        )
        native_rounds, extra_rounds = settings["repeats"], OVER[task]
        _require(
            native_rounds < extra_rounds and extra_rounds in settings["inference_repeats"],
            f"Missing registered beyond-training recurrence: {task}",
        )
        for arm in ARMS:
            run = output / task / "runs" / arm
            complete_path, grid_path = run / "complete.json", run / "grid.json"
            complete, grid = _json(complete_path), _json(grid_path)
            _require(
                complete.get("state") == "complete"
                and complete.get("step") == complete.get("budget") == settings["training"]["steps"]
                and complete.get("arm") == arm
                and complete.get("engineering_only") is False,
                f"Unfinished or engineering run: {task}/{arm}",
            )
            _require(
                grid.get("task") == task
                and grid.get("arm") == arm
                and grid.get("step") == settings["training"]["steps"]
                and grid.get("training_repeats") == native_rounds
                and grid.get("engineering_only") is False,
                f"Endpoint identity mismatch: {task}/{arm}",
            )
            _require(
                grid.get("model_sha256") == complete.get("model_sha256")
                and bool(grid.get("model_sha256")),
                f"Endpoint model identity mismatch: {task}/{arm}",
            )
            _require(
                grid.get("config_sha256") == complete.get("config_sha256")
                and bool(grid.get("config_sha256")),
                f"Config identity mismatch: {task}/{arm}",
            )
            config_hashes.add(grid["config_sha256"])
            if expected_config_hash is not None:
                _require(
                    grid["config_sha256"] == expected_config_hash,
                    f"Saved configuration differs from the supplied file: {task}/{arm}",
                )
            for key in ("world_sha256", "panels_sha256"):
                expected = settings["input_origin"].get(key)
                _require(
                    expected is not None and grid.get(key) == expected,
                    f"Pinned data identity mismatch: {task}/{arm}/{key}",
                )
            repeats = settings["inference_repeats"]
            _require(
                grid.get("inference_repeats") == repeats
                and set(grid["metrics"]) == {str(value) for value in repeats},
                f"Incomplete inference grid: {task}/{arm}",
            )
            grids[task, arm] = grid
            for path in (complete_path, grid_path):
                sources[str(path.relative_to(output))] = _sha(path)
            for count in repeats:
                metrics = grid["metrics"][str(count)]
                _require(
                    set(metrics) == set(ALL_PANELS),
                    f"Incomplete endpoint panel grid: {task}/{arm}/r{count}",
                )
                for pool in TESTS:
                    _require(pool in metrics, f"Missing test panel: {task}/{arm}/r{count}/{pool}")
                    path = run / "grid" / f"r{count}" / f"{pool}.npz"
                    prediction = _load_prediction(path, metrics[pool])
                    sources[str(path.relative_to(output))] = _sha(path)
                    reference_key = task, pool
                    if reference_key in kept_predictions:
                        _match(kept_predictions[reference_key], prediction, str(path))
                    else:
                        kept_predictions[reference_key] = prediction
                    if count in (native_rounds, extra_rounds):
                        kept_predictions[task, arm, pool, count] = prediction
    _require(len(config_hashes) == 1, "The eight runs do not share one frozen root configuration")
    result = {
        "schema": "dependency-gate-paired-report-v1",
        "complete": True,
        "batch": config["batch"],
        "scientific_runs": 8,
        "config_sha256": next(iter(config_hashes)),
        "supplied_config_bytes_verified": config_bytes_verified,
        "scoring": "full-vocabulary answer argmax; EOS generated after the predicted answer",
        "unit": "individual examples within one data realization and initialization per task",
        "sources_sha256": sources,
        "tasks": {},
    }
    for task in TASKS:
        settings = config["tasks"][task]
        native_rounds, extra_rounds = settings["repeats"], OVER[task]
        primary, groups, majority, paired, transitions = {}, {}, {}, {}, {}
        majority_token = settings["input_origin"]["train_majority_token"]
        for pool in TESTS:
            primary[pool], groups[pool] = {}, {}
            truth = kept_predictions[task, pool]["labels"][:, 0]
            majority[pool] = {
                "n": len(truth),
                "correct": int((truth == majority_token).sum()),
                "answer_accuracy": float((truth == majority_token).mean()),
            }
            for arm in ARMS:
                prediction = kept_predictions[task, arm, pool, native_rounds]
                primary[pool][arm] = {
                    "n": len(truth),
                    "correct": int(prediction["correct"].sum()),
                    "answer_accuracy": _rate(prediction["correct"]),
                    "answer_eos_correct": int(prediction["exact"].sum()),
                    "answer_eos_accuracy": _rate(prediction["exact"]),
                }
                groups[pool][arm] = {
                    str(count): {
                        key: value
                        for key, value in grids[task, arm]["metrics"][str(count)][pool].items()
                        if key.startswith("by_")
                    }
                    for count in (native_rounds, extra_rounds)
                }
        for method, baseline in (("only-u", "loop"), ("fixed-q-plus-u", "disco")):
            paired[f"{method}-minus-{baseline}"] = {
                pool: {
                    label: _transitions(
                        kept_predictions[task, baseline, pool, native_rounds][key],
                        kept_predictions[task, method, pool, native_rounds][key],
                    )
                    for label, key in (("answer", "correct"), ("answer_eos", "exact"))
                }
                for pool in TESTS
            }
        for arm in ARMS:
            transitions[arm] = {
                pool: {
                    label: _transitions(
                        kept_predictions[task, arm, pool, native_rounds][key],
                        kept_predictions[task, arm, pool, extra_rounds][key],
                    )
                    for label, key in (("answer", "correct"), ("answer_eos", "exact"))
                }
                for pool in TESTS
            }
        result["tasks"][task] = {
            "steps": settings["training"]["steps"],
            "training_repeats": native_rounds,
            "extra_repeats": extra_rounds,
            "primary": primary,
            "paired": paired,
            "training_majority_token": majority_token,
            "training_majority_baseline": majority,
            "extra_rounds": transitions,
            "groups": groups,
        }
    document = _markdown(result)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # All eight runs and every saved test grid were validated before any output.
    _write_new_or_identical(output / "comparison.json", encoded)
    _write_new_or_identical(output / "report.md", document)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.config, args.output)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "scientific_runs": result["scientific_runs"],
                "comparison": str(args.output / "comparison.json"),
                "report": str(args.output / "report.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
