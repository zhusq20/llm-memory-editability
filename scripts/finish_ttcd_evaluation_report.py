#!/usr/bin/env python3
"""Publish the requested result tables after the already-queued audited aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/mnt/disk2_from_server2/siqizhu4/llm-memory-editability")
RECOVERY = Path("/ossfs/workspace/llm-memory-editability/evaluation-recovery-20261016")


def replace_paragraph(path, prefix, replacement):
    previous = path.read_text()
    paragraphs = previous.split("\n\n")
    matched = [i for i, paragraph in enumerate(paragraphs) if paragraph.startswith(prefix)]
    if len(matched) != 1:
        return False
    paragraphs[matched[0]] = replacement
    if path.read_text() != previous:
        raise RuntimeError(
            "Documentation changed concurrently; preserve it for manual reconciliation"
        )
    temporary = path.with_name(path.name + ".ttcd-report.tmp")
    temporary.write_text("\n\n".join(paragraphs))
    os.replace(temporary, path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-script-sha256", required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 48 * 3600
    while time.monotonic() < deadline:
        path = RECOVERY / "summary-state.json"
        if path.is_file():
            state = json.loads(path.read_text())
            if state["state"] == "complete":
                if state.get("protected_evidence_unchanged") is not True:
                    raise ValueError("Original training and main report must remain unchanged")
                break
        recovery = json.loads((RECOVERY / "controller-state.json").read_text())
        if recovery["state"] == "needs_attention":
            raise RuntimeError("Finish diagnosed recovery failures before publishing conclusions")
        time.sleep(30)
    else:
        raise TimeoutError("Audited aggregate has not completed; no result report published")
    script = REPO / "scripts/summarize_ttcd_writing_results.py"
    if hashlib.sha256(script.read_bytes()).hexdigest() != args.summary_script_sha256:
        raise ValueError("The reviewed summary script changed while waiting")
    subprocess.run([sys.executable, str(script)], check=True)
    link = "[完整18条件结果](../results/ttcd-evaluation-summary-20261016/report.md)"
    completion = (
        "**2026-10-16恢复完成：三架构一步写入与merge评测、独立重放及云端完整JSON核验全部通过。** "
        "恢复新增训练0；全部原检查点、源码及原一步训练主报告保留并通过哈希核对。"
        "独立control-report及三种写法×三架构×训练内1/4步的18条件表已生成。"
        f"{link}列出写入成功、两跳更新、条件子集覆盖与旧知识保持；单图单初始化，旋转为重复测量。"
        "共享GPU耗时不作独占效率比较。"
        "[恢复与调度记录](development-artifacts/ttcd-one-step-second-hop-v1/evaluation-recovery-20261016.md)。"
        "此前运行中段落为历史，以本段及完成收据为准。"
    )
    results_updated = replace_paragraph(
        REPO / "docs/results.md", "**2026-10-16恢复状态：", completion
    )
    roadmap_updated = replace_paragraph(
        REPO / "docs/roadmap.md",
        "- **2026-10-16：按用户要求恢复剩余一步写入与merge评测，",
        "- " + completion,
    )
    report = REPO / "results/ttcd-evaluation-summary-20261016/report.md"
    receipt = {
        "state": "complete",
        "new_training_updates": 0,
        "report": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "results_document_updated": results_updated,
        "roadmap_updated": roadmap_updated,
    }
    (RECOVERY / "report-publication-state.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
