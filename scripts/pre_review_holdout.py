"""Create a stratified AI pre-review pilot without approving human labels."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_pilot(rows: list[dict], count: int) -> list[dict]:
    """Round-robin answerable pending rows across courses and categories."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["expected_behavior"] != "answer" or row["annotation"]["status"] != "pending_human_review":
            continue
        grouped[str(row["course"])].append(row)
    for course_rows in grouped.values():
        course_rows.sort(key=lambda row: (row["category"], row["id"]))
    selected: list[dict] = []
    while len(selected) < count and any(grouped.values()):
        for course in sorted(grouped):
            if grouped[course] and len(selected) < count:
                selected.append(grouped[course].pop(0))
    return selected


def preflight(row: dict, reviewed_at: str) -> dict:
    evidence = row.get("candidate_evidence", [])
    notes: list[str] = []
    if len(evidence) != 1:
        notes.append("候选证据数量不是 1；人工需确认是否需要多文件标注。")
    snippet = evidence[0]["text"].strip() if evidence else ""
    if len(snippet) < 120:
        notes.append("候选证据片段较短；人工需打开原始资料核对。")
    if snippet.startswith((",", "，", ")", "）")):
        notes.append("片段可能从段落中间开始；人工需核对前后文。")
    if row["category"] == "formula" and not any(token in snippet for token in ("$", "=", "\\")):
        notes.append("题目要求公式，但候选片段未发现明显公式标记。")
    if not row.get("required_keywords_draft"):
        notes.append("缺少关键词草稿；人工必须补充。")
    return {
        "reviewed_by": "Codex AI preflight (not human approval)",
        "reviewed_at": reviewed_at,
        "verdict": "needs_human_context_check" if notes else "ready_for_human_review",
        "checks": {
            "single_candidate_evidence": len(evidence) == 1,
            "snippet_length": len(snippet),
            "has_keyword_draft": bool(row.get("required_keywords_draft")),
            "question_is_answerable_candidate": row["expected_behavior"] == "answer",
        },
        "notes": notes or ["结构检查通过；仍需人工核对答案、来源和关键词。"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="对首批独立评测候选进行 AI 预审，不会批准人工标签")
    parser.add_argument("--queue", default="data/eval/holdout_review_queue.jsonl")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--report", default="reports/holdout_pre_review/pilot_20.md")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count 必须大于 0")
    queue_path = PROJECT_ROOT / args.queue
    rows = load_jsonl(queue_path)
    pilot = select_pilot(rows, args.count)
    if len(pilot) < args.count:
        raise RuntimeError(f"待审核的可回答题只有 {len(pilot)} 条，无法组成 {args.count} 条试点")
    reviewed_at = datetime.now(timezone.utc).isoformat()
    by_id = {row["id"]: row for row in rows}
    for item in pilot:
        by_id[item["id"]]["pre_review"] = preflight(item, reviewed_at)
    save_jsonl(queue_path, rows)

    report_path = PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 首批 20 条独立评测候选：AI 预审清单", "",
        "此文件是结构与证据完整性预审，不是人工审核，也不改变 `annotation.status`。",
        "人工复核时，请以原始课程资料为准，填写人工参考答案、目标文件、关键词和审核人。", "",
        "| ID | 课程 | 类型 | 预审结论 | 人工需要确认 |", "|---|---|---|---|---|",
    ]
    for item in pilot:
        review = by_id[item["id"]]["pre_review"]
        note = "；".join(review["notes"])
        lines.append(
            f"| {item['id']} | {item['course']} | {item['category']} | {review['verdict']} | {note} |"
        )
        lines.extend([
            "", f"**问题：** {item['question']}",
            f"**候选来源：** {item['candidate_evidence'][0]['source_file']} / {item['candidate_evidence'][0]['section']}",
            f"**关键词草稿：** {', '.join(item['required_keywords_draft']) or '无'}", "",
        ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ready = sum(by_id[item["id"]]["pre_review"]["verdict"] == "ready_for_human_review" for item in pilot)
    print(f"已预审 {len(pilot)} 条：{ready} 条结构就绪，{len(pilot) - ready} 条需要人工额外核对。")
    print(f"报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
