"""Validate human-reviewed rows and export a benchmark consumable by evaluate_answers.py."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def normalized_question(question: str) -> str:
    return re.sub(r"\s+", "", question).lower()


def approved_rows(queue: list[dict], existing_questions: set[str]) -> tuple[list[dict], list[str]]:
    output, errors, seen = [], [], set(existing_questions)
    for row in queue:
        annotation = row.get("annotation", {})
        if annotation.get("status") != "approved":
            continue
        identifier = row.get("id", "<missing>")
        question = str(row.get("question", "")).strip()
        normalized = normalized_question(question)
        if not question or normalized in seen:
            errors.append(f"{identifier}: 问题为空或与已有评测重复")
            continue
        expected = row.get("expected_behavior")
        if expected not in {"answer", "clarify", "refuse"}:
            errors.append(f"{identifier}: expected_behavior 无效")
            continue
        reference = str(annotation.get("reference_answer", "")).strip()
        if not reference:
            errors.append(f"{identifier}: 缺少人工参考答案")
            continue
        gold_files = annotation.get("gold_source_files", [])
        keywords = annotation.get("required_keywords", [])
        if expected == "answer" and (not gold_files or not keywords):
            errors.append(f"{identifier}: 可回答题必须标注目标文件和关键词")
            continue
        output.append({
            "id": identifier, "question": question, "course": row.get("course"),
            "expected_behavior": expected, "category": row.get("category", "holdout"),
            "gold_source_files": gold_files, "reference_answer": reference,
            "required_keywords": keywords,
            "annotation_metadata": {
                "reviewer": annotation.get("reviewer", ""),
                "reviewed_at": annotation.get("reviewed_at", ""),
                "notes": annotation.get("notes", ""),
            },
        })
        seen.add(normalized)
    return output, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="导出人工审核通过的独立评测集")
    parser.add_argument("--queue", default="data/eval/holdout_review_queue.jsonl")
    parser.add_argument("--output", default="data/eval/holdout_approved.jsonl")
    parser.add_argument("--min-approved", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="仅检查，不写出文件")
    args = parser.parse_args()
    queue = load_jsonl(PROJECT_ROOT / args.queue)
    existing = set()
    for source in ("data/eval/answer_pairs.jsonl", "data/eval/challenge_pairs.jsonl"):
        existing.update(normalized_question(row["question"]) for row in load_jsonl(PROJECT_ROOT / source))
    exported, errors = approved_rows(queue, existing)
    pending = sum(row.get("annotation", {}).get("status") == "pending_human_review" for row in queue)
    print(f"审核通过：{len(exported)}；待审核：{pending}；校验错误：{len(errors)}")
    for error in errors[:20]:
        print("- " + error)
    if errors or len(exported) < args.min_approved:
        print(f"未导出：至少需要 {args.min_approved} 条无错误的人工审核样本。", file=sys.stderr)
        return 1
    if not args.check:
        output = PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in exported) + "\n", encoding="utf-8")
        print(f"已导出：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
