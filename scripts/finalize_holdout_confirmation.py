"""Finalize rows after the user confirms the review was completed."""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def finalize(rows: list[dict]) -> list[dict]:
    timestamp = datetime.now(timezone.utc).isoformat()
    result = []
    for original in rows:
        row = copy.deepcopy(original)
        annotation = row.get("annotation", {})
        if annotation.get("status") == "pending_human_review":
            answerable = row.get("expected_behavior") == "answer"
            complete = bool(annotation.get("reference_answer", "").strip())
            if answerable:
                complete = complete and bool(annotation.get("gold_source_files")) and bool(annotation.get("required_keywords"))
            if complete:
                annotation["status"] = "approved"
                annotation["reviewer"] = annotation.get("reviewer") or "user-confirmed"
                annotation["reviewed_at"] = annotation.get("reviewed_at") or timestamp
                annotation["notes"] = (annotation.get("notes", "") + " User confirmed review completion.").strip()
        row["annotation"] = annotation
        result.append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="按用户确认结果收口评测集审核状态")
    parser.add_argument("--input", default="reports/holdout_review/ai_review_draft.jsonl")
    parser.add_argument("--output", default="reports/holdout_review/confirmed_review.jsonl")
    args = parser.parse_args()
    rows = finalize(load_jsonl(PROJECT_ROOT / args.input))
    write_jsonl(PROJECT_ROOT / args.output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        status = row["annotation"].get("status", "")
        counts[status] = counts.get(status, 0) + 1
    print("; ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print(f"输出：{PROJECT_ROOT / args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
