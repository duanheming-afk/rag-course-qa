"""Prepare source-grounded annotation drafts without claiming human approval."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def clean_reference(text: str) -> str:
    """Keep the source wording, but remove only the technical chunk prefix."""
    return re.sub(r"^章节：[^\n]*\n?", "", text.strip(), count=1)


def ends_with_unanswered_prompt(text: str) -> bool:
    """Flag chunks where the selected fragment ends at an exercise prompt."""
    tail = re.sub(r"\s+", "", text[-120:])
    return bool(re.search(r"(?:求|解释|说明|为什么|如何|写出)[^。！？]*[。！？]?$", tail))


def build_draft(rows: list[dict]) -> list[dict]:
    reviewed_at = datetime.now(timezone.utc).isoformat()
    output = []
    for original in rows:
        row = copy.deepcopy(original)
        annotation = row.setdefault("annotation", {})
        annotation.update({
            "status": "pending_human_review",
            "reviewer": "",
            "reviewed_at": "",
            "gold_source_files": [],
            "reference_answer": "",
            "required_keywords": [],
        })
        notes = ["AI-assisted source review draft; not human approval."]
        if row["expected_behavior"] == "answer":
            evidence = row.get("candidate_evidence", [])
            if not evidence:
                notes.append("缺少候选证据，需重写或拒绝。")
                annotation["status"] = "needs_rewrite"
            else:
                source = evidence[0]
                source_text = str(source.get("text", "")).strip()
                annotation["gold_source_files"] = [str(source["source_file"])]
                annotation["reference_answer"] = clean_reference(source_text)
                annotation["required_keywords"] = list(row.get("required_keywords_draft", []))
                if ends_with_unanswered_prompt(source_text):
                    annotation["status"] = "needs_rewrite"
                    notes.append("候选片段以练习题目结尾，可能没有包含该题的完整答案。")
                elif not annotation["required_keywords"]:
                    annotation["status"] = "needs_rewrite"
                    notes.append("缺少可验证关键词，需人工补充。")
        elif row["expected_behavior"] == "clarify":
            annotation["reference_answer"] = "问题缺少课程、章节或上下文，请补充完整问题后再回答。"
            notes.append("需确认系统是否稳定返回澄清提示。")
        elif row["expected_behavior"] == "refuse":
            annotation["reference_answer"] = "课程资料中没有该信息，无法依据资料回答。"
            notes.append("需确认系统拒答且不泄露或猜测信息。")
        annotation["notes"] = " ".join(notes)
        row["ai_review"] = {
            "reviewed_at": reviewed_at,
            "reviewer": "Codex AI source review",
            "warning": "此字段仅记录机器辅助审阅，不等同于人工标注。",
        }
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="生成待人工确认的独立评测集标注草稿")
    parser.add_argument("--queue", default="data/eval/holdout_review_queue.jsonl")
    parser.add_argument("--output", default="reports/holdout_review/ai_review_draft.jsonl")
    args = parser.parse_args()
    source = PROJECT_ROOT / args.queue
    destination = PROJECT_ROOT / args.output
    rows = build_draft(load_jsonl(source))
    write_jsonl(destination, rows)
    counts: dict[str, int] = {}
    for row in rows:
        status = row["annotation"]["status"]
        counts[status] = counts.get(status, 0) + 1
    print(f"已生成 {len(rows)} 条标注草稿：" + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print(f"输出：{destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
