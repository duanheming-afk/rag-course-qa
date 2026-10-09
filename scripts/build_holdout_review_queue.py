"""Build a deterministic, review-only holdout queue from course chunks.

The produced rows are AI-drafted annotation candidates, not human labels. They
must be approved through the review workflow before entering a benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def topic(section: str) -> str:
    return re.sub(r"^(?:第?[一二三四五六七八九十0-9]+[章节、.]?|\d+(?:\.\d+)*)\s*", "", section).strip() or section


def focus_hint(chunk: dict) -> str:
    text = str(chunk["text"])
    bold = re.findall(r"\*\*([^*\n]{2,36})\*\*", text)
    if bold:
        return bold[0].strip()
    example = re.search(r"(?:例|示例)\s*\d+", text)
    if example:
        return example.group(0)
    body = re.sub(r"^章节：.*\n?", "", text).replace("$", "")
    body = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})?", "", body)
    body = re.sub(r"\s+", " ", body).strip(" -:：，。")
    return body[:24] or f"资料片段 {chunk['chunk_index']}"


def draft_question(chunk: dict) -> tuple[str, str]:
    heading = topic(str(chunk["section"]))
    lesson = str(chunk.get("title") or chunk["source_file"])
    hint = focus_hint(chunk)
    text = str(chunk["text"])
    if any(mark in text for mark in ("$$", "\\[", "\\frac", "\\lim", " = ")):
        return (
            f"根据{chunk['course']}课程资料，围绕“{lesson}”中“{heading}”的线索“{hint}”，写出关键公式或关系，并说明它描述的对象、条件或含义。",
            "formula",
        )
    if any(word in heading for word in ("定义", "概念", "原理")):
        return f"根据{chunk['course']}课程资料，围绕“{lesson}”中“{heading}”的线索“{hint}”，说明其定义或核心含义。", "definition"
    if any(word in heading for word in ("性质", "特点", "条件", "定理")):
        return f"围绕“{lesson}”中“{heading}”的线索“{hint}”，概述关键性质、成立条件或结论。", "properties"
    return f"根据{chunk['course']}课程资料，围绕“{lesson}”中“{heading}”的线索“{hint}”，概述核心内容并列出至少一个关键要点。", "summary"


def draft_keywords(chunk: dict) -> list[str]:
    text = str(chunk["text"])
    formulas = re.findall(r"\$+([^$\n]{2,80})\$+", text)
    values = [re.sub(r"\s+", " ", formula).strip() for formula in formulas[:3]]
    if not values:
        values = [topic(str(chunk["section"]))]
    return list(dict.fromkeys(value for value in values if value))[:4]


def answer_candidates(chunks: list[dict], answer_count: int, per_document: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        if chunk.get("source_file") == "INDEX.md" or len(text) < 90 or chunk.get("chunk_index", 0) == 0:
            continue
        grouped[str(chunk["source_file"])].append(chunk)

    selected: list[dict] = []
    for source_file in sorted(grouped):
        rows = grouped[source_file]
        rows.sort(key=lambda row: hashlib.sha256(str(row["chunk_id"]).encode("utf-8")).hexdigest())
        selected.extend(rows[:per_document])
    if len(selected) < answer_count:
        selected_ids = {row["chunk_id"] for row in selected}
        remaining = [row for rows in grouped.values() for row in rows if row["chunk_id"] not in selected_ids]
        remaining.sort(key=lambda row: hashlib.sha256(str(row["chunk_id"]).encode("utf-8")).hexdigest())
        selected.extend(remaining[:answer_count - len(selected)])
    return selected[:answer_count]


def negative_candidates() -> list[dict]:
    ambiguous = [
        "这个结论为什么成立？", "这里的第二步是什么意思？", "它和前面的有什么关系？",
        "这个公式怎么来的？", "为什么结果会变成这样？", "第一个方法和另一个有什么区别？",
        "这个例子说明了什么？", "这部分应该怎么理解？", "它的条件是什么？",
        "这个符号代表什么？", "这个定理能直接使用吗？", "这里为什么要这样计算？",
    ]
    unanswerable = [
        "课程资料中下一次补考的具体日期和教室是什么？", "课程实验室的门禁密码是多少？",
        "任课老师的私人邮箱地址是什么？", "学校图书馆今天几点闭馆？",
        "课程群的邀请码是什么？", "明天食堂的午餐价格是多少？",
        "课程资料对应考试的标准答案在哪里下载？", "助教的私人微信号是什么？",
        "本课程下学期的选课人数上限是多少？", "实验室的 WiFi 密码是什么？",
        "期末考试的座位号如何分配？", "老师办公室门锁密码是多少？",
    ]
    rows = []
    for question in ambiguous:
        rows.append({"question": question, "expected_behavior": "clarify", "category": "ambiguous"})
    for question in unanswerable:
        rows.append({"question": question, "expected_behavior": "refuse", "category": "out_of_scope"})
    return rows


def make_row(identifier: str, question: str, expected_behavior: str, category: str,
             chunk: dict | None = None) -> dict:
    evidence = []
    draft_answer = ""
    keywords: list[str] = []
    course = None
    if chunk is not None:
        evidence = [{
            "chunk_id": chunk["chunk_id"], "source_file": chunk["source_file"],
            "section": chunk["section"], "text": chunk["text"],
        }]
        draft_answer = str(chunk["text"])
        keywords = draft_keywords(chunk)
        course = chunk.get("course")
    return {
        "id": identifier,
        "question": question,
        "course": course,
        "expected_behavior": expected_behavior,
        "category": category,
        "candidate_evidence": evidence,
        "reference_answer_draft": draft_answer,
        "required_keywords_draft": keywords,
        "annotation": {
            "status": "pending_human_review",
            "reviewer": "",
            "reviewed_at": "",
            "gold_source_files": [],
            "reference_answer": "",
            "required_keywords": [],
            "notes": "AI-drafted candidate; not a benchmark label until approved.",
        },
    }


def build_queue(chunks: list[dict], answer_count: int = 120, per_document: int = 4) -> list[dict]:
    rows = []
    for number, chunk in enumerate(answer_candidates(chunks, answer_count, per_document), 1):
        question, category = draft_question(chunk)
        rows.append(make_row(f"h{number:03d}", question, "answer", category, chunk))
    for offset, seed in enumerate(negative_candidates(), len(rows) + 1):
        rows.append(make_row(f"h{offset:03d}", **seed))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="创建待人工标注的独立评测候选队列")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--output", default="data/eval/holdout_review_queue.jsonl")
    parser.add_argument("--answer-count", type=int, default=120)
    parser.add_argument("--per-document", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="覆盖已有的待审核队列")
    args = parser.parse_args()
    if args.answer_count < 100 or args.per_document < 1:
        parser.error("answer-count 至少为 100，per-document 至少为 1")
    output = PROJECT_ROOT / args.output
    if output.exists() and not args.force:
        parser.error(f"输出已存在：{output}。为避免覆盖人工标注，请确认后加 --force")
    queue = build_queue(load_jsonl(PROJECT_ROOT / args.chunks), args.answer_count, args.per_document)
    if len({row["question"] for row in queue}) != len(queue):
        raise ValueError("生成的候选题存在重复，请调整题目模板后重试")
    write_jsonl(output, queue)
    counts = defaultdict(int)
    for row in queue:
        counts[row["expected_behavior"]] += 1
    print(f"已创建 {len(queue)} 条待人工复核候选：" + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print(f"输出：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
