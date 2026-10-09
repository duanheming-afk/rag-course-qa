"""Human answer review, kept separate from immutable automatic evaluation reports."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from threading import RLock
import tempfile


SCORE_FIELDS = ("correctness", "completeness", "evidence_support", "citation_support")
ERROR_LABELS = {
    "retrieval_miss": "检索遗漏",
    "answer_incomplete": "答案不完整",
    "unsupported_claim": "结论缺乏依据",
    "citation_mismatch": "引用不支持结论",
    "formula_error": "公式或条件错误",
    "wrong_refusal": "可回答题误拒答",
    "missed_refusal": "应拒答但未拒答",
    "missed_clarification": "应澄清但未澄清",
    "irrelevant_answer": "答非所问",
    "generation_error": "生成或服务异常",
    "annotation_issue": "题目、参考答案或关键词标注问题",
    "other": "其他问题",
}


def atomic_json(path: Path, value: dict) -> None:
    """Replace only the target review file after a complete, fsynced write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def keyword_groups(row: dict) -> tuple[list, list]:
    """Match the existing literal proxy; do not assign human correctness scores."""
    normalized = re.sub(r"\s+", "", row.get("answer", "")).lower()
    matched, missing = [], []
    for group in row.get("required_keywords", []):
        alternatives = group if isinstance(group, list) else [group]
        found = any(re.sub(r"\s+", "", term).lower() in normalized for term in alternatives)
        (matched if found else missing).append(group)
    return matched, missing


def risk_flags(row: dict) -> list[tuple[str, int]]:
    """Review-priority signals, not a list of confirmed model errors."""
    flags = []
    if not row.get("success", False) or row.get("error"):
        flags.append(("请求或生成异常", 100))
    behavior = row["expected_behavior"]
    if behavior == "answer":
        if row.get("refused") or row.get("clarified"):
            flags.append(("可回答题未正常回答", 90))
        if not row.get("retrieved_gold", False):
            flags.append(("目标文件未完全召回", 80))
        if row.get("invalid_citation_ids") or row.get("citation_missing"):
            flags.append(("引用编号无效或缺失", 75))
        if not row.get("citation_gold", False):
            flags.append(("目标文件未完全引用", 70))
        if row.get("evidence_coverage", 0) < 1 or row.get("removed_claim_count", 0):
            flags.append(("原始结论有删除或未全部通过语义筛选", 65))
        _, missing = keyword_groups(row)
        if missing:
            flags.append(("存在未命中的参考关键词（不等于错误）", 40))
        terms = [term for group in row.get("required_keywords", [])
                 for term in (group if isinstance(group, list) else [group])]
        if any(re.match(r"^\s*\d+(?:\.\d+)*[.、]?\s+\S", term) for term in terms):
            flags.append(("关键词疑似章节标题，需核对标注", 60))
    elif behavior == "refuse" and not row.get("refused"):
        flags.append(("未观察到预期拒答", 90))
    elif behavior == "clarify" and not row.get("clarified"):
        flags.append(("未观察到预期澄清", 90))
    return flags


def normalize_score(value, *, allow_na: bool = False):
    if value == "N/A" and allow_na:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, str)) or value not in (0, 1, 2, "0", "1", "2"):
        raise ValueError("评分必须选择 0、1、2；仅合理拒答/澄清的证据与引用可选 N/A。")
    return int(value)


class AnswerReviewStore:
    """One local review process, with atomic saves and stale-form detection."""

    def __init__(self, report_path: Path, output_dir: Path) -> None:
        self.report_path = report_path.resolve()
        self.output_dir = output_dir.resolve()
        if self.output_dir == self.report_path.parent:
            raise ValueError("评分目录必须与原始评测目录分开。")
        payload = self.report_path.read_bytes()
        self.source_hash = hashlib.sha256(payload).hexdigest()
        self.report = json.loads(payload.decode("utf-8-sig"))
        self.run_id = self.report.get("created_at", self.report_path.parent.name)
        self.records = {}
        for scheme, rows in self.report["details"].items():
            for row in rows:
                if row.get("expected_behavior") not in {"answer", "refuse", "clarify"}:
                    raise ValueError("评测样本缺少有效的 expected_behavior。")
                if not isinstance(row.get("id"), str) or not row["id"].strip():
                    raise ValueError("评测样本必须有字符串 ID。")
                key = f"{scheme}/{row['id']}"
                if key in self.records:
                    raise ValueError(f"重复评测记录：{key}")
                self.records[key] = {**row, "scheme": scheme}
        if not self.records:
            raise ValueError("评测报告没有逐题结果。")
        self.keys = sorted(self.records, key=lambda key: (
            -max((priority for _, priority in risk_flags(self.records[key])), default=0),
            self.records[key].get("keyword_recall", 0), key,
        ))
        self.path = self.output_dir / "reviews.json"
        self._lock = RLock()
        self._load()  # Fail early for mismatched/corrupt existing reviews; never overwrite them.

    def _load(self) -> dict:
        if hashlib.sha256(self.report_path.read_bytes()).hexdigest() != self.source_hash:
            raise ValueError("原始报告已变化，请重启并使用新的评分目录。")
        if not self.path.exists():
            return {"schema_version": 1, "run_id": self.run_id,
                    "source_sha256": self.source_hash, "rows": {}}
        data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if data.get("schema_version") != 1 or data.get("source_sha256") != self.source_hash:
            raise ValueError("评分文件版本或来源哈希不匹配；请勿合并不同评测运行。")
        if not isinstance(data.get("rows"), dict) or set(data["rows"]) - set(self.records):
            raise ValueError("评分文件包含未知记录或格式错误。")
        for key, review in data["rows"].items():
            if review.get("origin") != "human" or not review.get("reviewed_at"):
                raise ValueError(f"评分缺少人工来源和时间：{key}")
            normalized = self._validate(key, review)
            if any(review.get(field) != normalized[field] for field in SCORE_FIELDS):
                raise ValueError(f"评分值格式错误：{key}")
        return data

    def review(self, key: str) -> dict:
        if key not in self.records:
            raise ValueError("未知样本 ID。")
        with self._lock:
            return deepcopy(self._load()["rows"].get(key, {}))

    def _validate(self, key: str, review: dict) -> dict:
        record = self.records[key]
        correctness = normalize_score(review.get("correctness"))
        completeness = normalize_score(review.get("completeness"))
        allow_na = record["expected_behavior"] != "answer" and correctness == 2
        scores = {
            "correctness": correctness, "completeness": completeness,
            "evidence_support": normalize_score(review.get("evidence_support"), allow_na=allow_na),
            "citation_support": normalize_score(review.get("citation_support"), allow_na=allow_na),
        }
        if not isinstance(review.get("reviewer", ""), str) or not isinstance(review.get("notes", ""), str):
            raise ValueError("评分者和评分理由必须是文本。")
        reviewer = review.get("reviewer", "").strip()
        notes = review.get("notes", "").strip()
        tags = review.get("error_tags", [])
        if not reviewer:
            raise ValueError("请填写评分者，不能匿名生成正式人工评分。")
        if not isinstance(tags, list) or any(not isinstance(tag, str) or tag not in ERROR_LABELS for tag in tags):
            raise ValueError("问题分类无效。")
        if any(isinstance(value, int) and value < 2 for value in scores.values()) and (not tags or not notes):
            raise ValueError("有评分低于 2 时，请选择问题分类并填写理由。")
        return {**scores, "reviewer": reviewer, "notes": notes,
                "error_tags": sorted(set(tags))}

    def save(self, key: str, review: dict, *, expected_revision: int = 0) -> dict:
        if key not in self.records:
            raise ValueError("未知样本 ID。")
        validated = self._validate(key, review)
        with self._lock:
            data = self._load()
            previous = data["rows"].get(key, {})
            if previous.get("revision", 0) != expected_revision:
                raise ValueError("该题已被修改，请重新选择该题刷新后再保存。")
            saved = {**validated, "origin": "human",
                     "reviewed_at": datetime.now(timezone.utc).isoformat(),
                     "revision": expected_revision + 1}
            if previous:
                data.setdefault("history", []).append({"key": key, "previous_review": previous})
            data["rows"][key] = saved
            atomic_json(self.path, data)
            return deepcopy(saved)

    def next_pending(self, current: str) -> str:
        with self._lock:
            reviewed = self._load()["rows"]
        index = self.keys.index(current)
        order = self.keys[index + 1:] + self.keys[:index + 1]
        return next((key for key in order if key not in reviewed), current)

    def summary(self) -> dict:
        with self._lock:
            reviews = self._load()["rows"]
        schemes = {}
        for scheme in sorted({row["scheme"] for row in self.records.values()}):
            keys = [key for key, row in self.records.items() if row["scheme"] == scheme]
            scored = [reviews[key] for key in keys if key in reviews]
            def full_rate(field):
                applicable = [row[field] for row in scored if row[field] != "N/A"]
                return sum(value == 2 for value in applicable) / len(applicable) if applicable else None
            by_behavior = {}
            for behavior in ("answer", "refuse", "clarify"):
                selected = [key for key in keys if self.records[key]["expected_behavior"] == behavior]
                graded = [reviews[key] for key in selected if key in reviews]
                by_behavior[behavior] = {
                    "total": len(selected), "reviewed": len(graded),
                    "fully_correct_rate_on_reviewed": sum(row["correctness"] == 2 for row in graded) / len(graded) if graded else None,
                }
            schemes[scheme] = {
                "total": len(keys), "reviewed": len(scored),
                "coverage": len(scored) / len(keys), "complete": len(scored) == len(keys),
                "fully_correct_rate_on_reviewed": full_rate("correctness"),
                "fully_complete_rate_on_reviewed": full_rate("completeness"),
                "fully_supported_rate_on_applicable": full_rate("evidence_support"),
                "fully_citation_supported_rate_on_applicable": full_rate("citation_support"),
                "evidence_applicable": sum(row["evidence_support"] != "N/A" for row in scored),
                "citation_applicable": sum(row["citation_support"] != "N/A" for row in scored),
                "by_behavior": by_behavior,
            }
        return {
            "run_id": self.run_id, "source_sha256": self.source_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total": len(self.records), "reviewed": len(reviews),
            "reviewer_count": len({row["reviewer"] for row in reviews.values()}),
            "schemes": schemes,
            "human_issue_counts": dict(Counter(tag for row in reviews.values() for tag in row["error_tags"])),
            "automatic_risk_counts": dict(Counter(label for row in self.records.values() for label, _ in risk_flags(row))),
            "note": "人工评分与自动风险分开。未评分不计为零分；部分复核仅代表已评分子集，风险优先抽样有选择偏差。",
        }

    def export_reports(self) -> dict:
        summary = self.summary()
        atomic_json(self.output_dir / "review_summary.json", summary)
        lines = ["# RAG 模型答案人工复核汇总", "",
                 f"运行：{self.run_id}；来源 SHA256：`{self.source_hash}`", "",
                 f"已评分：{summary['reviewed']} / {summary['total']}；评分者人数：{summary['reviewer_count']}",
                 "", summary["note"], "",
                 "| 策略 | 已评分 / 总数 | 完全正确比例（已评分） | 完整回答比例（已评分） | 证据全支持比例（适用项） | 引用全支持比例（适用项） |",
                 "|---|---:|---:|---:|---:|---:|"]
        def percent(value):
            return "未评分 / 不适用" if value is None else f"{value:.2%}"
        for scheme, result in summary["schemes"].items():
            lines.append(f"| {scheme} | {result['reviewed']} / {result['total']} | "
                         f"{percent(result['fully_correct_rate_on_reviewed'])} | "
                         f"{percent(result['fully_complete_rate_on_reviewed'])} | "
                         f"{percent(result['fully_supported_rate_on_applicable'])}（{result['evidence_applicable']} 项） | "
                         f"{percent(result['fully_citation_supported_rate_on_applicable'])}（{result['citation_applicable']} 项） |")
            lines.extend(["", f"## {scheme} 题型覆盖", "",
                          "| 预期行为 | 已评分 / 总数 | 完全正确比例（已评分） |", "|---|---:|---:|"])
            for behavior, values in result["by_behavior"].items():
                lines.append(f"| {behavior} | {values['reviewed']} / {values['total']} | {percent(values['fully_correct_rate_on_reviewed'])} |")
        lines.extend(["", "## 人工记录的问题分类", "",
                      "允许一题多标签；标注问题不自动归因于模型。"])
        lines.extend(f"- {ERROR_LABELS[tag]}：{count}" for tag, count in summary["human_issue_counts"].items())
        if not summary["reviewed"]:
            lines.extend(["", "尚无人工评分，不能报告人工答案准确率。"])
        elif not all(result["complete"] for result in summary["schemes"].values()):
            lines.extend(["", "评分尚未覆盖全部样本，不得将已评分子集比例写成整个评测集的准确率。"])
        (self.output_dir / "review_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        diagnostic = ["# 自动风险清单（不是人工判错结果）", "",
                      "仅用于决定先审核哪些题；不改标签，不写人工评分，不等于真实失败样本数。", "",
                      "| 样本 | 预期行为 | 关键词代理分 | 自动提示 |", "|---|---|---:|---|"]
        for key in self.keys:
            row = self.records[key]
            flags = risk_flags(row)
            if flags:
                diagnostic.append(f"| {key} | {row['expected_behavior']} | {row.get('keyword_recall', 0):.2%} | {'；'.join(label for label, _ in flags)} |")
        (self.output_dir / "automatic_diagnostics.md").write_text("\n".join(diagnostic) + "\n", encoding="utf-8")
        return summary


def read_source_files(record: dict, raw_root: Path) -> str:
    """Display current local Markdown safely; recorded retrieval remains the historical source."""
    root = raw_root.resolve()
    sections = []
    for name in dict.fromkeys(record.get("gold_source_files", [])):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in {".md", ".markdown"}:
            sections.append(f"{name}：文件路径不合法，未读取。")
        elif path.is_file():
            sections.append(f"===== {name}（当前本地原文）=====\n{path.read_text(encoding='utf-8-sig')}")
        else:
            sections.append(f"{name}：当前本地找不到该原文，请参考报告中的检索片段。")
    return "\n\n".join(sections) or "无目标文件；请核对是否符合预期拒答或澄清行为。"
