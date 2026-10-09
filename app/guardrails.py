"""Deterministic handling for underspecified and out-of-scope requests."""
from __future__ import annotations

import re

from .citations import REFUSAL
from .retriever import SearchResult

_AMBIGUOUS_START = re.compile(
    r"^\s*(?:它|它们|这(?:个|段|条|些|里|部分)|那个|上述|前面|"
    r"第[一二三四五六七八九十]+个|为什么(?:结果|会变成这样|要这样))"
)
_SENSITIVE_PATTERNS = (
    ("wifi", "密码"), ("wi-fi", "密码"), ("私人", "手机"), ("私人", "号码"),
    ("手机号",), ("身份证",), ("银行卡",), ("住址",),
)


def clarification_message(question: str) -> str | None:
    """Ask for context instead of guessing what a deictic question means."""
    if _AMBIGUOUS_START.search(question.strip()):
        return "需要补充信息：请说明具体是哪门课程、哪个概念/公式/定理，或提供相关上下文。"
    return None


def immediate_refusal_reason(question: str) -> str | None:
    """Refuse sensitive or clearly non-course requests before model/index startup."""
    normalized = question.lower()
    if any(all(term in normalized for term in pattern) for pattern in _SENSITIVE_PATTERNS):
        return "课程问答不会提供密码、私人联系方式或其他敏感信息。"
    if "食堂" in normalized or "午餐菜单" in normalized:
        return "课程资料不包含食堂菜单等校园生活信息。"
    return None


def refusal_reason(question: str, results: list[SearchResult]) -> str | None:
    """Return a reason only when evidence is clearly unavailable or unsafe."""
    immediate = immediate_refusal_reason(question)
    if immediate:
        return immediate
    normalized = question.lower()

    # A schedule question may be valid only when the retrieved material actually
    # mentions the requested schedule details; do not infer dates/rooms from
    # ordinary course notes.
    asks_exam_schedule = ("考场" in normalized or "考试日期" in normalized or "期末考试" in normalized)
    evidence = "\n".join(f"{item.section}\n{item.text}" for item in results).lower()
    if asks_exam_schedule and not any(term in evidence for term in ("考场", "考试日期", "考试时间")):
        return "课程资料没有可核验的考试时间或考场信息。"
    return None


def refusal_message(reason: str) -> str:
    return f"{REFUSAL}\n{reason}"
