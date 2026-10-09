"""Resolve declared references, without claiming semantic faithfulness."""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from .retriever import SearchResult

# Accept both the documented ``[资料1]`` form and the plain ``资料1`` form.
# Matching the brackets here prevents evidence verification from leaving
# confusing empty placeholders such as ``[]`` in the final answer.
CITATION_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\[\s*)?资料\s*([0-9]+)(?:\s*\])?(?![0-9])"
)
REFUSAL = "资料中没有提到。"


def is_refusal(answer: str) -> bool:
    text = answer.strip().lstrip("#* \n")
    if text.startswith(
        ("资料中没有提到", "资料中未提到", "参考资料中没有提到")
    ):
        return True
    prefix = re.sub(r"\s+", "", text[:400])
    return (
        "参考资料" in prefix
        and any(signal in prefix for signal in ("没有相关信息", "未明确提及", "无法确定", "无法回答", "缺乏"))
    )


@dataclass
class CitationResolution:
    sources: list[dict]
    invalid_ids: list[int]
    missing: bool


def resolve_citations(answer: str, results: list[SearchResult]) -> CitationResolution:
    ids = sorted({int(m.group(1)) for m in CITATION_RE.finditer(unicodedata.normalize("NFKC", answer))})
    invalid = [i for i in ids if not 1 <= i <= len(results)]
    sources = []
    for index in ids:
        if 1 <= index <= len(results):
            item = results[index - 1]
            sources.append({
                "citation_id": index, "chunk_id": item.chunk_id, "source_file": item.source_file,
                "section": item.section, "page": item.page, "score": round(item.score, 4),
                "snippet": item.text,
            })
    return CitationResolution(sources, invalid, not sources and not is_refusal(answer))


def cited_source_files(answer: str, results: list[SearchResult]) -> set[str]:
    files = {source["source_file"] for source in resolve_citations(answer, results).sources}
    # File-level compatibility for historical reports; not used as UI evidence.
    files.update(item.source_file for item in results if item.source_file and item.source_file in answer)
    return files
