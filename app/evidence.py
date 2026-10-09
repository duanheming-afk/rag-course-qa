"""Sentence-level evidence matching and deterministic citation enforcement."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .citations import CITATION_RE, REFUSAL
from .retriever import SearchResult


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class EvidenceItem:
    claim_id: int
    claim: str
    source_ids: list[int]
    support_score: float


@dataclass(frozen=True)
class EvidenceVerification:
    answer: str
    evidence: list[EvidenceItem]
    unsupported_claims: list[str]
    coverage: float


def _strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


def split_claims(answer: str) -> list[str]:
    """Split prose/bullets into citeable claims without breaking LaTex blocks."""
    cleaned = _strip_citations(answer)
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s+.*$", "", cleaned, flags=re.MULTILINE)
    parts = re.split(r"(?<=[。！？；])\s*|\n(?=\s*(?:[-*+]\s+|\d+[.)、]\s+))", cleaned)
    claims = []
    for part in parts:
        claim = part.strip().lstrip("-*+ ").strip()
        if claim and claim not in claims:
            claims.append(claim)
    return claims


def _dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


def evidence_body(text: str) -> str:
    """Exclude headings and corpus notices, but preserve actual quoted evidence."""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or re.match(r"^(?:章节\s*[:：]|#{1,6}\s+)", line):
            continue
        if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", line):
            continue
        notice = line.lstrip("> ")
        if notice.startswith("合成讲义") and "仅供" in notice and "测试" in notice:
            continue
        lines.append(line)
    return "\n".join(lines)


class EvidenceVerifier:
    """Attach the strongest body-text match for each kept claim.

    This is an evidence-relevance gate, not a formal natural-language-inference
    proof. Matching headings alone cannot serve as evidence. Similarity ties do
    not justify additional citations; every claim uses one strongest candidate.
    Multi-source reasoning still needs finer claim decomposition/human review.
    """

    def __init__(self, embedder: Embedder, min_score: float = 0.52, min_coverage: float = 0.50) -> None:
        if not -1 <= min_score <= 1 or not 0 < min_coverage <= 1:
            raise ValueError("证据阈值不合法")
        self.embedder = embedder
        self.min_score = min_score
        self.min_coverage = min_coverage

    def verify(self, answer: str, results: list[SearchResult]) -> EvidenceVerification:
        claims = split_claims(answer)
        candidates = [(index + 1, body) for index, item in enumerate(results)
                      if (body := evidence_body(item.text))]
        if not claims or not candidates:
            return EvidenceVerification(REFUSAL, [], claims, 0.0)

        # Retain the original full-chunk threshold as a regression gate. Merely
        # changing embedding input must not resurrect formerly rejected claims.
        # Deduplicate identical full/body/claim strings within this verification.
        texts = list(dict.fromkeys([*claims, *(item.text for item in results),
                                   *(body for _, body in candidates)]))
        vectors = self.embedder.encode(texts)
        if len(vectors) != len(texts):
            raise RuntimeError("证据校验向量数量不匹配")
        by_text = dict(zip(texts, vectors))
        claim_vectors = [by_text[claim] for claim in claims]
        original_vectors = [by_text[item.text] for item in results]
        source_vectors = [by_text[body] for _, body in candidates]
        evidence: list[EvidenceItem] = []
        kept: list[str] = []
        unsupported: list[str] = []
        for claim_id, (claim, vector) in enumerate(zip(claims, claim_vectors), 1):
            scores = [_dot(vector, source) for source in source_vectors]
            # Stable ties retain retrieval order; the strongest match must never
            # be displaced by two earlier, merely close-scoring candidates.
            best_index = max(range(len(scores)), key=scores.__getitem__)
            best = scores[best_index]
            original_best = max(_dot(vector, source) for source in original_vectors)
            if best < self.min_score or original_best < self.min_score:
                unsupported.append(claim)
                continue
            # IDs still refer to the ORIGINAL retrieval list, not the filtered
            # body-text list, so API source resolution remains correct.
            source_ids = [candidates[best_index][0]]
            marker = " ".join(f"[资料{source_id}]" for source_id in source_ids)
            kept.append(f"{claim} {marker}")
            evidence.append(EvidenceItem(claim_id, claim, source_ids, round(best, 4)))

        coverage = len(kept) / len(claims)
        if not kept or coverage < self.min_coverage:
            return EvidenceVerification(REFUSAL, evidence, unsupported, coverage)
        return EvidenceVerification("\n\n".join(kept), evidence, unsupported, coverage)
