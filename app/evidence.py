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


class EvidenceVerifier:
    """Attach only retrieved sources that semantically support each kept claim.

    This is an evidence-relevance gate, not a formal natural-language-inference
    proof. The API exposes the score and matched chunk so a reviewer can audit it.
    """

    def __init__(self, embedder: Embedder, min_score: float = 0.52, min_coverage: float = 0.50) -> None:
        if not -1 <= min_score <= 1 or not 0 < min_coverage <= 1:
            raise ValueError("证据阈值不合法")
        self.embedder = embedder
        self.min_score = min_score
        self.min_coverage = min_coverage

    def verify(self, answer: str, results: list[SearchResult]) -> EvidenceVerification:
        claims = split_claims(answer)
        if not claims or not results:
            return EvidenceVerification(REFUSAL, [], claims, 0.0)

        vectors = self.embedder.encode([*claims, *(item.text for item in results)])
        if len(vectors) != len(claims) + len(results):
            raise RuntimeError("证据校验向量数量不匹配")
        claim_vectors, source_vectors = vectors[:len(claims)], vectors[len(claims):]
        evidence: list[EvidenceItem] = []
        kept: list[str] = []
        unsupported: list[str] = []
        for claim_id, (claim, vector) in enumerate(zip(claims, claim_vectors), 1):
            scores = [_dot(vector, source) for source in source_vectors]
            best = max(scores)
            if best < self.min_score:
                unsupported.append(claim)
                continue
            # Keep ties close to the strongest candidate, limited to two chunks
            # so citations remain useful rather than becoming a source dump.
            source_ids = [
                index + 1 for index, score in enumerate(scores)
                if score >= self.min_score and score >= best - 0.025
            ][:2]
            marker = " ".join(f"[资料{source_id}]" for source_id in source_ids)
            kept.append(f"{claim} {marker}")
            evidence.append(EvidenceItem(claim_id, claim, source_ids, round(best, 4)))

        coverage = len(kept) / len(claims)
        if not kept or coverage < self.min_coverage:
            return EvidenceVerification(REFUSAL, evidence, unsupported, coverage)
        return EvidenceVerification("\n\n".join(kept), evidence, unsupported, coverage)
