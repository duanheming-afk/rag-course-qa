"""Dense retrieval followed by evidence-based generation."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from .config import Settings
from .citations import REFUSAL, resolve_citations
from .evidence import EvidenceItem, EvidenceVerifier, evidence_body
from .generator import OllamaGenerator
from .guardrails import clarification_message, immediate_refusal_reason, refusal_message, refusal_reason
from .retriever import SearchResult, VectorRetriever


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[dict]
    retrieved: list[SearchResult]
    model: str
    warnings: list[str] = field(default_factory=list)
    refused: bool = False
    citation_mode: str = "model"
    evidence: list[EvidenceItem] = field(default_factory=list)
    evidence_coverage: float = 0.0
    removed_claim_count: int = 0


class RAGPipeline:
    def __init__(self, index_path: str | Path | None = None, collection: str | None = None,
                 embedding_model: str | None = None, llm_host: str | None = None,
                 llm_model: str | None = None, llm_max_tokens: int | None = None,
                 llm_timeout: float | None = None) -> None:
        settings = Settings.from_env()
        # Validate the generator settings before acquiring the index lock.
        self.generator = OllamaGenerator(llm_host, llm_model, max_tokens=llm_max_tokens, timeout=llm_timeout)
        self.retriever = VectorRetriever(
            index_path or settings.index_path, collection or settings.collection,
            embedding_model or settings.embedding_model,
        )
        self.evidence_verifier = EvidenceVerifier(
            self.retriever.embedder, settings.evidence_min_score, settings.evidence_min_coverage,
        )

    def ask(self, question: str, top_k: int = 5, course: str | None = None) -> RAGResponse:
        question = question.strip()
        if not question or len(question) > 2000 or not 1 <= top_k <= 8:
            raise ValueError("问题长度必须为 1—2000 字，top_k 必须在 1—8 之间")
        immediate_refusal = immediate_refusal_reason(question)
        if immediate_refusal:
            return RAGResponse(
                question, refusal_message(immediate_refusal), [], [], self.generator.model,
                [immediate_refusal], True, "not_required",
            )
        clarification = clarification_message(question)
        if clarification:
            return RAGResponse(question, clarification, [], [], self.generator.model, citation_mode="not_required")
        retrieved = self.retriever.search(question, top_k=top_k, course=course)
        refusal = refusal_reason(question, retrieved)
        if refusal:
            return RAGResponse(
                question, refusal_message(refusal), [], [], self.generator.model,
                [refusal], True, "not_required",
            )
        extractive = self._extractive_formula_quote(question, retrieved)
        if extractive:
            quote, source_id = extractive
            sources = resolve_citations(quote, retrieved).sources
            return RAGResponse(
                question, quote, sources, retrieved, self.generator.model,
                ["题目含明确线索且要求公式/关系；为避免改写公式条件，已直接摘录命中资料正文。"],
                False, "extractive_quote",
                [EvidenceItem(1, "资料原文摘录", [source_id], 1.0)], 1.0, 0,
            )
        answer = self.generator.generate(question, retrieved)
        if answer.refused:
            extractive = self._extractive_formula_fallback(question, retrieved)
            if extractive:
                verification = self.evidence_verifier.verify(extractive, retrieved)
                if verification.answer != REFUSAL:
                    sources = resolve_citations(verification.answer, retrieved).sources
                    warnings = [
                        "模型未生成答案，已从命中资料中摘录公式/复杂度并经过证据校验。",
                    ]
                    return RAGResponse(
                        question, verification.answer, sources, retrieved, answer.model,
                        warnings, False, "extractive_fallback", verification.evidence,
                        verification.coverage, len(verification.unsupported_claims),
                    )
            return RAGResponse(
                question, answer.answer, [], retrieved, answer.model,
                answer.warnings, True, "not_required",
            )
        verification = self.evidence_verifier.verify(answer.answer, retrieved)
        warnings = [warning for warning in answer.warnings if "模型未在正文标注引用" not in warning]
        if verification.unsupported_claims:
            warnings.append(f"已移除 {len(verification.unsupported_claims)} 条缺少足够证据的表述。")
        if verification.answer == REFUSAL:
            warnings.append("生成内容的可核验证据不足，已拒绝输出该回答。")
            return RAGResponse(
                question, REFUSAL, [], retrieved, answer.model, warnings, True, "not_required",
                verification.evidence, verification.coverage, len(verification.unsupported_claims),
            )
        sources = resolve_citations(verification.answer, retrieved).sources
        return RAGResponse(
            question, verification.answer, sources, retrieved, answer.model,
            warnings, False, "sentence_verified",
            verification.evidence, verification.coverage, len(verification.unsupported_claims),
        )

    @staticmethod
    def _extractive_formula_quote(question: str, results: list[SearchResult]) -> tuple[str, int] | None:
        """Return an exact body-text quote for a narrowly scoped formula query.

        The question must contain an explicit quoted line hint and ask for a
        formula, relationship or complexity.  The returned content is copied
        from one matched chunk's body only: it does not combine neighboring
        properties or paraphrase a condition as an equivalence.
        """
        if not re.search(r"关键公式|公式或关系|复杂度", question):
            return None
        if not VectorRetriever._anchor_terms(question):
            return None
        for index, item in enumerate(results, 1):
            if not VectorRetriever._anchor_matches(question, item):
                continue
            body = evidence_body(item.text)
            if "$" in body or "=" in body or "复杂度" in body:
                return f"以下为资料中与题目线索匹配的原文摘录：\n{body}\n[资料{index}]", index
        return None

    @staticmethod
    def _extractive_formula_fallback(question: str, results: list[SearchResult]) -> str | None:
        """Compatibility fallback for a model refusal on the same narrow query."""
        quote = RAGPipeline._extractive_formula_quote(question, results)
        return quote[0] if quote else None

    def close(self) -> None:
        self.retriever.close()
