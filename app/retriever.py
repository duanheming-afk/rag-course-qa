"""Single-owner local Qdrant retrieval with exception-safe resource handling."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from .embedding import TextEmbedder
from .errors import RAGError


@dataclass
class SearchResult:
    score: float
    text: str
    source_file: str
    section: str
    course: str
    page: int
    chunk_id: str


class VectorRetriever:
    _INTENT_TERMS = {"定义", "公式", "条件", "步骤", "区别", "原因", "特点", "应用", "例题", "计算"}
    _GENERIC_ANCHORS = {"内容", "方法", "应用", "定义", "特点", "概念", "典型例题"}
    def __init__(self, index_path: str | Path = "index/qdrant",
                 collection: str = "course_kb", model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        from qdrant_client import QdrantClient
        path = Path(index_path)
        if not path.is_dir():
            raise RAGError("index_missing", "索引不存在，请先运行数据处理和索引构建脚本")
        self.collection = collection
        self._lock = RLock()
        self._closed = False
        try:
            self.client = QdrantClient(path=str(path))
        except RuntimeError as exc:
            raise RAGError("index_busy", "本地索引已被占用；请关闭另一个 API 或检索/评测进程后重试") from exc
        try:
            info = self.client.get_collection(collection)
            if not info.points_count:
                raise RAGError("index_empty", "索引为空，请先构建课程索引")
            self.embedder = TextEmbedder(model_name=model_name, device=os.getenv("EMBEDDING_DEVICE") or None)
            if info.config.params.vectors.size != self.embedder.dimension:
                raise RAGError("index_dimension_mismatch", "当前向量模型维度与索引不一致，请使用构建索引时的模型")
            # Keep the small local corpus available for explicit line-hint
            # recovery. This is deliberately separate from dense retrieval:
            # ordinary questions never scan or inject lexical matches.
            self._anchor_chunks = self._load_anchor_chunks()
        except Exception:
            self.client.close()
            self._closed = True
            raise

    def _load_anchor_chunks(self) -> list[SearchResult]:
        points = []
        offset = None
        while True:
            page, offset = self.client.scroll(
                collection_name=self.collection, limit=1000, offset=offset,
                with_payload=True,
            )
            points.extend(page)
            if offset is None:
                break
        chunks = []
        for point in points:
            payload = point.payload or {}
            chunks.append(SearchResult(
                0.0, str(payload.get("text", "")), str(payload.get("source_file", "")),
                str(payload.get("section", "")), str(payload.get("course", "")),
                int(payload.get("page", 1)), str(payload.get("chunk_id", point.id)),
            ))
        return chunks

    def index_status(self) -> dict:
        with self._lock:
            if self._closed:
                raise RAGError("index_closed", "索引已关闭")
            count = int(self.client.get_collection(self.collection).points_count or 0)
            return {"status": "ok" if count > 0 else "degraded", "index_ready": count > 0, "points": count}

    @staticmethod
    def _query_terms(question: str) -> list[str]:
        """Extract a few meaningful lexical anchors without an extra tokenizer dependency."""
        cleaned = re.sub(r"(什么是|是什么|请问|如何|怎样|为什么|哪些|吗|呢|一下|的)", " ", question.lower())
        pieces = re.findall(r"[a-z0-9_+.#-]{2,}|[\u4e00-\u9fff]{2,}", cleaned)
        terms: list[str] = []
        for piece in pieces:
            variants = [piece]
            for part in re.split(r"[与和及或、]", piece):
                part = re.sub(r"^(?:比较|说明|写出|给出|计算|求|解释|介绍)", "", part)
                if part:
                    variants.append(part)
            for term in variants:
                if term not in terms:
                    terms.append(term)
            # Long Chinese phrases benefit from their short technical sub-terms,
            # e.g. “导数的定义” becomes “导数” and “定义”.
            if re.fullmatch(r"[\u4e00-\u9fff]{4,}", piece):
                for width in (2, 3, 4):
                    for start in range(len(piece) - width + 1):
                        term = piece[start:start + width]
                        if term not in {"什么", "请问", "如何", "怎样", "为什么"} and term not in terms:
                            terms.append(term)
        return terms[:12]

    @classmethod
    def _anchor_terms(cls, question: str) -> list[str]:
        """Extract only explicit quoted line hints, not arbitrary query words."""
        quoted = re.findall(r"[\u201c\"]([^\u201d\"]+)[\u201d\"]", question)
        terms: list[str] = []
        for term in quoted:
            term = re.sub(r"^\s*线索\s*[:：]?\s*", "", term).strip()
            normalized = re.sub(r"\s+", "", term).lower()
            if len(normalized) < 2 or normalized in cls._GENERIC_ANCHORS:
                continue
            if normalized not in terms:
                terms.append(normalized)
        return terms[:6]

    @classmethod
    def _anchor_matches(cls, question: str, item: SearchResult) -> list[str]:
        terms = cls._anchor_terms(question)
        if not terms:
            return []
        haystack = re.sub(
            r"\s+", "", f"{item.source_file}\n{item.section}\n{item.text}"
        ).lower()
        return [term for term in terms if term in haystack]

    def _anchor_candidates(self, question: str, course: str | None) -> list[SearchResult]:
        if not self._anchor_terms(question):
            return []
        candidates = []
        for item in self._anchor_chunks:
            if course and item.course != course:
                continue
            matched = self._anchor_matches(question, item)
            if matched:
                candidates.append((len(matched), max(map(len, matched)), len(item.text), item))
        candidates.sort(key=lambda value: value[:3], reverse=True)
        return [item for _, _, _, item in candidates[:24]]

    @classmethod
    def _lexical_rerank(cls, question: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Prefer chunks whose headings/content explicitly name the asked concept.

        Vector search supplies semantic recall; this is only a bounded tie-breaker
        over its candidates, so unrelated full-text matches cannot enter on their
        own. The public score remains the original vector score.
        """
        terms = cls._query_terms(question)
        anchors = cls._anchor_terms(question)
        if not terms and not anchors:
            return results[:top_k]

        def rank_key(item: SearchResult) -> tuple[float, float]:
            section = item.section.lower()
            source = item.source_file.lower()
            text = item.text.lower()
            bonus = 0.0
            for term in terms:
                # One-character terms are accepted only in headings. This covers
                # compact technical nouns such as “栈” without letting ordinary
                # Chinese characters dominate a full-text ranking.
                if len(term) == 1:
                    if term in section:
                        bonus += 0.12
                    continue
                if term in section:
                    # Question forms such as “导数的定义” identify the answer
                    # type explicitly; a matching heading should outrank a
                    # semantically close but different subsection.
                    bonus += 0.28 if term in cls._INTENT_TERMS else 0.12
                elif term in source:
                    bonus += 0.08
                elif term in text:
                    bonus += 0.035
            # An explicit quoted line hint is much stronger than a generic
            # semantic similarity score, but only applies when the user gave
            # that hint. This keeps normal retrieval behavior unchanged.
            matched_anchors = [
                anchor for anchor in anchors
                if anchor in re.sub(r"\s+", "", source + section + text)
            ]
            anchor_bonus = 0.0
            if matched_anchors:
                anchor_bonus = 0.85 + min(0.05 * (len(matched_anchors) - 1), 0.1)
            return (item.score + min(bonus, 0.28) + anchor_bonus, item.score)

        return sorted(results, key=rank_key, reverse=True)[:top_k]

    def search(self, question: str, top_k: int = 5, course: str | None = None) -> list[SearchResult]:
        from qdrant_client import models
        question = question.strip()
        if not question or len(question) > 2000 or not 1 <= top_k <= 100:
            raise ValueError("问题长度必须为 1—2000 字，top_k 必须在 1—100 之间")
        query_filter = models.Filter(must=[
            models.FieldCondition(key="course", match=models.MatchValue(value=course))
        ]) if course else None
        with self._lock:
            if self._closed:
                raise RAGError("index_closed", "索引已关闭")
            query = self.embedder.encode([question])[0]
            # Retrieve a wider semantic candidate set, then use terms from the
            # question as a deterministic local tie-breaker. It prevents exact
            # titled definitions/formulas from being excluded by a very close
            # vector-score cutoff.
            candidate_k = min(max(top_k * 4, 16), 100)
            points = self.client.query_points(
                collection_name=self.collection, query=query, query_filter=query_filter,
                limit=candidate_k, with_payload=True,
            ).points
        results = []
        for point in points:
            payload = point.payload or {}
            results.append(SearchResult(
                float(point.score), str(payload.get("text", "")), str(payload.get("source_file", "")),
                str(payload.get("section", "")), str(payload.get("course", "")),
                int(payload.get("page", 1)), str(payload.get("chunk_id", point.id)),
            ))
        anchor_candidates = self._anchor_candidates(question, course)
        if anchor_candidates:
            known = {(item.source_file, item.chunk_id) for item in results}
            results.extend(
                item for item in anchor_candidates
                if (item.source_file, item.chunk_id) not in known
            )
        return self._lexical_rerank(question, results, top_k)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self.client.close()
                self._closed = True
