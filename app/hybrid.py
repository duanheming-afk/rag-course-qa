"""Dense + BM25 稀疏检索与 RRF 融合。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .retriever import SearchResult, VectorRetriever


def tokenize(text: str) -> list[str]:
    """中文按字切分，英文和数字按连续词切分。"""
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())


class HybridRetriever:
    """使用 Dense 和 BM25 两路结果，通过 RRF 合并排名。"""

    def __init__(
        self,
        chunks_path: str | Path = "data/processed/chunks.jsonl",
        index_path: str | Path = "index/qdrant",
        collection: str = "course_kb",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        dense_retriever: VectorRetriever | None = None,
    ) -> None:
        self.chunks = [
            json.loads(line)
            for line in Path(chunks_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.chunks:
            raise ValueError("BM25 文本块为空，请先运行 ingest")
        self.bm25 = BM25Okapi([tokenize(row["text"]) for row in self.chunks])
        self._owns_dense = dense_retriever is None
        self.dense = dense_retriever or VectorRetriever(index_path, collection, embedding_model)

    @staticmethod
    def _to_result(row: dict, score: float) -> SearchResult:
        return SearchResult(
            score=score,
            text=str(row["text"]),
            source_file=str(row["source_file"]),
            section=str(row.get("section", "")),
            course=str(row.get("course", "")),
            page=int(row.get("page", 1)),
            chunk_id=str(row["chunk_id"]),
        )

    def sparse_search(self, question: str, candidate_k: int = 20, course: str | None = None) -> list[SearchResult]:
        scores = self.bm25.get_scores(tokenize(question))
        indices = sorted((i for i in range(len(scores)) if not course or self.chunks[i].get("course") == course), key=lambda i: float(scores[i]), reverse=True)
        return [self._to_result(self.chunks[i], float(scores[i])) for i in indices[:candidate_k]]

    def search(
        self, question: str, top_k: int = 5, candidate_k: int = 20, rrf_k: int = 60,
        course: str | None = None,
    ) -> list[SearchResult]:
        if not 1 <= top_k <= candidate_k <= 100 or rrf_k < 0:
            raise ValueError("要求 1 <= top_k <= candidate_k <= 100，rrf_k >= 0")
        dense_results = self.dense.search(question, top_k=candidate_k, course=course)
        sparse_results = self.sparse_search(question, candidate_k=candidate_k, course=course)
        dense_rank = {item.chunk_id: rank for rank, item in enumerate(dense_results, start=1)}
        sparse_rank = {item.chunk_id: rank for rank, item in enumerate(sparse_results, start=1)}
        by_id = {item.chunk_id: item for item in [*dense_results, *sparse_results]}

        fused: list[tuple[float, SearchResult]] = []
        for chunk_id, item in by_id.items():
            score = 0.0
            if chunk_id in dense_rank:
                score += 1.0 / (rrf_k + dense_rank[chunk_id])
            if chunk_id in sparse_rank:
                score += 1.0 / (rrf_k + sparse_rank[chunk_id])
            fused.append((score, item))
        fused.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SearchResult(
                score=score,
                text=item.text,
                source_file=item.source_file,
                section=item.section,
                course=item.course,
                page=item.page,
                chunk_id=item.chunk_id,
            )
            for score, item in fused[:top_k]
        ]

    def close(self) -> None:
        if self._owns_dense:
            self.dense.client.close()
