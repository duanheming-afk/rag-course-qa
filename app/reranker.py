"""候选文本重排序。"""

from __future__ import annotations

from .retriever import SearchResult


class CrossEncoderReranker:
    """使用 CrossEncoder 重新判断问题与文本的相关性。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("缺少 sentence-transformers，无法使用 Rerank") from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        results: list[SearchResult],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(question, result.text) for result in results]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return self._sort_results(results, scores, top_k)

    def rerank_many(
        self,
        requests: list[tuple[str, list[SearchResult]]],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[list[SearchResult]]:
        """批量重排多个问题，避免每个问题单独启动一次模型推理。"""
        pairs: list[tuple[str, str]] = []
        sizes: list[int] = []
        for question, results in requests:
            sizes.append(len(results))
            pairs.extend((question, result.text) for result in results)
        if not pairs:
            return [[] for _ in requests]

        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        output: list[list[SearchResult]] = []
        offset = 0
        for (_, results), size in zip(requests, sizes, strict=True):
            output.append(self._sort_results(results, scores[offset : offset + size], top_k))
            offset += size
        return output

    @staticmethod
    def _sort_results(
        results: list[SearchResult], scores, top_k: int
    ) -> list[SearchResult]:
        ranked = sorted(
            zip(scores, results, strict=True), key=lambda item: float(item[0]), reverse=True
        )
        return [
            SearchResult(
                score=float(score),
                text=result.text,
                source_file=result.source_file,
                section=result.section,
                course=result.course,
                page=result.page,
                chunk_id=result.chunk_id,
            )
            for score, result in ranked[:top_k]
        ]
