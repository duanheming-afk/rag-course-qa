"""Embedding 模型封装。

默认使用 BGE 中文向量模型，将课程文本映射为固定维度的语义向量。
"""

from __future__ import annotations

from collections.abc import Sequence


class TextEmbedder:
    """使用 sentence-transformers 生成文本向量。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 sentence-transformers，请先执行: pip install -r requirements.txt"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        get_dimension = getattr(
            self.model,
            "get_embedding_dimension",
            self.model.get_sentence_embedding_dimension,
        )
        dimension = get_dimension()
        if dimension is None:
            raise RuntimeError(f"无法获取 Embedding 维度: {model_name}")
        self.dimension = int(dimension)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """批量生成归一化向量，适合使用余弦距离检索。"""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()
