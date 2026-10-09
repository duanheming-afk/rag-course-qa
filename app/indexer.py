"""将标准化 JSONL 文本块向量化并写入本地 Qdrant。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .embedding import TextEmbedder


class QdrantIndexer:
    """本地 Qdrant 索引构建器。"""

    def __init__(
        self,
        index_path: str | Path = "index/qdrant",
        collection: str = "course_kb",
        model_name: str = "BAAI/bge-small-zh-v1.5",
        batch_size: int = 32,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("缺少 qdrant-client，请先执行: pip install -r requirements.txt") from exc

        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.collection = collection
        self.batch_size = batch_size
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.client = QdrantClient(path=str(self.index_path))
        try:
            self.embedder = TextEmbedder(model_name=model_name, batch_size=batch_size)
        except Exception:
            self.client.close()
            raise

    def _collection_exists(self) -> bool:
        names = {item.name for item in self.client.get_collections().collections}
        return self.collection in names

    def build(self, chunks_path: str | Path, reset: bool = True) -> int:
        """读取 JSONL，重建 collection，并返回写入数量。"""
        from qdrant_client import models

        chunks_path = Path(chunks_path)
        if not chunks_path.is_file():
            raise FileNotFoundError(f"文本块文件不存在: {chunks_path}")

        rows: list[dict[str, Any]] = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("文本块文件为空，无法建立索引")

        if self._collection_exists():
            if not reset:
                raise RuntimeError(f"collection 已存在: {self.collection}")
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension,
                distance=models.Distance.COSINE,
            ),
        )

        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            vectors = self.embedder.encode([row["text"] for row in batch])
            points = []
            for row, vector in zip(batch, vectors, strict=True):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, row["chunk_id"]))
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=row,
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            print(f"已写入 {min(start + len(batch), len(rows))}/{len(rows)} 个向量")

        return len(rows)

    def close(self) -> None:
        self.client.close()
