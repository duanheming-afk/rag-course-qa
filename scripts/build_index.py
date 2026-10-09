"""构建课程资料的本地 Qdrant 向量索引。

用法:
    python scripts/build_index.py
    python scripts/build_index.py --model BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.indexer import QdrantIndexer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="建立课程资料 Qdrant 向量索引")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--index", default="index/qdrant")
    parser.add_argument("--collection", default="course_kb")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    indexer = QdrantIndexer(
        index_path=PROJECT_ROOT / args.index,
        collection=args.collection,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    try:
        count = indexer.build(PROJECT_ROOT / args.chunks)
    finally:
        indexer.close()
    print(f"索引构建完成，共写入 {count} 个向量")
    print(f"索引位置: {PROJECT_ROOT / args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
