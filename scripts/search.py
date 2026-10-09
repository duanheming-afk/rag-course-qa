"""用向量索引检索课程资料。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retriever import VectorRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检索课程资料")
    parser.add_argument("question", help="要检索的问题")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--course", default=None)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    args = parser.parse_args()

    retriever = VectorRetriever(
        index_path=PROJECT_ROOT / "index/qdrant",
        model_name=args.model,
    )
    try:
        results = retriever.search(args.question, top_k=args.top_k, course=args.course)
    finally:
        retriever.client.close()

    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}] score={result.score:.4f} {result.source_file} | {result.section}")
        print(result.text[:300].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
