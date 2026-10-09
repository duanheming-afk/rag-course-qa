"""将 data/raw 中的课程资料处理为标准化 JSONL 文本块。

用法:
    python scripts/ingest.py
    python scripts/ingest.py --chunk-size 500 --chunk-overlap 80
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.loader import load_directory  # noqa: E402
from app.splitter import split_documents, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="处理高校课程 Markdown 资料")
    parser.add_argument("--input", default="data/raw", help="原始资料目录")
    parser.add_argument("--output", default="data/processed", help="处理结果目录")
    parser.add_argument("--chunk-size", type=int, default=500, help="文本块最大字符数")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="相邻文本块重叠字符数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = PROJECT_ROOT / args.input
    output_root = PROJECT_ROOT / args.output

    documents = load_directory(raw_root)
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    chunks_path = output_root / "chunks.jsonl"
    manifest_path = output_root / "manifest.json"
    write_jsonl(chunks, chunks_path)

    manifest = {
        "input": str(raw_root),
        "output": str(chunks_path),
        "source_file_count": len({doc.source_file for doc in documents}),
        "document_section_count": len(documents),
        "chunk_count": len(chunks),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "excluded_files": ["INDEX.md"],
        "courses": dict(Counter(chunk.course for chunk in chunks)),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"读取课程文件: {manifest['source_file_count']} 份")
    print(f"识别章节单元: {manifest['document_section_count']} 个")
    print(f"生成标准文本块: {manifest['chunk_count']} 个")
    print(f"文本块文件: {chunks_path}")
    print(f"处理清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
