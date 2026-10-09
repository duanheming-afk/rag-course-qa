"""执行一次完整的 RAG 问答。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import RAGPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="课程资料 RAG 问答")
    parser.add_argument("question", help="用户问题")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--course", default=None, help="可选，例如：机器学习")
    parser.add_argument("--model", default=None, help="Ollama 模型名")
    parser.add_argument("--host", default=None, help="Ollama 地址")
    parser.add_argument("--max-tokens", type=int, default=None, help="覆盖 .env 的生成 token 预算")
    parser.add_argument("--timeout", type=float, default=None, help="覆盖 .env 的超时秒数")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        llm_host=args.host,
        llm_model=args.model,
        llm_max_tokens=args.max_tokens,
        llm_timeout=args.timeout,
    )
    try:
        response = pipeline.ask(args.question, top_k=args.top_k, course=args.course)
    finally:
        pipeline.close()

    print("\n=== 答案 ===")
    print(response.answer)
    print("\n=== 检索来源 ===")
    label = {
        "model": "模型引用",
        "sentence_verified": "句子级证据引用",
        "extractive_fallback": "资料摘录兜底",
        "retrieval_fallback": "最相关检索片段",
    }.get(response.citation_mode, "检索来源")
    print(f"\n=== {label} ===")
    for source in response.sources:
        print(f"- [资料{source['citation_id']}] {source['source_file']} | {source['section']}")
    for warning in response.warnings:
        print(f"提示：{warning}")
    if response.evidence:
        print(f"\n=== 句子级证据校验（覆盖率 {response.evidence_coverage:.0%}）===")
        for item in response.evidence:
            refs = " ".join(f"[资料{source_id}]" for source_id in item.source_ids)
            print(f"- {item.claim} {refs}（匹配分 {item.support_score:.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
