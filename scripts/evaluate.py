"""Compare retrieval using file-level relevance and per-query end-to-end latency."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.retriever import VectorRetriever
from scripts.evaluate_answers import load_questions, percentile


def hit_rank(results, gold_file, k):
    return next((i for i, r in enumerate(results[:k], 1) if r.source_file == gold_file), None)


def summarize(rows, k):
    if not rows:
        raise ValueError("不能汇总空评测")
    ranks = [r["rank"] for r in rows if r["rank"] is not None]
    times = [r["latency_ms"] for r in rows]
    return {
        "hit_rate@1": round(sum(r["rank"] == 1 for r in rows) / len(rows), 4),
        f"hit_rate@{k}": round(len(ranks) / len(rows), 4),
        "mrr": round(sum(1 / r for r in ranks) / len(rows), 4),
        "avg_latency_ms": round(sum(times) / len(times), 2),
        "p50_latency_ms": round(percentile(times, .5), 2),
        "p95_latency_ms": round(percentile(times, .95), 2),
    }


def evaluate_scheme(name, questions, search_fn, k):
    rows = []
    for q in questions:
        start = time.perf_counter()
        results = search_fn(q["question"])
        rows.append({
            "id": q["id"], "question": q["question"], "gold_source_file": q["gold_source_file"],
            "rank": hit_rank(results, q["gold_source_file"], k),
            "top_source": results[0].source_file if results else None,
            "latency_ms": (time.perf_counter() - start) * 1000,
        })
    return {"scheme": name, **summarize(rows, k)}, rows


def evaluate_rerank_scheme(name, questions, candidate_fn, reranker, k):
    # Time retrieval AND reranking per query; batch throughput is a different metric.
    return evaluate_scheme(name, questions, lambda q: reranker.rerank(q, candidate_fn(q), top_k=k), k)


def main():
    p = argparse.ArgumentParser(description="检索评测（文件级相关性，不代表最终答案质量）")
    p.add_argument("--questions", default="data/eval/qa_pairs.jsonl")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    p.add_argument("--skip-rerank", action="store_true")
    p.add_argument("--output-dir", default="reports/retrieval_runs")
    args = p.parse_args()
    if not 1 <= args.top_k <= args.candidate_k <= 100:
        p.error("要求 1 <= top-k <= candidate-k <= 100")
    path = PROJECT_ROOT / args.questions
    questions = load_questions(path)
    if any(not q.get("gold_source_file") for q in questions):
        p.error("检索评测每题必须标注 gold_source_file")
    settings = Settings.from_env()
    with ExitStack() as stack:
        dense = VectorRetriever(settings.index_path, settings.collection, settings.embedding_model)
        stack.callback(dense.close)
        from app.hybrid import HybridRetriever
        hybrid = HybridRetriever(settings.chunks_path, settings.index_path, dense_retriever=dense)
        searches = {
            "Dense": lambda q: dense.search(q, top_k=args.top_k),
            "Hybrid_RRF": lambda q: hybrid.search(q, top_k=args.top_k, candidate_k=args.candidate_k),
        }
        summaries, details = [], {}
        for name, search in searches.items():
            summary, rows = evaluate_scheme(name, questions, search, args.top_k)
            summaries.append(summary)
            details[name] = rows
            print(summary, flush=True)
        if not args.skip_rerank:
            from app.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker(args.reranker_model)
            candidates = {
                "Dense_Rerank": lambda q: dense.search(q, top_k=args.candidate_k),
                "Hybrid_RRF_Rerank": lambda q: hybrid.search(q, top_k=args.candidate_k, candidate_k=args.candidate_k),
            }
            for name, search in candidates.items():
                summary, rows = evaluate_rerank_scheme(name, questions, search, reranker, args.top_k)
                summaries.append(summary)
                details[name] = rows
                print(summary, flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = PROJECT_ROOT / args.output_dir / stamp
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": stamp, "question_count": len(questions),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "chunks_sha256": hashlib.sha256(settings.chunks_path.read_bytes()).hexdigest(),
        "configuration": {"embedding_model": settings.embedding_model, "top_k": args.top_k,
                          "candidate_k": args.candidate_k, "reranker_model": args.reranker_model},
        "note": "合成讲义上的文件级检索指标。耗时为单题检索与重排总耗时（含首题热身），不含模型加载。旧版 Rerank 报告漏计检索耗时且使用批量均摊，不可直接对比。",
        "summaries": summaries, "details": details,
    }
    (output / "retrieval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 检索评测", "", report["note"], "", "| 方案 | Top-1 | Top-K | MRR | 平均 ms | P95 ms |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['scheme']} | {s['hit_rate@1']:.2%} | {s[f'hit_rate@{args.top_k}']:.2%} | "
                     f"{s['mrr']:.4f} | {s['avg_latency_ms']} | {s['p95_latency_ms']} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
