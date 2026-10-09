"""Reproducible answer evaluation; automatic proxies and manual review stay separate."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.citations import cited_source_files, is_refusal, resolve_citations
from app.config import Settings
from app.errors import RAGError
from app.evidence import EvidenceVerifier
from app.generator import OllamaGenerator
from app.guardrails import clarification_message, immediate_refusal_reason, refusal_message, refusal_reason
from app.pipeline import RAGPipeline
from app.retriever import VectorRetriever

SCHEMES = ("Dense", "Hybrid_RRF", "Dense_Rerank", "Hybrid_RRF_Rerank")


def load_questions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("评测集为空")
    ids = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or row["id"] in ids or not row.get("question", "").strip():
            raise ValueError("评测题必须有唯一字符串 id 和非空 question")
        ids.add(row["id"])
        if not isinstance(row.get("required_keywords", []), list):
            raise ValueError("required_keywords 必须是列表")
    return rows


def keyword_recall(answer: str, keywords: list) -> float:
    """Literal keyword proxy only: notation changes and synonyms can yield false negatives."""
    normalized = re.sub(r"\s+", "", answer).lower()
    matches = 0
    for keyword in keywords:
        alternatives = keyword if isinstance(keyword, list) else [keyword]
        matches += any(re.sub(r"\s+", "", term).lower() in normalized for term in alternatives)
    return matches / len(keywords) if keywords else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("不能汇总空评测")
    answerable = [r for r in rows if r["expected_behavior"] == "answer"]
    negative = [r for r in rows if r["expected_behavior"] == "refuse"]
    ambiguous = [r for r in rows if r["expected_behavior"] == "clarify"]
    citation_rows = [r for r in answerable if r["gold_source_files"]]
    def rate(items, field):
        return round(sum(float(r.get(field, 0) or 0) for r in items) / len(items), 4) if items else None
    return {
        "count": len(rows), "success_rate": rate(rows, "success"),
        "keyword_recall_proxy": rate(answerable, "keyword_recall"),
        "all_gold_files_retrieved_rate": rate(citation_rows, "retrieved_gold"),
        "all_gold_files_cited_rate": rate(citation_rows, "citation_gold"),
        "explicit_refusal_rate_on_unanswerable": rate(negative, "refused"),
        "clarification_rate_on_ambiguous": rate(ambiguous, "clarified"),
        "false_refusal_rate_on_answerable": rate(answerable, "refused"),
        "invalid_citation_case_count": sum(bool(r.get("invalid_citation_ids")) for r in rows),
        "missing_citation_case_count": sum(r.get("citation_missing", False) for r in answerable),
        "avg_evidence_coverage": rate(answerable, "evidence_coverage"),
        "fully_evidenced_answer_rate": rate(answerable, "fully_evidenced"),
        "avg_latency_ms": round(mean(r["latency_ms"] for r in rows), 2),
        "p50_latency_ms": round(percentile([r["latency_ms"] for r in rows], .5), 2),
        "p95_latency_ms": round(percentile([r["latency_ms"] for r in rows], .95), 2),
        "manual_correctness": None, "manual_evidence_support": None,
    }


def evaluate_scheme(name, questions, search_fn, generator, evidence_verifier, top_k, reranker=None, candidate_k=20):
    rows = []
    for number, question in enumerate(questions, 1):
        start = time.perf_counter()
        gold = question.get("gold_source_files") or (
            [question["gold_source_file"]] if question.get("gold_source_file") else []
        )
        row = {
            "id": question["id"], "question": question["question"],
            "expected_behavior": question.get("expected_behavior", "answer"),
            "reference_answer": question.get("reference_answer", ""),
            "required_keywords": question.get("required_keywords", []),
            "gold_source_files": gold, "success": False, "answer": "", "raw_answer": "", "refused": False,
            "clarified": False,
            "keyword_recall": 0.0, "retrieved_gold": False, "citation_gold": False,
            "sources": [], "retrieved": [], "warnings": [], "invalid_citation_ids": [],
            "citation_missing": True, "citation_mode": "not_generated", "error": None,
            "evidence": [], "evidence_coverage": 0.0, "fully_evidenced": False,
            "removed_claim_count": 0,
            "manual_correctness": None, "manual_evidence_support": None,
        }
        extractive_fallback_used = False
        try:
            immediate_refusal = immediate_refusal_reason(question["question"])
            if immediate_refusal:
                row.update(success=True, answer=refusal_message(immediate_refusal), refused=True,
                           citation_missing=False, citation_mode="not_required", warnings=[immediate_refusal])
                row["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
                rows.append(row)
                print(f"{name}: {number}/{len(questions)} OK", flush=True)
                continue
            clarification = clarification_message(question["question"])
            if clarification:
                row.update(success=True, answer=clarification, clarified=True, citation_missing=False,
                           citation_mode="not_required")
                row["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
                rows.append(row)
                print(f"{name}: {number}/{len(questions)} OK", flush=True)
                continue
            results = search_fn(question, candidate_k if reranker else top_k)
            if reranker:
                results = reranker.rerank(question["question"], results, top_k=top_k)
            row["retrieval_ms"] = round((time.perf_counter() - start) * 1000, 2)
            row["retrieved"] = [
                {"citation_id": i, "source_file": r.source_file, "section": r.section,
                 "chunk_id": r.chunk_id, "score": r.score, "text": r.text}
                for i, r in enumerate(results, 1)
            ]
            row["retrieved_gold"] = bool(gold) and set(gold) <= {r.source_file for r in results}
            refusal = refusal_reason(question["question"], results)
            if refusal:
                row.update(success=True, answer=refusal_message(refusal), refused=True,
                           citation_missing=False, citation_mode="not_required", warnings=[refusal])
                row["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
                rows.append(row)
                print(f"{name}: {number}/{len(questions)} OK", flush=True)
                continue
            response = generator.generate(question["question"], results)
            if response.refused:
                # Keep the evaluator aligned with the production pipeline's
                # narrow extractive fallback for explicit formula/complexity
                # line hints. It still goes through EvidenceVerifier below.
                extractive = RAGPipeline._extractive_formula_fallback(question["question"], results)
                if extractive:
                    response.answer = extractive
                    response.refused = False
                    response.citation_mode = "extractive_fallback"
                    extractive_fallback_used = True
                    response.warnings.append("模型未生成答案，已从命中资料中摘录公式/复杂度并经过证据校验。")
            row["raw_answer"] = response.answer
            if response.refused:
                row.update(success=True, answer=response.answer, refused=True, citation_missing=False,
                           citation_mode="not_required", warnings=response.warnings)
                row["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
                rows.append(row)
                print(f"{name}: {number}/{len(questions)} OK", flush=True)
                continue
            verification = evidence_verifier.verify(response.answer, results)
            evidence_rows = [
                {"claim_id": item.claim_id, "claim": item.claim, "source_ids": item.source_ids,
                 "support_score": item.support_score}
                for item in verification.evidence
            ]
            if verification.answer == "资料中没有提到。":
                row.update(
                    success=True, answer=verification.answer, refused=True, citation_missing=False,
                    citation_mode="not_required", evidence=evidence_rows,
                    evidence_coverage=verification.coverage, fully_evidenced=False,
                    removed_claim_count=len(verification.unsupported_claims),
                    warnings=[*response.warnings, "生成回答的可核验证据不足，已拒绝输出。"],
                )
                row["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
                rows.append(row)
                print(f"{name}: {number}/{len(questions)} OK", flush=True)
                continue
            citations = resolve_citations(verification.answer, results)
            cited_files = {s["source_file"] for s in citations.sources}
            row.update(
                success=True, answer=verification.answer, refused=is_refusal(verification.answer),
                keyword_recall=keyword_recall(verification.answer, row["required_keywords"]),
                citation_gold=bool(gold) and set(gold) <= cited_files,
                sources=citations.sources, invalid_citation_ids=citations.invalid_ids,
                citation_missing=citations.missing,
                evidence=evidence_rows, evidence_coverage=verification.coverage,
                fully_evidenced=verification.coverage == 1.0,
                removed_claim_count=len(verification.unsupported_claims),
                citation_mode="extractive_fallback" if extractive_fallback_used else "sentence_verified",
                warnings=[*(warning for warning in response.warnings if "模型未在正文标注引用" not in warning), *(
                    [f"已移除 {len(verification.unsupported_claims)} 条缺少证据的表述。"]
                    if verification.unsupported_claims else []
                )],
            )
        except Exception as exc:
            row["error"] = {"code": getattr(exc, "code", "evaluation_error"), "message": str(exc)}
        row["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        rows.append(row)
        print(f"{name}: {number}/{len(questions)} {'OK' if row['success'] else row['error']['code']}", flush=True)
    return {"scheme": name, **summarize(rows)}, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="最终答案评测，默认仅运行当前线上 Dense 策略")
    parser.add_argument("--questions", default="data/eval/answer_pairs.jsonl")
    parser.add_argument("--include-challenges", action="store_true")
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=["Dense"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--host", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--output-dir", default="reports/runs")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 8 or not args.top_k <= args.candidate_k <= 100 or args.limit < 0:
        parser.error("要求 1 <= top-k <= 8，top-k <= candidate-k <= 100，limit >= 0")
    paths = [PROJECT_ROOT / args.questions]
    if args.include_challenges:
        paths.append(PROJECT_ROOT / "data/eval/challenge_pairs.jsonl")
    questions = [row for path in paths for row in load_questions(path)]
    if len({q["id"] for q in questions}) != len(questions):
        parser.error("合并的评测集中存在重复 id")
    if args.limit:
        questions = questions[:args.limit]
    settings = Settings.from_env()
    generator = OllamaGenerator(args.host, args.model, max_tokens=args.max_tokens, timeout=args.timeout)
    try:
        generator.check_available()
    except RAGError as exc:
        print(f"评测未开始：{exc}", file=sys.stderr)
        return 2
    with ExitStack() as stack:
        dense = VectorRetriever(settings.index_path, settings.collection, settings.embedding_model)
        stack.callback(dense.close)
        evidence_verifier = EvidenceVerifier(
            dense.embedder, settings.evidence_min_score, settings.evidence_min_coverage,
        )
        hybrid = reranker = None
        if any(name.startswith("Hybrid") for name in args.schemes):
            from app.hybrid import HybridRetriever
            hybrid = HybridRetriever(settings.chunks_path, settings.index_path, dense_retriever=dense)
        if any("Rerank" in name for name in args.schemes):
            from app.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker(args.reranker_model)
        summaries, details = [], {}
        for scheme in dict.fromkeys(args.schemes):
            if scheme.startswith("Hybrid"):
                search = lambda q, k: hybrid.search(q["question"], top_k=k, candidate_k=args.candidate_k, course=q.get("course"))
            else:
                search = lambda q, k: dense.search(q["question"], top_k=k, course=q.get("course"))
            summary, rows = evaluate_scheme(
                scheme, questions, search, generator, evidence_verifier, args.top_k,
                reranker if "Rerank" in scheme else None, args.candidate_k,
            )
            summaries.append(summary)
            details[scheme] = rows
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = {
        "created_at": timestamp, "question_count": len(questions),
        "dataset_sources": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "selected_question_ids": [q["id"] for q in questions],
        "chunks_sha256": hashlib.sha256(settings.chunks_path.read_bytes()).hexdigest(),
        "configuration": {"model": generator.model, "host": generator.host, "max_tokens": generator.max_tokens,
                          "timeout": generator.timeout, "top_k": args.top_k, "candidate_k": args.candidate_k,
                          "embedding_model": settings.embedding_model, "temperature": generator.temperature,
                          "seed": 42, "strategy": args.schemes,
                          "evidence_min_score": settings.evidence_min_score,
                          "evidence_min_coverage": settings.evidence_min_coverage},
        "metric_note": "合成讲义、小规模自建测试。句子级证据是向量语义匹配，不等于自然语言蕴含证明；关键词命中和引用编号也不能替代人工审核。错误样本计入分母。",
        "summaries": summaries, "details": details,
    }
    output = PROJECT_ROOT / args.output_dir / timestamp
    output.mkdir(parents=True, exist_ok=False)
    (output / "answers.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manual = {
        "run_id": timestamp,
        "rubric": {"correctness": "0错误/未完成，1部分正确，2完整正确",
                   "evidence_support": "0无依据/未完成，1部分有依据，2关键结论均有依据",
                   "citation_support": "0引用错误/缺失，1部分支持，2全部支持；合理拒答可填N/A"},
        "review_instructions": "逐题查看 answers.json 的实际答案和检索原文；保留错误样本，不将关键词评分当人工结论。",
        "rows": [{"scheme": s, "id": r["id"], "correctness": None, "evidence_support": None,
                  "citation_support": None, "reviewer": "", "notes": ""} for s, rows in details.items() for r in rows],
    }
    (output / "manual_review.json").write_text(json.dumps(manual, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 最终答案评测", "", report["metric_note"], "", f"题数：{len(questions)}；模型：{generator.model}", "",
             "| 策略 | 请求成功率 | 关键词代理分 | 目标文件引用率 | 平均证据覆盖率 | 全句证据率 | 无依据题拒答率 | 歧义题澄清率 | P50 ms | P95 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        def fmt(value):
            return "N/A" if value is None else f"{value:.2%}"
        lines.append(f"| {s['scheme']} | {fmt(s['success_rate'])} | {fmt(s['keyword_recall_proxy'])} | "
                     f"{fmt(s['all_gold_files_cited_rate'])} | {fmt(s['avg_evidence_coverage'])} | "
                     f"{fmt(s['fully_evidenced_answer_rate'])} | {fmt(s['explicit_refusal_rate_on_unanswerable'])} | "
                     f"{fmt(s['clarification_rate_on_ambiguous'])} | {s['p50_latency_ms']} | {s['p95_latency_ms']} |")
    lines.extend(["", "配置、数据哈希、每题答案/错误及证据见 answers.json。人工审核见 manual_review.json。",
                  "耗时包含逐题检索与生成；首题可能包含冷启动，不作为纯模型性能基准。"])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已保存：{output}", flush=True)
    return 0 if all(row["success"] for rows in details.values() for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
