from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from app.generator import OllamaGenerator
from app.guardrails import clarification_message, immediate_refusal_reason, refusal_reason
from app.evidence import EvidenceVerifier, split_claims
from app.retriever import SearchResult
from scripts.evaluate_answers import cited_source_files


def result(source_file: str) -> SearchResult:
    return SearchResult(
        score=0.9,
        text="测试资料",
        source_file=source_file,
        section="测试章节",
        course="测试课程",
        page=1,
        chunk_id="chunk-1",
    )


class CitationTests(unittest.TestCase):
    def test_numeric_citation_maps_to_retrieved_result(self) -> None:
        results = [result("a.md"), result("b.md")]
        self.assertEqual(cited_source_files("结论 [资料2]。", results), {"b.md"})

    def test_citation_cleanup_removes_brackets_with_marker(self) -> None:
        from app.citations import CITATION_RE
        self.assertEqual(CITATION_RE.sub("", "结论 [资料1] 和资料2。"), "结论  和。")

    def test_filename_citation_is_supported(self) -> None:
        results = [result("a.md"), result("b.md")]
        self.assertEqual(cited_source_files("来源：a.md", results), {"a.md"})

    def test_out_of_range_citation_is_ignored(self) -> None:
        results = [result("a.md")]
        self.assertEqual(cited_source_files("来源：[资料3]。", results), set())


class RetrievalRankingTests(unittest.TestCase):
    def test_definition_heading_is_preferred_when_semantic_scores_are_close(self) -> None:
        from app.retriever import VectorRetriever
        candidates = [
            SearchResult(.64, "左右导数内容", "导数.md", "1.3 单侧导数", "高等数学", 1, "one-side"),
            SearchResult(.55, "函数在该点的导数定义", "导数.md", "1.2 定义", "高等数学", 1, "definition"),
        ]
        ranked = VectorRetriever._lexical_rerank("什么是导数的定义？", candidates, top_k=2)
        self.assertEqual(ranked[0].chunk_id, "definition")

    def test_compact_technical_noun_can_match_a_heading(self) -> None:
        from app.retriever import VectorRetriever
        self.assertIn("栈", VectorRetriever._query_terms("比较栈与队列"))

    def test_explicit_quoted_line_hint_is_extracted(self) -> None:
        from app.retriever import VectorRetriever
        self.assertEqual(
            VectorRetriever._anchor_terms("围绕“典型例题”的线索“例 3”回答"),
            ["例3"],
        )

    def test_explicit_line_hint_beats_unrelated_dense_match(self) -> None:
        from app.retriever import VectorRetriever
        candidates = [
            SearchResult(.86, "其他章节内容", "课程.md", "其他", "课程", 1, "other"),
            SearchResult(.10, "**例 3**：目标公式与条件。", "课程.md", "典型例题", "课程", 1, "target"),
        ]
        ranked = VectorRetriever._lexical_rerank(
            "围绕“典型例题”的线索“例 3”回答", candidates, top_k=2,
        )
        self.assertEqual(ranked[0].chunk_id, "target")

    def test_course_title_can_be_used_as_explicit_anchor(self) -> None:
        from app.retriever import VectorRetriever
        item = SearchResult(.1, "目标内容", "线代-05-二次型与矩阵分解.md", "其他章节", "线性代数", 1, "target")
        self.assertEqual(
            VectorRetriever._anchor_matches("围绕“二次型与矩阵分解”的线索回答", item),
            ["二次型与矩阵分解"],
        )


class GuardrailTests(unittest.TestCase):
    def test_deictic_question_requests_context(self) -> None:
        self.assertIsNotNone(clarification_message("第二个定理是什么意思？"))

    def test_common_deictic_question_variants_request_context(self) -> None:
        questions = [
            "这里的第二步是什么意思？",
            "为什么结果会变成这样？",
            "第一个方法和另一个有什么区别？",
            "这部分应该怎么理解？",
            "这里为什么要这样计算？",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNotNone(clarification_message(question))

    def test_wifi_password_is_refused_without_using_retrieval(self) -> None:
        self.assertIsNotNone(refusal_reason("神经网络实验室的 WiFi 密码是什么？", []))

    def test_sensitive_question_has_priority_over_deictic_wording(self) -> None:
        self.assertIsNotNone(immediate_refusal_reason("这些课程资料中 WiFi 密码是什么？"))

    def test_service_handles_ambiguous_question_without_creating_pipeline(self) -> None:
        import app.service as service
        previous = service._pipeline
        service._pipeline = None
        try:
            response = service.ask_question("它为什么是这样？")
            self.assertEqual(response.citation_mode, "not_required")
            self.assertIsNone(service._pipeline)
        finally:
            service._pipeline = previous


class EvidenceTests(unittest.TestCase):
    class FakeEmbedder:
        def encode(self, texts):
            vectors = []
            for text in texts:
                if "导数" in text:
                    vectors.append([1.0, 0.0])
                elif "太阳" in text:
                    vectors.append([-1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

    def test_split_claims_preserves_formula_with_its_claim(self) -> None:
        claims = split_claims("导数定义为极限：$$f'(x)=\\lim h$$。\n- 第二条结论。")
        self.assertEqual(len(claims), 2)
        self.assertIn("\\lim", claims[0])

    def test_unsupported_claim_is_removed_and_supported_claim_gets_citation(self) -> None:
        results = [
            SearchResult(.9, "导数是差商的极限。", "a.md", "定义", "高等数学", 1, "a"),
            SearchResult(.8, "积分的基本性质。", "b.md", "积分", "高等数学", 1, "b"),
        ]
        verification = EvidenceVerifier(self.FakeEmbedder(), min_score=.5, min_coverage=.5).verify(
            "导数是差商的极限。太阳从西边升起。", results,
        )
        self.assertIn("导数是差商的极限。 [资料1]", verification.answer)
        self.assertEqual(verification.unsupported_claims, ["太阳从西边升起。"])
        self.assertEqual(verification.coverage, .5)


class PipelineFallbackTests(unittest.TestCase):
    def test_formula_fallback_only_quotes_matching_explicit_anchor(self) -> None:
        from app.pipeline import RAGPipeline
        results = [
            SearchResult(
                .8, "章节：二、基本积分公式\n- $\\int x^a \\, dx = \\frac{x^{a+1}}{a+1} + C$ $(a \\neq -1)$",
                "calculus.md", "二、基本积分公式", "高等数学", 1, "formula",
            ),
        ]
        question = "围绕“基本积分公式”的线索“积分公式”，写出关键公式或关系。"
        fallback = RAGPipeline._extractive_formula_fallback(question, results)
        self.assertIn("\\int x^a", fallback)
        self.assertIn("[资料1]", fallback)

    def test_formula_fallback_does_not_run_without_explicit_anchor(self) -> None:
        from app.pipeline import RAGPipeline
        results = [SearchResult(.8, "- $x = 1$", "a.md", "公式", "课程", 1, "formula")]
        self.assertIsNone(RAGPipeline._extractive_formula_fallback("请写出关键公式。", results))


class HoldoutWorkflowTests(unittest.TestCase):
    def test_generated_review_queue_has_unique_questions(self) -> None:
        from scripts.build_holdout_review_queue import build_queue
        chunks = [
            {"chunk_id": "a", "source_file": "a.md", "course": "课程A", "title": "讲义A", "section": "1.1 定义", "text": "章节：1.1 定义\n**概念 A**：这是内容。", "chunk_index": 1},
            {"chunk_id": "b", "source_file": "b.md", "course": "课程B", "title": "讲义B", "section": "1.1 定义", "text": "章节：1.1 定义\n**概念 B**：这是内容。", "chunk_index": 1},
        ]
        rows = build_queue(chunks, answer_count=2, per_document=1)
        self.assertEqual(len(rows), len({row["question"] for row in rows}))

    def test_only_complete_human_approval_can_be_exported(self) -> None:
        from scripts.export_approved_holdout import approved_rows
        queue = [{
            "id": "h001", "question": "什么是测试概念？", "course": "测试课程",
            "expected_behavior": "answer", "category": "definition",
            "annotation": {"status": "approved", "reviewer": "reviewer", "reviewed_at": "2026-09-20",
                           "gold_source_files": ["test.md"], "reference_answer": "人工答案",
                           "required_keywords": ["测试概念"], "notes": ""},
        }]
        exported, errors = approved_rows(queue, set())
        self.assertEqual(errors, [])
        self.assertEqual(exported[0]["id"], "h001")


class GeneratorTests(unittest.TestCase):
    def test_prompt_requires_existing_numeric_citations(self) -> None:
        generator = OllamaGenerator()
        system, user = generator._prompt("问题", [result("a.md")])
        self.assertIn("必须使用上下文中已有的 [资料N] 编号", system)
        self.assertIn("优先摘录参考资料中的对应表述", system)
        self.assertIn("来源格式为 [资料1] [资料2]", user)

    def test_thinking_only_response_is_not_returned_as_answer(self) -> None:
        generator = OllamaGenerator(timeout=1)
        response = {"message": {"content": "", "thinking": "内部推理内容"}}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(response).encode("utf-8")

        class FakeOpener:
            def open(self, *args, **kwargs):
                return FakeResponse()

        with patch("app.generator.build_opener", return_value=FakeOpener()):
            with self.assertRaisesRegex(RuntimeError, "最终答案"):
                generator.generate("问题", [result("a.md")])

    def test_empty_citation_placeholders_are_removed(self) -> None:
        self.assertEqual(
            OllamaGenerator._clean_answer("[] [] 结论根据资料[]和[资料1]。"),
            "结论根据资料和[资料1]。",
        )

    def test_truncated_answer_is_retried_once_with_short_format(self) -> None:
        generator = OllamaGenerator(timeout=1)
        truncated = {"done": True, "done_reason": "length", "message": {"content": "未完成"}}
        complete = {"done": True, "done_reason": "stop", "message": {"content": "完整结论 [资料1]。"}}
        with patch.object(generator, "_chat", side_effect=[truncated, complete]) as chat:
            response = generator.generate("问题", [result("a.md")])
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(response.answer, "完整结论 [资料1]。")


if __name__ == "__main__":
    unittest.main()
