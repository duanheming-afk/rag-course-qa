from __future__ import annotations

import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

from app.citations import REFUSAL, resolve_citations
from app.evidence import EvidenceVerifier, evidence_body
from app.retriever import SearchResult


def result(text: str, chunk_id: str = "test") -> SearchResult:
    return SearchResult(.9, text, f"{chunk_id}.md", "定义", "课程", 1, chunk_id)


class RecordingEmbedder:
    def __init__(self, vectors=None):
        self.vectors = vectors
        self.texts = None

    def encode(self, texts):
        self.texts = list(texts)
        return self.vectors if self.vectors is not None else [[1., 0.] for _ in texts]


class EvidenceRegressionTests(unittest.TestCase):
    def test_title_and_test_notice_cannot_be_evidence(self):
        embedder = RecordingEmbedder()
        verified = EvidenceVerifier(embedder).verify("平均速度是位移与时间的比值。", [
            result("章节：质点运动学\n> 合成讲义 · 仅供 RAG 系统测试使用 · 学科:大学物理"),
        ])
        self.assertEqual(verified.answer, REFUSAL)
        self.assertEqual(verified.coverage, 0.)
        self.assertEqual(len(verified.unsupported_claims), 1)
        self.assertIsNone(embedder.texts)

    def test_filtered_candidates_keep_original_citation_id(self):
        results = [result("章节：课程标题", "title"), result("平均速度是位移与时间的比值。", "body")]
        verified = EvidenceVerifier(RecordingEmbedder()).verify("平均速度是位移与时间的比值。", results)
        self.assertEqual(verified.evidence[0].source_ids, [2])
        self.assertIn("[资料2]", verified.answer)
        self.assertEqual(resolve_citations(verified.answer, results).sources[0]["chunk_id"], "body")

    def test_body_match_is_separate_from_original_threshold_guard(self):
        embedder = RecordingEmbedder()
        EvidenceVerifier(embedder).verify("结论。", [result("章节：同名主题\n# 标题\n正文证据。")])
        self.assertEqual(embedder.texts, ["结论。", "章节：同名主题\n# 标题\n正文证据。", "正文证据。"])

    def test_body_matching_cannot_resurrect_original_low_score_claim(self):
        embedder = RecordingEmbedder([[1., 0.], [.4, .9165], [.9, .4359]])
        verified = EvidenceVerifier(embedder).verify("结论。", [result("章节：主题\n正文证据。")])
        self.assertEqual(verified.answer, REFUSAL)
        self.assertEqual(verified.unsupported_claims, ["结论。"])
        self.assertEqual(verified.coverage, 0.)

    def test_genuine_blockquote_is_not_discarded(self):
        self.assertEqual(evidence_body("# 定义\n> 平均速度是位移与时间的比值。"), "> 平均速度是位移与时间的比值。")

    def test_formula_lines_and_conditions_are_preserved(self):
        body = "$$\nR = \\rho L / S\n$$\n条件：材料和温度固定。"
        self.assertEqual(evidence_body("章节: 电阻\n" + body), body)

    def test_strongest_source_wins_over_two_earlier_near_ties(self):
        embedder = RecordingEmbedder([[1., 0.], [.98, .199], [.99, .141], [1., 0.]])
        verified = EvidenceVerifier(embedder).verify("结论。", [result("证据一"), result("证据二"), result("证据三")])
        self.assertEqual(verified.evidence[0].source_ids, [3])
        self.assertNotIn("[资料1]", verified.answer)
        self.assertNotIn("[资料2]", verified.answer)

    def test_exact_tie_uses_one_stable_source(self):
        verified = EvidenceVerifier(RecordingEmbedder()).verify("结论。", [result("证据一"), result("证据二")])
        self.assertEqual(verified.evidence[0].source_ids, [1])

    def test_coverage_threshold_still_uses_original_claim_count(self):
        embedder = RecordingEmbedder([[1., 0.], [-1., 0.], [1., 0.]])
        verified = EvidenceVerifier(embedder, min_coverage=.75).verify("第一条。第二条。", [result("证据。")])
        self.assertEqual(verified.coverage, .5)
        self.assertEqual(verified.answer, REFUSAL)
        self.assertEqual(verified.unsupported_claims, ["第二条。"])

    def test_missing_vectors_fail_explicitly_after_filtering(self):
        embedder = RecordingEmbedder([[1., 0.]])
        with self.assertRaisesRegex(RuntimeError, "向量数量"):
            EvidenceVerifier(embedder).verify("结论。", [result("章节：标题"), result("正文。")])

    def test_empty_retrieval_does_not_call_embedder(self):
        embedder = RecordingEmbedder()
        self.assertEqual(EvidenceVerifier(embedder).verify("结论。", []).answer, REFUSAL)
        self.assertIsNone(embedder.texts)

    @staticmethod
    def pipeline(results):
        from app.generator import GeneratedAnswer
        from app.pipeline import RAGPipeline
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.retriever = SimpleNamespace(search=lambda *args, **kwargs: results)
        pipeline.generator = SimpleNamespace(
            model="fake-model", generate=lambda *args: GeneratedAnswer("平均速度是位移与时间的比值。 [资料1]", [], "fake-model"),
        )
        pipeline.evidence_verifier = EvidenceVerifier(RecordingEmbedder())
        return pipeline

    def test_api_returns_original_id_and_body_snippet(self):
        from fastapi.testclient import TestClient
        # Import the API without creating its optional global Gradio demo and
        # associated event loops; this test exercises only the FastAPI route.
        with patch.dict(sys.modules, {"gradio": None}):
            from app.main import create_app
            pipeline = self.pipeline([result("章节：标题", "title"), result("平均速度是位移与时间的比值。", "body")])
            with patch("app.main.ask_question", side_effect=pipeline.ask), \
                    patch("app.service.get_pipeline", side_effect=AssertionError("接口回归不得启动真实模型或索引")), \
                    TestClient(create_app(include_ui=False)) as client:
                response = client.post("/ask", json={"question": "平均速度是什么？"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sources"][0]["citation_id"], 2)
        self.assertEqual(data["sources"][0]["snippet"], "平均速度是位移与时间的比值。")
        self.assertEqual(data["evidence"][0]["source_ids"], [2])
        self.assertEqual(len(data["retrieved"]), 2)
        self.assertFalse(data["refused"])

    def test_pipeline_refuses_metadata_only_results(self):
        pipeline = self.pipeline([result("章节：标题\n> 合成讲义 · 仅供 RAG 系统测试使用")])
        response = pipeline.ask("平均速度是什么？")
        self.assertTrue(response.refused)
        self.assertEqual(response.answer, REFUSAL)
        self.assertEqual(response.sources, [])
        self.assertEqual(response.evidence_coverage, 0.)


if __name__ == "__main__":
    unittest.main()
