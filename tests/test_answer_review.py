from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.answer_review import AnswerReviewStore, atomic_json, keyword_groups, normalize_score, read_source_files, risk_flags


def sample(identifier="a1", behavior="answer"):
    return {
        "id": identifier, "expected_behavior": behavior, "question": "什么是过采样？",
        "answer": "通过 SMOTE 扩充少数类样本。 [资料1]", "reference_answer": "过采样",
        "required_keywords": [["SMOTE", "合成少数类"], "2 方法"],
        "gold_source_files": ["lecture.md"], "success": True,
        "retrieved_gold": True, "citation_gold": True, "citation_missing": False,
        "evidence_coverage": 1.0, "keyword_recall": .5,
        "refused": behavior == "refuse", "clarified": behavior == "clarify",
    }


def rating(**changes):
    return {"correctness": "2", "completeness": "2", "evidence_support": "2",
            "citation_support": "2", "reviewer": "test-reviewer", "notes": "",
            "error_tags": [], **changes}


class AnswerReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "original" / "answers.json"
        self.source.parent.mkdir()
        self.payload = {"created_at": "test-run", "details": {"Dense": [sample(), sample("r1", "refuse"), sample("c1", "clarify")]}}
        self.source.write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf-8")
        self.original = self.source.read_bytes()
        self.output = self.root / "review-local"
        self.store = AnswerReviewStore(self.source, self.output)

    def tearDown(self):
        self.temporary.cleanup()

    def test_starting_review_does_not_assign_scores_or_create_review_file(self):
        summary = self.store.summary()
        self.assertEqual(summary["reviewed"], 0)
        self.assertIsNone(summary["schemes"]["Dense"]["fully_correct_rate_on_reviewed"])
        self.assertFalse(self.store.path.exists())

    def test_complete_human_rating_is_saved_separately_and_restored(self):
        saved = self.store.save("Dense/a1", rating())
        self.assertEqual(saved["correctness"], 2)
        self.assertEqual(saved["origin"], "human")
        self.assertTrue(saved["reviewed_at"])
        self.assertEqual(AnswerReviewStore(self.source, self.output).review("Dense/a1"), saved)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_unscored_records_are_not_counted_as_zero(self):
        self.store.save("Dense/a1", rating())
        result = self.store.summary()["schemes"]["Dense"]
        self.assertEqual(result["reviewed"], 1)
        self.assertEqual(result["coverage"], 1 / 3)
        self.assertEqual(result["fully_correct_rate_on_reviewed"], 1)
        self.assertFalse(result["complete"])

    def test_incomplete_score_form_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.save("Dense/a1", rating(correctness="未评分"))
        self.assertFalse(self.store.path.exists())

    def test_reviewer_is_required(self):
        with self.assertRaisesRegex(ValueError, "评分者"):
            self.store.save("Dense/a1", rating(reviewer=" "))

    def test_low_score_requires_category_and_reason(self):
        for tags, notes in [([], "missing detail"), (["answer_incomplete"], "")]:
            with self.subTest(tags=tags, notes=notes), self.assertRaises(ValueError):
                self.store.save("Dense/a1", rating(completeness="1", error_tags=tags, notes=notes))

    def test_bad_answer_rating_and_multiple_tags_are_counted_once(self):
        self.store.save("Dense/a1", rating(correctness="1", completeness="1",
                        error_tags=["answer_incomplete", "annotation_issue"], notes="核对原文后仍缺少条件"))
        result = self.store.summary()
        self.assertEqual(result["reviewed"], 1)
        self.assertEqual(result["human_issue_counts"]["annotation_issue"], 1)
        self.assertEqual(result["schemes"]["Dense"]["fully_correct_rate_on_reviewed"], 0)

    def test_na_is_allowed_only_for_correct_negative_behavior(self):
        self.store.save("Dense/r1", rating(evidence_support="N/A", citation_support="N/A"))
        result = self.store.summary()["schemes"]["Dense"]
        self.assertEqual(result["evidence_applicable"], 0)
        self.assertIsNone(result["fully_supported_rate_on_applicable"])
        with self.assertRaises(ValueError):
            self.store.save("Dense/a1", rating(citation_support="N/A"))
        with self.assertRaises(ValueError):
            self.store.save("Dense/c1", rating(correctness="1", citation_support="N/A"))

    def test_stale_form_cannot_overwrite_a_saved_review(self):
        self.store.save("Dense/a1", rating())
        with self.assertRaisesRegex(ValueError, "已被修改"):
            self.store.save("Dense/a1", rating(notes="stale"), expected_revision=0)
        self.assertEqual(self.store.review("Dense/a1")["notes"], "")

    def test_update_keeps_previous_review_and_increments_revision(self):
        self.store.save("Dense/a1", rating())
        self.store.save("Dense/a1", rating(notes="补充说明"), expected_revision=1)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        self.assertEqual(data["rows"]["Dense/a1"]["revision"], 2)
        self.assertEqual(data["history"][0]["previous_review"]["revision"], 1)

    def test_two_store_instances_preserve_other_records_when_saved_sequentially(self):
        second = AnswerReviewStore(self.source, self.output)
        self.store.save("Dense/a1", rating())
        second.save("Dense/r1", rating())
        self.assertEqual(self.store.summary()["reviewed"], 2)

    def test_changed_source_report_invalidates_reviews(self):
        self.store.save("Dense/a1", rating())
        self.source.write_text(json.dumps({**self.payload, "created_at": "another-run"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "原始报告已变化"):
            self.store.summary()
        with self.assertRaisesRegex(ValueError, "来源哈希不匹配"):
            AnswerReviewStore(self.source, self.output)

    def test_unknown_record_or_category_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.save("Dense/unknown", rating())
        with self.assertRaises(ValueError):
            self.store.save("Dense/a1", rating(error_tags=["unrecognized"]))

    def test_duplicate_record_is_rejected(self):
        report = deepcopy(self.payload)
        report["details"]["Dense"].append(sample())
        self.source.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "重复"):
            AnswerReviewStore(self.source, self.output)

    def test_same_id_in_different_schemes_has_independent_scores(self):
        report = {"created_at": "test-run", "details": {"Dense": [sample()], "Hybrid_RRF": [sample()]}}
        self.source.write_text(json.dumps(report), encoding="utf-8")
        store = AnswerReviewStore(self.source, self.output)
        store.save("Dense/a1", rating())
        self.assertEqual(store.summary()["schemes"]["Hybrid_RRF"]["reviewed"], 0)

    def test_next_pending_skips_reviewed_records(self):
        self.store.save("Dense/a1", rating())
        self.assertNotEqual(self.store.next_pending("Dense/a1"), "Dense/a1")

    def test_diagnostics_do_not_turn_proxy_failure_into_human_error(self):
        matched, missing = keyword_groups(sample())
        self.assertEqual(matched, [["SMOTE", "合成少数类"]])
        self.assertEqual(missing, ["2 方法"])
        self.assertTrue(any("章节标题" in label for label, _ in risk_flags(sample())))
        self.assertEqual(self.store.summary()["human_issue_counts"], {})

    def test_export_without_reviews_explicitly_has_no_accuracy(self):
        self.store.export_reports()
        output = (self.output / "review_summary.md").read_text(encoding="utf-8")
        self.assertIn("尚无人工评分", output)
        self.assertIn("未评分 / 不适用", output)
        self.assertFalse(self.store.path.exists())
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_partial_review_export_warns_about_subset_and_has_behavior_denominators(self):
        self.store.save("Dense/a1", rating())
        self.store.export_reports()
        output = (self.output / "review_summary.md").read_text(encoding="utf-8")
        self.assertIn("不得将已评分子集比例", output)
        self.assertIn("answer | 1 / 1", output)
        self.assertIn("refuse | 0 / 1", output)

    def test_corrupt_review_does_not_get_overwritten(self):
        self.output.mkdir()
        self.store.path.write_text("broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.save("Dense/a1", rating())
        self.assertEqual(self.store.path.read_text(encoding="utf-8"), "broken")

    def test_atomic_replace_failure_preserves_previous_file(self):
        target = self.root / "data.json"
        target.write_text("original", encoding="utf-8")
        with patch.object(Path, "replace", side_effect=OSError("failure")), self.assertRaises(OSError):
            atomic_json(target, {"new": True})
        self.assertEqual(target.read_text(encoding="utf-8"), "original")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_original_report_directory_is_not_a_review_output(self):
        with self.assertRaises(ValueError):
            AnswerReviewStore(self.source, self.source.parent)

    def test_source_view_cannot_read_outside_raw_root(self):
        raw = self.root / "raw"
        raw.mkdir()
        (self.root / "secret.md").write_text("secret", encoding="utf-8")
        text = read_source_files({"gold_source_files": ["../secret.md"]}, raw)
        self.assertIn("未读取", text)
        self.assertNotIn("secret\n", text)

    def test_source_view_reads_markdown_and_reports_missing_files(self):
        raw = self.root / "raw"
        raw.mkdir()
        (raw / "lecture.md").write_text("SMOTE 原文", encoding="utf-8")
        text = read_source_files({"gold_source_files": ["lecture.md", "missing.md"]}, raw)
        self.assertIn("SMOTE 原文", text)
        self.assertIn("找不到", text)

    def test_boolean_float_and_out_of_range_scores_are_rejected(self):
        for value in [True, False, 2.0, "3", None, "未评分"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_score(value)


if __name__ == "__main__":
    unittest.main()
