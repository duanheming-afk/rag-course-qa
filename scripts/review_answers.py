"""Review generated answers locally without modifying benchmark questions or reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.answer_review import AnswerReviewStore, ERROR_LABELS, SCORE_FIELDS, keyword_groups, read_source_files, risk_flags


def build_demo(store: AnswerReviewStore, raw_root: Path):
    import gradio as gr

    def progress():
        result = store.summary()
        return f"已评分 **{result['reviewed']} / {result['total']}**；评分者 {result['reviewer_count']} 人。未评分不会计为零分。"

    def display(key, current_reviewer):
        row = store.records[key]
        review = store.review(key)
        matched, missing = keyword_groups(row)
        info = (f"样本：{key}；预期行为：{row['expected_behavior']}\n"
                f"自动风险提示（不是判错）：{'；'.join(label for label, _ in risk_flags(row)) or '未检测到'}\n"
                f"关键词代理分：{row.get('keyword_recall', 0):.2%}；语义覆盖代理：{row.get('evidence_coverage', 0):.2%}\n"
                f"已删除结论：{row.get('removed_claim_count', 0)}；引用模式：{row.get('citation_mode', '')}\n"
                f"服务异常：{json.dumps(row.get('error'), ensure_ascii=False)}")
        evidence = "\n\n".join(
            f"[资料{item['citation_id']}] {item['source_file']} · {item.get('section', '')}\n{item['text']}"
            for item in row.get("retrieved", [])
        ) or "本题未检索资料。"
        citations = "\n\n".join(
            f"[资料{item['citation_id']}] {item['source_file']} · {item.get('section', '')}\n{item.get('snippet', '')}"
            for item in row.get("sources", [])
        ) or "没有正文引用；合理拒答或澄清时可不适用。"
        keywords = json.dumps({"matched": matched, "missing": missing}, ensure_ascii=False, indent=2)
        return (info, row["question"], row.get("reference_answer", ""), row.get("answer", ""),
                evidence, citations, row.get("raw_answer", ""), keywords,
                "\n".join(row.get("warnings", [])), read_source_files(row, raw_root),
                *(str(review[field]) if field in review else "未评分" for field in SCORE_FIELDS),
                review.get("error_tags", []), review.get("notes", ""),
                review.get("reviewer", current_reviewer or ""), review.get("revision", 0), key, progress())

    def save(key, correctness, completeness, evidence_support, citation_support, tags, notes, reviewer, revision, loaded_key):
        try:
            if key != loaded_key:
                raise ValueError("题目仍在切换，请等待页面内容加载完成再保存。")
            saved = store.save(key, dict(zip(SCORE_FIELDS, [correctness, completeness, evidence_support, citation_support]))
                               | {"error_tags": tags or [], "notes": notes or "", "reviewer": reviewer or ""},
                               expected_revision=revision)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return "已保存本题人工评分；可继续下一条。需要汇总时点击“导出当前汇总”。", saved["revision"], progress()

    def next_pending(key):
        selected = store.next_pending(key)
        message = "仅切换题目，未保存当前修改。" if selected != key else "没有其他未评分题目。请保存当前题或导出汇总。"
        return selected, message

    def export():
        result = store.export_reports()
        return f"已导出 {result['reviewed']} / {result['total']} 的汇总到：{store.output_dir}。人工评分与自动风险分开保存。"

    choices = [(f"{key} · {store.records[key]['question'][:38]}", key) for key in store.keys]
    with gr.Blocks(title="RAG 模型答案人工评分", analytics_enabled=False) as demo:
        gr.Markdown("# RAG 模型答案人工评分\n这是对**模型生成答案**评分，不是之前对题目和参考答案的审核。"
                    "排序仅优先展示风险样本，不能把自动提示当成真实错误。请先核对问题和原文，再判断模型回答。")
        progress_view = gr.Markdown(progress())
        identifier = gr.Dropdown(choices, value=store.keys[0], label="选择样本（风险优先排序）")
        revision = gr.State(0)
        loaded_key = gr.State(None)
        metadata = gr.Textbox(label="样本状态与自动风险提示", lines=5, interactive=False)
        question = gr.Textbox(label="问题", lines=3, interactive=False)
        with gr.Row():
            reference = gr.Textbox(label="已有参考答案（也需核对，不保证标注正确）", lines=8, interactive=False)
            answer = gr.Textbox(label="待评分：模型最终答案", lines=8, interactive=False)
        with gr.Row():
            evidence = gr.Textbox(label="评测时记录的全部检索片段", lines=12, interactive=False)
            citations = gr.Textbox(label="模型正文实际引用的片段（请逐条核对）", lines=12, interactive=False)
        with gr.Accordion("查看关键词、被筛选前的回答、警告与目标原文", open=False):
            keywords = gr.Textbox(label="命中 / 未命中关键词（仅字面匹配）", lines=4, interactive=False)
            raw_answer = gr.Textbox(label="证据筛选前的模型回答", lines=8, interactive=False)
            warnings = gr.Textbox(label="生成 / 证据筛选警告", lines=3, interactive=False)
            original = gr.Textbox(label="当前 data/raw 原文（可能与历史评测时版本不同）", lines=16, interactive=False)
        gr.Markdown("## 人工评分规则\n"
                    "正确性：0 错误/未完成，1 部分正确，2 完整正确（包括应澄清/拒答时行为正确）。\n\n"
                    "完整性：0 未覆盖要求，1 部分覆盖，2 覆盖问题要求；不要求机械复述整份参考答案。\n\n"
                    "证据支持：0 缺乏依据，1 部分支持，2 关键结论均有原文支持。\n\n"
                    "引用支持：0 错误/缺失，1 部分引用支持，2 每条引用支持相应结论。\n\n"
                    "仅完全正确的拒答/澄清题，证据和引用可选 N/A。有低于 2 的评分，必须填写分类与理由。")
        with gr.Row():
            correctness = gr.Dropdown(["未评分", "0", "1", "2"], value="未评分", label="正确性")
            completeness = gr.Dropdown(["未评分", "0", "1", "2"], value="未评分", label="完整性")
            support = gr.Dropdown(["未评分", "0", "1", "2", "N/A"], value="未评分", label="证据支持")
            citation = gr.Dropdown(["未评分", "0", "1", "2", "N/A"], value="未评分", label="引用支持")
        tags = gr.Dropdown([(label, key) for key, label in ERROR_LABELS.items()], multiselect=True, label="问题分类（可多选，含标注问题）")
        notes = gr.Textbox(label="评分理由 / 失败原因 / 建议", lines=3)
        reviewer = gr.Textbox(label="评分者（必填）")
        with gr.Row():
            save_button = gr.Button("保存本题评分", variant="primary")
            next_button = gr.Button("下一条未评分（不保存当前修改）")
            export_button = gr.Button("导出当前汇总")
        message = gr.Textbox(label="操作结果", interactive=False)
        outputs = [metadata, question, reference, answer, evidence, citations, raw_answer,
                   keywords, warnings, original, correctness, completeness, support, citation,
                   tags, notes, reviewer, revision, loaded_key, progress_view]
        identifier.change(display, [identifier, reviewer], outputs)
        demo.load(lambda: display(store.keys[0], ""), outputs=outputs)
        save_button.click(save, [identifier, correctness, completeness, support, citation, tags, notes, reviewer, revision, loaded_key],
                          [message, revision, progress_view])
        next_button.click(next_pending, identifier, [identifier, message])
        export_button.click(export, outputs=message)
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="本地模型答案人工评分；不自动填人工分数")
    parser.add_argument("--report", help="answers.json 路径；默认选择 holdout_runs 中最新一份")
    parser.add_argument("--output-dir", help="独立的本地评分目录")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--export-only", action="store_true", help="只生成风险清单和现有人工汇总，不启动网页")
    args = parser.parse_args()
    candidates = sorted((PROJECT_ROOT / "reports/holdout_runs").glob("*/answers.json"))
    if not args.report and not candidates:
        parser.error("没有找到 answers.json，请先运行答案评测。")
    report_path = PROJECT_ROOT / args.report if args.report else candidates[-1]
    output = PROJECT_ROOT / args.output_dir if args.output_dir else PROJECT_ROOT / "reports/manual_review_local" / report_path.parent.name
    store = AnswerReviewStore(report_path, output)
    result = store.export_reports()
    print(f"人工已评分 {result['reviewed']} / {result['total']}；输出目录：{output}", flush=True)
    if not args.export_only:
        build_demo(store, PROJECT_ROOT / "data/raw").launch(
            server_name="127.0.0.1", server_port=args.port, inbrowser=False, share=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
