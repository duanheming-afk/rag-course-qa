"""Local Gradio tool for reviewing holdout candidates before benchmark export."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class HoldoutReviewStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows = self._load()

    def _load(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

    def _save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in self.rows) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def ids(self) -> list[str]:
        return [row["id"] for row in self.rows]

    def get(self, identifier: str) -> dict:
        return next(row for row in self.rows if row["id"] == identifier)

    def save(self, identifier: str, status: str, reviewer: str, reference_answer: str,
             keywords: str, source_files: str, notes: str) -> None:
        row = self.get(identifier)
        annotation = row["annotation"]
        annotation.update({
            "status": status,
            "reviewer": reviewer.strip(),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "reference_answer": reference_answer.strip(),
            "required_keywords": [item.strip() for item in keywords.split(",") if item.strip()],
            "gold_source_files": [item.strip() for item in source_files.split(",") if item.strip()],
            "notes": notes.strip(),
        })
        self._save()


def build_demo(store: HoldoutReviewStore):
    import gradio as gr

    def display(identifier: str):
        row = store.get(identifier)
        annotation = row["annotation"]
        evidence = "\n\n".join(
            f"### {item['source_file']} · {item['section']}\n\n{item['text']}"
            for item in row["candidate_evidence"]
        ) or "此题不应依赖课程资料，应评价澄清或拒答行为。"
        metadata = (
            f"ID：`{row['id']}`  \\n"
            f"预期行为：`{row['expected_behavior']}`  \\n"
            f"类别：`{row['category']}`  \\n"
            f"课程：`{row['course'] or '不限'}`  \\n"
            f"当前状态：`{annotation['status']}`"
        )
        return (
            metadata, row["question"], evidence, row["reference_answer_draft"],
            annotation["reference_answer"], ", ".join(annotation["required_keywords"]),
            ", ".join(annotation["gold_source_files"]), annotation["reviewer"], annotation["notes"],
            annotation["status"],
        )

    def save(identifier, status, reviewer, reference_answer, keywords, source_files, notes):
        store.save(identifier, status, reviewer, reference_answer, keywords, source_files, notes)
        return "已保存。完成 100 条审核通过样本后运行 export_approved_holdout.py。"

    with gr.Blocks(title="RAG 独立评测集人工复核") as demo:
        gr.Markdown("# 独立评测集人工复核\n候选题由脚本生成，必须由人工核对资料后才能作为正式基准。")
        identifier = gr.Dropdown(store.ids(), value=store.ids()[0], label="候选题 ID")
        metadata = gr.Markdown()
        question = gr.Textbox(label="问题", interactive=False)
        evidence = gr.Markdown(label="候选证据与原文")
        draft = gr.Textbox(label="AI 起草的参考答案（仅供参考，不可直接批准）", lines=6, interactive=False)
        reference = gr.Textbox(label="人工确认的参考答案", lines=6)
        with gr.Row():
            keywords = gr.Textbox(label="关键词（逗号分隔）")
            source_files = gr.Textbox(label="目标文件（逗号分隔）")
        with gr.Row():
            reviewer = gr.Textbox(label="标注者")
            status = gr.Dropdown(["pending_human_review", "approved", "needs_rewrite", "rejected"], label="审核状态")
        notes = gr.Textbox(label="复核说明", lines=3)
        save_button = gr.Button("保存审核结果", variant="primary")
        message = gr.Markdown()
        outputs = [metadata, question, evidence, draft, reference, keywords, source_files, reviewer, notes, status]
        identifier.change(display, identifier, outputs)
        save_button.click(save, [identifier, status, reviewer, reference, keywords, source_files, notes], message)
        demo.load(lambda: display(store.ids()[0]), outputs=outputs)
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="启动独立评测集人工复核界面")
    parser.add_argument("--queue", default="data/eval/holdout_review_queue.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    store = HoldoutReviewStore(PROJECT_ROOT / args.queue)
    build_demo(store).launch(server_name=args.host, server_port=args.port, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
