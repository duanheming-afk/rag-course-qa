"""Gradio 网页问答界面。"""

from __future__ import annotations

import gradio as gr

from .service import ask_question, get_courses


def _format_sources(sources: list[dict[str, str | int | float]], citation_mode: str) -> str:
    if not sources:
        return "暂无来源"
    heading = "### 答案引用（模型声明，仍需核对内容）"
    if citation_mode == "retrieval_fallback":
        heading = "### 最相关检索片段（模型未声明引用，需人工核对）"
    elif citation_mode == "sentence_verified":
        heading = "### 句子级证据引用（系统语义匹配，仍需人工核对）"
    lines = [heading]
    for source in sources:
        lines.append(
            f"- [资料{source['citation_id']}] `{source['source_file']}` · {source['section']}\n"
            f"\n{source['snippet']}\n"
        )
    return "\n".join(lines)


def _format_retrieved(items: list) -> str:
    if not items:
        return "暂无检索片段"
    lines = ["### 检索候选（不代表全部被答案引用）"]
    for rank, item in enumerate(items, start=1):
        lines.append(f"[资料{rank}] `{item.source_file}` / {item.section}\n\n{item.text}\n")
    return "\n".join(lines)


def answer_question(question: str, course: str, top_k: int) -> tuple[str, str, str]:
    """网页按钮回调。"""
    if not question or not question.strip():
        return "请输入问题。", "", ""
    try:
        response = ask_question(
            question,
            top_k=int(top_k),
            course=None if not course or course == "全部课程" else course,
        )
    except Exception as exc:
        return f"请求失败：{exc}", "", "请检查 Ollama 服务、模型和 Qdrant 索引。"

    status = (
        f"模型：{response.model} · 已检索 {len(response.retrieved)} 条片段"
        f" · 证据覆盖率 {response.evidence_coverage:.0%}"
    )
    if response.removed_claim_count:
        status += f" · 已移除 {response.removed_claim_count} 条无充分证据表述"
    if response.warnings:
        status += "\n\n" + "\n\n".join(response.warnings)
    return response.answer, _format_sources(response.sources, response.citation_mode), f"{status}\n\n{_format_retrieved(response.retrieved)}"


def build_demo() -> gr.Blocks:
    """创建 Gradio 界面，可挂载到 FastAPI 的 /ui 路径。"""
    courses = ["全部课程", *get_courses()]
    with gr.Blocks(title="高校课程资料智能问答系统", analytics_enabled=False) as demo:
        gr.Markdown(
            "# 高校课程资料智能问答系统\n"
            "基于课程资料回答问题，并展示检索来源。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="请输入课程问题",
                    placeholder="例如：什么是反向传播算法？",
                    lines=3,
                )
                with gr.Row():
                    course = gr.Dropdown(courses, value="全部课程", label="课程筛选")
                    top_k = gr.Slider(1, 8, value=8, step=1, label="检索数量")
                submit = gr.Button("开始提问", variant="primary")
            with gr.Column(scale=5):
                answer = gr.Markdown(label="答案")
                sources = gr.Markdown(label="来源")
        details = gr.Markdown(label="检索详情")
        submit.click(
            answer_question,
            inputs=[question, course, top_k],
            outputs=[answer, sources, details],
        )
        question.submit(
            answer_question,
            inputs=[question, course, top_k],
            outputs=[answer, sources, details],
        )
    return demo
