"""课程资料标准化文本块切分。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .loader import SectionDocument


@dataclass
class TextChunk:
    """可直接用于 Embedding 和向量检索的标准化文本块。"""

    chunk_id: str
    text: str
    source_file: str
    source_path: str
    course: str
    title: str
    section: str
    page: int
    chunk_index: int
    char_count: int
    source_type: str = "markdown"


def _split_long_unit(text: str, size: int, overlap: int) -> list[str]:
    """优先按句号等边界切分，无法切分时使用滑动窗口。"""
    if len(text) <= size:
        return [text]

    sentences = [part.strip() for part in re.split(r"(?<=[。！？；!?;])\s*", text) if part.strip()]
    if len(sentences) > 1:
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current}{sentence}" if current else sentence
            if current and len(candidate) > size:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces.append(current)
        if all(len(piece) <= size for piece in pieces):
            return pieces

    step = max(1, size - overlap)
    return [text[start : start + size].strip() for start in range(0, len(text), step) if text[start : start + size].strip()]


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]


def split_documents(
    documents: list[SectionDocument], chunk_size: int = 500, chunk_overlap: int = 80
) -> list[TextChunk]:
    """将章节单元切分为固定大小的文本块，并保留溯源元数据。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")

    chunks: list[TextChunk] = []
    file_counters: dict[str, int] = {}

    for document in documents:
        section_prefix = f"章节：{document.section}\n"
        content_size = max(1, chunk_size - len(section_prefix))
        units: list[str] = []
        for paragraph in _paragraphs(document.text):
            units.extend(_split_long_unit(paragraph, content_size, chunk_overlap))

        current = ""
        section_chunks: list[str] = []
        for unit in units:
            candidate = f"{current}\n\n{unit}" if current else unit
            if current and len(candidate) > content_size:
                section_chunks.append(current.strip())
                tail = current[-chunk_overlap:].strip() if chunk_overlap else ""
                current = f"{tail}\n\n{unit}" if tail and len(tail) + len(unit) + 2 <= content_size else unit
            else:
                current = candidate
        if current.strip():
            section_chunks.append(current.strip())

        start_index = file_counters.get(document.source_file, 0)
        for offset, body in enumerate(section_chunks):
            index = start_index + offset
            text = f"{section_prefix}{body}"
            chunks.append(
                TextChunk(
                    chunk_id=f"{Path(document.source_file).stem}::p{document.page}::c{index:04d}",
                    text=text,
                    source_file=document.source_file,
                    source_path=document.source_path,
                    course=document.course,
                    title=document.title,
                    section=document.section,
                    page=document.page,
                    chunk_index=index,
                    char_count=len(text),
                )
            )
        file_counters[document.source_file] = start_index + len(section_chunks)

    return chunks


def write_jsonl(chunks: list[TextChunk], output_path: str | Path) -> None:
    """将标准化文本块写入 UTF-8 JSONL。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
