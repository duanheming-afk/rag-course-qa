"""课程资料读取与清洗。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SectionDocument:
    """一个课程文件中的章节单元。"""

    source_file: str
    source_path: str
    title: str
    section: str
    heading_level: int
    page: int
    text: str
    course: str


COURSE_NAMES = {
    "高数": "高等数学",
    "线代": "线性代数",
    "概率": "概率论与数理统计",
    "物理": "大学物理",
    "计科": "计算机科学",
    "ML": "机器学习",
    "经济": "经济学",
}


def course_from_filename(filename: str) -> str:
    """根据文件名的前缀识别课程名称。"""
    prefix = Path(filename).stem.split("-", 1)[0]
    return COURSE_NAMES.get(prefix, "未分类")


def clean_markdown(text: str) -> str:
    """统一换行、空格和空行，保留公式、列表和代码内容。"""
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def load_markdown(path: str | Path, raw_root: str | Path) -> list[SectionDocument]:
    """读取 Markdown，并按标题切分为章节单元。

    Markdown 没有真实页码，因此 page 固定为 1；真实 PDF 后续可替换为实际页码。
    """
    path = Path(path)
    raw_root = Path(raw_root)
    raw_text = path.read_text(encoding="utf-8-sig")
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    title = path.stem
    section = title
    heading_level = 1
    buffer: list[str] = []
    documents: list[SectionDocument] = []

    def flush() -> None:
        body = clean_markdown("\n".join(buffer))
        if not body:
            return
        documents.append(
            SectionDocument(
                source_file=path.name,
                source_path=path.relative_to(raw_root).as_posix(),
                title=title,
                section=section,
                heading_level=heading_level,
                page=1,
                text=body,
                course=course_from_filename(path.name),
            )
        )

    for line in lines:
        parsed = _heading(line)
        if parsed:
            flush()
            buffer.clear()
            heading_level, section = parsed
            if heading_level == 1 and title == path.stem:
                title = section
            continue
        buffer.append(line)

    flush()
    return documents


def load_directory(raw_root: str | Path) -> list[SectionDocument]:
    """读取目录中的 Markdown 课程资料，跳过目录说明文件。"""
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"原始资料目录不存在: {raw_root}")

    documents: list[SectionDocument] = []
    paths = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".md", ".markdown"}
        and path.name.upper() != "INDEX.MD"
    )
    for path in paths:
        documents.extend(load_markdown(path, raw_root))
    return documents
