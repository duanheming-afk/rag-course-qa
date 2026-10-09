"""Shared pipeline lifecycle; requests borrow it until inference has finished."""
from __future__ import annotations
import json
from threading import Condition, RLock
from .config import PROJECT_ROOT, Settings
from .errors import RAGError
from .guardrails import clarification_message, immediate_refusal_reason, refusal_message
from .generator import OllamaGenerator
from .pipeline import RAGPipeline, RAGResponse

_pipeline: RAGPipeline | None = None
_condition = Condition(RLock())
_active_requests = 0
_closing = False


def get_pipeline() -> RAGPipeline:
    global _pipeline
    with _condition:
        if _closing:
            raise RAGError("shutting_down", "服务正在关闭，请稍后重试")
        if _pipeline is None:
            _pipeline = RAGPipeline()
        return _pipeline


def ask_question(question: str, top_k: int = 8, course: str | None = None) -> RAGResponse:
    global _active_requests
    question = question.strip()
    if not question or len(question) > 2000 or not 1 <= top_k <= 8:
        raise ValueError("问题长度必须为 1—2000 字，top_k 必须在 1—8 之间")
    course = course.strip() if course else None
    if course and course not in get_courses():
        raise ValueError("所选课程不存在，请从课程列表中选择")
    # Avoid loading the embedding model/Qdrant for requests that have a safe,
    # deterministic response before retrieval is relevant.
    immediate_refusal = immediate_refusal_reason(question)
    if immediate_refusal:
        return RAGResponse(
            question, refusal_message(immediate_refusal), [], [], Settings.from_env().llm_model,
            [immediate_refusal], True, "not_required",
        )
    clarification = clarification_message(question)
    if clarification:
        return RAGResponse(question, clarification, [], [], Settings.from_env().llm_model, citation_mode="not_required")
    with _condition:
        pipeline = get_pipeline()
        _active_requests += 1
    try:
        return pipeline.ask(question, top_k, course)
    finally:
        with _condition:
            _active_requests -= 1
            _condition.notify_all()


def close_pipeline() -> None:
    global _pipeline, _closing
    with _condition:
        _closing = True
        try:
            while _active_requests:
                _condition.wait()
            if _pipeline is not None:
                _pipeline.close()
                _pipeline = None
        finally:
            _closing = False
            _condition.notify_all()


def get_index_status() -> dict:
    settings = Settings.from_env()
    # Use the same lock as lazy initialization, including before a pipeline exists.
    with _condition:
        if _pipeline is not None:
            try:
                return _pipeline.retriever.index_status()
            except Exception as exc:
                return {"status": "degraded", "index_ready": False, "points": 0, "detail": str(exc)}
        if not settings.index_path.is_dir():
            return {"status": "degraded", "index_ready": False, "points": 0, "detail": "索引目录不存在"}
        client = None
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(path=str(settings.index_path))
            count = int(client.get_collection(settings.collection).points_count or 0)
            return {"status": "ok" if count else "degraded", "index_ready": count > 0, "points": count}
        except Exception as exc:
            return {"status": "degraded", "index_ready": False, "points": 0, "detail": str(exc)}
        finally:
            if client is not None:
                client.close()


def get_health() -> dict:
    index = get_index_status()
    generator = OllamaGenerator()
    try:
        llm = generator.check_available()
    except RAGError as exc:
        llm = {"available": False, "model": generator.model, "host": generator.host,
               "code": exc.code, "detail": str(exc)}
    ready = index["index_ready"] and llm["available"]
    return {"status": "ok" if ready else "degraded", "ready": ready,
            "index_ready": index["index_ready"], "points": index["points"], "index": index, "llm": llm,
            "note": "检查索引和模型可用性；实际回答质量需另行评测"}


def get_courses() -> list[str]:
    chunks_path = Settings.from_env().chunks_path
    if not chunks_path.is_file():
        return []
    return sorted({str(row["course"]) for line in chunks_path.read_text(encoding="utf-8").splitlines()
                   if line.strip() for row in [json.loads(line)] if row.get("course")})
