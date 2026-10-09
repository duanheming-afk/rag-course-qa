"""FastAPI API with optional Gradio UI."""
from __future__ import annotations
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

# PyCharm's “Run current file” executes this module as a script. Keep that
# shortcut usable while preserving normal package-relative imports elsewhere.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from app.errors import RAGError
    from app.service import ask_question, close_pipeline, get_courses, get_health
else:
    from .errors import RAGError
    from .service import ask_question, close_pipeline, get_courses, get_health

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    course: str | None = Field(default=None, max_length=100)
    top_k: int = Field(default=8, ge=1, le=8)

    @field_validator("question", "course", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value


class RetrievedItem(BaseModel):
    source_file: str
    section: str
    page: int
    score: float
    chunk_id: str
    snippet: str
    citation_id: int


class SourceItem(RetrievedItem):
    pass


class EvidenceItemModel(BaseModel):
    claim_id: int
    claim: str
    source_ids: list[int]
    support_score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    elapsed_ms: int
    sources: list[SourceItem]
    retrieved: list[RetrievedItem]
    warnings: list[str]
    refused: bool
    citation_mode: str
    evidence: list[EvidenceItemModel]
    evidence_coverage: float
    removed_claim_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        await run_in_threadpool(close_pipeline)


def create_app(include_ui: bool = True) -> FastAPI:
    app = FastAPI(title="高校课程资料智能问答系统", version="0.2.0", lifespan=lifespan)

    @app.get("/")
    def root():
        return {"name": "高校课程资料智能问答系统", "docs": "/docs", "ui": "/ui"}

    @app.get("/live")
    def live():
        return {"status": "alive"}

    @app.get("/health")
    def health():
        result = get_health()
        return JSONResponse(result, status_code=200 if result["ready"] else 503)

    @app.get("/courses")
    def courses():
        values = get_courses()
        return {"courses": values, "count": len(values)}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        started = time.perf_counter()
        try:
            response = ask_question(request.question, request.top_k, request.course)
        except ValueError as exc:
            raise HTTPException(400, detail={"code": "invalid_request", "message": str(exc)}) from exc
        except RAGError as exc:
            raise HTTPException(exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc
        except Exception as exc:
            logger.exception("RAG request failed")
            raise HTTPException(503, detail={"code": "service_error", "message": "问答暂不可用，请查看服务日志"}) from exc
        return AskResponse(
            question=response.question, answer=response.answer, model=response.model,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            sources=[SourceItem(**source) for source in response.sources],
            retrieved=[RetrievedItem(
                source_file=item.source_file, section=item.section, page=item.page, score=item.score,
                chunk_id=item.chunk_id, snippet=item.text, citation_id=i,
            ) for i, item in enumerate(response.retrieved, 1)],
            warnings=response.warnings, refused=response.refused, citation_mode=response.citation_mode,
            evidence=[EvidenceItemModel(
                claim_id=item.claim_id, claim=item.claim, source_ids=item.source_ids,
                support_score=item.support_score,
            ) for item in response.evidence],
            evidence_coverage=response.evidence_coverage, removed_claim_count=response.removed_claim_count,
        )

    if include_ui:
        try:
            import gradio as gr
        except ImportError:
            logger.warning("Gradio 未安装，只有 API 可用")
        else:
            if __package__ in {None, ""}:
                from app.web import build_demo
            else:
                from .web import build_demo
            app = gr.mount_gradio_app(app, build_demo(), path="/ui")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
    )
