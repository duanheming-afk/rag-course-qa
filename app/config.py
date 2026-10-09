"""Configuration shared by API, UI and scripts."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def normalize_ollama_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        raise ValueError("Ollama 地址不能为空")
    if "://" not in value:
        value = "http://" + value
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Ollama 地址必须是 http(s)://主机:端口")
    if parts.username or parts.password or parts.query or parts.fragment or parts.path:
        raise ValueError("Ollama 地址不应包含凭据、路径或查询参数")
    host = parts.hostname
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = parts.port or (11434 if parts.scheme == "http" else 443)
    if ":" in host:
        host = f"[{host}]"
    return urlunsplit((parts.scheme, f"{host}:{port}", "", "", ""))


@dataclass(frozen=True)
class Settings:
    ollama_url: str
    llm_model: str
    max_tokens: int
    timeout: float
    index_path: Path
    chunks_path: Path
    collection: str
    embedding_model: str
    evidence_min_score: float
    evidence_min_coverage: float

    @classmethod
    def from_env(cls) -> "Settings":
        tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))
        min_score = float(os.getenv("EVIDENCE_MIN_SCORE", "0.52"))
        min_coverage = float(os.getenv("EVIDENCE_MIN_COVERAGE", "0.50"))
        if tokens <= 0 or not 0 < timeout <= 600 or not -1 <= min_score <= 1 or not 0 < min_coverage <= 1:
            raise ValueError("配置不合法：请检查 token、超时和证据阈值")
        return cls(
            normalize_ollama_url(os.getenv("RAG_OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "127.0.0.1:11434"),
            os.getenv("LLM_MODEL", "qwen2.5:3b"), tokens, timeout,
            PROJECT_ROOT / os.getenv("QDRANT_PATH", "index/qdrant"),
            PROJECT_ROOT / os.getenv("CHUNKS_PATH", "data/processed/chunks.jsonl"),
            os.getenv("QDRANT_COLLECTION", "course_kb"),
            os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            min_score, min_coverage,
        )
