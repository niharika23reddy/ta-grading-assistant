"""Settings read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    ollama_url: str
    ollama_model: str
    embedding_provider: str
    ollama_embed_model: str
    data_dir: Path
    ocr_enabled: bool
    top_k: int
    chunk_chars: int
    chunk_overlap: int
    unverified_as_missing: bool = True  # score "Unable to verify" like "Missing" (provisional)
    llm_parallel: int = 2  # requirements judged at the same time
    ollama_num_ctx: int = 8192
    whole_submission_chars: int = 12000  # send submissions up to this size whole (0 = always search)
    second_look: bool = True  # re-check anything not "Met" against the most related passages


def get_settings(**overrides) -> Settings:
    """Read settings from the environment at call time; keyword arguments override."""
    values = dict(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        data_dir=Path(os.getenv("DATA_DIR", "./data")),
        ocr_enabled=_bool("OCR_ENABLED", True),
        top_k=int(os.getenv("TOP_K", "6")),
        chunk_chars=int(os.getenv("CHUNK_CHARS", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        unverified_as_missing=_bool("UNVERIFIED_AS_MISSING", True),
        llm_parallel=max(1, int(os.getenv("LLM_PARALLEL", "2"))),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        whole_submission_chars=int(os.getenv("WHOLE_SUBMISSION_CHARS", "12000")),
        second_look=_bool("SECOND_LOOK", True),
    )
    values.update(overrides)
    return Settings(**values)
