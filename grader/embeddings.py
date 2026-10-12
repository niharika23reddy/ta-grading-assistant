"""Local embedding back-ends. Nothing is sent off the machine."""
from __future__ import annotations

import hashlib
import math
import re

import requests

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class HashEmbedder:
    """Deterministic bag-of-words hashing. Offline, crude; for tests and fallback."""

    name = "hash-512"

    def __init__(self, dims: int = 512):
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dims
            for tok in TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self.dims] += 1.0 if (h >> 64) & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class OllamaEmbedder:
    """Embeddings from a model served by a local Ollama instance."""

    def __init__(self, url: str, model: str, timeout: int = 120):
        self.url = url.rstrip("/")
        self.model = model
        self.name = f"ollama-{model}"
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts},
                             timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["embeddings"]


def get_embedder(settings):
    if settings.embedding_provider == "ollama":
        return OllamaEmbedder(settings.ollama_url, settings.ollama_embed_model)
    if settings.embedding_provider == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'")
