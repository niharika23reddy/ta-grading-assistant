"""Language-model back-ends. The default runs locally through Ollama.

Student work is only ever sent to the provider configured here. The shipped
providers are local ("ollama") or offline ("mock"); there is deliberately no
cloud provider, because this deployment must keep student work on the machine.
"""
from __future__ import annotations

import json
import re

import requests

from .models import Chunk, Requirement

STOP = set("""a an the and or of to in on for with by at from as is are be this that these those
it its your you their they student students must should will each all any into using use used
include includes including provide write make show give least one two three more than per""".split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


def key_terms(text: str) -> list[str]:
    seen, out = set(), []
    for w in WORD_RE.findall(text.lower()):
        if w not in STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return out


class OllamaProvider:
    def __init__(self, url: str, model: str, timeout: int = 300):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def judge(self, system: str, user: str, requirement: Requirement, chunks: list[Chunk]) -> dict:
        resp = requests.post(f"{self.url}/api/chat", timeout=self.timeout, json={
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])

    def health(self) -> tuple[bool, str]:
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=5).json()
        except Exception as exc:
            return False, f"Ollama is not reachable at {self.url} ({exc}). Start the Ollama app."
        names = [m["name"] for m in tags.get("models", [])]
        if not any(n == self.model or n.startswith(self.model + ":") for n in names):
            return False, f"Model '{self.model}' is not downloaded. Run: ollama pull {self.model}"
        return True, f"Ollama is running with {self.model}."


class MockProvider:
    """Keyword-overlap heuristic for offline testing. It makes no real judgment
    and never assesses correctness; do not use it to grade real work."""

    name = "mock"

    def judge(self, system: str, user: str, requirement: Requirement, chunks: list[Chunk]) -> dict:
        terms = key_terms(requirement.description)
        best, best_score, best_line = None, 0.0, ""
        for i, ch in enumerate(chunks, start=1):
            low = ch.text.lower()
            score = sum(1 for t in terms if t in low) / max(len(terms), 1)
            if score > best_score:
                line = max(ch.text.splitlines() or [""],
                           key=lambda ln: sum(1 for t in terms if t in ln.lower()))
                best, best_score, best_line = i, score, line.strip()
        if best is None or best_score < 0.3:
            return {"status": "Missing", "correctness": "Not assessed", "evidence": [],
                    "reasoning": "No passage in the submission mentions the key terms of this requirement.",
                    "improvement": "Add the missing work described by this requirement.",
                    "needs_ta_review": True, "review_reason": "Mock provider; confirm manually."}
        status = "Met" if best_score >= 0.6 else "Partially met"
        return {"status": status, "correctness": "Not assessed",
                "evidence": [{"chunk": f"C{best}", "quote": best_line}],
                "reasoning": f"{round(best_score * 100)}% of the requirement's key terms appear in the cited passage.",
                "improvement": "" if status == "Met" else "Address every part of this requirement explicitly.",
                "needs_ta_review": True, "review_reason": "Mock provider; confirm manually."}

    def health(self) -> tuple[bool, str]:
        return True, "Mock provider (keyword heuristic, for testing only)."


def get_provider(settings):
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings.ollama_url, settings.ollama_model)
    if settings.llm_provider == "mock":
        return MockProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")
