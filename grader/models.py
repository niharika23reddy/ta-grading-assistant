"""Core data structures shared across the app."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

STATUSES = ["Met", "Partially met", "Missing", "Unable to verify"]
CORRECTNESS = ["Correct", "Partially correct", "Incorrect", "Not assessed"]

# Roles keep instructor material and student work apart in the vector store.
ROLE_STUDENT = "student"
ROLE_REFERENCE = "reference"
ROLE_INSTRUCTIONS = "instructions"


@dataclass(frozen=True)
class Scope:
    """Identifies exactly one submission. Every student query is filtered by all four."""

    course: str
    assignment: str
    student: str
    version: str

    def as_filter(self) -> dict:
        return {
            "$and": [
                {"role": ROLE_STUDENT},
                {"course": self.course},
                {"assignment": self.assignment},
                {"student": self.student},
                {"version": self.version},
            ]
        }


@dataclass
class PageText:
    """Text from one page (PDF), whole file (DOCX/text/code) or image."""

    file: str
    page: Optional[int]  # 1-based page number, None when the format has no pages
    text: str
    method: str  # "text", "ocr", "none"
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExtractionReport:
    file: str
    file_type: str
    pages: list[PageText]
    readable: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.pages if p.text)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    file: str
    page: Optional[int]
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    ocr: bool = False  # text came from OCR and may contain recognition errors

    def location(self) -> str:
        where = f"{self.file}"
        if self.page is not None:
            where += f", page {self.page}"
        where += f", lines {self.line_start}-{self.line_end}"
        if self.ocr:
            where += " (OCR)"
        return where


@dataclass
class Requirement:
    req_id: str
    description: str
    criterion_id: str = ""  # rubric criterion this requirement counts toward
    check_correctness: bool = True
    needs_execution: bool = False  # cannot be verified without running code
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Criterion:
    criterion_id: str
    description: str
    max_points: float
    # Points per performance level, only as given in the official rubric.
    # Keys: "full", "partial", "none". Missing "partial" means no partial-credit rule.
    levels: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Evidence:
    quote: str
    location: str
    chunk_id: str
    verified: bool  # quote found verbatim (whitespace-insensitive) in the cited chunk


@dataclass
class Decision:
    req_id: str
    description: str
    status: str
    correctness: str
    evidence: list[Evidence]
    reasoning: str
    improvement: str
    ta_review: bool
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d
