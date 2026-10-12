"""Glue between extraction, chunking and the store, shared by the app and tests."""
from __future__ import annotations

from pathlib import Path

from .chunking import chunk_reports
from .extract import extract_file
from .models import Decision, ExtractionReport, Scope


def read_files(files: list[tuple[str, bytes]], ocr: bool) -> list[ExtractionReport]:
    reports = []
    for name, data in files:
        reports.extend(extract_file(Path(name).name, data, ocr))
    return reports


def ingest_submission(store, scope: Scope, files: list[tuple[str, bytes]], settings):
    reports = read_files(files, settings.ocr_enabled)
    prefix = f"{scope.course}|{scope.assignment}|{scope.student}|{scope.version}"
    chunks = chunk_reports(reports, prefix, settings.chunk_chars, settings.chunk_overlap)
    n = store.replace_submission(scope, chunks)
    return reports, n


def ingest_reference(store, course: str, assignment: str, files: list[tuple[str, bytes]], settings):
    reports = read_files(files, settings.ocr_enabled)
    chunks = chunk_reports(reports, f"ref|{course}|{assignment}", settings.chunk_chars, settings.chunk_overlap)
    return reports, store.replace_reference(course, assignment, chunks)


def draft_feedback(decisions: list[Decision]) -> str:
    """Plain draft built from the decisions; the TA edits it before anything is shared."""
    done = [d for d in decisions if d.status == "Met" and d.correctness in ("Correct", "Not assessed")]
    todo = [d for d in decisions if d not in done and d.status != "Unable to verify"]
    lines = []
    if done:
        lines.append("What you did well:")
        lines += [f"- {d.description}" for d in done]
    if todo:
        lines.append("")
        lines.append("What to improve:")
        for d in todo:
            tip = d.improvement or d.reasoning
            lines.append(f"- {d.description}: {tip}" if tip else f"- {d.description}")
    return "\n".join(lines).strip()
