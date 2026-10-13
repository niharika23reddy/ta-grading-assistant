"""Grading many students in one run: split the uploaded files by student, then review each one separately.

Every student still gets their own scope (course, assignment, student, version), their own index and their
own saved review, exactly as if the TA had done them one at a time in the app.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

from .evaluate import evaluate_submission
from .extract import has_unread_content, unreadable_parts
from .models import Decision, Evidence, Scope
from .pipeline import ingest_submission
from .static_code import analyze, summary_text

BY_FILE = "One file (or ZIP) per student, named with the student code"
BY_PREFIX = "File names start with the student code and an underscore (S001_report.pdf)"
BY_FOLDER = "One ZIP with a folder per student"
MODES = [BY_FILE, BY_PREFIX, BY_FOLDER]

# A class ZIP holds many students, so it may be bigger than one student's ZIP.
MAX_CLASS_ZIP_MEMBERS = 5000
MAX_CLASS_ZIP_BYTES = 1024 * 1024 * 1024


def slug(*parts: str) -> str:
    return "__".join(re.sub(r"[^A-Za-z0-9_-]", "_", p) for p in parts)


def student_code(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text.strip()).strip("._") or "unnamed"


def _skip(path: PurePosixPath) -> bool:
    return (path.is_absolute() or ".." in path.parts or "__MACOSX" in path.parts
            or any(p.startswith(".") for p in path.parts))


def _split_class_zip(name: str, data: bytes, groups: dict, warnings: list[str]) -> None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        warnings.append(f"{name}: the ZIP file is corrupt and was skipped.")
        return
    members = [m for m in zf.infolist() if not m.is_dir() and not _skip(PurePosixPath(m.filename))]
    if len(members) > MAX_CLASS_ZIP_MEMBERS or sum(m.file_size for m in members) > MAX_CLASS_ZIP_BYTES:
        warnings.append(f"{name}: the ZIP file is too large to unpack safely and was skipped.")
        return
    paths = [PurePosixPath(m.filename).parts for m in members]
    # Zipping a folder adds one wrapping folder around everything: look one level inside it.
    depth = 1 if paths and len({p[0] for p in paths}) == 1 and all(len(p) > 2 for p in paths) else 0
    for m, parts in zip(members, paths):
        if len(parts) <= depth + 1:
            warnings.append(f"{name}: {m.filename} is not inside a student folder and was skipped.")
            continue
        groups.setdefault(student_code(parts[depth]), []).append(("/".join(parts[depth + 1:]), zf.read(m)))


def group_by_student(files: list[tuple[str, bytes]], mode: str) -> tuple[dict[str, list[tuple[str, bytes]]], list[str]]:
    """Map each student code to that student's files. Returns (groups, warnings)."""
    groups: dict[str, list[tuple[str, bytes]]] = {}
    warnings: list[str] = []
    for name, data in files:
        base = Path(name).name
        if mode == BY_FOLDER:
            if base.lower().endswith(".zip"):
                _split_class_zip(base, data, groups, warnings)
            else:
                warnings.append(f"{base}: not a ZIP file, so it was skipped. Upload the class ZIP.")
        elif mode == BY_PREFIX:
            if "_" not in base:
                warnings.append(f"{base}: no underscore after a student code, so it was skipped.")
                continue
            groups.setdefault(student_code(base.split("_", 1)[0]), []).append((base, data))
        else:
            groups.setdefault(student_code(Path(base).stem), []).append((base, data))
    return dict(sorted(groups.items())), warnings


def review_path(data_dir: Path, scope: Scope) -> Path:
    return Path(data_dir) / "reviews" / f"{slug(scope.course, scope.assignment, scope.student, scope.version)}.json"


def save_review(path: Path, scope: Scope, decisions: list[Decision], hits: list[str], reports) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "course": scope.course, "assignment": scope.assignment, "student": scope.student, "version": scope.version,
        "decisions": [d.to_dict() for d in decisions], "injection_hits": hits,
        "unread_content": has_unread_content(reports) if reports is not None else True,
        "extraction_warnings": unreadable_parts(reports) if reports is not None else [],
    }, indent=2))


def load_review(path: Path) -> dict:
    saved = json.loads(Path(path).read_text())
    saved["decisions"] = [Decision(**{**d, "evidence": [Evidence(**e) for e in d["evidence"]]})
                          for d in saved["decisions"]]
    saved.setdefault("injection_hits", [])
    saved.setdefault("unread_content", True)  # reviews saved before this was recorded
    saved.setdefault("extraction_warnings", [])
    return saved


def reviewed_students(data_dir: Path, course: str, assignment: str, version: str) -> list[str]:
    """Student codes that already have a saved review for this assignment and version."""
    out = []
    for p in sorted((Path(data_dir) / "reviews").glob(f"{slug(course, assignment)}__*.json")):
        try:
            saved = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if saved.get("course") == course and saved.get("assignment") == assignment \
                and saved.get("version") == version and saved.get("student"):
            out.append(saved["student"])
    return out


def grade_student(store, provider, checklist, scope: Scope, files: list[tuple[str, bytes]], settings,
                  progress=None) -> dict:
    """Index one student's files, evaluate every requirement and save the review."""
    reports, n = ingest_submission(store, scope, files, settings)
    decisions, hits = evaluate_submission(
        checklist.requirements, scope, store, provider, unread_content=has_unread_content(reports),
        static_facts=summary_text(analyze(reports)), top_k=settings.top_k, progress=progress)
    path = review_path(settings.data_dir, scope)
    save_review(path, scope, decisions, hits, reports)
    return {"student": scope.student, "chunks": n, "decisions": decisions, "injection_hits": hits,
            "unread_content": has_unread_content(reports), "extraction_warnings": unreadable_parts(reports)}
