"""Split extracted text into line-aligned chunks that keep their source location."""
from __future__ import annotations

import hashlib
import re

from .models import Chunk, ExtractionReport


def chunk_reports(reports: list[ExtractionReport], prefix: str, chunk_chars: int = 900,
                  overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for report in reports:
        for page in report.pages:
            if page.text.strip():
                chunks.extend(_chunk_page(page.text, page.file or report.file, page.page, prefix, chunk_chars, overlap,
                                          page.method == "ocr"))
    return chunks


SHOT_OPEN_RE = re.compile(r"^\[Screenshot (\d+): OCR text of the image begins\]$")
SHOT_CLOSE_RE = re.compile(r"^\[Screenshot (\d+): OCR text of the image ends\]$")
SHOT_ANY_RE = re.compile(r"^\[Screenshot \d+:")


def _screenshot_lines(lines: list[str]) -> list[bool]:
    """Which lines hold a screenshot marker or OCR text from a screenshot inlined by extraction."""
    flags, inside = [], False
    for line in lines:
        s = line.strip()
        if SHOT_OPEN_RE.match(s):
            inside = True
        flags.append(inside or bool(SHOT_ANY_RE.match(s)))
        if SHOT_CLOSE_RE.match(s):
            inside = False
    return flags


def _chunk_page(text: str, file: str, page, prefix: str, size: int, overlap: int,
                ocr: bool = False) -> list[Chunk]:
    lines = text.splitlines() or [text]
    shot = _screenshot_lines(lines)
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    out: list[Chunk] = []
    start = 0
    while start < len(lines):
        end, length = start, 0
        while end < len(lines) and (length == 0 or length + len(lines[end]) + 1 <= size):
            length += len(lines[end]) + 1
            end += 1
        body = "\n".join(lines[start:end])
        if body.strip():
            key = f"{prefix}|{file}|{page}|{start}"
            out.append(Chunk(
                chunk_id=hashlib.sha1(key.encode()).hexdigest()[:16],
                text=body, file=file, page=page,
                line_start=start + 1, line_end=end,
                char_start=offsets[start], char_end=offsets[start] + len(body),
                ocr=ocr or any(shot[start:end]),
            ))
        if end >= len(lines):
            break
        # step back enough lines to give roughly `overlap` characters of context
        back, back_len = end, 0
        while back > start + 1 and back_len < overlap:
            back -= 1
            back_len += len(lines[back]) + 1
        start = back if back > start else end
    return out
