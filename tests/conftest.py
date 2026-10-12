import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grader.checklist import Checklist, draft_requirements, parse_rubric_csv  # noqa: E402
from grader.config import get_settings  # noqa: E402
from grader.embeddings import HashEmbedder  # noqa: E402
from grader.store import ChunkStore  # noqa: E402

SAMPLES = ROOT / "samples"
# Which rubric criterion each drafted sample requirement belongs to (set by the TA in the app).
SAMPLE_LINKS = {"R1": "C1", "R2": "C2", "R3": "C3", "R4": "C4", "R5": "C5", "R6": "C6", "R7": "C6"}


@pytest.fixture
def settings():
    return get_settings(llm_provider="mock", embedding_provider="hash", ocr_enabled=False)


@pytest.fixture
def store(tmp_path):
    return ChunkStore(HashEmbedder(), path=str(tmp_path / "chroma"))


@pytest.fixture
def sample_checklist():
    reqs = draft_requirements((SAMPLES / "assignment/instructions.md").read_text())
    for r in reqs:
        r.criterion_id = SAMPLE_LINKS[r.req_id]
    crits = parse_rubric_csv((SAMPLES / "assignment/rubric.csv").read_text())
    return Checklist("DATA101", "A1", reqs, crits, approved=True)


def submission_files(name):
    folder = SAMPLES / "submissions" / name
    return [(p.name, p.read_bytes()) for p in sorted(folder.iterdir())]


def make_pdf(pages):
    """pages: list of str (text page) or None (image-only page that looks scanned)."""
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        if text is None:
            img = Image.new("RGB", (400, 200), "white")
            c.drawImage(ImageReader(img), 50, 500, 400, 200)
        else:
            y = 800
            for line in text.splitlines():
                c.drawString(50, y, line)
                y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()
