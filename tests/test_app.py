"""Smoke test of the Streamlit app with the offline mock model."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from grader.embeddings import HashEmbedder
from grader.models import Scope
from grader.pipeline import ingest_submission
from grader.store import ChunkStore
from tests.conftest import submission_files

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_review_end_to_end(tmp_path, monkeypatch, settings, sample_checklist):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("OCR_ENABLED", "false")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "checklists").mkdir()
    (tmp_path / "checklists" / "DATA101__A1.json").write_text(sample_checklist.to_json())
    scope = Scope("DATA101", "A1", "S001", "v1")
    ingest_submission(ChunkStore(HashEmbedder(), path=str(tmp_path / "chroma")), scope,
                      submission_files("S001_v1"), settings)

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    at.sidebar.text_input[0].set_value("DATA101")
    at.sidebar.text_input[1].set_value("A1")
    at.run()
    assert not at.exception
    next(b for b in at.button if b.label == "Run evaluation").click()
    at.run()
    assert not at.exception
    decisions = at.session_state["decisions"]
    assert len(decisions) == 7
    assert any("Suggested total" in m.value for m in at.markdown)
    assert (tmp_path / "reviews" / "DATA101__A1__S001__v1.json").exists()
    assert any(c.value.startswith("This evaluation took") for c in at.caption)

    # A TA edit belongs to that run: running the evaluation again starts the fields from the new suggestions.
    first = next(s for s in at.selectbox if s.label == "Your status")
    first.set_value("Unable to verify" if first.value != "Unable to verify" else "Met")
    at.run()
    next(b for b in at.button if b.label == "Run evaluation").click()
    at.run()
    assert not at.exception
    shown = [s.value for s in at.selectbox if s.label == "Your status"]
    assert shown == [d.status for d in at.session_state["decisions"]]
