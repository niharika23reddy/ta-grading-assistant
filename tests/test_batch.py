"""Grading many students in one run."""
import io
import zipfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

from grader import batch
from grader.config import get_settings
from grader.embeddings import HashEmbedder
from grader.llm import get_provider
from grader.models import Scope
from grader.store import ChunkStore
from tests.conftest import submission_files

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


def test_one_file_per_student():
    groups, warns = batch.group_by_student([("S001.pdf", b"a"), ("S002.docx", b"b"), ("S002.py", b"c")],
                                           batch.BY_FILE)
    assert list(groups) == ["S001", "S002"]
    assert [n for n, _ in groups["S002"]] == ["S002.docx", "S002.py"]
    assert not warns


def test_student_code_prefix():
    groups, warns = batch.group_by_student(
        [("S001_report.pdf", b"a"), ("S001_code.py", b"b"), ("S002_report.pdf", b"c"), ("notes.txt", b"d")],
        batch.BY_PREFIX)
    assert {k: len(v) for k, v in groups.items()} == {"S001": 2, "S002": 1}
    assert any("notes.txt" in w for w in warns)


def test_class_zip_with_a_folder_per_student():
    data = _zip([("class/S001/report.pdf", b"a"), ("class/S001/sub/code.py", b"b"), ("class/S002/r.docx", b"c"),
                 ("__MACOSX/class/S001/._report.pdf", b"x"), ("class/.DS_Store", b"x")])
    groups, warns = batch.group_by_student([("all.zip", data)], batch.BY_FOLDER)
    assert {k: [n for n, _ in v] for k, v in groups.items()} == {"S001": ["report.pdf", "sub/code.py"],
                                                                 "S002": ["r.docx"]}
    assert not warns
    # Without a wrapping folder, and with a stray file at the top.
    groups, warns = batch.group_by_student([("all.zip", _zip([("S003/a.pdf", b"a"), ("readme.txt", b"r")]))],
                                           batch.BY_FOLDER)
    assert list(groups) == ["S003"]
    assert any("readme.txt" in w for w in warns)


def test_each_student_is_graded_and_saved_separately(tmp_path, settings, sample_checklist):
    from dataclasses import replace
    settings = replace(settings, data_dir=tmp_path)
    store = ChunkStore(HashEmbedder(), path=str(tmp_path / "chroma"))
    for code in ("S001", "S002"):
        scope = Scope("DATA101", "A1", code, "v1")
        out = batch.grade_student(store, get_provider(settings), sample_checklist, scope,
                                  submission_files(f"{code}_v1"), settings)
        assert len(out["decisions"]) == len(sample_checklist.requirements)
        saved = batch.load_review(batch.review_path(tmp_path, scope))
        assert saved["student"] == code and saved["unread_content"] is False
    assert batch.reviewed_students(tmp_path, "DATA101", "A1", "v1") == ["S001", "S002"]
    assert batch.reviewed_students(tmp_path, "DATA101", "A1", "v2") == []


def test_app_grades_many_students_and_switches_between_them(tmp_path, monkeypatch, sample_checklist):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("OCR_ENABLED", "false")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "checklists").mkdir()
    (tmp_path / "checklists" / "DATA101__A1.json").write_text(sample_checklist.to_json())
    class_zip = _zip([(f"{code}/{name}", data) for code in ("S001", "S002")
                      for name, data in submission_files(f"{code}_v1")])

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.text_input[0].set_value("DATA101")
    at.sidebar.text_input[1].set_value("A1")
    at.run()
    # AppTest has no file uploader, so feed the class ZIP to the grouping step the tab uses.
    groups, _ = batch.group_by_student([("class.zip", class_zip)], batch.BY_FOLDER)
    settings = get_settings()
    store = ChunkStore(HashEmbedder(), path=str(tmp_path / "chroma"))
    for code, files in groups.items():
        batch.grade_student(store, get_provider(settings), sample_checklist, Scope("DATA101", "A1", code, "v1"),
                            files, settings)
    at.run()
    assert not at.exception
    picker = next(s for s in at.sidebar.selectbox if s.label.startswith("Or pick"))
    assert picker.options == ["S001", "S002"]
    summary = at.dataframe[-1].value
    assert list(summary["student"]) == ["S001", "S002"]

    picker.set_value("S002")
    at.run()
    assert not at.exception
    assert at.sidebar.text_input[2].value == "S002"
    s2 = at.session_state["decisions"]
    assert s2 and at.session_state["decisions_scope"].student == "S002"

    # Switching student never shows the previous student's results.
    at.sidebar.text_input[2].set_value("S999")
    at.run()
    assert not at.exception
    assert at.session_state["decisions"] is None
