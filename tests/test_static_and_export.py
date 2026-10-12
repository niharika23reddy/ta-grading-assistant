import csv
import io
import json

from grader import export
from grader.evaluate import evaluate_submission
from grader.extract import extract_file
from grader.llm import MockProvider
from grader.models import Scope
from grader.pipeline import draft_feedback, ingest_submission
from grader.scoring import score
from grader.static_code import analyze
from tests.conftest import submission_files


def test_static_analysis_never_executes(tmp_path):
    marker = tmp_path / "pwned"
    code = f"open({str(marker)!r}, 'w').write('x')\ndef clean_data(df):\n    return df\n"
    facts = analyze(extract_file("a.py", code.encode()))
    assert facts[0]["syntax_ok"] and facts[0]["functions"] == ["clean_data"]
    assert not marker.exists()


def test_syntax_error_reported():
    facts = analyze(extract_file("a.py", b"print(1\n"))
    assert facts[0]["syntax_ok"] is False


def test_exports_keep_suggested_and_approved_separate(store, settings, sample_checklist):
    scope = Scope("DATA101", "A1", "S001", "v1")
    ingest_submission(store, scope, submission_files("S001_v1"), settings)
    decisions, hits = evaluate_submission(sample_checklist.requirements, scope, store, MockProvider(),
                                          unread_content=False)
    suggested = score(sample_checklist.criteria, sample_checklist.requirements, decisions)
    ta_rows = [{"req_id": d.req_id, "status": "Met", "correctness": "Correct", "comment": ""} for d in decisions]
    rec = export.review_record(scope, decisions, suggested, ta_rows, {"C1": 2}, None,
                               draft_feedback(decisions), [], hits, "mock")
    data = json.loads(export.to_json(rec))
    assert data["suggested"]["scores"]["suggested_total"] == suggested.suggested_total
    assert data["ta_approved"]["criterion_points"] == {"C1": 2} and data["ta_approved"]["total"] is None
    rows = list(csv.DictReader(io.StringIO(export.requirements_csv(rec))))
    assert len(rows) == len(decisions) and rows[0]["student"] == "S001"
    srows = list(csv.DictReader(io.StringIO(export.scores_csv(rec))))
    assert srows[-1]["criterion_id"] == "TOTAL" and srows[0]["ta_approved_points"] == "2"
