"""JSON and CSV exports of a review. Exports are files for the TA; nothing is sent anywhere."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from .models import Decision, Scope
from .scoring import ScoreSummary


def review_record(scope: Scope, decisions: list[Decision], suggested: ScoreSummary,
                  ta_decisions: list[dict], approved_points: dict, approved_total: float | None,
                  ta_feedback: str, extraction_warnings: list[str], injection_hits: list[str],
                  model: str) -> dict:
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "course": scope.course, "assignment": scope.assignment,
        "student": scope.student, "version": scope.version,
        "model": model,
        "status_note": "Suggested results are machine-generated. Only TA-approved values are grades.",
        "extraction_warnings": extraction_warnings,
        "possible_grader_manipulation": injection_hits,
        "suggested": {
            "requirements": [d.to_dict() for d in decisions],
            "scores": suggested.to_dict(),
        },
        "ta_approved": {
            "requirements": ta_decisions,
            "criterion_points": approved_points,
            "total": approved_total,
            "feedback": ta_feedback,
        },
    }


def to_json(record: dict) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False)


def requirements_csv(record: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["course", "assignment", "student", "version", "req_id", "description",
                "suggested_status", "suggested_correctness", "evidence", "evidence_location",
                "evidence_verified", "reasoning", "improvement", "ta_review_required", "review_reasons",
                "ta_status", "ta_correctness", "ta_comment"])
    ta = {r["req_id"]: r for r in record["ta_approved"]["requirements"]}
    for d in record["suggested"]["requirements"]:
        t = ta.get(d["req_id"], {})
        w.writerow([record["course"], record["assignment"], record["student"], record["version"],
                    d["req_id"], d["description"], d["status"], d["correctness"],
                    " || ".join(e["quote"] for e in d["evidence"]),
                    " || ".join(e["location"] for e in d["evidence"]),
                    " || ".join(str(e["verified"]) for e in d["evidence"]),
                    d["reasoning"], d["improvement"], d["ta_review"], " | ".join(d["review_reasons"]),
                    t.get("status", ""), t.get("correctness", ""), t.get("comment", "")])
    return buf.getvalue()


def scores_csv(record: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["course", "assignment", "student", "version", "criterion_id", "description",
                "max_points", "suggested_level", "suggested_points", "provisional", "notes", "ta_approved_points"])
    approved = record["ta_approved"]["criterion_points"]
    sc = record["suggested"]["scores"]
    for c in sc["criteria"]:
        w.writerow([record["course"], record["assignment"], record["student"], record["version"],
                    c["criterion_id"], c["description"], c["max_points"], c["level"],
                    "" if c["suggested_points"] is None else c["suggested_points"],
                    c["provisional"], " | ".join(c["notes"]),
                    "" if approved.get(c["criterion_id"]) is None else approved[c["criterion_id"]]])
    w.writerow([record["course"], record["assignment"], record["student"], record["version"],
                "TOTAL", "", sc["max_total"], "", sc["suggested_total"],
                bool(sc["provisional"]) or not sc["complete"], "Suggested total covers resolved criteria only",
                "" if record["ta_approved"]["total"] is None else record["ta_approved"]["total"]])
    return buf.getvalue()
