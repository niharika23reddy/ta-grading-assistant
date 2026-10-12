"""Draft, load, save and sanity-check the grading checklist.

Drafts are only suggestions extracted from the documents the TA uploaded. The
TA confirms or edits them before any student is evaluated, and nothing here
invents points or partial-credit rules that the rubric does not state.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field

from .models import Criterion, Requirement

REQ_LINE = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)]|\(?[a-z][.)])\s+(.*\S)\s*$", re.I)
REQ_WORDS = re.compile(r"\b(must|should|include|includes|submit|provide|implement|write|explain|"
                       r"describe|discuss|compare|report|plot|create|use|cite|analy[sz]e|calculate|"
                       r"answer|define|design|test|document|install|configure|set up|setup|take|"
                       r"screenshot|capture|run|open|download|upload|save|record|verify|connect|"
                       r"enable|launch|add|attach|paste|show|list|identify|complete|required)\b", re.I)
# Plain sentences (not list items) count only when they read as an instruction.
IMPERATIVE_START = re.compile(r"^\s*(please\s+)?(install|configure|set up|setup|take|capture|run|open|"
                              r"download|upload|save|record|verify|connect|enable|launch|add|attach|paste|"
                              r"create|write|explain|describe|discuss|compare|submit|provide|implement|"
                              r"include|plot|use|cite|analy[sz]e|calculate|answer|define|design|test|"
                              r"document|list|identify|complete|show|make|ensure|log in|sign in|navigate|"
                              r"click|select|go to|start|restart|update|check)\b", re.I)
OBLIGATION = re.compile(r"\b(you must|you should|you need to|students must|is required|are required)\b", re.I)
POINTS_RE = re.compile(r"\(?\b(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)\b\)?", re.I)
TOTAL_RE = re.compile(r"(?:out of|total(?: of)?|worth)\s+(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)", re.I)
EXEC_WORDS = re.compile(r"\b(runs?|executes?|output|passes|tests? pass|compiles?|runtime|performance)\b", re.I)


@dataclass
class Checklist:
    course: str = ""
    assignment: str = ""
    requirements: list[Requirement] = field(default_factory=list)
    criteria: list[Criterion] = field(default_factory=list)
    approved: bool = False

    def to_json(self) -> str:
        return json.dumps({
            "course": self.course, "assignment": self.assignment, "approved": self.approved,
            "requirements": [r.to_dict() for r in self.requirements],
            "criteria": [c.to_dict() for c in self.criteria],
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Checklist":
        d = json.loads(text)
        return cls(
            course=d.get("course", ""), assignment=d.get("assignment", ""),
            approved=bool(d.get("approved", False)),
            requirements=[Requirement(**r) for r in d.get("requirements", [])],
            criteria=[Criterion(**c) for c in d.get("criteria", [])],
        )


def draft_requirements(instructions: str) -> list[Requirement]:
    """List-item lines that read like a deliverable become draft requirements."""
    reqs, seen = [], set()
    for line in instructions.splitlines():
        m = REQ_LINE.match(line)
        if m:
            text = m.group(1)
            if not REQ_WORDS.search(text):
                continue
        elif IMPERATIVE_START.match(line) or OBLIGATION.search(line):
            text = line
        else:
            continue
        text = POINTS_RE.sub("", text).strip(" -:;\t")
        if len(text) < 8 or "|" in text or text.lower() in seen:
            continue
        seen.add(text.lower())
        reqs.append(Requirement(req_id=f"R{len(reqs) + 1}", description=text,
                                needs_execution=bool(EXEC_WORDS.search(text))))
    return reqs


def parse_requirements_csv(text: str) -> list[Requirement]:
    """Parse the requirements template (templates/requirements_template.csv)."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
        if not row.get("req_id") or not row.get("description"):
            continue
        out.append(Requirement(row["req_id"], row["description"], row.get("criterion_id", ""),
                               row.get("check_correctness", "true").lower() != "false",
                               row.get("needs_execution", "false").lower() == "true",
                               row.get("notes", "")))
    return out


def parse_rubric_csv(text: str) -> list[Criterion]:
    """Parse the structured rubric template (templates/rubric_template.csv)."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
        if not row.get("criterion_id"):
            continue
        max_pts = float(row["max_points"])
        levels = {}
        for key in ("full", "partial", "none"):
            val = row.get(f"{key}_points", "")
            if val != "":
                levels[key] = float(val)
        out.append(Criterion(row["criterion_id"], row.get("description", ""), max_pts, levels))
    return out


def draft_rubric_from_text(text: str) -> list[Criterion]:
    """Lines that state a point value become draft criteria. Partial credit is left blank."""
    out = []
    for line in text.splitlines():
        if "|" in line:
            crit = _criterion_from_table_row(line, len(out) + 1)
            if crit:
                out.append(crit)
            continue
        m = POINTS_RE.search(line)
        if not m or TOTAL_RE.search(line):
            continue
        desc = POINTS_RE.sub("", line).strip(" -:;*•\t()")
        desc = re.sub(r"^\(?\d+[.)]\s*", "", desc)
        if len(desc) < 4:
            continue
        out.append(Criterion(f"C{len(out) + 1}", desc, float(m.group(1)), {}))
    return out


NUMBER_CELL = re.compile(r"^\(?(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\)?$", re.I)
RANGE_CELL = re.compile(r"^\d+(?:\.\d+)?\s*[-\u2013]\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?$", re.I)


def _criterion_from_table_row(line: str, index: int) -> Criterion | None:
    """A rubric table row: the first text cell is the criterion, the largest number is its max points.
    Level columns (e.g. Excellent 5 / Partial 3 / Missing 0) are not mapped automatically; the TA
    fills partial credit from the official rubric."""
    cells = [c.strip() for c in line.split("|") if c.strip()]
    texts, numbers = [], []
    for c in cells:
        m = NUMBER_CELL.match(c) or RANGE_CELL.match(c)
        if m:
            numbers.append(float(m.group(1)))
        else:
            pm = POINTS_RE.search(c)
            if pm and len(c) < 25:
                numbers.append(float(pm.group(1)))
            else:
                texts.append(c)
    if not texts or not numbers:
        return None
    desc = texts[0]
    if re.fullmatch(r"(total|grand total|sum)\b.*", desc, re.I):
        return None
    return Criterion(f"C{index}", desc, max(numbers), {})


def stated_total(text: str) -> float | None:
    m = TOTAL_RE.search(text)
    return float(m.group(1)) if m else None


def find_conflicts(checklist: Checklist, instructions_text: str = "", rubric_text: str = "") -> list[str]:
    """Inconsistencies the TA should resolve before grading."""
    issues = []
    crit_ids = {c.criterion_id for c in checklist.criteria}
    seen = set()
    for r in checklist.requirements:
        if r.req_id in seen:
            issues.append(f"Duplicate requirement ID {r.req_id}.")
        seen.add(r.req_id)
        if not r.criterion_id:
            issues.append(f"{r.req_id} is not linked to any rubric criterion, so it cannot affect the score.")
        elif r.criterion_id not in crit_ids:
            issues.append(f"{r.req_id} points to criterion {r.criterion_id}, which is not in the rubric.")
    linked = {r.criterion_id for r in checklist.requirements}
    for c in checklist.criteria:
        if c.criterion_id not in linked:
            issues.append(f"Rubric criterion {c.criterion_id} ({c.description}) has no matching requirement in the instructions.")
        if "partial" not in c.levels:
            issues.append(f"{c.criterion_id} has no partial-credit rule; partial work will be left for the TA.")
        for lvl, pts in c.levels.items():
            if pts > c.max_points or pts < 0:
                issues.append(f"{c.criterion_id} level '{lvl}' gives {pts} points, outside 0-{c.max_points}.")
    rubric_sum = sum(c.max_points for c in checklist.criteria)
    for label, text in (("instructions", instructions_text), ("rubric", rubric_text)):
        total = stated_total(text) if text else None
        if total is not None and checklist.criteria and abs(total - rubric_sum) > 1e-9:
            issues.append(f"The {label} state a total of {total:g} points, but rubric criteria add up to {rubric_sum:g}.")
    return issues
