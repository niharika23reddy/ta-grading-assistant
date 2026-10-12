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


@dataclass
class RubricCSV:
    criteria: list[Criterion]
    requirements: list[Requirement]  # drafted from level descriptions, already linked
    total: float | None  # total stated in a "Total" row, if any


def parse_rubric_csv(text: str) -> list[Criterion]:
    return read_rubric_csv(text).criteria


def read_rubric_csv(text: str) -> RubricCSV:
    """Read either the app's rubric template or a level-based rubric export
    (columns like Criteria, Expected, Expected Pts, Satisfactory, Satisfactory Pts, ..., Max Pts)."""
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    if not rows:
        return RubricCSV([], [], None)
    headers = [h.strip().lower() for h in rows[0]]
    if "criterion_id" in headers:
        return RubricCSV(_parse_template_rubric(text), [], None)
    return _parse_level_rubric(rows)


def _num(val: str) -> float | None:
    m = re.fullmatch(r"\s*\(?(-?\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)?\)?\s*", val or "", re.I)
    return float(m.group(1)) if m else None


PTS_HEADER = re.compile(r"\b(pts?|points?|marks?)\b", re.I)
NUMBERED_ITEM = re.compile(r"(?:^|\s)\d+[.)]\s+")
PER_ITEM_RULE = re.compile(r"\b(each|per)\b", re.I)


def _parse_level_rubric(rows: list[list[str]]) -> RubricCSV:
    raw = [h.strip() for h in rows[0]]
    low = [h.lower() for h in raw]
    max_col = next((i for i, h in enumerate(low) if "max" in h or h == "total"), None)
    level_cols = [i for i, h in enumerate(low) if PTS_HEADER.search(h) and i != max_col and i > 0]
    levels_meta = []  # (name, text column or None, points column)
    for i in level_cols:
        text_col = i - 1 if i - 1 > 0 and i - 1 not in level_cols and i - 1 != max_col else None
        name = raw[text_col] if text_col is not None else PTS_HEADER.sub("", raw[i]).strip()
        levels_meta.append((name or f"Level {len(levels_meta) + 1}", text_col, i))

    criteria, reqs, total = [], [], None
    for row in rows[1:]:
        row = row + [""] * (len(raw) - len(row))
        name = row[0].strip()
        if not name:
            continue
        if re.fullmatch(r"(grand )?total\b.*", name, re.I):
            vals = [_num(v) for v in row[1:]]
            total = next((v for v in reversed(vals) if v is not None), None)
            continue
        pts = [(n, row[t].strip() if t is not None else "", _num(row[p])) for n, t, p in levels_meta]
        pts = [x for x in pts if x[2] is not None]
        max_pts = _num(row[max_col]) if max_col is not None else None
        if max_pts is None and pts:
            max_pts = max(p for _, _, p in pts)
        if max_pts is None:
            continue
        cid = f"C{len(criteria) + 1}"
        levels, notes = {}, []
        if pts:
            levels["full"] = pts[0][2]
        if len(pts) >= 2:
            levels["none"] = pts[-1][2]
        middles = pts[1:-1]
        if middles:
            levels["partial"] = middles[0][2]
            for lname, ltext, lpts in middles:
                notes.append(f"{lname} ({lpts:g} pts): {ltext}".strip())
            if len(middles) > 1:
                notes.append(f"Rubric has {len(middles)} partial levels; the app suggests the first. "
                             "Choose the right one when approving points.")
        expected = pts[0][1] if pts else ""
        criteria.append(Criterion(cid, name, max_pts, levels, " | ".join(notes)))
        items = [x.strip(" .;") for x in NUMBERED_ITEM.split(expected) if x.strip(" .;")] or (
            [expected.strip()] if expected.strip() else [])
        for item in items:
            reqs.append(Requirement(f"R{len(reqs) + 1}", f"{name}: {item}", cid))
    return RubricCSV(criteria, reqs, total)


def attach_instruction_context(reqs: list[Requirement], criteria: list[Criterion], instructions: str,
                               limit: int = 700) -> None:
    """Copy the part of the instructions that belongs to each criterion (e.g. the "Lab Exercise 2"
    section) into its requirements' notes, so the reviewer knows the exact command or question."""
    if not instructions.strip():
        return
    starts = {}
    for c in criteria:
        m = re.search(re.escape(c.description), instructions, re.I)
        if not m:
            num = re.search(r"(\d+)\s*$", c.description)
            word = re.sub(r"\s*\d+\s*$", "", c.description).split()[-1:] if num else []
            if num and word:
                m = re.search(rf"\b{re.escape(word[0])}\s*#?\s*{num.group(1)}\b", instructions, re.I)
        if m:
            starts[c.criterion_id] = m.start()
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    sections = {}
    for i, (cid, pos) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(instructions)
        sections[cid] = re.sub(r"\n{2,}", "\n", instructions[pos:end]).strip()
    for r in reqs:
        sec = sections.get(r.criterion_id)
        if sec and not r.notes:
            clipped = sec if len(sec) <= limit else sec[:limit].rsplit(" ", 1)[0] + " ..."
            r.notes = "From the instructions: " + clipped


def _parse_template_rubric(text: str) -> list[Criterion]:
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


def find_conflicts(checklist: Checklist, instructions_text: str = "", rubric_text: str = "",
                   rubric_total: float | None = None) -> list[str]:
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
        elif PER_ITEM_RULE.search(c.notes or ""):
            issues.append(f"{c.criterion_id} ({c.description}) describes a per-item partial-credit rule "
                          f"that differs from its fixed partial points; partial scores there stay provisional.")
        for lvl, pts in c.levels.items():
            if pts > c.max_points or pts < 0:
                issues.append(f"{c.criterion_id} level '{lvl}' gives {pts} points, outside 0-{c.max_points}.")
    rubric_sum = sum(c.max_points for c in checklist.criteria)
    for label, text in (("instructions", instructions_text), ("rubric", rubric_text)):
        total = stated_total(text) if text else None
        if total is not None and checklist.criteria and abs(total - rubric_sum) > 1e-9:
            issues.append(f"The {label} state a total of {total:g} points, but rubric criteria add up to {rubric_sum:g}.")
    if rubric_total is not None and checklist.criteria and abs(rubric_total - rubric_sum) > 1e-9:
        issues.append(f"The rubric's Total row says {rubric_total:g} points, but criteria add up to {rubric_sum:g}.")
    return issues
