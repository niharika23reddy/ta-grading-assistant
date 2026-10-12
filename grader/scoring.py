"""Rubric arithmetic. All totals are computed here, never by the language model.

Points come only from the rubric: a criterion's max points (full credit), its
stated "none" level (default 0) and its stated "partial" level. When the rubric
has no partial-credit rule, partial work gets no suggested points and is left
to the TA. Unverifiable criteria are left unresolved rather than scored zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .checklist import PER_ITEM_RULE
from .models import Criterion, Decision, Requirement


@dataclass
class CriterionScore:
    criterion_id: str
    description: str
    max_points: float
    level: str  # full | partial | none | unresolved
    suggested_points: float | None
    provisional: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoreSummary:
    criteria: list[CriterionScore]
    suggested_total: float  # sum of resolved criteria only
    max_total: float
    unresolved: list[str]
    provisional: list[str]
    complete: bool  # every criterion has suggested points

    def to_dict(self) -> dict:
        d = asdict(self)
        d["criteria"] = [c.to_dict() for c in self.criteria]
        return d


def requirement_level(decision: Decision, requirement: Requirement) -> tuple[str, bool, str]:
    """Map a decision to a rubric level. Returns (level, provisional, note)."""
    if decision.status == "Unable to verify":
        return "unresolved", False, f"{decision.req_id} could not be verified."
    if decision.status == "Missing":
        return "none", False, ""
    if decision.status == "Partially met":
        return "partial", False, ""
    # Met
    if decision.correctness in ("Partially correct", "Incorrect"):
        return "partial", False, f"{decision.req_id} is complete but {decision.correctness.lower()}."
    if decision.correctness == "Not assessed" and requirement.check_correctness:
        return "full", True, f"{decision.req_id} is complete; correctness was not assessed."
    return "full", False, ""


def score(criteria: list[Criterion], requirements: list[Requirement], decisions: list[Decision]) -> ScoreSummary:
    by_req = {d.req_id: d for d in decisions}
    results = []
    for c in criteria:
        reqs = [r for r in requirements if r.criterion_id == c.criterion_id]
        notes: list[str] = []
        if not reqs:
            results.append(CriterionScore(c.criterion_id, c.description, c.max_points, "unresolved", None,
                                          False, ["No requirement is linked to this criterion."]))
            continue
        levels, provisional = [], False
        for r in reqs:
            d = by_req.get(r.req_id)
            if d is None:
                levels.append("unresolved")
                notes.append(f"{r.req_id} has not been evaluated.")
                continue
            lvl, prov, note = requirement_level(d, r)
            levels.append(lvl)
            provisional |= prov
            if note:
                notes.append(note)

        if "unresolved" in levels:
            level = "unresolved"
        elif all(lv == "full" for lv in levels):
            level = "full"
        elif all(lv == "none" for lv in levels):
            level = "none"
        else:
            level = "partial"

        if level == "full":
            pts = c.levels.get("full", c.max_points)
        elif level == "none":
            pts = c.levels.get("none", 0.0)
        elif level == "partial" and "partial" in c.levels:
            pts = c.levels["partial"]
            if PER_ITEM_RULE.search(c.notes or ""):
                provisional = True
                notes.append("Rubric describes a per-item rule for partial credit; check these points.")
        elif level == "partial":
            pts, provisional = None, True
            notes.append("Rubric gives no partial-credit rule; TA must decide the points.")
        else:
            pts = None
        results.append(CriterionScore(c.criterion_id, c.description, c.max_points, level,
                                      None if pts is None else float(pts), provisional, notes))

    resolved = [r.suggested_points for r in results if r.suggested_points is not None]
    return ScoreSummary(
        criteria=results,
        suggested_total=round(sum(resolved), 2),
        max_total=round(sum(c.max_points for c in criteria), 2),
        unresolved=[r.criterion_id for r in results if r.suggested_points is None],
        provisional=[r.criterion_id for r in results if r.provisional],
        complete=all(r.suggested_points is not None for r in results),
    )


def approved_total(approved: dict[str, float | None], criteria: list[Criterion]) -> tuple[float | None, list[str]]:
    """Total of TA-approved points. Returns (total or None if incomplete, problems)."""
    problems, total = [], 0.0
    for c in criteria:
        pts = approved.get(c.criterion_id)
        if pts is None:
            problems.append(f"{c.criterion_id} has no approved points yet.")
            continue
        if pts < 0 or pts > c.max_points:
            problems.append(f"{c.criterion_id}: {pts:g} is outside 0-{c.max_points:g}.")
            continue
        total += pts
    return (round(total, 2) if not problems else None), problems
