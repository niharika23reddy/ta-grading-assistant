"""How close the app's suggested totals are to the grades the TA actually gave.

The TA's grades are typed into the app and saved on this computer only (data/ta_grades)."""
from __future__ import annotations

import json
from pathlib import Path


def grades_path(data_dir: Path, course: str, assignment: str, version: str) -> Path:
    from .batch import slug
    return Path(data_dir) / "ta_grades" / f"{slug(course, assignment, version)}.json"


def load_grades(path: Path) -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(Path(path).read_text()).items() if v is not None}
    except (OSError, ValueError):
        return {}


def save_grades(path: Path, grades: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(grades, indent=2, sort_keys=True))


def compare(pairs: dict[str, tuple[float, float]]) -> dict | None:
    """pairs maps student -> (TA grade, suggested total). None when there is nothing to compare."""
    if not pairs:
        return None
    gaps = {s: round(sug - ta, 2) for s, (ta, sug) in pairs.items()}
    n = len(gaps)
    return {
        "students": n,
        "within_1": round(100 * sum(abs(g) <= 1 for g in gaps.values()) / n),
        "within_2": round(100 * sum(abs(g) <= 2 for g in gaps.values()) / n),
        "average_gap": round(sum(abs(g) for g in gaps.values()) / n, 2),
        "too_low": sum(g < 0 for g in gaps.values()),
        "too_high": sum(g > 0 for g in gaps.values()),
        "gaps": sorted(gaps.items(), key=lambda kv: -abs(kv[1])),
    }
