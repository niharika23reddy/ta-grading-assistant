"""Comparing suggested totals with the TA's own grades."""
from grader import accuracy


def test_compare_counts_closeness_and_direction():
    r = accuracy.compare({"a": (10, 10), "b": (10, 9.2), "c": (10, 8.5), "d": (5, 8)})
    assert r["students"] == 4
    assert r["within_1"] == 50 and r["within_2"] == 75
    assert r["average_gap"] == round((0 + 0.8 + 1.5 + 3) / 4, 2)
    assert r["too_low"] == 2 and r["too_high"] == 1
    assert r["gaps"][0] == ("d", 3.0)
    assert accuracy.compare({}) is None


def test_grades_are_saved_locally_and_reloaded(tmp_path):
    path = accuracy.grades_path(tmp_path, "C1", "A1", "v1")
    assert accuracy.load_grades(path) == {}
    accuracy.save_grades(path, {"S001": 9.5})
    assert accuracy.load_grades(path) == {"S001": 9.5}
    assert path.is_relative_to(tmp_path)
