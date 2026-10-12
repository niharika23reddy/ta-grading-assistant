from grader.models import Scope
from grader.pipeline import ingest_reference, ingest_submission
from tests.conftest import submission_files

A = Scope("DATA101", "A1", "S001", "v1")
B = Scope("DATA101", "A1", "S002", "v1")


def test_students_never_mix(store, settings):
    ingest_submission(store, A, submission_files("S001_v1"), settings)
    ingest_submission(store, B, submission_files("S002_v1"), settings)
    hits = store.search(B, "clean_data function removes missing temperature", k=20)
    assert hits, "S002 has chunks"
    assert all("clean_data" not in c.text for c in hits)  # only S001 wrote clean_data
    assert all("Fanaee" not in c.text for c in store.all_chunks(B))


def test_reference_is_never_student_evidence(store, settings):
    ingest_submission(store, B, submission_files("S002_v1"), settings)
    ingest_reference(store, "DATA101", "A1", [("ref.md", b"INSTRUCTOR REFERENCE mean 4504 median 4548")], settings)
    assert all("INSTRUCTOR REFERENCE" not in c.text for c in store.all_chunks(B))
    assert all("INSTRUCTOR" not in c.text for c in store.search(B, "mean 4504 median 4548", k=20))
    assert store.reference_chunks("DATA101", "A1", "median")


def test_submission_versions_are_separate_and_reingest_replaces(store, settings):
    v1, v2 = A, Scope("DATA101", "A1", "S001", "v2")
    ingest_submission(store, v1, [("r.md", b"first draft about bikes")], settings)
    ingest_submission(store, v2, [("r.md", b"second draft with histogram")], settings)
    assert [c.text for c in store.all_chunks(v1)] == ["first draft about bikes"]
    assert [c.text for c in store.all_chunks(v2)] == ["second draft with histogram"]
    ingest_submission(store, v1, [("r.md", b"replaced v1 text")], settings)
    assert [c.text for c in store.all_chunks(v1)] == ["replaced v1 text"]


def test_other_course_or_assignment_not_visible(store, settings):
    ingest_submission(store, Scope("DATA101", "A2", "S001", "v1"), [("x.md", b"assignment two text")], settings)
    ingest_submission(store, Scope("STAT200", "A1", "S001", "v1"), [("x.md", b"other course text")], settings)
    assert store.count(A) == 0 and store.search(A, "text") == []
