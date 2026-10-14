"""Streamlit interface: one assignment at a time, reviewing one student at a time (tab 5 grades many).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from grader import accuracy, batch, export
from grader.checklist import (Checklist, attach_instruction_context, draft_requirements, draft_rubric_from_text,
                              find_conflicts, parse_requirements_csv, read_rubric_csv,
                              split_per_item_requirements)
from grader.config import get_settings
from grader.embeddings import get_embedder
from grader.evaluate import evaluate_submission
from grader.extract import has_unread_content, ocr_available, unreadable_parts
from grader.llm import get_provider
from grader.models import CORRECTNESS, STATUSES, Criterion, Decision, Requirement, Scope
from grader.pipeline import draft_feedback, ingest_reference, ingest_submission, read_files
from grader.scoring import approved_total, score
from grader.static_code import analyze, summary_text
from grader.store import ChunkStore

st.set_page_config(page_title="TA Grading Assistant", layout="wide")
settings = get_settings()
DATA = settings.data_dir
(DATA / "checklists").mkdir(parents=True, exist_ok=True)
(DATA / "reviews").mkdir(parents=True, exist_ok=True)


slug = batch.slug


def get_store():
    return _store(str(DATA / "chroma"))


@st.cache_resource
def _store(path: str):
    return ChunkStore(get_embedder(settings), path=path)


provider = get_provider(settings)


def reset_timing() -> None:
    if hasattr(provider, "take_stats"):
        provider.take_stats()


def timing_note(started: float) -> str:
    """How long a run took and, for the local model, how much it read and wrote."""
    secs = time.monotonic() - started
    note = f"took {int(secs // 60)} min {int(secs % 60)} s"
    stats = provider.take_stats() if hasattr(provider, "take_stats") else None
    if stats and stats["calls"]:
        note += (f"; the model read {stats['read_tokens']:,} new tokens and wrote "
                 f"{stats['written_tokens']:,} over {stats['calls']} questions")
    return note
ss = st.session_state
ss.setdefault("checklist", None)
ss.setdefault("reports", [])
ss.setdefault("decisions", None)

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("TA Grading Assistant")
    st.caption("Everything runs on this computer. Nothing is sent or published automatically.")
    ok, msg = provider.health()
    (st.success if ok else st.error)(msg)
    if settings.llm_provider == "mock":
        st.warning("Mock mode is a keyword heuristic for testing. Do not use it for real grading.")
    if settings.ocr_enabled and not ocr_available():
        st.info("OCR is on, but Tesseract is not installed. Scanned pages will be flagged, not read. "
                "Install with: brew install tesseract")
    st.divider()
    course = st.text_input("Course", value="COURSE101")
    assignment = st.text_input("Assignment", value="A1")
    st.caption("Use anonymized student codes such as S001.")
    ss.setdefault("student_code", "S001")
    student = st.text_input("Student code", key="student_code")
    version = st.text_input("Submission version", value="v1")
    graded = batch.reviewed_students(DATA, course.strip(), assignment.strip(), version.strip())
    if graded:
        def _pick():
            ss.student_code = ss.pick_student
        st.selectbox(f"Or pick one of {len(graded)} graded students", graded, index=None,
                     placeholder="Choose a student", key="pick_student", on_change=_pick)

scope = Scope(course.strip(), assignment.strip(), student.strip(), version.strip())
if ss.get("decisions_scope") != scope:  # another student or assignment: never show the previous one's results
    ss.decisions, ss.decisions_scope = None, scope
checklist_path = DATA / "checklists" / f"{slug(scope.course, scope.assignment)}.json"
if ss.checklist is None and checklist_path.exists():
    ss.checklist = Checklist.from_json(checklist_path.read_text())

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Assignment setup", "2. Student submission", "3. Review", "4. Export",
                                        "5. Many students"])

# ---------------- 1. assignment setup ----------------
with tab1:
    st.header("Instructions, rubric and grading checklist")
    st.write("Upload the official documents. The app drafts a checklist from them; "
             "you confirm or correct it before any student is evaluated.")
    c1, c2, c3 = st.columns(3)
    instr_files = c1.file_uploader("Assignment instructions (or a requirements CSV)", accept_multiple_files=True,
                                   type=["pdf", "docx", "txt", "md", "csv"])
    rubric_files = c2.file_uploader("Grading rubric (CSV template preferred)", accept_multiple_files=True,
                                    type=["csv", "pdf", "docx", "txt", "md"])
    ref_files = c3.file_uploader("Reference answers (optional, instructor only)", accept_multiple_files=True)

    if st.button("Draft checklist from documents", disabled=not instr_files):
        csv_reqs = [r for f in instr_files if f.name.lower().endswith(".csv")
                    for r in parse_requirements_csv(f.getvalue().decode("utf-8", errors="replace"))]
        instr_reports = read_files([(f.name, f.getvalue()) for f in instr_files
                                    if not f.name.lower().endswith(".csv")], settings.ocr_enabled)
        instr_text = "\n".join(r.text for r in instr_reports)
        criteria, rubric_reqs, rubric_text, rubric_total = [], [], "", None
        for f in rubric_files or []:
            if f.name.lower().endswith(".csv"):
                parsed = read_rubric_csv(f.getvalue().decode("utf-8", errors="replace"))
                criteria += parsed.criteria
                rubric_reqs += parsed.requirements
                rubric_total = parsed.total if parsed.total is not None else rubric_total
            else:
                reps = read_files([(f.name, f.getvalue())], settings.ocr_enabled)
                rubric_text += "\n".join(r.text for r in reps)
                instr_reports += reps
        if rubric_text:
            criteria += draft_rubric_from_text(rubric_text)
        if csv_reqs:
            reqs = csv_reqs
        elif rubric_reqs:
            # The rubric's level descriptions name the deliverables and are already linked to
            # criteria; the instructions supply the exact commands and questions as notes.
            reqs = split_per_item_requirements(rubric_reqs, criteria, instr_text)
            attach_instruction_context(reqs, criteria, instr_text)
        else:
            reqs = draft_requirements(instr_text)
        ss.checklist = Checklist(scope.course, scope.assignment, reqs, criteria)
        ss.instr_text, ss.rubric_text, ss.rubric_total = instr_text, rubric_text, rubric_total
        ss.setup_warnings = unreadable_parts(instr_reports)
        if ref_files:
            _, n = ingest_reference(get_store(), scope.course, scope.assignment,
                                    [(f.name, f.getvalue()) for f in ref_files], settings)
            st.success(f"Stored {n} reference chunks separately from student work.")

    for w in ss.get("setup_warnings", []):
        st.warning(w)

    cl: Checklist | None = ss.checklist
    if cl:
        st.subheader("Requirements")
        st.caption("Edit, add or delete rows. Link each requirement to a rubric criterion.")
        req_df = st.data_editor(
            pd.DataFrame([r.to_dict() for r in cl.requirements] or
                         [Requirement("R1", "").to_dict()]),
            num_rows="dynamic", use_container_width=True, key="req_editor",
            column_config={"needs_execution": st.column_config.CheckboxColumn(
                "needs execution", help="Can only be verified by running code")})
        st.subheader("Rubric criteria")
        st.caption("Leave partial_points blank if the rubric has no partial-credit rule. "
                   "Points are never invented.")
        crit_rows = [{"criterion_id": c.criterion_id, "description": c.description, "max_points": c.max_points,
                      "full_points": c.levels.get("full"), "partial_points": c.levels.get("partial"),
                      "none_points": c.levels.get("none"), "per_item_points": c.levels.get("per_item"),
                      "notes": c.notes} for c in cl.criteria]
        crit_df = st.data_editor(
            pd.DataFrame(crit_rows or [{"criterion_id": "C1", "description": "", "max_points": 0.0,
                                        "full_points": None, "partial_points": None, "none_points": None,
                                        "per_item_points": None, "notes": ""}]),
            num_rows="dynamic", use_container_width=True, key="crit_editor")

        def from_editors() -> Checklist:
            reqs = [Requirement(str(r["req_id"]).strip(), str(r["description"]).strip(),
                                str(r.get("criterion_id") or "").strip(), bool(r.get("check_correctness", True)),
                                bool(r.get("needs_execution", False)), str(r.get("notes") or ""))
                    for r in req_df.to_dict("records") if str(r.get("description") or "").strip()]
            crits = []
            for r in crit_df.to_dict("records"):
                if not str(r.get("criterion_id") or "").strip():
                    continue
                levels = {k: float(r[f"{k}_points"]) for k in ("full", "partial", "none", "per_item")
                          if r.get(f"{k}_points") is not None and not pd.isna(r.get(f"{k}_points"))}
                crits.append(Criterion(str(r["criterion_id"]).strip(), str(r["description"] or ""),
                                       float(r["max_points"] or 0), levels, str(r.get("notes") or "")))
            return Checklist(scope.course, scope.assignment, reqs, crits, False)

        current = from_editors()
        conflicts = find_conflicts(current, ss.get("instr_text", ""), ss.get("rubric_text", ""),
                                   ss.get("rubric_total"))
        if conflicts:
            st.subheader("Check these before approving")
            for c in conflicts:
                st.warning(c)
        n_exec = sum(r.needs_execution for r in current.requirements)
        st.info(f"{len(current.requirements) - n_exec} requirement(s) can be checked from the text. "
                f"{n_exec} need code to run, which this app never does, so they go to TA review.")
        approve = st.checkbox("I have checked this list against the official instructions and rubric",
                              value=cl.approved)
        if st.button("Save checklist"):
            current.approved = approve
            checklist_path.write_text(current.to_json())
            ss.checklist = current
            st.success(f"Saved{' and approved' if approve else ' (not yet approved)'}.")
        st.download_button("Download checklist JSON", current.to_json(),
                           file_name=f"{slug(scope.course, scope.assignment)}_checklist.json")
    up = st.file_uploader("Or load a saved checklist JSON", type=["json"], key="load_cl")
    if up and st.button("Load checklist"):
        ss.checklist = Checklist.from_json(up.getvalue().decode())
        st.rerun()

# ---------------- 2. submission ----------------
with tab2:
    st.header(f"Submission: {scope.student} ({scope.version})")
    st.caption("Files are read as text only. Student code is never executed.")
    sub_files = st.file_uploader("Student files (PDF, DOCX, code, notebooks, images, ZIP)",
                                 accept_multiple_files=True, key="sub_files")
    if st.button("Read and index submission", disabled=not sub_files):
        with st.spinner("Reading files and building the search index..."):
            reports, n = ingest_submission(get_store(), scope, [(f.name, f.getvalue()) for f in sub_files],
                                           settings)
        ss.reports, ss.decisions, ss.indexed_scope = reports, None, scope
        st.success(f"Indexed {n} chunks for {scope.student} {scope.version}.")

    if ss.reports and ss.get("indexed_scope") == scope:
        st.subheader("Extraction quality")
        rows = [{"file": r.file, "type": r.file_type, "readable": r.readable,
                 "pages/parts": len(r.pages), "characters": len(r.text),
                 "OCR pages": sum(p.method == "ocr" for p in r.pages),
                 "screenshots": sum(p.text.count(": image present;") + p.text.count(": OCR text of the image begins]")
                                    for p in r.pages),
                 "unread pages": sum(p.method == "none" for p in r.pages)} for r in ss.reports]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        warns = unreadable_parts(ss.reports)
        for w in warns:
            st.warning(w)
        if not warns:
            st.success("All files were read without warnings.")
        facts = analyze(ss.reports)
        if facts:
            st.subheader("Static code facts")
            st.code(summary_text(facts))
    store_count = get_store().count(scope)
    if store_count:
        st.subheader("Search this submission")
        q = st.text_input("Search text", key="chunk_search")
        if q:
            for c in get_store().search(scope, q, k=settings.top_k):
                with st.expander(c.location()):
                    st.text(c.text)
        with st.expander(f"All {store_count} chunks"):
            for c in get_store().all_chunks(scope):
                st.markdown(f"**{c.location()}**")
                st.text(c.text)

# ---------------- 3. review ----------------
with tab3:
    st.header("Requirement review")
    cl = ss.checklist
    ready = cl is not None and cl.approved and get_store().count(scope) > 0
    if cl is None or not cl.approved:
        st.info("Approve the checklist in step 1 first.")
    elif not get_store().count(scope):
        st.info("Index this student's submission in step 2 first.")
    review_path = batch.review_path(DATA, scope)

    if ready and st.button("Run evaluation"):
        started = time.monotonic()
        reset_timing()
        bar = st.progress(0.0)
        unread = has_unread_content(ss.reports) if ss.get("indexed_scope") == scope else True
        decisions, hits = evaluate_submission(
            cl.requirements, scope, get_store(), provider, unread_content=unread,
            static_facts=summary_text(analyze(ss.reports)) if ss.get("indexed_scope") == scope else "",
            top_k=settings.top_k, parallel=settings.llm_parallel,
            whole_chars=settings.whole_submission_chars, second_look=settings.second_look,
            progress=lambda i, n, r: bar.progress(i / n, text=f"Checked {i} of {n} requirements..."))
        bar.progress(1.0, text="Done")
        ss.decisions, ss.injection_hits = decisions, hits
        ss.review_run = ss.get("review_run", 0) + 1  # fresh TA fields that start from the new suggestions
        ss.ta_feedback = draft_feedback(decisions)
        batch.save_review(review_path, scope, decisions, hits,
                          ss.reports if ss.get("indexed_scope") == scope else None)
        ss.review_unread = unread
        ss.review_warnings = unreadable_parts(ss.reports) if ss.get("indexed_scope") == scope else []
        ss.last_timing = (scope, "This evaluation " + timing_note(started) + ".")
    if ss.get("last_timing") and ss.last_timing[0] == scope:
        st.caption(ss.last_timing[1])

    if ss.decisions is None and review_path.exists() and ready:
        saved = batch.load_review(review_path)
        ss.decisions, ss.injection_hits = saved["decisions"], saved["injection_hits"]
        ss.review_unread, ss.review_warnings = saved["unread_content"], saved["extraction_warnings"]
        ss.ta_feedback = draft_feedback(ss.decisions)

    if ss.decisions and ready:
        if ss.get("injection_hits"):
            st.error("The submission contains text that looks like instructions to the grader. "
                     "It was ignored, and every item is flagged for your review:\n\n" +
                     "\n".join(f"- {h}" for h in ss.injection_hits))
        if ss.get("review_unread"):
            st.warning("Some pictures or pages had little or no readable text, so items they may answer are "
                       "'Unable to verify'. With 'Deduct points for work that could not be found' on (under "
                       "Scores), those items lose points and are marked provisional: check them yourself.")
            if ss.get("review_warnings"):
                with st.expander("What could not be read"):
                    st.markdown("\n".join(f"- {w}" for w in ss.review_warnings))

        # Widget keys include the student and the evaluation run, so TA fields never carry
        # values over from another student or from an earlier run.
        wk = f"{review_path.stem}-{ss.get('review_run', 0)}"
        ta_rows = []
        for d in ss.decisions:
            flag = " 🔎 TA review" if d.ta_review else ""
            with st.expander(f"{d.req_id} · {d.status} · {d.correctness}{flag}: {d.description[:80]}"):
                st.markdown(f"**Suggested status:** {d.status}  \n**Suggested correctness:** {d.correctness}")
                if d.evidence:
                    st.markdown("**Evidence**")
                    for e in d.evidence:
                        mark = "✅ verified" if e.verified else "⚠️ not found in submission"
                        st.markdown(f"> {e.quote}\n\n{e.location} · {mark}")
                else:
                    st.markdown("**Evidence:** none cited")
                st.markdown(f"**Reasoning:** {d.reasoning or '-'}")
                st.markdown(f"**Suggested improvement:** {d.improvement or '-'}")
                if d.review_reasons:
                    st.markdown("**Why TA review:**\n" + "\n".join(f"- {r}" for r in d.review_reasons))
                c1, c2 = st.columns(2)
                ta_status = c1.selectbox("Your status", STATUSES, index=STATUSES.index(d.status),
                                         key=f"st_{wk}_{d.req_id}")
                ta_corr = c2.selectbox("Your correctness", CORRECTNESS, index=CORRECTNESS.index(d.correctness),
                                       key=f"co_{wk}_{d.req_id}")
                comment = st.text_input("Your comment", key=f"cm_{wk}_{d.req_id}")
                ta_rows.append({"req_id": d.req_id, "status": ta_status, "correctness": ta_corr,
                                "comment": comment})
        ss.ta_rows = ta_rows

        st.subheader("Scores")
        as_missing = st.checkbox(
            "Deduct points for work that could not be found", value=settings.unverified_as_missing,
            help="On: items marked \"Unable to verify\" are scored like \"Missing\" and flagged provisional. "
                 "Off: they stay unresolved until you decide.")
        suggested = score(cl.criteria, cl.requirements, ss.decisions, as_missing)
        ss.suggested = suggested
        st.caption("Suggested points come from the model's findings and the rubric only. "
                   "Approved points are yours and are kept separate.")
        ta_decisions = [Decision(r["req_id"], "", r["status"], r["correctness"], [], "", "", False)
                        for r in ta_rows]
        ta_based = score(cl.criteria, cl.requirements, ta_decisions, as_missing)
        approved = {}
        for cs, tb in zip(suggested.criteria, ta_based.criteria):
            c1, c2, c3 = st.columns([3, 2, 2])
            sug = "unresolved" if cs.suggested_points is None else f"{cs.suggested_points:g}"
            c1.markdown(f"**{cs.criterion_id}** {cs.description}  \nmax {cs.max_points:g} · suggested **{sug}**"
                        f"{' (provisional)' if cs.provisional else ''}")
            if cs.notes:
                c1.caption("; ".join(cs.notes))
            c2.caption(f"From your reviewed statuses: "
                       f"{'unresolved' if tb.suggested_points is None else f'{tb.suggested_points:g}'}")
            approved[cs.criterion_id] = c3.number_input(
                f"Approved points {cs.criterion_id}", min_value=0.0, max_value=float(cs.max_points),
                value=None, step=0.5, key=f"ap_{wk}_{cs.criterion_id}")
        st.markdown(f"**Suggested total:** {suggested.suggested_total:g} / {suggested.max_total:g}"
                    + ("" if suggested.complete else f" (unresolved: {', '.join(suggested.unresolved)})"))
        total, problems = approved_total(approved, cl.criteria)
        ss.approved_points, ss.approved_total = approved, total
        if total is None:
            st.info("Approved total appears once every criterion has approved points. " + " ".join(problems))
        else:
            st.success(f"Approved total: {total:g} / {suggested.max_total:g}")

        st.subheader("Draft feedback")
        ss.ta_feedback = st.text_area("Edit before sharing. The app never sends feedback.",
                                      value=ss.get("ta_feedback", ""), height=220)

# ---------------- 4. export ----------------
with tab4:
    st.header("Export")
    if not (ss.decisions and ss.get("suggested")):
        st.info("Run and review an evaluation first.")
    else:
        record = export.review_record(
            scope, ss.decisions, ss.suggested, ss.get("ta_rows", []), ss.get("approved_points", {}),
            ss.get("approved_total"), ss.get("ta_feedback", ""),
            unreadable_parts(ss.reports) if ss.get("indexed_scope") == scope else ss.get("review_warnings", []),
            ss.get("injection_hits", []), provider.name)
        base = slug(scope.course, scope.assignment, scope.student, scope.version)
        st.download_button("Download JSON", export.to_json(record), file_name=f"{base}.json")
        st.download_button("Download requirements CSV", export.requirements_csv(record),
                           file_name=f"{base}_requirements.csv")
        st.download_button("Download scores CSV", export.scores_csv(record), file_name=f"{base}_scores.csv")
        st.caption("Exports stay on this computer. Grades are not published anywhere.")

# ---------------- 5. many students ----------------
with tab5:
    st.header(f"Grade many students: {scope.course} {scope.assignment} ({scope.version})")
    st.caption("Each student is read, indexed and reviewed separately, exactly as in tabs 2 and 3. "
               "Afterwards, pick a student in the sidebar to check and approve their review in tab 3.")
    cl = ss.checklist
    if cl is None or not cl.approved:
        st.info("Approve the checklist in step 1 first.")
    else:
        mode = st.radio("How are the files named?", batch.MODES, key="batch_mode")
        batch_files = st.file_uploader("All students' files", accept_multiple_files=True, key="batch_files")
        groups, split_warnings = batch.group_by_student(
            [(f.name, f.getvalue()) for f in batch_files or []], mode)
        for w in split_warnings:
            st.warning(w)
        if groups:
            done = set(batch.reviewed_students(DATA, scope.course, scope.assignment, scope.version))
            st.dataframe(pd.DataFrame([{"student": code, "files": ", ".join(n for n, _ in files),
                                        "already graded": code in done} for code, files in groups.items()]),
                         use_container_width=True)
            skip = st.checkbox("Skip students who are already graded", value=True,
                               help="Lets you continue where you stopped. Untick to grade everyone again.")
            todo = [c for c in groups if not (skip and c in done)]
            st.caption(f"{len(todo)} of {len(groups)} students will be graded. Each one takes several minutes "
                       "with a local model, so keep the computer awake and this page open until it finishes.")
            if st.button(f"Grade {len(todo)} students", disabled=not todo):
                bar = st.progress(0.0)
                reset_timing()
                for i, code in enumerate(todo):
                    started = time.monotonic()

                    def tick(j, n, r, i=i, code=code):
                        bar.progress((i + j / max(n, 1)) / len(todo),
                                     text=f"Student {i + 1} of {len(todo)} ({code}): checked {j} of {n} "
                                          "requirements...")
                    s_scope = Scope(scope.course, scope.assignment, code, scope.version)
                    try:
                        batch.grade_student(get_store(), provider, cl, s_scope, groups[code], settings, tick)
                        st.caption(f"{code}: {timing_note(started)}.")
                    except Exception as exc:  # one bad submission must not stop the others
                        st.error(f"{code}: grading failed ({exc}). The other students continue.")
                bar.progress(1.0, text="All done")
                ss.decisions = None  # reload the sidebar student's review from disk

    graded = batch.reviewed_students(DATA, scope.course, scope.assignment, scope.version)
    if cl is not None and graded:
        st.subheader(f"Graded students ({len(graded)})")
        as_missing_all = st.checkbox("Deduct points for work that could not be found",
                                     value=settings.unverified_as_missing, key="batch_as_missing")
        rows = []
        for code in graded:
            saved = batch.load_review(batch.review_path(DATA, Scope(scope.course, scope.assignment, code,
                                                                    scope.version)))
            sc = score(cl.criteria, cl.requirements, saved["decisions"], as_missing_all)
            row = {"student": code, "suggested total": sc.suggested_total, "max": sc.max_total,
                   "provisional": bool(sc.provisional) or not sc.complete,
                   "items for TA review": sum(d.ta_review for d in saved["decisions"]),
                   "unreadable content": saved["unread_content"]}
            rows.append(row)
        table = pd.DataFrame(rows)
        st.dataframe(table, use_container_width=True)
        st.caption("These are suggested points only. Approve each student's grade in tab 3.")
        csv_table = table.assign(course=scope.course, assignment=scope.assignment, version=scope.version)
        st.download_button("Download suggested scores for all students (CSV)", csv_table.to_csv(index=False),
                           file_name=f"{slug(scope.course, scope.assignment, scope.version)}_all_suggested.csv")

        st.subheader("Check accuracy against your grades")
        st.caption("Type the grade you gave each student, then save. Your grades stay on this computer.")
        g_path = accuracy.grades_path(DATA, scope.course, scope.assignment, scope.version)
        mine = accuracy.load_grades(g_path)
        edited = st.data_editor(
            table[["student", "suggested total"]].assign(**{"your grade": [mine.get(c) for c in table["student"]]}),
            disabled=["student", "suggested total"], use_container_width=True, key="ta_grades_editor",
            column_config={"your grade": st.column_config.NumberColumn(min_value=0.0, max_value=float(table["max"].max()))})
        typed = {r["student"]: float(r["your grade"]) for _, r in edited.iterrows() if pd.notna(r["your grade"])}
        if st.button("Save my grades"):
            accuracy.save_grades(g_path, typed)
            st.success(f"Saved {len(typed)} grades.")
        result = accuracy.compare({r["student"]: (typed[r["student"]], r["suggested total"])
                                   for _, r in table.iterrows() if r["student"] in typed})
        if result:
            c1, c2, c3 = st.columns(3)
            c1.metric("Within 1 point of you", f"{result['within_1']}%")
            c2.metric("Within 2 points of you", f"{result['within_2']}%")
            c3.metric("Average gap", f"{result['average_gap']:g} points")
            st.caption(f"Compared {result['students']} students: the app was lower than you for "
                       f"{result['too_low']} and higher for {result['too_high']}. Biggest gaps first:")
            st.dataframe(pd.DataFrame(result["gaps"], columns=["student", "app minus you"]),
                         use_container_width=True)
