import io
import shutil

import docx
import pytest
from PIL import Image, ImageDraw, ImageFont

from grader.checklist import Checklist, attach_instruction_context, find_conflicts, read_rubric_csv
from grader.extract import extract_file, has_unread_content
from grader.models import Decision
from grader.scoring import score

# Synthetic rubric in the same layout as a level-based LMS rubric export.
LEVEL_RUBRIC = """Criteria,Expected,Expected Pts,Satisfactory,Satisfactory Pts,Insufficient,Insufficient Pts,Max Pts
Lab Exercise 1,"1. Screenshot of the command and its output. 2. Description of the command.",2,Missing description or screenshot.,1,Missing both.,0,2
Lab Exercise 2,All tools installed with screenshots.,3,"Some tools missing; each tool is worth 0.5 points.",1.5,Nothing installed.,0,3
Total,,,,,,,5
"""


def dec(req_id, status):
    return Decision(req_id, "", status, "Correct", [], "", "", False)


def test_level_rubric_criteria_levels_and_total():
    r = read_rubric_csv(LEVEL_RUBRIC)
    c1, c2 = r.criteria
    assert (c1.description, c1.max_points, c1.levels) == ("Lab Exercise 1", 2.0, {"full": 2, "partial": 1, "none": 0})
    assert "Missing description or screenshot" in c1.notes
    assert c2.levels == {"full": 3, "partial": 1.5, "none": 0}
    assert r.total == 5


def test_level_rubric_drafts_linked_requirements_from_expected_column():
    r = read_rubric_csv(LEVEL_RUBRIC)
    assert [(q.req_id, q.criterion_id, q.description) for q in r.requirements] == [
        ("R1", "C1", "Lab Exercise 1: Screenshot of the command and its output"),
        ("R2", "C1", "Lab Exercise 1: Description of the command"),
        ("R3", "C2", "Lab Exercise 2: All tools installed with screenshots")]


def test_per_item_rule_is_flagged_and_partial_stays_provisional():
    r = read_rubric_csv(LEVEL_RUBRIC)
    cl = Checklist("X", "A", r.requirements, r.criteria)
    issues = find_conflicts(cl, rubric_total=r.total)
    assert len(issues) == 1 and "per-item" in issues[0] and "C2" in issues[0]
    s = score(r.criteria, r.requirements, [dec("R1", "Met"), dec("R2", "Missing"), dec("R3", "Partially met")])
    c1, c2 = s.criteria
    assert c1.suggested_points == 1 and not c1.provisional  # one of two items missing = Satisfactory
    assert c2.suggested_points == 1.5 and c2.provisional
    assert s.suggested_total == 2.5


def test_total_row_mismatch_reported():
    bad = LEVEL_RUBRIC.replace(",5\n", ",6\n")
    r = read_rubric_csv(bad)
    issues = find_conflicts(Checklist("X", "A", r.requirements, r.criteria), rubric_total=r.total)
    assert any("Total row says 6" in i for i in issues)


def test_template_rubric_still_supported():
    r = read_rubric_csv("criterion_id,description,max_points,full_points,partial_points,none_points\n"
                        "C1,Intro,2,2,,0\n")
    assert [(c.criterion_id, c.levels) for c in r.criteria] == [("C1", {"full": 2.0, "none": 0.0})]


def test_instruction_sections_become_requirement_notes():
    r = read_rubric_csv(LEVEL_RUBRIC)
    instructions = ("Week 2 lab\nLab Exercise 1\nRun `uname -a` and explain the output.\n"
                    "Lab Exercise 2\nInstall git, python and docker.\n")
    attach_instruction_context(r.requirements, r.criteria, instructions)
    assert "uname -a" in r.requirements[0].notes and "docker" not in r.requirements[0].notes
    assert "docker" in r.requirements[2].notes


def _docx_with_screenshot(text):
    img = Image.new("RGB", (900, 120), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 32)
    except OSError:
        font = ImageFont.load_default(size=32)
    draw.text((10, 40), text, fill="black", font=font)
    png = io.BytesIO()
    img.save(png, format="PNG")
    png.seek(0)
    d = docx.Document()
    d.add_paragraph("Exercise 1 screenshot:")
    d.add_picture(png)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_word_screenshots_flagged_when_ocr_off():
    [rep] = extract_file("lab.docx", _docx_with_screenshot("uname -a Linux ubuntu"), ocr=False)
    assert has_unread_content([rep])
    assert "1 image(s)" in rep.pages[0].warnings[0]


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs the tesseract OCR binary")
def test_word_screenshots_read_with_ocr():
    [rep] = extract_file("lab.docx", _docx_with_screenshot("uname -a Linux ubuntu 6.8"), ocr=True)
    image_page = rep.pages[1]
    assert image_page.file == "lab.docx (image 1)" and image_page.method == "ocr"
    assert "uname" in image_page.text and "Linux" in image_page.text
