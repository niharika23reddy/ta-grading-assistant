"""Text extraction with explicit quality warnings.

Nothing here executes student files. Code and notebooks are read as plain text.
Pages that cannot be read are reported, never silently dropped, so the grader
can mark related requirements "Unable to verify" instead of "Missing".
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .models import ExtractionReport, PageText

TEXT_EXTS = {
    ".txt", ".md", ".py", ".java", ".c", ".cpp", ".h", ".hpp", ".js", ".ts",
    ".sql", ".r", ".csv", ".tsv", ".json", ".html", ".css", ".m", ".sh", ".yaml", ".yml",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
MIN_PAGE_CHARS = 25  # below this a PDF page is treated as image-only / scanned
MAX_ZIP_MEMBERS = 200
MAX_ZIP_BYTES = 100 * 1024 * 1024


def ocr_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def _ocr_image(img) -> str:
    import pytesseract

    return pytesseract.image_to_string(_prepare_for_ocr(img))


def _prepare_for_ocr(img):
    """Tesseract reads dark text on a light background best, at a decent size. Terminal
    screenshots are often light text on a dark background, and small once pasted."""
    from PIL import Image, ImageOps

    gray = ImageOps.grayscale(img.convert("RGB"))
    hist = gray.histogram()
    if sum(hist[:128]) > sum(hist[128:]):  # mostly dark: invert
        gray = ImageOps.invert(gray)
    if gray.width < 1200:
        gray = gray.resize((gray.width * 2, gray.height * 2), Image.LANCZOS)
    return gray


def _ocr_page(name: str, page_no: int | None, img, ocr: bool) -> PageText:
    if not ocr:
        return PageText(name, page_no, "", "none",
                        ["Image-only content; OCR is off, so it was not read. TA must check it."])
    if not ocr_available():
        return PageText(name, page_no, "", "none",
                        ["Image-only content; Tesseract is not installed, so it was not read. TA must check it."])
    text = _ocr_image(img).strip()
    if len(text) < MIN_PAGE_CHARS:
        return PageText(name, page_no, text, "ocr",
                        ["OCR found little or no text (handwriting, diagram or poor scan). TA must check it."])
    return PageText(name, page_no, text, "ocr",
                    ["Read with OCR; wording may contain recognition errors. Verify quoted evidence."])


MIN_IMAGE_PT = 40  # smaller pictures (bullets, icons, logos) are ignored
OCR_RENDER_SCALE = 3  # about 216 dpi when a page has to be rendered for OCR


def extract_pdf(name: str, data: bytes, ocr: bool) -> ExtractionReport:
    import pdfplumber

    pages: list[PageText] = []
    k = 0  # screenshot number across the whole file
    prev_line = ""  # last line of readable text before the current page
    use_ocr = ocr and ocr_available()
    try:
        doc = _open_pdfium(data)
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                images = _pdf_images(doc, i - 1, float(page.height))
                if len(text) >= MIN_PAGE_CHARS and not images:
                    pages.append(PageText(name, i, text, "text", []))
                    prev_line = _last_text_line(text)
                    continue
                outcomes: dict[str, list[int]] = {"read": [], "little": [], "unread": []}
                if len(text) < MIN_PAGE_CHARS:
                    # the whole page is pictures: a scan, or screenshots saved as their own page.
                    # Tie them to the exercise text they follow on the previous page.
                    lines = [f"[Screenshot {k + 1} follows: {prev_line[:100]}]"] if prev_line and not text else []
                    lines += [text] if text else []
                    for im in images or [None]:
                        k += 1
                        if not use_ocr:
                            picture = None
                        elif im is None:  # no embedded picture (vector page): OCR the rendered page
                            picture = _render_pdf_page(data, i - 1, OCR_RENDER_SCALE)
                        else:
                            picture = im["image"]()
                        shot, outcome = _ocr_screenshot(k, picture, ocr)
                        lines += shot
                        outcomes[outcome].append(k)
                    warns = ["Page has little or no selectable text (likely scanned or a screenshot)."]
                    method = "none" if not outcomes["read"] and not outcomes["little"] else "ocr"
                    pages.append(PageText(name, i, "\n".join(lines), method, warns + _image_warnings(outcomes, ocr)))
                    prev_line = _last_text_line(text) or prev_line
                    continue
                lines, k = _pdf_page_with_images(page, images, use_ocr, ocr, k, outcomes)
                pages.append(PageText(name, i, "\n".join(lines), "text+ocr", _image_warnings(outcomes, ocr)))
                prev_line = _last_text_line(text)
    except Exception as exc:  # corrupt or encrypted PDFs
        return ExtractionReport(name, "pdf", [], False, [f"PDF could not be opened: {exc}"])
    return _report(name, "pdf", pages)


def _open_pdfium(data: bytes):
    try:
        import pypdfium2 as pdfium

        return pdfium.PdfDocument(data)
    except Exception:
        return None


def _pdf_images(doc, index: int, page_height: float) -> list[dict]:
    """Pictures on one page, top to bottom, each with a loader for its original pixels.

    Reading the embedded picture at its own resolution OCRs far better than a rendered
    page, where a pasted desktop screenshot's terminal text shrinks to a few pixels."""
    if doc is None:
        return []
    try:
        import pypdfium2.raw as pdfium_c

        objects = list(doc[index].get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=5))
    except Exception:
        return []
    out = []
    for obj in objects:
        try:
            left, bottom, right, top = obj.get_bounds()
        except Exception:
            continue
        if right - left < MIN_IMAGE_PT or top - bottom < MIN_IMAGE_PT:
            continue
        out.append({"x0": left, "x1": right, "top": page_height - top, "bottom": page_height - bottom,
                    "image": lambda obj=obj: _pdf_image_pixels(obj)})
    return sorted(out, key=lambda im: im["top"])


def _pdf_image_pixels(obj):
    for render in (False, True):  # raw pixels first; rendered applies masks and odd colour spaces
        try:
            return obj.get_bitmap(render=render).to_pil()
        except Exception:
            continue
    return None


def _pdf_page_with_images(page, images, use_ocr: bool, ocr: bool, k: int, outcomes: dict[str, list[int]]):
    """Text and pictures in reading order: the text above each picture, then the picture's OCR text."""
    lines: list[str] = []
    top = 0.0
    for im in images:
        lines.append(_crop_text(page, top, im["top"]))
        k += 1
        shot, outcome = _ocr_screenshot(k, im["image"]() if use_ocr else None, ocr)
        lines.extend(shot)
        outcomes[outcome].append(k)
        top = max(top, im["bottom"])
    lines.append(_crop_text(page, top, page.height))
    return [ln for ln in lines if ln], k


def _last_text_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _crop_text(page, top: float, bottom: float) -> str:
    top, bottom = max(0.0, top), min(float(page.height), bottom)
    if bottom - top < 1:
        return ""
    try:
        return (page.crop((0, top, page.width, bottom), strict=False).extract_text() or "").strip()
    except Exception:
        return ""


def _render_pdf_page(data: bytes, index: int, scale: float = 2):
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(data)
        return doc[index].render(scale=scale).to_pil()
    except Exception:
        return None


SCREENSHOT_OPEN = "[Screenshot {k}: OCR text of the image begins]"
SCREENSHOT_CLOSE = "[Screenshot {k}: OCR text of the image ends]"


def _element_images(element, part) -> list[bytes]:
    """Pictures inside one paragraph or table, in order (DrawingML and older VML)."""
    rids = element.xpath(".//a:blip/@r:embed") + element.xpath('.//*[local-name()="imagedata"]/@r:id')
    blobs = []
    for rid in rids:
        try:
            blobs.append(part.related_parts[rid].blob)
        except Exception:
            blobs.append(b"")
    return blobs


def _paragraph_line(para) -> str:
    # Word's automatic numbering and bullets are not part of the paragraph text,
    # so mark list items explicitly; checklist drafting relies on them.
    style = (para.style.name if para.style is not None else "") or ""
    is_list = style.lower().startswith("list") or para._p.pPr is not None and para._p.pPr.numPr is not None
    return f"- {para.text}" if is_list and para.text.strip() else para.text


def _screenshot_lines(k: int, blob: bytes, ocr: bool) -> tuple[list[str], str]:
    """Inline text for one embedded picture, plus its outcome: "read", "little", "unread"."""
    from PIL import Image

    if not (ocr and ocr_available()):
        return _ocr_screenshot(k, None, ocr)
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
    except Exception:
        return [f"[Screenshot {k}: image present; its format could not be opened]"], "unread"
    return _ocr_screenshot(k, img, ocr)


def _ocr_screenshot(k: int, img, ocr: bool) -> tuple[list[str], str]:
    if not (ocr and ocr_available()):
        return [f"[Screenshot {k}: image present; not read because OCR is unavailable]"], "unread"
    if img is None:
        return [f"[Screenshot {k}: image present; it could not be rendered]"], "unread"
    text = _ocr_image(img).strip()
    if len(text) < MIN_PAGE_CHARS:
        body = [text] if text else []
        return [f"[Screenshot {k}: image present; OCR found little or no text]", *body], "little"
    return [SCREENSHOT_OPEN.format(k=k), text, SCREENSHOT_CLOSE.format(k=k)], "read"


def _image_warnings(outcomes: dict[str, list[int]], ocr: bool) -> list[str]:
    warnings = []
    if outcomes["read"]:
        warnings.append(f"{len(outcomes['read'])} screenshot(s) read with OCR; wording may contain recognition "
                        "errors. Verify quoted evidence.")
    if outcomes["little"]:
        warnings.append(f"Screenshot(s) {_numbers(outcomes['little'])}: OCR found little or no text (diagram, "
                        "photo or blurry image). TA must check them.")
    if outcomes["unread"]:
        why = "OCR is off" if not ocr else "Tesseract is not installed or the image could not be opened"
        warnings.append(f"Document contains {len(outcomes['unread'])} image(s) (screenshots, figures) that were not "
                        f"read because {why}. TA must check them.")
    return warnings


def extract_docx(name: str, data: bytes, ocr: bool = True) -> ExtractionReport:
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        return ExtractionReport(name, "docx", [], False, [f"DOCX could not be opened: {exc}"])
    part = document.part
    lines: list[str] = []
    outcomes: dict[str, list[int]] = {"read": [], "little": [], "unread": []}

    def add_images(element):
        # Screenshots stay where the student pasted them, next to the exercise they belong to.
        for blob in _element_images(element, part):
            k = sum(len(v) for v in outcomes.values()) + 1
            shot, outcome = _screenshot_lines(k, blob, ocr)
            lines.extend(shot)
            outcomes[outcome].append(k)

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            lines.append(_paragraph_line(Paragraph(child, document)))
            add_images(child)
        elif child.tag == qn("w:tbl"):
            for row in Table(child, document).rows:
                lines.append(" | ".join(cell.text.strip() for cell in row.cells))
            add_images(child)
    text = "\n".join(lines).strip()

    warnings = _image_warnings(outcomes, ocr)
    method = "text+ocr" if outcomes["read"] or outcomes["little"] else "text"
    return _report(name, "docx", [PageText(name, None, text, method, warnings)])


def _numbers(ks: list[int]) -> str:
    return ", ".join(str(k) for k in ks)


def extract_text(name: str, data: bytes) -> ExtractionReport:
    warnings = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        warnings.append("File is not valid UTF-8; some characters were replaced.")
    if "\x00" in text:
        return ExtractionReport(name, "binary", [], False, ["File looks binary, not text. It was not read."])
    return _report(name, Path(name).suffix.lstrip(".").lower() or "text",
                   [PageText(name, None, text, "text", warnings)])


def extract_ipynb(name: str, data: bytes) -> ExtractionReport:
    try:
        nb = json.loads(data.decode("utf-8"))
    except Exception as exc:
        return ExtractionReport(name, "ipynb", [], False, [f"Notebook could not be parsed: {exc}"])
    parts, warnings, n_images = [], [], 0
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        src = "".join(cell.get("source", []))
        parts.append(f"# [cell {idx}: {cell.get('cell_type', '?')}]\n{src}")
        for out in cell.get("outputs", []):
            if "text" in out:
                parts.append("# [output]\n" + "".join(out["text"]))
            data_out = out.get("data", {})
            if "text/plain" in data_out:
                parts.append("# [output]\n" + "".join(data_out["text/plain"]))
            if any(k.startswith("image/") for k in data_out):
                n_images += 1
    if n_images:
        warnings.append(f"Notebook has {n_images} image output(s) (plots) that were not read. TA must check them.")
    warnings.append("Notebook outputs are as saved by the student; the code was not re-run.")
    return _report(name, "ipynb", [PageText(name, None, "\n".join(parts), "text", warnings)])


def extract_image(name: str, data: bytes, ocr: bool) -> ExtractionReport:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        return ExtractionReport(name, "image", [], False, [f"Image could not be opened: {exc}"])
    return _report(name, "image", [_ocr_page(name, None, img, ocr)])


def extract_zip(name: str, data: bytes, ocr: bool) -> list[ExtractionReport]:
    reports = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return [ExtractionReport(name, "zip", [], False, ["ZIP archive is corrupt."])]
    members = [m for m in zf.infolist() if not m.is_dir()]
    if len(members) > MAX_ZIP_MEMBERS or sum(m.file_size for m in members) > MAX_ZIP_BYTES:
        return [ExtractionReport(name, "zip", [], False, ["ZIP archive is too large to unpack safely."])]
    for m in members:
        path = PurePosixPath(m.filename)
        if path.is_absolute() or ".." in path.parts or path.name.startswith(".") or "__MACOSX" in path.parts:
            continue
        reports.extend(extract_file(f"{name}/{m.filename}", zf.read(m), ocr))
    return reports


def extract_file(name: str, data: bytes, ocr: bool = True) -> list[ExtractionReport]:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return [extract_pdf(name, data, ocr)]
    if ext == ".docx":
        return [extract_docx(name, data, ocr)]
    if ext == ".ipynb":
        return [extract_ipynb(name, data)]
    if ext in IMAGE_EXTS:
        return [extract_image(name, data, ocr)]
    if ext == ".zip":
        return extract_zip(name, data, ocr)
    if ext in TEXT_EXTS:
        return [extract_text(name, data)]
    return [ExtractionReport(name, ext.lstrip(".") or "unknown", [], False,
                             [f"Unsupported file type '{ext or 'none'}'. It was not read. TA must check it."])]


def _report(name: str, file_type: str, pages: list[PageText]) -> ExtractionReport:
    readable = any(p.text.strip() for p in pages)
    warnings = [] if readable else ["No readable text was found in this file."]
    return ExtractionReport(name, file_type, pages, readable, warnings)


def unreadable_parts(reports: list[ExtractionReport]) -> list[str]:
    """Human-readable list of anything that could not be read fully."""
    out = []
    for r in reports:
        for w in r.warnings:
            out.append(f"{r.file}: {w}")
        for p in r.pages:
            where = p.file if p.page is None else f"{p.file} page {p.page}"
            for w in p.warnings:
                out.append(f"{where}: {w}")
    return out


def has_unread_content(reports: list[ExtractionReport]) -> bool:
    """True when some content may exist that the grader could not read."""
    for r in reports:
        if not r.readable:
            return True
        for p in r.pages:
            if p.method == "none" or any("not read" in w or "OCR found little" in w for w in p.warnings):
                return True
    return False
