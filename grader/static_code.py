"""Static facts about submitted code. Code is parsed, never executed."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .models import ExtractionReport

CODE_EXTS = {".py", ".java", ".c", ".cpp", ".h", ".hpp", ".js", ".ts", ".r", ".sql", ".m", ".sh", ".ipynb"}


def analyze(reports: list[ExtractionReport]) -> list[dict]:
    out = []
    for r in reports:
        ext = Path(r.file).suffix.lower()
        if ext not in CODE_EXTS or not r.readable:
            continue
        text = r.text
        info = {"file": r.file, "lines": len(text.splitlines()),
                "todo_markers": len(re.findall(r"\b(TODO|FIXME|XXX)\b", text))}
        if ext == ".py":
            try:
                tree = ast.parse(text)
                info["syntax_ok"] = True
                info["functions"] = sorted({n.name for n in ast.walk(tree)
                                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))})
                info["classes"] = sorted({n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)})
                imports = set()
                for n in ast.walk(tree):
                    if isinstance(n, ast.Import):
                        imports.update(a.name.split(".")[0] for a in n.names)
                    elif isinstance(n, ast.ImportFrom) and n.module:
                        imports.add(n.module.split(".")[0])
                info["imports"] = sorted(imports)
            except SyntaxError as exc:
                info["syntax_ok"] = False
                info["syntax_error"] = f"line {exc.lineno}: {exc.msg}"
        out.append(info)
    return out


def summary_text(facts: list[dict]) -> str:
    lines = []
    for f in facts:
        parts = [f"{f['file']}: {f['lines']} lines"]
        if "syntax_ok" in f:
            parts.append("parses without syntax errors" if f["syntax_ok"] else f"SYNTAX ERROR {f['syntax_error']}")
        if f.get("functions"):
            parts.append("functions: " + ", ".join(f["functions"]))
        if f.get("classes"):
            parts.append("classes: " + ", ".join(f["classes"]))
        if f["todo_markers"]:
            parts.append(f"{f['todo_markers']} TODO/FIXME marker(s)")
        lines.append("; ".join(parts))
    return "\n".join(lines)
