"""Lint a PRD (``specs/NNN-*/spec.md``) against the WORKFLOW.md format and against the code it describes.

Usage:
    .venv/Scripts/python scripts/prd_lint.py specs/115-album-workbench/spec.md [--render]

Errors (exit 1):
  * sections "## 1. Objective" .. "## 6. Modified files" present, in order; the legend is present
  * objective at most 3 lines; at most 15 requirement rows
  * every requirement row has a status glyph (● ◐ ○) and a check that names evidence: a backticked file or
    test, or a real-data result ("real ...") -- bare "tests" / "Playwright" fails
  * every requirement id appears in the plan diagram (ranges like R1-R5 count)
  * new files (section 5) do not exist yet unless marked ●; modified files (section 6) exist
  * class-diagram members marked changed (✎), removed (✖) or unmarked must exist on that class in the code
    (Python AST); a class that is not in the code may only have new (✚) members
Warnings: a requirement no file in sections 5/6 serves; with --render, a desktop page over 6,000 px.
``--render`` also opens the generated HTML in a browser and checks every diagram drew without errors.
"""
from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ("sim_bench", "app", "face_cluster", "scripts")
SECTIONS = ("1. Objective", "2. Requirements", "3. Plan", "4. Class diagram", "5. New files", "6. Modified files")
MAX_OBJECTIVE_LINES, MAX_REQUIREMENTS, MAX_DESKTOP_PX = 3, 15, 6000
STATUS = ("●", "◐", "○")
NEW, CHANGED, REMOVED = "✚", "✎", "✖"
SKIP_STEREOTYPES = ("page", "api")          # not Python classes: a Streamlit page, a FastAPI router


@dataclass
class LintResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _sections(text: str) -> Dict[str, str]:
    parts = re.split(r"^## ", text, flags=re.M)
    return {p.split("\n", 1)[0].strip(): p.split("\n", 1)[1] if "\n" in p else "" for p in parts[1:]}


def _table_rows(body: str) -> List[List[str]]:
    rows = [l for l in body.splitlines() if l.startswith("|")]
    return [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]   # skip header + rule


def _mermaid_blocks(body: str) -> List[str]:
    return re.findall(r"```mermaid\n(.*?)```", body, flags=re.S)


def _req_ids(text: str) -> Set[str]:
    ids = set()
    for a, b in re.findall(r"R(\d+)\s*-\s*R(\d+)", text):
        ids.update(f"R{i}" for i in range(int(a), int(b) + 1))
    ids.update(re.findall(r"\bR\d+\b", text))
    return ids


def _paths(cell: str) -> List[str]:
    """Backticked file paths in a table cell; ``a/{b,c}.py`` expands to ``a/b.py``, ``a/c.py``."""
    out = []
    for raw in re.findall(r"`([^`]+)`", cell):
        m = re.match(r"(.*)\{(.+)\}(.*)", raw)
        for n in ([m.group(1) + x + m.group(3) for x in m.group(2).split(",")] if m else [raw]):
            if "/" in n and "." in n.rsplit("/", 1)[-1]:
                out.append(n)
    return out


def code_index(root: Path) -> Dict[str, Set[str]]:
    """class name -> member names (methods, class attributes, ``self.x`` assigned in any method)."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for d in CODE_DIRS:
        for py in (root / d).rglob("*.py") if (root / d).is_dir() else []:
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                members = index[node.name]
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        members.add(item.name)
                        for sub in ast.walk(item):
                            if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                                    and sub.value.id == "self" and isinstance(sub.ctx, ast.Store)):
                                members.add(sub.attr)
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        members.add(item.target.id)
                    elif isinstance(item, ast.Assign):
                        members.update(t.id for t in item.targets if isinstance(t, ast.Name))
    return index


def _class_blocks(diagram: str):
    for m in re.finditer(r"^\s*class (\w+)\s*\{(.*?)^\s*\}", diagram, flags=re.S | re.M):
        yield m.group(1), [l.strip() for l in m.group(2).splitlines() if l.strip()]


def _member_names(line: str) -> List[str]:
    body = line.lstrip(NEW + CHANGED + REMOVED + "+-#~ ").rstrip("$*")
    if "(" in body:
        return [body.split("(", 1)[0].strip()]
    return [n.split(":", 1)[0].strip() for n in body.split(",") if n.strip()]


def check_diagram(diagram: str, index: Dict[str, Set[str]], res: LintResult) -> None:
    for cls, lines in _class_blocks(diagram):
        stereo = next((l.strip("<>").lower() for l in lines if l.startswith("<<")), "")
        if any(s in stereo for s in SKIP_STEREOTYPES):
            continue
        known = index.get(cls)
        for line in lines:
            if line.startswith("<<"):
                continue
            marker = line[0] if line[0] in (NEW, CHANGED, REMOVED) else ""
            if marker == NEW:
                continue
            kind = {CHANGED: "changed", REMOVED: "removed", "": "unchanged"}[marker]
            for name in _member_names(line):
                if known is None:
                    res.errors.append(f"class diagram: {cls} is not in the code, but '{name}' is marked {kind} "
                                      f"(mark new members with {NEW})")
                elif name not in known:
                    res.errors.append(f"class diagram: {cls}.{name} is marked {kind} but does not exist in the code")


def lint(spec: Path, root: Path = REPO_ROOT) -> LintResult:
    res = LintResult()
    text = spec.read_text(encoding="utf-8")
    secs = _sections(text)
    found = [s for s in secs if s in SECTIONS]
    if found != list(SECTIONS):
        res.errors.append(f"sections must be, in order: {', '.join(SECTIONS)}; found: {', '.join(found) or 'none'}")
    if 'class="legend"' not in text:
        res.errors.append("legend line missing (copy it from the WORKFLOW.md template)")

    objective = [l for l in secs.get(SECTIONS[0], "").splitlines() if l.strip()]
    if len(objective) > MAX_OBJECTIVE_LINES:
        res.errors.append(f"objective is {len(objective)} lines (max {MAX_OBJECTIVE_LINES})")

    reqs = [r for r in _table_rows(secs.get(SECTIONS[1], "")) if r and re.fullmatch(r"R\d+", r[0])]
    if len(reqs) > MAX_REQUIREMENTS:
        res.errors.append(f"{len(reqs)} requirements (max {MAX_REQUIREMENTS}); split the spec")
    for r in reqs:
        if len(r) < 4:
            res.errors.append(f"{r[0]}: expected columns # | Status | Requirement | How we know it works")
            continue
        if not r[1].startswith(STATUS):
            res.errors.append(f"{r[0]}: status must start with one of {' '.join(STATUS)}")
        if not (re.search(r"`[^`]*(\.py|::|\.md|\.json)[^`]*`", r[3]) or re.search(r"\breal\b", r[3])):
            res.errors.append(f"{r[0]}: the check must name a test file / test, or a real-data result")

    plan = "\n".join(_mermaid_blocks(secs.get(SECTIONS[2], "")))
    in_plan = _req_ids(plan)
    for r in reqs:
        if r[0] not in in_plan:
            res.errors.append(f"{r[0]} is not closed by any check in the plan diagram")

    file_reqs: Set[str] = set()
    for row in _table_rows(secs.get(SECTIONS[4], "")):
        status = row[-1] if row else ""
        for p in _paths(row[0] if row else ""):
            exists = (root / p).exists()
            if exists and not status.startswith("●"):
                res.errors.append(f"new file {p} already exists but is not marked ● done")
            if not exists and status.startswith("●"):
                res.errors.append(f"new file {p} is marked ● done but does not exist")
    for row in _table_rows(secs.get(SECTIONS[5], "")):
        for p in _paths(row[0] if row else ""):
            if not (root / p).exists():
                res.errors.append(f"modified file {p} does not exist")
        file_reqs |= _req_ids(" ".join(row[1:]))
    for r in reqs:
        if r[0] not in file_reqs:
            res.warnings.append(f"{r[0]}: no modified file names it in its Req column")

    index = code_index(root)
    for block in _mermaid_blocks(secs.get(SECTIONS[3], "")):
        check_diagram(block, index, res)
    return res


def render_check(spec: Path, res: LintResult) -> None:
    from playwright.sync_api import sync_playwright

    html = spec.with_suffix(".html")
    if not html.is_file() or html.stat().st_mtime < spec.stat().st_mtime:
        res.errors.append(f"{html.name} is missing or older than {spec.name}: run scripts/md2html.py first")
        return
    n_blocks = spec.read_text(encoding="utf-8").count("```mermaid")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors: List[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(html.resolve().as_uri())
        if n_blocks:
            page.wait_for_selector("div.mermaid svg", timeout=30000)
            page.wait_for_timeout(1000)
        drawn = page.locator("div.mermaid svg").count()
        height = page.evaluate("document.documentElement.scrollHeight")
        browser.close()
    if drawn != n_blocks:
        res.errors.append(f"{drawn} of {n_blocks} diagrams drew")
    if errors:
        res.errors.append(f"browser errors: {errors[:3]}")
    if height > MAX_DESKTOP_PX:
        res.warnings.append(f"desktop page is {height} px tall (aim for {MAX_DESKTOP_PX} or less)")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", type=Path)
    ap.add_argument("--render", action="store_true", help="also check the generated HTML in a browser")
    args = ap.parse_args(argv)
    spec = args.spec if args.spec.is_absolute() else REPO_ROOT / args.spec
    res = lint(spec)
    if args.render:
        render_check(spec, res)
    for w in res.warnings:
        print(f"WARN  {w}")
    for e in res.errors:
        print(f"ERROR {e}")
    print(f"{spec.relative_to(REPO_ROOT) if spec.is_relative_to(REPO_ROOT) else spec}: "
          f"{len(res.errors)} errors, {len(res.warnings)} warnings")
    return 1 if res.errors else 0


if __name__ == "__main__":
    sys.exit(main())
