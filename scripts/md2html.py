"""Render a Markdown doc to a styled, self-contained HTML view next to it.

Usage:
    .venv/Scripts/python scripts/md2html.py ROADMAP.md [more.md ...]
    .venv/Scripts/python scripts/md2html.py --check ROADMAP.md   # exit 1 if HTML is stale

The HTML is GENERATED OUTPUT. Never hand-edit it; edit the .md and re-run.
Requires pandoc on PATH (or at the default Windows install location).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CSS_PATH = REPO_ROOT / "docs" / "assets" / "md.css"
PANDOC_FALLBACKS = [Path("C:/Program Files/Pandoc/pandoc.exe")]

# Mermaid diagrams (```mermaid blocks; required in specs since 2026-10-23, see WORKFLOW.md). pandoc emits
# <pre class="mermaid"><code>...</code></pre>; mermaid wants the bare diagram text, so unwrap it first.
MERMAID = """<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>
document.querySelectorAll("pre.mermaid").forEach(function (pre) {
  var div = document.createElement("div");
  div.className = "mermaid";
  div.textContent = pre.textContent;
  pre.replaceWith(div);
});
mermaid.initialize({startOnLoad: false, securityLevel: "loose"});
mermaid.run({querySelector: "div.mermaid"});
</script>"""

BANNER = (
    '<div class="md-banner">Generated from <a href="{md}">{md}</a> by '
    "scripts/md2html.py &middot; do not edit this HTML by hand</div>"
)


def find_pandoc() -> str:
    found = shutil.which("pandoc")
    if found:
        return found
    for candidate in PANDOC_FALLBACKS:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("pandoc not found on PATH; install from https://pandoc.org")


def render(md_path: Path, pandoc: str) -> Path:
    """Convert md_path to a sibling .html with embedded CSS. Returns the HTML path."""
    html_path = md_path.with_suffix(".html")
    css = CSS_PATH.read_text(encoding="utf-8")
    header = f"<style>\n{css}\n</style>"
    before = BANNER.format(md=md_path.name) + '<div class="md-wrap">'
    after = "</div>" + (MERMAID if "```mermaid" in md_path.read_text(encoding="utf-8") else "")
    cmd = [
        pandoc,
        str(md_path),
        "--from", "gfm+task_lists",
        "--to", "html5",
        "--standalone",
        "--toc", "--toc-depth=2",
        "--metadata", "pagetitle=" + md_path.stem,
        "--output", str(html_path),
    ]
    # pandoc reads header/before/after blocks from files, so stage them in-memory via stdin-free temp files.
    tmp_dir = html_path.parent
    header_f = tmp_dir / f".{md_path.stem}.header.tmp.html"
    before_f = tmp_dir / f".{md_path.stem}.before.tmp.html"
    after_f = tmp_dir / f".{md_path.stem}.after.tmp.html"
    try:
        header_f.write_text(header, encoding="utf-8")
        before_f.write_text(before, encoding="utf-8")
        after_f.write_text(after, encoding="utf-8")
        cmd += [
            "--include-in-header", str(header_f),
            "--include-before-body", str(before_f),
            "--include-after-body", str(after_f),
        ]
        subprocess.run(cmd, check=True)
    finally:
        for f in (header_f, before_f, after_f):
            f.unlink(missing_ok=True)
    logger.info("rendered %s -> %s", md_path, html_path)
    return html_path


def is_stale(md_path: Path) -> bool:
    html_path = md_path.with_suffix(".html")
    return (not html_path.exists()) or html_path.stat().st_mtime < md_path.stat().st_mtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("md_files", nargs="+", type=Path, help="Markdown files to render")
    parser.add_argument("--check", action="store_true", help="Only report whether the HTML is older than the .md")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    md_files = [p if p.is_absolute() else REPO_ROOT / p for p in args.md_files]
    missing = [p for p in md_files if not p.exists()]
    if missing:
        logger.error("not found: %s", ", ".join(str(p) for p in missing))
        return 2

    if args.check:
        stale = [p for p in md_files if is_stale(p)]
        for p in stale:
            logger.warning("STALE: %s (re-run scripts/md2html.py %s)", p.with_suffix(".html").name, p.name)
        return 1 if stale else 0

    pandoc = find_pandoc()
    for p in md_files:
        render(p, pandoc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
