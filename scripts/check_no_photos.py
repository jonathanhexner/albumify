"""Keep photos out of git (2026-10-16 photo purge, docs/project/photo_purge_list.md).

Fails when git would carry a photo, in any of the three forms found in the purge:

    image files     *.jpg / *.png / *.heic / ...  unless listed in .photo-allowlist (charts)
    notebooks       *.ipynb with saved cell outputs (outputs embed images)
    text pages      *.html / *.htm / *.md / *.svg with an inlined base64 picture

    python scripts/check_no_photos.py            # every tracked file (CI)
    python scripts/check_no_photos.py --staged   # files staged for the next commit (pre-commit hook)
    python scripts/check_no_photos.py --install-hook

Photos belong on the data drive or in private storage; reports keep their sample images on disk, untracked.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOWLIST = REPO / ".photo-allowlist"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
TEXT_EXT = {".html", ".htm", ".md", ".svg"}
INLINE_IMAGE = re.compile(rb"data:image/[a-z+]+;base64,[A-Za-z0-9+/=]{200}")

HOOK = """#!/bin/sh
# Installed by scripts/check_no_photos.py --install-hook. Works from any worktree: the venv lives
# next to the main checkout's .git; a branch that predates the guard script is skipped (CI checks it).
root=$(cd "$(git rev-parse --git-common-dir)/.." && pwd)
py="$root/.venv/Scripts/python"; [ -x "$py" ] || py="$root/.venv/bin/python"
[ -f scripts/check_no_photos.py ] || { echo "photo guard: script not on this branch, skipped"; exit 0; }
exec "$py" scripts/check_no_photos.py --staged
"""


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def _allowed() -> set[str]:
    if not ALLOWLIST.exists():
        return set()
    return {ln.strip() for ln in ALLOWLIST.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")}


def _files(staged: bool) -> list[str]:
    out = (_git("diff", "--cached", "--name-only", "--diff-filter=ACMR") if staged
           else _git("ls-files"))
    return [f for f in out.splitlines() if f]


def _content(path: str) -> bytes:
    """The file as git stores it (the index), not the working copy: local notebooks may keep
    their outputs on disk, since the nbstripout filter clears them on the way into git."""
    return subprocess.run(["git", "show", f":{path}"], cwd=REPO, capture_output=True,
                          check=True).stdout


def _notebook_has_outputs(data: bytes) -> bool:
    try:
        nb = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return any(c.get("outputs") for c in nb.get("cells", []) if c.get("cell_type") == "code")


def find_problems(staged: bool) -> list[str]:
    allowed, problems = _allowed(), []
    for path in _files(staged):
        ext = Path(path).suffix.lower()
        if ext in IMAGE_EXT:
            if path not in allowed:
                problems.append(f"{path}: image file (charts go in .photo-allowlist)")
            continue
        if ext not in TEXT_EXT and ext != ".ipynb":
            continue
        try:
            data = _content(path)
        except (OSError, subprocess.CalledProcessError):
            continue
        if ext == ".ipynb" and _notebook_has_outputs(data):
            problems.append(f"{path}: notebook saved with outputs (nbstripout clears them)")
        elif ext in TEXT_EXT and INLINE_IMAGE.search(data):
            problems.append(f"{path}: page with an inlined base64 image")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staged", action="store_true", help="check the files staged for commit")
    ap.add_argument("--install-hook", action="store_true", help="install the git pre-commit hook")
    args = ap.parse_args(argv)
    if args.install_hook:
        hook = REPO / ".git" / "hooks" / "pre-commit"
        hook.write_text(HOOK, encoding="utf-8", newline="\n")
        hook.chmod(0o755)
        print(f"installed {hook}")
        return 0
    problems = find_problems(args.staged)
    for p in problems:
        print(f"PHOTO GUARD: {p}")
    if problems:
        print(f"{len(problems)} problem(s). Photos stay out of git: see docs/project/photo_purge_list.md")
        return 1
    print("photo guard: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
