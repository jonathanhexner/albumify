"""spec-048 — `face_cluster/repositories/` must use the in-process
Alembic API (``alembic.command.upgrade``), never a subprocess shell-out
to ``alembic.exe``.

Also bans hardcoded ``.venv`` paths anywhere in ``face_cluster/`` —
those are non-portable and obstruct CI/macOS/Linux usage.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_CLUSTER = PROJECT_ROOT / "face_cluster"
REPOS_DIR = FACE_CLUSTER / "repositories"


def _py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "docs" not in p.parts]


def test_no_subprocess_import_in_repositories() -> None:
    pattern = re.compile(r"^\s*import\s+subprocess|^\s*from\s+subprocess\s+import")
    offenders: list[str] = []
    for py in _py_files(REPOS_DIR):
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{py.relative_to(PROJECT_ROOT)}:{lineno}  {line.strip()}")
    assert not offenders, (
        "subprocess is banned in face_cluster/repositories/ — use the "
        "in-process Alembic API (alembic.command.upgrade) instead:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "needle,reason",
    [
        ("alembic.exe", "Alembic CLI binary — use alembic.command instead"),
        (".venv", "Hardcoded venv path — non-portable, breaks CI/macOS/Linux"),
    ],
)
def test_no_hardcoded_venv_or_alembic_exe(needle: str, reason: str) -> None:
    offenders: list[str] = []
    for py in _py_files(FACE_CLUSTER):
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if needle in line and not line.lstrip().startswith("#"):
                offenders.append(f"{py.relative_to(PROJECT_ROOT)}:{lineno}  {line.strip()}")
    assert not offenders, f"{reason}:\n  " + "\n  ".join(offenders)
