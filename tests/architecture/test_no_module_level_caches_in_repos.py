"""spec-048 — no module-level mutable cache dicts in face_cluster/repositories/.

Catches regressions of spec-046 smell S5 (the ``_engine_cache`` dict)
where state was held across Repository instances at module scope. If
caching ever becomes a real need, do it via a properly-scoped factory
that's easy to reason about and override in tests — not a module dict.
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOS_DIR = PROJECT_ROOT / "face_cluster" / "repositories"

# Matches module-scope (zero-indent) assignments of empty dicts to names
# ending in `_cache`. Tolerates optional type annotation.
_PATTERN = re.compile(r"""^(_\w+_cache)\s*(:\s*[^=]+)?=\s*\{\}\s*$""")


def test_no_module_level_cache_dicts() -> None:
    offenders: list[str] = []
    for py in REPOS_DIR.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _PATTERN.match(line):
                offenders.append(f"{py.relative_to(PROJECT_ROOT)}:{lineno}  {line.strip()}")
    assert not offenders, (
        "Module-level mutable cache dicts are banned in repositories/. "
        "Use a scoped factory instead:\n  " + "\n  ".join(offenders)
    )
