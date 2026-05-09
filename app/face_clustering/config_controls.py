"""Config control rendering helpers (merge params, etc.).

Delegates to app/shared/merge_controls.py — single source of truth
for merge parameter UI shared with the main Album App.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure app/shared/ is importable
_shared_dir = str(Path(__file__).resolve().parents[1] / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)

from merge_controls import render_merge_params


def _render_merge_params(key_prefix: str = "rc_") -> dict:
    """Render merge parameter controls. Returns dict of PipelineConfig-valid params."""
    return render_merge_params(key_prefix=key_prefix)
