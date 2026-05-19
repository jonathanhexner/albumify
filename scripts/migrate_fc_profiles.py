"""spec-040 Phase 5b — migrate legacy FC App profiles to the v2 step-config shape.

Legacy profiles in ``~/.sim_bench/profiles/<name>.json`` are a flat dict
of ``FCConfig`` fields (``K``, ``distance_threshold``, ``merge_enabled``,
``cluster_diameter_cap_enabled``, ...). The v2 FC App expects
``step_configs[<step_name>] = {<flat_dict>}`` — each unified step picks
its own relevant keys via ``_build_fc_config``.

This script reshapes legacy profiles in place (writes ``<name>.json``
adjacent to a ``<name>.v1.json`` backup), so the v2 app can load them
unmodified. Idempotent: a v2-shaped profile is left alone.

Usage:
    .venv/Scripts/python scripts/migrate_fc_profiles.py
    .venv/Scripts/python scripts/migrate_fc_profiles.py --dir /custom/path
    .venv/Scripts/python scripts/migrate_fc_profiles.py --dry-run

Idempotence test: re-running on a migrated profile is a no-op.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".sim_bench" / "profiles"


def is_v2_shape(profile: Dict[str, Any]) -> bool:
    """A v2 profile has a 'step_configs' key whose value is a dict of dicts."""
    sc = profile.get("step_configs")
    if not isinstance(sc, dict) or not sc:
        return False
    return any(isinstance(v, dict) for v in sc.values())


def reshape_v1_to_v2(legacy: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a flat legacy profile into the v2 step_configs shape.

    Every unified clustering step gets the same flat dict — each step's
    ``_build_fc_config`` picks only the keys it knows about, so unused
    keys are harmless. We also preserve the legacy flat dict under
    ``legacy_flat`` so a future re-export back to v1 is possible.
    """
    return {
        "version": 2,
        "step_configs": {name: dict(legacy) for name in UNIFIED_CLUSTERING_STEPS},
        "legacy_flat": dict(legacy),
    }


def migrate_one(path: Path, dry_run: bool = False) -> str:
    """Return one of 'migrated', 'already_v2', 'invalid'."""
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Skipping %s — could not parse: %s", path.name, e)
        return "invalid"
    if not isinstance(profile, dict):
        logger.warning("Skipping %s — top-level is not a dict", path.name)
        return "invalid"
    if is_v2_shape(profile):
        logger.info("Skipping %s — already v2 shape", path.name)
        return "already_v2"
    reshaped = reshape_v1_to_v2(profile)
    if dry_run:
        logger.info("[dry-run] would migrate %s (%d step configs)",
                    path.name, len(reshaped["step_configs"]))
        return "migrated"
    # Backup, then overwrite.
    backup = path.with_suffix(".v1.json")
    if not backup.exists():
        backup.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    path.write_text(json.dumps(reshaped, indent=2), encoding="utf-8")
    logger.info("Migrated %s (backup at %s)", path.name, backup.name)
    return "migrated"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=_DEFAULT_DIR,
                        help=f"Profiles directory (default: {_DEFAULT_DIR})")
    parser.add_argument("--dry-run", action="store_true", help="Report only; don't write.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.dir.exists():
        logger.info("Profiles directory %s does not exist; nothing to migrate.", args.dir)
        return 0

    files = sorted(args.dir.glob("*.json"))
    files = [f for f in files if not f.name.endswith(".v1.json")]
    if not files:
        logger.info("No profiles found in %s.", args.dir)
        return 0

    counts = {"migrated": 0, "already_v2": 0, "invalid": 0}
    for f in files:
        counts[migrate_one(f, dry_run=args.dry_run)] += 1

    logger.info("Done: %d migrated, %d already-v2, %d invalid.",
                counts["migrated"], counts["already_v2"], counts["invalid"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
