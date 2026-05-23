"""spec-041 — migrate FC App profiles to the FCParams JSON shape.

Two legacy shapes exist on disk:

* **Flat v1** — a plain dict of ``FCConfig`` fields written by the original
  FC App. Example: ``{"K": 5, "distance_threshold": 0.35, ...}``.
* **spec-040 v2 shape** — the transitional ``{"version": 2, "step_configs":
  {<step_name>: {<flat_dict>}, ...}, "legacy_flat": {...}}`` shape written
  by the previous incarnation of this script.

After spec-041, the canonical shape is the JSON output of
``FCParams.model_dump_json()`` — a flat dict of every knob with full
type/range validation. This script reads either legacy form, validates it
through ``FCParams``, and rewrites the file. The original content is
backed up to ``<name>.v1.json`` before overwrite.

Idempotent: when a ``.v1.json`` backup is already present the file is
treated as already migrated and left alone.

Usage:
    .venv/Scripts/python scripts/migrate_fc_profiles.py
    .venv/Scripts/python scripts/migrate_fc_profiles.py --dir /custom/path
    .venv/Scripts/python scripts/migrate_fc_profiles.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError

from face_cluster.fc_params import FCParams

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".sim_bench" / "profiles"


def _extract_flat(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a flat FCConfig-shaped dict from either legacy shape."""
    sc = data.get("step_configs")
    if isinstance(sc, dict) and sc:
        sample = next(iter(sc.values()))
        if isinstance(sample, dict):
            return sample
    return data


def migrate_one(path: Path, dry_run: bool = False) -> str:
    """Return one of 'migrated', 'already_v2', 'invalid'."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Skipping %s — could not parse: %s", path.name, e)
        return "invalid"
    if not isinstance(data, dict):
        logger.warning("Skipping %s — top-level is not a dict", path.name)
        return "invalid"

    backup = path.with_suffix(".v1.json")
    if backup.exists():
        logger.info("Skipping %s — backup already present", path.name)
        return "already_v2"

    flat = _extract_flat(data)
    try:
        params = FCParams.model_validate(flat)
    except ValidationError as e:
        logger.warning("Skipping %s — does not validate as FCParams: %s", path.name, e)
        return "invalid"

    if dry_run:
        logger.info("[dry-run] would migrate %s", path.name)
        return "migrated"

    backup.write_text(json.dumps(data, indent=2), encoding="utf-8")
    params.save(path)
    logger.info("Migrated %s (backup at %s)", path.name, backup.name)
    return "migrated"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=_DEFAULT_DIR,
                        help=f"Profiles directory (default: {_DEFAULT_DIR})")
    parser.add_argument("--dry-run", action="store_true", help="Report only; don't write.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.dir.exists():
        logger.info("Profiles directory %s does not exist; nothing to migrate.", args.dir)
        return 0

    files = [f for f in sorted(args.dir.glob("*.json")) if not f.name.endswith(".v1.json")]
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
