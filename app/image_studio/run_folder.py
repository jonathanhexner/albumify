"""RunFolder (spec-094 Slice 5) — persist a studio run as a sub-folder.

A *run* = one execution: a set of methods scored over a folder, plus the
per-image ``AnalysisColumn`` results. RunFolder saves / lists / loads runs as
``<folder>/.studio_runs/<run_id>/``:

    run.json     — manifest (methods, image list, timestamps, spec version)
    columns.json — full render payload so Browse re-opens with NO recompute
    results.csv  — human / export artifact (image x method) + metadata mandate

PURE — no Streamlit. Scores themselves live in ``universal_cache``; this is the
run manifest + a rendering snapshot. A future DB-backed variant implements the
same ``save`` / ``list_runs`` / ``load`` and can replace this behind the app.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

RUNS_DIRNAME = ".studio_runs"
SPEC_VERSION = "spec-094"


def _runs_dir(folder: str) -> str:
    return os.path.join(folder, RUNS_DIRNAME)


def make_run_id(created_ts: str, methods: List[str]) -> str:
    """Deterministic run id: ``<created_ts>_<methods-slug>`` (ts passed in, not generated)."""
    slug = "-".join(methods)[:60] or "run"
    return f"{created_ts}_{slug}"


def save(folder: str, run_id: str, methods: List[str], paths: List[str],
         columns: Dict[str, Dict[str, object]], created_ts: str) -> str:
    """Write a run to ``<folder>/.studio_runs/<run_id>/``; return that dir.

    ``columns`` is ``{full_path: {method_key: AnalysisColumn}}`` (from
    ``engine.run_methods``). Stored keyed by BASENAME so a run stays valid if the
    album is reached via a different absolute path.
    """
    d = os.path.join(_runs_dir(folder), run_id)
    os.makedirs(d, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "source_folder": folder,
        "methods": list(methods),
        "images": [os.path.basename(p) for p in paths],
        "created_ts": created_ts,
        "spec_version": SPEC_VERSION,
        "n_images": len(paths),
    }
    _write_json(os.path.join(d, "run.json"), manifest)

    cols_by_name = {
        os.path.basename(p): {k: asdict(c) for k, c in per.items()}
        for p, per in columns.items()
    }
    _write_json(os.path.join(d, "columns.json"), cols_by_name)

    _write_csv(os.path.join(d, "results.csv"), methods, paths, columns, created_ts)
    logger.info("RunFolder.save: wrote %s (%d images, %d methods)", d, len(paths), len(methods))
    return d


def list_runs(folder: str) -> List[dict]:
    """Return run manifests for ``folder``, newest-first. Malformed dirs skipped; never raises."""
    root = _runs_dir(folder)
    if not os.path.isdir(root):
        return []
    out: List[dict] = []
    for name in os.listdir(root):
        mp = os.path.join(root, name, "run.json")
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp, encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception as e:  # malformed run — skip, don't sink the list
            logger.warning("RunFolder.list: skipping %s (%s)", name, e)
    out.sort(key=lambda m: m.get("created_ts", ""), reverse=True)
    return out


def load(folder: str, run_id: str) -> Optional[dict]:
    """Load a saved run → ``{paths, columns, methods, manifest}`` (None if missing/bad).

    ``columns`` is rebuilt as ``{full_path: {key: AnalysisColumn}}`` with paths
    re-anchored to ``folder`` (join folder + stored basename).
    """
    from app.image_studio.engine import AnalysisColumn  # lazy: keep module light
    d = os.path.join(_runs_dir(folder), run_id)
    try:
        with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        with open(os.path.join(d, "columns.json"), encoding="utf-8") as f:
            cols_by_name = json.load(f)
    except Exception as e:
        logger.warning("RunFolder.load: cannot read run %s (%s)", run_id, e)
        return None

    columns: Dict[str, Dict[str, AnalysisColumn]] = {}
    for name, per in cols_by_name.items():
        full = os.path.join(folder, name)
        columns[full] = {k: AnalysisColumn(**c) for k, c in per.items()}
    paths = [os.path.join(folder, n) for n in manifest.get("images", [])]
    return {"paths": paths, "columns": columns,
            "methods": manifest.get("methods", []), "manifest": manifest}


# --------------------------------------------------------------------------- #
def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path, methods, paths, columns, created_ts) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["file", "source_path", "run_timestamp", "spec_version"]
        for k in methods:
            header += [k, f"{k}_score"]
        w.writerow(header)
        for p in sorted(paths, key=os.path.basename):
            row = [os.path.basename(p), p, created_ts, SPEC_VERSION]
            for k in methods:
                c = columns.get(p, {}).get(k)
                row += [getattr(c, "display", "") if c else "",
                        getattr(c, "sort_value", "") if (c and c.sort_value is not None) else ""]
            w.writerow(row)
