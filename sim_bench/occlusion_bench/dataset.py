"""Occlusion benchmark dataset builder (spec-096 Slice 1).

Builds ``D:\\occlusion_dataset\\{positives,negatives}`` from user-labeled sources with
full provenance:

- files are copied with dataset-prefixed ids (``budapest__20250822_1226.jpg``) so the
  origin is visible in the filename;
- ``manifest.csv`` is the source of truth: label, source dataset + original path, sha1,
  frozen train/test split, hard-negative flag;
- sha1 dedupe — the same photo appearing in two sources (e.g. the finger images that
  exist in both ``examples/finger_occlusion`` and the Budapest album) is kept ONCE,
  with positives processed first so an occluded duplicate can never land in negatives;
- the split is DETERMINISTIC (derived from sha1, no RNG): reproducible and frozen.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")
TEST_BUCKETS = {0}  # sha1 % 5 == 0 -> test (~20%), frozen


@dataclass
class SourceDir:
    key: str          # provenance prefix, e.g. "budapest"
    path: str         # folder to scan
    label: int        # 1 = occluded, 0 = clean


@dataclass
class BuildResult:
    n_positives: int = 0
    n_negatives: int = 0
    n_duplicates_skipped: int = 0
    rows: List[dict] = field(default_factory=list)


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_for(sha1_hex: str) -> str:
    return "test" if int(sha1_hex[:8], 16) % 5 in TEST_BUCKETS else "train"


def _images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        logger.warning("dataset: source folder missing: %s", folder)
        return []
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    )


def build(sources: List[SourceDir], out_dir: str,
          hard_negatives: Optional[Set[str]] = None) -> BuildResult:
    """Build the dataset folder + manifest. Positives-first source order matters:
    a duplicate file (same sha1) keeps its FIRST occurrence's label."""
    hard_negatives = hard_negatives or set()
    res = BuildResult()
    seen: Dict[str, str] = {}  # sha1 -> id that claimed it

    pos_dir = os.path.join(out_dir, "positives")
    neg_dir = os.path.join(out_dir, "negatives")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    # positives first so occluded duplicates can never be claimed as negatives
    for src in sorted(sources, key=lambda s: -s.label):
        for p in _images(src.path):
            sha = _sha1(p)
            base = os.path.basename(p)
            if sha in seen:
                res.n_duplicates_skipped += 1
                logger.info("dedupe: %s (%s) already present as %s", base, src.key, seen[sha])
                continue
            new_id = f"{src.key}__{base}"
            seen[sha] = new_id
            dest = os.path.join(pos_dir if src.label == 1 else neg_dir, new_id)
            shutil.copy2(p, dest)
            res.rows.append({
                "id": new_id,
                "label": src.label,
                "level": "",  # 0-3 severity: Haiku proposes, user confirms (Track A)
                "source_dataset": src.key,
                "source_path": p,
                "sha1": sha,
                "split": _split_for(sha),
                "hard_negative": base in hard_negatives and src.label == 0,
                "notes": "",
            })
            if src.label == 1:
                res.n_positives += 1
            else:
                res.n_negatives += 1

    manifest = os.path.join(out_dir, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res.rows[0].keys()) if res.rows else
                           ["id", "label", "level", "source_dataset", "source_path",
                            "sha1", "split", "hard_negative", "notes"])
        w.writeheader()
        w.writerows(res.rows)
    logger.info("dataset: %d positives, %d negatives, %d dupes skipped -> %s",
                res.n_positives, res.n_negatives, res.n_duplicates_skipped, out_dir)
    return res


def load_manifest(out_dir: str) -> List[dict]:
    with open(os.path.join(out_dir, "manifest.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
