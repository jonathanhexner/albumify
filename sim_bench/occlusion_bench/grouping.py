"""Near-duplicate grouping for leakage-safe splits (spec-096 T1.4).

sha1 dedupe only catches byte-identical files. Personal albums are full of NEAR
duplicates (burst shots, immediate retakes) — if two frames of the same scene land
on opposite sides of a train/test split, every score is inflated. Fix: cluster
near-duplicates into groups and derive the split from the GROUP, so a scene can
never straddle the split.

Grouping (union-find over two cues, per source dataset; over-grouping is the safe
direction for leakage — worst case we lose a little effective data):
  - dHash hamming distance <= DHASH_MAX  (perceptual near-duplicate)
  - capture time from the filename (YYYYMMDD_HHMMSS) within TIME_WINDOW_S seconds
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

from PIL import Image

try:  # albums contain .heic (grouping must hash them, not skip them)
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

DHASH_MAX = 8        # hamming distance on 64-bit dHash considered "same scene"
TIME_WINDOW_S = 15   # frames captured within this window (same source) = one burst

_TS_RE = re.compile(r"(20\d{6})_(\d{6})")


def dhash(path: str) -> Optional[int]:
    """64-bit difference hash (row-wise gradient of a 9x8 grayscale thumb)."""
    try:
        img = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    except Exception as e:
        logger.warning("dhash failed for %s: %s", path, e)
        return None
    px = list(img.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (1 if px[r * 9 + c] > px[r * 9 + c + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def filename_ts(name: str) -> Optional[datetime]:
    m = _TS_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def group_rows(rows: List[dict], image_root: str) -> Dict[str, str]:
    """Assign a group id to every manifest row -> {row_id: group_id}.

    ``group_id`` = the lexicographically smallest sha1 in the group, so it is
    deterministic and the split can be derived from it exactly like before.
    Grouping never crosses source datasets (a Budapest photo can't burst-match
    an Austria photo).
    """
    n = len(rows)
    uf = _UnionFind(n)
    hashes: List[Optional[int]] = []
    stamps: List[Optional[datetime]] = []
    for r in rows:
        sub = "positives" if str(r["label"]) == "1" else "negatives"
        hashes.append(dhash(os.path.join(image_root, sub, r["id"])))
        stamps.append(filename_ts(r["id"]))

    for i in range(n):
        for j in range(i + 1, n):
            if rows[i]["source_dataset"] != rows[j]["source_dataset"]:
                continue
            near = (hashes[i] is not None and hashes[j] is not None
                    and hamming(hashes[i], hashes[j]) <= DHASH_MAX)
            burst = (stamps[i] is not None and stamps[j] is not None
                     and abs((stamps[i] - stamps[j]).total_seconds()) <= TIME_WINDOW_S)
            if near or burst:
                uf.union(i, j)

    roots: Dict[int, List[int]] = {}
    for i in range(n):
        roots.setdefault(uf.find(i), []).append(i)
    out: Dict[str, str] = {}
    for members in roots.values():
        gid = min(rows[i]["sha1"] for i in members)
        for i in members:
            out[rows[i]["id"]] = gid
    return out
