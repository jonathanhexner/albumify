"""Ingest new positive captures from D:/occlusion_dataset/positives into the
manifest + clip_embeddings.npz (batch-3, 2026-07-10).

- Files not yet in the manifest are renamed on disk to occl__<name> (dataset id
  convention) unless already prefixed.
- Burst grouping: shots <=15s apart share a group_id (leak-free CV), matching
  the grouping granularity of earlier batches.
- Embeds each new image (CLIP global + 9 tiles via OcclusionScorer.embed) and
  rewrites clip_embeddings.npz with the appended rows.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
from datetime import datetime

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("ingest")

ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")
BURST_GAP_S = 15


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ts_of(name):
    m = re.search(r"(\d{8})_(\d{6})", name)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def main():
    from sim_bench.occlusion_bench.scorer import OcclusionScorer
    man_path = os.path.join(ROOT, "manifest.csv")
    with open(man_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    known = {r["id"] for r in rows}

    pos_dir = os.path.join(ROOT, "positives")
    new = []
    for fn in sorted(os.listdir(pos_dir)):
        rid = fn if fn.startswith("occl__") else "occl__" + fn
        if fn in known or rid in known:
            continue
        if rid != fn:
            os.rename(os.path.join(pos_dir, fn), os.path.join(pos_dir, rid))
        new.append(rid)
    if not new:
        logger.info("nothing to ingest")
        return
    logger.info("ingesting %d new positives", len(new))

    # burst grouping by timestamp gap
    stamped = [(rid, ts_of(rid)) for rid in new]
    groups, cur = [], [stamped[0]]
    for prev, item in zip(stamped, stamped[1:]):
        if item[1] and prev[1] and (item[1] - prev[1]).total_seconds() <= BURST_GAP_S:
            cur.append(item)
        else:
            groups.append(cur)
            cur = [item]
    groups.append(cur)
    logger.info("%d burst groups: %s", len(groups), [len(g) for g in groups])

    scorer = OcclusionScorer()
    d = np.load(os.path.join(ROOT, "clip_embeddings.npz"), allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    y = list(d["y"].astype(int))
    grp = [str(x) for x in d["groups"]]
    eg = list(d["emb_global"])
    et = list(d["emb_tiles"])

    added = []
    for g in groups:
        gid = sha1(os.path.join(pos_dir, g[0][0]))  # group id = sha1 of first member
        for rid, _ in g:
            path = os.path.join(pos_dir, rid)
            pair = scorer.embed(path)
            if pair is None:
                logger.warning("SKIP unreadable %s", rid)
                continue
            ids.append(rid)
            y.append(1)
            grp.append(gid)
            eg.append(pair[0])
            et.append(pair[1])
            added.append({"id": rid, "label": "1", "level": "",
                          "source_dataset": "occl", "source_path": path,
                          "sha1": sha1(path), "split": "train",
                          "hard_negative": "False", "notes": "batch3 2026-07-10",
                          "group_id": gid})
            logger.info("embedded %s (group %s)", rid, gid[:8])

    np.savez_compressed(os.path.join(ROOT, "clip_embeddings.npz"),
                        ids=np.array(ids), y=np.array(y), groups=np.array(grp),
                        emb_global=np.array(eg), emb_tiles=np.array(et))
    with open(man_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        for r in added:
            w.writerow(r)
    logger.info("DONE: manifest %d -> %d rows; embeddings %d rows",
                len(rows), len(rows) + len(added), len(ids))


if __name__ == "__main__":
    main()
