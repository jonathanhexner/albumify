"""Build the Gate 1 picks review page for one Albumify run (default: Slovakia26 run 52776cc5).

Writes downscaled copies of every photo + face crop samples into this folder (untracked: images are
gitignored) and items.json, which review.html reads. Usage:
    .venv/Scripts/python reports/2026-11-13_slovakia26_picks_review/build_review.py
"""
import argparse
import csv
import json
import logging
import os
import sqlite3
from pathlib import Path

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

OUT = Path(__file__).resolve().parent
DB = Path(os.path.expanduser("~/.sim_bench/sim_bench.db"))
REPO = OUT.parent.parent
NO_SCENE_PAGE = 20
CROPS_PER_PERSON = 30


def thumb(src: Path, dst: Path, edge: int) -> None:
    if dst.exists():
        return
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((edge, edge))
        im.save(dst, "JPEG", quality=82)


def photo_entry(path: str, metrics: dict, picked: set, img_dir: Path) -> dict:
    p = Path(path)
    dst = img_dir / f"{p.stem}.jpg"
    thumb(p, dst, 1100)
    m = metrics.get(path, {})
    return {
        "key": p.name,
        "img": f"img/{dst.name}",
        "full": p.resolve().as_uri(),
        "pick": path in picked,
        "why": m.get("filter_reason") or "",
        "score": round(m.get("composite_score") or 0, 3),
        "sharp": round(m.get("sharpness") or 0, 3),
        "faces": m.get("face_count") or 0,
    }


def build(run_prefix: str) -> None:
    con = sqlite3.connect(DB)
    row = con.execute(
        "select r.id, a.name, res.scene_clusters, res.selected_images, res.image_metrics, res.fc_export_dir "
        "from pipeline_runs r join albums a on a.id = r.album_id join pipeline_results res on res.run_id = r.id "
        "where r.id like ?", (run_prefix + "%",)).fetchone()
    run_id, album, scenes_js, sel_js, metrics_js, fc_dir = row
    scenes = json.loads(scenes_js)
    picked = set(json.loads(sel_js))
    metrics = json.loads(metrics_js)
    img_dir = OUT / "img"
    img_dir.mkdir(exist_ok=True)

    groups = []
    named = sorted((k for k in scenes if k != "-1"), key=lambda k: min(Path(p).name for p in scenes[k]))
    for k in named:
        paths = sorted(scenes[k], key=lambda p: Path(p).name)
        groups.append({"id": f"scene-{k}", "title": f"Scene {k}", "kind": "scene",
                       "photos": [photo_entry(p, metrics, picked, img_dir) for p in paths]})
    lone = sorted(scenes.get("-1", []), key=lambda p: Path(p).name)
    pages = [lone[i:i + NO_SCENE_PAGE] for i in range(0, len(lone), NO_SCENE_PAGE)]
    for i, chunk in enumerate(pages):
        groups.append({"id": f"lone-{i}", "title": f"In no scene ({i + 1}/{len(pages)}), each judged alone",
                       "kind": "lone", "photos": [photo_entry(p, metrics, picked, img_dir) for p in chunk]})
    in_scene = {p for v in scenes.values() for p in v}
    removed = sorted(set(metrics) - in_scene, key=lambda p: Path(p).name)
    for i in range(0, len(removed), NO_SCENE_PAGE):
        chunk = removed[i:i + NO_SCENE_PAGE]
        groups.append({"id": f"removed-{i // NO_SCENE_PAGE}",
                       "title": f"Removed before scenes ({i // NO_SCENE_PAGE + 1}/{-(-len(removed) // NO_SCENE_PAGE)})",
                       "kind": "removed", "photos": [photo_entry(p, metrics, picked, img_dir) for p in chunk]})

    people = []
    fc = REPO / fc_dir
    crop_dir = OUT / "crops"
    crop_dir.mkdir(exist_ok=True)
    with open(fc / "faces.csv", newline="") as f:
        faces = list(csv.DictReader(f))
    by_cluster: dict[int, list[dict]] = {}
    for face in faces:
        by_cluster.setdefault(int(face["cluster_id"]), []).append(face)
    for cid in sorted(by_cluster, key=lambda c: (c < 0, -len(by_cluster[c]))):
        members = by_cluster[cid]
        step = max(1, len(members) // CROPS_PER_PERSON)
        crops = []
        for face in members[::step][:CROPS_PER_PERSON]:
            dst = crop_dir / Path(face["crop_path"]).name
            thumb(fc / face["crop_path"], dst, 160)
            crops.append({"img": f"crops/{dst.name}", "photo": face["image_id"]})
        people.append({"id": cid, "size": len(members), "crops": crops})

    data = {"run": run_id, "album": album, "n_picks": len(picked), "groups": groups, "people": people}
    (OUT / "items.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (OUT / "items.js").write_text("window.REVIEW=" + json.dumps(data) + ";", encoding="utf-8")
    logger.info("run %s: %d groups, %d picks, %d people -> %s", run_id[:8], len(groups), len(picked), len(people), OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="52776cc5")
    build(ap.parse_args().run)
