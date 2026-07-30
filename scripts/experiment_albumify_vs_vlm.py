"""Spec-102 experiment harness — Albumify vs VLM.

T1 slice: the `prep` command only. Downsamples a trip to the shared 768px working set (both
arms consume this), writes a manifest with the A1 input-set hash, and emits a coverage-roster
template for the owner to hand-label. Later slices add `albumify`, `vlm`, `judge`, `report`.

Usage (Windows, always the venv python):
    .venv/Scripts/python scripts/experiment_albumify_vs_vlm.py prep --trip budapest
    .venv/Scripts/python scripts/experiment_albumify_vs_vlm.py prep --trip budapest --max-edge 768

Working set lands on the data drive (not the repo); the roster (small, human-labelled, worth
versioning) lands under the spec dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from sim_bench.albumify_vs_vlm.downsample import DownsampleConfig, downsample_trip
from sim_bench.albumify_vs_vlm.roster import build_roster_template, save_roster

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("albumify_vs_vlm")

# Trip registry (spec-102 datasets). Budapest is the pilot (D2).
TRIPS: dict[str, Path] = {
    "budapest": Path(r"D:\Budapest2025_Google"),
    "austria": Path(r"D:\Austria_24"),
    "germany": Path(r"D:\Google_Germany"),
}

WORK_ROOT = Path(r"D:\albumify_vs_vlm")  # data-drive working area (downsampled sets, manifests)
REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO_ROOT / "specs" / "102-albumify-vs-vlm"


def _work_dir(trip: str) -> Path:
    return WORK_ROOT / trip


def cmd_prep(args: argparse.Namespace) -> int:
    trip = args.trip
    src = TRIPS.get(trip)
    if src is None:
        logger.error("unknown trip '%s' (choices: %s)", trip, ", ".join(TRIPS))
        return 2
    if not src.exists():
        logger.error("source not found: %s", src)
        return 2

    out_dir = _work_dir(trip) / f"imgs{args.max_edge}"
    cfg = DownsampleConfig(max_edge=args.max_edge, quality=args.quality)
    result = downsample_trip(src, out_dir, trip=trip, config=cfg)
    if result.n_images == 0:
        logger.error("no images downsampled for %s", trip)
        return 1

    manifest_path = _work_dir(trip) / "manifest.json"
    manifest = {
        "trip": trip,
        "source": str(src),
        "out_dir": result.out_dir,
        "n_images": result.n_images,
        "input_set_hash": result.input_set_hash,
        "config": asdict(cfg),
        "records": [asdict(r) for r in result.records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("manifest -> %s", manifest_path)

    roster_path = SPEC_DIR / "rosters" / f"{trip}.roster.json"
    if roster_path.exists() and not args.force_roster:
        logger.info("roster exists, kept (use --force-roster to overwrite): %s", roster_path)
    else:
        save_roster(build_roster_template(result), roster_path)

    print(
        f"\nprep done: {trip}\n"
        f"  images     : {result.n_images}\n"
        f"  input hash : {result.input_set_hash[:16]}...\n"
        f"  working set: {result.out_dir}\n"
        f"  manifest   : {manifest_path}\n"
        f"  roster     : {roster_path}  (hand-label persons + scenes next)\n"
    )
    return 0


def _load_manifest(trip: str) -> dict:
    path = _work_dir(trip) / "manifest.json"
    if not path.exists():
        raise SystemExit(f"no manifest for {trip} — run `prep --trip {trip}` first ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_albumify(args: argparse.Namespace) -> int:
    from sim_bench.albumify_vs_vlm.albumify_arm import (
        AlbumifyArmConfig,
        run_albumify_arm,
        save_album_result,
    )

    trip = args.trip
    manifest = _load_manifest(trip)
    imgs_dir = Path(manifest["out_dir"])
    cfg = AlbumifyArmConfig(target_k=args.k, pipeline=args.pipeline)
    res = run_albumify_arm(imgs_dir, trip, manifest["input_set_hash"], cfg)

    out = _work_dir(trip) / "albumify_picks.json"
    save_album_result(res, out)
    # Scene-cluster map is a shared artifact (VLM contact sheets + dup-survival metric).
    scenes_path = _work_dir(trip) / "scene_clusters.json"
    scenes_path.write_text(
        json.dumps({str(c): s for c, s in res.scene_clusters.items()}, indent=2),
        encoding="utf-8",
    )
    print(
        f"\nalbumify done: {trip}\n"
        f"  pipeline   : {res.pipeline}\n"
        f"  picks      : {len(res.order)} (target K={res.k})\n"
        f"  scenes     : {len(res.scene_clusters)} clusters\n"
        f"  picks file : {out}\n"
        f"  scenes file: {scenes_path}\n"
        f"  order      : {', '.join(res.order[:6])}{' ...' if len(res.order) > 6 else ''}\n"
    )
    return 0


def cmd_vlm(args: argparse.Namespace) -> int:
    from sim_bench.albumify_vs_vlm.schema import save_album_result
    from sim_bench.albumify_vs_vlm.vlm_arm import VLMArmConfig, run_vlm_arm

    trip = args.trip
    manifest = _load_manifest(trip)
    imgs_dir = Path(manifest["out_dir"])
    stems = [r["stem"] for r in manifest["records"]]
    cfg = VLMArmConfig(target_k=args.k, batch_size=args.batch_size)
    shortlist_cache = _work_dir(trip) / f"{trip}_vlm_shortlist.json"
    res = run_vlm_arm(imgs_dir, stems, trip, manifest["input_set_hash"], cfg,
                      shortlist_cache=shortlist_cache)

    out = _work_dir(trip) / "vlm_picks.json"
    save_album_result(res, out)
    m = res.meta
    print(
        f"\nvlm done: {trip}\n"
        f"  model      : {res.pipeline}\n"
        f"  picks      : {len(res.order)} (target K={res.k})\n"
        f"  shortlisted: {m.get('n_shortlisted')} over {m.get('n_batches')} batches\n"
        f"  tokens     : in={m.get('input_tokens')} out={m.get('output_tokens')}\n"
        f"  picks file : {out}\n"
        f"  order      : {', '.join(res.order[:6])}{' ...' if len(res.order) > 6 else ''}\n"
    )
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    from sim_bench.albumify_vs_vlm.annotate import run_annotation, save_annotation

    trip = args.trip
    manifest = _load_manifest(trip)
    imgs_dir = Path(manifest["out_dir"])
    roster_path = SPEC_DIR / "rosters" / f"{trip}.roster.json"
    if not roster_path.exists():
        raise SystemExit(f"no roster for {trip} — run `prep --trip {trip}` first")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))

    ann = run_annotation(imgs_dir, roster, trip, manifest["input_set_hash"])
    out = _work_dir(trip) / "vlm_annotation.json"
    save_annotation(ann, out)
    print(
        f"\nannotate done: {trip}\n"
        f"  album_type : {ann['album_type']} / {ann['trip_subtype']}\n"
        f"  narrative  : {ann['narrative']}\n"
        f"  groups     : {ann['n_groups']}\n"
        f"  tokens     : in={ann['meta']['input_tokens']} out={ann['meta']['output_tokens']}\n"
        f"  file       : {out}\n"
    )
    return 0


def cmd_viewer(args: argparse.Namespace) -> int:
    from sim_bench.albumify_vs_vlm.ab_viewer import generate_ab_viewer

    trip = args.trip
    manifest = _load_manifest(trip)
    imgs_dir = Path(manifest["out_dir"])
    wd = _work_dir(trip)
    albumify_json, vlm_json = wd / "albumify_picks.json", wd / "vlm_picks.json"
    for p in (albumify_json, vlm_json):
        if not p.exists():
            raise SystemExit(f"missing {p} — run both `albumify` and `vlm` for {trip} first")

    out_html = wd / f"ab_viewer_{trip}.html"
    out_key = wd / f"ab_UNBLIND_KEY_{trip}.json"
    generate_ab_viewer(trip, imgs_dir, albumify_json, vlm_json, out_html, out_key)
    print(
        f"\nviewer done: {trip}\n"
        f"  open       : {out_html}\n"
        f"  unblind key: {out_key}  (DO NOT open until you've saved your answers)\n"
    )
    return 0


def cmd_exp2(args: argparse.Namespace) -> int:
    """EXP-2: build cluster cases (Albumify best = argmax composite) + VLM best-per-cluster."""
    from sim_bench.albumify_vs_vlm.exp2 import (
        Exp2Result,
        save_exp2,
        select_cluster_cases,
        vlm_best_per_cluster,
    )

    trip = args.trip
    manifest = _load_manifest(trip)
    imgs_dir = Path(manifest["out_dir"])
    wd = _work_dir(trip)
    picks_path = wd / "albumify_picks.json"
    if not picks_path.exists():
        raise SystemExit(f"missing {picks_path} — run `albumify --trip {trip} --pipeline default` first")
    alb = json.loads(picks_path.read_text(encoding="utf-8"))
    scores = (alb.get("meta") or {}).get("composite_scores")
    if not scores:
        raise SystemExit(
            f"{picks_path} has no composite_scores — re-run `albumify --trip {trip} --pipeline default` "
            "(the arm now dumps them)."
        )
    scene_clusters = {int(c): s for c, s in alb["scene_clusters"].items()}

    cases = select_cluster_cases(scene_clusters, scores)
    if not cases:
        raise SystemExit(f"no eligible clusters for {trip} (need multi-frame scene clusters)")

    in_tok = out_tok = 0
    if not args.no_vlm:
        in_tok, out_tok = vlm_best_per_cluster(imgs_dir, cases, model=args.model)

    res = Exp2Result(
        trip=trip, input_set_hash=alb["input_set_hash"], cases=cases,
        meta={"input_tokens": in_tok, "output_tokens": out_tok,
              "n_clusters": len(cases), "model": args.model or "claude-opus-4-8"},
    )
    out = wd / "exp2.json"
    save_exp2(res, out)
    agree = sum(1 for c in cases if c.vlm_best and c.albumify_best == c.vlm_best)
    print(
        f"\nexp2 done: {trip}\n"
        f"  clusters   : {len(cases)} (frames {sum(len(c.stems) for c in cases)})\n"
        f"  vlm picks  : {'skipped' if args.no_vlm else 'done'}  tokens in={in_tok} out={out_tok}\n"
        f"  system agree (albumify==vlm): {agree}/{len(cases)}\n"
        f"  file       : {out}\n"
        f"  next       : `exp2-viewer --trip {trip}`, judge, save exp2_judging.json, `exp2-metrics`\n"
    )
    return 0


def cmd_exp2_viewer(args: argparse.Namespace) -> int:
    from sim_bench.albumify_vs_vlm.exp2_viewer import generate_exp2_viewer

    trip = args.trip
    manifest = _load_manifest(trip)
    imgs_dir = Path(manifest["out_dir"])
    wd = _work_dir(trip)
    exp2_json = wd / "exp2.json"
    if not exp2_json.exists():
        raise SystemExit(f"missing {exp2_json} — run `exp2 --trip {trip}` first")
    out_html = wd / f"exp2_picker_{trip}.html"
    generate_exp2_viewer(trip, imgs_dir, exp2_json, out_html)
    print(
        f"\nexp2-viewer done: {trip}\n"
        f"  open  : {out_html}\n"
        f"  then  : click best per cluster -> Copy answers -> save as {wd / 'exp2_judging.json'}\n"
    )
    return 0


def cmd_exp2_metrics(args: argparse.Namespace) -> int:
    """Fold the human judging back in and report top-1 accuracy per system."""
    from sim_bench.albumify_vs_vlm.exp2 import load_exp2, save_exp2, top1_accuracy

    trip = args.trip
    wd = _work_dir(trip)
    exp2_json = wd / "exp2.json"
    if not exp2_json.exists():
        raise SystemExit(f"missing {exp2_json} — run `exp2 --trip {trip}` first")
    res = load_exp2(exp2_json)

    judging = wd / "exp2_judging.json"
    if judging.exists():
        picks = json.loads(judging.read_text(encoding="utf-8"))
        by_id = {c.cluster_id: c for c in res.cases}
        applied = 0
        for key, stem in picks.items():
            cid = int(str(key).replace("cluster_", ""))
            if cid in by_id and stem in by_id[cid].stems:
                by_id[cid].human_best = stem
                applied += 1
        save_exp2(res, exp2_json)  # persist human_best back into exp2.json
        print(f"applied {applied} human picks from {judging}")
    else:
        print(f"(no {judging} yet — reporting system agreement only; judge via `exp2-viewer` first)")

    m = top1_accuracy(res.cases)
    alb_detail = f"  ({m['albumify_hits']}/{m['n_judged']})" if m["n_judged"] else ""
    vlm_detail = f"  ({m['vlm_hits']}/{m['n_judged']})" if m["n_judged"] else ""
    print(
        f"\nexp2-metrics: {trip}\n"
        f"  clusters judged : {m['n_judged']} / {m['n_total']}\n"
        f"  Albumify top-1  : {m['albumify_top1']}{alb_detail}\n"
        f"  VLM top-1       : {m['vlm_top1']}{vlm_detail}\n"
        f"  system agreement: {m['system_agreement']}\n"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from sim_bench.albumify_vs_vlm.report import generate_report

    trip = args.trip
    manifest = _load_manifest(trip)
    imgs_dir = Path(manifest["out_dir"])
    wd = _work_dir(trip)
    albumify_json, vlm_json = wd / "albumify_picks.json", wd / "vlm_picks.json"
    for p in (albumify_json, vlm_json):
        if not p.exists():
            raise SystemExit(f"missing {p} — run both arms for {trip} first")

    annotation_json = wd / "vlm_annotation.json"
    out_dir = REPO_ROOT / "reports" / f"{args.date}_albumify_vs_vlm_{trip}"
    generate_report(trip, imgs_dir, albumify_json, vlm_json, out_dir,
                    annotation_json=annotation_json if annotation_json.exists() else None)
    print(f"\nreport done: {out_dir / 'report.html'}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spec-102 Albumify-vs-VLM experiment harness")
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prep", help="downsample a trip + emit manifest and roster template")
    prep.add_argument("--trip", required=True, choices=sorted(TRIPS))
    prep.add_argument("--max-edge", type=int, default=768)
    prep.add_argument("--quality", type=int, default=80)
    prep.add_argument("--force-roster", action="store_true", help="overwrite an existing roster")
    prep.set_defaults(func=cmd_prep)

    alb = sub.add_parser("albumify", help="run the Albumify arm -> ordered K-sequence")
    alb.add_argument("--trip", required=True, choices=sorted(TRIPS))
    alb.add_argument("--k", type=int, default=20, help="target album size (both arms match)")
    alb.add_argument("--pipeline", choices=["faces", "default", "minimal"], default="faces",
                     help="faces = people+quality+scenes minus OOM steps (recommended); "
                          "default = full 33-step (OOMs on low RAM); minimal = scenes-only")
    alb.set_defaults(func=cmd_albumify)

    vlm = sub.add_parser("vlm", help="run the VLM arm (Claude Opus 4.8) -> ordered K-sequence")
    vlm.add_argument("--trip", required=True, choices=sorted(TRIPS))
    vlm.add_argument("--k", type=int, default=20, help="target album size (both arms match)")
    vlm.add_argument("--batch-size", type=int, default=15, help="images per map-phase batch")
    vlm.set_defaults(func=cmd_vlm)

    vw = sub.add_parser("viewer", help="build the blind A/B judging HTML from both arms' picks")
    vw.add_argument("--trip", required=True, choices=sorted(TRIPS))
    vw.set_defaults(func=cmd_viewer)

    ann = sub.add_parser("annotate", help="EXP-3: VLM structured annotation (moments + album type)")
    ann.add_argument("--trip", required=True, choices=sorted(TRIPS))
    ann.set_defaults(func=cmd_annotate)

    rep = sub.add_parser("report", help="build the experiment report HTML + summary.md")
    rep.add_argument("--trip", required=True, choices=sorted(TRIPS))
    rep.add_argument("--date", default="2026-07-17", help="report folder date prefix")
    rep.set_defaults(func=cmd_report)

    e2 = sub.add_parser("exp2", help="EXP-2: build cluster cases + VLM best-per-cluster")
    e2.add_argument("--trip", required=True, choices=sorted(TRIPS))
    e2.add_argument("--model", default=None, help="VLM model (default claude-opus-4-8)")
    e2.add_argument("--no-vlm", action="store_true", help="skip VLM calls (Albumify + agreement only)")
    e2.set_defaults(func=cmd_exp2)

    e2v = sub.add_parser("exp2-viewer", help="EXP-2: blind best-frame picker HTML")
    e2v.add_argument("--trip", required=True, choices=sorted(TRIPS))
    e2v.set_defaults(func=cmd_exp2_viewer)

    e2m = sub.add_parser("exp2-metrics", help="EXP-2: fold in judging -> top-1 accuracy per system")
    e2m.add_argument("--trip", required=True, choices=sorted(TRIPS))
    e2m.set_defaults(func=cmd_exp2_metrics)
    return p


def main() -> int:
    # VLM captions carry accented chars; the Windows console codepage (cp1255 here) can't encode
    # them and crashes on print. Degrade gracefully rather than abort a completed run.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
