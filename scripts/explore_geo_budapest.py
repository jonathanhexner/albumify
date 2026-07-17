"""Exploration: run all four spec-022 geo/semantic signals over an album and
report what each gives — EXIF (GPS+time), StreetCLIP (city), GeoCLIP (lat/lon),
BLIP (caption) — plus the segmentation competition outcome.

Usage (Windows):
    .venv/Scripts/python scripts/explore_geo_budapest.py [DIR] [LIMIT]

Defaults: DIR=D:/Budapest2025_Google  LIMIT=24
Writes specs/022-trip-detection/BUDAPEST_EXPLORATION.html and prints a summary.
Per-image results are disk-cached, so re-runs are fast.
"""

from __future__ import annotations

import html
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("explore")

from geo_cluster.exif_reader import ExifInputs, GeoMetadataExtractor
from geo_cluster.streetclip import StreetClipInputs, StreetCLIPLocator
from geo_cluster.geoclip_locator import GeoClipInputs, GeoCLIPLocator
from geo_cluster.captioning import BlipCaptioner, CaptionInputs
from geo_cluster.home import HomeAnchor
from geo_cluster.selector import SegmentationSelector
from geo_cluster.axes.base import AxisInputs

EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")


def discover(folder: str, limit: int) -> list[str]:
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))
    return [os.path.join(folder, n) for n in names[:limit]]


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else "D:/Budapest2025_Google"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    paths = discover(folder, limit)
    log.info("album=%s  images=%d (of folder)", folder, len(paths))

    log.info("[1/5] EXIF geo/time ...")
    meta = GeoMetadataExtractor().calc(ExifInputs(paths)).metadata

    log.info("[2/5] StreetCLIP city guesses ...")
    sclip = StreetCLIPLocator(top_k=3).calc(StreetClipInputs(paths)).predictions

    log.info("[3/5] GeoCLIP (lat,lon) ...")
    gclip = GeoCLIPLocator(top_k=3).calc(GeoClipInputs(paths)).predictions

    log.info("[4/5] BLIP captions ...")
    caps = BlipCaptioner().calc(CaptionInputs(paths)).captions

    log.info("[5/5] Home anchor + segmentation competition ...")
    home = HomeAnchor().calc(meta)
    outcome = SegmentationSelector().calc(
        AxisInputs(metadata=meta, home=home, captions=caps)
    )

    n_gps = sum(1 for m in meta.values() if m.has_geo)
    n_time = sum(1 for m in meta.values() if m.has_time)

    # ---- console summary ----
    print("\n" + "=" * 70)
    print(f"ALBUM: {folder}   images analysed: {len(paths)}")
    print(f"EXIF GPS: {n_gps}/{len(paths)}   EXIF time: {n_time}/{len(paths)}")
    print(f"Home anchor: {home}")
    print(f"Segmentation: {'FLAT (declined)' if outcome.flat else 'SEGMENTED'}  floor={outcome.floor}")
    for s in outcome.scores:
        print(f"   axis {s.axis:9s} score={s.overall:.3f}  {s.detail}")
    if outcome.winner:
        print(f"Winner: {len(outcome.winner.segments)} segments")
    print("=" * 70 + "\n")

    # ---- HTML report ----
    rows = []
    for p in paths:
        m = meta[p]
        gps = f"{m.lat:.4f}, {m.lon:.4f}" if m.has_geo else "&mdash;"
        tm = m.timestamp.strftime("%Y-%m-%d %H:%M") if m.has_time else "&mdash;"
        sc = "<br>".join(f"{html.escape(d['label'])} <i>{d['score']:.2f}</i>"
                         for d in sclip.get(p, [])[:3]) or "&mdash;"
        gc_list = gclip.get(p, [])
        gc = "&mdash;"
        if gc_list:
            d0 = gc_list[0]
            place = html.escape(d0.get("place") or "")
            gc = f"{d0['lat']:.3f}, {d0['lon']:.3f}<br><b>{place}</b> <i>{d0['prob']:.2f}</i>"
        cap = html.escape(caps.get(p, "") or "&mdash;")
        rows.append(
            f"<tr><td class=mono>{html.escape(os.path.basename(p))}</td>"
            f"<td class=mono>{tm}</td><td class=mono>{gps}</td>"
            f"<td>{sc}</td><td>{gc}</td><td class=cap>{cap}</td></tr>"
        )

    score_rows = "".join(
        f"<tr><td class=mono>{s.axis}</td><td class=mono>{s.overall:.3f}</td>"
        f"<td class=mono>{html.escape(str(s.detail))}</td></tr>"
        for s in outcome.scores
    )
    seg_line = ("<b>FLAT</b> — even the best axis fell below the floor "
                f"({outcome.floor}); album left ungrouped."
                if outcome.flat else
                f"<b>SEGMENTED</b> into {len(outcome.winner.segments)} segments.")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>Budapest geo/semantic exploration</title><style>
body{{margin:0;background:#0f1115;color:#e8ecf3;font:14px/1.55 system-ui,"Segoe UI",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:34px 22px 90px}}
h1{{font-size:25px;margin:0 0 4px}} h2{{font-size:18px;margin:34px 0 10px;border-left:3px solid #6ea8fe;padding-left:10px}}
.sub{{color:#9aa4b2}} .mono{{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#bcd0f5}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin:8px 0}}
th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid #2a2f3a;vertical-align:top}}
th{{color:#9aa4b2;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
td.cap{{color:#cbe5cf;max-width:240px}} tr:hover td{{background:#161a22}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}}
.card{{background:#171a21;border:1px solid #2a2f3a;border-radius:11px;padding:13px 18px;min-width:150px}}
.big{{font-size:22px;font-weight:700}} .k{{color:#9aa4b2;font-size:12px}}
.note{{border-left:3px solid #f0a843;background:#1d1a12;padding:11px 15px;border-radius:0 9px 9px 0;margin:12px 0}}
</style></head><body><div class=wrap>
<h1>Budapest album — geo &amp; semantic signal exploration</h1>
<p class=sub>spec-022 · {stamp} · {len(paths)} images from <span class=mono>{html.escape(folder)}</span> ·
each image run through all four signals. Per-image results are disk-cached.</p>

<div class=cards>
  <div class=card><div class=big>{n_gps}/{len(paths)}</div><div class=k>have EXIF GPS</div></div>
  <div class=card><div class=big>{n_time}/{len(paths)}</div><div class=k>have EXIF time</div></div>
  <div class=card><div class=big>{('flat' if outcome.flat else len(outcome.winner.segments))}</div><div class=k>segments ({'declined' if outcome.flat else 'best axis'})</div></div>
  <div class=card><div class=big class=mono>{('%.3f,%.3f'%home) if home else '&mdash;'}</div><div class=k>home anchor</div></div>
</div>

<h2>Segmentation competition</h2>
<p class=sub>{seg_line}</p>
<table><thead><tr><th>axis</th><th>score</th><th>detail</th></tr></thead><tbody>{score_rows}</tbody></table>

<h2>Per-image signals</h2>
<p class=sub>StreetCLIP = zero-shot pick from a fixed city list (GPS-less fallback).
GeoCLIP = direct (lat,lon) regression, reverse-geocoded offline. BLIP = free-text caption.</p>
<table><thead><tr><th>file</th><th>EXIF time</th><th>EXIF GPS</th>
<th>StreetCLIP top-3</th><th>GeoCLIP best</th><th>BLIP caption</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>

<div class=note><b>Reading this:</b> where EXIF GPS exists it is ground truth; compare the
CLIP guesses against it to judge model quality. Where GPS is blank, StreetCLIP/GeoCLIP are
the only location signal. BLIP captions are the raw material for the (still-unbuilt) semantic axis.</div>
</div></body></html>"""

    out_path = "specs/022-trip-detection/BUDAPEST_EXPLORATION.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
