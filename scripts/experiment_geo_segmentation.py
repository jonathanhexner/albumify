"""Standalone experiment: the multi-axis segmentation competition (spec-022).

Runs the pluggable-axis selector on synthetic albums (and, optionally, a real
photo directory) to see which axis wins and whether the album should be
segmented at all. Writes a visual HTML report.

Usage (Windows):
    .venv/Scripts/python scripts/experiment_geo_segmentation.py
    .venv/Scripts/python scripts/experiment_geo_segmentation.py --dir D:/Budapest2025_Google

Nothing here touches the production pipeline — it exercises geo_cluster/ directly
so we can tune axes, weights, and the floor before any wiring.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from geo_cluster.axes.base import AxisInputs
from geo_cluster.home import HomeAnchor
from geo_cluster.selector import SegmentationSelector
from geo_cluster.types import GeoMetadata, SegmentationOutcome

logging.basicConfig(level=logging.WARNING)
REPORT = Path("specs/022-trip-detection/EXPERIMENT_RESULTS.html")


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def _block(rng, center, n, jitter_km, start: datetime, end: datetime) -> list[GeoMetadata]:
    """n photos jittered around a center, with random times in [start, end]."""
    dlat = rng.normal(0, jitter_km / 111.0, n)
    dlon = rng.normal(0, jitter_km / (111.0 * np.cos(np.radians(center[0]))), n)
    span = (end - start).total_seconds()
    out = []
    for i in range(n):
        ts = start + timedelta(seconds=float(rng.uniform(0, span)))
        out.append(GeoMetadata("", ts, center[0] + dlat[i], center[1] + dlon[i]))
    return out


def _label(metas: list[GeoMetadata], scenario: str) -> dict[str, GeoMetadata]:
    """Assign synthetic paths and return a path->GeoMetadata dict."""
    d = {}
    for i, m in enumerate(metas):
        p = f"{scenario}/img_{i:04d}.jpg"
        d[p] = GeoMetadata(p, m.timestamp, m.lat, m.lon)
    return d


def scenarios() -> dict[str, dict[str, GeoMetadata]]:
    rng = np.random.default_rng(7)
    TLV, DXB, ALP = (32.08, 34.78), (25.20, 55.27), (47.26, 11.39)
    y = 2024

    # 1) Traveler: home all year + two far multi-day trips -> GEO should win.
    traveler = (
        _block(rng, TLV, 200, 12, datetime(y, 1, 1), datetime(y, 12, 31))
        + _block(rng, DXB, 90, 8, datetime(y, 5, 12), datetime(y, 5, 18))
        + _block(rng, ALP, 60, 15, datetime(y, 7, 3), datetime(y, 7, 7))
    )

    # 2) Kids at home: one location, four seasonal time-bursts -> TIME should win.
    kids = []
    for mo in (1, 4, 7, 10):
        kids += _block(rng, TLV, 60, 10, datetime(y, mo, 5), datetime(y, mo, 8))

    # 3) Wedding: one venue, one continuous day -> single burst -> FLAT.
    wedding = _block(rng, (47.50, 19.04), 600, 3,
                     datetime(y, 6, 22, 10), datetime(y, 6, 22, 23))

    # 4) No metadata at all -> every axis insufficient -> FLAT.
    no_meta = [GeoMetadata("", None, None, None) for _ in range(50)]

    return {
        "traveler (a year + 2 trips)": _label(traveler, "traveler"),
        "kids at home (4 seasons)": _label(kids, "kids"),
        "wedding (one venue, one day)": _label(wedding, "wedding"),
        "no metadata": _label(no_meta, "nometa"),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(v):
    return "  -  " if v is None else f"{v:5.2f}"


def run_one(name: str, meta: dict[str, GeoMetadata]) -> dict:
    home = HomeAnchor().calc(meta)
    inputs = AxisInputs(metadata=meta, home=home,
                        config={"geo_radius_km": 30.0, "time_gap_hours": 8.0})
    outcome = SegmentationSelector().calc(inputs)

    print(f"\n=== {name} ===")
    print(f"  images: {len(meta)} | home: "
          f"{('%.2f,%.2f' % home) if home else 'n/a'} | floor: {outcome.floor}")
    print(f"  {'axis':10} {'overall':>8} {'separ':>6} {'cover':>6} {'balan':>6} "
          f"{'stab':>6} {'parsi':>6}  segs")
    for s in outcome.scores:
        if s.detail.get("insufficient"):
            print(f"  {s.axis:10} {'  -  ':>8}  (insufficient signal)")
            continue
        segs = s.detail.get("n_segments", "-")
        flag = " [single bucket]" if s.detail.get("single_bucket") else ""
        print(f"  {s.axis:10} {s.overall:8.2f} {_fmt(s.separation)} {_fmt(s.coverage)} "
              f"{_fmt(s.balance)} {_fmt(s.stability)} {_fmt(s.parsimony)}  {segs}{flag}")
    verdict = "FLAT (no segmentation)" if outcome.flat else \
        f"WINNER = {outcome.winning_axis} ({len(outcome.winner.segments)} segments)"
    print(f"  -> {verdict}")
    return {"name": name, "home": home, "outcome": outcome, "n": len(meta)}


def write_html(results: list[dict]) -> None:
    def bar(v, color):
        w = 0 if v is None else int(round(v * 100))
        return (f"<div style='background:#1a1d24;border-radius:5px;height:14px;width:160px;display:inline-block'>"
                f"<div style='background:{color};height:14px;width:{w}%;border-radius:5px'></div></div>"
                f"<span style='color:#9aa4b2;font-size:12px;margin-left:6px'>{'-' if v is None else f'{v:.2f}'}</span>")

    rows = []
    for r in results:
        o: SegmentationOutcome = r["outcome"]
        verdict = ("<span style='color:#7c8493'>FLAT — no story imposed</span>" if o.flat
                   else f"<span style='color:#7fe0a0'>WINNER: {o.winning_axis} · "
                        f"{len(o.winner.segments)} segments</span>")
        axis_rows = ""
        for s in sorted(o.scores, key=lambda s: s.overall, reverse=True):
            if s.detail.get("insufficient"):
                axis_rows += (f"<tr><td>{s.axis}</td><td colspan='5' style='color:#7c8493'>"
                              f"insufficient signal</td></tr>")
                continue
            win = "color:#7fe0a0;font-weight:700" if (not o.flat and s.axis == o.winning_axis) else ""
            axis_rows += (
                f"<tr><td style='{win}'>{s.axis}</td>"
                f"<td>{bar(s.overall, '#6ea8fe')}</td>"
                f"<td style='color:#9aa4b2'>{_fmt(s.separation)}</td>"
                f"<td style='color:#9aa4b2'>{_fmt(s.coverage)}</td>"
                f"<td style='color:#9aa4b2'>{_fmt(s.balance)}</td>"
                f"<td style='color:#9aa4b2'>{_fmt(s.stability)}</td>"
                f"<td style='color:#9aa4b2'>{_fmt(s.parsimony)}</td></tr>")
        home = ("%.3f, %.3f" % r["home"]) if r["home"] else "n/a"
        rows.append(f"""
        <div class="card">
          <h2>{r['name']}</h2>
          <div class="meta">{r['n']} images · home {home} · floor {o.floor:.2f}</div>
          <table>
            <tr><th>axis</th><th>overall (≥floor wins)</th><th>separ</th><th>cover</th><th>balan</th><th>stab</th><th>parsi</th></tr>
            {axis_rows}
          </table>
          <div class="verdict">{verdict}</div>
        </div>""")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Spec-022 · Segmentation experiment</title><style>
body{{margin:0;background:#0f1115;color:#e8ecf3;font:15px/1.6 system-ui,Segoe UI,sans-serif}}
.wrap{{max-width:840px;margin:0 auto;padding:40px 22px 90px}}
h1{{font-size:26px}} .sub{{color:#9aa4b2;max-width:70ch}}
.card{{background:#171a21;border:1px solid #2a2f3a;border-radius:12px;padding:18px 22px;margin:16px 0}}
h2{{font-size:18px;margin:0 0 4px}} .meta{{color:#9aa4b2;font-size:13px;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid #20242e}}
th{{color:#9aa4b2;font-size:12px;text-transform:uppercase}}
.verdict{{margin-top:12px;font-size:15px;font-weight:600}}
code{{font-family:ui-monospace,Consolas,monospace;background:#11141b;padding:1px 6px;border-radius:5px}}
</style></head><body><div class="wrap">
<h1>Multi-axis segmentation — experiment results</h1>
<p class="sub">Each album runs through every enabled axis (<code>geo</code>, <code>time</code>, <code>identity</code>).
The highest <b>overall</b> score wins, but only if it clears the <b>floor</b> — otherwise the album is shown
<b>FLAT</b> with no story imposed. Identity reports "insufficient" here because these synthetic albums carry no
face clusters; that's the graceful-plug-in behaviour.</p>
{''.join(rows)}
<p class="sub" style="margin-top:30px;border-top:1px solid #2a2f3a;padding-top:16px">
Generated by <code>scripts/experiment_geo_segmentation.py</code> · spec-022 planning experiment.</p>
</div></body></html>"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(html, encoding="utf-8")
    print(f"\nHTML report -> {REPORT}")


# --------------------------------------------------------------------------- #
def from_directory(path: Path) -> dict[str, GeoMetadata]:
    from geo_cluster.exif_reader import ExifInputs, GeoMetadataExtractor
    exts = {".jpg", ".jpeg", ".png", ".heic"}
    files = [str(p) for p in path.rglob("*") if p.suffix.lower() in exts]
    print(f"scanning {len(files)} images under {path} ...")
    return GeoMetadataExtractor().calc(ExifInputs(image_paths=files)).metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None,
                    help="optional real photo directory to segment by EXIF")
    args = ap.parse_args()

    results = [run_one(name, meta) for name, meta in scenarios().items()]
    if args.dir:
        meta = from_directory(Path(args.dir))
        n_geo = sum(1 for m in meta.values() if m.has_geo)
        print(f"\nreal album: {len(meta)} images, {n_geo} with GPS")
        if meta:
            results.append(run_one(f"REAL: {args.dir}", meta))
    write_html(results)


if __name__ == "__main__":
    main()
