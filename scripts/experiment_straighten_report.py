"""spec-101 T4: before/after auto-straighten report on real confident tilts.

Takes the confident (unc<=1.5) + tilted (|roll|>=3) Budapest photos from the
spec-100 benchmark, runs the REAL subject-aware gate (YOLO person boxes), and for
each shows original vs straightened (or the decline reason), the retained-area
FOV cost, and the fixability-scaled tilt_penalty it would receive. The retained-
area column is the number the fov_weight default (0.4) should be tuned against.

Output: reports/2026-07-17_auto_straighten/{report.html, summary.md}
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_bench.quality_assessment.straighten import retained_area_fraction, straighten
from sim_bench.quality_assessment.straighten_gate import GateConfig, decide

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("straighten_report")

SRC = Path(r"D:\Budapest2025_Google")
BENCH = Path(__file__).resolve().parents[1] / "reports" / "2026-07-12_geocalib_tilt" / "metrics.json"
OUT = Path(__file__).resolve().parents[1] / "reports" / "2026-07-17_auto_straighten"
IMG = OUT / "img"
CFG = GateConfig()  # production defaults (area 0.70, person 0.15)
FOV_W, SLOPE, GATE_DEG, CAP = 0.4, 0.02, 3.0, 0.15
DW = 460


def penalty(roll, retained, straightened):
    if straightened:
        return -min(FOV_W * (1 - retained), CAP)
    return -min(SLOPE * (abs(roll) - GATE_DEG), CAP)


def person_bbox(detector, path):
    if detector is None:
        return None
    try:
        p = detector.detect_person(Path(path))
    except Exception:
        return None
    if p is None or p.bbox is None:
        return None
    b = p.bbox
    return (float(b.x), float(b.y), float(b.w), float(b.h))


def save_disp(rgb, out):
    d = cv2.resize(rgb, (DW, int(DW * rgb.shape[0] / rgb.shape[1])))
    cv2.imwrite(str(out), cv2.cvtColor(d, cv2.COLOR_RGB2BGR))


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    rows_m = json.load(open(BENCH))["rows"]
    picks = [r for r in rows_m if r["base_unc"] <= 1.5 and abs(r["base_angle"]) >= 3.0]
    picks.sort(key=lambda r: -abs(r["base_angle"]))
    log.info("confident tilted photos: %d", len(picks))

    try:
        from sim_bench.pipeline.person_detection.yolo_detector import YOLOPersonDetector
        detector = YOLOPersonDetector({})  # defaults
    except Exception as e:
        log.warning("YOLO unavailable (%s); person gating skipped", e)
        detector = None

    cards, retained_fixable = [], []
    for r in picks:
        stem, roll, unc = r["photo"], r["base_angle"], r["base_unc"]
        conf = float(np.exp(-unc / 2.0))
        p = next(SRC.rglob(stem + ".jp*"))
        rgb = np.array(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
        h, w = rgb.shape[:2]
        pb = person_bbox(detector, str(p))
        g = decide(roll, conf, w, h, pb, CFG)
        pen = penalty(roll, g.retained_area, g.straighten)
        save_disp(rgb, IMG / f"{stem}_orig.jpg")
        if g.straighten:
            save_disp(straighten(rgb, roll, preserve_aspect=True), IMG / f"{stem}_straight.jpg")
            retained_fixable.append(g.retained_area)
        cards.append({"stem": stem, "roll": roll, "conf": conf, "wh": (w, h),
                      "reason": g.reason, "retained": g.retained_area, "penalty": pen,
                      "straight": g.straighten, "person": pb is not None})
        log.info("%s roll=%.1f %s retained=%.2f pen=%.3f", stem, roll, g.reason, g.retained_area, pen)

    med = float(np.median(retained_fixable)) if retained_fixable else float("nan")
    n_str = sum(c["straight"] for c in cards)
    write_html(cards, med, n_str)
    OUT.joinpath("summary.md").write_text(
        f"# Auto-straighten before/after (spec-101 T4)\n\n"
        f"- {len(cards)} confident tilted photos; {n_str} straightened, {len(cards)-n_str} declined.\n"
        f"- Median retained area on straightened: {med*100:.0f}% "
        f"(-> fov penalty {FOV_W*(1-med):.3f} at fov_weight={FOV_W}).\n"
        f"- Decline reasons: " + ", ".join(sorted({c['reason'] for c in cards if not c['straight']})) + "\n"
        f"- Report: report.html.\n", encoding="utf-8")
    log.info("done -> %s", OUT / "report.html")


def write_html(cards, med, n_str):
    def card(c):
        s = c["stem"]
        right = (f'<figure><img src="img/{s}_straight.jpg"><figcaption>straightened '
                 f'({-c["roll"]:+.1f}&deg;, kept {c["retained"]*100:.0f}%)</figcaption></figure>'
                 if c["straight"] else
                 f'<div class="declined"><b>declined</b><br>{c["reason"].replace("_"," ")}'
                 f'<br><span>left as-shot; penalty {c["penalty"]:+.3f}</span></div>')
        badge = ("straighten" if c["straight"] else c["reason"])
        cls = "ok" if c["straight"] else "no"
        return f"""<div class="card">
  <h3>{s} &middot; <span class="roll">roll {c['roll']:+.1f}&deg;</span>
      <span class="mut">conf {c['conf']:.2f} &middot; {c['wh'][0]}&times;{c['wh'][1]}
      {'&middot; person' if c['person'] else ''}</span>
      <span class="badge {cls}">{badge}</span>
      <span class="pen">penalty {c['penalty']:+.3f}</span></h3>
  <div class="ba">
    <figure><img src="img/{s}_orig.jpg"><figcaption>original</figcaption></figure>
    {right}
  </div></div>"""

    body = "\n".join(card(c) for c in cards)
    tune = "\n".join(
        f"<tr><td>{c['stem']}</td><td class=n>{c['retained']*100:.0f}%</td>"
        f"<td class=n>{c['penalty']:+.3f}</td></tr>"
        for c in cards if c["straight"])
    OUT.joinpath("report.html").write_text(f"""<title>Auto-straighten before/after — spec-101</title>
<style>
:root{{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--card:#fafafa;--ok:#1a8a4a;--no:#c0392b;--accent:#7b3fb3}}
@media(prefers-color-scheme:dark){{:root{{--bg:#141414;--fg:#e8e8e8;--mut:#9a9a9a;--line:#2c2c2c;--card:#1d1d1d;--ok:#4fd07f;--no:#ff6b5b;--accent:#c08bff}}}}
:root[data-theme=dark]{{--bg:#141414;--fg:#e8e8e8;--mut:#9a9a9a;--line:#2c2c2c;--card:#1d1d1d;--ok:#4fd07f;--no:#ff6b5b;--accent:#c08bff}}
:root[data-theme=light]{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--card:#fafafa;--ok:#1a8a4a;--no:#c0392b;--accent:#7b3fb3}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:0 auto;padding:26px}}
h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:19px;margin:30px 0 8px;border-bottom:2px solid var(--line);padding-bottom:5px}}
.sub{{color:var(--mut);margin:0 0 16px}}.mut{{color:var(--mut);font-weight:400;font-size:13px}}
.card{{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:14px 0;background:var(--card)}}
.card h3{{margin:0 0 10px;font-size:14px;font-weight:600;display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
.roll{{color:var(--accent);font-weight:700}}.pen{{color:var(--no);font-family:ui-monospace,Consolas,monospace;font-size:13px}}
.badge{{font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px}}
.badge.ok{{background:color-mix(in srgb,var(--ok) 20%,transparent);color:var(--ok)}}
.badge.no{{background:color-mix(in srgb,var(--no) 20%,transparent);color:var(--no)}}
.ba{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.ba img{{width:100%;border-radius:6px;display:block}}
figure{{margin:0}}figcaption{{color:var(--mut);font-size:12px;text-align:center;margin-top:4px}}
.declined{{display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;background:color-mix(in srgb,var(--no) 8%,transparent);border:1px dashed var(--no);border-radius:6px;color:var(--no);font-size:14px}}
.declined span{{color:var(--mut);font-size:12px}}
table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}}th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left}}
td.n{{text-align:right;font-family:ui-monospace,Consolas,monospace}}
.note{{background:var(--card);border-left:3px solid var(--accent);padding:10px 14px;border-radius:4px;font-size:14px;color:var(--mut)}}
</style>
<h1>Auto-straighten &mdash; before / after</h1>
<p class="sub">{len(cards)} confident, tilted Budapest photos ({n_str} straightened, {len(cards)-n_str} declined) &middot; spec-101 option A</p>
<div class="note"><b>How to read this.</b> Each photo is straightened only if the inscribed crop keeps
&ge;70% of the frame AND doesn't clip a prominent person. Declines stay as-shot and take the full tilt
penalty. Straightened ones take a small penalty = <code>0.4&times;(1&minus;retained area)</code>.
<b>Median retained area on straightened shots: {med*100:.0f}%</b>
(&rarr; typical fov penalty {FOV_W*(1-med):+.3f}).</div>
<h2>Weight-tuning table (straightened photos)</h2>
<table><thead><tr><th>photo</th><th>kept area</th><th>fov penalty</th></tr></thead><tbody>
{tune}
</tbody></table>
<h2>Gallery</h2>
{body}
""", encoding="utf-8")


if __name__ == "__main__":
    main()
