"""
spec-100: GeoCalib tilt report on the real Budapest album.

Runs GeoCalib on each photo (its ACTUAL tilt, not injected), then builds a
self-contained HTML report:
  1) table of detected tilt per image, sorted largest->smallest, rows link to detail
  2) per image: original vs straightened (rotated to level) view
  3) "evidence": GeoCalib's per-pixel up-vector field + horizon line overlay
  4) a step-by-step deep-dive on one high-tilt photo showing HOW roll is computed

Output: reports/2026-07-13_geocalib_budapest_tilt/{report.html,summary.md,img/}
Usage: python scripts/geocalib_budapest_report.py [--limit N]
"""

import argparse
import html
import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocalib import GeoCalib
from geocalib.utils import numpy_image_to_torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("geocalib_report")

SRC = Path(r"D:\Budapest2025_Google")
OUT = Path(__file__).resolve().parents[1] / "reports" / "2026-07-13_geocalib_budapest_tilt"
IMG = OUT / "img"
CARD_W = 560          # display width for original / straightened
_SIGN = -1.0          # GeoCalib roll -> spec-099 clockwise-positive convention (spec-100 T1)


def load_rgb(path: Path, work_side: int) -> np.ndarray:
    with Image.open(path) as pil:
        rgb = np.array(ImageOps.exif_transpose(pil).convert("RGB"))
    s = work_side / max(rgb.shape[:2])
    if s < 1:
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)))
    return rgb


def straighten(rgb: np.ndarray, roll_deg: float) -> np.ndarray:
    """Rotate content to level it. roll_deg>0 = content tilted clockwise -> rotate CCW."""
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), roll_deg, 1.0)  # + = CCW in cv2
    return cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def calibrate(model: GeoCalib, rgb: np.ndarray) -> dict:
    t = numpy_image_to_torch(rgb).to("cpu")
    with torch.no_grad():
        return model.calibrate(t)


def up_field_2hw(res: dict) -> np.ndarray:
    """Return up-field as (2,H,W) numpy, whatever GeoCalib's layout."""
    uf = res["up_field"]
    uf = uf.detach().cpu().numpy() if isinstance(uf, torch.Tensor) else np.asarray(uf)
    uf = np.squeeze(uf)
    if uf.shape[0] != 2:                    # (H,W,2) -> (2,H,W)
        uf = np.transpose(uf, (2, 0, 1))
    return uf


def draw_evidence(rgb: np.ndarray, res: dict, out_path: Path, width: int = 480):
    """Original with up-vector arrows (green) + horizon line (orange)."""
    uf = up_field_2hw(res)
    H, W = uf.shape[1:]
    disp = cv2.resize(rgb, (W, H))
    fig, ax = plt.subplots(figsize=(6, 6 * H / W), dpi=110)
    ax.imshow(disp)
    ax.set_axis_off()
    from geocalib import viz2d
    viz2d.plot_vector_fields([torch.from_numpy(uf)], axes=[ax], subsample=16, cmap="lime")
    try:
        viz2d.plot_horizon_lines([res["camera"]], [res["gravity"]], ax=[ax], lw=2.5)
    except Exception as e:  # horizon off-frame for large pitch
        log.debug("horizon skip: %s", e)
    fig.subplots_adjust(0, 0, 1, 1)
    s = width / W
    fig.set_size_inches(width / fig.dpi, (H * s) / fig.dpi)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def consensus_up(res: dict):
    """Mean up-vector and its angle from screen-vertical (~ the roll)."""
    uf = up_field_2hw(res)
    mx, my = float(uf[0].mean()), float(uf[1].mean())
    n = np.hypot(mx, my) or 1.0
    mx, my = mx / n, my / n
    ang = np.degrees(np.arctan2(mx, -my))   # deviation of 'up' from screen-up
    return (mx, my), ang


def make_explainer(model, stem, out_path):
    """4-panel figure: input -> up-field -> consensus/roll -> straightened."""
    p = next(SRC.rglob(stem + ".jp*"))
    rgb = load_rgb(p, 1024)
    res = calibrate(model, rgb)
    roll = _SIGN * float(torch.rad2deg(res["gravity"].roll))
    uf = up_field_2hw(res)
    H, W = uf.shape[1:]
    disp = cv2.resize(rgb, (W, H))
    (cx, cy), _ = consensus_up(res)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), dpi=115)
    for a in axes:
        a.set_axis_off()
    from geocalib import viz2d

    axes[0].imshow(disp)
    axes[0].set_title("1. Input photo\n(camera held crooked)", fontsize=12)

    axes[1].imshow(disp)
    viz2d.plot_vector_fields([torch.from_numpy(uf)], axes=[axes[1]], subsample=14, cmap="lime")
    axes[1].set_title("2. Predict 'up' at every pixel\n(from walls / people / verticals)",
                      fontsize=12)

    axes[2].imshow(disp)
    L = 0.40 * min(H, W)
    ox, oy = W * 0.5, H * 0.62
    # true screen-vertical reference (what "level" would be)
    axes[2].annotate("", xy=(ox, oy - L), xytext=(ox, oy),
                     arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, ls=(0, (5, 3))))
    axes[2].text(ox - 8, oy - L - 10, "screen up", color="white", fontsize=11,
                 ha="right", fontweight="bold")
    # consensus 'up' (what the arrows agree on)
    axes[2].annotate("", xy=(ox + cx * L, oy + cy * L), xytext=(ox, oy),
                     arrowprops=dict(arrowstyle="-|>", color="lime", lw=3.5))
    axes[2].text(ox + cx * L + 10, oy + cy * L, "scene up", color="lime", fontsize=11,
                 ha="left", fontweight="bold")
    # angle wedge between the two directions
    import matplotlib.patches as mpatches
    a_scene = np.degrees(np.arctan2(-(cy), cx))   # matplotlib angle (ccw from +x)
    axes[2].add_patch(mpatches.Wedge((ox, oy), L * 0.5, min(90, a_scene), max(90, a_scene),
                                     color="yellow", alpha=0.35))
    axes[2].text(ox + 14, oy - L * 0.35, f"{abs(roll):.1f} deg", color="yellow",
                 fontsize=13, fontweight="bold")
    axes[2].set_title("3. Angle between them = roll", fontsize=12)

    axes[3].imshow(straighten(disp, roll))
    axes[3].axhline(H * 0.5, color="orange", lw=1.5, ls="--", alpha=0.7)
    axes[3].set_title(f"4. Rotate by {-roll:+.1f} deg -> level", fontsize=12)

    fig.suptitle(f"How GeoCalib measures tilt  -  {stem}  (roll {roll:+.1f} deg)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=115)
    plt.close(fig)
    return roll


def build_html(records, explainer_stem, explainer_roll):
    def card(r):
        s = r["stem"]
        return f"""
<div class="card" id="card-{s}">
  <h3>#{r['rank']} &middot; {html.escape(s)} &middot;
      <span class="roll">roll {r['roll']:+.1f}&deg;</span>
      <span class="unc">&plusmn;{r['unc']:.1f}&deg; ({r['conflabel']})</span></h3>
  <div class="triptych">
    <figure><img src="img/{s}_orig.jpg" loading="lazy"><figcaption>original</figcaption></figure>
    <figure><img src="img/{s}_straight.jpg" loading="lazy"><figcaption>straightened ({-r['roll']:+.1f}&deg;)</figcaption></figure>
    <figure><img src="img/{s}_evidence.jpg" loading="lazy"><figcaption>evidence: up-field + horizon</figcaption></figure>
  </div>
  <a class="back" href="#top">&uarr; back to table</a>
</div>"""

    rows = "\n".join(
        f'<tr class="{"lowconf" if r["conflabel"]=="low" else ""}" '
        f'onclick="location.hash=\'#card-{r["stem"]}\'">'
        f'<td>{r["rank"]}</td>'
        f'<td><img class="thumb" src="img/{r["stem"]}_orig.jpg" loading="lazy"></td>'
        f'<td class="mono">{html.escape(r["stem"])}</td>'
        f'<td class="num roll">{r["roll"]:+.1f}&deg;</td>'
        f'<td class="num">&plusmn;{r["unc"]:.1f}&deg;</td>'
        f'<td>{r["conflabel"]}</td>'
        f'<td class="link">view &rarr;</td></tr>'
        for r in records)

    cards = "\n".join(card(r) for r in records)
    n = len(records)
    tilted = sum(abs(r["roll"]) >= 3 for r in records)
    return f"""<title>GeoCalib tilt - Budapest album</title>
<style>
:root {{ color-scheme: light dark; --bg:#fff; --fg:#1a1a1a; --mut:#666; --line:#e3e3e3;
        --card:#fafafa; --accent:#c0392b; --link:#1565c0; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141414; --fg:#e8e8e8; --mut:#9a9a9a; --line:#2c2c2c; --card:#1d1d1d;
           --accent:#ff6b5b; --link:#6ea8ff; }} }}
:root[data-theme=dark] {{ --bg:#141414; --fg:#e8e8e8; --mut:#9a9a9a; --line:#2c2c2c;
           --card:#1d1d1d; --accent:#ff6b5b; --link:#6ea8ff; }}
:root[data-theme=light] {{ --bg:#fff; --fg:#1a1a1a; --mut:#666; --line:#e3e3e3;
           --card:#fafafa; --accent:#c0392b; --link:#1565c0; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
        max-width:1040px; margin:0 auto; padding:24px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:20px; margin:34px 0 10px;
        border-bottom:2px solid var(--line); padding-bottom:6px; }}
.sub {{ color:var(--mut); margin:0 0 18px; }}
.mono {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; }}
.roll {{ color:var(--accent); font-weight:700; }} .unc {{ color:var(--mut); font-weight:400; font-size:13px; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0 4px; }}
th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
th {{ position:sticky; top:0; background:var(--bg); font-size:12px; text-transform:uppercase;
      letter-spacing:.04em; color:var(--mut); }}
tbody tr {{ cursor:pointer; }} tbody tr:hover {{ background:var(--card); }}
tr.lowconf {{ opacity:.42; }} tr.lowconf:hover {{ opacity:.7; }}
.note {{ background:var(--card); border-left:3px solid var(--accent); padding:8px 12px;
         border-radius:4px; font-size:13px; color:var(--mut); margin:8px 0 14px; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }} td.link {{ color:var(--link); }}
.thumb {{ width:64px; height:44px; object-fit:cover; border-radius:4px; display:block; }}
.wrap {{ overflow-x:auto; }}
.explain img {{ width:100%; border:1px solid var(--line); border-radius:8px; }}
.steps {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:14px 0; }}
.step {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
.step b {{ color:var(--accent); }}
.card {{ border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:16px 0;
         background:var(--card); scroll-margin-top:14px; }}
.card h3 {{ margin:0 0 10px; font-size:15px; font-weight:600; }}
.triptych {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
.triptych img {{ width:100%; border-radius:6px; display:block; }}
figure {{ margin:0; }} figcaption {{ color:var(--mut); font-size:12px; margin-top:4px; text-align:center; }}
.back {{ display:inline-block; margin-top:8px; color:var(--link); text-decoration:none; font-size:13px; }}
a {{ color:var(--link); }}
</style>

<div id="top"></div>
<h1>GeoCalib tilt detection &mdash; Budapest album</h1>
<p class="sub">{n} photos &middot; {tilted} with &ge;3&deg; detected tilt &middot; learned single-image
roll estimation (GeoCalib, ECCV 2024) &middot; spec-100</p>

<h2>How the method works</h2>
<div class="explain">
<p>A crooked photo has no metadata saying so &mdash; EXIF only records 90&deg; steps. GeoCalib recovers
the roll from the picture itself, but unlike edge/line tricks it reads <b>meaning</b>: it knows people
stand up, walls are vertical, horizons are level. Worked example below on a high-tilt shot
(<span class="mono">{html.escape(explainer_stem)}</span>, roll {explainer_roll:+.1f}&deg;):</p>
<div class="steps">
<div class="step"><b>1. Input</b><br>A photo taken with a tilted camera. The scene content is rotated.</div>
<div class="step"><b>2. Predict "up" everywhere</b><br>A neural net outputs, at <i>every pixel</i>, the
direction gravity points &mdash; the green arrows. It infers this from semantic cues, so a slanted
<i>painting</i> doesn't fool it the way a line detector would.</div>
<div class="step"><b>3. Measure the roll</b><br>All the arrows agree on one "scene up" (green).
The angle between that and the screen's true vertical (white dashed) <i>is</i> the roll. A geometric
optimiser fits the single camera roll that best explains the whole field.</div>
<div class="step"><b>4. Straighten</b><br>Rotate the image by the negative of that roll and the horizon
levels out.</div>
</div>
<img src="img/explainer.jpg" alt="step-by-step explainer">
</div>

<h2>Detected tilt per image (largest first)</h2>
<p class="sub">Click any row to jump to its original / straightened / evidence view.
Confidence = GeoCalib's roll uncertainty (smaller = surer).</p>
<p class="note"><b>Greyed rows are low-confidence.</b> The biggest reported angles are mostly
mirror / kaleidoscope shots that have no true "up" (uncertainty 10&ndash;30&deg;) &mdash; GeoCalib
abstains, and any tilt penalty would ignore them. Trust the non-grey rows.</p>
<div class="wrap"><table>
<thead><tr><th>#</th><th>thumb</th><th>file</th><th>roll</th><th>uncert.</th><th>conf</th><th></th></tr></thead>
<tbody>
{rows}
</tbody></table></div>

<h2>Per-image detail</h2>
{cards}

<script>
// respect the app's theme toggle if present
const mo=new MutationObserver(()=>{{}});mo.observe(document.documentElement,{{attributes:true}});
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--html-only", action="store_true",
                    help="rebuild report.html from the benchmark metrics + existing images")
    args = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)

    if args.html_only:
        import json
        bench = json.load(open(Path(__file__).resolve().parents[1] / "reports"
                               / "2026-07-12_geocalib_tilt" / "metrics.json"))
        records = [{"stem": r["photo"], "roll": r["base_angle"], "unc": r["base_unc"]}
                   for r in bench["rows"]]
        records.sort(key=lambda r: abs(r["roll"]), reverse=True)
        for k, r in enumerate(records, 1):
            r["rank"] = k
            r["conflabel"] = ("high" if r["unc"] <= 1.0 else "med" if r["unc"] <= 2.0 else "low")
        conf = [r for r in records if r["unc"] <= 1.5] or records
        ex = conf[0]
        OUT.joinpath("report.html").write_text(
            build_html(records, ex["stem"], ex["roll"]), encoding="utf-8")
        log.info("html-only rebuild done -> %s", OUT / "report.html")
        return

    model = GeoCalib().to("cpu").eval()
    photos = sorted(SRC.rglob("*.jpg")) + sorted(SRC.rglob("*.jpeg"))
    if args.limit:
        photos = photos[:args.limit]
    log.info("photos=%d", len(photos))

    records, t0 = [], time.time()
    for i, p in enumerate(photos):
        rgb = load_rgb(p, 1024)
        res = calibrate(model, rgb)
        roll = _SIGN * float(torch.rad2deg(res["gravity"].roll))
        unc = float(torch.rad2deg(res["roll_uncertainty"]))
        disp = cv2.resize(rgb, (CARD_W, int(CARD_W * rgb.shape[0] / rgb.shape[1])))
        cv2.imwrite(str(IMG / f"{p.stem}_orig.jpg"), cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(IMG / f"{p.stem}_straight.jpg"),
                    cv2.cvtColor(straighten(disp, roll), cv2.COLOR_RGB2BGR))
        draw_evidence(rgb, res, IMG / f"{p.stem}_evidence.jpg")
        records.append({"stem": p.stem, "roll": roll, "unc": unc})
        if (i + 1) % 10 == 0:
            log.info("%d/%d (%.1fs/img)", i + 1, len(photos), (time.time() - t0) / (i + 1))

    records.sort(key=lambda r: abs(r["roll"]), reverse=True)
    for k, r in enumerate(records, 1):
        r["rank"] = k
        r["conflabel"] = ("high" if r["unc"] <= 1.0 else "med" if r["unc"] <= 2.0 else "low")

    # explainer: highest-tilt photo that the model is confident about
    conf = [r for r in records if r["unc"] <= 1.5] or records
    ex = conf[0]
    ex_roll = make_explainer(model, ex["stem"], IMG / "explainer.jpg")
    log.info("explainer=%s roll=%.1f", ex["stem"], ex_roll)

    OUT.joinpath("report.html").write_text(
        build_html(records, ex["stem"], ex_roll), encoding="utf-8")
    top = records[:5]
    OUT.joinpath("summary.md").write_text(
        "# GeoCalib tilt on Budapest album (spec-100)\n\n"
        f"- {len(records)} photos scored; "
        f"{sum(abs(r['roll'])>=3 for r in records)} with >=3 deg detected tilt.\n"
        f"- Most tilted: " + ", ".join(f"{r['stem']} ({r['roll']:+.1f} deg)" for r in top) + "\n"
        f"- Explainer image: {ex['stem']} (roll {ex_roll:+.1f} deg).\n"
        "- Report: report.html (table sorted by tilt, original vs straightened, up-field evidence).\n",
        encoding="utf-8")
    log.info("done -> %s", OUT / "report.html")


if __name__ == "__main__":
    main()
