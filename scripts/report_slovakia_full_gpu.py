"""Build the Stage A judging report for the full-roll Slovakia26 GPU run (spec-102 / spec-110).

Reads reports/_modal_job_slovakia_full.json (album.json from the Modal job flow) and the originals in
D:/Slovakia26, writes reports/<date>_slovakia_full_gpu/{report.html, summary.md, imgs/} with downscaled
copies of EVERY pick and EVERY set-aside photo (the owner judges keep/not per pick), plus the first
near-duplicate each pick beat, so the dedup decisions can be eyeballed too.
"""
from __future__ import annotations

import collections
import html
import json
import logging
import sys
from pathlib import Path

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
SRC = Path("D:/Slovakia26")
RESULT = REPO / "reports" / "_modal_job_slovakia_full.json"
OUT = REPO / "reports" / "2026-10-09_slovakia_full_gpu"
THUMB = 320


def thumb(stem: str, size: int = THUMB) -> str | None:
    src = next(SRC.glob(stem + ".*"), None)
    if src is None:
        return None
    dst = OUT / "imgs" / f"{stem}.jpg"
    if not dst.exists():
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size))
            im.save(dst, "JPEG", quality=80)
    return f"imgs/{stem}.jpg"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    (OUT / "imgs").mkdir(parents=True, exist_ok=True)
    d = json.loads(RESULT.read_text(encoding="utf-8"))
    run = d["runs"][0]
    album = run["album"]
    timing = run["timing"]
    steps = {s["step"]: s["ms"] / 1000 for s in timing["steps"]}
    n = run["n_inputs"]
    picks = album["picks"]
    dropped = album["dropped"]
    by_reason = collections.Counter(x["reason"] for x in dropped)
    staged = [Path(x).stem for x in d["staged"]["staged"]]
    in_album = {p["id"] for p in picks} | {x["id"] for x in dropped}
    missing = sorted(set(staged) - in_album)
    people = sorted((p["face_count"] for p in album["people"]), reverse=True)
    set_aside = [x for x in dropped if x["reason"] == "set_aside"]
    near = {x["id"]: x for x in dropped if x["reason"] == "near_dup"}

    # --- summary.md ---
    top = sorted(steps.items(), key=lambda kv: -kv[1])[:6]
    summary = f"""# Slovakia26 full roll on the Modal L4 worker (2026-10-09)

**Goal**: produce the Stage A quality-gate input (ROADMAP Stage A, spec-102 updates): the real
`default_pipeline` over the WHOLE 715-photo trip, on the GPU worker, as `album.json`, so the owner can
judge the picks against their own keeper list. Also the first at-scale run of the spec-110 job flow.

**Method**: `scripts/modal_curate_driver.py --src D:/Slovakia26 --n 800 --job-id slovakia_full`
(768 px downsample, EXIF kept, staged to the `albumify-jobs` Volume, `curate_job` on an L4 with
protobuf 4.25 + per-step `device=cuda`, deterministic torch). Full result:
`reports/_modal_job_slovakia_full.json`; originals on `D:\\Slovakia26`; displays in the job store.

## Outcome
| | |
|---|---|
| photos sent | {n} (97 MB at 768 px) |
| pipeline time | {timing['total_ms']/1000:.0f} s = **{timing['total_ms']/1000/n:.2f} s/img**; wall {run.get('wall_s', 0):.0f} s incl. upload |
| picks | **{len(picks)}** across {album['counts']['scenes']} scene clusters (max 2 per cluster, no global cap yet) |
| near-duplicates dropped | {by_reason.get('near_dup', 0)} |
| set aside by triage | {by_reason.get('set_aside', 0)} (screenshots, receipts, signs, posters) |
| people clusters | {len(people)}; largest {people[:5]} |
| **missing from album.json** | **{len(missing)}** of {n} — removed before scoring (`filter_quality`), invisible (SIGHTING-128) |

Slowest steps (s, whole roll): {', '.join(f'{k} {v:.0f}' for k, v in top)}.

## What this says
- The worker is production-shaped: {timing['total_ms']/1000/n:.2f} s/img all-in; 100 photos ≈ 1 min warm.
- 147 picks is not an album: selection is per scene cluster with no total cap. The Stage B item
  `album_max_picks` (global top-K) is the next engine change; Stage A can still be judged as
  "would you keep each of these" + "which keepers of yours are missing, and what reason did we give".
- {len(missing)} photos silently vanish (dropped by the quality filter before scene clustering); the
  show-everything principle requires them in `dropped` with a reason. SIGHTING-128.
- {len(people)} people clusters with the largest at {people[0]} faces on a family trip = identities
  over-split at this scale; spec-111 checkpoint #1 (merge candidates) is the designed fix.

## Stage A — what the owner does next
Open `report.html`, mark each pick keep / not, list keepers of yours that are absent (then find them in
the Dropped section and note the reason given). That verdict + the missing-keeper reasons = Gate 1.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")

    # --- report.html ---
    def card(stem: str, caption: str, cls: str = "") -> str:
        src = thumb(stem)
        img = f'<img src="{src}" loading="lazy" alt="{stem}">' if src else '<div class="noimg">no file</div>'
        return f'<figure class="{cls}">{img}<figcaption>{html.escape(caption)}</figcaption></figure>'

    pick_cards = []
    for p in picks:
        sc = p["score"]
        beat = p.get("beat") or []
        cap = (f"{p['id']}  ·  composite {sc.get('composite', 0):.2f}  ·  sharp {sc.get('sharpness') or 0:.2f}  "
               f"aes {sc.get('aesthetic') or 0:.2f}  ·  scene {p['scene_cluster']}  ·  beat {len(beat)}")
        pick_cards.append(card(p["id"], cap, "pick"))
    by_day = collections.defaultdict(list)
    for p, c in zip(picks, pick_cards):
        by_day[p["id"][:8]].append(c)
    pick_sections = "".join(
        f"<h3>{day[:4]}-{day[4:6]}-{day[6:]} · {len(cards)} picks</h3><div class='grid'>{''.join(cards)}</div>"
        for day, cards in sorted(by_day.items()))

    aside_cards = "".join(card(x["id"], f"{x['id']} · {x.get('reason_detail', '')}", "aside") for x in set_aside)
    # near-dup pairs: first 24 picks that beat something
    pair_cards = []
    for p in picks:
        beat = p.get("beat") or []
        if beat and len(pair_cards) < 24:
            b = beat[0]
            pair_cards.append(f"<div class='pair'>{card(p['id'], 'KEPT ' + p['id'])}{card(b, 'dropped ' + b)}</div>")
    missing_list = ", ".join(missing)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Slovakia26 full roll — GPU worker — Stage A judging sheet</title>
<style>
:root{{--bg:#F5F0E7;--card:#fff;--line:#E4DACB;--ink:#1B1714;--dim:#4A4239;--accent:#A96E12;--crit:#B25B3C;--good:#3E7A46;color-scheme:light}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0E0D0C;--card:#17130F;--line:#2C2620;--ink:#F2EEE7;--dim:#C8BFB2;--accent:#E8B04B;--crit:#D98A6A;--good:#7FB682;color-scheme:dark}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:1240px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:20px;margin:36px 0 8px;border-bottom:1px solid var(--line)}} h3{{font-size:15px;margin:18px 0 6px;color:var(--dim)}}
table{{border-collapse:collapse;margin:12px 0}} td,th{{padding:5px 12px;border-bottom:1px solid var(--line);text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
figure img{{width:100%;height:170px;object-fit:cover;display:block}} figcaption{{font:11.5px/1.35 ui-monospace,monospace;padding:6px 8px;color:var(--dim)}}
figure.aside{{border-color:var(--crit)}} .pair{{display:grid;grid-template-columns:1fr 1fr;gap:4px;border:1px dashed var(--line);padding:4px;border-radius:8px}}
.note{{background:var(--card);border-left:3px solid var(--accent);padding:10px 14px;margin:10px 0}} .small{{font-size:12px;color:var(--dim)}}
</style></head><body>
<h1>Slovakia26 — full roll on the Modal L4 worker</h1>
<p class="small">2026-10-09 · {n} photos · pipeline {timing['total_ms']/1000:.0f} s ({timing['total_ms']/1000/n:.2f} s/img) · album.json from <code>curate_job(slovakia_full)</code> · originals <code>D:\\Slovakia26</code> · full data <code>reports/_modal_job_slovakia_full.json</code></p>
<div class="note"><b>Stage A judging sheet.</b> For each pick below: keep or not? Then list the photos you would have kept that are absent, find them under Dropped, and note the reason given. That verdict is Gate 1 (ROADMAP Stage A).</div>
<h2>Outcome</h2>
<table><tr><th>sent</th><td>{n}</td></tr><tr><th>picks</th><td><b>{len(picks)}</b> across {album['counts']['scenes']} scenes (max 2 per scene, no global cap yet)</td></tr>
<tr><th>near-duplicates dropped</th><td>{by_reason.get('near_dup', 0)}</td></tr><tr><th>set aside by triage</th><td>{by_reason.get('set_aside', 0)}</td></tr>
<tr><th>people clusters</th><td>{len(people)} (largest {people[:5]})</td></tr>
<tr><th>missing from album.json</th><td><b>{len(missing)}</b> — removed by the quality filter before scene clustering, no record (SIGHTING-128)</td></tr>
<tr><th>slowest steps (s)</th><td>{', '.join(f'{k} {v:.0f}' for k, v in top)}</td></tr></table>
<h2>Picks — {len(picks)} — by day</h2>{pick_sections}
<h2>Set aside by triage — {len(set_aside)}</h2><p class="small">Should any of these be in the album? They are promotable in the product.</p><div class="grid">{aside_cards}</div>
<h2>Near-duplicate decisions — first 24 pairs</h2><p class="small">Left = kept, right = the first photo it beat in the same scene.</p><div class="grid">{''.join(pair_cards)}</div>
<h2>Missing from album.json — {len(missing)}</h2><p class="small">{missing_list}</p>
</body></html>"""
    (OUT / "report.html").write_text(page, encoding="utf-8")
    size = sum(p.stat().st_size for p in (OUT / "imgs").glob("*.jpg")) / 1e6
    logger.info("report: %s (%d thumbs, %.1f MB)", OUT / "report.html", len(list((OUT / 'imgs').glob('*.jpg'))), size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
