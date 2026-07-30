"""Spec-102 T4.1 — the blind A/B judging page.

Loads both arms' `AlbumResult` JSONs and renders ONE self-contained HTML: two albums shown in
their intended order, side by side, labelled only "Album A / Album B" with left/right assignment
shuffled. The arm->label mapping is seeded by the trip name (reproducible) and written to a
SEPARATE unblind-key file the rater must not open until judging is done — so the page itself
carries no tell of which album is Albumify.

The page captures the rater sheet (PROMPTS.md v1): forced overall pick, the "one moment", the
story-vs-slideshow axis, 5-pt sub-ratings per album, and the human-rescue note. A "copy answers"
button emits JSON to paste into `judging.json`. Solo N=1 pilot; images are embedded as base64 so
the page is portable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _seeded_swap(trip: str) -> bool:
    """Deterministic per-trip coin flip (no Date/random): does Albumify go on the RIGHT?"""
    return hashlib.sha256(trip.encode()).digest()[0] % 2 == 1


def _b64(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.standard_b64encode(path.read_bytes()).decode()


def _album_column(label: str, order: list[str], imgs_dir: Path) -> str:
    tiles = []
    for i, stem in enumerate(order, 1):
        p = imgs_dir / f"{stem}.jpg"
        src = _b64(p) if p.exists() else ""
        tiles.append(
            f'<figure class="shot"><span class="idx">{i}</span>'
            f'<img loading="lazy" src="{src}" alt="{stem}"></figure>'
        )
    return (
        f'<section class="album"><h2>Album {label}</h2>'
        f'<div class="strip">{"".join(tiles)}</div></section>'
    )


def build_ab_html(trip: str, imgs_dir: Path, left_label: str, left_order: list[str],
                  right_label: str, right_order: list[str]) -> str:
    left = _album_column(left_label, left_order, imgs_dir)
    right = _album_column(right_label, right_order, imgs_dir)
    return _TEMPLATE.replace("{{TRIP}}", trip).replace("{{LEFT}}", left).replace("{{RIGHT}}", right)


def generate_ab_viewer(trip: str, imgs_dir: Path, albumify_json: Path, vlm_json: Path,
                       out_html: Path, out_key: Path) -> None:
    alb = json.loads(albumify_json.read_text(encoding="utf-8"))
    vlm = json.loads(vlm_json.read_text(encoding="utf-8"))
    if alb["input_set_hash"] != vlm["input_set_hash"]:
        raise ValueError("A1 fairness violated: arms ran on different input sets (hash mismatch)")

    albumify_right = _seeded_swap(trip)
    # The page labels are A (left) / B (right). Assign arms to left/right by the seeded swap.
    if albumify_right:
        left_arm, right_arm = ("vlm", vlm), ("albumify", alb)
    else:
        left_arm, right_arm = ("albumify", alb), ("vlm", vlm)

    html = build_ab_html(
        trip, imgs_dir,
        "A", left_arm[1]["order"], "B", right_arm[1]["order"],
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    key = {
        "trip": trip,
        "Album A (left)": left_arm[0],
        "Album B (right)": right_arm[0],
        "note": "UNBLIND KEY — do not open until judging.json is written.",
        "input_set_hash": alb["input_set_hash"],
    }
    out_key.write_text(json.dumps(key, indent=2), encoding="utf-8")
    logger.info("wrote A/B viewer -> %s (key -> %s)", out_html, out_key)


_TEMPLATE = r"""<!-- spec-102 blind A/B judging page -->
<style>
  :root{--bg:#faf8f5;--fg:#1c1a17;--muted:#6b6560;--line:#e2ddd6;--card:#fff;--accent:#b5623a;}
  @media (prefers-color-scheme:dark){:root{--bg:#17150f;--fg:#ece7df;--muted:#9a938a;--line:#332e26;--card:#201d16;--accent:#e08a5a;}}
  :root[data-theme="dark"]{--bg:#17150f;--fg:#ece7df;--muted:#9a938a;--line:#332e26;--card:#201d16;--accent:#e08a5a;}
  :root[data-theme="light"]{--bg:#faf8f5;--fg:#1c1a17;--muted:#6b6560;--line:#e2ddd6;--card:#fff;--accent:#b5623a;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:20px 24px;border-bottom:1px solid var(--line)}
  header h1{margin:0 0 4px;font-size:19px}
  header p{margin:0;color:var(--muted);font-size:13px}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px 24px}
  .album h2{position:sticky;top:0;background:var(--bg);margin:0;padding:8px 0;font-size:16px;border-bottom:2px solid var(--accent)}
  .strip{display:flex;flex-direction:column;gap:10px;margin-top:10px}
  .shot{position:relative;margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .shot img{display:block;width:100%;height:auto}
  .idx{position:absolute;top:6px;left:6px;background:rgba(0,0,0,.7);color:#fff;font-size:12px;font-weight:600;padding:2px 7px;border-radius:10px}
  form{padding:16px 24px 40px;max-width:820px}
  fieldset{border:1px solid var(--line);border-radius:10px;margin:0 0 16px;padding:14px 16px;background:var(--card)}
  legend{padding:0 6px;font-weight:600;color:var(--accent)}
  label{display:block;margin:8px 0 4px}
  textarea,input[type=text]{width:100%;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
  .row{display:flex;gap:20px;flex-wrap:wrap}
  .rate{display:flex;align-items:center;gap:6px}
  button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:10px 18px;font-weight:600;cursor:pointer;font-size:14px}
  pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-top:12px}
  @media (max-width:720px){.cols{grid-template-columns:1fr}}
</style>
<header>
  <h1>Blind album comparison — {{TRIP}}</h1>
  <p>Two albums, shown in their intended order. You don't know which system made which. Spend ~10s
     forming a gut reaction before analysing. Do NOT open the unblind key until you've saved answers.</p>
</header>
<div class="cols">{{LEFT}}{{RIGHT}}</div>
<form id="f">
  <fieldset><legend>1. Overall — forced choice</legend>
    <div class="row">
      <label class="rate"><input type="radio" name="winner" value="A"> Album A is better</label>
      <label class="rate"><input type="radio" name="winner" value="B"> Album B is better</label>
    </div>
    <label>2. The one moment — what single image/moment made you pick it?</label>
    <textarea name="one_moment" rows="2"></textarea>
  </fieldset>
  <fieldset><legend>3. Story vs slideshow (1 = pile of nice photos … 5 = a story)</legend>
    <div class="row">
      <span class="rate">Album A <input type="number" name="story_A" min="1" max="5" step="1"></span>
      <span class="rate">Album B <input type="number" name="story_B" min="1" max="5" step="1"></span>
    </div>
  </fieldset>
  <fieldset><legend>4. Sub-ratings (1–5 each, per album)</legend>
    <div class="row" id="subs"></div>
  </fieldset>
  <fieldset><legend>5. Human rescue</legend>
    <label>Up to 5 photos BOTH albums missed that you'd add back (filename + why):</label>
    <textarea name="rescue" rows="3"></textarea>
  </fieldset>
  <button type="button" onclick="dump()">Copy answers as JSON</button>
  <pre id="out"></pre>
</form>
<script>
  const AXES=["coverage","quality","redundancy","narrative"];
  const subs=document.getElementById("subs");
  for(const ax of AXES){for(const al of ["A","B"]){
    const s=document.createElement("span");s.className="rate";
    s.innerHTML=`${ax} ${al} <input type="number" name="${ax}_${al}" min="1" max="5" step="1">`;
    subs.appendChild(s);}}
  function dump(){
    const d=new FormData(document.getElementById("f"));const o={};
    for(const [k,v] of d.entries())o[k]=v;
    document.getElementById("out").textContent=JSON.stringify(o,null,2);
  }
</script>
"""
