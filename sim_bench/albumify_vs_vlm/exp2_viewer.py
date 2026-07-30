"""Spec-102 EXP-2 — the blind best-frame picker page.

One self-contained HTML: for each scene cluster, every frame is shown (order shuffled by a seeded
hash so position carries no tell of which frame a system picked), and the rater clicks the SINGLE
best. A "copy answers" button emits `{"cluster_<id>": "<stem>", ...}` to paste into
`exp2_judging.json`. The page is blind by construction — it never marks which frame Albumify or the
VLM chose. Images embedded as base64 for portability. Solo N=1 pilot (D6), upgradeable to 3 raters.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _b64(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.standard_b64encode(path.read_bytes()).decode()


def _shuffled(trip: str, cid: int, stems: list[str]) -> list[str]:
    """Deterministic per-cluster frame order (no Date/random): sort by a seeded hash."""
    return sorted(stems, key=lambda s: hashlib.sha256(f"{trip}:{cid}:{s}".encode()).hexdigest())


def _cluster_block(trip: str, cid: int, stems: list[str], imgs_dir: Path) -> str:
    tiles = []
    for stem in _shuffled(trip, cid, stems):
        p = imgs_dir / f"{stem}.jpg"
        src = _b64(p) if p.exists() else ""
        # The stem lives only in the radio value (not shown), so the choice maps back to a frame
        # without cueing the rater. Clicking anywhere on the tile selects it.
        tiles.append(
            f'<label class="shot"><input type="radio" name="cluster_{cid}" value="{stem}">'
            f'<img loading="lazy" src="{src}" alt=""><span class="tick">✓ best</span></label>'
        )
    return (
        f'<section class="cluster"><h2>Cluster {cid} '
        f'<span class="cnt">{len(stems)} frames — pick the one best</span></h2>'
        f'<div class="grid">{"".join(tiles)}</div></section>'
    )


def generate_exp2_viewer(trip: str, imgs_dir: Path, exp2_json: Path, out_html: Path) -> None:
    from sim_bench.albumify_vs_vlm.exp2 import load_exp2

    res = load_exp2(exp2_json)
    blocks = "".join(
        _cluster_block(trip, c.cluster_id, c.stems, imgs_dir) for c in res.cases
    )
    html = _TEMPLATE.replace("{{TRIP}}", trip).replace("{{N}}", str(len(res.cases))).replace(
        "{{BLOCKS}}", blocks
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    logger.info("wrote EXP-2 picker -> %s (%d clusters)", out_html, len(res.cases))


_TEMPLATE = r"""<!-- spec-102 EXP-2 blind best-frame picker -->
<style>
  :root{--bg:#faf8f5;--fg:#1c1a17;--muted:#6b6560;--line:#e2ddd6;--card:#fff;--accent:#b5623a;--good:#2e7d52;}
  @media (prefers-color-scheme:dark){:root{--bg:#17150f;--fg:#ece7df;--muted:#9a938a;--line:#332e26;--card:#201d16;--accent:#e08a5a;--good:#5fbf8f;}}
  :root[data-theme="dark"]{--bg:#17150f;--fg:#ece7df;--muted:#9a938a;--line:#332e26;--card:#201d16;--accent:#e08a5a;--good:#5fbf8f;}
  :root[data-theme="light"]{--bg:#faf8f5;--fg:#1c1a17;--muted:#6b6560;--line:#e2ddd6;--card:#fff;--accent:#b5623a;--good:#2e7d52;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:20px 24px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
  header h1{margin:0 0 4px;font-size:19px}
  header p{margin:0;color:var(--muted);font-size:13px}
  header .prog{font-weight:600;color:var(--accent)}
  .cluster{padding:16px 24px;border-bottom:1px solid var(--line)}
  .cluster h2{margin:0 0 10px;font-size:16px}
  .cluster .cnt{color:var(--muted);font-weight:400;font-size:13px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
  .shot{position:relative;margin:0;border:2px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer;display:block}
  .shot img{display:block;width:100%;height:auto}
  .shot input{position:absolute;opacity:0;pointer-events:none}
  .shot .tick{position:absolute;top:6px;left:6px;background:var(--good);color:#fff;font-size:12px;font-weight:700;padding:2px 8px;border-radius:10px;opacity:0}
  .shot:has(input:checked){border-color:var(--good);box-shadow:0 0 0 2px var(--good)}
  .shot:has(input:checked) .tick{opacity:1}
  .bar{padding:16px 24px 44px}
  button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:10px 18px;font-weight:600;cursor:pointer;font-size:14px}
  pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-top:12px}
</style>
<header>
  <h1>EXP-2 — pick the best frame per cluster · {{TRIP}}</h1>
  <p>For each of {{N}} clusters, click the single frame you'd keep for the album. You don't know
     which frame either system picked. <span class="prog" id="prog">0 / {{N}} chosen</span></p>
</header>
<form id="f">{{BLOCKS}}
  <div class="bar">
    <button type="button" onclick="dump()">Copy answers as JSON</button>
    <pre id="out"></pre>
  </div>
</form>
<script>
  const form=document.getElementById("f");
  const total={{N}};
  function upd(){
    const names=new Set();
    form.querySelectorAll("input[type=radio]:checked").forEach(i=>names.add(i.name));
    document.getElementById("prog").textContent=names.size+" / "+total+" chosen";
  }
  form.addEventListener("change",upd);
  function dump(){
    const d=new FormData(form);const o={};
    for(const [k,v] of d.entries())o[k]=v;
    document.getElementById("out").textContent=JSON.stringify(o,null,2);
  }
</script>
"""
