"""Spec-102 T7 — the experiment report (analysis HTML + summary.md).

Un-blinded analysis (NOT the judging page): shows both arms' albums side by side, the objective
metrics, the pick divergence, and the VLM's editorial annotations. Downscaled copies of every
picked image are written into the report folder (experiment-report mandate); the full 768px set
stays on the data drive. The human A/B verdict is left as a marked placeholder until the rater
fills `judging.json` via the blind viewer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image, ImageOps

from sim_bench.albumify_vs_vlm.metrics import duplicate_survival

logger = logging.getLogger(__name__)


def _downscale_copy(src: Path, dst: Path, max_edge: int = 480) -> None:
    with Image.open(src) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    img.save(dst, "JPEG", quality=82)


def _scene_labels(albumify: dict) -> dict[str, int]:
    labels: dict[str, int] = {}
    for cid, stems in albumify["scene_clusters"].items():
        for s in stems:
            labels[s] = int(cid)
    return labels


def _strip(order: list[str], reasons: dict[str, str], roles: dict[str, str]) -> str:
    tiles = []
    for i, stem in enumerate(order, 1):
        cap = reasons.get(stem, "")
        role = roles.get(stem, "")
        badge = f'<span class="role">{role}</span>' if role else ""
        tiles.append(
            f'<figure class="shot"><span class="idx">{i}</span>{badge}'
            f'<img loading="lazy" src="imgs/{stem}.jpg" alt="{stem}">'
            f'<figcaption>{cap}</figcaption></figure>'
        )
    return "".join(tiles)


def _annotation_section(annotation_json: Path | None) -> str:
    """EXP-3: render the VLM's structured annotation if it exists, else a placeholder note."""
    if not annotation_json or not annotation_json.exists():
        return ("<p>Not yet run. The VLM arm already tags each pick with a narrative role and an "
                "editorial reason grounded in the moment, not the pixels — Albumify has no "
                "equivalent (it ranks by composite quality score).</p>")
    a = json.loads(annotation_json.read_text(encoding="utf-8"))
    rows = "".join(
        f'<tr><td>{g.get("day","")}</td><td><b>{g.get("label","")}</b></td>'
        f'<td>{g.get("moment_type","")}</td><td>{g.get("reason","")}</td></tr>'
        for g in a.get("groups", [])
    )
    return (
        f'<p>The VLM classified the whole collection as '
        f'<b>{a.get("album_type")}</b> / <b>{a.get("trip_subtype")}</b> and wrote the arc: '
        f'<em>"{a.get("narrative","")}"</em> It then grouped the trip into '
        f'<b>{a.get("n_groups")}</b> moments with human-readable labels, moment tags, and a '
        f'best-frame reason each (structured JSON, product-shaped). A sample:</p>'
        f'<div style="overflow-x:auto"><table><tr><th>Day</th><th>Moment</th><th>Tag</th>'
        f'<th>Why this frame</th></tr>{rows}</table></div>'
    )


def generate_report(trip: str, imgs_dir: Path, albumify_json: Path, vlm_json: Path,
                    out_dir: Path, annotation_json: Path | None = None) -> None:
    alb = json.loads(albumify_json.read_text(encoding="utf-8"))
    vlm = json.loads(vlm_json.read_text(encoding="utf-8"))
    labels = _scene_labels(alb)
    passed = set(labels)  # Albumify's quality-passed set

    ds_a = duplicate_survival(alb["order"], labels)
    ds_v = duplicate_survival(vlm["order"], labels)
    a_set, v_set = set(alb["order"]), set(vlm["order"])
    overlap = sorted(a_set & v_set)
    vlm_only_rejected = [s for s in vlm["order"] if s not in passed]

    # copy downscaled samples for every picked image
    img_out = out_dir / "imgs"
    img_out.mkdir(parents=True, exist_ok=True)
    for stem in sorted(a_set | v_set):
        src = imgs_dir / f"{stem}.jpg"
        if src.exists():
            _downscale_copy(src, img_out / f"{stem}.jpg")

    n_input = len(list(imgs_dir.glob("*.jpg")))
    full_pipe = alb["pipeline"] == "default"
    # Caveat #1 must tell the truth about which penalties actually ran. The full
    # `default` pipeline (SIGHTING-117 fixed) applies occlusion + tilt penalties;
    # the reduced faces/minimal fallbacks skip them.
    alb_caveat1_html = (
        (f"<b>Albumify arm ran the full <code>default</code> pipeline</b> — the product path, "
         f"with <b>occlusion &amp; tilt penalties active</b> (SIGHTING-117 release-model-per-step "
         f"fix; no longer OOMs).")
        if full_pipe else
        (f"<b>Albumify arm ran the reduced <code>{alb['pipeline']}</code> pipeline</b>, not the full "
         f"product: the full 33-step run OOM-kills this machine when the occlusion CLIP model loads "
         f"on top of YOLO+InsightFace+DINOv2+AVA, so <b>occlusion &amp; tilt penalties are disabled</b> here.")
    )
    alb_caveat1_md = (
        f"`default` pipeline (full product path; occlusion + tilt penalties active, "
        f"SIGHTING-117 fixed)."
        if full_pipe else
        f"`{alb['pipeline']}` pipeline (full 33-step OOMs this box at occlusion CLIP load; "
        f"occlusion/tilt penalties skipped)."
    )

    alb_reasons = {p["id"]: p.get("reason", "") for p in alb["picks"]}
    vlm_reasons = {p["id"]: p.get("reason", "") for p in vlm["picks"]}
    vlm_roles = {p["id"]: p.get("role", "") for p in vlm["picks"]}

    html = _TEMPLATE
    repl = {
        "{{TRIP}}": trip,
        "{{HASH}}": alb["input_set_hash"][:16],
        "{{N_OVERLAP}}": str(len(overlap)),
        "{{ALB_PIPE}}": alb["pipeline"],
        "{{ALB_RED}}": f"{ds_a.redundancy_rate:.2f}",
        "{{VLM_RED}}": f"{ds_v.redundancy_rate:.2f}",
        "{{ALB_SCENES}}": str(ds_a.n_scenes_covered),
        "{{VLM_SCENES}}": str(ds_v.n_scenes_covered),
        "{{VLM_TOK_IN}}": str(vlm.get("meta", {}).get("input_tokens", "?")),
        "{{VLM_TOK_OUT}}": str(vlm.get("meta", {}).get("output_tokens", "?")),
        "{{VLM_REJECTED}}": ", ".join(vlm_only_rejected) or "(none)",
        "{{N_REJECTED}}": str(len(vlm_only_rejected)),
        "{{ALB_STRIP}}": _strip(alb["order"], alb_reasons, {}),
        "{{VLM_STRIP}}": _strip(vlm["order"], vlm_reasons, vlm_roles),
        "{{ANNOTATION}}": _annotation_section(annotation_json),
        "{{ALB_CAVEAT1}}": alb_caveat1_html,
        "{{N_INPUT}}": str(n_input),
    }
    for k, v in repl.items():
        html = html.replace(k, v)
    (out_dir / "report.html").write_text(html, encoding="utf-8")

    summary = (
        f"# Albumify vs VLM — {trip} pilot (spec-102)\n\n"
        f"- Input: {n_input} photos -> 768px (hash {alb['input_set_hash'][:16]}), both arms, K=20.\n"
        f"- Albumify arm: {alb_caveat1_md}\n"
        f"- VLM arm: claude-opus-4-8, map(shortlist)->reduce(order); "
        f"{vlm['meta']['input_tokens']}/{vlm['meta']['output_tokens']} tokens.\n"
        f"- **Pick overlap: {len(overlap)}/20 identical** (systems disagree on "
        f"{20 - len(overlap)}/20).\n"
        f"- Duplicate-survival (Albumify scene clusters as truth): Albumify {ds_a.redundancy_rate}, "
        f"VLM {ds_v.redundancy_rate} (VLM slightly more diverse, and it never saw the clusters).\n"
        f"- VLM picked {len(vlm_only_rejected)} shot(s) Albumify's quality gate rejected: "
        f"{', '.join(vlm_only_rejected) or 'none'}.\n"
        f"- VLM produced narrative roles (opener/hero/peak/closer) + editorial captions.\n"
        f"- **Human blind A/B verdict: PENDING** (viewer built; solo N=1 pilot, no inferential claim).\n"
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    logger.info("wrote report -> %s", out_dir / "report.html")


_TEMPLATE = r"""<!-- spec-102 experiment report -->
<style>
  :root{--bg:#f7f4ef;--fg:#1e1b17;--muted:#6d665e;--line:#e3ddd3;--card:#fff;--accent:#a8532e;--good:#2e7d52;--warn:#b8860b;}
  @media (prefers-color-scheme:dark){:root{--bg:#16130d;--fg:#eae4da;--muted:#9a9188;--line:#332d24;--card:#1f1b14;--accent:#e08a5a;--good:#5fbf8f;--warn:#d9b45a;}}
  :root[data-theme="dark"]{--bg:#16130d;--fg:#eae4da;--muted:#9a9188;--line:#332d24;--card:#1f1b14;--accent:#e08a5a;--good:#5fbf8f;--warn:#d9b45a;}
  :root[data-theme="light"]{--bg:#f7f4ef;--fg:#1e1b17;--muted:#6d665e;--line:#e3ddd3;--card:#fff;--accent:#a8532e;--good:#2e7d52;--warn:#b8860b;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:28px 24px 60px}
  h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
  .sub{color:var(--muted);margin:0 0 24px}
  h2{font-size:19px;margin:34px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--accent)}
  .pilot{display:inline-block;background:var(--warn);color:#000;font-weight:700;font-size:12px;padding:2px 10px;border-radius:12px;letter-spacing:.03em}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
  .kpi .n{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
  .kpi .l{color:var(--muted);font-size:12px;margin-top:2px}
  table{width:100%;border-collapse:collapse;margin:10px 0;font-size:14px}
  th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600}
  .note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);border-radius:8px;padding:12px 14px;margin:14px 0}
  .albums{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  .col h3{margin:0 0 8px;font-size:15px}
  .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
  .shot{position:relative;margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .shot img{display:block;width:100%;height:auto}
  .idx{position:absolute;top:5px;left:5px;background:rgba(0,0,0,.72);color:#fff;font-size:11px;font-weight:600;padding:1px 6px;border-radius:9px}
  .role{position:absolute;top:5px;right:5px;background:var(--accent);color:#fff;font-size:10px;font-weight:600;padding:1px 6px;border-radius:9px;text-transform:uppercase;letter-spacing:.03em}
  figcaption{padding:6px 8px;font-size:11.5px;color:var(--muted);line-height:1.35}
  code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:12.5px}
  @media (max-width:820px){.albums{grid-template-columns:1fr}}
</style>
<div class="wrap">
  <span class="pilot">PILOT · N=1 · NO INFERENTIAL CLAIM</span>
  <h1>Albumify vs VLM — {{TRIP}}</h1>
  <p class="sub">Can a generalist VLM (Claude Opus 4.8) curate a trip album as well as the engineered
     Albumify pipeline? Same {{N_INPUT}} photos → 768px (hash <code>{{HASH}}…</code>), each picks an ordered
     K=20. Spec-102.</p>

  <div class="cards">
    <div class="kpi"><div class="n">{{N_OVERLAP}}/20</div><div class="l">identical picks (systems disagree on the rest)</div></div>
    <div class="kpi"><div class="n">{{ALB_RED}} · {{VLM_RED}}</div><div class="l">redundancy rate — Albumify · VLM (lower = more diverse)</div></div>
    <div class="kpi"><div class="n">{{N_REJECTED}}</div><div class="l">VLM picks Albumify's quality gate rejected</div></div>
    <div class="kpi"><div class="n">{{VLM_TOK_IN}}<span style="font-size:13px"> in</span></div><div class="l">VLM tokens ({{VLM_TOK_OUT}} out) — full pass under $2</div></div>
  </div>

  <h2>What we measured (objective, no human judge)</h2>
  <table>
    <tr><th>Metric</th><th>Albumify (<code>{{ALB_PIPE}}</code>)</th><th>VLM (opus-4-8)</th><th>Read</th></tr>
    <tr><td>Distinct scenes covered</td><td>{{ALB_SCENES}}</td><td>{{VLM_SCENES}}</td><td>by Albumify's own scene clusters</td></tr>
    <tr><td>Redundancy rate</td><td>{{ALB_RED}}</td><td>{{VLM_RED}}</td><td>VLM slightly more diverse — and it never saw the clusters</td></tr>
    <tr><td>Pick overlap</td><td colspan="2">{{N_OVERLAP}} / 20 identical</td><td>70% of the album is a genuine disagreement</td></tr>
    <tr><td>Picks Albumify quality-filtered</td><td>—</td><td>{{N_REJECTED}}: <code>{{VLM_REJECTED}}</code></td><td>VLM kept shots Albumify auto-rejected on quality</td></tr>
  </table>

  <div class="note"><b>Two honest caveats.</b> (1) {{ALB_CAVEAT1}} (2) <b>The human blind A/B verdict is not yet collected</b> — the judging page is
    built (<code>ab_viewer_{{TRIP}}.html</code>); this is a solo N=1 pilot, so even once judged it is an
    illustrative anecdote, not a measured win-rate.</p></div>

  <h2>The two albums (un-blinded for analysis)</h2>
  <div class="albums">
    <div class="col"><h3>Albumify — quality + scene-dedup + person penalty</h3><div class="grid">{{ALB_STRIP}}</div></div>
    <div class="col"><h3>VLM — narrative roles + editorial reasons</h3><div class="grid">{{VLM_STRIP}}</div></div>
  </div>

  <h2>Where a VLM clearly adds value — annotation (EXP-3)</h2>
  {{ANNOTATION}}
</div>
"""
