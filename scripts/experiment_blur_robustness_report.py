"""Assemble reports/2026-07-10_blur_robustness/report.html + summary.md from the
part A/B/C result JSONs (experiment-report convention, CLAUDE.md)."""

from __future__ import annotations

import html as H
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "2026-07-10_blur_robustness")


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def card(img, caption):
    return (f"<div class='card'><img src='{img}' loading='lazy'>"
            f"<div class='cap'>{caption}</div></div>")


def main():
    A, Anm = load("partA_results.json"), load("partA_nearmisses.json")
    B, C, Cg = load("partB_results.json"), load("partC_results.json"), load("partC_gallery.json")
    a, b = A["stats"], B["stats"]

    partA_cards = "".join(card(g["img"],
        f"{H.escape(g['id'])}<br>lapvar {g['lapvar']} &middot; P={g['p_artifact']:.2f} "
        f"(oof {g['p_oof']:.2f})") for g in A["gallery"][:8])
    nm_cards = "".join(card(g["img"],
        f"{H.escape(g['id'])}<br>P={g['p_artifact']:.2f} &middot; oof {g['p_oof']:.2f}")
        for g in Anm)
    partB_cards = "".join(
        card(g["img"], f"{H.escape(g['file'])} &middot; <b>P={g['p']:.2f}</b>") +
        (card(g["gt_img"], "sharp reference (gt)") if g.get("gt_img") else "")
        for g in B["gallery"][:8])
    partC_cards = "".join(card(g["img"],
        f"{H.escape(g['file'])}<br>v1 <b class='bad'>P={g['p_v1']:.2f}</b> &rarr; "
        f"v2 <b class='good'>P={g['p_v2']:.2f}</b>") for g in Cg)

    nmrows = "".join(
        f"<tr><td>{H.escape(k)}</td><td>{v['v1']:.2f}</td><td>{v['v2']:.2f}</td></tr>"
        for k, v in C["near_misses"].items())

    doc = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Blur robustness: own albums + RealBlur (2026-07-10)</title><style>
body{{font-family:'Segoe UI',sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}}
h1{{border-bottom:3px solid #2563eb}} h2{{margin-top:36px;border-bottom:1px solid #ccc}}
table{{border-collapse:collapse;margin:12px 0}} td,th{{border:1px solid #bbb;padding:6px 12px;text-align:left}}
th{{background:#eef2ff}} .grid{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{width:280px}} .card img{{width:100%;border:1px solid #999;border-radius:4px}}
.cap{{font-size:12px;color:#444;padding:2px}} .good{{color:#15803d}} .bad{{color:#b91c1c}}
.verdict{{background:#f0fdf4;border-left:5px solid #15803d;padding:10px 16px;margin:16px 0}}
.note{{background:#fffbeb;border-left:5px solid #d97706;padding:10px 16px;margin:16px 0}}
code{{background:#f3f4f6;padding:1px 5px;border-radius:3px}}</style></head><body>
<h1>Does real blur fool the occlusion detector?</h1>
<p><b>Date</b>: 2026-07-10 &middot; <b>Detector</b>: clip_b32_gmax_v1 (spec-096 winner, gate P&ge;{a['gate']})
&middot; <b>Spec</b>: 097 research thread</p>

<h2>Goal</h2>
<p>The shipped blur-separation gate used <i>synthetic</i> blur only (score-and-delete, no gallery
— the gap that triggered the experiment-report convention). Here we test with (A) natural blur
mined from the user's own albums, (B) real camera-shake blur (RealBlur-J, Rim et al. ECCV 2020),
and (C) whether adding RealBlur negatives to training fixes what B finds — measuring both
<b>detection probability</b> and <b>explainability</b>.</p>

<h2>Data</h2>
<table><tr><th>Set</th><th>Images</th><th>Access</th></tr>
<tr><td>A: user's clean negatives (austria24/germany1/budapest)</td><td>{a['n_negatives']}</td>
<td><code>D:\\occlusion_dataset\\negatives\\</code></td></tr>
<tr><td>B: RealBlur-J sample, 3 blur imgs/scene</td><td>{b['n_images']} ({b['n_scenes']} scenes)</td>
<td><code>D:\\occlusion_dataset\\realblur\\j_sample\\</code> (full tar: <code>realblur\\RealBlur.tar.gz</code>, 12.2GB)</td></tr>
<tr><td>C: retrain adds 1 blur img/train-scene</td><td>{C['n_rb_train']} train / {C['n_rb_holdout']} holdout imgs</td>
<td>candidate artifact <code>realblur\\clip_b32_gmax_v2rb.npz</code></td></tr></table>

<h2>Part A — natural blur in the user's own albums: PASS</h2>
<table>
<tr><th></th><th>over gate (P&ge;0.8)</th><th>max P</th></tr>
<tr><td>40 blurriest photos (production artifact)</td><td class='good'><b>0/40</b></td><td>{a['blurriest_artifact_max']:.2f}</td></tr>
<tr><td>40 blurriest (honest out-of-fold)</td><td class='good'><b>0/40</b></td><td>{a['blurriest_oof_max']:.2f}</td></tr>
<tr><td>ALL {a['n_negatives']} negatives (artifact)</td><td>1*</td><td>{a['artifact_max']:.2f}</td></tr>
<tr><td>ALL {a['n_negatives']} negatives (out-of-fold)</td><td>3</td><td>{a['oof_max']:.2f}</td></tr></table>
<p>*the one &ldquo;negative&rdquo; over gate is <code>germany1__20240816_151011.heic</code> — the hidden
positive found during adjudication (manifest label stale). The detector is <b>right</b>; the three
out-of-fold hits are one Austria burst (below, judge yourself).</p>
<h3>The 8 blurriest album photos (of 40 in <code>imgs_partA/</code>)</h3>
<div class='grid'>{partA_cards}</div>
<h3>Near-misses + the hidden positive (tile heatmap, white box = hottest tile)</h3>
<div class='grid'>{nm_cards}</div>

<h2>Part B — RealBlur-J real camera shake: <span class='bad'>FAIL for v1</span></h2>
<table>
<tr><th>metric</th><th>value</th></tr>
<tr><td>images over gate</td><td class='bad'><b>{b['over_gate']}/{b['n_images']} ({100*b['over_gate']/b['n_images']:.1f}%)</b></td></tr>
<tr><td>scenes over gate</td><td class='bad'><b>{b['scenes_over_gate']}/{b['n_scenes']}</b></td></tr>
<tr><td>images over 0.5</td><td>{b['over_half']}/{b['n_images']}</td></tr>
<tr><td>median / mean P</td><td>{b['p_median']:.2f} / {b['p_mean']:.2f}</td></tr></table>
<p>Synthetic blur (0/120) was too easy. Real low-light handshake blur sits in a very different
part of CLIP space — much closer to &ldquo;defocused occluder&rdquo;. In an album pipeline these would
be wrongly penalized as occlusions (though they ARE bad photos — but for the wrong reason,
with the wrong explanation).</p>
<h3>Worst offenders under v1 (blur | sharp reference)</h3>
<div class='grid'>{partB_cards}</div>

<h2>Part C — retrain with {C['n_rb_train']} RealBlur negatives (1/scene): <span class='good'>FIXED</span></h2>
<table>
<tr><th>check</th><th>v1 (production)</th><th>v2 candidate</th></tr>
<tr><td><b>RealBlur HOLDOUT ({C['n_rb_holdout']} imgs, unseen scenes) over gate</b></td>
<td class='bad'>{C['holdout_v1']['over_gate']} (max {C['holdout_v1']['max']:.2f}, mean {C['holdout_v1']['mean']:.2f})</td>
<td class='good'><b>{C['holdout_v2']['over_gate']}</b> (max {C['holdout_v2']['max']:.2f}, mean {C['holdout_v2']['mean']:.2f})</td></tr>
<tr><td>scene PR-AUC on original 832 (same folds/seed)</td>
<td>{C['cv_scene_pr_auc_orig']['v1_recipe']:.3f}</td><td>{C['cv_scene_pr_auc_orig']['v2_recipe_rb_pinned']:.3f}</td></tr>
<tr><td>positive recall at gate (in-sample)</td><td>{C['pos_recall_gate_v1']}</td><td>{C['pos_recall_gate_v2']}</td></tr>
<tr><td>synthetic blur separation</td><td>0/60 &middot; 0/60 &middot; 0/60</td>
<td>{C['synthetic_blur_v2']['clean']}/60 &middot; {C['synthetic_blur_v2']['motion']}/60 &middot; {C['synthetic_blur_v2']['defocus']}/60</td></tr>
<tr><td>explainability: hot tile in LoG box</td><td>{C['explain_v1']}</td><td>{C['explain_v2']}</td></tr></table>
<h3>Same worst offenders, before &rarr; after</h3>
<div class='grid'>{partC_cards}</div>
<h3>Own-album near-misses, before &rarr; after</h3>
<table><tr><th>image</th><th>v1</th><th>v2</th></tr>{nmrows}</table>

<div class='verdict'><b>Verdict</b>: 140 RealBlur negatives (1 per scene — the representation-over-quantity
rule) eliminate ALL real-blur false alarms on 282 held-out images (31&rarr;0 over gate, mean P
0.60&rarr;0.04) at <b>zero cost</b> to benchmark PR-AUC (0.777 vs 0.774, identical folds) and zero cost
to positive recall (56/56). Synthetic separation stays perfect.</div>
<div class='note'><b>Caveats</b>: (1) tile explainability dips {C['explain_v1']}&rarr;{C['explain_v2']}
— both research-grade, tiles only modulate &plusmn;30% of the penalty; hot tile moved on
{C['hot_tile_changed']}/56 positives. (2) The 0.777 CV number differs from the stored 0.86 because
fold seed differs — the fair comparison is v1-vs-v2 on identical folds (0.777 vs 0.774).
(3) Austria burst barely moves (still under gate in-artifact) — RealBlur teaches &ldquo;shake blur&rdquo;,
not &ldquo;whatever confused that scene&rdquo;.</div>

<h2>Reproduce</h2>
<p><code>scripts/experiment_blur_negative_mining.py</code> (A) &middot;
<code>scripts/experiment_realblur_score.py</code> (B) &middot;
<code>scripts/experiment_realblur_retrain.py</code> (C) &middot; this report:
<code>scripts/experiment_blur_robustness_report.py</code></p>
</body></html>"""

    with open(os.path.join(OUT, "report.html"), "w", encoding="utf-8") as f:
        f.write(doc)

    md = f"""# Blur robustness: own albums + RealBlur (2026-07-10)

**Question**: does real (non-synthetic) blur fool the occlusion detector, and does
training on RealBlur negatives fix it? (spec-097 research thread; both detection P
and explainability measured.)

| Part | Data | Result |
|---|---|---|
| A. natural blur, own albums | 40 blurriest of {a['n_negatives']} negatives | **0/40 over gate** (max {a['blurriest_artifact_max']:.2f}); pop. scan re-found the hidden Germany positive at P=0.91 |
| B. RealBlur-J real shake | {b['n_images']} imgs / {b['n_scenes']} scenes | **v1 FAILS: {b['over_gate']} imgs ({b['scenes_over_gate']} scenes) over gate**, median P {b['p_median']:.2f} |
| C. retrain +{C['n_rb_train']} RB negatives (1/scene) | {C['n_rb_holdout']} holdout imgs | **31 -> 0 over gate**; PR-AUC {C['cv_scene_pr_auc_orig']['v1_recipe']:.3f}->{C['cv_scene_pr_auc_orig']['v2_recipe_rb_pinned']:.3f}; recall 56/56; synthetic 0/60x3; explain {C['explain_v1']}->{C['explain_v2']} |

**Verdict**: synthetic blur was too easy; real handshake blur is a genuine v1 failure mode
(7.4% over gate) and 140 well-chosen negatives eliminate it on held-out scenes at zero
benchmark/recall cost. Candidate artifact: `D:\\occlusion_dataset\\realblur\\clip_b32_gmax_v2rb.npz`
(NOT promoted to production — user decision pending).

Details + galleries: [report.html](report.html)
"""
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("report + summary written to", OUT)


if __name__ == "__main__":
    main()
