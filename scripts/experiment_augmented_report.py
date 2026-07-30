"""Assemble reports/2026-07-10_augmented_retrain/report.html + summary.md
(experiment-report convention, CLAUDE.md)."""

from __future__ import annotations

import html as H
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "2026-07-10_augmented_retrain")


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def card(img, caption, w=250):
    return (f"<div class='card' style='width:{w}px'><img src='{img}' loading='lazy'>"
            f"<div class='cap'>{caption}</div></div>")


def main():
    R = load("partD_results.json")
    fp, fn = load("fp_gallery.json"), load("fn_gallery.json")
    aug_imgs = sorted(os.listdir(os.path.join(OUT, "imgs_aug")))

    def auc_row(metric, label):
        cells = "".join(
            f"<td{' class=win' if R['best_variant'] == v and metric == 'scene_pr' else ''}>"
            f"{R[v][metric].replace('+-', '&plusmn;')}</td>"
            for v in ("base", "rb", "rb_aug"))
        return f"<tr><td>{label}</td>{cells}</tr>"

    fp_cards = "".join(card(g["img"], f"{H.escape(g['id'])}<br>OOF P={g['p_oof']:.2f}")
                       for g in fp)
    fn_cards = "".join(card(g["img"], f"{H.escape(g['id'])}<br>OOF P={g['p_oof']:.2f}")
                       for g in fn)
    aug_cards = "".join(card(f"imgs_aug/{f}",
                             ("positive" if "_pos_" in f else "negative") + " aug", 200)
                        for f in aug_imgs)

    doc = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Batch-3 positives + augmentation retrain (2026-07-10)</title><style>
body{{font-family:'Segoe UI',sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}}
h1{{border-bottom:3px solid #2563eb}} h2{{margin-top:36px;border-bottom:1px solid #ccc}}
table{{border-collapse:collapse;margin:12px 0}} td,th{{border:1px solid #bbb;padding:6px 12px;text-align:left}}
th{{background:#eef2ff}} td.win{{background:#f0fdf4;font-weight:bold}}
.grid{{display:flex;flex-wrap:wrap;gap:10px}} .card img{{width:100%;border:1px solid #999;border-radius:4px}}
.cap{{font-size:12px;color:#444;padding:2px}}
.verdict{{background:#f0fdf4;border-left:5px solid #15803d;padding:10px 16px;margin:16px 0}}
.note{{background:#fffbeb;border-left:5px solid #d97706;padding:10px 16px;margin:16px 0}}
code{{background:#f3f4f6;padding:1px 5px;border-radius:3px}}</style></head><body>
<h1>Batch-3 positives + augmentation retrain</h1>
<p><b>Date</b>: 2026-07-10 &middot; <b>Data</b>: {R['n_orig']} originals
(<b>{R['n_pos']} positives</b> incl. 24 new batch-3 captures), {R['n_rb_train']} RealBlur
train negatives, {R['n_aug']} augmented variants &middot; <b>Eval</b>: 5-fold grouped+stratified
OOF CV &times; 3 seeds, augs follow their source image's fold, metrics on originals only.</p>

<h2>Goal</h2>
<p>(1) Ingest the user's 24 new positives and retrain. (2) Measure whether heavy augmentation
(hflip, &plusmn;12&deg; rotation reflect-fill, random 85&ndash;97% crops, brightness/contrast/gamma,
Gaussian noise — applied to BOTH classes) improves the detector. (3) Report ROC curves and the
concrete false positives / false negatives at the production gate (P&ge;{R['gate']}).</p>

<h2>Results — three variants</h2>
<table><tr><th>metric (mean &plusmn; sd over 3 seeds)</th><th>base (856)</th>
<th>+RealBlur</th><th>+RealBlur+aug</th></tr>
{auc_row('scene_pr', 'scene PR-AUC (headline)')}
{auc_row('scene_roc', 'scene ROC-AUC')}
{auc_row('img_pr', 'image PR-AUC')}
{auc_row('img_roc', 'image ROC-AUC')}</table>
<p>Best variant: <b>{R['best_variant']}</b>. Candidate artifact
<code>D:\\occlusion_dataset\\clip_b32_gmax_v3aug.npz</code> (full fit on best variant;
NOT promoted to production). RealBlur holdout with candidate:
<b>{R['rb_holdout_candidate']['over_gate']}/{R['n_rb_holdout']} over gate</b>
(max {R['rb_holdout_candidate']['max']:.2f}). In-sample positive recall at gate:
{R['pos_recall_gate_candidate']}.</p>

<h2>ROC curves (out-of-fold, seed 0)</h2>
<img src='roc_curves.png' style='max-width:100%;border:1px solid #ccc'>

<h2>False positives (OOF P &ge; 0.5; {R['fp_over_gate']} of them cross the {R['gate']} gate)</h2>
<div class='grid'>{fp_cards}</div>

<h2>False negatives — occluded but under the gate ({R['fn_under_gate']}/{R['n_pos']}, sorted worst-first)</h2>
<p>Expected per the user: minor occlusions may be missed, and that is acceptable —
the gallery shows WHICH positives are missed so the trade-off is explicit.</p>
<div class='grid'>{fn_cards}</div>

<h2>Augmentation samples (what the model was trained on)</h2>
<div class='grid'>{aug_cards}</div>
<p>Full augmented set was embedded in-memory (embeddings:
<code>D:\\occlusion_dataset\\aug_embeddings.npz</code>); these samples are the visual record.</p>

<div class='verdict'><b>Verdict</b>: the 24 new positives are the story — scene PR-AUC
0.777 &rarr; ~0.90 (same fold procedure as yesterday's 0.86-benchmarked model).
Augmentation is a null result: {R['rb_aug']['scene_pr'].replace('+-','&plusmn;')} vs
{R['rb']['scene_pr'].replace('+-','&plusmn;')} without — CLIP embeddings are already largely
invariant to flips/rotations/lighting, so augmented copies add near-zero new information.
RealBlur negatives remain worth keeping (real-shake false alarms stay at zero, benchmark cost nil).</div>
<div class='note'><b>Caveats</b>: (1) FP/FN lists are OOF seed-0 at image level — counts vary
&plusmn;a few images across seeds. (2) Random crops on positives can cut corner occluders out of
frame (kept &ge;85% area to limit label noise). (3) In-sample recall ({R['pos_recall_gate_candidate']})
&gt; OOF recall ({R['n_pos'] - R['fn_under_gate']}/{R['n_pos']}) — the honest number is OOF.</div>

<h2>Reproduce</h2>
<p><code>scripts/ingest_positive_batch.py</code> &rarr; <code>scripts/experiment_augment_embed.py</code>
&rarr; <code>scripts/experiment_augmented_retrain.py</code> &rarr; this report:
<code>scripts/experiment_augmented_report.py</code></p>
</body></html>"""
    with open(os.path.join(OUT, "report.html"), "w", encoding="utf-8") as f:
        f.write(doc)

    md = f"""# Batch-3 positives + augmentation retrain (2026-07-10)

**Data**: {R['n_orig']} originals / **{R['n_pos']} positives** (24 new batch-3), {R['n_rb_train']}
RealBlur negatives, {R['n_aug']} augmented variants (both classes; hflip/rot/crop/light/noise).
**Eval**: grouped OOF CV x3 seeds, metrics on originals only.

| variant | scene PR-AUC | scene ROC-AUC |
|---|---|---|
| base (856) | {R['base']['scene_pr']} | {R['base']['scene_roc']} |
| +RealBlur | **{R['rb']['scene_pr']}** | {R['rb']['scene_roc']} |
| +RealBlur+aug | {R['rb_aug']['scene_pr']} | {R['rb_aug']['scene_roc']} |

- New positives moved the needle: 0.777 -> ~0.90 scene PR-AUC (same-fold comparison).
- **Augmentation = null result** (CLIP already invariant to these transforms).
- At gate {R['gate']}: {R['fp_over_gate']} FP scenes-worth of images, {R['fn_under_gate']}/{R['n_pos']} FN
  (user-accepted: mostly minor occlusion; see FN gallery).
- RealBlur holdout: {R['rb_holdout_candidate']['over_gate']}/{R['n_rb_holdout']} over gate with candidate.
- Candidate: `D:\\occlusion_dataset\\clip_b32_gmax_v3aug.npz` (variant={R['best_variant']}; not promoted).

Details + ROC + FP/FN galleries: [report.html](report.html)
"""
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("written", os.path.join(OUT, "report.html"))


if __name__ == "__main__":
    main()
