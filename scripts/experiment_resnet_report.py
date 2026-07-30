"""Comparison report: fine-tuned ResNet18+aug vs CLIP+LR probe (2026-07-10).
Recomputes the CLIP '+rb' OOF (seed 0, fast) for the ROC overlay, renders
FP/FN thumbnails for the ResNet, writes report.html + summary.md."""

from __future__ import annotations

import csv
import html as H
import json
import os

import cv2
import numpy as np

ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "reports", "2026-07-10_resnet_finetune")
GATE = 0.8


def corrected_labels(ids, y_orig):
    with open(os.path.join(ROOT, "corrections.csv"), newline="", encoding="utf-8") as f:
        corr = {r["id"]: r["decision"] for r in csv.DictReader(f)}
    y = y_orig.copy().astype(int)
    for i, rid in enumerate(ids):
        d = corr.get(str(rid))
        if d and d != "foreground_object":
            y[i] = 1 if d.startswith("occluded") else 0
    return y


def clip_rb_oof():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    d = np.load(os.path.join(ROOT, "clip_embeddings.npz"), allow_pickle=True)
    ids = np.array([str(x) for x in d["ids"]])
    y0 = corrected_labels(ids, d["y"].astype(int))
    g0 = np.array([str(x) for x in d["groups"]])
    X0 = np.concatenate([d["emb_global"], d["emb_tiles"].max(axis=1)], axis=1)
    rb = np.load(os.path.join(ROOT, "realblur", "realblur_embeddings.npz"),
                 allow_pickle=True)
    names = [str(x) for x in rb["names"]]
    scenes = sorted({n.split("__")[0] for n in names})
    tr_scenes = set(scenes[:int(len(scenes) * 0.6)])
    seen, sel = set(), []
    for i, n in enumerate(names):
        s = n.split("__")[0]
        if s in tr_scenes and s not in seen:
            seen.add(s)
            sel.append(i)
    Xrb = np.concatenate([rb["emb_global"][sel],
                          rb["emb_tiles"][sel].max(axis=1)], axis=1)
    grb = np.array(["rb_" + names[i].split("__")[0] for i in sel])
    oof = np.full(len(y0), np.nan)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X0, y0, g0):
        X = np.vstack([X0[tr], Xrb])
        y = np.concatenate([y0[tr], np.zeros(len(sel), int)])
        g = np.concatenate([g0[tr], grb])
        gsize = {}
        for gg in g:
            gsize[gg] = gsize.get(gg, 0) + 1
        cw = {1: len(y) / (2.0 * y.sum()), 0: len(y) / (2.0 * (y == 0).sum())}
        w = np.array([cw[int(yi)] / gsize[gg] for yi, gg in zip(y, g)])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=0.01).fit(sc.transform(X), y,
                                                            sample_weight=w)
        oof[te] = clf.predict_proba(sc.transform(X0[te]))[:, 1]
    return ids, y0, g0, oof


def thumb(src, dst, max_side=360):
    img = cv2.imread(src)
    if img is None:
        try:
            from PIL import Image, ImageOps
            from pillow_heif import register_heif_opener
            register_heif_opener()
            img = cv2.cvtColor(np.array(ImageOps.exif_transpose(
                Image.open(src)).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return False
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return True


def scene_level(oof, y, g):
    sp, sy = {}, {}
    for p, yy, gg in zip(oof, y, g):
        sp[gg] = max(sp.get(gg, 0.0), float(p))
        sy[gg] = max(sy.get(gg, 0), int(yy))
    ks = sorted(sp)
    return np.array([sy[k] for k in ks]), np.array([sp[k] for k in ks])


def main():
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    R = json.load(open(os.path.join(OUT, "resnet_results.json"), encoding="utf-8"))
    rn = np.load(os.path.join(OUT, "resnet_oof.npz"), allow_pickle=True)
    ids_r = [str(x) for x in rn["ids"]]
    y_r, g_r, oof_r = rn["y"].astype(int), rn["groups"], rn["oof"]

    ids_c, y_c, g_c, oof_c = clip_rb_oof()

    # ROC overlay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for (tag, y, o, g) in (("CLIP ViT-B/32 + LR (+RealBlur)", y_c, oof_c, g_c),
                           ("ResNet18 fine-tuned + aug", y_r, oof_r, g_r)):
        fpr, tpr, _ = roc_curve(y, o)
        axes[0].plot(fpr, tpr, label=f"{tag} (AUC {roc_auc_score(y, o):.3f})")
        sy, sp = scene_level(o, y, g)
        fpr, tpr, _ = roc_curve(sy, sp)
        axes[1].plot(fpr, tpr, label=f"{tag} (AUC {roc_auc_score(sy, sp):.3f})")
    for ax, t in zip(axes, ["Image-level ROC (OOF)", "Scene-level ROC (OOF)"]):
        ax.plot([0, 1], [0, 1], "k--", lw=0.5)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(t)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roc_overlay.png"), dpi=110)

    # CLIP metrics for the table
    sy, sp = scene_level(oof_c, y_c, g_c)
    clip_m = {"img_roc": roc_auc_score(y_c, oof_c),
              "img_pr": average_precision_score(y_c, oof_c),
              "scene_roc": roc_auc_score(sy, sp),
              "scene_pr": average_precision_score(sy, sp),
              "fp_over_gate": int(((oof_c >= GATE) & (y_c == 0)).sum()),
              "fn_under_gate": int(((oof_c < GATE) & (y_c == 1)).sum())}

    # ResNet FP/FN thumbnails
    with open(os.path.join(ROOT, "manifest.csv"), newline="", encoding="utf-8") as f:
        sub = {r["id"]: ("positives" if r["label"] == "1" else "negatives")
               for r in csv.DictReader(f)}
    os.makedirs(os.path.join(OUT, "imgs"), exist_ok=True)
    fp = sorted([(float(oof_r[i]), ids_r[i]) for i in range(len(y_r))
                 if y_r[i] == 0 and oof_r[i] >= GATE], reverse=True)[:8]
    fn = sorted([(float(oof_r[i]), ids_r[i]) for i in range(len(y_r))
                 if y_r[i] == 1 and oof_r[i] < GATE])[:8]
    cards = {"fp": [], "fn": []}
    for kind, lst in (("fp", fp), ("fn", fn)):
        for p, rid in lst:
            name = f"{kind}__{rid}".replace(".heic", ".jpg").replace(".HEIC", ".jpg")
            if thumb(os.path.join(ROOT, sub[rid], rid),
                     os.path.join(OUT, "imgs", name)):
                cards[kind].append(
                    f"<div class='card'><img src='imgs/{name}' loading='lazy'>"
                    f"<div class='cap'>{H.escape(rid)}<br>OOF P={p:.2f}</div></div>")

    row = ("<tr><td>{}</td>" + "<td>{:.4f}</td>" * 4 + "<td>{}</td><td>{}</td></tr>")
    doc = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>ResNet18 fine-tune vs CLIP probe (2026-07-10)</title><style>
body{{font-family:'Segoe UI',sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#222}}
h1{{border-bottom:3px solid #2563eb}} h2{{margin-top:32px;border-bottom:1px solid #ccc}}
table{{border-collapse:collapse;margin:12px 0}} td,th{{border:1px solid #bbb;padding:6px 12px}}
th{{background:#eef2ff}} .grid{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{width:250px}} .card img{{width:100%;border:1px solid #999;border-radius:4px}}
.cap{{font-size:12px;color:#444}} .verdict{{background:#f0fdf4;border-left:5px solid #15803d;padding:10px 16px}}
code{{background:#f3f4f6;padding:1px 5px;border-radius:3px}}</style></head><body>
<h1>CNN (ResNet18 fine-tuned + augmentation) vs CLIP+LR probe</h1>
<p><b>Date</b>: 2026-07-10 · identical protocol for both: 856 originals (80 pos) + 140 RealBlur
negatives, 5-fold grouped+stratified OOF CV seed 0, scene = max over burst group.
ResNet18: ImageNet weights, ALL layers fine-tuned, 6 epochs/fold, weighted BCE,
on-the-fly aug (flip / ±12° rot / crop 85-100% / color jitter / noise).</p>

<h2>Head-to-head (OOF, seed 0)</h2>
<table><tr><th>model</th><th>scene PR-AUC</th><th>scene ROC-AUC</th><th>image PR-AUC</th>
<th>image ROC-AUC</th><th>FP ≥ gate</th><th>FN &lt; gate (of 80)</th></tr>
{row.format('CLIP ViT-B/32 + LR (+RealBlur) — production recipe',
            clip_m['scene_pr'], clip_m['scene_roc'], clip_m['img_pr'], clip_m['img_roc'],
            clip_m['fp_over_gate'], clip_m['fn_under_gate'])}
{row.format('ResNet18 fine-tuned + aug',
            R['scene_pr'], R['scene_roc'], R['img_pr'], R['img_roc'],
            R['fp_over_gate'], R['fn_under_gate'])}</table>

<h2>ROC overlay</h2>
<img src='roc_overlay.png' style='max-width:100%;border:1px solid #ccc'>

<h2>ResNet's mistakes (context for the gap)</h2>
<h3>False positives at gate (top {len(cards['fp'])})</h3>
<div class='grid'>{''.join(cards['fp'])}</div>
<h3>Worst false negatives (top {len(cards['fn'])})</h3>
<div class='grid'>{''.join(cards['fn'])}</div>

<div class='verdict'><b>Verdict</b>: the frozen-CLIP probe beats the fine-tuned CNN by
~{100 * (clip_m['scene_pr'] - R['scene_pr']):.0f} scene PR-AUC points
({clip_m['scene_pr']:.3f} vs {R['scene_pr']:.3f}) with 5× fewer false positives at the gate.
Training loss fell to ~0.01 (the CNN memorized the training set) while OOF stayed far behind —
80 positives is simply not enough data to fine-tune a CNN, even with heavy augmentation,
whereas CLIP's 400M-image pretraining already encodes the needed invariances and the LR head
only has ~1k parameters to fit. Augmentation helped neither model (CLIP: null; ResNet: still
overfits through it). Consistent with spec-096's frozen-ResNet50 result (0.66).</div>

<h2>Reproduce</h2>
<p><code>scripts/experiment_resnet_finetune.py</code> → <code>scripts/experiment_resnet_report.py</code>.
OOF arrays: <code>resnet_oof.npz</code>. Cache: <code>D:\\occlusion_dataset\\_cache256</code>.</p>
</body></html>"""
    with open(os.path.join(OUT, "report.html"), "w", encoding="utf-8") as f:
        f.write(doc)

    md = f"""# ResNet18 fine-tune vs CLIP probe (2026-07-10)

Same protocol both sides (856 originals / 80 pos / +140 RealBlur / grouped OOF CV seed 0).

| model | scene PR-AUC | image PR-AUC | FP≥gate | FN<gate |
|---|---|---|---|---|
| CLIP+LR (+RealBlur) | **{clip_m['scene_pr']:.3f}** | {clip_m['img_pr']:.3f} | {clip_m['fp_over_gate']} | {clip_m['fn_under_gate']}/80 |
| ResNet18 fine-tuned + aug | {R['scene_pr']:.3f} | {R['img_pr']:.3f} | {R['fp_over_gate']} | {R['fn_under_gate']}/80 |

CNN loses by ~{100 * (clip_m['scene_pr'] - R['scene_pr']):.0f} points: train loss ~0.01 = memorization,
OOF lags = overfit on 80 positives despite on-the-fly augmentation. CLIP's pretraining is the
advantage a small dataset can't buy back. Details: [report.html](report.html)
"""
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("written", os.path.join(OUT, "report.html"))
    print(json.dumps({"clip": {k: round(float(v), 4) if isinstance(v, float) else v
                               for k, v in clip_m.items()}}, indent=2))


if __name__ == "__main__":
    main()
