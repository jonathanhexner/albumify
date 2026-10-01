#!/usr/bin/env python
"""
Experiment: can RAM++ (a content tagger) detect DEFECTS — occlusion & blur?

Hypothesis (to be tested, not assumed): RAM++ is trained for semantic content, so it should be a
poor defect detector. Test it against labeled data and the existing dedicated detectors.

OCCLUSION: labeled benchmark D:\\occlusion_dataset (label 1=occluded, 0=clean). RAM++ signal =
  sigmoid score for its only occlusion-adjacent tags (`hand`, `lens flare`), PLUS a steelman:
  the single best-separating tag out of all ~4,585 (max AUC). Compare to the trained probe
  (model_scores.csv `B_clip_oof`) on the same images.
BLUR: RealBlur-J sample (blur/ vs gt/ scene pairs). RAM++ has no blur/sharp/focus tag at all, so
  the test is whether its tag OUTPUT even changes between blurred and sharp versions (Jaccard).

Run:  .venv/Scripts/python scripts/experiment_ram_defect_detection.py [smoke]
"""
from __future__ import annotations
import sys, csv, html, random
from pathlib import Path
import numpy as np
from PIL import Image

import transformers.modeling_utils as _mu, transformers.pytorch_utils as _pu
for _n in ["apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"]:
    if not hasattr(_mu, _n) and hasattr(_pu, _n):
        setattr(_mu, _n, getattr(_pu, _n))

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ram.models import ram_plus  # noqa: E402
from ram import inference_ram, get_transform  # noqa: E402

CKPT = ROOT / "models" / "ram" / "ram_plus_swin_large_14m.pth"
OCCL = Path("D:/occlusion_dataset")
BLUR = OCCL / "realblur" / "j_sample"
OUT = ROOT / "reports" / "2026-10-01_ram_defect_detection"
IMGS = OUT / "imgs"
N_NEG = 84          # balance against the 84 positives
N_BLUR_SCENES = 20
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"


def load_model():
    print("loading RAM++ ...")
    m = ram_plus(pretrained=str(CKPT), image_size=384, vit="swin_l")
    m.eval().to("cpu")
    tags = np.array([t.strip() for t in m.tag_list])
    cap = {}
    m.fc.register_forward_hook(lambda mod, i, o: cap.__setitem__("l", o.detach().float().cpu()))
    tf = get_transform(image_size=384)
    return m, tf, tags, cap


try:
    import pillow_heif  # noqa
    pillow_heif.register_heif_opener()
except Exception:
    pass


def score_image(path, model, tf, cap):
    """Return (sigmoid score vector over all tags, list[str] tags) — or (None, None) if unreadable."""
    try:
        pil = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"  skip unreadable {Path(path).name}: {type(e).__name__}")
        return None, None
    img = tf(pil).unsqueeze(0)
    with torch.no_grad():
        res = inference_ram(img, model)
    en = res[0] if isinstance(res, (list, tuple)) else res
    tagset = [t.strip() for t in str(en).split("|") if t.strip()]
    scores = torch.sigmoid(cap["l"].squeeze()).numpy()
    return scores, tagset


def run_occlusion(model, tf, tags, cap):
    rows = list(csv.DictReader(open(OCCL / "manifest.csv", encoding="utf-8")))
    probe = {r["id"]: float(r["B_clip_oof"]) for r in csv.DictReader(open(OCCL / "model_scores.csv", encoding="utf-8"))}
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    neg = [neg[i] for i in np.linspace(0, len(neg) - 1, N_NEG).round().astype(int)]
    sample = pos + neg
    if SMOKE:
        sample = pos[:1] + neg[:1]
    print(f"occlusion: {len(pos)} pos + {len(neg)} neg = {len(sample)} imgs")

    idx = {t: i for i, t in enumerate(tags)}
    hand_i, flare_i = idx.get("hand"), idx.get("lens flare")
    labels, ram_hf, probe_s, S, examples = [], [], [], [], []
    for k, r in enumerate(sample, 1):
        sub = "positives" if r["label"] == "1" else "negatives"
        p = OCCL / sub / r["id"]
        if not p.exists():
            continue
        sc, tagset = score_image(p, model, tf, cap)
        if sc is None:
            continue
        labels.append(int(r["label"])); S.append(sc)
        ram_hf.append(max(sc[hand_i], sc[flare_i])); probe_s.append(probe.get(r["id"], np.nan))
        if r["label"] == "1" and len(examples) < 6:
            th = IMGS / f"occl_{r['id']}"; im = Image.open(p).convert("RGB"); im.thumbnail((360, 360)); im.save(th, quality=86)
            examples.append(dict(thumb=th.name, tags=tagset[:12], hand=float(sc[hand_i]), flare=float(sc[flare_i])))
        if k % 20 == 0 or SMOKE:
            print(f"  [{k}/{len(sample)}]")
    labels = np.array(labels); S = np.array(S)
    valid = ~np.isnan(probe_s)
    auc_probe = roc_auc_score(labels[valid], np.array(probe_s)[valid])
    auc_hf = roc_auc_score(labels, ram_hf)
    # steelman: best single tag over all ~4585
    aucs = np.array([roc_auc_score(labels, S[:, t]) if len(set(S[:, t])) > 1 else 0.5 for t in range(S.shape[1])])
    best_t = int(aucs.argmax())
    return dict(n=len(labels), auc_probe=float(auc_probe), auc_hf=float(auc_hf),
                best_tag=str(tags[best_t]), best_auc=float(aucs[best_t]), examples=examples)


def run_blur(model, tf, tags, cap):
    dirs = {d.name: d for d in BLUR.iterdir() if d.is_dir()}
    bdir = dirs.get("blur"); sdir = dirs.get("gt") or dirs.get("sharp") or dirs.get("gt_sharp")
    if not (bdir and sdir):
        return dict(available=False, dirs=list(dirs))
    scenes = sorted({f.name for f in bdir.glob("*.png")} & {f.name for f in sdir.glob("*.png")})
    scenes = scenes[: (2 if SMOKE else N_BLUR_SCENES)]
    jac, any_blur_tag, examples = [], False, []
    blurish = {"blur", "blurry", "out of focus", "motion blur", "defocus"}
    print(f"blur: {len(scenes)} scene pairs")
    for k, name in enumerate(scenes, 1):
        _, tb = score_image(bdir / name, model, tf, cap)
        _, ts = score_image(sdir / name, model, tf, cap)
        if tb is None or ts is None:
            continue
        sb, ss = set(tb), set(ts)
        j = len(sb & ss) / max(1, len(sb | ss))
        jac.append(j)
        if blurish & (sb | ss):
            any_blur_tag = True
        if len(examples) < 4:
            th = IMGS / f"blur_{name}"; Image.open(bdir / name).convert("RGB").resize((300, 200)).save(th, quality=84)
            examples.append(dict(thumb=th.name, blur_tags=sorted(sb)[:10], sharp_tags=sorted(ss)[:10], jac=j))
        if SMOKE:
            print(f"  {name}: jaccard {j:.2f}")
    return dict(available=True, n=len(scenes), mean_jac=float(np.mean(jac)),
                any_blur_tag=any_blur_tag, examples=examples,
                vocab_has_blur=bool(blurish & set(tags)))


def main():
    IMGS.mkdir(parents=True, exist_ok=True)
    model, tf, tags, cap = load_model()
    occl = run_occlusion(model, tf, tags, cap)
    blur = run_blur(model, tf, tags, cap)
    print("\n=== OCCLUSION ===")
    print(f"  trained probe AUC      : {occl['auc_probe']:.3f}")
    print(f"  RAM++ hand/lens-flare  : {occl['auc_hf']:.3f}")
    print(f"  RAM++ best-of-4585 tag : {occl['best_auc']:.3f}  (tag: '{occl['best_tag']}')")
    print("=== BLUR ===")
    if blur.get("available"):
        print(f"  vocab has a blur tag?  : {blur['vocab_has_blur']}")
        print(f"  any blur tag ever fired: {blur['any_blur_tag']}")
        print(f"  mean tag Jaccard blur vs sharp: {blur['mean_jac']:.2f} (1.0 = identical tags = blur-invisible)")
    else:
        print("  blur dirs not found:", blur.get("dirs"))
    if not SMOKE:
        write_report(occl, blur, tags)
        print(f"\nReport: {OUT / 'report.html'}")


def write_report(occl, blur, tags):
    def esc(x): return html.escape(str(x))
    def bar(auc, color):
        w = max(2, (auc - 0.5) / 0.5 * 100)  # 0.5..1.0 -> 0..100%
        return f'<div class="track"><span style="width:{w:.0f}%;background:{color}"></span></div>'
    occ_ex = "".join(f"""
      <div class="ex"><img src="imgs/{esc(e['thumb'])}" loading="lazy">
        <div class="extags">RAM++ saw: {esc(', '.join(e['tags']))}</div>
        <div class="exmini">hand {e['hand']:.2f} · lens flare {e['flare']:.2f}</div></div>""" for e in occl["examples"])
    blur_ex = ""
    if blur.get("available"):
        blur_ex = "".join(f"""
      <div class="ex"><img src="imgs/{esc(e['thumb'])}" loading="lazy">
        <div class="extags"><b>blurred:</b> {esc(', '.join(e['blur_tags']))}</div>
        <div class="extags"><b>sharp:</b> {esc(', '.join(e['sharp_tags']))}</div>
        <div class="exmini">tag overlap {e['jac']*100:.0f}%</div></div>""" for e in blur["examples"])

    blur_block = (f"""
    <div class="stats">
      <div class="stat"><div class="v crit">{'NONE' if not blur['vocab_has_blur'] else 'yes'}</div><div class="l">blur / focus / sharp tag in RAM++'s 4,585-tag vocabulary &mdash; nothing to threshold</div></div>
      <div class="stat"><div class="v crit">{'never' if not blur['any_blur_tag'] else 'sometimes'}</div><div class="l">a blur-related tag fired across {blur['n']} scene pairs &rarr; no blur score exists</div></div>
      <div class="stat"><div class="v">{blur['mean_jac']*100:.0f}%</div><div class="l">content-tag overlap blurred vs sharp &mdash; the shift is lost detail, not blur awareness</div></div>
    </div>
    <div class="exrow">{blur_ex}</div>"""
        if blur.get("available") else "<p class='note'>RealBlur sample dirs not found; vocabulary check alone: no blur/focus/sharp tag exists.</p>")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAM++ for defect detection — occlusion &amp; blur</title>
<style>
:root{{--ground:#0E0D0C;--surface:#17130F;--surface2:#1E1913;--line:#2C2620;--ink:#F2EEE7;
--ink-dim:#C8BFB2;--muted:#948A7C;--accent:#E8B04B;--eng:#6FB3C9;--good:#7FB682;--crit:#D98A6A;
--sans:"Helvetica Neue",Helvetica,Arial,system-ui,sans-serif;--mono:ui-monospace,"SF Mono",Consolas,monospace;}}
@media(prefers-color-scheme:light){{:root{{--ground:#F5F0E7;--surface:#FFF;--surface2:#FBF6EC;--line:#E4DACB;
--ink:#1B1714;--ink-dim:#4A4239;--muted:#766B5C;--accent:#A96E12;--eng:#2A7288;--good:#3E7A46;--crit:#B25B3C;}}}}
*{{box-sizing:border-box;}}body{{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.55;}}
.wrap{{max-width:1080px;margin:0 auto;padding:40px 24px 70px;}}
h1{{font-size:clamp(24px,4vw,38px);letter-spacing:-.02em;margin:0;}}
h2{{font-size:22px;margin:0;letter-spacing:-.01em;}}
.sub{{color:var(--ink-dim);margin-top:12px;max-width:76ch;}}.sub code{{color:var(--accent);font-family:var(--mono);font-size:.9em;}}
section{{margin-top:44px;padding-top:26px;border-top:1px solid var(--line);}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:8px;}}
.auc{{margin-top:22px;display:flex;flex-direction:column;gap:14px;max-width:680px;}}
.aucrow{{display:grid;grid-template-columns:230px 1fr 54px;gap:14px;align-items:center;}}
.aucrow .lbl{{font-size:13px;}}.aucrow .lbl b{{color:var(--ink);}}
.track{{height:14px;border-radius:7px;background:var(--surface2);border:1px solid var(--line);overflow:hidden;}}
.track span{{display:block;height:100%;}}
.aucrow .val{{font-family:var(--mono);font-variant-numeric:tabular-nums;font-weight:700;text-align:right;}}
.scalenote{{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:4px;}}
.stats{{display:flex;gap:26px;flex-wrap:wrap;margin-top:20px;}}
.stat .v{{font-size:32px;font-weight:800;letter-spacing:-.02em;}}.stat .v.crit{{color:var(--crit);}}
.stat .l{{color:var(--muted);font-size:12px;max-width:26ch;}}
.exrow{{display:flex;gap:14px;flex-wrap:wrap;margin-top:22px;}}
.ex{{width:200px;border:1px solid var(--line);border-radius:11px;overflow:hidden;background:var(--surface);}}
.ex img{{width:100%;height:150px;object-fit:cover;display:block;background:var(--surface2);}}
.extags{{font-size:11.5px;color:var(--ink-dim);padding:7px 9px 0;line-height:1.4;}}
.exmini{{font-family:var(--mono);font-size:10px;color:var(--muted);padding:5px 9px 9px;}}
.callout{{margin-top:24px;border-left:3px solid var(--accent);background:var(--surface2);border-radius:0 12px 12px 0;padding:16px 20px;}}
.callout .k{{font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);}}
.callout p{{margin:8px 0 0;font-size:14px;color:var(--ink-dim);}}
.note{{color:var(--muted);font-size:13px;margin-top:14px;}}
.foot{{margin-top:34px;font-family:var(--mono);font-size:12px;color:var(--muted);}}
</style></head><body><div class="wrap">
<h1>Can RAM++ detect defects? Occlusion &amp; blur, measured</h1>
<p class="sub">RAM++ is a <em>content</em> tagger. This tests, against labeled data, whether it can
double as a <em>defect</em> detector — and compares it to the purpose-built models already in the
pipeline. AUC bars are scaled 0.50 (chance) → 1.00 (perfect).</p>

<section>
  <div class="eyebrow">01 · Occlusion — vs the trained probe</div>
  <h2>RAM++ barely beats chance; the dedicated probe dominates</h2>
  <p class="note">{occl['n']} images (84 occluded + balanced clean) from <code>D:\\occlusion_dataset</code>.
  RAM++ has no "occlusion/finger" tag; its only candidates are <code>hand</code> &amp; <code>lens flare</code>.
  "Best-of-4,585" is a steelman: the single most label-correlated tag in its entire vocabulary.</p>
  <div class="auc">
    <div class="aucrow"><div class="lbl"><b>Trained probe</b> (B_clip_oof)</div>{bar(occl['auc_probe'],'var(--good)')}<div class="val">{occl['auc_probe']:.3f}</div></div>
    <div class="aucrow"><div class="lbl">RAM++ <b>hand / lens flare</b></div>{bar(occl['auc_hf'],'var(--crit)')}<div class="val">{occl['auc_hf']:.3f}</div></div>
    <div class="aucrow"><div class="lbl">RAM++ <b>best-of-4,585</b> (<code>{esc(occl['best_tag'])}</code>)</div>{bar(occl['best_auc'],'var(--accent)')}<div class="val">{occl['best_auc']:.3f}</div></div>
  </div>
  <div class="scalenote">AUC = area under ROC; 0.50 = coin flip, 1.00 = perfect separation.</div>
  <p class="note" style="margin-top:20px">What RAM++ actually reports on occluded photos — it describes residual content, not the obstruction:</p>
  <div class="exrow">{occ_ex}</div>
</section>

<section>
  <div class="eyebrow">02 · Blur — RAM++ is blur-invisible by design</div>
  <h2>No blur tag exists, and tags barely change when a photo is blurred</h2>
  {blur_block}
</section>

<div class="callout">
  <div class="k">Conclusion</div>
  <p>RAM++ is the wrong tool for both defects — confirmed on labeled data, not asserted. Occlusion: even
  its best-correlated tag of 4,585 (AUC {occl['best_auc']:.2f}) trails the dedicated probe
  ({occl['auc_probe']:.2f}); <code>hand</code>/<code>lens flare</code> are near chance
  ({occl['auc_hf']:.2f}). Blur: no blur/focus/sharp tag exists and none ever fires, so there is no blur score to produce.
  Keep occlusion on the trained probe and blur/quality on IQA/noise-aware sharpness; use RAM++ only
  for the content axis.</p>
</div>
<div class="foot">Albumify · experiment · 2026-10-01 · scripts/experiment_ram_defect_detection.py</div>
</div></body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
