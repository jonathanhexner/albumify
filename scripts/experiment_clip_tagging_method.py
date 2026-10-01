#!/usr/bin/env python
"""
Experiment: CLIP tagging — scoring METHOD before/after (same model, same prompts).

BEFORE = exactly what sim_bench/photo_analysis/clip_tagger.py does today:
  raw cosine similarity to all 57 prompts -> take the top-5 highest numbers.

AFTER = same ViT-B/32 (laion2b), same prompt text, but read differently:
  - scene / human  : softmax WITHIN the category (prompts compete only with peers)
  - quality / comp : contrastive pairs (sharp vs blurry, ...) -> winner + margin
  - drop CLIP-hopeless abstract composition prompts (rule-of-thirds, leading lines, ...)

No prompt wording is changed. This isolates "method" from "model".
Outputs an HTML report + summary.md with the sample thumbnails inline.
"""
from __future__ import annotations
import sys, shutil, html
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim_bench.vision_language import CLIPModel  # noqa: E402

DATA = Path("D:/albumify_vs_vlm")
OUT = ROOT / "reports" / "2026-10-01_clip_tagging_method"
IMGS = OUT / "imgs"

SAMPLES = [
    DATA / "budapest/imgs768/20250822_112331.jpg",
    DATA / "budapest/imgs768/20250822_121039.jpg",
    DATA / "austria/imgs768/20230817_142056.jpg",
    DATA / "austria/imgs768/20230817_151932.jpg",
    DATA / "germany/imgs768/20240816_080028.jpg",
    DATA / "germany/imgs768/20240816_133958.jpg",
]

T = 100.0  # CLIP logit scale for softmax (standard)

# contrastive pairs / small groups for the AFTER reading
QUALITY = {
    "sharpness": ["a sharp photo", "a blurry photo"],
    "exposure": ["a well-exposed photo", "an underexposed photo", "an overexposed photo"],
    "lighting": ["a photo with good lighting", "a photo with bad lighting"],
    "noise": ["a clean image", "a noisy image"],
    "overall": ["a high-quality photo", "a low-quality photo"],
}
COMPOSITION = {
    "composed": ["a well-composed photograph", "a poorly composed photograph"],
    "balance": ["a balanced composition", "an unbalanced composition"],
    "framing": ["a photo with good framing", "a photo with bad framing"],
}
DROPPED_COMP = [
    "a centered composition", "a rule of thirds composition", "a minimalist composition",
    "a cluttered composition", "a photo with leading lines", "a photo with symmetry",
    "a photo with depth and perspective", "a flat composition",
]
PERSON_SCENE = {"a photo of a person", "a group photo", "a close-up face portrait", "a crowd of people"}


def softmax(x):
    z = np.exp((x - x.max()) * 1.0)
    return z / z.sum()


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "photo_analysis_prompts.yaml").read_text())
    groups = {k: cfg[k] for k in ["scene_content", "quality_technical", "composition_aesthetic", "human_focused"]}
    all_prompts, g_of = [], {}
    for g, ps in groups.items():
        for p in ps:
            all_prompts.append(p); g_of[p] = g
    idx = {p: i for i, p in enumerate(all_prompts)}

    print(f"Loading CLIP ViT-B-32 / laion2b_s34b_b79k (cpu) ... {len(all_prompts)} prompts")
    clip = CLIPModel(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu", enable_cache=True)
    txt = clip.encode_texts(all_prompts)                       # [57,d] normalized
    paths = [str(p) for p in SAMPLES if p.exists()]
    img = clip.encode_images(paths)                            # [n,d] normalized
    sims = img @ txt.T                                         # cosine

    IMGS.mkdir(parents=True, exist_ok=True)

    def pick(prompts, s):
        sub = np.array([s[idx[p]] for p in prompts])
        pr = softmax(sub * T)
        order = np.argsort(pr)[::-1]
        return [(prompts[i], float(pr[i]), float(sub[i])) for i in order]

    results = []
    for p, s in zip(paths, sims):
        name = Path(p).name
        thumb = IMGS / name
        im = Image.open(p).convert("RGB"); im.thumbnail((440, 440)); im.save(thumb, quality=88)

        # BEFORE: raw top-5 across ALL 57 (clip_tagger behavior)
        top = np.argsort(s)[::-1][:5]
        before = [(all_prompts[i], float(s[i]), g_of[all_prompts[i]]) for i in top]

        # AFTER
        scene = pick(groups["scene_content"], s)[:3]
        person_mass = sum(pr for pr, in [(pr,) for (pp, pr, _) in pick(groups["scene_content"], s) if pp in PERSON_SCENE])
        human = pick(groups["human_focused"], s)[:2] if person_mass > 0.30 else None
        quality = {k: pick(v, s)[0] for k, v in QUALITY.items()}
        comp = {k: pick(v, s)[0] for k, v in COMPOSITION.items()}
        results.append(dict(name=name, thumb=thumb.name, before=before, scene=scene,
                            human=human, quality=quality, comp=comp, person_mass=person_mass))

        # stdout
        print("\n" + "=" * 70 + f"\n{name}")
        print("  BEFORE (raw top-5):")
        for pr, sc, g in before:
            print(f"    {sc:+.3f}  [{g:<20}] {pr}")
        print("  AFTER:")
        print("    scene : " + ", ".join(f"{pp} {pr*100:.0f}%" for pp, pr, _ in scene))
        if human:
            print("    people: " + ", ".join(f"{pp} {pr*100:.0f}%" for pp, pr, _ in human))
        else:
            print(f"    people: (none — person-scene mass {person_mass*100:.0f}%)")
        for k, (pp, pr, raw) in quality.items():
            print(f"    {k:<9}: {pp}  ({pr*100:.0f}%)")
        for k, (pp, pr, raw) in comp.items():
            print(f"    comp.{k:<6}: {pp}  ({pr*100:.0f}%)")

    write_report(results)
    print(f"\nReport: {OUT / 'report.html'}")


def write_report(results):
    def esc(x): return html.escape(str(x))

    def pills(items, cls="pill"):
        return "".join(f'<span class="{cls}">{esc(p)} <b>{pr*100:.0f}%</b></span>' for p, pr, _ in items)

    cards = []
    for r in results:
        before_rows = "".join(
            f'<tr><td class="num">{sc:+.3f}</td><td class="g">{esc(g)}</td><td>{esc(p)}</td></tr>'
            for p, sc, g in r["before"])
        q = r["quality"]; c = r["comp"]
        qa = lambda k: f'<span class="pill"><i>{k}</i> {esc(q[k][0])} <b>{q[k][1]*100:.0f}%</b></span>'
        ca = lambda k: f'<span class="pill"><i>{k}</i> {esc(c[k][0])} <b>{c[k][1]*100:.0f}%</b></span>'
        people = pills(r["human"]) if r["human"] else f'<span class="muted">no person (scene mass {r["person_mass"]*100:.0f}%)</span>'
        cards.append(f"""
      <div class="card">
        <img src="imgs/{esc(r['thumb'])}" alt="{esc(r['name'])}" loading="lazy">
        <div class="body">
          <div class="name">{esc(r['name'])}</div>
          <div class="col before">
            <div class="lbl bad">BEFORE — raw top-5 across all 57</div>
            <table>{before_rows}</table>
          </div>
          <div class="col after">
            <div class="lbl good">AFTER — per-category softmax + contrastive pairs</div>
            <div class="grp"><span class="k">scene</span>{pills(r['scene'])}</div>
            <div class="grp"><span class="k">people</span>{people}</div>
            <div class="grp"><span class="k">quality</span>{qa('sharpness')}{qa('exposure')}{qa('lighting')}{qa('noise')}{qa('overall')}</div>
            <div class="grp"><span class="k">composition</span>{ca('composed')}{ca('balance')}{ca('framing')}</div>
          </div>
        </div>
      </div>""")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CLIP tagging — method before/after</title>
<style>
:root{{--ground:#0E0D0C;--surface:#17130F;--surface2:#1E1913;--line:#2C2620;--ink:#F2EEE7;
--ink-dim:#C8BFB2;--muted:#948A7C;--accent:#E8B04B;--good:#7FB682;--crit:#D98A6A;--eng:#6FB3C9;
--sans:"Helvetica Neue",Helvetica,Arial,system-ui,sans-serif;--mono:ui-monospace,"SF Mono",Consolas,monospace;}}
@media(prefers-color-scheme:light){{:root{{--ground:#F5F0E7;--surface:#FFF;--surface2:#FBF6EC;--line:#E4DACB;
--ink:#1B1714;--ink-dim:#4A4239;--muted:#766B5C;--accent:#A96E12;--good:#3E7A46;--crit:#B25B3C;--eng:#2A7288;}}}}
*{{box-sizing:border-box;}}body{{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.5;}}
.wrap{{max-width:1180px;margin:0 auto;padding:40px 24px 70px;}}
h1{{font-size:clamp(24px,4vw,38px);letter-spacing:-.02em;margin:0;}}
.sub{{color:var(--ink-dim);margin-top:12px;max-width:72ch;}}
.sub code{{color:var(--accent);font-family:var(--mono);font-size:.9em;}}
.cards{{display:flex;flex-direction:column;gap:20px;margin-top:30px;}}
.card{{display:grid;grid-template-columns:260px 1fr;gap:18px;border:1px solid var(--line);border-radius:14px;background:var(--surface);overflow:hidden;}}
@media(max-width:800px){{.card{{grid-template-columns:1fr;}}}}
.card img{{width:100%;height:100%;max-height:300px;object-fit:cover;background:var(--surface2);}}
.body{{padding:16px 18px;}}
.name{{font-family:var(--mono);font-size:12px;color:var(--muted);margin-bottom:12px;}}
.col{{margin-bottom:16px;}}
.lbl{{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;}}
.lbl.bad{{color:var(--crit);}}.lbl.good{{color:var(--good);}}
table{{border-collapse:collapse;font-size:13px;width:100%;max-width:520px;}}
td{{padding:3px 10px 3px 0;border-bottom:1px solid var(--line);}}
td.num{{font-family:var(--mono);color:var(--muted);text-align:right;width:58px;}}
td.g{{font-family:var(--mono);font-size:10.5px;color:var(--eng);width:150px;}}
.grp{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:8px;}}
.grp .k{{font-family:var(--mono);font-size:10.5px;color:var(--muted);text-transform:uppercase;width:84px;flex-shrink:0;}}
.pill{{font-size:12px;border:1px solid var(--line);border-radius:999px;padding:3px 10px;background:var(--surface2);}}
.pill i{{font-style:normal;color:var(--muted);font-family:var(--mono);font-size:10px;margin-right:4px;}}
.pill b{{color:var(--accent);font-variant-numeric:tabular-nums;}}
.muted{{color:var(--muted);font-size:12px;}}
.foot{{margin-top:30px;font-family:var(--mono);font-size:12px;color:var(--muted);}}
</style></head><body><div class="wrap">
<h1>CLIP tagging — same model, two ways of reading it</h1>
<p class="sub">Model <code>ViT-B/32 · laion2b_s34b_b79k</code> · 57 prompts from <code>configs/photo_analysis_prompts.yaml</code>, unchanged.
<b>BEFORE</b> = today's <code>clip_tagger.py</code>: raw cosine, top-5 across all 57.
<b>AFTER</b> = per-category softmax + contrastive pairs (no prompt text changed). Note how BEFORE fills with
always-on tags (<code>a color photo</code>, <code>a daytime scene</code>) and mixes axes, while AFTER gives a structured read.</p>
<div class="cards">{''.join(cards)}</div>
<div class="foot">Albumify · experiment · 2026-10-01 · scripts/experiment_clip_tagging_method.py</div>
</div></body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
