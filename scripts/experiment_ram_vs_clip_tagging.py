#!/usr/bin/env python
"""
Experiment: RAM++ vs fixed-method CLIP for CONTENT tagging, on 30 real trip photos.

- RAM++  (recognize-anything-plus, swin_l, 384): open-vocabulary multi-label tagger (~4.5k tags).
- CLIP   (ViT-B/32 laion2b): the SAME fixed-method read from the previous experiment —
         per-category softmax over the 20 scene prompts + people gating. Closed 20-class vocab.

Point of the comparison: not quality (CLIP can't do that) but whether RAM++'s much larger open
vocabulary is worth adding a ~2.8 GB model to the pipeline's memory floor. Emits a comprehensive
HTML report (thumbnail | RAM++ tags | CLIP tags) + summary.md.
"""
from __future__ import annotations
import sys, html
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

# --- transformers-version shim: RAM's bundled BERT imports helpers from their old location ---
import transformers.modeling_utils as _mu, transformers.pytorch_utils as _pu
for _n in ["apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"]:
    if not hasattr(_mu, _n) and hasattr(_pu, _n):
        setattr(_mu, _n, getattr(_pu, _n))

import torch  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim_bench.vision_language import CLIPModel  # noqa: E402
from ram.models import ram_plus  # noqa: E402
from ram import inference_ram, get_transform  # noqa: E402

DATA = Path("D:/albumify_vs_vlm")
OUT = ROOT / "reports" / "2026-10-01_ram_vs_clip_tagging"
IMGS = OUT / "imgs"
CKPT = ROOT / "models" / "ram" / "ram_plus_swin_large_14m.pth"
N_PER_TRIP = 10
T = 100.0
PERSON_SCENE = {"a photo of a person", "a group photo", "a close-up face portrait", "a crowd of people"}


def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


def pick_samples():
    out = []
    for trip in ["budapest", "austria", "germany"]:
        files = sorted((DATA / trip / "imgs768").glob("*.jpg"))
        if not files:
            continue
        idx = np.linspace(0, len(files) - 1, N_PER_TRIP).round().astype(int)
        out += [(trip, files[i]) for i in dict.fromkeys(idx)]  # dedup, keep order
    return out


def main():
    samples = pick_samples()
    print(f"{len(samples)} images; loading models (cpu)...")

    # CLIP (fixed method)
    cfg = yaml.safe_load((ROOT / "configs" / "photo_analysis_prompts.yaml").read_text())
    scene_prompts = cfg["scene_content"]
    human_prompts = cfg["human_focused"]
    clip = CLIPModel(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu", enable_cache=True)
    scene_txt = clip.encode_texts(scene_prompts)
    human_txt = clip.encode_texts(human_prompts)

    # RAM++
    print("loading RAM++ (swin_l, 384)...")
    ram = ram_plus(pretrained=str(CKPT), image_size=384, vit="swin_l")
    ram.eval().to("cpu")
    ram_tf = get_transform(image_size=384)

    IMGS.mkdir(parents=True, exist_ok=True)
    rows = []
    ram_counts, clip_counts, vocab = [], [], set()

    for i, (trip, path) in enumerate(samples, 1):
        name = path.name
        thumb = IMGS / f"{trip}_{name}"
        im = Image.open(path).convert("RGB"); im.thumbnail((460, 460)); im.save(thumb, quality=88)

        # CLIP fixed-method content read
        iemb = clip.encode_images([str(path)])[0]
        sp = softmax((iemb @ scene_txt.T) * T)
        scene = sorted(zip(scene_prompts, sp), key=lambda t: -t[1])[:3]
        person_mass = sum(p for pp, p in zip(scene_prompts, sp) if pp in PERSON_SCENE)
        clip_tags = [pp.replace("a photo of ", "").replace("a ", "", 1) for pp, p in scene if p > 0.08]
        if person_mass > 0.30:
            hp = softmax((iemb @ human_txt.T) * T)
            top_h = max(zip(human_prompts, hp), key=lambda t: t[1])
            clip_tags.append(top_h[0])

        # RAM++ open-vocab tags
        img_t = ram_tf(Image.open(path).convert("RGB")).unsqueeze(0).to("cpu")
        with torch.no_grad():
            res = inference_ram(img_t, ram)
        en = res[0] if isinstance(res, (list, tuple)) else res
        ram_tags = [t.strip() for t in str(en).split("|") if t.strip()]

        ram_counts.append(len(ram_tags)); clip_counts.append(len(clip_tags)); vocab.update(ram_tags)
        rows.append(dict(i=i, trip=trip, name=name, thumb=thumb.name,
                         ram=ram_tags, clip=clip_tags, scene_top=scene[0][0], scene_conf=float(scene[0][1])))
        print(f"[{i:2}/{len(samples)}] {trip}/{name}")
        print(f"     RAM++ ({len(ram_tags):2}): {', '.join(ram_tags)}")
        print(f"     CLIP  ({len(clip_tags)}): {', '.join(clip_tags)}")

    stats = dict(n=len(rows), ram_avg=float(np.mean(ram_counts)), clip_avg=float(np.mean(clip_counts)),
                 ram_vocab=len(vocab), clip_vocab=20)
    write_report(rows, stats)
    print(f"\nRAM++ avg {stats['ram_avg']:.1f} tags/img (vocab seen {stats['ram_vocab']}); "
          f"CLIP avg {stats['clip_avg']:.1f} (closed 20-class).")
    print(f"Report: {OUT / 'report.html'}")


def write_report(rows, st):
    def esc(x): return html.escape(str(x))
    chips = lambda tags, cls: "".join(f'<span class="{cls}">{esc(t)}</span>' for t in tags) or '<span class="none">—</span>'
    trs = "".join(f"""
      <tr>
        <td class="idx">{r['i']}<br><span class="trip">{esc(r['trip'])}</span></td>
        <td><img src="imgs/{esc(r['thumb'])}" alt="{esc(r['name'])}" loading="lazy"><div class="fn">{esc(r['name'])}</div></td>
        <td class="ramc">{chips(r['ram'], 'chip ram')}</td>
        <td class="clipc">{chips(r['clip'], 'chip clip')}<div class="conf">scene conf {r['scene_conf']*100:.0f}%</div></td>
      </tr>""" for r in rows)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAM++ vs fixed CLIP — content tagging</title>
<style>
:root{{--ground:#0E0D0C;--surface:#17130F;--surface2:#1E1913;--line:#2C2620;--ink:#F2EEE7;
--ink-dim:#C8BFB2;--muted:#948A7C;--accent:#E8B04B;--eng:#6FB3C9;--good:#7FB682;
--sans:"Helvetica Neue",Helvetica,Arial,system-ui,sans-serif;--mono:ui-monospace,"SF Mono",Consolas,monospace;}}
@media(prefers-color-scheme:light){{:root{{--ground:#F5F0E7;--surface:#FFF;--surface2:#FBF6EC;--line:#E4DACB;
--ink:#1B1714;--ink-dim:#4A4239;--muted:#766B5C;--accent:#A96E12;--eng:#2A7288;--good:#3E7A46;}}}}
*{{box-sizing:border-box;}}body{{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.5;}}
.wrap{{max-width:1240px;margin:0 auto;padding:40px 24px 70px;}}
h1{{font-size:clamp(24px,4vw,38px);letter-spacing:-.02em;margin:0;}}
.sub{{color:var(--ink-dim);margin-top:12px;max-width:80ch;}}
.sub code{{color:var(--accent);font-family:var(--mono);font-size:.9em;}}
.stats{{display:flex;gap:26px;flex-wrap:wrap;margin:26px 0 6px;}}
.stat .v{{font-size:30px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;}}
.stat .v.ram{{color:var(--accent);}}.stat .v.clip{{color:var(--eng);}}
.stat .l{{color:var(--muted);font-size:12px;max-width:22ch;}}
.tscroll{{overflow-x:auto;margin-top:24px;border:1px solid var(--line);border-radius:14px;}}
table{{border-collapse:collapse;width:100%;min-width:900px;}}
th,td{{text-align:left;padding:14px 14px;border-bottom:1px solid var(--line);vertical-align:top;}}
thead th{{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);background:var(--surface2);position:sticky;top:0;}}
thead th.ramh{{color:var(--accent);}}thead th.cliph{{color:var(--eng);}}
td.idx{{font-family:var(--mono);color:var(--muted);font-size:13px;width:52px;}}
td.idx .trip{{font-size:10px;color:var(--eng);}}
td img{{width:190px;height:140px;object-fit:cover;border-radius:8px;background:var(--surface2);display:block;}}
.fn{{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:5px;}}
td.ramc{{width:40%;}}td.clipc{{width:26%;}}
.chip{{display:inline-block;font-size:12px;border-radius:999px;padding:3px 10px;margin:0 4px 5px 0;border:1px solid var(--line);}}
.chip.ram{{background:color-mix(in srgb,var(--accent) 12%,var(--surface));border-color:color-mix(in srgb,var(--accent) 40%,transparent);}}
.chip.clip{{background:color-mix(in srgb,var(--eng) 12%,var(--surface));border-color:color-mix(in srgb,var(--eng) 40%,transparent);}}
.none{{color:var(--muted);}}
.conf{{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:6px;}}
.foot{{margin-top:26px;font-family:var(--mono);font-size:12px;color:var(--muted);}}
</style></head><body><div class="wrap">
<h1>RAM++ vs fixed-method CLIP — content tagging on 30 trip photos</h1>
<p class="sub"><span style="color:var(--accent)">RAM++</span> = open-vocabulary tagger (swin_l/384, ~4.5k tags, ~2.8 GB).
<span style="color:var(--eng)">CLIP</span> = <code>ViT-B/32 laion2b</code> read with the fixed method (per-category softmax over the 20 scene prompts + people gating) — a closed 20-class vocabulary.
The question isn't quality (CLIP can't judge that) — it's whether RAM++'s larger open vocabulary earns a new model on the pipeline's memory floor.</p>
<div class="stats">
  <div class="stat"><div class="v ram">{st['ram_avg']:.1f}</div><div class="l">RAM++ tags / image (avg)</div></div>
  <div class="stat"><div class="v ram">{st['ram_vocab']}</div><div class="l">distinct RAM++ tags seen across {st['n']} imgs</div></div>
  <div class="stat"><div class="v clip">{st['clip_avg']:.1f}</div><div class="l">CLIP tags / image (avg)</div></div>
  <div class="stat"><div class="v clip">{st['clip_vocab']}</div><div class="l">CLIP closed vocabulary (fixed)</div></div>
</div>
<div class="tscroll"><table>
<thead><tr><th>#</th><th>Photo</th><th class="ramh">RAM++ (open vocab)</th><th class="cliph">CLIP (fixed, 20-class)</th></tr></thead>
<tbody>{trs}</tbody>
</table></div>
<div class="foot">Albumify · experiment · 2026-10-01 · scripts/experiment_ram_vs_clip_tagging.py</div>
</div></body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
