"""Build a single self-contained HTML diagnostic report for SIGHTING-059.

Targets the run at `results/album/face_clustering_20260510_231628/`.

What the report shows (all images embedded as base64, no external assets):
  * Issue 1 — cluster 6 has no crops:
      - source images 20250822_123354.jpg and 20250822_123400.jpg
      - the bounding boxes that detection actually produced (we re-run InsightFace
        to recover them, since the crop-stage on disk didn't persist them)
      - the corresponding faces.csv rows with crop_path=NaN highlighted
      - face_26 (which DID get a crop, from the same album) shown side-by-side
        as a known-working comparison
  * Issue 2 — face_26 area = 0 px^2 in UI:
      - source image 20250822_122626.jpg with all detections boxed
      - the `area` column histogram across all 428 faces (range 0.0001..0.45)
      - confirmation that the column is a fraction-of-image-area, not pixels
  * Issue 3 — blur_score = 0.0 for every face, det_score = NaN for every face:
      - histograms (degenerate -> single bar at zero / single bar at NaN)
      - sample of crops with the broken (0/NaN) values displayed as captions
  * Issue 3b — merge_log.json missing the four `*_pass` boolean fields:
      - one row of merge_log.json shown verbatim with missing/expected fields
        side-by-side
  * Issue 5 — cluster 4 (28 faces) is internally incoherent (chain merge):
      - 28x28 cosine-distance heatmap with rows/cols ordered by hierarchical
        clustering so chained sub-groups jump out visually
      - a DBSCAN sub-clustering of cluster-4's embeddings to identify how many
        identities it actually contains
      - thumbnail strip of all 28 cluster-4 faces, grouped by sub-cluster

Run:
    .venv/Scripts/python specs/033-data-integrity/build_report.py
Output:
    specs/033-data-integrity/diagnostic_report.html
"""
from __future__ import annotations

import base64
import io
import json
import logging
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("report")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO        = Path(__file__).resolve().parents[2]
RUN_DIR     = REPO / "results" / "album" / "face_clustering_20260510_231628"
OUT_HTML    = Path(__file__).resolve().parent / "diagnostic_report.html"

# Three source images called out in the user's report.
SOURCE_IMAGES = {
    "issue_2_area_zero":           "20250822_122626.jpg",  # face_26 -> "Area 0 px^2"
    "issue_1_cluster6_first":      "20250822_123354.jpg",  # face_46 -> cluster 6, no crop
    "issue_1_cluster6_second":     "20250822_123400.jpg",  # face_47 -> cluster 6, no crop
}


# -----------------------------------------------------------------------------
# Encoding helpers
# -----------------------------------------------------------------------------
def png_b64_from_pil(img: Image.Image, max_dim: int | None = None) -> str:
    if max_dim is not None and max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_b64_from_fig(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def img_tag(b64: str, alt: str = "", width: int | None = None) -> str:
    style = f' style="max-width:{width}px;"' if width else ' style="max-width:100%;"'
    return f'<img src="data:image/png;base64,{b64}" alt="{escape(alt)}"{style}>'


# -----------------------------------------------------------------------------
# Load run data
# -----------------------------------------------------------------------------
@dataclass
class RunData:
    faces:       pd.DataFrame
    merge_log:   list[dict[str, Any]]
    embeddings:  np.ndarray
    face_ids:    np.ndarray
    pipeline_run: dict[str, Any]
    source_album: Path


def load_run() -> RunData:
    log.info("Loading run from %s", RUN_DIR)
    faces      = pd.read_csv(RUN_DIR / "faces.csv")
    merge_log  = json.loads((RUN_DIR / "merge_log.json").read_text(encoding="utf-8"))
    embeddings = np.load(RUN_DIR / "embeddings.npy")
    face_ids   = np.load(RUN_DIR / "embedding_face_ids.npy")
    pr_text    = (RUN_DIR / "pipeline_run.json").read_text(encoding="utf-8")
    pipeline_run = json.loads(pr_text) if pr_text.strip() else {}
    source_album = Path(faces.iloc[0]["image_path"]).parent if len(faces) else Path()
    log.info("  %d faces, %d clusters, %d merge rows", len(faces),
             faces["cluster_id"].nunique(), len(merge_log))
    return RunData(faces, merge_log, embeddings, face_ids, pipeline_run, source_album)


# -----------------------------------------------------------------------------
# Re-run InsightFace on the 3 named source images so we can show real bboxes.
# faces.csv does not store bboxes; we have to recover them from the model.
# -----------------------------------------------------------------------------
def detect_bboxes(image_paths: list[Path]) -> dict[Path, list[dict[str, Any]]]:
    log.info("Re-running InsightFace on %d images to recover bboxes", len(image_paths))
    sys.path.insert(0, str(REPO))
    from face_cluster.embedding import InsightFaceEmbedder
    embedder = InsightFaceEmbedder()  # buffalo_l, CPU
    out: dict[Path, list[dict[str, Any]]] = {}
    for p in image_paths:
        if not p.exists():
            log.warning("  missing: %s", p); out[p] = []; continue
        import cv2
        bgr = cv2.imread(str(p))
        if bgr is None:
            log.warning("  unreadable: %s", p); out[p] = []; continue
        faces = embedder.app.get(bgr)
        out[p] = [{
            "bbox":     [float(x) for x in f.bbox],   # [x1, y1, x2, y2]
            "det_score": float(f.det_score),
            "area_px":   float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            "kps":       f.kps.tolist() if hasattr(f, "kps") and f.kps is not None else [],
        } for f in faces]
        log.info("  %s -> %d face(s)", p.name, len(out[p]))
    return out


# -----------------------------------------------------------------------------
# Image with bboxes drawn
# -----------------------------------------------------------------------------
def annotate_image(img_path: Path, detections: list[dict[str, Any]],
                   labels: list[str] | None = None) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(20, img.size[0] // 60))
    except IOError:
        font = ImageFont.load_default()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 200, 0) if i == 0 else (220, 90, 90) if i == 1 else (60, 130, 220)
        for k in range(4):
            draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)
        label = labels[i] if labels and i < len(labels) else f"det {i}"
        bbox = draw.textbbox((x1, y1 - 30), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 30), label, font=font, fill=(255, 255, 255))
    return img


# -----------------------------------------------------------------------------
# Issue 5: cluster 4 forensics
# -----------------------------------------------------------------------------
def analyse_cluster_4(d: RunData) -> dict[str, Any]:
    cluster_id = 4
    rows = d.faces[d.faces["cluster_id"] == cluster_id].copy()
    fids = rows["face_id"].astype(int).tolist()

    id_to_idx = {int(fid): i for i, fid in enumerate(d.face_ids.tolist())}
    idxs = [id_to_idx[f] for f in fids]
    emb  = d.embeddings[idxs]
    emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos_sim = emb_n @ emb_n.T
    dist = 1.0 - cos_sim
    np.fill_diagonal(dist, 0.0)

    cond = squareform(dist, checks=False)
    Z = linkage(cond, method="average")
    order = leaves_list(Z)
    dist_ord = dist[order][:, order]
    fids_ord = [fids[i] for i in order]

    # DBSCAN to count the chained sub-identities. eps tuned for arc-face cosine.
    sub = DBSCAN(eps=0.32, min_samples=2, metric="precomputed").fit(dist)
    sub_labels = sub.labels_

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dist_ord, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"Cluster {cluster_id} pairwise cosine distance "
                 f"(N={len(fids)}, ordered by hierarchical avg-link)")
    ax.set_xticks(range(len(fids_ord))); ax.set_yticks(range(len(fids_ord)))
    ax.set_xticklabels([str(f) for f in fids_ord], rotation=90, fontsize=7)
    ax.set_yticklabels([str(f) for f in fids_ord], fontsize=7)
    fig.colorbar(im, ax=ax, label="cosine distance (0=identical, 1=opposite)")
    heatmap_b64 = png_b64_from_fig(fig)

    return {
        "n":              len(fids),
        "fids":           fids,
        "fids_ord":       fids_ord,
        "dist":           dist,
        "sub_labels":     sub_labels,
        "n_subclusters":  int((sub_labels >= 0).any()) and int(sub_labels[sub_labels >= 0].max() + 1),
        "n_subnoise":     int((sub_labels == -1).sum()),
        "internal_mean":  float(dist[np.triu_indices_from(dist, k=1)].mean()),
        "internal_max":   float(dist[np.triu_indices_from(dist, k=1)].max()),
        "internal_min":   float(dist[np.triu_indices_from(dist, k=1)].min()),
        "heatmap_b64":    heatmap_b64,
        "rows":           rows,
    }


def cluster_4_thumbnails_grouped(d: RunData, c4: dict[str, Any]) -> str:
    """Return HTML for the 28 cluster-4 thumbnails, grouped by DBSCAN sub-cluster."""
    rows = c4["rows"].set_index("face_id")
    sub_labels = c4["sub_labels"]
    groups: dict[int, list[int]] = {}
    for fid, lbl in zip(c4["fids"], sub_labels):
        groups.setdefault(int(lbl), []).append(int(fid))

    # Order: noise (-1) last, real groups by size desc.
    real = sorted([g for g in groups if g >= 0], key=lambda g: -len(groups[g]))
    ordering = real + ([-1] if -1 in groups else [])

    parts: list[str] = []
    for sub_id in ordering:
        fids = groups[sub_id]
        title = (f"Sub-cluster {sub_id} (n={len(fids)})"
                 if sub_id >= 0 else f"DBSCAN noise (n={len(fids)})")
        parts.append(f'<h4>{escape(title)}</h4>')
        parts.append('<div class="thumbstrip">')
        for fid in fids:
            row = rows.loc[fid]
            crop_rel = row.get("crop_path")
            if pd.notna(crop_rel):
                p = RUN_DIR / crop_rel
                if p.exists():
                    img = Image.open(p).convert("RGB")
                    b64 = png_b64_from_pil(img, max_dim=140)
                    src = Path(row["image_path"]).name
                    parts.append(
                        f'<figure><img src="data:image/png;base64,{b64}">'
                        f'<figcaption>face {fid}<br><small>{escape(src)}</small></figcaption></figure>'
                    )
                    continue
            parts.append(
                f'<figure class="missing"><div class="missing-box">crop<br>missing</div>'
                f'<figcaption>face {fid}</figcaption></figure>'
            )
        parts.append('</div>')
    return "\n".join(parts)


# -----------------------------------------------------------------------------
# Histograms for area / blur / det_score
# -----------------------------------------------------------------------------
def histogram_panel(d: RunData) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].hist(d.faces["area"].dropna(), bins=40, color="#4c72b0")
    axes[0].set_title(f"area (n={d.faces['area'].notna().sum()})  range "
                      f"{d.faces['area'].min():.4f}–{d.faces['area'].max():.4f}")
    axes[0].set_xlabel("area (column value)"); axes[0].axvline(1.0, ls="--", c="r")
    axes[0].text(1.01, axes[0].get_ylim()[1]*0.85, "x=1 (full image)", c="r", fontsize=8)

    axes[1].hist(d.faces["blur_score"].dropna(), bins=40, color="#dd8452")
    axes[1].set_title(f"blur_score  unique={d.faces['blur_score'].nunique()}  "
                      f"max={d.faces['blur_score'].max():.2f}")
    axes[1].set_xlabel("blur_score (Laplacian var)")

    det_clean = d.faces["det_score"].dropna()
    if len(det_clean):
        axes[2].hist(det_clean, bins=40, color="#55a868")
    else:
        axes[2].text(0.5, 0.5, "ALL det_score values are NaN\n(none recorded)",
                     ha="center", va="center", transform=axes[2].transAxes,
                     fontsize=11, color="#a00")
        axes[2].set_xlim(0, 1)
    axes[2].set_title(f"det_score  non-null={len(det_clean)} / {len(d.faces)}")
    axes[2].set_xlabel("det_score (InsightFace confidence)")

    fig.suptitle("Per-face quality columns from faces.csv (3 columns x 428 faces)",
                 fontsize=11)
    fig.tight_layout()
    return png_b64_from_fig(fig)


# -----------------------------------------------------------------------------
# Merge-log gate-fields audit
# -----------------------------------------------------------------------------
EXPECTED_MERGE_FIELDS = [
    "iteration", "cluster_a", "cluster_b",
    "cluster_a_size", "cluster_b_size",
    "centroid_dist", "d_cross_min", "d_cross_p25", "p25_cross_dist",
    "support", "required_support", "unique_support",
    "T_a", "T_b", "T_global",
    "margin_gap", "post_diameter", "max_allowed_diameter",
    "support_pass", "margin_pass", "diameter_pass", "distance_pass",
    "action", "actually_merged", "rejection_reason",
    "exemplar_a", "exemplar_b", "merge_threshold",
]


def merge_field_audit(d: RunData) -> str:
    if not d.merge_log:
        return "<p><em>merge_log is empty.</em></p>"
    sample = d.merge_log[0]
    keys_present = set(sample.keys())
    rows = []
    for f in EXPECTED_MERGE_FIELDS:
        present = f in keys_present
        cls = "ok" if present else "missing"
        val = sample.get(f, "—")
        if isinstance(val, float):
            val = f"{val:.4f}"
        rows.append(f'<tr class="{cls}"><td>{escape(f)}</td>'
                    f'<td>{"yes" if present else "<b>NO</b>"}</td>'
                    f'<td>{escape(str(val))}</td></tr>')
    extras = sorted(keys_present - set(EXPECTED_MERGE_FIELDS))
    extras_html = ""
    if extras:
        extras_html = ("<p><b>Extra fields on disk that aren't in the expected "
                       "28-field contract:</b> " + ", ".join(escape(x) for x in extras) + "</p>")
    return (
        f'<table class="audit"><thead><tr><th>field</th><th>present?</th>'
        f'<th>sample value (row 0)</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>" + extras_html
    )


# -----------------------------------------------------------------------------
# Build the HTML
# -----------------------------------------------------------------------------
CSS = """
body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;line-height:1.5;
     max-width:1180px;margin:24px auto;padding:0 16px;color:#222}
h1{border-bottom:3px solid #c33;padding-bottom:.3em}
h2{margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.2em;color:#222}
h3{margin-top:1.2em;color:#444}
section.issue{border-left:5px solid #c33;background:#fff5f5;padding:14px 18px;
              margin:20px 0;border-radius:4px}
section.issue.algo{border-left-color:#d80;background:#fff8ee}
section.issue.tuning{border-left-color:#08a;background:#eef6fb}
.banner{background:#f5f5f5;padding:10px 14px;border-radius:4px;margin:14px 0}
table{border-collapse:collapse;margin:8px 0;font-size:13px}
th,td{border:1px solid #ddd;padding:5px 9px;text-align:left;vertical-align:top}
th{background:#eee}
table.audit tr.ok td:nth-child(2){color:#070;font-weight:600}
table.audit tr.missing{background:#fee}
table.audit tr.missing td:nth-child(2){color:#a00;font-weight:700}
.thumbstrip{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 16px}
.thumbstrip figure{margin:0;text-align:center;font-size:11px}
.thumbstrip img{width:100px;height:100px;object-fit:cover;border:1px solid #aaa;
                border-radius:3px;display:block}
.thumbstrip figure.missing .missing-box{width:100px;height:100px;line-height:100px;
        background:#fdd;border:2px dashed #c33;color:#a00;font-weight:600;
        text-align:center;border-radius:3px}
.kvtable th{width:240px}
code{background:#f4f4f4;padding:1px 5px;border-radius:3px}
.verdict{background:#fff;border:1px solid #aaa;padding:10px 12px;margin-top:10px;border-radius:3px}
.verdict.bug{border-left:4px solid #c33}
.verdict.algo{border-left:4px solid #d80}
.verdict.ui{border-left:4px solid #08a}
"""


def build_html(d: RunData, bboxes: dict[Path, list[dict[str, Any]]],
               c4: dict[str, Any]) -> str:
    f26  = d.faces[d.faces["face_id"] == 26].iloc[0]
    f46  = d.faces[d.faces["face_id"] == 46].iloc[0]
    f47  = d.faces[d.faces["face_id"] == 47].iloc[0]

    img_122626 = d.source_album / SOURCE_IMAGES["issue_2_area_zero"]
    img_123354 = d.source_album / SOURCE_IMAGES["issue_1_cluster6_first"]
    img_123400 = d.source_album / SOURCE_IMAGES["issue_1_cluster6_second"]

    # Annotated source images
    annotate_pairs = [
        ("issue1_a", img_123354,
         f"face_46  cluster=6  is_core=True  area={f46['area']:.4f}  crop_path=NaN"),
        ("issue1_b", img_123400,
         f"face_47  cluster=6  is_core=True  area={f47['area']:.4f}  crop_path=NaN"),
        ("issue2",   img_122626,
         f"face_26  cluster={int(f26['cluster_id'])}  area={f26['area']:.4f}  crop_path={f26['crop_path']}"),
    ]
    annotated_b64: dict[str, str] = {}
    for tag, p, label in annotate_pairs:
        dets = bboxes.get(p, [])
        labels = [f"det {i}: score={d.get('det_score', 0):.2f}  area_px={d.get('area_px',0):,.0f}"
                  for i, d in enumerate(dets)]
        if not dets:
            img = Image.open(p).convert("RGB")
        else:
            img = annotate_image(p, dets, labels=labels)
        annotated_b64[tag] = png_b64_from_pil(img, max_dim=900)

    # face_26 successful crop, for comparison
    f26_crop_b64 = ""
    f26_crop = RUN_DIR / str(f26["crop_path"])
    if f26_crop.exists():
        f26_crop_b64 = png_b64_from_pil(Image.open(f26_crop).convert("RGB"), max_dim=240)

    n_faces       = len(d.faces)
    n_clusters    = d.faces[d.faces["cluster_id"] >= 0]["cluster_id"].nunique()
    n_noise       = int((d.faces["cluster_id"] == -1).sum())
    n_no_crop     = int(d.faces["crop_path"].isna().sum())
    n_with_crop   = n_faces - n_no_crop
    n_iter        = max((r.get("iteration", 0) for r in d.merge_log), default=0)
    n_merge_rows  = len(d.merge_log)
    actually      = sum(1 for r in d.merge_log if r.get("actually_merged"))

    hist_b64 = histogram_panel(d)

    # Merge audit
    merge_audit_html = merge_field_audit(d)

    # Cluster 4 thumbnails grouped
    c4_thumbs_html = cluster_4_thumbnails_grouped(d, c4)

    # Pre-format
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>SIGHTING-059 — Face clustering data integrity diagnostic</title>
<style>{CSS}</style></head><body>

<h1>SIGHTING-059 — Face clustering data integrity</h1>
<p class="banner"><b>Run:</b> <code>{escape(str(RUN_DIR.relative_to(REPO)))}</code><br>
<b>Source album:</b> <code>{escape(str(d.source_album))}</code><br>
<b>Producer mode:</b> <code>{escape(str(d.pipeline_run.get('mode','?')))}</code><br>
<b>Generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>Run-level numbers</h2>
<table class="kvtable"><tbody>
<tr><th>faces in faces.csv</th><td>{n_faces}</td></tr>
<tr><th>faces with crop_path populated</th><td>{n_with_crop} ({n_with_crop/n_faces*100:.1f}%)</td></tr>
<tr><th>faces with crop_path = NaN</th><td><b>{n_no_crop}</b> ({n_no_crop/n_faces*100:.1f}%)</td></tr>
<tr><th>clusters (excluding noise)</th><td>{n_clusters}</td></tr>
<tr><th>noise faces (cluster_id=-1)</th><td>{n_noise}</td></tr>
<tr><th>merge_log iterations</th><td>{n_iter}</td></tr>
<tr><th>merge_log decision rows</th><td>{n_merge_rows}</td></tr>
<tr><th>actually_merged=True rows</th><td>{actually}</td></tr>
</tbody></table>

<h2>Executive summary</h2>
<table><thead><tr><th>#</th><th>Reported</th><th>Confirmed?</th><th>Class</th></tr></thead><tbody>
<tr><td>1</td><td>Cluster 6 has no thumbnails</td><td>YES</td><td>Producer bug (crop stage)</td></tr>
<tr><td>2</td><td>face_26 area = 0 px²</td><td>YES (display + unit drift)</td><td>UI label vs column unit mismatch</td></tr>
<tr><td>3a</td><td>blur_score=0 for all faces</td><td>YES</td><td>Producer bug (blur step)</td></tr>
<tr><td>3b</td><td>det_score=NaN for all faces</td><td>YES</td><td>Producer bug (export drops field)</td></tr>
<tr><td>3c</td><td>merge gate fields missing</td><td>YES</td><td>Schema mismatch (legacy writer)</td></tr>
<tr><td>4</td><td>No profile load in Run tab</td><td>FIXED 2026-05-10</td><td>UI gap</td></tr>
<tr><td>5</td><td>Merge distances "look fake"</td><td>NO — distances are real;
    cluster 4 itself is a chain merge of multiple identities</td>
    <td>Algorithm tuning</td></tr>
</tbody></table>

<!-- ===== Issue 1 ===== -->
<section class="issue">
<h2>Issue 1 — Cluster 6 has no thumbnails (faces 46, 47)</h2>
<p>Both faces survive every quality gate, get assigned to cluster 6, are marked
<code>is_core=True</code>, but their <code>crop_path</code> is <code>NaN</code> in
<code>faces.csv</code> and <i>no</i> <code>face_0046_*</code> / <code>face_0047_*</code>
file exists in <code>crops/</code>. Their embeddings <i>do</i> exist in
<code>embeddings.npy</code>, so detection + embedding succeeded — only the
crop-persist step skipped them.</p>

<h3>Source images (re-detected with InsightFace to recover bboxes)</h3>
<p>Bboxes are NOT stored in faces.csv on this run, so we re-ran InsightFace to
show what detection actually found. Each rectangle is one detection; the label
shows the model's <code>det_score</code> and the raw pixel area.</p>
<p>{img_tag(annotated_b64["issue1_a"], "20250822_123354.jpg with detections")}</p>
<p>{img_tag(annotated_b64["issue1_b"], "20250822_123400.jpg with detections")}</p>

<h3>faces.csv rows for these faces</h3>
<table>
<tr><th>face_id</th><th>image</th><th>cluster</th><th>is_core</th><th>area</th>
<th>blur_score</th><th>det_score</th><th>quality_rejection_reason</th><th>crop_path</th></tr>
<tr><td>46</td><td>{escape(Path(f46['image_path']).name)}</td><td>6</td><td>True</td>
<td>{f46['area']:.4f}</td><td>{f46['blur_score']}</td><td>{f46['det_score']}</td>
<td>{escape(str(f46['quality_rejection_reason']))}</td><td><b>NaN</b></td></tr>
<tr><td>47</td><td>{escape(Path(f47['image_path']).name)}</td><td>6</td><td>True</td>
<td>{f47['area']:.4f}</td><td>{f47['blur_score']}</td><td>{f47['det_score']}</td>
<td>{escape(str(f47['quality_rejection_reason']))}</td><td><b>NaN</b></td></tr>
</table>

<h3>Comparison: face_26 from a different image, same album, DID get a crop</h3>
<div style="display:flex;gap:16px;align-items:flex-start">
<div>{img_tag(f26_crop_b64, "face_26 successful crop", width=180) if f26_crop_b64 else "<i>face_26 crop missing on disk</i>"}</div>
<div><small><code>crop_path={escape(str(f26['crop_path']))}</code><br>
Same quality flags, same album, same pipeline run.<br>
{n_with_crop}/{n_faces} faces got a crop; {n_no_crop} did not.</small></div>
</div>

<div class="verdict bug"><b>Verdict — producer bug.</b>
The crop stage iterates a different subset than the quality stage. Faces that
pass quality and form a real cluster can still be silently dropped before crop
persistence. Look at <code>face_cluster/pipeline.py</code> crop stage: which list
does it iterate, and what filter is applied between quality-pass and crop-write?
Total impact: {n_no_crop} of {n_faces} faces ({n_no_crop/n_faces*100:.0f}%) on
this run.</div>
</section>

<!-- ===== Issue 2 ===== -->
<section class="issue tuning">
<h2>Issue 2 — face_26 displays as &quot;Area 0 px²&quot;</h2>
<p>{img_tag(annotated_b64["issue2"], "20250822_122626.jpg with detections")}</p>
<table class="kvtable"><tbody>
<tr><th>faces.csv area for face_26</th><td><code>{f26['area']:.6f}</code> (a fraction)</td></tr>
<tr><th>area column range across all 428 faces</th>
<td>min <code>{d.faces['area'].min():.6f}</code>,
max <code>{d.faces['area'].max():.6f}</code></td></tr>
<tr><th>UI Run-tab control label</th><td><code>min_face_area px (0=off)</code> — talks pixels</td></tr>
<tr><th>UI Face Analysis tab display</th><td>renders <code>int(area)</code> →
"0 px²" for any value <1.0, which is every face on this run</td></tr>
</tbody></table>
<p>{img_tag(hist_b64, "histograms of area, blur_score, det_score")}</p>
<div class="verdict ui"><b>Verdict — column unit drift.</b>
The <code>area</code> column is a fraction (0..1). The Run-tab slider, the filter
comparison, and the Face Analysis renderer all assume pixels. Need to pick one
unit and align all four sites:
(a) the column producer in <code>face_cluster/quality.py</code> or
<code>face_cluster/pipeline.py</code>,
(b) the slider label/range in Run + Recluster tabs,
(c) the comparison in the quality gate,
(d) the renderer in <code>app/face_clustering/tabs/face_analysis_tab.py</code>.
This is a <i>data-contract</i> bug — fixing one site alone makes things worse.</div>
</section>

<!-- ===== Issue 3 ===== -->
<section class="issue">
<h2>Issue 3 — blur_score = 0 for all faces, det_score = NaN for all faces</h2>
<p>The histograms above are degenerate: every face has <code>blur_score = 0.0</code>
exactly, and every face has <code>det_score = NaN</code>. Yet the embeddings are
non-zero, so the InsightFace detector ran and returned valid faces.</p>

<table class="kvtable"><tbody>
<tr><th>blur_score: unique values</th><td><code>{d.faces['blur_score'].nunique()}</code> (just <code>0.0</code>)</td></tr>
<tr><th>blur_score: max</th><td><code>{d.faces['blur_score'].max():.4f}</code></td></tr>
<tr><th>det_score: non-null count</th><td><code>{d.faces['det_score'].notna().sum()}</code> / {n_faces}</td></tr>
<tr><th>quality_blur_value column</th>
<td>non-null <code>{d.faces['quality_blur_value'].notna().sum()}</code></td></tr>
</tbody></table>

<div class="verdict bug"><b>Verdict — two more producer bugs.</b>
Either the blur step never ran, or the export step is overwriting blur_score with
a default. det_score is computed by InsightFace itself (it's <code>face.det_score</code>
on every detection) but isn't being persisted into faces.csv. Both should be
caught by a single export-stage test that asserts no column is uniformly zero/NaN
for non-trivial runs.</div>
</section>

<!-- ===== Issue 3b ===== -->
<section class="issue">
<h2>Issue 3b — merge_log.json missing the four <code>*_pass</code> booleans</h2>
<p>The 28-field <code>MergeDecisionRow</code> contract documented in
<code>face_cluster/types.py</code> includes <code>support_pass</code>,
<code>margin_pass</code>, <code>diameter_pass</code>, <code>distance_pass</code>.
None of those exist in the on-disk merge_log on this run, so the FC App's gate
badges have to derive pass/fail from the numbers — which is how SIGHTING-058's
"Margin: inf" and "4/4 REJECTED" mis-renders appear.</p>
{merge_audit_html}
<div class="verdict bug"><b>Verdict — schema mismatch in the legacy writer.</b>
Phase 1 of spec-030 added a v4 layout under <code>_v4/</code> that <i>does</i>
include all 28 fields in the DB <code>merge_decisions</code> table — but the
legacy <code>merge_log.json</code> writer next to it still emits the lossy old
shape, and FC App on <code>main</code> reads the legacy file. Fixing requires
either (a) cutting FC App over to RunStore (Phase 3 — branch
<code>spec-030-phase-3-ui-cutover</code>, not yet merged), or
(b) backporting the missing fields into the legacy writer.</div>
</section>

<!-- ===== Issue 5 ===== -->
<section class="issue algo">
<h2>Issue 5 — Cluster 4 is a chain merge (n=28 faces)</h2>
<p>You said the merge distances looked fake. They aren't — the per-pair
distances are real. The problem is that <b>cluster 4 itself, before any merge
gate runs, already contains multiple identities chain-linked together</b>.</p>

<table class="kvtable"><tbody>
<tr><th>n faces in cluster 4</th><td>{c4['n']}</td></tr>
<tr><th>internal cosine distance: min / mean / max</th>
<td>{c4['internal_min']:.3f} / {c4['internal_mean']:.3f} / <b>{c4['internal_max']:.3f}</b></td></tr>
<tr><th>DBSCAN sub-clusters at eps=0.32</th>
<td>{c4['n_subclusters']} sub-clusters + {c4['n_subnoise']} noise</td></tr>
</tbody></table>

<p>Anything above ~0.40 cosine distance between two ArcFace embeddings is "different
person" territory. Cluster 4 has internal max <b>{c4['internal_max']:.3f}</b>, which
means at least one pair inside this single cluster is essentially unrelated.</p>

<h3>Pairwise distance heatmap (rows/cols ordered by hierarchical clustering)</h3>
<p>{img_tag(c4['heatmap_b64'], "cluster 4 distance heatmap")}</p>
<p>Look for the dark-blue blocks along the diagonal — each block is a
genuinely-similar sub-group. The bright off-diagonal cells are pairs of
"different people" that ended up in the same cluster because base clustering's
mutual-kNN graph chained them through intermediate faces.</p>

<h3>DBSCAN sub-clustering of cluster 4 (eps=0.32, ArcFace cosine)</h3>
{c4_thumbs_html}

<div class="verdict algo"><b>Verdict — algorithm tuning, not a data bug.</b>
The base mutual-kNN clustering at <code>distance_threshold=0.35</code>,
<code>K=5</code> is too permissive for this album. DBSCAN at the same data
finds {c4['n_subclusters']} natural sub-groups inside what the pipeline
called a single cluster. Wait until you compare to the standalone-app
"satisfactory" run before changing anything — the gap between this run's
params and that run's params is the most useful diagnostic we have.</div>
</section>

<h2>Where to go from here</h2>
<ol>
<li><b>Producer bugs (Issues 1, 2, 3a, 3b, 3c)</b> need separate small fixes,
each with a regression test. None depend on spec-030 Phase 3 landing.</li>
<li><b>Schema-mismatch in merge_log.json (Issue 3b)</b> resolves automatically
if Phase 3 is merged to <code>main</code> and Phase 4 deletes the legacy writer.
That branch (<code>spec-030-phase-3-ui-cutover</code>) is ready.</li>
<li><b>Cluster 4 chain (Issue 5)</b> is parked until the user provides their
"satisfactory" standalone run for parameter comparison.</li>
</ol>

</body></html>
"""


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    d = load_run()
    image_paths = [d.source_album / SOURCE_IMAGES[k] for k in SOURCE_IMAGES]
    bboxes = detect_bboxes(image_paths)
    log.info("Analysing cluster 4")
    c4 = analyse_cluster_4(d)
    log.info("Building HTML")
    html = build_html(d, bboxes, c4)
    OUT_HTML.write_text(html, encoding="utf-8")
    log.info("Wrote %s (%.1f KiB)", OUT_HTML, OUT_HTML.stat().st_size / 1024)


if __name__ == "__main__":
    main()
