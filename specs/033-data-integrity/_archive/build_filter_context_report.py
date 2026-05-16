"""Build a self-contained HTML inventory of every filter in the pipeline.

The point of this report (per user feedback on SIGHTING-059):

  "I think we need to understand what filter types we have ... these should
   be part of the context passing through the pipeline. As we propagate
   through the pipeline we can also write which step caused the image to
   be filtered. ... Issue 2: please explain where the ratio criteria comes
   from and why it's not exposed."

So the report is split into three parts:

  PART A — Inventory of every filter in the codebase
           (image-level, face-level, with thresholds, defaults, units, UI status)
  PART B — Tracing the "Min Face Size (px)" slider and the hidden min_face_ratio
           filter, showing why one is exposed and one isn't, and showing where
           the units drift
  PART C — Proposed architecture: every filter emits a StepDecision; the
           per-face audit log makes it into faces.csv. This closes Issue 1
           (no observability on crop drops) and Issue 2 (unit drift) at the
           SAME architectural change.

Run:
    .venv/Scripts/python specs/033-data-integrity/build_filter_context_report.py
Output:
    specs/033-data-integrity/filter_context.html
"""
from __future__ import annotations

import logging
from html import escape
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("filter_report")

REPO     = Path(__file__).resolve().parents[2]
OUT_HTML = Path(__file__).resolve().parent / "filter_context.html"


# -----------------------------------------------------------------------------
# Part A — the filter inventory.
#
# Each row is a real filter found by reading the codebase.  Columns:
#   step             — the module that owns the gate
#   level            — what gets filtered (image / face / cluster)
#   criterion        — what is measured
#   unit             — what units the threshold is in
#   default          — the default threshold
#   exposed_in_ui    — which UI lets you change it
#   emits_decision   — does it emit a StepDecision per item?
#   propagates_to_csv — does the outcome land in faces.csv?
# -----------------------------------------------------------------------------
FILTERS = [
    # --- Image-level (whether to consider the image at all) ---
    dict(step="filter_quality",         level="image",
         criterion="IQA score",         unit="0..1",   default="0.30",
         exposed_in_ui="Album Configure & Run (Min IQA Score)",
         emits_decision="YES (StepDecision item_type=image)",
         propagates_to_csv="n/a (image-level, not in faces.csv)",
         file="sim_bench/pipeline/steps/filter_quality.py"),
    dict(step="filter_quality",         level="image",
         criterion="sharpness score",   unit="0..1",   default="0.20",
         exposed_in_ui="Album Configure & Run (Min Sharpness)",
         emits_decision="YES",
         propagates_to_csv="n/a",
         file="sim_bench/pipeline/steps/filter_quality.py"),
    dict(step="filter_portraits",       level="image",
         criterion="min_face_ratio (largest face)",
         unit="ratio (0..1)", default="0.02",
         exposed_in_ui="NO",
         emits_decision="NO",
         propagates_to_csv="n/a",
         file="sim_bench/pipeline/steps/filter_portraits.py"),
    dict(step="filter_portraits",       level="image",
         criterion="require_frontal_pose",  unit="bool",  default="False",
         exposed_in_ui="NO",
         emits_decision="NO",
         propagates_to_csv="n/a",
         file="sim_bench/pipeline/steps/filter_portraits.py"),

    # --- Face-level: detection + size ---
    dict(step="insightface_detect_faces", level="face",
         criterion="detection_threshold", unit="0..1 conf", default="0.5",
         exposed_in_ui="Album Configure & Run (Min Detection Confidence)",
         emits_decision="PARTIAL (StepDecision emitted at IMAGE level, "
                        "not per face)",
         propagates_to_csv="implicitly (face never appears if rejected)",
         file="sim_bench/pipeline/steps/insightface_detect_faces.py"),
    dict(step="insightface_detect_faces", level="face",
         criterion="min_face_size", unit="px (bbox side)",  default="50",
         exposed_in_ui="Album Configure & Run (Min Face Size (px))",
         emits_decision="NO (slider value is stored in config_used but never "
                        "compared — see Part B)",
         propagates_to_csv="NO",
         file="sim_bench/pipeline/steps/insightface_detect_faces.py"),

    dict(step="filter_faces", level="face",
         criterion="min_confidence",     unit="0..1",  default="0.5",
         exposed_in_ui="NO (uses config-file default)",
         emits_decision="NO (sets face['filter_passed']=False on the face dict, "
                        "no StepDecision)",
         propagates_to_csv="NO",
         file="sim_bench/pipeline/steps/filter_faces.py"),
    dict(step="filter_faces", level="face",
         criterion="min_bbox_ratio (bbox_w / image_w)", unit="ratio (0..1)",
         default="0.02",
         exposed_in_ui="NO",
         emits_decision="NO",
         propagates_to_csv="NO",
         file="sim_bench/pipeline/steps/filter_faces.py"),
    dict(step="filter_faces", level="face",
         criterion="min_relative_size (this bbox / largest bbox in image)",
         unit="ratio (0..1)", default="0.30",
         exposed_in_ui="NO",
         emits_decision="NO",
         propagates_to_csv="NO",
         file="sim_bench/pipeline/steps/filter_faces.py"),
    dict(step="filter_faces", level="face",
         criterion="min_eye_ratio (inter_eye_dist / image_w)",
         unit="ratio (0..1)", default="0.01",
         exposed_in_ui="NO",
         emits_decision="NO",
         propagates_to_csv="NO",
         file="sim_bench/pipeline/steps/filter_faces.py"),

    # --- Face-level: scoring gates (mark faces as ineligible, not removed) ---
    dict(step="insightface_score_expression / eyes / pose", level="face",
         criterion="min_face_size (per-step copy)", unit="px (bbox side)",
         default="50",
         exposed_in_ui="Album Configure & Run (Min Face Size (px)) — same "
                        "slider feeds all 3",
         emits_decision="NO — returns default 0.5 score instead of rejecting",
         propagates_to_csv="NO",
         file="sim_bench/pipeline/steps/insightface_score_*.py"),

    # --- Face Clustering App: standalone quality gate ---
    dict(step="QualityGater (FC App)",  level="face",
         criterion="blur_min (Laplacian variance)", unit="float",
         default="50.0",
         exposed_in_ui="FC App Run tab (blur_min slider)",
         emits_decision="indirect — writes quality_blur_pass + "
                        "quality_blur_value columns to faces.csv",
         propagates_to_csv="YES (quality_blur_pass, quality_blur_value)",
         file="face_cluster/quality.py"),
    dict(step="QualityGater (FC App)",  level="face",
         criterion="yaw_max", unit="degrees", default="30.0",
         exposed_in_ui="FC App Run tab (yaw_max)",
         emits_decision="indirect",
         propagates_to_csv="YES (quality_pose_yaw_pass, quality_pose_yaw_value)",
         file="face_cluster/quality.py"),
    dict(step="QualityGater (FC App)",  level="face",
         criterion="pitch_max", unit="degrees", default="25.0",
         exposed_in_ui="FC App Run tab (pitch_max)",
         emits_decision="indirect",
         propagates_to_csv="YES",
         file="face_cluster/quality.py"),
    dict(step="QualityGater (FC App)",  level="face",
         criterion="roll_max", unit="degrees", default="25.0",
         exposed_in_ui="FC App Run tab (roll_max)",
         emits_decision="indirect",
         propagates_to_csv="YES",
         file="face_cluster/quality.py"),
    dict(step="QualityGater (FC App)",  level="face",
         criterion="min_face_area", unit="<b>UNIT DRIFT — see Part B</b>",
         default="0 (OFF)",
         exposed_in_ui='FC App Run tab ("min_face_area px (0=off)") — '
                        'label says px but data is fraction',
         emits_decision="indirect",
         propagates_to_csv="YES (quality_area_pass, quality_area_value)",
         file="face_cluster/quality.py"),
    dict(step="QualityGater (FC App)",  level="face",
         criterion="max_faces_per_image_core (top-K)", unit="count",
         default="3",
         exposed_in_ui="FC App Run tab (max_faces_per_image_core)",
         emits_decision='indirect — sets quality_rejection_reason="top_k_per_image"',
         propagates_to_csv="YES (quality_rejection_reason)",
         file="face_cluster/quality.py"),
    dict(step="QualityGater (FC App)",  level="face",
         criterion="det_score_min", unit="0..1", default="None (OFF)",
         exposed_in_ui="FC App Run tab (det_score_min) — but column is NaN "
                        "for every face on the reference run",
         emits_decision="WOULD emit if det_score column populated; "
                        "currently silently inactive",
         propagates_to_csv="YES in schema, NO in practice",
         file="face_cluster/quality.py"),

    # --- Cluster-level (merge) ---
    dict(step="ConservativeMerger gate A",  level="cluster pair",
         criterion="merge_exemplar_threshold OR cross-distance",
         unit="cosine distance",
         default="0.35 / 0.40",
         exposed_in_ui="FC App Recluster tab / Run tab (Merge Parameters)",
         emits_decision="YES — merge_log.json row per pair",
         propagates_to_csv="YES (merge_log.json + new v4 DB)",
         file="face_cluster/merge.py"),
    dict(step="ConservativeMerger gate B",  level="cluster pair",
         criterion="support count (kNN edges crossing)", unit="count",
         default="2 abs + 30% frac",
         exposed_in_ui="FC App Merge Parameters",
         emits_decision="YES",
         propagates_to_csv="YES",
         file="face_cluster/merge.py"),
    dict(step="ConservativeMerger gate C",  level="cluster pair",
         criterion="margin vs next-best", unit="cosine distance",
         default="0 (OFF)",
         exposed_in_ui="FC App Merge Parameters",
         emits_decision="YES",
         propagates_to_csv="YES",
         file="face_cluster/merge.py"),
    dict(step="ConservativeMerger gate D",  level="cluster pair",
         criterion="post-merge diameter <= factor × max(d_a, d_b)",
         unit="cosine distance × factor", default="1.5×",
         exposed_in_ui="FC App Merge Parameters",
         emits_decision="YES",
         propagates_to_csv="YES",
         file="face_cluster/merge.py"),
    dict(step="(PROPOSED) absolute max_diameter step",  level="cluster",
         criterion="diameter(merged_cluster) <= max_diameter",
         unit="cosine distance", default="1.2 (proposed)",
         exposed_in_ui="(to be added)",
         emits_decision="will emit per cluster (or per merge that survived gate D)",
         propagates_to_csv="(to be added)",
         file="(new — face_cluster/cluster_diameter_cap.py)"),
]


# -----------------------------------------------------------------------------
# Part B — Min Face Size trace and the hidden ratio filter
# -----------------------------------------------------------------------------

MIN_FACE_SIZE_TRACE = [
    ("Slider in UI",
     "app/streamlit/components/pipeline_runner.py:202-205",
     "<code>st.slider(\"Min Face Size (px)\", 20, 100, value=50, step=10, "
     "key=\"config_min_face_size\")</code>"),
    ("Wired to 4 step configs",
     "app/streamlit/components/pipeline_runner.py:307-312",
     "Same value goes into <code>insightface_detect_faces.min_face_size</code>, "
     "<code>insightface_score_expression.min_face_size</code>, "
     "<code>insightface_score_eyes.min_face_size</code>, "
     "<code>insightface_score_pose.min_face_size</code>."),
    ("Consumer 1 — insightface_detect_faces",
     "sim_bench/pipeline/steps/insightface_detect_faces.py:128-129",
     "Stored in <code>config_used</code> for the StepDecision and <b>nothing else</b>. "
     "The step does <i>not</i> compare detected faces to min_face_size — "
     "every face InsightFace returns is kept regardless of size."),
    ("Consumer 2 — insightface_score_expression",
     "sim_bench/pipeline/steps/insightface_score_expression.py:212-213",
     "<code>if face_size &lt; min_face_size or confidence &lt; min_confidence: "
     "return 0.5</code> — small face still kept, but gets a neutral 0.5 score "
     "instead of being measured."),
    ("Consumer 3/4 — insightface_score_eyes / pose",
     "sim_bench/pipeline/steps/insightface_score_*.py",
     "Same pattern: skip scoring on small faces, return default. Not a rejector."),
]


MIN_FACE_RATIO_TRACE = [
    ("filter_portraits.py:54-67",
     "Image-level: drops the whole image if NO face is &gt;= "
     "<code>min_face_ratio</code> (default 0.02 = 2% of image area). "
     "Not exposed in UI."),
    ("filter_best_faces.py:56, 68",
     "Face-level: removes faces with <code>face_ratio &lt; "
     "min_face_ratio</code> (default 0.02). Not exposed in UI."),
    ("detect_faces.py:183, configs/pipeline.yaml:70",
     "Image-level via <code>crop_service.crop_faces(...)</code>. Config-file "
     "default is <code>0.005</code> (= 0.5%) — much looser than the schema "
     "default 0.02. Not exposed in UI."),
]


# The three sources of `face.area` that disagree on unit ----------------------
AREA_UNIT_DRIFT = [
    ("face_cluster/embedding.py:99",
     "FC App standalone run — area in raw <b>pixels</b>",
     "<code>area = (x2 - x1) * (y2 - y1)</code> &nbsp; (bbox in absolute coords)"),
    ("sim_bench/pipeline/steps/face_cluster_bridge.py:49",
     "Albumify export to FC App — area as <b>fraction of image area</b>",
     "<code>area = float(bbox.get(\"w\", 0) * bbox.get(\"h\", 0))</code> "
     "&nbsp; (bbox.w / bbox.h are normalized 0..1)"),
    ("sim_bench/pipeline/steps/filter_quality_gate.py:111",
     "Experimental FC pipeline — area in raw <b>pixels</b>",
     "<code>area = w_px * h_px</code>"),
]


# Where step_decisions go (or don't) -----------------------------------------
CURRENT_AUDIT_FLOW = """
  Image-level filter (filter_quality) ─────┐
                                            │  emits StepDecision per image
                                            ▼
                              context.step_decisions list
                                            │  pipeline_service persists
                                            ▼
                              pipeline_results DB table

  Face-level filter (filter_faces) ─────► face["filter_passed"] = False
                                          face["filter_reason"]  = "..."
                                          (lives on the face dict, lost when
                                           Albumify exports to FC App)

  FC App quality gate ────────────────► faces.csv columns
                                          quality_*_pass / quality_*_value
                                          quality_rejection_reason
                                          (but only for the 4 standalone gates,
                                           not for the upstream Albumify ones)

  Crop stage ─────────────────────────► crop_path is either set or NaN
                                          NO reason column when NaN
                                          (this is the SIGHTING-059 #1 gap)
"""


# -----------------------------------------------------------------------------
# HTML rendering
# -----------------------------------------------------------------------------
CSS = """
body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;line-height:1.55;
     max-width:1280px;margin:24px auto;padding:0 18px;color:#222}
h1{border-bottom:3px solid #265;padding-bottom:.3em}
h2{margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.2em;color:#222}
h3{margin-top:1.2em;color:#444}
table{border-collapse:collapse;font-size:13px;margin:8px 0;width:100%}
th,td{border:1px solid #ddd;padding:6px 9px;text-align:left;vertical-align:top}
th{background:#eef;position:sticky;top:0}
tr.image-level td:first-child{background:#e8f4e8}
tr.face-level  td:first-child{background:#fff3e6}
tr.cluster-level td:first-child{background:#eee8f7}
tr.proposed td{background:#fff8c8;font-style:italic}
code{background:#f3f3f3;padding:1px 5px;border-radius:3px;font-size:90%}
pre{background:#1f1f1f;color:#e8e8e8;padding:14px;border-radius:4px;
    overflow-x:auto;font-size:12.5px;line-height:1.45}
.banner{background:#fafafa;border-left:5px solid #265;padding:10px 14px;
        margin:14px 0;border-radius:3px}
.callout{background:#fffbe7;border-left:5px solid #d80;padding:10px 14px;
        margin:14px 0;border-radius:3px}
.callout.crit{background:#fff5f5;border-left-color:#c33}
.legend span{display:inline-block;padding:3px 9px;margin-right:8px;
             border-radius:3px;font-size:12px}
.legend .img{background:#e8f4e8}
.legend .face{background:#fff3e6}
.legend .cluster{background:#eee8f7}
.legend .prop{background:#fff8c8;font-style:italic}
.diagram{font-family:Consolas,Menlo,monospace;font-size:12.5px;
        background:#0e1721;color:#cfe;padding:14px 18px;border-radius:5px;
        white-space:pre;overflow-x:auto}
.proposal{background:#f0f9ff;border-left:5px solid #08a;padding:10px 14px;
         margin:14px 0;border-radius:3px}
"""


def render_filter_table() -> str:
    rows = []
    for f in FILTERS:
        cls = ("image-level" if f["level"] == "image"
               else "face-level" if f["level"] == "face"
               else "cluster-level")
        if "(PROPOSED)" in f["step"]:
            cls += " proposed"
        rows.append(
            f'<tr class="{cls}">'
            f'<td><code>{escape(f["step"])}</code></td>'
            f'<td>{escape(f["level"])}</td>'
            f'<td>{escape(f["criterion"])}</td>'
            f'<td>{f["unit"]}</td>'  # may contain HTML
            f'<td>{escape(f["default"])}</td>'
            f'<td>{escape(f["exposed_in_ui"])}</td>'
            f'<td>{escape(f["emits_decision"])}</td>'
            f'<td>{escape(f["propagates_to_csv"])}</td>'
            f'<td><code>{escape(f["file"])}</code></td>'
            f'</tr>'
        )
    return (
        '<table><thead><tr>'
        '<th>step</th><th>level</th><th>criterion</th><th>unit</th>'
        '<th>default</th><th>exposed in UI?</th><th>emits StepDecision?</th>'
        '<th>propagates to faces.csv?</th><th>file</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )


def render_trace(rows: list[tuple[str, str, str]]) -> str:
    return "<ol>" + "".join(
        f"<li><b>{escape(name)}</b><br>"
        f"<small><code>{escape(loc)}</code></small><br>{body}</li>"
        for (name, loc, body) in rows
    ) + "</ol>"


def render_simple_list(rows: list[tuple[str, str]] | list[tuple[str, str, str]]) -> str:
    parts = ["<ul>"]
    for r in rows:
        if len(r) == 2:
            loc, body = r
            parts.append(f"<li><code>{escape(loc)}</code> — {body}</li>")
        else:
            loc, body, snippet = r
            parts.append(
                f"<li><code>{escape(loc)}</code> — {body}"
                f"<div style='margin-top:4px'>{snippet}</div></li>"
            )
    parts.append("</ul>")
    return "".join(parts)


PROPOSED_DIAGRAM = """
            ┌───────────────────────────────────────┐
            │       PipelineContext (existing)      │
            │                                       │
            │   step_decisions : list[StepDecision] │◄────────────┐
            └────────────┬──────────────────────────┘             │
                         │                                        │
   every filter step (image OR face OR cluster) MUST emit         │
   one StepDecision per item it touches, with                     │
       decision   = "passed" | "rejected" | "skipped" | "merged"  │
       reason     = human-readable                                │
       config_used= the threshold actually used                   │
       metrics    = the measured values                           │
                                                                  │
            ┌─────────────────────────────────────────┐           │
            │  export_for_analysis (PROPOSED)         │           │
            │                                         │           │
            │  per-face step_decisions ──► faces.csv  │           │
            │      crop_skip_reason (new column)      │           │
            │      filter_steps (new JSON column)     │           │
            │      area_unit (new column: px or frac) │           │
            │                                         │           │
            │  per-image step_decisions ──► images.csv│───────────┘
            └─────────────────────────────────────────┘
"""


def main() -> None:
    log.info("Rendering filter context report")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>SIGHTING-059 — Filter context & pipeline observability</title>
<style>{CSS}</style></head><body>

<h1>SIGHTING-059 follow-up — Filter context &amp; pipeline observability</h1>
<p class="banner">This is the structural answer to the three feedback items
the user raised after seeing the first diagnostic report:</p>
<ul>
  <li><b>Issue 1 follow-up</b> — "we shouldn't need to read code to answer
      where face_46's crop went."  Answer:  <i>StepDecision exists already;
      it's emitted by some filters and dropped on the floor by others.  Fix the
      gap, not the symptom.</i></li>
  <li><b>Issue 2 follow-up</b> — "the UI says Min Face Size (px) but I can't
      see where the ratio criteria comes from."  Answer:  <i>min_face_size
      is plumbed but not enforced as a rejector; the real rejector
      (min_face_ratio) lives in 3 different files with 3 different defaults
      and isn't exposed.</i></li>
  <li><b>Issue 3</b> — "add max_diameter as a separate step, not a 5th gate."
      Answer: filed (see end).</li>
</ul>

<!-- ===== Part A ===== -->
<h2>Part A — Filter inventory</h2>
<p class="legend">
  <span class="img">image-level</span>
  <span class="face">face-level</span>
  <span class="cluster">cluster-level</span>
  <span class="prop">proposed (not yet built)</span>
</p>
{render_filter_table()}

<div class="callout"><b>What the inventory shows:</b><br>
1. There are <b>{sum(1 for f in FILTERS if 'PROPOSED' not in f['step'])}
   real filters</b> active today across image-, face-, and cluster-level.<br>
2. Only <b>2 of them</b> (filter_quality and the 4 merge gates) emit
   <code>StepDecision</code> with structured config + metrics. The other
   {sum(1 for f in FILTERS if f['emits_decision'].startswith('NO'))} either
   set a flag on the face dict (lost on export) or emit nothing.<br>
3. Of {sum(1 for f in FILTERS if f['level']=='face')} face-level filters,
   only 6 columns make it into faces.csv (the FC-App quality columns).
   The Albumify filters (filter_faces, insightface_*_score, filter_portraits)
   contribute zero columns to the per-face audit, despite being the filters
   that actually decided whether a face even reaches the FC App.
</div>

<!-- ===== Part B.1 ===== -->
<h2>Part B.1 — Tracing "Min Face Size (px)"</h2>
<p>The slider value is read in the Album App, wired into 4 step configs,
and then quietly does almost nothing:</p>
{render_trace(MIN_FACE_SIZE_TRACE)}

<div class="callout crit"><b>Surprising fact:</b>
<code>min_face_size</code> is <b>not a rejector</b> anywhere. It only changes
behavior in the 3 scoring steps, where it makes them <i>return 0.5</i> instead
of measuring. Detection itself ignores it. So changing the slider from 50 to
80 doesn't actually reject more faces — it just makes more faces get neutral
expression/eyes/pose scores.
</div>

<!-- ===== Part B.2 ===== -->
<h2>Part B.2 — The hidden <code>min_face_ratio</code> filter (3 copies, 3 defaults)</h2>
<p>The real face-size rejector is <code>min_face_ratio</code>, which lives in
three independent places:</p>
{render_simple_list(MIN_FACE_RATIO_TRACE)}

<p>None of these is exposed in the UI. The configured default in
<code>configs/pipeline.yaml</code> is <code>0.005</code> for detect_faces
(0.5% of image area), but the schema defaults in two of the steps say
<code>0.02</code> (2%). Whichever default wins is determined by how the
config dict is built up at runtime, which is not user-visible.</p>

<h3>Why isn't it exposed?</h3>
<p>Because no one ever moved it into the Configure &amp; Run page. The plumbing
exists — every step has a <code>config_schema</code> with a JSON-Schema entry
for <code>min_face_ratio</code> — but the UI only renders sliders for the
handful of params it explicitly enumerates. Adding it is mechanical (one slider
in <code>pipeline_runner.py</code>), but it should probably be merged with
<code>min_face_size</code> so the user picks a single unit on a single control.</p>

<!-- ===== Part B.3 — unit drift ===== -->
<h2>Part B.3 — <code>face.area</code> unit drift across 3 producers</h2>
<p>The same field name (<code>face.area</code>) carries different units depending
on which producer populated it:</p>
{render_simple_list(AREA_UNIT_DRIFT)}

<p>The reference run (<code>face_clustering_20260510_231628</code>, mode
<code>main_app_export</code>) goes through Albumify's bridge, so its
<code>face.area</code> column is in <b>fraction-of-image</b>. The FC App's
quality gate compares it directly against the slider value (default 0, label
says "px"). The Face Analysis UI renders it as <code>int(area) px²</code>,
which is "0" for any fraction. Hence "Area 0 px²" on the reported face_26.
</p>

<!-- ===== Part C — proposed architecture ===== -->
<h2>Part C — Proposed architecture</h2>

<h3>Current audit flow (what we have today)</h3>
<div class="diagram">{escape(CURRENT_AUDIT_FLOW)}</div>

<h3>Proposed audit flow</h3>
<div class="diagram">{escape(PROPOSED_DIAGRAM)}</div>

<div class="proposal"><b>Concrete deltas</b>:
<ol>
  <li>Every filter step in the inventory above gains one
      <code>StepDecision</code> append per item it inspects, with
      <code>decision</code>, <code>reason</code>, <code>config_used</code>,
      <code>metrics</code> populated. This is mechanical for the ones that
      already half-emit (filter_faces sets <code>face["filter_passed"]</code>
      but no StepDecision — just write the decision both places).</li>
  <li>The Albumify→FC export step
      (<code>sim_bench/pipeline/steps/face_cluster_export.py</code> and the
      bridge) inherits per-face decisions into faces.csv as new columns:
      <ul>
        <li><code>filter_steps_passed</code> — comma-joined list of step names
            the face passed</li>
        <li><code>filter_steps_failed</code> — comma-joined list of step names
            the face failed (with the per-step reason)</li>
        <li><code>crop_skip_reason</code> — NaN on success, else an enum
            (<code>landmarks_missing</code> / <code>alignment_failed</code> /
            <code>top_k_culled</code> / <code>write_failed</code>) populated
            by the crop stage</li>
        <li><code>area_unit</code> — explicit unit declaration:
            <code>"px"</code> or <code>"image_ratio"</code></li>
      </ul></li>
  <li>The FC App reads faces.csv via RunStore (already done in spec-030),
      and the Face Analysis tab renders area using <code>area_unit</code>
      instead of assuming px.</li>
  <li>Static-check test:
      <code>test_every_filter_emits_step_decision</code> grepping the steps
      directory for filter classes that don't append to
      <code>context.step_decisions</code> in their <code>process()</code>.</li>
</ol>
</div>

<h3>How this closes the SIGHTING-059 issues</h3>
<table>
<tr><th>Issue</th><th>How the new architecture closes it</th></tr>
<tr><td>#1 cluster 6 has no crop</td>
  <td>The crop stage emits a StepDecision per face. If
      <code>decision="skipped"</code> with reason
      <code>"landmarks_missing"</code>, that's persisted to
      <code>crop_skip_reason</code> in faces.csv. Querying "show me cluster 6"
      surfaces the reason inline instead of an empty thumbnail strip.</td></tr>
<tr><td>#2 face_26 area = 0 px²</td>
  <td>The export step writes <code>area_unit="image_ratio"</code> alongside
      the value, and the UI renderer dispatches on it.
      No more "0 px²" for a 0.001 fraction.  The "Min Face Size (px)" slider
      becomes a <i>real</i> rejector after we either (a) make
      insightface_detect_faces enforce it, or (b) collapse it into
      <code>min_face_ratio</code> with explicit unit.</td></tr>
<tr><td>#3 max_diameter</td>
  <td>The proposed step is one more StepDecision producer — fits the same
      pattern. See feature request for spec.</td></tr>
</table>

<h2>Open question for you</h2>
<p>Is the <i>plumbing</i> direction (every filter emits StepDecision, exporter
propagates them to faces.csv) the architecture you want, or do you prefer
something different (e.g. a separate <code>filter_log.csv</code> per run,
or a row-per-(face,step) long table instead of wide columns on faces.csv)?
Both fit the same need; the choice affects how the UI consumes them and how
historical runs compare. I'd default to wide columns on faces.csv to match
the existing pattern, but the long table is more queryable for "show me every
face filter_faces rejected" questions.</p>

</body></html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    log.info("Wrote %s (%.1f KiB)", OUT_HTML, OUT_HTML.stat().st_size / 1024)


if __name__ == "__main__":
    main()
