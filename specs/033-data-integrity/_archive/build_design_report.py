"""Build the SIGHTING-059 Issues 1 & 2 design HTML — v2 (corrected framing).

What changed from v1:
  v1 framed the problem as "every step should emit a decision log". That's
  observability. The user clarified the actual concern is that **filter
  decisions are advisory, not enforced** — face_46 ends up in faces.csv
  because nothing prevents downstream steps from iterating raw collections.
  v2 reframes around in-pipeline filter STATE: named filters write to a
  typed structure, downstream steps query active(...) instead of iterating
  raw lists. Observability falls out for free.

The user also raised a new concern: FC App and Albumify have two pipeline
frameworks; how do we keep them from drifting? v2 has a dedicated section
on that (shared primitive in face_cluster/, single output file, parity
test, static checks).

Output: specs/033-data-integrity/design.html (overwrites v1)
"""
from __future__ import annotations

import logging
from html import escape
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("design")

REPO     = Path(__file__).resolve().parents[2]
OUT_HTML = Path(__file__).resolve().parent / "design.html"


CSS = """
body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;line-height:1.55;
     max-width:1180px;margin:24px auto;padding:0 18px;color:#222}
h1{border-bottom:3px solid #265;padding-bottom:.3em}
h2{margin-top:1.8em;border-bottom:1px solid #ddd;padding-bottom:.2em}
h3{margin-top:1.2em;color:#444}
h4{margin-top:1em;color:#555}
.box{background:#fafafa;border-left:5px solid #265;padding:12px 16px;
     margin:14px 0;border-radius:3px}
.box.bad{border-left-color:#c33;background:#fff5f5}
.box.good{border-left-color:#080;background:#f4fbf4}
.box.note{border-left-color:#08a;background:#f0f9ff}
.box.warn{border-left-color:#d80;background:#fffbe7}
pre{background:#0e1721;color:#cfe;padding:14px 18px;border-radius:5px;
    overflow-x:auto;font-size:12.5px;line-height:1.45}
code{background:#f3f3f3;padding:1px 5px;border-radius:3px;font-size:90%}
pre code{background:transparent;padding:0;color:inherit;font-size:inherit}
table{border-collapse:collapse;font-size:13px;margin:8px 0;width:100%}
th,td{border:1px solid #ddd;padding:6px 9px;text-align:left;vertical-align:top}
th{background:#eee}
tr.bad td:first-child{background:#fee}
tr.good td:first-child{background:#efe}
.diagram{font-family:Consolas,Menlo,monospace;font-size:12.5px;
        background:#0e1721;color:#cfe;padding:14px 18px;border-radius:5px;
        white-space:pre;overflow-x:auto}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;
     font-weight:600;color:#fff;vertical-align:middle}
.tag.p1{background:#c33}.tag.p2{background:#d80}.tag.p3{background:#08a}
.tag.done{background:#080}
.changed{background:#fff8c8;padding:1px 4px;border-radius:2px}
"""


# -----------------------------------------------------------------------------
# Section blocks
# -----------------------------------------------------------------------------

PROBLEM_DIAGRAM = """
   Today's pipeline (simplified) — filter decisions are ADVISORY, not enforced
   ─────────────────────────────────────────────────────────────────────────

      filter_quality
         ↓ writes context.quality_passed = {paths that passed}
         ↓ (a *set* — nothing forces anyone to use it)

      detect_faces (uses context.image_paths, NOT quality_passed)
         ↓ writes context.insightface_faces = {path: {faces: [...]}}
         ↓ all images get face-detected, even ones quality_passed dropped

      filter_faces (sets face["filter_passed"] = False on dropped faces)
         ↓ but the face DICT stays in context.insightface_faces

      cluster_people (iterates ctx.insightface_faces directly)
         ↓ includes faces from images filter_quality rejected
         ↓ includes faces marked filter_passed=False

      ...result: cluster 6 contains face_46 even though nothing in the
                 pipeline ever decided it should be there. No single
                 step is at fault; the contract was advisory.
"""


CURRENT_STATE_TABLE = [
    # (filter, what it writes, where it lives, who consumes it, problem)
    ("filter_quality",
     "context.quality_passed: set[str]",
     "PipelineContext field",
     "advisory — no enforcement",
     "downstream iterates context.image_paths or context.insightface_faces directly"),
    ("filter_portraits",
     "context.portrait_passed: set[str]",
     "PipelineContext field",
     "advisory",
     "same"),
    ("filter_faces",
     'face["filter_passed"] = False',
     "mutated face dict inside context.insightface_faces",
     'face dicts STAY in the collection',
     "cluster_people iterates the collection raw and includes filter_passed=False faces"),
    ("QualityGater (FC App)",
     "faces.csv columns quality_*_pass / quality_rejection_reason",
     "written at export time only",
     "post-hoc audit",
     "no runtime effect on downstream steps"),
    ("save_crops",
     "(nothing — drops faces silently)",
     "—",
     "—",
     "the SIGHTING-059 #1 bug exactly"),
]


PROPOSED_DIAGRAM = """
   Proposed: filter decisions are STATE on the context, downstream steps
   query active() instead of iterating raw collections.
   ─────────────────────────────────────────────────────────────────────────

   shared module:  face_cluster/filter_context.py
   ────────────────────────────────────────────────
     class FilterDecision:
         filter_name: str        # canonical, one of KNOWN_FILTERS
         rejected:    bool
         reason:      str
         measured:    dict       # the value that triggered the verdict

     class ItemState:
         item_id:   str
         item_type: \"image\" | \"face\" | \"cluster\"
         parent_id: Optional[str]      # face → image; cluster → noise
         decisions: list[FilterDecision]

         @property
         def active(self) -> bool:
             return not any(d.rejected for d in self.decisions)

     class FilterContext:
         items: dict[str, ItemState]

         def record(self, item_id, *, filter_name, rejected, reason, measured):
             ...

         def active(self, item_type, after: str | None = None):
             \"\"\"Yields items still alive, walking parent inheritance:
                a face whose parent image was rejected is itself inactive.\"\"\"

         def summary(self) -> dict:
             \"\"\"End-of-run breakdown: per filter, how many rejected.\"\"\"

   ─────────────────────────────────────────────────────────────────────────
   Both pipelines hold an attribute   ctx.filters: FilterContext
                                      ───────────────────────────
   ▼ Albumify (sim_bench.pipeline)        ▼ FC App (face_cluster.pipeline)
   PipelineContext.filters: FilterContext   _RunContext.filters: FilterContext

   Each filter step calls:
      ctx.filters.record(item_id,
                         filter_name=\"image_quality\",
                         rejected=(iqa < 0.30 or sharp < 0.20),
                         reason=\"IQA 0.08 < 0.30\",
                         measured={\"iqa\": 0.08, \"sharpness\": 0.6})

   Each downstream step calls:
      for face in ctx.filters.active(\"face\"):
          ...

   At export:
      ctx.filters.summary()  →  filter_decisions.csv (long table,
                                                       same schema in both apps)
"""


KNOWN_FILTERS = [
    # (name, item_type, file, description, ui_binding)
    #
    # ui_binding grammar (one of three forms):
    #   "UI: <freeform label> (<widget_key>) [+ (<widget_key>)...]"
    #   "HIDDEN(permanent): <reason>"        — genuinely not user-tunable
    #   "HIDDEN(todo, target=<phase>): <reason>"  — must be resolved by <phase>
    #
    # The static check fails CI if:
    #   * a UI: binding references a widget key not present in the UI source
    #   * a widget key matching a filter-knob prefix is not referenced by exactly one filter
    #   * a HIDDEN entry does not start with one of the two parenthesized forms
    #   * a HIDDEN(todo, target=Px) entry exists when Px has been declared shipped
    ("image_quality",        "image",  "filter_quality.py",            "IQA / sharpness thresholds",
        "UI: Album Configure & Run sliders 'Min IQA Score' (config_min_iqa) + 'Min Sharpness' (config_min_sharpness)"),
    ("image_portrait",       "image",  "filter_portraits.py",          "min_face_ratio for largest face",
        "HIDDEN(todo, target=P3): not exposed today; add 'Min largest-face ratio' slider to Album Configure"),
    ("face_confidence",      "face",   "filter_faces.py / detect",     "InsightFace det_score",
        "UI: 'Min Detection Confidence' (config_det_conf)"),
    ("face_bbox_ratio",      "face",   "filter_faces.py",              "bbox_w / image_w >= threshold",
        "HIDDEN(todo, target=P6): merge with face_area in SIGHTING-060 cleanup; today uses min_bbox_ratio=0.02 default"),
    ("face_relative_size",   "face",   "filter_faces.py",              "this bbox / largest in image",
        "HIDDEN(todo, target=P3): decide: expose as slider or remove (talks past min_bbox_ratio)"),
    ("face_eye_ratio",       "face",   "filter_faces.py",              "inter-eye distance / image_w",
        "HIDDEN(todo, target=P3): decide: expose as slider or remove (talks past det_score)"),
    ("face_blur",            "face",   "face_cluster/quality.py",      "Laplacian variance >= blur_min",
        "UI: FC App Run tab slider 'blur_min' (run_blur_min)"),
    ("face_pose_yaw",        "face",   "face_cluster/quality.py",      "|yaw| <= yaw_max",
        "UI: FC App Run tab slider 'yaw_max' (run_yaw_max)"),
    ("face_pose_pitch",      "face",   "face_cluster/quality.py",      "|pitch| <= pitch_max",
        "UI: FC App Run tab slider 'pitch_max' (run_pitch_max)"),
    ("face_pose_roll",       "face",   "face_cluster/quality.py",      "|roll| <= roll_max",
        "UI: FC App Run tab slider 'roll_max' (run_roll_max)"),
    ("face_area",            "face",   "face_cluster/quality.py",      "area >= min_face_area (px) — SIGHTING-060 unit drift",
        "UI: FC App 'min_face_area px' (run_min_face_area); UNIT MISMATCH with column today, fixed in P6"),
    ("face_top_k_per_image", "face",   "face_cluster/quality.py",      "keep N largest faces per image",
        "UI: FC App 'max_faces_per_image_core' (run_max_faces)"),
    ("face_crop",            "face",   "face_cluster/pipeline._save_crops", "crop persisted to disk — SIGHTING-059 #1",
        "HIDDEN(permanent): binary success/skip, not a tunable threshold"),
    ("cluster_diameter_cap", "cluster","face_cluster/cluster_diameter_cap.py", "spec-031 (already shipped)",
        "UI: FC App merge controls (run_cap_enabled, run_cap_max_full, run_cap_max_exemplar)"),
]


CODE_PRIMITIVES = """from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional

ItemType = Literal[\"image\", \"face\", \"cluster\"]


@dataclass(frozen=True)
class FilterDecision:
    \"\"\"One filter's verdict for one item. Immutable once recorded.\"\"\"
    filter_name: str       # canonical name from KNOWN_FILTERS
    rejected:    bool
    reason:      str       # human-readable
    measured:    Dict      # the value(s) that produced the verdict


@dataclass
class ItemState:
    item_id:   str
    item_type: ItemType
    parent_id: Optional[str] = None       # face → image; cluster → None
    decisions: List[FilterDecision] = field(default_factory=list)

    @property
    def self_active(self) -> bool:
        \"\"\"Item is active IGNORING ancestor state.\"\"\"
        return not any(d.rejected for d in self.decisions)


class FilterContext:
    \"\"\"Shared filter state. Lives on PipelineContext.filters in Albumify
    and on _RunContext.filters in FC App. Same class, same module.\"\"\"

    def __init__(self):
        self._items: Dict[str, ItemState] = {}

    def register(self, item_id: str, item_type: ItemType,
                 parent_id: Optional[str] = None) -> None:
        \"\"\"Called by the step that produces the item (detect_faces for faces,
        discover_images for images, ...). Idempotent on the same item_id.\"\"\"
        if item_id not in self._items:
            self._items[item_id] = ItemState(item_id, item_type, parent_id)

    def record(self, item_id: str, *, filter_name: str, rejected: bool,
               reason: str, measured: Dict) -> None:
        assert filter_name in KNOWN_FILTERS, f\"unknown filter: {filter_name}\"
        self._items[item_id].decisions.append(FilterDecision(
            filter_name=filter_name, rejected=rejected,
            reason=reason, measured=dict(measured),
        ))

    def active(self, item_type: ItemType,
               after: Optional[str] = None) -> Iterable[ItemState]:
        \"\"\"Yield items still alive AT the named filter (or right now if
        after=None). A face is inactive if EITHER it failed a filter OR
        its parent image is inactive (recursively).\"\"\"
        for it in self._items.values():
            if it.item_type != item_type: continue
            if not self._active_recursive(it, after): continue
            yield it

    def _active_recursive(self, it: ItemState, after: Optional[str]) -> bool:
        decisions = it.decisions if after is None else \\
                    [d for d in it.decisions
                     if _filter_position(d.filter_name) <= _filter_position(after)]
        if any(d.rejected for d in decisions):
            return False
        if it.parent_id:
            parent = self._items.get(it.parent_id)
            if parent is not None and not self._active_recursive(parent, after):
                return False
        return True

    def summary(self) -> Dict[str, Dict[str, int]]:
        \"\"\"End-of-run report: per filter, count of rejections by item_type.\"\"\"
        out: Dict[str, Dict[str, int]] = {}
        for it in self._items.values():
            for d in it.decisions:
                if d.rejected:
                    out.setdefault(d.filter_name, {}) \\
                       .setdefault(it.item_type, 0)
                    out[d.filter_name][it.item_type] += 1
        return out
"""


CODE_USAGE_BEFORE_AFTER = """# Before (today)
# ────────────────────────────────────────────────────────────────────
class FilterQualityStep(BaseStep):
    def process(self, context, config):
        passed = set()
        for path, iqa in context.iqa_scores.items():
            if iqa >= 0.30 and context.sharpness_scores[path] >= 0.20:
                passed.add(path)
        context.quality_passed = passed             # ← advisory set
        context.active_images = passed              # ← also advisory


class DetectFacesStep(BaseStep):
    def process(self, context, config):
        for path in context.image_paths:            # ← ignores active_images!
            faces = self._detect(path)
            context.insightface_faces[path] = {\"faces\": faces}

# Result: detect_faces processes images that filter_quality rejected.
# Wasted compute AND filter intent silently violated.


# After (proposed)
# ────────────────────────────────────────────────────────────────────
class FilterQualityStep(BaseStep):
    def process(self, context, config):
        for path, iqa in context.iqa_scores.items():
            sharp = context.sharpness_scores.get(path, 1.0)
            rejected = iqa < 0.30 or sharp < 0.20
            context.filters.record(
                path,
                filter_name=\"image_quality\",
                rejected=rejected,
                reason=(f\"IQA {iqa:.2f} < 0.30\" if iqa < 0.30
                        else f\"sharpness {sharp:.2f} < 0.20\"),
                measured={\"iqa\": iqa, \"sharpness\": sharp},
            )


class DetectFacesStep(BaseStep):
    def process(self, context, config):
        for item in context.filters.active(\"image\"):  # ← enforced
            faces = self._detect(item.item_id)
            for i, face in enumerate(faces):
                face_id = f\"{item.item_id}:face_{i}\"
                context.filters.register(
                    face_id, item_type=\"face\", parent_id=item.item_id,
                )
                # ... store face data ...

# Result: detect_faces never sees a rejected image. cluster_people never
# sees a rejected face. filter intent is enforced by the contract.
"""


CROSS_PIPELINE_DRIFT_DIAGRAM = """
   How we keep FC App and Albumify from drifting
   ─────────────────────────────────────────────────────────────────────────

   ┌──────────────────────────────────────────────────────────────────────┐
   │   face_cluster/filter_context.py    (SHARED MODULE — one copy)       │
   │   - FilterDecision, ItemState, FilterContext, KNOWN_FILTERS          │
   │   - Both pipelines import from here. No duplicate definitions.       │
   └──────────────────────────────────────────────────────────────────────┘
                 │                                       │
                 │ imported by                           │ imported by
                 ▼                                       ▼
   ┌─────────────────────────────────┐    ┌─────────────────────────────────┐
   │ sim_bench/pipeline/context.py   │    │ face_cluster/pipeline.py        │
   │                                 │    │                                 │
   │  class PipelineContext:         │    │  class _RunContext:             │
   │    filters: FilterContext = …   │    │    filters: FilterContext = …   │
   └─────────────────────────────────┘    └─────────────────────────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                  ┌────────────────────────────────────────┐
                  │  RunExporter.export()                  │
                  │  (already shared — spec-030 Phase 1)   │
                  │                                        │
                  │  writes filter_decisions.csv with the  │
                  │  SAME schema regardless of producer    │
                  └────────────────────────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │  4 layers of drift protection           │
                  │  ───────────────────────────────       │
                  │  1. Shared primitive (one definition)  │
                  │  2. Shared exporter (spec-030)         │
                  │  3. Parity pytest (next §)             │
                  │  4. Static checks (next §)             │
                  └────────────────────────────────────────┘
"""


CODE_PARITY_TEST = """# tests/architecture/test_filter_context_parity.py

def test_both_apps_produce_same_filter_decision_schema(tmp_path):
    \"\"\"Run both pipelines on the same tiny fixture; assert the
    filter_decisions.csv they produce has the same columns, same filter
    names, and consistent semantics on a known item.\"\"\"

    fixture = REPO / \"tests\" / \"data\" / \"shared_filter_fixture\" / \"images\"
    out_albumify = tmp_path / \"albumify\"
    out_fcapp    = tmp_path / \"fcapp\"

    # Albumify path
    run_albumify_pipeline(source=fixture, output=out_albumify)
    # FC App path (full_run preset)
    FaceClusteringPipeline().run(
        PipelineConfig.full_run(source_dir=fixture, output_dir=out_fcapp)
    )

    df_a = pd.read_csv(out_albumify / \"filter_decisions.csv\")
    df_f = pd.read_csv(out_fcapp    / \"filter_decisions.csv\")

    # 1. Same column names.
    assert list(df_a.columns) == list(df_f.columns)

    # 2. Same set of filter names emitted (subject to which filters apply
    #    to a given run; intersection must not be empty).
    common = set(df_a.filter_name) & set(df_f.filter_name)
    assert {\"image_quality\", \"face_pose_yaw\", \"face_top_k_per_image\"} <= common

    # 3. Semantics: for a known synthetic image with iqa=0.05, both
    #    pipelines must reject it under image_quality with the same reason
    #    shape.
    known = \"bad_iqa_image.jpg\"
    row_a = df_a[(df_a.item_id.str.endswith(known)) &
                 (df_a.filter_name == \"image_quality\")].iloc[0]
    row_f = df_f[(df_f.item_id.str.endswith(known)) &
                 (df_f.filter_name == \"image_quality\")].iloc[0]
    assert row_a.rejected == row_f.rejected == True
    # measured values may differ in precision; check the keys match.
    assert set(json.loads(row_a.measured)) == set(json.loads(row_f.measured))
"""


CODE_UI_ALIGNMENT_CHECK = """# tests/architecture/test_ui_aligns_with_filters.py
\"\"\"Prevent the SIGHTING-059 Issue 2 scenario: UI shows a slider that does
nothing, while a hidden filter actually decides.

Bidirectional contract:
  (a) Every KNOWN_FILTERS entry whose ui_binding starts with \"UI: ...\"
      must reference a widget key that EXISTS in the UI source files.
  (b) Every widget key that LOOKS LIKE a filter knob (matches the prefix
      registry below) must be referenced by exactly one KNOWN_FILTERS entry.
  (c) HIDDEN filters must include a written reason; no silent hiding.
\"\"\"

import re
import pathlib
from face_cluster.filter_context import KNOWN_FILTERS

UI_SOURCES = [
    \"app/streamlit/components/pipeline_runner.py\",  # Album Configure & Run
    \"app/face_clustering/tabs/run_tab.py\",          # FC App Run tab
    \"app/face_clustering/tabs/recluster_tab.py\",    # FC App Recluster tab
    \"app/shared/merge_controls.py\",                 # shared merge sliders
]

# Phases that have shipped (cleared the bar) — bumped on each PR that closes
# a phase. HIDDEN(todo, target=Px) entries pointing at a shipped phase fail CI.
SHIPPED_PHASES: set[str] = set()  # extended by the file owner via separate PR

# Widget-key prefixes that signal \"this is a filter knob\".  Anything matching
# but not bound to a KNOWN_FILTERS entry is a phantom control.
FILTER_KEY_PREFIXES = (
    \"config_min_\", \"config_det_\", \"config_max_\",
    \"run_blur_\", \"run_yaw_\", \"run_pitch_\", \"run_roll_\",
    \"run_min_face_\", \"run_max_faces\", \"run_det_score_\",
    \"run_cap_\", \"rc_cap_\",
)
# Explicit non-filter widget keys that match the prefixes but aren't filters
# (e.g., clustering hyperparameters, not gates).
NON_FILTER_KEYS = {
    \"run_min_cluster\",   # clustering parameter, not a filter
    \"run_max_faces\",     # this IS a filter (top_k); kept in KNOWN_FILTERS
}

def _extract_widget_keys(path: pathlib.Path) -> set[str]:
    src = path.read_text(encoding=\"utf-8\")
    return set(re.findall(r'key=[\"\\'](\\w+)[\"\\']', src))

def _ui_keys_from_binding(binding: str) -> list[str]:
    # \"UI: 'Min Face Size (px)' (config_min_face_size); ...\"
    return re.findall(r'\\((\\w+)\\)', binding)

def test_ui_filter_alignment():
    all_ui_keys: set[str] = set()
    for src in UI_SOURCES:
        all_ui_keys |= _extract_widget_keys(pathlib.Path(src))

    bound_keys: set[str] = set()
    errors: list[str] = []

    for name, _level, _file, _desc, ui in KNOWN_FILTERS:
        if ui.startswith(\"UI:\"):
            keys = _ui_keys_from_binding(ui)
            if not keys:
                errors.append(f\"{name}: ui_binding says 'UI:' but no \"
                              f\"widget key extracted from: {ui!r}\")
                continue
            for k in keys:
                if k not in all_ui_keys:
                    errors.append(f\"{name}: ui_binding references widget \"
                                  f\"key {k!r} not found in any UI source\")
                bound_keys.add(k)
        elif ui.startswith(\"HIDDEN(permanent):\"):
            reason = ui[len(\"HIDDEN(permanent):\"):].strip()
            if not reason:
                errors.append(f\"{name}: HIDDEN(permanent) without reason\")
        elif ui.startswith(\"HIDDEN(todo,\"):
            # Form: HIDDEN(todo, target=Px): reason
            m = re.match(r\"HIDDEN\\(todo,\\s*target=(P\\d+)\\):\\s*(.+)\", ui)
            if not m:
                errors.append(f\"{name}: HIDDEN(todo,...) must be \"
                              f\"'HIDDEN(todo, target=Px): reason' (got {ui!r})\")
                continue
            target_phase, reason = m.group(1), m.group(2).strip()
            if not reason:
                errors.append(f\"{name}: HIDDEN(todo, target={target_phase}) \"
                              f\"without reason\")
            if target_phase in SHIPPED_PHASES:
                errors.append(f\"{name}: HIDDEN(todo, target={target_phase}) \"
                              f\"is overdue — {target_phase} has shipped. \"
                              f\"Resolve: expose, remove, or convert to \"
                              f\"HIDDEN(permanent).\")
        else:
            errors.append(f\"{name}: ui_binding must start with 'UI:', \"
                          f\"'HIDDEN(permanent):', or 'HIDDEN(todo, target=Px):' \"
                          f\"(got {ui!r})\")

    phantom_keys = {k for k in all_ui_keys
                    if k.startswith(FILTER_KEY_PREFIXES)
                    and k not in bound_keys
                    and k not in NON_FILTER_KEYS}
    for k in phantom_keys:
        errors.append(f\"phantom control: widget key {k!r} looks like a \"
                      f\"filter knob but no KNOWN_FILTERS entry binds it\")

    assert not errors, (
        \"UI/filter alignment violated. Either add a KNOWN_FILTERS entry, \"
        \"bind to an existing widget, or move to HIDDEN with reason:\\n  \"
        + \"\\n  \".join(errors)
    )
"""


CODE_STATIC_CHECK = """# tests/architecture/test_no_raw_collection_iteration.py

FORBIDDEN_PATTERNS = [
    # Iterating raw image collections instead of going through filters.active()
    r\"for\\s+\\w+\\s+in\\s+context\\.image_paths\\b\",
    r\"for\\s+\\w+\\s+in\\s+context\\.insightface_faces\\b\",
    r\"for\\s+\\w+\\s+in\\s+ctx\\.faces\\b\",
    # Direct iteration of legacy advisory sets
    r\"for\\s+\\w+\\s+in\\s+context\\.quality_passed\\b\",
]

ALLOW_LIST = {
    # Legitimate raw iteration with documented justification.
    \"sim_bench/pipeline/steps/discover_images.py\":
        \"discover IS the step that registers items; no upstream filters exist\",
    \"sim_bench/pipeline/steps/score_iqa.py\":
        \"IQA scores every image; filter_quality consumes scores via filters.record\",
    # ... explicit, greppable, requires reviewer sign-off to add ...
}

def test_no_raw_iteration_outside_allow_list():
    violations = []
    for path in REPO.rglob(\"*.py\"):
        if str(path.relative_to(REPO)) in ALLOW_LIST: continue
        src = path.read_text(encoding=\"utf-8\")
        for pat in FORBIDDEN_PATTERNS:
            for m in re.finditer(pat, src):
                line = src[:m.start()].count(\"\\n\") + 1
                violations.append(f\"{path}:{line}  matches  {pat!r}\")
    assert not violations, (
        \"Steps must consume context.filters.active(...) rather than \"
        \"iterating raw collections. If you genuinely need raw iteration, \"
        \"add the file to ALLOW_LIST with a one-line justification:\\n  \"
        + \"\\n  \".join(violations)
    )
"""


PRINCIPLES_TABLE = [
    ("Generic, not per-step",
     "B+",
     "Primitive is one class in a shared module. New step = call <code>ctx.filters.record(...)</code>. No framework changes."),
    ("Context flows through",
     "A",
     "FilterContext IS the flow. Same instance threaded through every step. Faces inherit parent state automatically."),
    ("One owner per fact",
     "A",
     "Each filter_name has exactly one producer. Static check forbids duplicate producers."),
    ("Fail loud",
     "A-",
     "Unknown filter_name asserts at <code>record()</code> time, not at export time. Static check rejects raw iteration."),
    ("Enforced, not advisory",
     "A",
     "This was missing in v1. Downstream steps go through <code>active()</code>; raw iteration fails static check."),
    ("Cross-framework drift protection",
     "A",
     "Shared module + shared exporter + parity test + static checks. Four independent guards."),
]


# -----------------------------------------------------------------------------
def render_filter_table() -> str:
    rows = []
    for name, level, file, desc, ui in KNOWN_FILTERS:
        is_hidden = ui.startswith("HIDDEN")
        cls = "bad" if is_hidden else "good"
        ui_html = (f'<span style="color:#a00">{escape(ui)}</span>'
                   if is_hidden else escape(ui))
        rows.append(
            f'<tr class="{cls}"><td><code>{escape(name)}</code></td>'
            f'<td>{escape(level)}</td>'
            f'<td><code>{escape(file)}</code></td>'
            f'<td>{escape(desc)}</td>'
            f'<td>{ui_html}</td></tr>'
        )
    return (
        '<table><thead><tr><th>canonical name</th><th>item type</th>'
        '<th>file</th><th>what it checks</th>'
        '<th>UI binding (or HIDDEN with reason)</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table>'
    )


def render_principles_table() -> str:
    rows = []
    for name, grade, body in PRINCIPLES_TABLE:
        rows.append(
            f'<tr><td><b>{escape(name)}</b></td>'
            f'<td><b>{escape(grade)}</b></td>'
            f'<td>{body}</td></tr>'
        )
    return ('<table><thead><tr><th>principle</th><th>grade</th><th>why</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>')


def render_current_state_table() -> str:
    rows = []
    for filt, writes, where, consumer, problem in CURRENT_STATE_TABLE:
        rows.append(
            f'<tr class="bad"><td><code>{escape(filt)}</code></td>'
            f'<td>{escape(writes)}</td>'
            f'<td>{escape(where)}</td>'
            f'<td>{escape(consumer)}</td>'
            f'<td>{escape(problem)}</td></tr>'
        )
    return (
        '<table><thead><tr><th>filter step</th><th>writes</th>'
        '<th>where it lives</th><th>consumer behavior</th>'
        '<th>problem</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table>'
    )


def main() -> None:
    log.info("Rendering design HTML v2")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>SIGHTING-059 Issues 1 &amp; 2 — Design (v2)</title>
<style>{CSS}</style></head><body>

<h1>SIGHTING-059 Issues 1 &amp; 2 — Design <small>(v2, corrected framing)</small></h1>
<p>v1 of this doc framed the problem as "every step should emit a decision
log". After user pushback, v2 reframes around the actual concern:
<span class="changed">filter decisions are advisory, not enforced</span>.
Face_46 ends up in faces.csv because nothing prevents downstream steps from
iterating the raw <code>insightface_faces</code> collection regardless of
filter intent.</p>

<p>v2 also adds a section explicitly addressing the drift risk between
the two pipeline frameworks (Albumify and FC App) — see §6.</p>

<div class="box note">
<b>One-line summary of the design</b><br>
Define a <code>FilterContext</code> in a shared module
(<code>face_cluster/filter_context.py</code>). Both pipelines hold one
on their context object. Every filter step calls
<code>ctx.filters.record(...)</code>; every downstream step iterates
<code>ctx.filters.active(item_type)</code>. End-of-run report is a free
byproduct. Drift across frameworks is mitigated by 4 independent guards.
</div>

<!-- ============================================================== -->
<h2>1. The actual problem (corrected from v1)</h2>

<div class="diagram">{escape(PROBLEM_DIAGRAM)}</div>

<p>Today every filter writes <i>something somewhere</i>, but no contract
forces downstream steps to consume it:</p>

{render_current_state_table()}

<p>Concrete consequences on the user's reference run
<code>face_clustering_20260510_231628</code>:</p>
<ul>
  <li>Cluster 6 contains face_46/47 — both passed every quality gate, but
      the crop stage silently skipped them and no decision exists explaining
      why. No filter recorded the drop.</li>
  <li>face_26 (area = 0.001 in fraction units, no enforcement) survives
      the "area" gate because the FC App quality gate's threshold defaults
      to 0 — making the filter a no-op rather than a deliberate decision.</li>
</ul>

<!-- ============================================================== -->
<h2>2. The right primitive</h2>

<div class="diagram">{escape(PROPOSED_DIAGRAM)}</div>

<h3>2.1 Types</h3>
<pre><code>{escape(CODE_PRIMITIVES)}</code></pre>

<h3>2.2 How a step changes — before vs. after</h3>
<pre><code>{escape(CODE_USAGE_BEFORE_AFTER)}</code></pre>

<div class="box good">
<b>Why this works for new steps automatically.</b><br>
Adding a new filter step tomorrow:
<ol>
  <li>Add its canonical name to <code>KNOWN_FILTERS</code> (one constant, one line).</li>
  <li>In the step's <code>process()</code>, call
      <code>ctx.filters.record(...)</code>.</li>
  <li>Downstream steps already use <code>ctx.filters.active(...)</code> —
      no edits needed.</li>
  <li>End-of-run report shows the new filter in its summary — no edits to
      the reporter.</li>
</ol>
That's the "flexible to new steps" constraint the user named, satisfied
by construction.
</div>

<!-- ============================================================== -->
<h2>3. Today's filter inventory (named)</h2>

<p>Every filter currently in the codebase gets one canonical name. These
go into <code>KNOWN_FILTERS</code>. <code>record()</code> asserts the
filter_name is in the set — typos fail fast.</p>

{render_filter_table()}

<!-- ============================================================== -->
<h2>4. End-of-run report (free byproduct)</h2>

<p>One method on FilterContext:</p>
<pre><code>ctx.filters.summary()
# →
# {{
#   \"image_quality\":        {{\"image\": 164}},
#   \"face_confidence\":      {{\"face\": 1284}},
#   \"face_bbox_ratio\":      {{\"face\": 219}},
#   \"face_top_k_per_image\": {{\"face\": 1463}},
#   \"face_crop\":            {{\"face\": 7}},     # ← THIS IS THE SIGHTING-059 #1 BUG
#   \"cluster_diameter_cap\": {{\"cluster\": 0}},
# }}</code></pre>

<p>Rendered in the FC App run summary, the UI immediately shows what every
filter did. Querying "where did face_46's crop go?" is a SQL row lookup,
not a code-reading exercise.</p>

<!-- ============================================================== -->
<h2>5. Senior-SWE evaluation of the design principles</h2>

<p>The user asked: "are these design principles a good idea?" Honest
grades after the v1→v2 correction:</p>

{render_principles_table()}

<div class="box note">
<b>What I'd push on if I were reviewing this PR.</b>
<ul>
  <li>The <code>active(after=...)</code> argument relies on a
      <code>_filter_position(name)</code> ordering. That ordering must be
      a declared constant, not derived implicitly from registration order.
      Otherwise "active after image_quality" becomes brittle.</li>
  <li>Faces inherit parent-image state recursively. If we add cluster-level
      filters that affect faces, the parent chain grows. Keep
      <code>_active_recursive</code> bounded — assert no cycles, depth limit.</li>
  <li>FilterContext.record() must be idempotent on (item_id, filter_name).
      Re-running a step (recluster, remerge) shouldn't append duplicate
      decisions. Add a <code>.reset(filter_name)</code> or
      <code>.replace()</code> semantics — TBD which.</li>
</ul>
</div>

<!-- ============================================================== -->
<h2>6. Keeping FC App and Albumify from drifting</h2>

<p>The user's question after reviewing v1: <i>"I have a concern that the
face clustering app will work differently from the pipeline. How do we
mitigate it?"</i></p>

<p>This is a real risk. Spec-020 deliberately kept the two apps' pipeline
frameworks separate ("connected by data, not code"). Adding a new shared
primitive means a new place where they can drift. Four independent
mitigations:</p>

<div class="diagram">{escape(CROSS_PIPELINE_DRIFT_DIAGRAM)}</div>

<h3>6.1 Single source of truth for the primitive</h3>
<p><code>face_cluster/filter_context.py</code> defines
<code>FilterDecision</code>, <code>ItemState</code>,
<code>FilterContext</code>, and <code>KNOWN_FILTERS</code>. Both
<code>sim_bench.pipeline.context</code> and
<code>face_cluster.pipeline._RunContext</code> import the same classes.
There is exactly one definition. Adding a filter name in one app
automatically makes it available in the other (and the
<code>record()</code> assert fires symmetrically).</p>

<h3>6.2 Single exporter (already shared post-spec-030)</h3>
<p><code>RunExporter</code> (spec-030 Phase 1, already on main) writes
<code>filter_decisions.csv</code> with the same schema regardless of which
pipeline called it. The file format is the contract; spec-020 already
established this pattern for faces.csv / merge_log.json / etc.</p>

<h3>6.3 Cross-pipeline parity test</h3>
<pre><code>{escape(CODE_PARITY_TEST)}</code></pre>
<p>Runs both pipelines on a synthetic 5-image fixture, asserts the
<code>filter_decisions.csv</code> they produce has the same columns,
overlapping filter names, and consistent semantics on a known item.
If either app silently diverges on what filters it applies, this fails
in CI.</p>

<h3>6.4 Static check: no raw-collection iteration</h3>
<pre><code>{escape(CODE_STATIC_CHECK)}</code></pre>
<p>Greps both apps for forbidden patterns (<code>for x in
context.image_paths</code>, <code>for f in ctx.faces</code>). Steps that
need raw iteration must opt in via the explicit
<code>ALLOW_LIST</code> with a one-line justification.</p>

<div class="box good">
<b>Net effect on drift risk.</b>
A new filter added in Albumify but not in FC App produces a parity-test
failure. A step that bypasses <code>filters.active()</code> produces a
static-check failure. A typo in a filter name fails at runtime via the
KNOWN_FILTERS assert. Three independent guard rails that don't rely on
human discipline.
</div>

<!-- ============================================================== -->
<h2>6.5 UI ↔ filter alignment (new, user-requested)</h2>

<p>The user's concern: "I want to avoid the situation where face_size in
pixels is ignored and you're using some hidden filter on face area ratio."
The "Min Face Size (px)" slider in Configure &amp; Run plumbs to
<code>min_face_size</code>, which today is not a rejector anywhere — small
faces still survive detection and clustering. Meanwhile a different
filter (<code>min_face_ratio</code>) is the real gate, in three places,
all hidden from the UI. That is the exact failure mode this section
prevents.</p>

<h3>Bidirectional contract</h3>
<table>
<tr><th>Direction</th><th>Rule</th><th>Enforced by</th></tr>
<tr class="good"><td>filter → UI</td>
  <td>Every <code>KNOWN_FILTERS</code> entry declares one of three:
      <code>UI:</code> (with widget keys), <code>HIDDEN(permanent):</code>
      (genuinely not user-tunable), or <code>HIDDEN(todo, target=Px):</code>
      (must be exposed/resolved by phase Px). No middle ground.</td>
  <td>Static check parses the entry and verifies the binding form.</td></tr>
<tr class="good"><td>UI → filter</td>
  <td>Every Streamlit widget key in the UI source files that looks like a
      filter knob (matches <code>FILTER_KEY_PREFIXES</code>) must be
      referenced by exactly one <code>KNOWN_FILTERS</code> entry. No
      phantom controls.</td>
  <td>Static check greps the UI source files and asserts the keys it finds
      are all bound.</td></tr>
<tr class="good"><td>filter exists</td>
  <td>Every binding <code>(config_min_face_size)</code> in a
      <code>UI:</code> string must actually exist as a
      <code>key=</code> in the UI source files. Stale bindings fail CI.</td>
  <td>Same static check.</td></tr>
<tr class="good"><td>hidden is documented and time-boxed</td>
  <td><code>HIDDEN(permanent)</code> needs a reason. <code>HIDDEN(todo,
      target=Px)</code> needs a reason AND a target phase. When phase
      Px ships, all matching entries must be resolved (exposed, removed,
      or promoted to <code>HIDDEN(permanent)</code>) — otherwise CI fails.
      Prevents the escape-hatch decay where "hidden with reason"
      accumulates forever.</td>
  <td>Static check cross-references against <code>SHIPPED_PHASES</code>.</td></tr>
</table>

<div class="box warn">
<b>The two-form distinction (your direct question).</b><br>
"HIDDEN" is not a decision to leave filters hidden. It is a current-state
label that comes in two forms:
<ul>
<li><code>HIDDEN(permanent)</code> — the filter is not a tunable knob, period
    (e.g. <code>face_crop</code> is a binary save/skip outcome).
    Stays this way forever; one entry today.</li>
<li><code>HIDDEN(todo, target=Px)</code> — the filter exists in code but isn't
    exposed yet. The target phase is a contract: once Px ships, this entry
    must be resolved. CI rejects the merge that ships Px if any
    <code>HIDDEN(todo, target=Px)</code> entries are still present.
    4 entries today (image_portrait, face_bbox_ratio, face_relative_size,
    face_eye_ratio).</li>
</ul>
This is the mechanism that prevents the SIGHTING-059 anti-pattern from
re-occurring as architecture decay.
</div>

<pre><code>{escape(CODE_UI_ALIGNMENT_CHECK)}</code></pre>

<div class="box good">
<b>Why this closes the user's concern.</b><br>
The "Min Face Size (px)" example reads in the new world like:
<ul>
  <li>Today's broken state: filter <code>face_area</code> has
      <code>UI: (run_min_face_area)</code> but the slider is a no-op
      because units mismatch (SIGHTING-060).</li>
  <li>After P6: filter <code>face_area</code> uses pixels, the slider's
      "px" label is honest, and the static check verifies the binding still
      works.</li>
  <li>Today's hidden state: <code>face_bbox_ratio</code> /
      <code>face_relative_size</code> / <code>face_eye_ratio</code> are
      currently <code>HIDDEN</code> with documented reason
      ("not exposed yet; flagged for UI in Phase 2"). The static check
      forces those flags to be present and reviewed.</li>
</ul>
The bug pattern is structurally impossible to land silently.
</div>

<!-- ============================================================== -->
<h2>7. Implementation phases</h2>

<table>
<tr><th>Phase</th><th>Scope</th><th>Risk</th><th>Tag</th></tr>
<tr><td>P0</td>
  <td>Define the primitive in <code>face_cluster/filter_context.py</code>.
      Add <code>filters: FilterContext</code> field to both
      <code>PipelineContext</code> and <code>_RunContext</code>. No
      existing step changes. Static check + unit tests only.</td>
  <td>Low — additive.</td>
  <td><span class="tag p1">P1</span></td></tr>
<tr><td>P1</td>
  <td>Migrate <code>filter_quality</code> (Albumify) and
      <code>QualityGater</code> (FC App) to call
      <code>filters.record(...)</code>. Both still also write their legacy
      <code>quality_passed</code> / <code>quality_*_pass</code> outputs
      during the transition.</td>
  <td>Low — dual-write.</td>
  <td><span class="tag p1">P1</span></td></tr>
<tr><td>P2</td>
  <td>Migrate the SIGHTING-059 culprit: <code>save_crops</code> in FC App
      emits <code>filter_name="face_crop"</code> with the actual reason
      (landmarks_missing / alignment_failed / etc). This is the
      user-visible fix for cluster 6 having no thumbnails.</td>
  <td>Medium — needs source reading to enumerate skip reasons.</td>
  <td><span class="tag p1">P1</span></td></tr>
<tr><td>P3</td>
  <td>Add <code>RunExporter</code> support for
      <code>filter_decisions.csv</code>. Wire FC App and Albumify export
      paths.</td>
  <td>Low — additive, uses existing v4 layout.</td>
  <td><span class="tag p1">P1</span></td></tr>
<tr><td>P4</td>
  <td>Parity test + static check land in CI. Static check starts with
      a large allow-list, shrinks over time as each step migrates.</td>
  <td>Medium — every step author has to think about it from this point.</td>
  <td><span class="tag p2">P2</span></td></tr>
<tr><td>P5</td>
  <td>Migrate remaining steps off the allow-list, one PR each. Each PR:
      add <code>filters.record</code> calls, switch from raw iteration to
      <code>filters.active</code>, remove allow-list entry.</td>
  <td>Low — incremental.</td>
  <td><span class="tag p3">P3 (ongoing)</span></td></tr>
<tr><td>P6</td>
  <td>SIGHTING-060 fix lands here: <code>face_area</code> filter is
      canonicalized in <code>face_cluster/face_geometry.py</code>
      (pixels everywhere). The filter records the same shape regardless
      of which producer ran.</td>
  <td>Medium — touches the unit-drift bug.</td>
  <td><span class="tag p1">P1</span></td></tr>
</table>

<!-- ============================================================== -->
<h2>8. Locked decisions (formerly open questions)</h2>

<div class="box good">
User confirmed v2 framing and asked to move to implementation. The four
questions are resolved as follows:
<ol>
<li><b>Idempotency on re-run.</b> <code>record()</code> <span class="changed">REPLACES</span>
    any prior decision for the same (item_id, filter_name) pair, and logs a
    debug message. Re-running recluster/remerge produces clean output
    without duplicate rows. Asserting "must already exist" would be too
    strict (a recluster might run a filter the original didn't).</li>
<li><b>Filter ordering.</b> <span class="changed">List-ordered
    <code>KNOWN_FILTERS</code></span> is the canonical ordering. One list,
    one place to look. <code>active(after="image_quality")</code> uses
    list position.</li>
<li><b>Allow-list size.</b> <span class="changed">Initial allow-list
    includes every existing step</span> (P4 lands the check with the
    current state grandfathered in). P5 shrinks one step per PR. CI
    blocks NEW step-author regressions immediately; existing-step
    migration is incremental and reviewed.</li>
<li><b>Backward-compat fields.</b> <span class="changed">Keep
    <code>context.quality_passed</code> / <code>active_images</code> /
    <code>face["filter_passed"]</code></span> through P5 with deprecation
    warnings. Remove in a P7 cleanup PR once nothing reads them.</li>
</ol>
</div>

<h3>One additional decision baked into v2.1:</h3>
<div class="box good">
<b>UI/filter alignment is enforced as a first-class guard</b> (new §6.5),
not a soft principle. The "Min Face Size (px)" / hidden
<code>min_face_ratio</code> bug pattern fails CI through the bidirectional
<code>KNOWN_FILTERS</code> ↔ widget-key static check.
</div>

<!-- ============================================================== -->
<h2>Appendix — what v1 got wrong</h2>

<table>
<tr><th>v1 claim</th><th>v2 correction</th></tr>
<tr><td>"Every step emits a Decision."</td>
  <td>Wrong emphasis. The point isn't logging — it's that filter
      decisions must be queryable STATE that gates downstream steps.</td></tr>
<tr><td>"PER_ITEM_OBSERVABILITY=False as opt-out flag."</td>
  <td>Mechanical decorator; weakens the contract. v2 uses an explicit
      <code>ALLOW_LIST</code> in the static check, requiring a code
      review to add entries.</td></tr>
<tr><td>"Bridging the FC App framework" framed as a problem.</td>
  <td>Spec-020 deliberately kept the frameworks separate.
      v2 accepts that and addresses drift with shared module + shared
      exporter + parity test + static checks.</td></tr>
<tr><td>"Wide columns on faces.csv + long table on decisions.csv".</td>
  <td>v2 keeps the long table (now <code>filter_decisions.csv</code>) as
      the source of truth. Wide aggregate columns on faces.csv are
      derived view, optional.</td></tr>
</table>

</body></html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    log.info("Wrote %s (%.1f KiB)", OUT_HTML, OUT_HTML.stat().st_size / 1024)


if __name__ == "__main__":
    main()
