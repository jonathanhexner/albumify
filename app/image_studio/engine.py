"""Image Analysis Studio — generalized comparison engine (spec-094 Slice 2).

PURE orchestration, no Streamlit. Runs a user-selected set of per-image analysis
methods over a folder and normalizes every method's output into a uniform
``AnalysisColumn`` (the abstraction that makes "flat vs category tabs" trivial).

Each method is backed by a real, ``universal_cache``-persisted pipeline step
(spec-094 Slice 1), so a second run is incremental. The UI (``main.py`` /
``view.py``, Slice 3) is a thin consumer of ``run_methods``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.extract_geo_metadata import ExtractGeoMetadataStep
from sim_bench.pipeline.steps.infer_geo_clip import InferGeoClipStep
from sim_bench.pipeline.steps.infer_geo_coords import InferGeoCoordsStep
from sim_bench.pipeline.steps.caption_images import CaptionImagesStep
from sim_bench.pipeline.steps.score_iqa import ScoreIQAStep
from sim_bench.pipeline.steps.score_ava import ScoreAVAStep
from sim_bench.pipeline.steps.score_quality import ScoreQualityStep  # spec-093
from sim_bench.image_quality_models.pyiqa_model_wrapper import PyIQAModel, PYIQA_METRICS

logger = logging.getLogger(__name__)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff")

CATEGORY_QUALITY = "image_quality"
CATEGORY_GEO = "geo_location_and_caption"

# kinds
NUMERIC, LABEL_CONF, TEXT, COORD = "numeric", "label_conf", "text", "coord"

_DEFAULT_AVA_CKPT = "models/album_app/ava_resnet50.pt"


@dataclass
class AnalysisColumn:
    """One method's output for one image, normalized for display + sorting."""

    key: str
    category: str
    kind: str  # numeric | label_conf | text | coord
    sort_value: Optional[float] = None  # what the column sorts by (None = unsortable)
    display: str = "-"  # short cell text
    topk: list = field(default_factory=list)  # full payload (top-k + confidence)


# --------------------------------------------------------------------------- #
# Image discovery
# --------------------------------------------------------------------------- #
def discover_images(folder: str, limit: Optional[int] = None) -> List[str]:
    """Return sorted image paths in ``folder`` (capped at ``limit``).

    Missing/empty/non-dir folder -> ``[]`` (never raises). Logs how many were
    skipped by the cap (no silent truncation).
    """
    if not folder or not os.path.isdir(folder):
        return []
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))
    total = len(names)
    if limit is not None and limit >= 0:
        names = names[:limit]
    if total > len(names):
        logger.info("discover_images: showing %d of %d images in %s (capped)",
                    len(names), total, folder)
    return [os.path.join(folder, n) for n in names]


# --------------------------------------------------------------------------- #
# Column mappers — context field -> {path: AnalysisColumn}
# --------------------------------------------------------------------------- #
def _map_exif(ctx: PipelineContext) -> Dict[str, AnalysisColumn]:
    out = {}
    for p, m in ctx.geo_metadata.items():
        gps = f"{m.lat:.4f}, {m.lon:.4f}" if m.has_geo else ""
        tm = m.timestamp.strftime("%Y-%m-%d %H:%M") if m.has_time else ""
        disp = " | ".join(x for x in (gps, tm) if x) or "-"
        out[p] = AnalysisColumn("exif", CATEGORY_GEO, COORD, None, disp,
                                [{"lat": m.lat, "lon": m.lon, "timestamp": tm or None}])
    return out


def _map_streetclip(ctx: PipelineContext) -> Dict[str, AnalysisColumn]:
    out = {}
    for p, preds in ctx.geo_clip_predictions.items():
        if preds:
            top = preds[0]
            out[p] = AnalysisColumn("streetclip", CATEGORY_GEO, LABEL_CONF,
                                    float(top["score"]),
                                    f"{top['label']} ({top['score']:.2f})", preds)
        else:
            out[p] = AnalysisColumn("streetclip", CATEGORY_GEO, LABEL_CONF, None, "-", [])
    return out


def _map_geoclip(ctx: PipelineContext) -> Dict[str, AnalysisColumn]:
    out = {}
    for p, preds in ctx.geo_coord_predictions.items():
        if preds:
            top = preds[0]
            place = top.get("place") or f"{top['lat']:.3f}, {top['lon']:.3f}"
            out[p] = AnalysisColumn("geoclip", CATEGORY_GEO, LABEL_CONF,
                                    float(top["prob"]),
                                    f"{place} ({top['prob']:.2f})", preds)
        else:
            out[p] = AnalysisColumn("geoclip", CATEGORY_GEO, LABEL_CONF, None, "-", [])
    return out


def _map_blip(ctx: PipelineContext) -> Dict[str, AnalysisColumn]:
    return {p: AnalysisColumn("blip", CATEGORY_GEO, TEXT, None, cap or "-", [])
            for p, cap in ctx.image_captions.items()}


def _scalar_mapper(field_name: str, key: str) -> Callable[[PipelineContext], Dict[str, AnalysisColumn]]:
    def mapper(ctx: PipelineContext) -> Dict[str, AnalysisColumn]:
        scores = getattr(ctx, field_name, {}) or {}
        return {p: AnalysisColumn(key, CATEGORY_QUALITY, NUMERIC, float(v), f"{float(v):.3f}", [])
                for p, v in scores.items()}
    return mapper


# --------------------------------------------------------------------------- #
# Method registry
# --------------------------------------------------------------------------- #
@dataclass
class _Method:
    key: str
    category: str
    label: str
    order: int
    step_id: str  # methods sharing a step_id run that step once
    make_step: Callable[[], Any]
    config: dict
    available: Callable[[], bool]
    to_columns: Callable[[PipelineContext], Dict[str, AnalysisColumn]]


def _have(*mods: str) -> bool:
    try:
        return all(importlib.util.find_spec(m) is not None for m in mods)
    except (ImportError, ValueError):
        return False


def _merge_configs(cfgs: List[dict]) -> dict:
    """Merge configs of methods sharing a backing step: list values union
    (order-preserving), dict values shallow-merge, scalars last-wins."""
    out: dict = {}
    for c in cfgs:
        for k, v in c.items():
            if isinstance(v, list):
                cur = out.setdefault(k, [])
                cur.extend(x for x in v if x not in cur)
            elif isinstance(v, dict):
                out.setdefault(k, {}).update(v)
            else:
                out[k] = v
    return out


_VISION = lambda: _have("transformers", "torch")  # noqa: E731

METHODS: Dict[str, _Method] = {m.key: m for m in [
    # --- geo / location / caption family -----------------------------------
    _Method("exif", CATEGORY_GEO, "EXIF (GPS + time)", 10, "extract_geo_metadata",
            lambda: ExtractGeoMetadataStep(), {}, lambda: True, _map_exif),
    _Method("streetclip", CATEGORY_GEO, "StreetCLIP (city)", 11, "infer_geo_clip",
            lambda: InferGeoClipStep(), {"top_k": 3}, _VISION, _map_streetclip),
    _Method("geoclip", CATEGORY_GEO, "GeoCLIP (lat/lon)", 12, "infer_geo_coords",
            lambda: InferGeoCoordsStep(), {"top_k": 3}, lambda: _have("geoclip"), _map_geoclip),
    _Method("blip", CATEGORY_GEO, "BLIP (caption)", 13, "caption_images",
            lambda: CaptionImagesStep(), {}, _VISION, _map_blip),
    # --- image-quality family: existing rule-based/AVA steps ----------------
    _Method("iqa", CATEGORY_QUALITY, "Rule-based IQA", 20, "score_iqa",
            lambda: ScoreIQAStep(), {}, lambda: True, _scalar_mapper("iqa_scores", "iqa")),
    _Method("sharpness", CATEGORY_QUALITY, "Sharpness", 21, "score_iqa",
            lambda: ScoreIQAStep(), {}, lambda: True, _scalar_mapper("sharpness_scores", "sharpness")),
    _Method("ava", CATEGORY_QUALITY, "AVA aesthetic", 22, "score_ava",
            lambda: ScoreAVAStep(), {"checkpoint_path": _DEFAULT_AVA_CKPT},
            lambda: os.path.exists(_DEFAULT_AVA_CKPT), _scalar_mapper("ava_scores", "ava")),
]}


# --------------------------------------------------------------------------- #
# spec-093 pyiqa metrics — image_quality family, backed by ScoreQualityStep.
# --------------------------------------------------------------------------- #
def _method_scores_mapper(key: str) -> Callable[[PipelineContext], Dict[str, AnalysisColumn]]:
    """Column mapper reading ``ctx.method_scores[path][key]`` (score = higher-better)."""
    def mapper(ctx: PipelineContext) -> Dict[str, AnalysisColumn]:
        out = {}
        for p, per_method in (getattr(ctx, "method_scores", None) or {}).items():
            v = per_method.get(key)
            if v is not None:
                out[p] = AnalysisColumn(key, CATEGORY_QUALITY, NUMERIC, float(v), f"{float(v):.3f}", [])
        return out
    return mapper


_PYIQA_LABELS = {"maniqa": "MANIQA", "musiq": "MUSIQ", "hyperiqa": "HyperIQA",
                 "brisque": "BRISQUE", "niqe": "NIQE", "clipiqa": "CLIP-IQA"}

# All pyiqa metrics share ONE ScoreQualityStep run: it scores N methods in a
# single call and writes them together into ctx.method_scores. They therefore
# share step_id "score_quality"; run_methods merges their per-method
# config ({"methods": [...]}) into one union list so a single step run produces
# every selected metric. (A distinct step_id per metric would be wrong: each
# ScoreQualityStep.process REPLACES ctx.method_scores, so the last run would
# wipe the others.) Per-metric caching still happens inside the step
# (feature_type="quality_<metric>").
for _i, _mk in enumerate(PYIQA_METRICS):
    METHODS[_mk] = _Method(
        key=_mk, category=CATEGORY_QUALITY, label=_PYIQA_LABELS.get(_mk, _mk),
        order=30 + _i, step_id="score_quality",
        make_step=lambda: ScoreQualityStep(), config={"methods": [_mk]},
        available=PyIQAModel.is_available, to_columns=_method_scores_mapper(_mk),
    )


def available_methods() -> List[dict]:
    """List every registered method with its category + availability (for checkboxes)."""
    return [{"key": m.key, "category": m.category, "label": m.label, "available": m.available()}
            for m in sorted(METHODS.values(), key=lambda x: x.order)]


def categories() -> List[str]:
    """Categories in display order (dedup, order-preserving)."""
    seen, out = set(), []
    for m in sorted(METHODS.values(), key=lambda x: x.order):
        if m.category not in seen:
            seen.add(m.category)
            out.append(m.category)
    return out


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run_methods(
    paths: List[str],
    selected: List[str],
    *,
    cache_handler: Any = None,
    config: Optional[Dict[str, dict]] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Dict[str, AnalysisColumn]]:
    """Run ``selected`` methods over ``paths``; return ``{path: {method_key: AnalysisColumn}}``.

    Only selected methods run. Methods sharing a backing step run it once.
    A failing method is logged and skipped (its column simply absent) — never
    raises. Pass ``cache_handler`` to persist/read via ``universal_cache``.
    """
    config = config or {}
    paths = [str(p) for p in paths]
    methods = [METHODS[k] for k in selected if k in METHODS]
    methods.sort(key=lambda m: m.order)

    ctx = PipelineContext(image_paths=list(paths), cache_handler=cache_handler)

    # 1) group selected methods by backing step and run each step ONCE with the
    #    merged config (so e.g. maniqa+niqe -> one score_quality run scoring both).
    groups: "OrderedDict[str, list]" = OrderedDict()
    for m in methods:
        groups.setdefault(m.step_id, []).append(m)

    for gi, (step_id, ms) in enumerate(groups.items()):
        if progress:
            progress(gi / max(len(groups), 1), f"Running {ms[0].label}")
        merged = _merge_configs([m.config for m in ms])
        for m in ms:  # per-method user overrides (methods union preserved)
            merged = {**merged, **config.get(m.key, {})}
        try:
            ms[0].make_step().process(ctx, merged)
        except Exception as e:  # a broken step must not sink the whole run
            logger.warning("step %s (%s) failed: %s",
                           step_id, ",".join(m.key for m in ms), e)

    # 2) map context -> columns
    columns: Dict[str, Dict[str, AnalysisColumn]] = {p: {} for p in paths}
    for m in methods:
        try:
            per = m.to_columns(ctx)
        except Exception as e:
            logger.warning("mapper %s failed: %s", m.key, e)
            per = {}
        for p, col in per.items():
            columns.setdefault(p, {})[m.key] = col

    if progress:
        progress(1.0, "Done")
    return columns
