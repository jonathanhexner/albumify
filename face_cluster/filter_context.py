"""Filter context — typed, queryable state for filter decisions in the pipeline.

Spec: specs/032-filter-context/spec.md
Design: specs/033-data-integrity/design.html

Background: today, filter decisions are advisory.  filter_quality writes a
set of paths to context.quality_passed, but nothing forces downstream steps
to consume it — they iterate context.image_paths or context.insightface_faces
directly and silently bypass the filter.  Result: SIGHTING-059 #1 (face_46's
crop disappears with no recorded reason), SIGHTING-060 (face.area unit drift
hides which filter actually runs), and the "Min Face Size (px)" slider that
controls nothing.

This module makes filter decisions a first-class typed state on the pipeline
context.  Both Albumify (sim_bench.pipeline) and the FC App
(face_cluster.pipeline) hold a ``FilterContext`` on their context object —
same class, one source of truth.

Contract:
    step.process()  →  ctx.filters.record(item_id, filter_name=..., ...)
    downstream      →  ctx.filters.active(item_type)   (NOT raw iteration)
    export          →  ctx.filters.summary()           writes to CSV
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


ItemType = Literal["image", "face", "cluster"]


# -----------------------------------------------------------------------------
# Canonical filter registry
# -----------------------------------------------------------------------------
# (name, item_type, file, description, ui_binding)
#
# ui_binding grammar — exactly one of:
#   "UI: <label> (<widget_key>) [+ (<widget_key>)]..."
#   "HIDDEN(permanent): <reason>"
#   "HIDDEN(todo, target=Px): <reason>"
#
# The static check in tests/architecture/test_ui_aligns_with_filters.py
# parses this column and fails CI on malformed entries or unresolved
# HIDDEN(todo) entries whose target phase has shipped.
KNOWN_FILTERS: List[Tuple[str, ItemType, str, str, str]] = [
    ("image_quality",        "image",  "filter_quality.py",
        "IQA / sharpness thresholds",
        "UI: Album Configure & Run 'Min IQA Score' (config_min_iqa) + 'Min Sharpness' (config_min_sharpness)"),
    ("image_portrait",       "image",  "filter_portraits.py",
        "min_face_ratio for largest face",
        "HIDDEN(todo, target=P3): not exposed today; add 'Min largest-face ratio' slider"),
    ("face_confidence",      "face",   "filter_faces.py / detect",
        "InsightFace det_score",
        "UI: 'Min Detection Confidence' (config_det_conf)"),
    ("face_bbox_ratio",      "face",   "filter_faces.py",
        "bbox_w / image_w >= threshold",
        "UI: Album Configure & Run 'Min Face Bbox Ratio' (config_min_bbox_ratio)"),
    ("face_relative_size",   "face",   "filter_faces.py",
        "this bbox / largest in image",
        "HIDDEN(todo, target=P3): decide: expose as slider or remove"),
    ("face_eye_ratio",       "face",   "filter_faces.py",
        "inter-eye distance / image_w",
        "HIDDEN(todo, target=P3): decide: expose as slider or remove"),
    ("face_blur",            "face",   "face_cluster/quality.py",
        "Laplacian variance >= blur_min",
        "UI: FC App Run tab 'blur_min' (run_blur_min)"),
    ("face_pose_yaw",        "face",   "face_cluster/quality.py",
        "|yaw| <= yaw_max",
        "UI: FC App Run tab 'yaw_max' (run_yaw_max)"),
    ("face_pose_pitch",      "face",   "face_cluster/quality.py",
        "|pitch| <= pitch_max",
        "UI: FC App Run tab 'pitch_max' (run_pitch_max)"),
    ("face_pose_roll",       "face",   "face_cluster/quality.py",
        "|roll| <= roll_max",
        "UI: FC App Run tab 'roll_max' (run_roll_max)"),
    ("face_area",            "face",   "face_cluster/quality.py",
        "area >= min_face_area",
        "UI: FC App 'min_face_area' (run_min_face_area); UNIT MISMATCH today — fixed in P6 (SIGHTING-060)"),
    ("face_top_k_per_image", "face",   "face_cluster/quality.py",
        "keep N largest faces per image",
        "UI: FC App 'max_faces_per_image_core' (run_max_faces)"),
    ("face_crop",            "face",   "face_cluster/pipeline._save_crops",
        "crop persisted to disk — SIGHTING-059 #1",
        "HIDDEN(permanent): binary success/skip, not a tunable threshold"),
    ("cluster_diameter_cap", "cluster", "face_cluster/cluster_diameter_cap.py",
        "spec-031 absolute diameter ceiling",
        "UI: FC App merge controls (run_cap_enabled) + (run_cap_max_full) + (run_cap_max_exemplar) "
        "+ (rc_cap_enabled) + (rc_cap_max_full) + (rc_cap_max_exemplar)"),
]


_FILTER_NAMES = frozenset(name for name, *_ in KNOWN_FILTERS)
_FILTER_ITEM_TYPE: Dict[str, ItemType] = {n: t for n, t, *_ in KNOWN_FILTERS}
_FILTER_ORDER: Dict[str, int] = {n: i for i, (n, *_) in enumerate(KNOWN_FILTERS)}


def filter_position(name: str) -> int:
    """Position of a filter in KNOWN_FILTERS (the canonical ordering).

    Used by ``FilterContext.active(after=...)`` to bound the visible
    decisions.  Locked decision (spec-032): KNOWN_FILTERS list order IS
    the canonical filter order.
    """
    return _FILTER_ORDER[name]


# -----------------------------------------------------------------------------
# Core types
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterDecision:
    """One filter's verdict for one item.

    Immutable once recorded.  Re-recording for the same (item_id, filter_name)
    replaces the previous decision in FilterContext (locked decision: replace,
    not append).
    """
    filter_name: str
    rejected:    bool
    reason:      str
    measured:    Dict   # measured values that produced the verdict


@dataclass
class ItemState:
    """One item's full filter history."""
    item_id:   str
    item_type: ItemType
    parent_id: Optional[str] = None
    decisions: List[FilterDecision] = field(default_factory=list)

    @property
    def self_active(self) -> bool:
        """True if this item passed every recorded filter (ignoring ancestors)."""
        return not any(d.rejected for d in self.decisions)


# -----------------------------------------------------------------------------
# FilterContext — the single primitive both pipelines hold
# -----------------------------------------------------------------------------
class FilterContext:
    """Typed, queryable filter state shared by both pipeline frameworks.

    Lives on ``PipelineContext.filters`` in Albumify and on
    ``_RunContext.filters`` in the FC App.  Same class, one definition.
    """

    def __init__(self) -> None:
        self._items: Dict[str, ItemState] = {}

    # -- registration -------------------------------------------------------
    def register(
        self,
        item_id:   str,
        item_type: ItemType,
        parent_id: Optional[str] = None,
    ) -> None:
        """Declare an item exists.  Idempotent on item_id.

        Called by the producer step (``discover_images`` for images,
        ``detect_faces`` for faces, the merge step for clusters).  Late
        ``record()`` calls auto-register if missing — that keeps existing
        steps mostly unchanged during P1 migration.
        """
        existing = self._items.get(item_id)
        if existing is None:
            self._items[item_id] = ItemState(item_id, item_type, parent_id)
            return
        # Already registered.  If the caller supplied a parent_id and the
        # existing record has none, fill it in.  Otherwise it's a no-op.
        if parent_id is not None and existing.parent_id is None:
            existing.parent_id = parent_id

    # -- recording ----------------------------------------------------------
    def record(
        self,
        item_id:     str,
        *,
        filter_name: str,
        rejected:    bool,
        reason:      str,
        measured:    Optional[Dict] = None,
        item_type:   Optional[ItemType] = None,
        parent_id:   Optional[str] = None,
    ) -> None:
        """Record a filter's verdict for an item.

        If the (item_id, filter_name) pair has been recorded before, this
        REPLACES the previous decision (and logs DEBUG).  Locked decision
        per spec-032 — supports recluster/remerge re-runs.

        If the item hasn't been registered yet, auto-register it.  The
        ``item_type`` of the filter (looked up in KNOWN_FILTERS) takes
        precedence; callers may pass item_type/parent_id for clarity.
        """
        if filter_name not in _FILTER_NAMES:
            raise ValueError(
                f"Unknown filter_name {filter_name!r}.  Add it to KNOWN_FILTERS "
                f"in face_cluster/filter_context.py first."
            )
        canonical_type = _FILTER_ITEM_TYPE[filter_name]
        effective_type = item_type or canonical_type

        if item_id not in self._items:
            self.register(item_id, effective_type, parent_id)

        item = self._items[item_id]
        # Replace prior decision on (item_id, filter_name) collisions.
        for i, d in enumerate(item.decisions):
            if d.filter_name == filter_name:
                logger.debug("FilterContext: replacing decision %s on %s",
                             filter_name, item_id)
                item.decisions[i] = FilterDecision(
                    filter_name=filter_name, rejected=rejected,
                    reason=reason, measured=dict(measured or {}),
                )
                return
        item.decisions.append(FilterDecision(
            filter_name=filter_name, rejected=rejected,
            reason=reason, measured=dict(measured or {}),
        ))

    # -- query --------------------------------------------------------------
    def active(
        self,
        item_type: ItemType,
        after:     Optional[str] = None,
    ) -> Iterable[ItemState]:
        """Yield items of ``item_type`` that are still active.

        ``after``: if given, restrict visible decisions to filters at or
        before that position in KNOWN_FILTERS (the canonical ordering).
        Useful for "what's still alive at this stage of the pipeline".

        Faces inherit parent-image state.  A face is inactive if either
        the face failed a filter or its parent image is inactive.
        """
        cutoff = filter_position(after) if after is not None else None
        for item in self._items.values():
            if item.item_type != item_type:
                continue
            if not self._active(item, cutoff):
                continue
            yield item

    def is_active(self, item_id: str, after: Optional[str] = None) -> bool:
        """True iff the named item is still active (parent chain respected)."""
        cutoff = filter_position(after) if after is not None else None
        item = self._items.get(item_id)
        if item is None:
            # An unknown item is conservatively treated as inactive — callers
            # should register() before querying.  Returning False prevents
            # silent passes on typos.
            return False
        return self._active(item, cutoff)

    def get(self, item_id: str) -> Optional[ItemState]:
        return self._items.get(item_id)

    def _active(self, item: ItemState, cutoff: Optional[int]) -> bool:
        if cutoff is None:
            decisions = item.decisions
        else:
            decisions = [d for d in item.decisions
                         if filter_position(d.filter_name) <= cutoff]
        if any(d.rejected for d in decisions):
            return False
        if item.parent_id:
            parent = self._items.get(item.parent_id)
            if parent is not None and not self._active(parent, cutoff):
                return False
        return True

    # -- reporting ----------------------------------------------------------
    def summary(self) -> Dict[str, Dict[str, int]]:
        """End-of-run report: filter_name -> item_type -> rejection count.

        Used by the FC App Run Summary tab and by exporters.
        """
        out: Dict[str, Dict[str, int]] = {}
        for item in self._items.values():
            for d in item.decisions:
                if d.rejected:
                    out.setdefault(d.filter_name, {})
                    out[d.filter_name].setdefault(item.item_type, 0)
                    out[d.filter_name][item.item_type] += 1
        return out

    def all_decisions(self) -> List[Tuple[ItemState, FilterDecision]]:
        """Flat iteration over (item, decision) pairs for export."""
        out = []
        for item in self._items.values():
            for d in item.decisions:
                out.append((item, d))
        return out

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items


# -----------------------------------------------------------------------------
# UI-alignment introspection helpers
# -----------------------------------------------------------------------------
# Used by tests/architecture/test_ui_aligns_with_filters.py.  Kept here so the
# parser and the data live in one module.

_RE_HIDDEN_TODO = re.compile(r"^HIDDEN\(todo,\s*target=(P\d+)\):\s*(.+)$")
_RE_HIDDEN_PERM = re.compile(r"^HIDDEN\(permanent\):\s*(.+)$")
_RE_UI_KEYS     = re.compile(r"\((\w+)\)")


def parse_ui_binding(binding: str) -> Tuple[str, Optional[str], Optional[str], List[str]]:
    """Parse a ui_binding string into (kind, target_phase, reason, widget_keys).

    kind is one of: 'ui', 'hidden_permanent', 'hidden_todo', 'invalid'.
    target_phase is the 'Px' string for hidden_todo, else None.
    reason is the trailing reason for hidden_*, else None.
    widget_keys is the list of widget keys parsed out of UI: bindings.
    """
    if binding.startswith("UI:"):
        keys = _RE_UI_KEYS.findall(binding)
        return ("ui", None, None, keys)
    m = _RE_HIDDEN_PERM.match(binding)
    if m:
        return ("hidden_permanent", None, m.group(1).strip(), [])
    m = _RE_HIDDEN_TODO.match(binding)
    if m:
        return ("hidden_todo", m.group(1), m.group(2).strip(), [])
    return ("invalid", None, None, [])
