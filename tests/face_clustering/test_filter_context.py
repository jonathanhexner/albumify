"""Tests for spec-032 P0: face_cluster/filter_context.py primitive.

Covers:
  - register: idempotent on item_id
  - record: replace-on-duplicate, unknown filter_name raises, auto-register
  - active: parent inheritance, after= cutoff, item_type filtering
  - is_active: same semantics, unknown id returns False
  - summary: rejection counts grouped by filter and item type
  - parse_ui_binding: UI:/HIDDEN(permanent):/HIDDEN(todo,...) recognition

All tests use the public API; internal data structures (_items) are
inspected only to assert idempotency.
"""
from __future__ import annotations

import pytest

from face_cluster.filter_context import (
    KNOWN_FILTERS,
    FilterContext,
    FilterDecision,
    ItemState,
    filter_position,
    parse_ui_binding,
)


# -----------------------------------------------------------------------------
# Registry sanity
# -----------------------------------------------------------------------------
class ut_KnownFiltersRegistry:
    def test_no_duplicate_filter_names(self):
        names = [n for n, *_ in KNOWN_FILTERS]
        assert len(names) == len(set(names)), f"duplicate names: {names}"

    def test_every_filter_has_required_columns(self):
        for entry in KNOWN_FILTERS:
            assert len(entry) == 5, f"malformed: {entry}"
            name, item_type, file, desc, ui = entry
            assert isinstance(name, str) and name
            assert item_type in ("image", "face", "cluster")
            assert isinstance(file, str) and file
            assert isinstance(desc, str) and desc
            assert isinstance(ui, str) and ui

    def test_filter_position_matches_list_index(self):
        for i, (name, *_) in enumerate(KNOWN_FILTERS):
            assert filter_position(name) == i

    def test_ui_binding_parses_for_every_entry(self):
        # Catches malformed entries in the registry itself.
        for name, _t, _f, _d, ui in KNOWN_FILTERS:
            kind, target, reason, keys = parse_ui_binding(ui)
            assert kind in ("ui", "hidden_permanent", "hidden_todo"), \
                f"{name}: invalid binding {ui!r} (kind={kind})"
            if kind == "ui":
                assert keys, f"{name}: UI binding has no widget keys: {ui!r}"
            else:
                assert reason, f"{name}: {kind} without reason"
            if kind == "hidden_todo":
                assert target and target.startswith("P") and target[1:].isdigit(), \
                    f"{name}: bad target phase {target!r}"


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------
class ut_Register:
    def test_register_creates_item(self):
        fc = FilterContext()
        fc.register("img.jpg", "image")
        assert "img.jpg" in fc
        assert fc.get("img.jpg").item_type == "image"
        assert fc.get("img.jpg").parent_id is None

    def test_register_is_idempotent(self):
        fc = FilterContext()
        fc.register("img.jpg", "image")
        fc.register("img.jpg", "image")
        assert len(fc) == 1

    def test_register_fills_in_late_parent_id(self):
        # Auto-registration via record() (no parent_id), then explicit
        # register() with parent_id — should fill the gap.
        fc = FilterContext()
        fc.register("face_0", "face")           # no parent yet
        fc.register("face_0", "face", parent_id="img.jpg")
        assert fc.get("face_0").parent_id == "img.jpg"

    def test_register_preserves_existing_parent_id(self):
        fc = FilterContext()
        fc.register("face_0", "face", parent_id="img.jpg")
        fc.register("face_0", "face", parent_id="other.jpg")  # ignored
        assert fc.get("face_0").parent_id == "img.jpg"


# -----------------------------------------------------------------------------
# Recording
# -----------------------------------------------------------------------------
class ut_Record:
    def test_record_unknown_filter_raises(self):
        fc = FilterContext()
        with pytest.raises(ValueError, match="Unknown filter_name"):
            fc.record("img.jpg", filter_name="bogus_filter",
                      rejected=False, reason="?")

    def test_record_auto_registers_if_needed(self):
        fc = FilterContext()
        # No prior register() call.
        fc.record("img.jpg", filter_name="image_quality",
                  rejected=True, reason="IQA low",
                  measured={"iqa": 0.05})
        assert "img.jpg" in fc
        # Item type was inferred from the filter's canonical type.
        assert fc.get("img.jpg").item_type == "image"

    def test_record_replaces_prior_decision_on_same_pair(self):
        fc = FilterContext()
        fc.record("img.jpg", filter_name="image_quality",
                  rejected=False, reason="first", measured={"iqa": 0.5})
        fc.record("img.jpg", filter_name="image_quality",
                  rejected=True, reason="second",
                  measured={"iqa": 0.05})
        decisions = fc.get("img.jpg").decisions
        assert len(decisions) == 1
        assert decisions[0].reason == "second"
        assert decisions[0].rejected is True

    def test_record_appends_distinct_filters(self):
        fc = FilterContext()
        fc.record("face_0", filter_name="face_blur",
                  rejected=False, reason="ok", measured={})
        fc.record("face_0", filter_name="face_pose_yaw",
                  rejected=True, reason="yaw too high",
                  measured={"yaw": 45.0})
        names = [d.filter_name for d in fc.get("face_0").decisions]
        assert names == ["face_blur", "face_pose_yaw"]

    def test_recorded_measured_is_a_copy(self):
        fc = FilterContext()
        original = {"iqa": 0.5}
        fc.record("img.jpg", filter_name="image_quality",
                  rejected=False, reason="ok", measured=original)
        original["iqa"] = 99.0  # caller-side mutation must not leak in
        assert fc.get("img.jpg").decisions[0].measured["iqa"] == 0.5


# -----------------------------------------------------------------------------
# Query: active() and is_active()
# -----------------------------------------------------------------------------
class ut_Active:
    def test_active_yields_only_non_rejected(self):
        fc = FilterContext()
        fc.record("img1.jpg", filter_name="image_quality",
                  rejected=False, reason="ok", measured={})
        fc.record("img2.jpg", filter_name="image_quality",
                  rejected=True, reason="bad", measured={})
        active = {it.item_id for it in fc.active("image")}
        assert active == {"img1.jpg"}

    def test_active_filters_by_item_type(self):
        fc = FilterContext()
        fc.record("img.jpg", filter_name="image_quality",
                  rejected=False, reason="ok", measured={})
        fc.record("face_0", filter_name="face_blur",
                  rejected=False, reason="ok", measured={},
                  parent_id="img.jpg")
        assert [it.item_id for it in fc.active("image")] == ["img.jpg"]
        assert [it.item_id for it in fc.active("face")]  == ["face_0"]

    def test_face_inactive_when_parent_image_rejected(self):
        # Parent inheritance: if image is rejected, face is too.
        fc = FilterContext()
        fc.record("img.jpg", filter_name="image_quality",
                  rejected=True, reason="bad", measured={})
        fc.register("face_0", "face", parent_id="img.jpg")
        fc.record("face_0", filter_name="face_blur",
                  rejected=False, reason="ok", measured={})
        active_faces = list(fc.active("face"))
        assert active_faces == []

    def test_face_active_when_parent_and_self_pass(self):
        fc = FilterContext()
        fc.record("img.jpg", filter_name="image_quality",
                  rejected=False, reason="ok", measured={})
        fc.register("face_0", "face", parent_id="img.jpg")
        fc.record("face_0", filter_name="face_blur",
                  rejected=False, reason="ok", measured={})
        assert [it.item_id for it in fc.active("face")] == ["face_0"]

    def test_active_after_excludes_later_decisions(self):
        # Image fails face_pose_yaw — but yaw is recorded AFTER image_quality.
        # active(after="image_quality") must IGNORE the yaw rejection.
        fc = FilterContext()
        fc.record("face_0", filter_name="face_blur",
                  rejected=False, reason="ok", measured={})
        fc.record("face_0", filter_name="face_pose_yaw",
                  rejected=True, reason="bad", measured={})
        snapshot_after_blur = list(fc.active("face", after="face_blur"))
        assert [it.item_id for it in snapshot_after_blur] == ["face_0"]
        final = list(fc.active("face"))
        assert final == []

    def test_is_active_returns_false_for_unknown_id(self):
        fc = FilterContext()
        assert fc.is_active("never_registered") is False

    def test_is_active_respects_parent_inheritance(self):
        fc = FilterContext()
        fc.record("img.jpg", filter_name="image_quality",
                  rejected=True, reason="bad", measured={})
        fc.register("face_0", "face", parent_id="img.jpg")
        assert fc.is_active("face_0") is False


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
class ut_Summary:
    def test_summary_counts_only_rejections(self):
        fc = FilterContext()
        fc.record("img1", filter_name="image_quality",
                  rejected=False, reason="ok", measured={})
        fc.record("img2", filter_name="image_quality",
                  rejected=True, reason="bad", measured={})
        fc.record("img3", filter_name="image_quality",
                  rejected=True, reason="bad", measured={})
        assert fc.summary() == {"image_quality": {"image": 2}}

    def test_summary_buckets_by_item_type(self):
        fc = FilterContext()
        fc.record("img1", filter_name="image_quality",
                  rejected=True, reason="bad", measured={})
        fc.record("face_0", filter_name="face_blur",
                  rejected=True, reason="blur", measured={})
        fc.record("face_1", filter_name="face_blur",
                  rejected=True, reason="blur", measured={})
        summary = fc.summary()
        assert summary["image_quality"]["image"] == 1
        assert summary["face_blur"]["face"]      == 2

    def test_summary_empty_when_no_rejections(self):
        fc = FilterContext()
        fc.record("img1", filter_name="image_quality",
                  rejected=False, reason="ok", measured={})
        assert fc.summary() == {}


# -----------------------------------------------------------------------------
# Parse ui_binding
# -----------------------------------------------------------------------------
class ut_ParseUiBinding:
    def test_ui_binding_with_one_key(self):
        kind, target, reason, keys = parse_ui_binding(
            "UI: 'Min Face Size' (config_min_face_size)"
        )
        assert kind == "ui"
        assert target is None
        assert keys == ["config_min_face_size"]

    def test_ui_binding_with_multiple_keys(self):
        kind, _, _, keys = parse_ui_binding(
            "UI: sliders (config_min_iqa) + (config_min_sharpness)"
        )
        assert kind == "ui"
        assert keys == ["config_min_iqa", "config_min_sharpness"]

    def test_hidden_permanent(self):
        kind, target, reason, keys = parse_ui_binding(
            "HIDDEN(permanent): binary outcome"
        )
        assert kind == "hidden_permanent"
        assert target is None
        assert reason == "binary outcome"
        assert keys == []

    def test_hidden_todo_with_target(self):
        kind, target, reason, _ = parse_ui_binding(
            "HIDDEN(todo, target=P3): expose as slider"
        )
        assert kind == "hidden_todo"
        assert target == "P3"
        assert reason == "expose as slider"

    def test_invalid_binding(self):
        kind, *_ = parse_ui_binding("just some random text")
        assert kind == "invalid"


# -----------------------------------------------------------------------------
# Integration: filters field is wired into both pipeline contexts
# -----------------------------------------------------------------------------
class ut_PipelineContextsCarryFilters:
    def test_albumify_PipelineContext_has_filters_field(self):
        from sim_bench.pipeline.context import PipelineContext
        ctx = PipelineContext()
        assert isinstance(ctx.filters, FilterContext)
        assert len(ctx.filters) == 0

    def test_fc_app_RunContext_has_filters_field(self):
        # _RunContext is internal but its field signature is part of the
        # contract spec-032 establishes.
        from face_cluster.pipeline import _RunContext
        import inspect
        sig = inspect.signature(_RunContext)
        # _RunContext is a dataclass; fields() reveals filters.
        from dataclasses import fields
        field_names = {f.name for f in fields(_RunContext)}
        assert "filters" in field_names

    def test_both_contexts_use_same_FilterContext_class(self):
        from sim_bench.pipeline.context import PipelineContext
        from face_cluster.pipeline import _RunContext
        from dataclasses import fields
        rc_filter_field = next(f for f in fields(_RunContext)
                               if f.name == "filters")
        # default_factory should be FilterContext (the same class).
        assert rc_filter_field.default_factory is FilterContext
        pc = PipelineContext()
        assert type(pc.filters) is FilterContext
