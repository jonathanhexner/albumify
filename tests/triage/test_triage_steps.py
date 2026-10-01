"""spec-107 P2 — triage pipeline steps (integration-lite + apply helper)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps._triage_apply import partition_and_record
from sim_bench.triage.metadata import TriageResult

SLOVAKIA = Path("D:/Slovakia26")


class ut_TriageApply:
    """The shared partition/record helper."""

    def _ctx(self, names):
        c = PipelineContext(source_directory=Path("."))
        c.image_paths = [Path(n) for n in names]
        c.active_images = {n for n in names}
        return c

    def test_set_aside_removed_review_kept_keep_passes(self):
        c = self._ctx(["a.jpg", "b.jpg", "c.jpg"])
        decisions = {
            "a.jpg": TriageResult("set_aside", "screenshot", 0.98, []),
            "b.jpg": TriageResult("review", "no exif", 0.4, []),
            "c.jpg": TriageResult("keep", "camera", 0.0, []),
        }
        n = partition_and_record(c, decisions, tier="metadata")
        assert n == 1
        survivors = {p.name for p in c.image_paths}
        assert survivors == {"b.jpg", "c.jpg"}            # set_aside removed; review kept
        assert len(c.set_aside) == 1 and c.set_aside[0]["image"] == "a.jpg"
        assert c.set_aside[0]["tier"] == "metadata"
        assert len(c.triage_review) == 1 and c.triage_review[0]["image"] == "b.jpg"
        assert c.active_images == {"b.jpg", "c.jpg"}

    def test_missing_decision_passes_through(self):
        c = self._ctx(["x.jpg"])
        n = partition_and_record(c, {}, tier="metadata")
        assert n == 0 and len(c.image_paths) == 1


@pytest.mark.needs_data
class ut_TriageMetadataStepOnRealRoll:
    """Run the metadata step on the real Slovakia roll (free, no model)."""

    def test_screenshots_set_aside_no_false_positives(self):
        if not SLOVAKIA.exists():
            pytest.skip("Slovakia26 data not present")
        from sim_bench.pipeline.steps.discover_images import DiscoverImagesStep
        from sim_bench.pipeline.steps.triage_metadata import TriageMetadataStep

        ctx = PipelineContext(source_directory=SLOVAKIA)
        DiscoverImagesStep().process(ctx, {"extensions": [".jpg"]})
        total = len(ctx.image_paths)
        TriageMetadataStep().process(ctx, {})

        set_aside_names = {Path(e["image"]).name for e in ctx.set_aside}
        assert total == 715
        assert len(set_aside_names) == 8                          # all 8 screenshots
        assert all(n.lower().startswith("screenshot") for n in set_aside_names)
        assert len(ctx.image_paths) == total - 8                 # survivors = rest
