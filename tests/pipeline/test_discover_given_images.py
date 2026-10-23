"""discover_images uses images handed over by the caller (spec-112 TW1.3) and never scans for more."""
from pathlib import Path

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.discover_images import DiscoverImagesStep


class ut_DiscoverGivenImages:

    def test_given_images_are_used_as_is(self, tmp_path):
        a, b = tmp_path / "x" / "b.jpg", tmp_path / "y" / "a.jpg"
        (tmp_path / "stray.jpg").write_bytes(b"x")   # beside them, must not be picked up
        ctx = PipelineContext(image_paths=[a, b])
        step = DiscoverImagesStep()
        assert step.validate(ctx) == []
        step.process(ctx, {})
        assert ctx.image_paths == sorted([a, b])
        assert ctx.active_images == {str(a), str(b)}

    def test_without_images_or_folder_validation_fails(self):
        assert DiscoverImagesStep().validate(PipelineContext()) == ["source_directory is not set"]

    def test_a_folder_is_still_scanned(self, tmp_path):
        (tmp_path / "p.jpg").write_bytes(b"x")
        ctx = PipelineContext(source_directory=tmp_path)
        DiscoverImagesStep().process(ctx, {})
        assert [Path(p).name for p in ctx.image_paths] == ["p.jpg"]
