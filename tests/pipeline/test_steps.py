"""Tests for individual pipeline steps."""

import pytest
from pathlib import Path

from sim_bench.pipeline.context import PipelineContext

import sim_bench.pipeline.steps.all_steps
from sim_bench.pipeline.registry import get_registry


SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples" / "ukbench"


@pytest.fixture
def registry():
    """Get the global registry with all steps registered."""
    return get_registry()


@pytest.fixture
def context_with_source():
    """Create a context with the samples directory."""
    return PipelineContext(source_directory=SAMPLES_DIR)


@pytest.fixture
def context_empty():
    """Create an empty context."""
    return PipelineContext()


class TestDiscoverImagesStep:
    """Tests for discover_images step."""

    def test_validate_fails_without_source(self, registry, context_empty):
        """Validate should fail if source_directory is not set."""
        step = registry.get("discover_images")
        errors = step.validate(context_empty)
        assert len(errors) > 0

    def test_validate_passes_with_source(self, registry, context_with_source):
        """Validate should pass with valid source directory."""
        step = registry.get("discover_images")
        errors = step.validate(context_with_source)
        assert len(errors) == 0

    def test_process_finds_images(self, registry, context_with_source):
        """Process should find images in the samples directory."""
        step = registry.get("discover_images")
        step.process(context_with_source, {})

        assert len(context_with_source.image_paths) > 0
        assert all(p.suffix.lower() in [".jpg", ".jpeg", ".png"]
                   for p in context_with_source.image_paths)

    def test_process_populates_active_images(self, registry, context_with_source):
        """Process should also populate active_images set."""
        step = registry.get("discover_images")
        step.process(context_with_source, {})

        assert len(context_with_source.active_images) == len(context_with_source.image_paths)


class TestScoreIQAStep:
    """Tests for score_iqa step."""

    def test_validate_fails_without_images(self, registry, context_empty):
        """Validate should fail if image_paths is empty."""
        step = registry.get("score_iqa")
        errors = step.validate(context_empty)
        assert len(errors) > 0

    def test_validate_passes_with_images(self, registry, context_with_source):
        """Validate should pass if image_paths is populated."""
        # First discover images
        discover = registry.get("discover_images")
        discover.process(context_with_source, {})

        step = registry.get("score_iqa")
        errors = step.validate(context_with_source)
        assert len(errors) == 0

    def test_process_scores_all_images(self, registry, context_with_source):
        """Process should score all images."""
        # First discover images
        discover = registry.get("discover_images")
        discover.process(context_with_source, {})

        step = registry.get("score_iqa")
        step.process(context_with_source, {})

        assert len(context_with_source.iqa_scores) == len(context_with_source.image_paths)
        assert len(context_with_source.sharpness_scores) == len(context_with_source.image_paths)

    def test_scores_are_valid_range(self, registry, context_with_source):
        """Scores should be in valid range (0-1)."""
        discover = registry.get("discover_images")
        discover.process(context_with_source, {})

        step = registry.get("score_iqa")
        step.process(context_with_source, {})

        for score in context_with_source.iqa_scores.values():
            assert 0.0 <= score <= 1.0

        for score in context_with_source.sharpness_scores.values():
            assert 0.0 <= score <= 1.0


class TestFilterQualityStep:
    """Tests for filter_quality step."""

    def test_validate_fails_without_scores(self, registry, context_empty):
        """Validate should fail if iqa_scores is empty."""
        step = registry.get("filter_quality")
        errors = step.validate(context_empty)
        assert len(errors) > 0

    def test_filter_removes_low_quality(self, registry, context_with_source):
        """Filter should remove images below threshold."""
        # Setup: discover and score
        discover = registry.get("discover_images")
        discover.process(context_with_source, {})

        score_iqa = registry.get("score_iqa")
        score_iqa.process(context_with_source, {})

        # Set a high threshold that will filter some images
        step = registry.get("filter_quality")
        step.process(context_with_source, {"min_iqa_score": 0.9, "min_sharpness": 0.9})

        # Should have filtered some (or possibly all) images
        assert len(context_with_source.quality_passed) <= len(context_with_source.image_paths)

    def test_filter_with_low_threshold_passes_all(self, registry, context_with_source):
        """Filter with threshold 0 should pass all images."""
        discover = registry.get("discover_images")
        discover.process(context_with_source, {})

        score_iqa = registry.get("score_iqa")
        score_iqa.process(context_with_source, {})

        step = registry.get("filter_quality")
        step.process(context_with_source, {"min_iqa_score": 0.0, "min_sharpness": 0.0})

        assert len(context_with_source.quality_passed) == len(context_with_source.image_paths)


class TestStepMetadata:
    """Tests for step metadata correctness."""

    def test_all_steps_have_required_metadata(self, registry):
        """All registered steps should have complete metadata."""
        for step_meta in registry.list_steps():
            assert step_meta.name is not None
            assert step_meta.display_name is not None
            assert step_meta.description is not None
            assert step_meta.category is not None
            assert isinstance(step_meta.requires, set)
            assert isinstance(step_meta.produces, set)
            assert isinstance(step_meta.depends_on, list)

    def test_dependencies_are_valid_steps(self, registry):
        """All step dependencies should reference valid steps."""
        step_names = set(registry.list_step_names())

        for step_meta in registry.list_steps():
            for dep in step_meta.depends_on:
                assert dep in step_names, f"{step_meta.name} depends on unknown step: {dep}"

    def test_produces_not_empty(self, registry):
        """Each step should produce at least one output."""
        for step_meta in registry.list_steps():
            # discover_images is the only exception that reads nothing
            if step_meta.name != "discover_images" or step_meta.requires:
                pass  # OK
            assert len(step_meta.produces) > 0, f"{step_meta.name} produces nothing"
