"""Integration tests - full pipeline end-to-end."""

import pytest
from pathlib import Path

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import get_registry
from sim_bench.pipeline.clustering_labels import NOISE_LABEL, is_noise

import sim_bench.pipeline.steps.all_steps


SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples" / "ukbench"


@pytest.fixture
def executor():
    """Create an executor with all steps registered."""
    return PipelineExecutor(get_registry())


@pytest.fixture
def context():
    """Create a context with the samples directory."""
    return PipelineContext(source_directory=SAMPLES_DIR)


class TestFullPipeline:
    """Tests for the complete pipeline execution."""

    def test_default_pipeline_succeeds(self, executor, context):
        """The default pipeline should complete successfully."""
        steps = [
            "discover_images",
            "score_iqa",
            "filter_quality",
            "extract_scene_embedding",
            "cluster_scenes",
            "select_best"
        ]

        config = PipelineConfig(
            fail_fast=True,
            step_configs={
                "filter_quality": {"min_iqa_score": 0.1, "min_sharpness": 0.1}
            }
        )

        result = executor.execute(context, steps, config)

        assert result.success, f"Pipeline failed: {result.error_message}"
        assert len(result.step_results) == 6
        assert all(r.success for r in result.step_results)

    def test_pipeline_produces_expected_outputs(self, executor, context):
        """Pipeline should produce all expected context outputs."""
        steps = [
            "discover_images",
            "score_iqa",
            "filter_quality",
            "extract_scene_embedding",
            "cluster_scenes",
            "select_best"
        ]

        config = PipelineConfig(
            step_configs={
                "filter_quality": {"min_iqa_score": 0.0, "min_sharpness": 0.0}
            }
        )

        executor.execute(context, steps, config)

        # Check all expected outputs are populated
        assert len(context.image_paths) > 0, "No images discovered"
        assert len(context.iqa_scores) > 0, "No IQA scores"
        assert len(context.sharpness_scores) > 0, "No sharpness scores"
        assert len(context.active_images) > 0, "No active images after filter"
        assert len(context.scene_embeddings) > 0, "No scene embeddings"
        assert len(context.scene_clusters) > 0, "No clusters created"
        assert len(context.selected_images) > 0, "No images selected"

    def test_selected_images_exist(self, executor, context):
        """Selected images should be valid paths that exist."""
        steps = [
            "discover_images",
            "score_iqa",
            "filter_quality",
            "extract_scene_embedding",
            "cluster_scenes",
            "select_best"
        ]

        config = PipelineConfig(
            step_configs={
                "filter_quality": {"min_iqa_score": 0.0, "min_sharpness": 0.0}
            }
        )

        executor.execute(context, steps, config)

        for image_path in context.selected_images:
            assert Path(image_path).exists(), f"Selected image doesn't exist: {image_path}"

    def test_selected_from_each_cluster(self, executor, context):
        """select_best must cover every real scene cluster.

        Coverage contract — each non-noise cluster contributes at least one
        image to context.selected_images. We don't pin max_images_per_cluster
        on purpose: the selector is free to pick 1 or N per cluster; the only
        thing we care about here is that no real cluster is dropped.

        Two configs ARE pinned to lock the contract under test:
        - cluster_scenes.min_cluster_size = 2 (the producer's definition of
          "what counts as a cluster" must not drift under us)
        - select_best.include_noise = False (the NOISE_LABEL bucket is not a
          cluster and is not expected to contribute selections)
        """
        steps = [
            "discover_images",
            "score_iqa",
            "filter_quality",
            "extract_scene_embedding",
            "cluster_scenes",
            "select_best"
        ]

        config = PipelineConfig(
            step_configs={
                "filter_quality": {"min_iqa_score": 0.0, "min_sharpness": 0.0},
                "cluster_scenes": {"min_cluster_size": 2},
                "select_best": {"include_noise": False},
            }
        )

        executor.execute(context, steps, config)

        real_cluster_ids = {cid for cid in context.scene_clusters if not is_noise(cid)}
        assert real_cluster_ids, (
            "Test precondition: cluster_scenes produced no real clusters on the "
            "ukbench fixture — the coverage assertion would be vacuous."
        )

        selected_cluster_ids = {
            context.scene_cluster_labels[p] for p in context.selected_images
        }
        missing = real_cluster_ids - selected_cluster_ids
        assert not missing, (
            f"select_best dropped clusters: {sorted(missing)}. "
            f"real clusters={sorted(real_cluster_ids)}, "
            f"covered={sorted(selected_cluster_ids - {NOISE_LABEL})}"
        )

    def test_progress_callback_called(self, executor, context):
        """Progress callback should be called during execution."""
        progress_updates = []

        def on_progress(step: str, progress: float, message: str):
            progress_updates.append((step, progress, message))

        config = PipelineConfig(
            progress_callback=on_progress,
            step_configs={
                "filter_quality": {"min_iqa_score": 0.0, "min_sharpness": 0.0}
            }
        )

        steps = ["discover_images", "score_iqa", "filter_quality"]
        executor.execute(context, steps, config)

        assert len(progress_updates) > 0, "Progress callback was never called"

    def test_step_timings_recorded(self, executor, context):
        """Each step should have timing information."""
        steps = ["discover_images", "score_iqa"]
        result = executor.execute(context, steps, PipelineConfig())

        for step_result in result.step_results:
            assert step_result.duration_ms >= 0

        assert result.total_duration_ms >= 0


class TestPipelineErrorHandling:
    """Tests for error handling."""

    def test_fail_fast_stops_on_error(self, executor):
        """With fail_fast=True, pipeline should stop on first error."""
        # Create context with non-existent directory
        context = PipelineContext(source_directory=Path("/nonexistent/path"))

        config = PipelineConfig(fail_fast=True)
        result = executor.execute(context, ["discover_images"], config)

        assert not result.success
        assert result.error_message is not None

    def test_missing_dependency_validation_fails(self, executor, context):
        """Step validation flags a None-valued required key.

        PipelineContext defaults image_paths to [] (not None), so to exercise
        the "missing required key" path we have to null it explicitly. Empty
        collections are no longer flagged — that conflated "producer never
        ran" with "producer ran and emitted nothing".
        """
        context_empty = PipelineContext(source_directory=SAMPLES_DIR)
        context_empty.image_paths = None  # type: ignore[assignment]

        registry = get_registry()
        step = registry.get("score_iqa")
        errors = step.validate(context_empty)

        assert len(errors) > 0
        assert "image_paths" in errors[0]


class TestAutoResolve:
    """Tests for automatic dependency resolution in executor."""

    def test_auto_resolves_dependencies(self, executor, context):
        """Executor should auto-resolve missing dependencies."""
        # Only request the last step, dependencies should be auto-added
        steps = ["select_best"]

        config = PipelineConfig(
            step_configs={
                "filter_quality": {"min_iqa_score": 0.0, "min_sharpness": 0.0}
            }
        )

        result = executor.execute(context, steps, config)

        # Should have executed all dependent steps
        assert result.success
        executed_steps = [r.step_name for r in result.step_results]
        assert "discover_images" in executed_steps
        assert "score_iqa" in executed_steps
        assert "filter_quality" in executed_steps
        assert "extract_scene_embedding" in executed_steps
        assert "cluster_scenes" in executed_steps
        assert "select_best" in executed_steps
