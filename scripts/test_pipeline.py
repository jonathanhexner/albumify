"""Quick test script for the pipeline engine."""

from pathlib import Path

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import get_registry

import sim_bench.pipeline.steps.all_steps


# New 13-step pipeline from PIPELINE_DESIGN.md
NEW_PIPELINE = [
    "discover_images",
    "detect_faces",           # Early face detection
    "score_iqa",
    "score_ava",
    "score_face_pose",        # Only for images with significant faces
    "score_face_eyes",
    "score_face_smile",
    "filter_quality",
    "extract_scene_embedding",
    "cluster_scenes",
    "extract_face_embeddings",
    "cluster_by_identity",    # Sub-cluster by face identity + count
    "select_best"             # Smart selection with branching logic
]

# Minimal pipeline for quick testing (no face processing)
MINIMAL_PIPELINE = [
    "discover_images",
    "score_iqa",
    "filter_quality",
    "extract_scene_embedding",
    "cluster_scenes",
    "select_best"
]


def test_pipeline(source_dir: str, use_full_pipeline: bool = True):
    """Test the pipeline with a source directory."""

    print(f"\n{'='*60}")
    print("Testing Pipeline Engine")
    print(f"{'='*60}\n")

    registry = get_registry()
    print(f"Registered steps: {registry.list_step_names()}\n")

    context = PipelineContext(source_directory=Path(source_dir))

    def progress_callback(step: str, progress: float, message: str):
        bar_len = 20
        filled = int(bar_len * progress)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"  [{step:25}] [{bar}] {progress*100:5.1f}% {message}")

    # Get paths relative to script location
    script_dir = Path(__file__).parent.parent
    ava_checkpoint = script_dir / "models" / "album_app" / "ava_aesthetic_model.pt"
    siamese_checkpoint = script_dir / "models" / "album_app" / "siamese_comparison_model.pt"

    config = PipelineConfig(
        fail_fast=True,
        progress_callback=progress_callback,
        step_configs={
            "filter_quality": {"min_iqa_score": 0.2, "min_sharpness": 0.1},
            "score_ava": {
                "checkpoint_path": str(ava_checkpoint),
                "device": "cpu"
            },
            "select_best": {
                "max_images_per_cluster": 2,
                "min_score_threshold": 0.3,
                "max_score_gap": 0.3,
                "siamese_checkpoint": str(siamese_checkpoint),
                "tiebreaker_threshold": 0.05
            }
        }
    )

    executor = PipelineExecutor(registry)

    steps = NEW_PIPELINE if use_full_pipeline else MINIMAL_PIPELINE
    pipeline_name = "FULL (13-step)" if use_full_pipeline else "MINIMAL (6-step)"

    print(f"Running {pipeline_name} pipeline:\n")
    for i, step in enumerate(steps, 1):
        print(f"  {i:2}. {step}")
    print()

    result = executor.execute(context, steps, config)

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Success: {result.success}")
    if not result.success:
        print(f"Error: {result.error_message}")
    print(f"Total duration: {result.total_duration_ms:.0f}ms")
    print()

    print("Step timings:")
    for step_result in result.step_results:
        status = "OK" if step_result.success else "FAIL"
        print(f"  [{status:4}] {step_result.step_name:30} {step_result.duration_ms:6.0f}ms")

    print()
    print(f"Images found: {len(context.image_paths)}")
    print(f"Images after quality filter: {len(context.active_images)}")
    print(f"Scene clusters: {len(context.scene_clusters)}")

    if context.face_clusters:
        total_subclusters = sum(len(sc) for sc in context.face_clusters.values())
        print(f"Face subclusters: {total_subclusters}")

    print(f"Selected images: {len(context.selected_images)}")

    # Show faces detected
    if context.faces:
        total_faces = sum(len(f) for f in context.faces.values())
        images_with_faces = len([p for p, f in context.faces.items() if f])
        print(f"\nFace detection:")
        print(f"  Total faces found: {total_faces}")
        print(f"  Images with faces: {images_with_faces}")

    # Show cluster details
    if context.scene_clusters:
        print(f"\nScene clusters:")
        for cluster_id, images in sorted(context.scene_clusters.items()):
            print(f"  Cluster {cluster_id}: {len(images)} images")

    # Show face subclusters
    if context.face_clusters:
        print(f"\nFace subclusters:")
        for scene_id, subclusters in sorted(context.face_clusters.items()):
            for subcluster_id, info in subclusters.items():
                face_count = info.get('face_count', '?')
                has_faces = info.get('has_faces', False)
                num_images = len(info.get('images', []))
                face_str = f"faces={face_count}" if has_faces else "no faces"
                print(f"  Scene {scene_id}, Sub {subcluster_id}: {num_images} images ({face_str})")

    if context.selected_images:
        print(f"\nSelected images:")
        for img in context.selected_images[:10]:
            score_info = ""
            path = str(img)
            if path in context.iqa_scores:
                score_info += f" IQA={context.iqa_scores[path]:.2f}"
            if path in context.ava_scores:
                score_info += f" AVA={context.ava_scores[path]:.1f}"
            print(f"  - {Path(img).name}{score_info}")
        if len(context.selected_images) > 10:
            print(f"  ... and {len(context.selected_images) - 10} more")

    return result.success


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <source_directory> [--minimal]")
        print("Example: python test_pipeline.py ./samples/ukbench")
        print("         python test_pipeline.py ./samples/ukbench --minimal")
        sys.exit(1)

    source_dir = sys.argv[1]
    use_full = "--minimal" not in sys.argv

    if not Path(source_dir).exists():
        print(f"Error: Directory does not exist: {source_dir}")
        sys.exit(1)

    success = test_pipeline(source_dir, use_full_pipeline=use_full)
    sys.exit(0 if success else 1)
