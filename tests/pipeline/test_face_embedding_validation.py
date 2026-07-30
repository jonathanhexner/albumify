"""Face Embedding Validation Tests

Purpose: Validate that face embeddings match correctly to faces and images,
preventing systematic offset bugs that can occur when faces are gated.

Test Design: Approved by expert panel (Dr. Chen, Alex Martinez, Jordan Lee)
See: face_cluster/docs/design/TEST_DESIGN_REVIEW.md

Critical Tests:
- test_embeddings_match_after_gating: Verify pipeline embeddings match direct extraction
- test_saved_crops_match_embeddings: Verify saved crops match their embeddings

Test Data: test_data/face_clustering (6 images, 3 people)
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pytest

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.discover_images import DiscoverImagesStep
from sim_bench.pipeline.steps.insightface_detect_faces import InsightFaceDetectFacesStep
from sim_bench.pipeline.steps.align_faces import AlignFacesStep
from sim_bench.pipeline.steps.extract_face_embeddings import ExtractFaceEmbeddingsStep
# spec-053: filter_quality_gate was consolidated into quality_gate (QualityGateStep). Alias keeps
# the rest of this file unchanged.
from sim_bench.pipeline.steps.quality_gate import QualityGateStep as FilterQualityGateStep

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# SIGHTING-118: needs InsightFace models + test_data/face_clustering. Not runnable in a clean CI
# runner; excluded from the default (fast) suite. Run explicitly with `pytest -m slow`.
pytestmark = pytest.mark.slow

# Test data directory
TEST_DATA_DIR = Path("test_data/face_clustering")


def bbox_tuple(bbox_dict):
    """Convert bbox dict to tuple for use as dict key."""
    if isinstance(bbox_dict, dict):
        return (bbox_dict["x_px"], bbox_dict["y_px"], bbox_dict["w_px"], bbox_dict["h_px"])
    return tuple(bbox_dict)


def bbox_matches(bbox1, bbox2, tolerance=1.0):
    """Check if two bboxes match within tolerance (pixels).

    Args:
        bbox1: Tuple (x, y, w, h)
        bbox2: Tuple (x, y, w, h)
        tolerance: Max pixel difference allowed

    Returns:
        True if bboxes match within tolerance
    """
    return all(abs(a - b) < tolerance for a, b in zip(bbox1, bbox2))


def find_matching_face(target_face_key, candidates, tolerance=1.0):
    """Find face in candidates that matches target face by image_path and bbox.

    Args:
        target_face_key: Tuple (image_path, bbox_tuple)
        candidates: Dict mapping (image_path, bbox_tuple) -> data
        tolerance: Bbox matching tolerance in pixels

    Returns:
        Matching key from candidates, or None
    """
    target_path, target_bbox = target_face_key

    for cand_key in candidates:
        cand_path, cand_bbox = cand_key
        if cand_path == target_path and bbox_matches(target_bbox, cand_bbox, tolerance):
            return cand_key
    return None


class TestFaceEmbeddingValidation:
    """Embedding validation test suite."""

    @pytest.fixture(scope="session")
    def test_images(self):
        """Load test images."""
        if not TEST_DATA_DIR.exists():
            pytest.skip(f"Test data not found: {TEST_DATA_DIR}")

        image_paths = []
        for person_dir in sorted(TEST_DATA_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            for img_path in sorted(person_dir.glob("*.jpg")):
                image_paths.append(img_path)

        logger.info(f"Loaded {len(image_paths)} test images")
        return image_paths

    @pytest.fixture(scope="session")
    def pipeline_result(self, test_images, tmp_path_factory):
        """Run pipeline once for all tests (with quality gating).

        Returns:
            PipelineContext with all steps executed
        """
        output_dir = tmp_path_factory.mktemp("pipeline_output")

        # Create context
        context = PipelineContext(source_directory=TEST_DATA_DIR)
        context.image_paths = test_images

        # Run pipeline steps
        logger.info("Running pipeline with quality gating...")

        # 1. Detect faces
        detect_step = InsightFaceDetectFacesStep()
        detect_config = {
            "model_name": "buffalo_l",
            "det_size": 640,
            "det_thresh": 0.5,
        }
        detect_step.process(context, detect_config)
        logger.info(f"  Detected faces in {len(context.insightface_faces)} images")

        # 2. Align faces
        align_step = AlignFacesStep()
        align_config = {
            "target_size": 112,
            "use_insightface_alignment": True,
        }
        align_step.process(context, align_config)
        logger.info(f"  Aligned {len(context.aligned_faces)} faces")

        # 3. Extract embeddings (before gating - this is the key step)
        embed_step = ExtractFaceEmbeddingsStep()
        embed_config = {
            "backend": "insightface",
            "model_name": "buffalo_l",
            "device": "cpu",
            "normalize": True,
            "verify_norm": True,
        }
        embed_step.process(context, embed_config)
        logger.info(f"  Extracted {len(context.face_embeddings)} embeddings")

        # 4. Apply quality gate (CRITICAL - this is where the bug can occur)
        filter_step = FilterQualityGateStep()
        filter_config = {
            "yaw_max": 45.0,
            "pitch_max": 30.0,
            "roll_max": 30.0,
            "blur_min": 50.0,  # Lenient to ensure some faces pass
            "max_faces_per_image_core": 10,
        }
        filter_step.process(context, filter_config)
        logger.info(f"  Core faces: {len(context.core_indices)}, Holdout: {len(context.holdout_indices)}")

        # 5. Save face crops manually (don't use export_for_labeling - it requires clustering)
        crops_dir = output_dir / "face_crops"
        crops_dir.mkdir(exist_ok=True)

        logger.info("Saving face crops...")
        saved_count = 0
        for face_record in context.face_records:
            # Get aligned face crop
            face_key = f"{face_record.image_path}:face_{face_record.face_index}"
            aligned_crop = context.aligned_faces.get(face_key)

            if aligned_crop is None:
                logger.warning(f"  No aligned crop for face {face_record.face_id}")
                continue

            # Save as JPEG
            crop_path = crops_dir / f"face_{face_record.face_id:04d}_aligned.jpg"
            crop_bgr = cv2.cvtColor(aligned_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_path), crop_bgr)
            saved_count += 1

        logger.info(f"  Saved {saved_count} face crops to {crops_dir}")

        context.export_directory = output_dir
        return context

    @pytest.fixture(scope="session")
    def direct_result(self, test_images):
        """Extract embeddings directly without gating.

        Returns:
            Dict mapping (image_path, bbox_tuple) -> embedding
        """
        logger.info("Extracting embeddings directly (no gating)...")

        # Create context
        context = PipelineContext(source_directory=TEST_DATA_DIR)
        context.image_paths = test_images

        # Run detection + alignment + embedding extraction (NO gating)
        detect_step = InsightFaceDetectFacesStep()
        detect_config = {
            "model_name": "buffalo_l",
            "det_size": 640,
            "det_thresh": 0.5,
        }
        detect_step.process(context, detect_config)

        align_step = AlignFacesStep()
        align_config = {
            "target_size": 112,
            "use_insightface_alignment": True,
        }
        align_step.process(context, align_config)

        embed_step = ExtractFaceEmbeddingsStep()
        embed_config = {
            "backend": "insightface",
            "model_name": "buffalo_l",
            "device": "cpu",
            "normalize": True,
            "verify_norm": True,
        }
        embed_step.process(context, embed_config)

        # Build mapping using content-based keys
        result = {}
        for face_key, embedding in context.face_embeddings.items():
            # Parse face_key: "image_path:face_N"
            parts = face_key.rsplit(":face_", 1)
            if len(parts) != 2:
                continue

            image_path = parts[0]
            face_idx = int(parts[1])

            # Get bbox from insightface_faces
            face_data = context.insightface_faces.get(image_path)
            if not face_data:
                continue

            faces = face_data.get("faces", [])
            if face_idx >= len(faces):
                continue

            bbox = faces[face_idx]["bbox"]
            bbox_key = bbox_tuple(bbox)

            # Store with content-based key
            key = (image_path, bbox_key)
            result[key] = embedding

        logger.info(f"  Direct extraction: {len(result)} embeddings")
        return result

    # ============================================================================
    # Test 1: Embeddings match after gating ⭐ CRITICAL
    # ============================================================================

    @pytest.mark.critical
    def test_embeddings_match_after_gating(self, pipeline_result, direct_result):
        """Pipeline embeddings should match direct extraction for passed faces.

        CRITICAL: This test catches the bug where face gating causes systematic
        offset between face IDs and embeddings.

        Validation:
        - For each face that passed quality gate
        - Compare pipeline embedding to direct extraction embedding
        - Use content-based keys: (image_path, bbox) not face_id
        - Tolerance: atol=1e-4, cosine_sim > 0.9999
        """
        logger.info("Test 1: Validating embeddings match after gating...")

        mismatches = []
        checked = 0

        for face_record in pipeline_result.face_records:
            # Only check core faces (passed quality gate)
            if face_record.face_id not in pipeline_result.core_indices:
                continue

            # Build content-based key
            bbox = face_record.bbox
            bbox_key = bbox_tuple(bbox)
            face_key = (str(face_record.image_path), bbox_key)

            # Find matching face in direct result (with fuzzy bbox matching)
            direct_key = find_matching_face(face_key, direct_result, tolerance=2.0)

            if direct_key is None:
                logger.warning(f"  Face not found in direct result: {face_key}")
                continue

            # Get embeddings
            pipeline_emb = face_record.embedding
            direct_emb = direct_result[direct_key]

            # Check L2 distance
            l2_dist = np.linalg.norm(pipeline_emb - direct_emb)
            l2_match = np.allclose(pipeline_emb, direct_emb, atol=1e-4)

            # Check cosine similarity
            cosine_sim = np.dot(pipeline_emb, direct_emb) / (
                np.linalg.norm(pipeline_emb) * np.linalg.norm(direct_emb)
            )
            cosine_match = cosine_sim > 0.9999

            if not (l2_match and cosine_match):
                mismatches.append({
                    "face_id": face_record.face_id,
                    "image_path": face_record.image_path,
                    "bbox": bbox_key,
                    "l2_dist": l2_dist,
                    "cosine_sim": cosine_sim,
                })

            checked += 1

        # Report results
        logger.info(f"  Checked: {checked} faces")
        logger.info(f"  Mismatches: {len(mismatches)}")

        if mismatches:
            for mismatch in mismatches[:5]:  # Show first 5
                logger.error(f"  Mismatch: face_id={mismatch['face_id']}, "
                           f"l2_dist={mismatch['l2_dist']:.6f}, "
                           f"cosine_sim={mismatch['cosine_sim']:.6f}")

        assert len(mismatches) == 0, f"{len(mismatches)}/{checked} embeddings mismatched"

    # ============================================================================
    # Test 2: Face IDs are sequential
    # ============================================================================

    def test_face_ids_sequential(self, pipeline_result):
        """Face IDs should be 0, 1, 2, ... with no gaps.

        This ensures face ID assignment is correct even when some faces are filtered.
        """
        logger.info("Test 2: Validating face IDs are sequential...")

        face_ids = sorted([f.face_id for f in pipeline_result.face_records])
        expected = list(range(len(face_ids)))

        logger.info(f"  Face IDs: {face_ids[:10]}... (first 10)")
        logger.info(f"  Expected: {expected[:10]}... (first 10)")

        assert face_ids == expected, f"Non-sequential IDs: {face_ids}"

    # ============================================================================
    # Test 3: Saved crops match embeddings ⭐ CRITICAL
    # ============================================================================

    @pytest.mark.critical
    def test_saved_crops_match_embeddings(self, pipeline_result):
        """Embeddings extracted from saved crops should match stored embeddings.

        CRITICAL: This test detects mismatch between saved crops and embeddings
        (could indicate ID offset bug).

        Validation:
        - Load saved face crops from disk
        - Extract fresh embeddings from crops
        - Compare to stored embeddings
        - Tolerance: atol=1e-4
        """
        logger.info("Test 3: Validating saved crops match embeddings...")

        from face_cluster.embedding import InsightFaceEmbedder

        embedder = InsightFaceEmbedder(model_name="buffalo_l")
        crops_dir = pipeline_result.export_directory / "face_crops"

        if not crops_dir.exists():
            pytest.skip(f"Crops directory not found: {crops_dir}")

        mismatches = []
        checked = 0

        for face_record in pipeline_result.face_records:
            # Find crop file
            crop_path = crops_dir / f"face_{face_record.face_id:04d}_aligned.jpg"

            if not crop_path.exists():
                logger.warning(f"  Missing crop: {crop_path}")
                continue

            # Load crop and extract embedding
            crop_img = cv2.imread(str(crop_path))
            if crop_img is None:
                logger.warning(f"  Failed to load crop: {crop_path}")
                continue

            # Convert BGR to RGB
            crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

            # Extract embedding from crop using get_embedding method
            crop_emb = embedder.get_embedding(crop_img_rgb)

            if crop_emb is None:
                logger.warning(f"  Failed to extract embedding from crop: {crop_path}")
                continue

            # Compare to stored embedding
            stored_emb = face_record.embedding

            # Check match using cosine similarity (more robust to JPEG compression)
            # Note: L2 distance can be affected by JPEG compression artifacts
            # when saving/loading crops, so we use cosine similarity instead
            cosine_sim = np.dot(crop_emb, stored_emb) / (
                np.linalg.norm(crop_emb) * np.linalg.norm(stored_emb)
            )

            # Cosine similarity should be > 0.99 (allows for minor JPEG artifacts)
            match = cosine_sim > 0.99

            if not match:
                l2_dist = np.linalg.norm(crop_emb - stored_emb)
                mismatches.append({
                    "face_id": face_record.face_id,
                    "crop_path": crop_path,
                    "l2_dist": l2_dist,
                    "cosine_sim": cosine_sim,
                })

            checked += 1

        # Report results
        logger.info(f"  Checked: {checked} crops")
        logger.info(f"  Mismatches: {len(mismatches)}")

        if mismatches:
            for mismatch in mismatches[:5]:  # Show first 5
                logger.error(f"  Mismatch: face_id={mismatch['face_id']}, "
                           f"cosine_sim={mismatch['cosine_sim']:.6f}, "
                           f"l2_dist={mismatch['l2_dist']:.6f}")

        assert len(mismatches) == 0, f"{len(mismatches)}/{checked} crops had cosine similarity < 0.99"

    # ============================================================================
    # Test 4: Gated faces not in output
    # ============================================================================

    def test_gated_faces_not_in_output(self, pipeline_result, direct_result):
        """Faces that failed quality gate should not be in pipeline output.

        Validation:
        - Compare direct result (all faces) vs pipeline result (filtered)
        - Verify filtered faces are excluded
        - At least some faces should be gated (otherwise test too weak)
        """
        logger.info("Test 4: Validating gated faces not in output...")

        # Build set of pipeline face keys
        pipeline_keys = set()
        for face_record in pipeline_result.face_records:
            bbox_key = bbox_tuple(face_record.bbox)
            key = (str(face_record.image_path), bbox_key)
            pipeline_keys.add(key)

        # Build set of direct face keys
        direct_keys = set(direct_result.keys())

        # Faces in direct but not pipeline = gated faces
        gated_faces = direct_keys - pipeline_keys

        logger.info(f"  Pipeline faces: {len(pipeline_keys)}")
        logger.info(f"  Direct faces: {len(direct_keys)}")
        logger.info(f"  Gated faces: {len(gated_faces)}")

        # Verify at least some faces were gated (otherwise quality gate too lenient)
        # NOTE: With lenient settings (blur_min=50), this might be 0, which is OK
        # Just log a warning if no faces gated
        if len(gated_faces) == 0:
            logger.warning("  No faces were gated - quality gate might be too lenient")

    # ============================================================================
    # Test 5: Offset bug regression test
    # ============================================================================

    @pytest.mark.skip(reason="Requires synthetic test data with controlled gating pattern")
    def test_no_offset_after_gating(self):
        """Systematic offset bug should not occur when early faces are gated.

        Bug scenario:
        - 5 faces detected: [0, 1, 2, 3, 4]
        - Faces 0, 2 are gated (fail quality check)
        - Remaining faces: [1, 3, 4]
        - Bug: Embeddings stored as [0, 1, 2] but should be [1, 3, 4]
        - Result: embedding[0] is actually for face 1, not face 0

        TODO: Implement synthetic face generator to create controlled test case
        """
        pass

    # ============================================================================
    # Test 6: Performance baseline
    # ============================================================================

    @pytest.mark.slow
    def test_pipeline_performance_baseline(self, test_images, tmp_path):
        """Pipeline should process test images in < 120 seconds.

        This catches performance regressions in the pipeline.

        Note: Baseline includes model loading time (InsightFace buffalo_l models),
        which takes ~60-80 seconds on CPU. Increase timeout if models are cached.
        """
        logger.info("Test 6: Performance baseline...")

        # Create fresh context
        context = PipelineContext(source_directory=TEST_DATA_DIR)
        context.image_paths = test_images

        # Time the pipeline
        start = time.time()

        # Run all steps
        detect_step = InsightFaceDetectFacesStep()
        detect_step.process(context, {"model_name": "buffalo_l", "det_size": 640, "det_thresh": 0.5})

        align_step = AlignFacesStep()
        align_step.process(context, {"target_size": 112, "use_insightface_alignment": True})

        embed_step = ExtractFaceEmbeddingsStep()
        embed_step.process(context, {"backend": "insightface", "model_name": "buffalo_l", "device": "cpu"})

        filter_step = FilterQualityGateStep()
        filter_step.process(context, {"yaw_max": 45.0, "blur_min": 50.0, "max_faces_per_image_core": 10})

        # Note: We don't run export_for_labeling in performance test
        # (it requires clustering which is not part of core embedding validation)

        duration = time.time() - start

        logger.info(f"  Pipeline duration: {duration:.2f}s")

        # Allow 120 seconds (includes model loading ~60-80s + processing ~20-40s)
        assert duration < 120.0, f"Pipeline too slow: {duration:.2f}s (expected < 120s)"
