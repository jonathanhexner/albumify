"""
Full pipeline test: Run complete face detection → crop → align → embed on source images.

Tests the entire pipeline from original photos to embeddings, verifying against ground truth.

Run with `-s -o log_cli=true --log-cli-level=INFO` to stream the diagnostic output
from `test_diagnostic_distance_matrices` to the console. The same test also writes an
HTML report under `docs/project/`.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from face_cluster import InsightFaceEmbedder

logger = logging.getLogger(__name__)


# Ground truth labels (same as embedding test)
LABELS = {
    545: 1, 569: 1, 573: 1, 634: 1, 637: 1,  # Person 1
    550: 2, 551: 2, 558: 2, 580: 2,          # Person 2
    546: 3, 557: 3, 562: 3, 584: 3,          # Person 3 (557 is profile)
    587: 4, 589: 4,                          # Person 4
}

MAX_WITHIN = {
    1: 0.30, 2: 0.31, 3: 0.63, 4: 0.05
}

MIN_BETWEEN = 0.60


@pytest.fixture(scope="module")
def ground_truth_mapping():
    """Load face_id → (image, face_index) mapping."""
    mapping_file = Path("test_data/ground_truth_mapping.csv")
    mapping = {}

    with open(mapping_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                face_id = int(parts[0])
                image_name = Path(parts[1]).name
                face_index = int(parts[2])
                mapping[face_id] = (image_name, face_index)

    return mapping


@pytest.fixture(scope="module")
def ground_truth_embeddings():
    """Load ground truth embeddings from pre-extracted crops."""
    test_dir = Path("test_data/face_crops_ground_truth")
    embedder = InsightFaceEmbedder(model_name='buffalo_l')

    embeddings = {}
    for face_id in LABELS.keys():
        crop_path = test_dir / f"face_{face_id:04d}_aligned.jpg"

        if crop_path.exists():
            img = np.array(Image.open(crop_path))

            # Handle grayscale/RGBA
            if len(img.shape) == 2:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            embeddings[face_id] = embedder.get_embedding(img)

    return embeddings


@pytest.fixture(scope="module")
def pipeline_detection(ground_truth_mapping):
    """
    Run detection + alignment once. For each ground-truth face_id, returns:
      embedding       — L2-normalized 512-D vector from the recognition model
      aligned_crop    — 112x112 RGB crop produced by norm_crop on face.kps
                         (what the recognition model effectively "sees")
      image_name      — source image filename
      face_index      — detection_order index in that image

    Both `pipeline_embeddings` and the crop-dump diagnostic derive from this.
    """
    source_dir = Path("test_data/source_images_ground_truth")
    embedder = InsightFaceEmbedder(model_name='buffalo_l', face_ordering='detection_order')

    # InsightFace alignment helper. Kept inside the fixture so a missing
    # install fails the diagnostic only, not collection.
    from insightface.utils import face_align

    results: dict[int, dict] = {}

    logger.info("=" * 70)
    logger.info("RUNNING FULL PIPELINE")
    logger.info("=" * 70)

    images_to_process: dict[str, list[tuple[int, int]]] = {}
    for face_id, (image_name, face_index) in ground_truth_mapping.items():
        images_to_process.setdefault(image_name, []).append((face_id, face_index))

    logger.info("Processing %d source images...", len(images_to_process))

    from PIL import ImageOps
    from pillow_heif import register_heif_opener
    register_heif_opener()

    for image_name, faces_in_image in sorted(images_to_process.items()):
        image_path = source_dir / image_name

        if not image_path.exists():
            logger.warning("SKIP %s: File not found", image_name)
            continue

        logger.info("%s:", image_name)

        try:
            with Image.open(image_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img_rgb = np.array(pil_img)
        except Exception as e:
            logger.warning("  ERROR: Failed to load image - %s", e)
            continue

        detected_faces = embedder.app.get(img_rgb)
        logger.info("  Detected %d faces", len(detected_faces))

        for face_id, face_index in faces_in_image:
            if face_index >= len(detected_faces):
                logger.warning(
                    "  Face %d (index %d): Index out of range (only %d detected)",
                    face_id, face_index, len(detected_faces),
                )
                continue
            face = detected_faces[face_index]
            aligned = face_align.norm_crop(img_rgb, landmark=face.kps, image_size=112)
            results[face_id] = {
                "embedding": face.embedding / np.linalg.norm(face.embedding),
                "aligned_crop": aligned,  # RGB, 112x112
                "image_name": image_name,
                "face_index": face_index,
            }
            logger.info("  Face %d (index %d): OK", face_id, face_index)

    logger.info("Total embeddings extracted: %d/%d", len(results), len(LABELS))
    return results


@pytest.fixture(scope="module")
def pipeline_embeddings(pipeline_detection):
    """Thin view over pipeline_detection returning only the embeddings."""
    return {fid: r["embedding"] for fid, r in pipeline_detection.items()}


def cosine_distance(emb1, emb2):
    """Compute cosine distance."""
    return 1.0 - np.dot(emb1, emb2)


def compute_distance_matrix(embeddings):
    """Compute pairwise distance matrix."""
    face_ids = sorted(embeddings.keys())
    n = len(face_ids)

    distances = np.zeros((n, n))
    for i, fid_a in enumerate(face_ids):
        for j, fid_b in enumerate(face_ids):
            if i != j:
                distances[i, j] = cosine_distance(embeddings[fid_a], embeddings[fid_b])

    return pd.DataFrame(distances, index=face_ids, columns=face_ids)


class TestFullPipeline:
    """Test full pipeline from source images to embeddings."""

    def test_pipeline_extracts_all_faces(self, pipeline_embeddings):
        """Verify pipeline extracted all ground truth faces."""
        missing = [fid for fid in LABELS.keys() if fid not in pipeline_embeddings]

        if missing:
            logger.info("Missing faces: %s", missing)

        # Allow some tolerance (e.g., 90% success rate)
        success_rate = len(pipeline_embeddings) / len(LABELS)
        assert success_rate >= 0.90, \
            f"Pipeline only extracted {len(pipeline_embeddings)}/{len(LABELS)} faces ({success_rate:.1%})"

    def test_pipeline_embeddings_match_ground_truth(self, pipeline_embeddings, ground_truth_embeddings):
        """Verify pipeline embeddings are similar to ground truth crops."""
        mismatches = []

        for face_id in pipeline_embeddings.keys():
            if face_id in ground_truth_embeddings:
                pipe_emb = pipeline_embeddings[face_id]
                gt_emb = ground_truth_embeddings[face_id]

                similarity = np.dot(pipe_emb, gt_emb)

                # Embeddings should be very similar (>0.95)
                if similarity < 0.95:
                    mismatches.append(f"Face {face_id}: similarity = {similarity:.3f}")

        if mismatches:
            logger.info("Embedding mismatches:")
            for msg in mismatches:
                logger.info("  %s", msg)

        assert not mismatches, f"{len(mismatches)} faces have embeddings that don't match ground truth"

    def test_pipeline_preserves_identity_structure(self, pipeline_embeddings):
        """Verify pipeline preserves person identity structure (within/between distances)."""

        # Only test faces that were successfully extracted
        available_faces = set(pipeline_embeddings.keys())

        if len(available_faces) < len(LABELS) * 0.9:
            pytest.skip(f"Too few faces extracted ({len(available_faces)}/{len(LABELS)})")

        dist_df = compute_distance_matrix(pipeline_embeddings)

        failures = []

        # Test within-person distances
        for person_id, max_dist in MAX_WITHIN.items():
            faces = [fid for fid, label in LABELS.items() if label == person_id and fid in available_faces]

            if len(faces) < 2:
                continue

            for i, face_a in enumerate(faces):
                for face_b in faces[i+1:]:
                    dist = dist_df.loc[face_a, face_b]
                    if dist > max_dist:
                        failures.append(
                            f"Person {person_id}: {face_a}<->{face_b} = {dist:.3f} (exceeds {max_dist:.2f})"
                        )

        # Test between-people distances
        face_list = list(available_faces)
        for i, face_a in enumerate(face_list):
            for face_b in face_list[i+1:]:
                if LABELS[face_a] != LABELS[face_b]:
                    dist = dist_df.loc[face_a, face_b]
                    if dist < MIN_BETWEEN:
                        failures.append(
                            f"Different people ({LABELS[face_a]} vs {LABELS[face_b]}): "
                            f"{face_a}<->{face_b} = {dist:.3f} (below {MIN_BETWEEN:.2f})"
                        )

        if failures:
            logger.info("Pipeline distance matrix:\n%s", dist_df.round(3).to_string())
            logger.info("Failures:")
            for f in failures[:10]:
                logger.info("  %s", f)

        assert not failures, f"{len(failures)} distance violations in pipeline output"


class TestPipelineVsGroundTruth:
    """Compare pipeline output to ground truth crops."""

    def test_distance_matrix_correlation(self, pipeline_embeddings, ground_truth_embeddings):
        """Verify pipeline distance matrix correlates highly with ground truth."""

        # Only compare faces that are in both sets
        common_faces = set(pipeline_embeddings.keys()) & set(ground_truth_embeddings.keys())

        if len(common_faces) < 10:
            pytest.skip(f"Too few common faces ({len(common_faces)})")

        pipe_dist = compute_distance_matrix({fid: pipeline_embeddings[fid] for fid in common_faces})
        gt_dist = compute_distance_matrix({fid: ground_truth_embeddings[fid] for fid in common_faces})

        # Compute correlation of distance matrices
        pipe_dists = []
        gt_dists = []

        face_list = sorted(common_faces)
        for i, fid_a in enumerate(face_list):
            for fid_b in face_list[i+1:]:
                pipe_dists.append(pipe_dist.loc[fid_a, fid_b])
                gt_dists.append(gt_dist.loc[fid_a, fid_b])

        correlation = np.corrcoef(pipe_dists, gt_dists)[0, 1]

        logger.info("Distance matrix correlation: %.4f", correlation)
        logger.info("Common faces: %d/%d", len(common_faces), len(LABELS))

        assert correlation > 0.90, \
            f"Pipeline distance matrix has low correlation with ground truth: {correlation:.4f}"


# ---------------------------------------------------------------------------
# Diagnostic test — always passes, never asserts. Collects matrices so a
# human can localize where pipeline/GT embeddings diverge. No interpretation
# is performed here; it just dumps tables to logs + HTML.
# ---------------------------------------------------------------------------

DIAGNOSTIC_HTML = Path("docs/project/face_pipeline_diagnostic.html")


def _summarize_within_between(dist_df: pd.DataFrame) -> pd.DataFrame:
    """Per-person within-group and overall between-group distance stats."""
    rows = []
    face_ids = list(dist_df.index)
    by_person: dict[int, list[int]] = {}
    for fid in face_ids:
        by_person.setdefault(LABELS[fid], []).append(fid)

    for person_id, members in sorted(by_person.items()):
        within = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                within.append(dist_df.loc[a, b])
        rows.append({
            "scope": f"within person {person_id} (n={len(members)})",
            "count": len(within),
            "min": np.min(within) if within else np.nan,
            "mean": np.mean(within) if within else np.nan,
            "max": np.max(within) if within else np.nan,
        })

    between = []
    for i, a in enumerate(face_ids):
        for b in face_ids[i + 1:]:
            if LABELS[a] != LABELS[b]:
                between.append(dist_df.loc[a, b])
    rows.append({
        "scope": "between people",
        "count": len(between),
        "min": np.min(between) if between else np.nan,
        "mean": np.mean(between) if between else np.nan,
        "max": np.max(between) if between else np.nan,
    })
    return pd.DataFrame(rows)


def test_diagnostic_distance_matrices(pipeline_embeddings, ground_truth_embeddings):
    """Always-passes diagnostic: dump all matrices for human inspection."""
    common = sorted(set(pipeline_embeddings) & set(ground_truth_embeddings))
    if len(common) < 2:
        pytest.skip(f"Too few common faces ({len(common)})")

    pipe_dist = compute_distance_matrix({fid: pipeline_embeddings[fid] for fid in common})
    gt_dist = compute_distance_matrix({fid: ground_truth_embeddings[fid] for fid in common})
    residual = (pipe_dist - gt_dist).round(3)

    # Per-face cosine similarity between pipeline and GT embeddings for the same face_id.
    per_face = pd.DataFrame({
        "face_id": common,
        "person": [LABELS[fid] for fid in common],
        "pipe_vs_gt_dot": [
            float(np.dot(pipeline_embeddings[fid], ground_truth_embeddings[fid]))
            for fid in common
        ],
        "pipe_norm": [float(np.linalg.norm(pipeline_embeddings[fid])) for fid in common],
        "gt_norm": [float(np.linalg.norm(ground_truth_embeddings[fid])) for fid in common],
    }).set_index("face_id")

    summary_pipe = _summarize_within_between(pipe_dist)
    summary_gt = _summarize_within_between(gt_dist)

    logger.info("=" * 70)
    logger.info("DIAGNOSTIC: per-face pipeline vs ground-truth (dot product)")
    logger.info("=" * 70)
    logger.info("\n%s", per_face.round(4).to_string())

    logger.info("Pipeline distance matrix:\n%s", pipe_dist.round(3).to_string())
    logger.info("Ground-truth distance matrix:\n%s", gt_dist.round(3).to_string())
    logger.info("Residual (pipe - gt):\n%s", residual.to_string())
    logger.info("Pipeline within/between summary:\n%s", summary_pipe.round(3).to_string(index=False))
    logger.info("Ground-truth within/between summary:\n%s", summary_gt.round(3).to_string(index=False))

    # HTML report — heatmaps via pandas Styler so reviewers can eyeball clusters.
    def _style(df: pd.DataFrame, title: str) -> str:
        try:
            html = df.style.background_gradient(cmap="RdYlGn_r", axis=None).format(precision=3).to_html()
        except Exception:
            html = df.round(3).to_html()
        return f"<h2>{title}</h2>\n{html}\n"

    DIAGNOSTIC_HTML.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Face pipeline diagnostic</title>",
        "<style>body{font-family:sans-serif;margin:24px;} table{border-collapse:collapse;} "
        "th,td{padding:4px 8px;border:1px solid #ccc;font-size:12px;}</style>",
        f"<h1>Face pipeline diagnostic — {len(common)} common faces</h1>",
        f"<p>Person labels: {LABELS}</p>",
        _style(per_face, "Per-face pipeline-vs-GT dot product (1.0 = identical)"),
        _style(pipe_dist, "Pipeline distance matrix"),
        _style(gt_dist, "Ground-truth distance matrix"),
        _style(residual, "Residual (pipe - gt)"),
        f"<h2>Pipeline within/between summary</h2>{summary_pipe.round(3).to_html(index=False)}",
        f"<h2>Ground-truth within/between summary</h2>{summary_gt.round(3).to_html(index=False)}",
    ]
    DIAGNOSTIC_HTML.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Diagnostic HTML written to %s", DIAGNOSTIC_HTML)


CROPS_DIR = Path("docs/project/face_pipeline_crops")
CROPS_INDEX_HTML = CROPS_DIR / "index.html"


def test_diagnostic_dump_aligned_crops(pipeline_detection, ground_truth_embeddings, pipeline_embeddings):
    """Dump pipeline-aligned vs ground-truth-aligned crops side-by-side.

    Always passes. Lets a human eyeball whether:
    - the same person is selected at the given detection_index (face_index)
    - the alignment matches (pose, rotation, crop tightness)
    - any face_id is silently picking up the wrong face from a multi-face image
    """
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    gt_src_dir = Path("test_data/face_crops_ground_truth")

    rows: list[str] = []
    for face_id in sorted(pipeline_detection):
        entry = pipeline_detection[face_id]
        person = LABELS[face_id]

        # Pipeline crop — aligned via norm_crop on the detected landmarks.
        pipe_path = CROPS_DIR / f"face_{face_id:04d}_pipeline.jpg"
        Image.fromarray(entry["aligned_crop"]).save(pipe_path, quality=92)

        # GT crop — copy/normalize from the existing fixture so they sit next
        # to each other in this folder (no symlinks on Windows in CI).
        gt_src = gt_src_dir / f"face_{face_id:04d}_aligned.jpg"
        gt_path = CROPS_DIR / f"face_{face_id:04d}_ground_truth.jpg"
        if gt_src.exists():
            with Image.open(gt_src) as g:
                g.convert("RGB").save(gt_path, quality=92)
            gt_present = True
        else:
            gt_present = False

        dot = (
            float(np.dot(pipeline_embeddings[face_id], ground_truth_embeddings[face_id]))
            if face_id in ground_truth_embeddings else float("nan")
        )

        rows.append(
            f"<tr>"
            f"<td>{face_id}</td><td>person {person}</td>"
            f"<td>{entry['image_name']}<br><small>idx={entry['face_index']}</small></td>"
            f"<td><img src='{pipe_path.name}' width='128'></td>"
            f"<td>"
            + (f"<img src='{gt_path.name}' width='128'>" if gt_present else "<em>(missing)</em>")
            + f"</td>"
            f"<td>{dot:.3f}</td>"
            f"</tr>"
        )

    preamble = """
    <h2>What this page is</h2>
    <p>One row per <strong>face_id</strong> in the ground-truth fixture. Each row shows
    two 112&times;112 crops of the same face_id and asks: <em>do they look like the same
    human, and does the embedder think so?</em></p>
    <h2>Column definitions</h2>
    <dl>
      <dt><strong>face_id</strong></dt>
      <dd>Arbitrary integer ID for each fixture face (e.g. 545). Stable key, no semantics.</dd>
      <dt><strong>label</strong></dt>
      <dd>Which of the four ground-truth identities this face belongs to.
      "person 1" groups faces that share a human; the digit itself is arbitrary.</dd>
      <dt><strong>source / det_idx</strong></dt>
      <dd>Original photo + which face within it (InsightFace returns multiple faces
      per image, sorted by detection confidence: idx=0 is the most confident).</dd>
      <dt><strong>pipeline crop</strong></dt>
      <dd><strong>Generated live by this test run.</strong> Source photo &rarr;
      <code>embedder.app.get()</code> &rarr; pick face at det_idx &rarr;
      <code>norm_crop</code> on its 5 landmarks &rarr; this 112&times;112 RGB image.
      Literally what today's recognition model sees.</dd>
      <dt><strong>ground-truth crop</strong></dt>
      <dd><strong>Static fixture</strong> from
      <code>test_data/face_crops_ground_truth/face_NNNN_aligned.jpg</code>, checked
      into the repo at commit 01d292c. Treated as the "correct" pre-aligned reference.</dd>
      <dt><strong>pipe&middot;gt</strong></dt>
      <dd>Cosine similarity between two embeddings (unit vectors, so this is just
      the dot product): <code>embed(pipeline_crop)</code> &middot;
      <code>embed(ground_truth_crop)</code>. <strong>1.0</strong> = identical,
      <strong>~0</strong> = unrelated, <strong>negative</strong> = pointing opposite ways.</dd>
    </dl>
    <h2>How to read it</h2>
    <ul>
      <li>Two crops look like the same human + high pipe&middot;gt &rarr; healthy row.</li>
      <li>Two crops look like the same human + low pipe&middot;gt &rarr; embedder changed
          since the GT was generated.</li>
      <li>Two crops show <em>different</em> humans &rarr; det_idx ordering shifted, so the
          live pipeline is picking up the wrong face from the source image.</li>
      <li>Pipeline crop is tilted / off-center vs the GT &rarr; alignment routine drift.</li>
    </ul>
    """
    CROPS_INDEX_HTML.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Pipeline vs GT aligned crops</title>"
        "<style>body{font-family:sans-serif;margin:24px;max-width:1000px;}"
        "table{border-collapse:collapse;margin-top:16px;} "
        "th,td{padding:6px;border:1px solid #ccc;vertical-align:top;}"
        "img{display:block;} dt{margin-top:6px;} dd{margin-left:20px;}</style>"
        f"<h1>Pipeline vs ground-truth aligned crops ({len(pipeline_detection)} faces)</h1>"
        + preamble +
        "<table>"
        "<tr><th>face_id</th><th>label</th><th>source / det_idx</th>"
        "<th>pipeline crop</th><th>ground-truth crop</th><th>pipe&middot;gt</th></tr>"
        + "\n".join(rows)
        + "</table>",
        encoding="utf-8",
    )
    logger.info("Crop comparison written to %s", CROPS_INDEX_HTML)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-o', 'log_cli=true', '--log-cli-level=INFO'])
