"""End-to-end pipeline tests using labeled test images (person_1, person_2, person_3).

Ground truth: test_data/face_clustering/ground_truth.csv
Source images: test_data/face_clustering/person_1/, person_2/, person_3/
Each folder contains solo portraits of one person — directory name is the identity label.
"""
import csv
from pathlib import Path

import pytest
import pandas as pd

from face_cluster import FaceClusteringPipeline, PipelineConfig
from tests.conftest import get_test_data_dir

TEST_DATA_DIR = get_test_data_dir() / "face_clustering"
GROUND_TRUTH_CSV = TEST_DATA_DIR / "ground_truth.csv"


def load_ground_truth() -> dict:
    """Return {relative_image_path: person_id} from ground_truth.csv."""
    gt = {}
    with open(GROUND_TRUTH_CSV, newline="") as f:
        for row in csv.DictReader(f):
            gt[row["image_path"]] = row["person_id"]
    return gt


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    """Run pipeline once on test data; reuse result across all tests in this module."""
    tmp_path = tmp_path_factory.mktemp("e2e_output")
    config = PipelineConfig()  # production defaults — no relaxation
    pipeline = FaceClusteringPipeline(config)
    return pipeline.run(TEST_DATA_DIR, tmp_path / "output")


@pytest.fixture(scope="module")
def faces_with_labels(pipeline_result):
    """faces.csv joined with ground truth person labels."""
    gt = load_ground_truth()
    faces_df = pd.read_csv(pipeline_result.output_dir / "faces.csv")

    if faces_df.empty:
        pytest.fail("faces.csv is empty — pipeline produced no faces")

    def lookup_person(image_path: str) -> str:
        rel = Path(image_path).relative_to(TEST_DATA_DIR)
        return gt.get(rel.as_posix(), "unknown")

    faces_df["person_id"] = faces_df["image_path"].apply(lookup_person)

    unknown = faces_df[faces_df["person_id"] == "unknown"]
    if not unknown.empty:
        pytest.fail(
            f"{len(unknown)} faces have image_path not found in ground_truth.csv: "
            f"{unknown['image_path'].tolist()}"
        )

    return faces_df


class ut_FaceClusteringPipeline:
    """E2E tests for FaceClusteringPipeline on labeled 3-person test data."""

    def test_output_files_exist(self, pipeline_result):
        out = pipeline_result.output_dir
        for fname in ("faces.csv", "clusters.csv", "export_summary.json", "crop_manifest.json"):
            assert (out / fname).exists(), f"Missing output file: {fname}"

    def test_no_null_image_paths(self, pipeline_result):
        faces_df = pd.read_csv(pipeline_result.output_dir / "faces.csv")
        assert not faces_df.empty, "faces.csv is empty"
        null_count = faces_df["image_path"].isna().sum()
        assert null_count == 0, f"{null_count} faces have null image_path"

    def test_face_ids_are_globally_unique(self, pipeline_result):
        """Every face must have a unique face_id across the entire run.

        Regression test for SIGHTING-012: face_id_counter resets to 0 on each
        call to detect_and_embed(), causing all faces to share only N IDs where
        N = max faces in any single image. Crops overwrite each other and the
        cluster browser shows the same few images repeatedly.
        """
        faces_df = pd.read_csv(pipeline_result.output_dir / "faces.csv")
        assert not faces_df.empty, "faces.csv is empty"
        n_total = len(faces_df)
        n_unique = faces_df["face_id"].nunique()
        assert n_unique == n_total, (
            f"face_id not globally unique: {n_unique} unique IDs for {n_total} faces. "
            "This means face_id_counter is resetting between images."
        )

    def test_holdout_path_exercised(self, pipeline_result):
        """Some faces must fail quality gating (cluster_id == -1).

        With production-default config (blur_min=50), at least some faces should
        fail quality gating. This exercises the core_indices index-mapping path —
        the bug that caused SIGHTING-010.
        """
        faces_df = pd.read_csv(pipeline_result.output_dir / "faces.csv")
        assert not faces_df.empty, "faces.csv is empty"
        n_holdout = (faces_df["cluster_id"] == -1).sum()
        assert n_holdout > 0, (
            "No faces went to holdout under production-default config. "
            "Quality gating may be broken. "
            "This test exists to exercise the core_indices index mapping."
        )

    def test_produces_three_clusters(self, pipeline_result):
        """Pipeline must produce exactly 3 clusters — one per person."""
        n = pipeline_result.cluster_result.n_clusters
        assert n == 3, f"Expected 3 clusters (one per person), got {n}"

    def test_cluster_purity(self, faces_with_labels):
        """Each cluster must contain faces from only one person (purity = 1.0).

        Catches the case where different people are merged into one cluster.
        """
        clustered = faces_with_labels[faces_with_labels["cluster_id"] != -1]
        assert len(clustered) > 0, "No clustered faces to evaluate purity on"

        impure = []
        for cluster_id, group in clustered.groupby("cluster_id"):
            persons = group["person_id"].unique()
            if len(persons) > 1:
                impure.append((cluster_id, list(persons)))

        assert not impure, f"Mixed-person clusters found: {impure}"

    def test_cluster_completeness(self, faces_with_labels):
        """All faces of the same person must end up in the same cluster.

        Catches over-clustering: person_1's faces split across 2+ clusters.
        Purity alone passes when every face is its own cluster — completeness does not.
        """
        clustered = faces_with_labels[faces_with_labels["cluster_id"] != -1]
        assert len(clustered) > 0, "No clustered faces to evaluate completeness on"

        split_persons = []
        for person_id, group in clustered.groupby("person_id"):
            clusters = group["cluster_id"].unique()
            if len(clusters) > 1:
                split_persons.append((person_id, list(clusters)))

        assert not split_persons, (
            f"Person(s) split across multiple clusters (over-clustering): {split_persons}"
        )

    def test_all_persons_represented(self, faces_with_labels):
        """Every person in ground truth must have at least one clustered face."""
        gt = load_ground_truth()
        expected_persons = set(gt.values())

        clustered = faces_with_labels[faces_with_labels["cluster_id"] != -1]
        assert len(clustered) > 0, "No clustered faces"

        represented = set(clustered["person_id"].unique())
        missing = expected_persons - represented
        assert not missing, f"These people have no clustered faces: {missing}"
