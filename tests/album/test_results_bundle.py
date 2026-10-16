"""spec-112 TA1.2 - ResultsBundler: PipelineContext -> per-photo / per-face rows.

Plain-data fakes stand in for the finished context; what is under test is the mapping (ids, scores,
decisions, person membership, embeddings), not the models that produced them.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np

from sim_bench.album.results_bundle import BUNDLE_VERSION, BundleInputs, ResultsBundler


def _face(fid, path, emb=None, aligned=None):
    return SimpleNamespace(face_id=fid, image_path=path, image_id=path.rsplit("/", 1)[-1],
                           bbox=(1.0, 2.0, 11.0, 12.0), det_score=0.9, blur_score=40.0,
                           pose=(1.0, 2.0, 3.0), is_core=True, rejection_reason=None,
                           embedding_normalized=emb, aligned_face=aligned)


def _context():
    w = "/scratch"
    faces = [_face(0, f"{w}/a.jpg", np.ones(4), np.zeros((112, 112, 3), np.uint8)),
             _face(1, f"{w}/a.jpg", np.full(4, 2.0)),
             _face(2, f"{w}/c.jpg")]
    return SimpleNamespace(
        straightened_from={f"{w}/c_straight.jpg": f"{w}/c.jpg"},
        composite_scores={f"{w}/a.jpg": 0.8, f"{w}/c_straight.jpg": 0.7},
        iqa_scores={f"{w}/a.jpg": 0.6}, ava_scores={}, sharpness_scores={},
        occlusion_scores={}, tilt_angles={f"{w}/a.jpg": float("nan")}, tilt_confidences={},
        scene_cluster_labels={f"{w}/a.jpg": 3, f"{w}/c.jpg": 5},
        geo_metadata={f"{w}/a.jpg": SimpleNamespace(timestamp=datetime(2026, 8, 22, 18, 12),
                                                     lat=48.1, lon=17.1)},
        step_decisions=[
            SimpleNamespace(item_id=f"{w}/q.jpg", item_type="image", step="filter_quality",
                            decision="rejected", reason="IQA 0.08 < 0.20"),
            SimpleNamespace(item_id=f"{w}/a.jpg:face_0", item_type="face", step="quality_gate",
                            decision="passed", reason=""),
        ],
        face_records=faces,
        people_clusters={7: [faces[0], faces[1]]},
    )


def _run(album_ids=("a", "c")):
    return ResultsBundler().calc(BundleInputs(context=_context(), input_ids=["c", "a", "q"],
                                              album_ids=list(album_ids)))


def test_one_photo_row_per_input_sorted_by_id():
    res = _run()
    assert [p["photo_id"] for p in res.photos] == ["a", "c", "q"]


def test_photo_scores_time_and_scene_are_mapped():
    a = _run().photos[0]
    assert a["scores"]["composite"] == 0.8 and a["scores"]["iqa"] == 0.6
    assert a["scores"]["tilt"] is None            # NaN is not a score
    assert a["taken_at"] == "2026-08-22T18:12:00" and a["lat"] == 48.1
    assert a["scene_label"] == 3 and a["in_album"] is True


def test_straightened_derivative_folds_back_to_original_id():
    c = _run().photos[1]
    assert c["scores"]["composite"] == 0.7


def test_photo_filtered_before_scoring_keeps_its_decision():
    """SIGHTING-128: a photo absent from album.json still carries why it left."""
    q = _run().photos[2]
    assert q["in_album"] is False
    assert q["decisions"] == [{"step": "filter_quality", "decision": "rejected",
                               "reason": "IQA 0.08 < 0.20"}]


def test_faces_carry_photo_and_person():
    faces = _run().faces
    assert [(f["face_id"], f["photo_id"], f["person_id"]) for f in faces] == \
        [(0, "a", 7), (1, "a", 7), (2, "c", None)]
    assert faces[0]["bbox"] == [1.0, 2.0, 11.0, 12.0] and faces[0]["pose"] == [1.0, 2.0, 3.0]


def test_embeddings_aligned_with_face_ids():
    res = _run()
    assert res.embedding_face_ids == [0, 1]
    assert res.embeddings.shape == (2, 4) and res.embeddings.dtype == np.float32
    assert res.embeddings[1, 0] == 2.0


def test_crops_only_for_faces_with_aligned_crop():
    assert list(_run().crops) == [0]


def test_manifest_counts():
    m = _run().manifest()
    assert m["bundle_version"] == BUNDLE_VERSION and m["n_photos"] == 3 and m["n_faces"] == 3


def test_empty_context_gives_photo_rows_only():
    res = ResultsBundler().calc(BundleInputs(context=SimpleNamespace(), input_ids=["x"]))
    assert len(res.photos) == 1 and res.faces == [] and res.embeddings is None
