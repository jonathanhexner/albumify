"""spec-086 (slice 1): ImageRepository + person-images metric join.

AC1/AC2 (data reaches the endpoint), AC4 (equivalence with the blob), and the
repository unit behaviour. Uses an in-memory SQLite seeded with one run.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import (
    Base, Album, PipelineRun, PipelineResult, Person,
    ImageMetricRow, FaceMetricRow,
)
from sim_bench.api.repositories.image_repository import ImageRepository
from sim_bench.api.services.people_service import PeopleService
from sim_bench.api.services.pipeline_service import PipelineService
from sim_bench.api.schemas.people import PersonImageResponse


ALBUM = "alb1"
RUN = "run1"
PID = "person-A"

IMG_SEL = "D:/x/a.jpg"      # selected, has the person
IMG_REJ = "D:/x/b.jpg"      # not selected, has the person
IMG_OTHER = "D:/x/c.jpg"    # no person link


def _metric(path, is_selected, composite, faces):
    """A minimal ImageMetrics-shaped blob dict, like _build_image_metrics emits."""
    return {
        "iqa_score": 0.5, "ava_score": 5.0, "sharpness": 0.4,
        "composite_score": composite, "cluster_id": 1,
        "face_count": len(faces), "is_selected": is_selected,
        "face_pose_scores": [f["pose"] for f in faces],
        "face_eyes_scores": [f["eyes"] for f in faces],
        "filter_scores": [
            {"face_index": f["idx"], "confidence": f["conf"], "filter_passed": True,
             "bbox": {"x": f["x"], "y": 0.1, "w": 0.2, "h": 0.2,
                      "x_px": f["xpx"], "y_px": 100, "w_px": 200, "h_px": 200}}
            for f in faces
        ],
    }


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(Album(id=ALBUM, name="a", source_path="D:/x"))
    s.add(PipelineRun(id=RUN, album_id=ALBUM, pipeline_name="p"))
    image_metrics = {
        IMG_SEL: _metric(IMG_SEL, True, 0.9,
                         [{"idx": 0, "conf": 0.8, "pose": 0.86, "eyes": 0.7, "x": 0.4, "xpx": 1700},
                          {"idx": 1, "conf": 0.7, "pose": 0.10, "eyes": 0.5, "x": 0.7, "xpx": 2500}]),
        IMG_REJ: _metric(IMG_REJ, False, 0.3,
                         [{"idx": 0, "conf": 0.6, "pose": 0.2, "eyes": 0.4, "x": 0.5, "xpx": 1800}]),
        IMG_OTHER: _metric(IMG_OTHER, True, 0.8,
                           [{"idx": 0, "conf": 0.9, "pose": 0.9, "eyes": 0.9, "x": 0.3, "xpx": 1000}]),
    }
    s.add(PipelineResult(id="res1", run_id=RUN, image_metrics=image_metrics,
                         selected_images=[IMG_SEL, IMG_OTHER]))
    # Person A appears in IMG_SEL (face 0) and IMG_REJ (face 0) — not IMG_OTHER.
    s.add(Person(id=PID, album_id=ALBUM, run_id=RUN, person_index=0,
                 face_instances=[
                     {"image_path": IMG_SEL, "face_index": 0},
                     {"image_path": IMG_REJ, "face_index": 0},
                 ]))
    s.commit()
    # Write the normalized tables (the production write path).
    PipelineService(s)._write_metric_tables(RUN, image_metrics,
                                            s.query(Person).all())
    s.commit()
    yield s
    s.close()


def test_write_links_faces_to_person(session):
    linked = session.query(FaceMetricRow).filter(FaceMetricRow.person_id == PID).count()
    assert linked == 2  # face 0 of IMG_SEL and IMG_REJ
    assert session.query(ImageMetricRow).count() == 3
    assert session.query(FaceMetricRow).count() == 4  # 2 + 1 + 1


def test_repository_returns_person_images_with_metrics(session):
    imgs = ImageRepository(session).get_images_for_person(RUN, PID)
    paths = {m.path for m in imgs}
    assert paths == {IMG_SEL, IMG_REJ}          # IMG_OTHER excluded (no person link)
    by_path = {m.path: m for m in imgs}
    sel = by_path[IMG_SEL]
    assert sel.is_selected is True
    assert sel.composite_score == 0.9
    assert sel.face_count == 2                   # ALL faces in image, not just the person's
    assert sel.face_pose_scores == [0.86, 0.10]
    assert sel.filter_scores[0]["bbox"]["x_px"] == 1700


def test_selected_and_not_selected_partition(session):
    imgs = ImageRepository(session).get_images_for_person(RUN, PID)
    assert [m.is_selected for m in sorted(imgs, key=lambda x: x.path)] == [True, False]


def test_endpoint_carries_metrics_and_validates(session):
    """AC2: get_person_images output has the metrics AND validates as the response model."""
    rows = PeopleService(session).get_person_images(ALBUM, PID)
    assert len(rows) == 2
    validated = [PersonImageResponse(**r) for r in rows]   # would raise if a field is wrong
    sel = next(v for v in validated if v.image_path == IMG_SEL)
    assert sel.is_selected is True
    assert sel.filter_scores and "bbox" in sel.filter_scores[0]
    assert sel.faces                                       # person face instances preserved


def test_equivalence_with_blob(session):
    """AC4: per-image is_selected/composite from the repo == the blob image_metrics."""
    blob = session.query(PipelineResult).filter(PipelineResult.run_id == RUN).first().image_metrics
    for m in ImageRepository(session).get_images_for_person(RUN, PID):
        assert m.is_selected == blob[m.path]["is_selected"]
        assert m.composite_score == blob[m.path]["composite_score"]
        assert m.face_count == blob[m.path]["face_count"]
