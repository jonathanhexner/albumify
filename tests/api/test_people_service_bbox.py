"""Unit tests for People thumbnail bbox normalization (SIGHTING-104).

The Streamlit consumers (people_browser._crop_face, draw_face_bboxes) assume
``thumbnail_bbox`` is normalized to [0, 1]. The spec-079 ``FaceRecord.bbox`` is
in PIXELS, so the unnormalized value rendered the gray placeholder instead of a
cropped face. These tests lock the contract: the persisted bbox is always [0, 1].
"""

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import Base
from sim_bench.api.services.people_service import (
    PeopleService, _bbox_to_xywh, _normalized_bbox,
)
from app.streamlit.components.people_browser import _crop_face
from face_cluster.types import FaceRecord


def _face(**overrides) -> FaceRecord:
    """Minimal FaceRecord with a pixel bbox; overrides set spec-040 geometry."""
    base = dict(face_id=1, image_id="img.jpg", bbox=(1719.0, 730.0, 2230.0, 1506.0))
    base.update(overrides)
    return FaceRecord(**base)


class ut_NormalizedBbox:
    """_normalized_bbox must always return values in [0, 1] when it can."""

    def test_prefers_spec040_ratio_fields(self):
        face = _face(bbox_x_ratio=0.40, bbox_y_ratio=0.15,
                     bbox_w_ratio=0.12, bbox_h_ratio=0.25)
        assert _normalized_bbox(face) == [0.40, 0.15, 0.12, 0.25]

    def test_normalizes_pixels_via_image_dims(self):
        # No ratio fields -> fall back to pixel bbox / image dims.
        face = _face(image_width_px=4000, image_height_px=3000)
        bbox = _normalized_bbox(face)
        # (1719/4000, 730/3000, 511/4000, 776/3000)
        assert bbox == [1719 / 4000, 730 / 3000, 511 / 4000, 776 / 3000]
        assert all(0.0 <= v <= 1.0 for v in bbox)

    def test_last_resort_returns_pixels_for_consumer_guard(self):
        # No ratios, no dims -> raw reshape; values stay pixel-scale so the
        # consumer guard (_crop_face) is the safety net.
        face = _face()
        bbox = _normalized_bbox(face)
        assert bbox == [1719.0, 730.0, 511.0, 776.0]
        assert max(bbox) > 1.5  # the signal the consumer guard keys on

    def test_none_face_returns_none(self):
        assert _normalized_bbox(None) is None

    def test_bbox_to_xywh_is_unit_preserving(self):
        # Sanity: the reshape helper does NOT normalize (pixels in, pixels out).
        assert _bbox_to_xywh((10, 20, 60, 120)) == [10, 20, 50, 100]


def _in_memory_session():
    """Fresh in-memory SQLite session with all tables created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


class ut_ServiceToAppBboxContract:
    """End-to-end: what PeopleService persists must be what the app can render.

    This drives the REAL write path (`create_from_clusters`) and asserts the
    bbox the app reads back (`Person.thumbnail_bbox` + `face_instances[].bbox`)
    is normalized [0, 1] -- the contract that was silently violated (SIGHTING-104).
    """

    def test_persisted_thumbnail_bbox_is_normalized(self):
        session = _in_memory_session()
        svc = PeopleService(session)
        # spec-079 FaceRecord: PIXEL bbox + spec-040 image dims (no ratio fields).
        face = FaceRecord(
            face_id=1, image_id="20250822_112331.jpg",
            bbox=(1719.0, 730.0, 2230.0, 1506.0),
            image_width_px=3468, image_height_px=4624,
            image_path="D:/Budapest2025_Google/20250822_112331.jpg",
            face_index=0,
        )
        created = svc.create_from_clusters("album1", "run1", {0: [face]})

        assert len(created) == 1
        person = created[0]
        # The exact fields the Streamlit app reads.
        assert person.thumbnail_bbox is not None
        assert all(0.0 <= v <= 1.0 for v in person.thumbnail_bbox), person.thumbnail_bbox
        inst_bbox = person.face_instances[0]["bbox"]
        assert all(0.0 <= v <= 1.0 for v in inst_bbox), inst_bbox

    def test_persisted_bbox_renders_a_real_crop(self):
        # Close the loop: feed the persisted bbox to the app's crop helper and
        # confirm it yields a non-degenerate region (not the gray placeholder).
        session = _in_memory_session()
        svc = PeopleService(session)
        face = FaceRecord(
            face_id=1, image_id="img.jpg", bbox=(1719.0, 730.0, 2230.0, 1506.0),
            image_width_px=3468, image_height_px=4624,
            image_path="img.jpg", face_index=0,
        )
        person = svc.create_from_clusters("album1", "run1", {0: [face]})[0]

        img = Image.new("RGB", (3468, 4624), "white")
        crop = _crop_face(img, person.thumbnail_bbox)
        assert crop.size[0] > 0 and crop.size[1] > 0
        assert crop.size[0] < img.size[0]


class ut_CropFaceGuard:
    """_crop_face must render existing pixel-scale rows (no re-run needed)."""

    def test_pixel_bbox_and_equivalent_ratio_crop_identically(self):
        img = Image.new("RGB", (4000, 3000), "white")
        pixel_bbox = [1719.0, 730.0, 511.0, 776.0]
        ratio_bbox = [1719 / 4000, 730 / 3000, 511 / 4000, 776 / 3000]

        crop_px = _crop_face(img, pixel_bbox)
        crop_ratio = _crop_face(img, ratio_bbox)

        # Pixel-scale input is detected and normalized -> same crop as the
        # already-normalized input, and not a degenerate (empty) region.
        assert crop_px.size == crop_ratio.size
        assert crop_px.size[0] > 0 and crop_px.size[1] > 0
        assert crop_px.size[0] < img.size[0]  # actually cropped, not full image
