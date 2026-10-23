"""spec-091: the FC export reuses the in-memory aligned crop instead of
re-decoding source photos per face (SIGHTING-110 / API starvation)."""

import json

import numpy as np

import sim_bench.pipeline.steps.face_cluster_export as fce
from sim_bench.pipeline.steps.face_cluster_export import (
    _crop_from_aligned,
    _generate_crops_from_bboxes,
)
from face_cluster.types import FaceRecord


class ut_CropFromAligned:

    def test_rgb_array_keeps_its_colours(self):
        aligned = np.zeros((256, 256, 3), dtype=np.uint8)
        aligned[:, :, 0] = 255  # pure red, RGB (align_faces output is RGB)
        img = _crop_from_aligned(aligned)
        assert img is not None
        assert img.size == (112, 112)
        assert img.mode == "RGB"
        assert img.getpixel((0, 0)) == (255, 0, 0)

    def test_a_real_alignment_of_a_red_photo_gives_a_red_crop(self, tmp_path):
        """SIGHTING-141: run the real chain (image_cache read -> align_faces -> crop) so the two ends of the colour
        contract are tested together; the old test asserted a swap and passed while every crop came out swapped."""
        from PIL import Image as PILImage

        from sim_bench.pipeline.steps.align_faces import align_face_with_orientation
        from sim_bench.pipeline.utils.image_cache import ImageCache

        photo = tmp_path / "red.jpg"
        PILImage.new("RGB", (400, 400), (220, 30, 30)).save(photo, quality=95)
        img = ImageCache(tmp_path / "cache").get(photo)          # how align_faces reads photos
        landmarks = [[170, 180], [230, 180], [200, 210], [175, 240], [225, 240]]
        aligned = align_face_with_orientation(img, landmarks, 0, bbox={"x": 150, "y": 150, "w": 100, "h": 120})
        assert aligned is not None
        r, g, b = _crop_from_aligned(aligned).getpixel((56, 56))
        assert r > 150 and b < 80, f"crop colour {(r, g, b)} is not red"

    def test_none_and_bad_shape_return_none(self):
        assert _crop_from_aligned(None) is None
        assert _crop_from_aligned(np.zeros((10, 10), dtype=np.uint8)) is None


class ut_ExportReusesAligned:

    def test_reuses_aligned_without_opening_source(self, tmp_path, monkeypatch):
        # If the export tries to open the source photo, fail loudly.
        def _boom(*a, **k):
            raise AssertionError("source photo was opened - aligned crop not reused")
        monkeypatch.setattr(fce.Image, "open", _boom)

        face = FaceRecord(
            face_id=1, image_id="img.jpg", bbox=(0.0, 0.0, 10.0, 10.0),
            image_path="D:/does/not/exist.jpg",
            aligned_face=np.zeros((256, 256, 3), dtype=np.uint8),
        )
        manifest = _generate_crops_from_bboxes([face], tmp_path)

        assert manifest == {1: "crops/face_0001_aligned.jpg"}
        assert (tmp_path / "crops" / "face_0001_aligned.jpg").exists()
        saved = json.loads((tmp_path / "crop_manifest.json").read_text())
        assert saved == {"1": "crops/face_0001_aligned.jpg"}
