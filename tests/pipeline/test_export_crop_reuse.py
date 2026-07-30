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

    def test_bgr_array_becomes_112_rgb(self):
        aligned = np.zeros((256, 256, 3), dtype=np.uint8)
        aligned[:, :, 0] = 255  # all-blue in BGR -> should map to RGB blue channel
        img = _crop_from_aligned(aligned)
        assert img is not None
        assert img.size == (112, 112)
        assert img.mode == "RGB"
        assert img.getpixel((0, 0)) == (0, 0, 255)  # BGR->RGB swap applied

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
