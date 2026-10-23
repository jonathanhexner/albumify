"""spec-115 W2 (R10, R14): the server makes display images and face crops from its own copies.

Photos are painted so the face box is red and everything else blue: a crop cut from the right place is red in the
middle. One photo carries an EXIF rotation, because the detector's boxes are in the upright frame (T2.1).
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from sim_bench.control.local_renders import LocalRenders, LocalRendersInputs
from sim_bench.serving.job_store import JobStore

BOX = [40, 20, 80, 70]   # x1, y1, x2, y2 in the upright photo


def _photo(rotate_exif: int = 1) -> bytes:
    """160x120 upright scene with a red face box; stored rotated with an EXIF tag when ``rotate_exif`` is 6."""
    img = Image.new("RGB", (160, 120), (20, 40, 220))
    img.paste((230, 20, 20), (BOX[0], BOX[1], BOX[2], BOX[3]))
    buf = io.BytesIO()
    if rotate_exif == 6:                      # camera held sideways: pixels stored 90° ccw, tag says "rotate cw"
        stored = img.transpose(Image.Transpose.ROTATE_90)
        exif = Image.Exif()
        exif[274] = 6
        stored.save(buf, "JPEG", quality=95, exif=exif)
    else:
        img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs")
    s.stage_inputs("j", [("a.jpg", _photo()), ("b.jpg", _photo(6)), ("c.jpg", b"not a jpeg")])
    return s


ALBUM = {"picks": [{"id": "a"}, {"id": "b"}], "dropped": [{"id": "c"}]}
BUNDLE = {"faces": [{"face_id": 0, "photo_id": "a", "bbox": BOX},
                    {"face_id": 1, "photo_id": "b", "bbox": BOX},
                    {"face_id": 2, "photo_id": "c", "bbox": BOX},
                    {"face_id": 3, "photo_id": "a", "bbox": None}]}


def _centre(path):
    img = Image.open(path).convert("RGB")
    return img.getpixel((img.width // 2, img.height // 2))


class ut_LocalRenders:

    def test_displays_for_the_picks_only(self, store):
        res = LocalRenders(store).calc(LocalRendersInputs("j", ALBUM, BUNDLE))
        assert sorted(p.name for p in store.paths("j").display_dir.iterdir()) == ["a.jpg", "b.jpg"]
        assert res.n_displays == 2

    def test_crop_is_cut_from_the_face_box(self, store):
        LocalRenders(store, margin=0.0).calc(LocalRendersInputs("j", ALBUM, BUNDLE))
        r, g, b = _centre(store.paths("j").faces_dir / "0.jpg")
        assert r > 180 and b < 80

    def test_rotated_photo_is_cut_upright_like_the_detector_reads_it(self, store):
        LocalRenders(store, margin=0.0).calc(LocalRendersInputs("j", ALBUM, BUNDLE))
        crop = Image.open(store.paths("j").faces_dir / "1.jpg")
        assert crop.size == (BOX[2] - BOX[0], BOX[3] - BOX[1])     # upright box, not its rotated shape
        r, g, b = _centre(store.paths("j").faces_dir / "1.jpg")
        assert r > 180 and b < 80

    def test_margin_adds_context(self, store):
        LocalRenders(store, margin=0.25).calc(LocalRendersInputs("j", ALBUM, BUNDLE))
        assert Image.open(store.paths("j").faces_dir / "0.jpg").size == (60, 74)

    def test_unreadable_photo_and_faces_without_a_box_are_skipped(self, store):
        res = LocalRenders(store).calc(LocalRendersInputs("j", ALBUM, BUNDLE))
        assert sorted(p.name for p in store.paths("j").faces_dir.iterdir()) == ["0.jpg", "1.jpg"]
        assert res.unreadable_photos == ["c"] and res.n_crops == 2

    def test_existing_files_of_an_older_job_are_kept(self, store):
        """R14: a job whose worker sent displays and crops keeps them."""
        paths = store.paths("j")
        paths.faces_dir.mkdir(parents=True)
        (paths.faces_dir / "0.jpg").write_bytes(b"worker crop")
        paths.display_dir.mkdir(parents=True)
        (paths.display_dir / "a.jpg").write_bytes(b"worker display")
        res = LocalRenders(store).calc(LocalRendersInputs("j", ALBUM, BUNDLE))
        assert (paths.faces_dir / "0.jpg").read_bytes() == b"worker crop"
        assert (paths.display_dir / "a.jpg").read_bytes() == b"worker display"
        assert res.n_kept == 2
