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

    def test_a_bad_box_is_skipped_not_fatal_and_files_are_written_whole(self, store):
        """review M6."""
        bundle = {"faces": [{"face_id": 0, "photo_id": "a", "bbox": BOX},
                            {"face_id": 5, "photo_id": "a", "bbox": [500, 500, 400, 400]}]}
        res = LocalRenders(store).calc(LocalRendersInputs("j", ALBUM, bundle))
        faces = store.paths("j").faces_dir
        assert res.bad_boxes == [5] and (faces / "0.jpg").is_file() and not (faces / "5.jpg").exists()
        assert not list(faces.glob("*.part")) and not list(store.paths("j").display_dir.glob("*.part"))


class _SlowStore:
    """An object store where every PUT takes 50 ms, like R2 from far away (0.5-1 s in real life)."""

    def __init__(self, real):
        import threading
        self._real, self.is_local, self.max_at_once, self._now, self._lock = real, False, 0, 0, threading.Lock()

    def list(self, prefix):
        return self._real.list(prefix)

    def _slow(self, fn, *args):
        import time
        with self._lock:
            self._now += 1
            self.max_at_once = max(self.max_at_once, self._now)
        time.sleep(0.05)
        fn(*args)
        with self._lock:
            self._now -= 1

    def put(self, key, data):
        self._slow(self._real.put, key, data)

    def put_file(self, key, path):
        self._slow(self._real.put_file, key, path)


def test_uploads_run_in_parallel_and_all_arrive(tmp_path):
    """spec-117 T8.2: ~1,300 files one PUT after another took 10-20 min and ran the L4 worker into its timeout."""
    import time
    store = JobStore(tmp_path / "jobs")
    store.stage_inputs("j", [(f"p{i}.jpg", _photo()) for i in range(20)])
    store.objects = _SlowStore(store.objects)
    album = {"picks": [{"id": f"p{i}"} for i in range(20)]}
    bundle = {"faces": [{"face_id": i * 3 + k, "photo_id": f"p{i}", "bbox": BOX} for i in range(20) for k in range(3)]}
    t0 = time.perf_counter()
    res = LocalRenders(store).calc(LocalRendersInputs("j", album, bundle))
    took = time.perf_counter() - t0
    assert (res.n_displays, res.n_crops) == (20, 60)
    assert len(list(store.paths("j").faces_dir.iterdir())) == 60 and len(list(store.paths("j").display_dir.iterdir())) == 20
    assert store.objects.max_at_once > 1, "uploads went one at a time"
    assert took < 80 * 0.05 / 2, f"{took:.2f}s for 80 uploads of 50 ms"       # one at a time would be >= 4 s
