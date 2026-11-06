"""spec-116 R5: stored paths point at the originals; pixel boxes scale; [0,1] boxes stay."""
from __future__ import annotations

import pytest
from PIL import Image

from sim_bench.serving.job_store import JobStore
from sim_bench.serving.path_map import PathMap
from sim_bench.serving.run_output import RunOutput
from sim_bench.serving.staging import downsample

KEY = "0123456789abcdef/IMG_1.jpg"


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs")
    original = tmp_path / "trip" / "IMG_1.jpg"
    original.parent.mkdir()
    Image.new("RGB", (4032, 3024), "gray").save(original)
    s.inputs.put_bytes(KEY, downsample(original))
    s.write_input_manifest("j1", [KEY], {KEY: str(original)})
    s.original = str(original)
    return s


class ut_PathMap:
    @pytest.mark.parametrize("staged", [
        "/inputs/0123456789abcdef/IMG_1.jpg",                                        # the L4
        "C:\\Users\\Jonathan Hexner\\.sim_bench\\jobs\\_inputs\\0123456789abcdef\\IMG_1.jpg",  # this PC
        "C:/Users/Jonathan Hexner/.sim_bench/jobs/_inputs/0123456789abcdef/IMG_1.jpg",
    ])
    def test_a_staged_path_from_either_machine_finds_its_original(self, store, staged):
        assert PathMap.from_job(store, "j1").original(staged) == store.original

    def test_the_face_key_keeps_its_suffix(self, store):
        assert PathMap.from_job(store, "j1").original(f"/inputs/{KEY}:face_2") == store.original + ":face_2"

    def test_anything_else_comes_back_unchanged(self, store):
        m = PathMap.from_job(store, "j1")
        for text in ("Best in cluster", "/inputs/ffffffffffffffff/other.jpg", "D:/x/IMG_1.jpg", ""):
            assert m.original(text) == text

    def test_scale_is_original_over_staged_pixels(self, store):
        assert PathMap.from_job(store, "j1").scale(f"/inputs/{KEY}") == pytest.approx(4032 / 768)

    def test_a_moved_original_keeps_its_path_and_scale_one(self, store, tmp_path):
        (tmp_path / "trip" / "IMG_1.jpg").unlink()
        m = PathMap.from_job(store, "j1")
        assert m.original(f"/inputs/{KEY}") == store.original and m.scale(f"/inputs/{KEY}") == 1.0

    def test_a_job_from_before_this_spec_has_an_empty_map(self, tmp_path):
        s = JobStore(tmp_path / "jobs")
        s.inputs.put_bytes(KEY, b"x")
        s.write_input_manifest("old", [KEY])
        assert PathMap.from_job(s, "old") == PathMap()


class ut_Remap:
    def _output(self):
        path = f"/inputs/{KEY}"
        return RunOutput(
            image_metrics={path: {"path": path, "filter_scores": [
                {"face_index": 0, "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "x_px": 76.8, "y_px": 115.2,
                                           "w_px": 230.4, "h_px": 230.4}}]}},
            scene_clusters={"0": [path]},
            selected_images=[path],
            people=[{"person_index": 0, "thumbnail_image_path": path, "thumbnail_face_index": 0,
                     "thumbnail_bbox": [0.1, 0.2, 0.3, 0.4],
                     "face_instances": [{"image_path": path, "face_index": 0, "bbox": [76.8, 115.2, 230.4, 230.4]}]}],
            step_decisions=[{"item_id": f"{path}:face_0", "item_type": "face", "reason": "kept"}],
            siamese_comparisons=[{"img1": path, "img2": path}],
        )

    def test_every_path_is_the_original(self, store):
        out = self._output().remap(PathMap.from_job(store, "j1"))
        assert "/inputs/" not in out.to_json() and "0123456789abcdef" not in out.to_json()
        assert list(out.image_metrics) == [store.original] and out.selected_images == [store.original]
        assert out.scene_clusters == {"0": [store.original]}
        assert out.step_decisions[0]["item_id"] == store.original + ":face_0"
        assert out.siamese_comparisons[0]["img2"] == store.original

    def test_pixel_boxes_scale_and_ratio_boxes_stay(self, store):
        out = self._output().remap(PathMap.from_job(store, "j1"))
        box = out.image_metrics[store.original]["filter_scores"][0]["bbox"]
        assert box["x_px"] == pytest.approx(403.2) and box["w_px"] == pytest.approx(1209.6)
        assert box["x"] == 0.1 and box["w"] == 0.3
        assert out.people[0]["thumbnail_bbox"] == [0.1, 0.2, 0.3, 0.4]
        assert out.people[0]["face_instances"][0]["bbox"] == pytest.approx([403.2, 604.8, 1209.6, 1209.6])

    def test_a_worker_crop_thumbnail_falls_back_to_a_face_in_its_photo(self, store):
        output = self._output()
        output.people[0].update(thumbnail_image_path="/root/.sim_bench/crops/a.jpg", thumbnail_bbox=None)
        person = output.remap(PathMap.from_job(store, "j1")).people[0]
        assert person["thumbnail_image_path"] == store.original and person["thumbnail_bbox"]
