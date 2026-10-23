"""spec-114 - JobComparer: two finished jobs on the same photos, and what each did with every photo."""
from __future__ import annotations

import pytest
from PIL import Image

from sim_bench.control.album_builder import record_edit
from sim_bench.control.compare import CompareJobsInputs, DifferentPhotos, JobComparer
from sim_bench.serving.job_service import SubmitInputs

from tests.control.fakes import NAMES
from tests.control.test_job_service import _in_process, _service


@pytest.fixture
def svc(tmp_path):
    svc = _service(tmp_path, _in_process)
    src = tmp_path / "trip"
    src.mkdir()
    for n in NAMES:
        Image.new("RGB", (900, 600), "gray").save(src / n)
    for jid in ("A", "B"):
        assert svc.submit(SubmitInputs(source=src, job_id=jid)).status == "succeeded"
    svc.src = src
    return svc


def _compare(svc, a="A", b="B"):
    return JobComparer(svc.db).calc(CompareJobsInputs(a=a, b=b))


class ut_JobComparer:
    def test_two_runs_of_the_same_photos_match(self, svc):
        res = _compare(svc)
        assert res.result.ok and res.result.picks.same == ["a", "c"]
        assert res.result.picks.missing == res.result.picks.extra == []

    def test_every_photo_has_a_decision_in_both_jobs(self, svc):
        d = _compare(svc).decisions
        assert set(d["A"]) == set(d["B"]) == {n.split(".")[0] for n in NAMES}
        a = d["A"]
        assert a["a"]["state"] == "pick" and "pick #" in a["a"]["detail"]
        assert a["b"]["state"] == "near_dup" and "near-duplicate of a" in a["b"]["detail"]
        assert a["q"]["state"] == "removed" and "filter_quality" in a["q"]["detail"]
        assert a["s"]["state"] == "set_aside"

    def test_an_owner_edit_in_b_shows_as_a_moved_pick(self, svc):
        """The comparison uses the album as the DB builds it, owner edits included."""
        record_edit(svc.db, "B", "demote", {"photo_id": "c"})
        res = _compare(svc)
        assert res.result.picks.missing == ["c"] and res.decisions["B"]["c"]["state"] != "pick"

    def test_people_largest_first_with_a_face_to_show(self, svc):
        people = _compare(svc).people["B"]
        assert [p["face_count"] for p in people] == sorted((p["face_count"] for p in people), reverse=True)
        assert all(p["face_id"] is not None for p in people)

    def test_jobs_on_different_photos_are_refused(self, svc):
        from sim_bench.control.models import Photo
        svc.submit(SubmitInputs(source=svc.src, job_id="C"))
        with svc.db.session() as s:          # the fake pipeline always reports the same six photos
            s.delete(s.get(Photo, ("C", "s")))
        with pytest.raises(DifferentPhotos, match="same photos"):
            _compare(svc, b="C")

    def test_result_serialises_for_the_api(self, svc):
        out = _compare(svc).to_dict()
        assert out["a"] == "A" and out["ok"] is True and out["checks"][0]["name"] == "uploaded"
        assert set(out["picks"]) == {"same", "swapped", "missing", "extra", "also_kept"}
