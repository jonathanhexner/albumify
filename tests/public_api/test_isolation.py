"""spec-120 R4 (a user reaches only their own albums) and R11 (every public read takes at most two queries).

Job tables and API tables in one database, as on Neon (spec-117 R2)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import Album, Base, ImageMetricRow, Person, PipelineResult, PipelineRun
from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import JobRepo
from sim_bench.control.models import Job
from sim_bench.public_api.album_repo import ALL, BEST, BEST_N, PublicAlbumRepo
from tests._db import fresh_database_url

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


@pytest.fixture
def world(tmp_path):
    db = ControlDB(fresh_database_url(tmp_path, "one.db"))
    Base.metadata.create_all(db.engine)
    sessions = sessionmaker(bind=db.engine, expire_on_commit=False)

    def album(job_id, owner, n_photos, n_picked, when):
        JobRepo(db).create(job_id, runner="modal_gpu", source=None)
        with db.session() as s:
            job = s.get(Job, job_id)
            job.owner_id, job.n_inputs, job.created_at = owner, n_photos, when
        with sessions() as s:
            s.add(Album(id=f"alb-{job_id}", name=f"Trip {job_id}", source_path="upload"))
            s.add(PipelineRun(id=job_id, album_id=f"alb-{job_id}", status="completed"))
            s.add(PipelineResult(id=f"res-{job_id}", run_id=job_id, total_images=n_photos, num_selected=n_picked))
            s.flush()                                # metric rows have no ORM link to the run: insert it first
            for i in range(n_photos):
                s.add(ImageMetricRow(run_id=job_id, image_path=f"{job_id}/p{i:03d}.jpg", composite_score=i / n_photos,
                                     is_selected=i >= n_photos - n_picked,
                                     filter_reason=None if i >= n_photos - n_picked else "near-duplicate",
                                     cluster_id=i % 5))
            for k in range(3):
                s.add(Person(id=f"{job_id}-p{k}", album_id=f"alb-{job_id}", run_id=job_id, person_index=k,
                             face_count=10 - k, image_count=5 - k, face_instances=[{"x": 1}] * 50))
            s.commit()

    album("a1", "ann", 60, 40, T0)
    album("a2", "ann", 10, 4, T0 + timedelta(days=1))
    album("b1", "bob", 20, 8, T0)
    album("own", None, 5, 2, T0)                          # the owner's own run: nobody's on the public site
    return PublicAlbumRepo(sessions), db.engine


class ut_Isolation:
    def test_each_user_lists_only_their_albums_newest_first(self, world):
        repo, _ = world
        assert [a.job_id for a in repo.list("ann")] == ["a2", "a1"]
        assert [a.job_id for a in repo.list("bob")] == ["b1"]
        assert repo.list("carol") == []

    @pytest.mark.parametrize("call", ["album", "photos_best", "photos_all", "people"])
    def test_another_users_album_is_not_found(self, world, call):
        repo, _ = world
        do = {"album": lambda u, j: repo.album(u, j), "photos_best": lambda u, j: repo.photos(u, j, BEST),
              "photos_all": lambda u, j: repo.photos(u, j, ALL), "people": lambda u, j: repo.people(u, j)}[call]
        assert do("ann", "a1")                               # own: found
        assert do("bob", "a1") is None                       # someone else's
        assert do("ann", "own") is None                      # the owner's runs are nobody's here
        assert do("ann", "missing") is None

    def test_rename_reaches_only_your_own_person(self, world):
        repo, _ = world
        assert repo.rename_person("bob", "a1", "a1-p0", "Mallory") is None
        assert repo.rename_person("ann", "b1", "a1-p0", "Wrong album") is None
        renamed = repo.rename_person("ann", "a1", "a1-p0", "  Grandma   Rose ")
        assert renamed.name == "Grandma Rose"
        assert repo.people("ann", "a1")[0].name == "Grandma Rose"
        with pytest.raises(ValueError):
            repo.rename_person("ann", "a1", "a1-p0", "   ")
        with pytest.raises(ValueError):
            repo.rename_person("ann", "a1", "a1-p0", "x" * 81)


class ut_Reads:
    def test_best_is_the_top_30_picks_by_score(self, world):
        repo, _ = world
        best = repo.photos("ann", "a1", BEST)
        assert len(best) == BEST_N and all(p.picked for p in best)
        assert [p.score for p in best] == sorted((p.score for p in best), reverse=True)

    def test_all_lists_picks_first_then_the_rest_with_their_reason(self, world):
        repo, _ = world
        photos = repo.photos("ann", "a1", ALL)
        assert len(photos) == 60
        assert [p.picked for p in photos] == [True] * 40 + [False] * 20
        assert {p.reason for p in photos if not p.picked} == {"near-duplicate"}
        assert all(p.reason is None for p in photos if p.picked)

    def test_album_counts_and_people_order(self, world):
        repo, _ = world
        a = repo.album("ann", "a1")
        assert (a.title, a.n_photos, a.n_picked) == ("Trip a1", 60, 40)
        assert [p.face_count for p in repo.people("ann", "a1")] == [10, 9, 8]

    def test_unknown_tier_is_refused(self, world):
        with pytest.raises(ValueError):
            world[0].photos("ann", "a1", "top")


class ut_QueryCounts:
    """R11: on Neon each query is a ~0.16 s round trip."""

    @pytest.fixture
    def count(self, world):
        _, engine = world
        seen = []

        def on(conn, cursor, statement, *a):
            if statement.lstrip().upper().startswith(("SELECT", "UPDATE", "INSERT", "DELETE")):
                seen.append(statement)
        event.listen(engine, "before_cursor_execute", on)
        yield seen
        event.remove(engine, "before_cursor_execute", on)

    @pytest.mark.parametrize("call,limit", [("list", 1), ("album", 1), ("photos_best", 2), ("photos_all", 2),
                                            ("people", 2), ("rename", 2)])
    def test_at_most_two_queries(self, world, count, call, limit):
        repo, _ = world
        {"list": lambda: repo.list("ann"), "album": lambda: repo.album("ann", "a1"),
         "photos_best": lambda: repo.photos("ann", "a1", BEST), "photos_all": lambda: repo.photos("ann", "a1", ALL),
         "people": lambda: repo.people("ann", "a1"),
         "rename": lambda: repo.rename_person("ann", "a1", "a1-p1", "Leo")}[call]()
        assert len(count) <= limit, count
