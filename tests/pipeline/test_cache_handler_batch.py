"""spec-117 R12: one feature cache for this PC and the L4. Reads and writes are one query per step, and a staged
photo has the same key on both machines (content, not folder), so a run on one reuses the other's work."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import UniversalCache
from sim_bench.pipeline.cache_handler import CacheKey, UniversalCacheHandler

from tests._db import fresh_api_engine

N = 715          # Slovakia26


@pytest.fixture
def engine(tmp_path):
    return fresh_api_engine(tmp_path)


def _count_cache_queries(engine):
    seen = {"select": 0, "insert": 0, "commit": 0}

    def before(conn, cursor, statement, params, context, executemany):
        sql = statement.lstrip().upper()
        if "UNIVERSAL_CACHE" in sql:
            if sql.startswith("SELECT"):
                seen["select"] += 1
            elif sql.startswith("INSERT"):
                seen["insert"] += 1

    event.listen(engine, "before_cursor_execute", before)
    event.listen(engine, "commit", lambda conn: seen.__setitem__("commit", seen["commit"] + 1))
    return seen


def _staged(root: Path, n: int):
    """``n`` photos in an input store under ``root``: <sha16>/<name>.jpg."""
    paths = []
    for i in range(n):
        p = root / f"{i:016x}" / f"IMG_{i:04d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        paths.append(str(p))
    return paths


def _keys(paths, feature="insightface_faces"):
    return [CacheKey(image_path=p, feature_type=feature, model_name="buffalo_l") for p in paths]


class ut_CacheBatch:
    def test_a_step_reads_with_one_query_not_one_per_photo(self, engine, tmp_path):
        paths = _staged(tmp_path / "inputs", N)
        handler = UniversalCacheHandler(sessionmaker(bind=engine)(), inputs_root=str(tmp_path / "inputs"))
        handler.store_many([(k, b"d") for k in _keys(paths)], {"model_version": "v2"})
        seen = _count_cache_queries(engine)
        got = handler.load_many(_keys(paths))
        assert len(got) == N
        assert seen["select"] == 1                  # 715 photos (+ their older path keys), one query

    def test_a_read_leaves_no_transaction_open(self, engine, tmp_path):
        """The step computes for minutes after its cache read; Postgres (Neon) ends a connection idle in a
        transaction, which failed triage_content on the L4 (spec-117 T8.2, 8.5 min step)."""
        paths = _staged(tmp_path / "inputs", 5)
        session = sessionmaker(bind=engine)()
        handler = UniversalCacheHandler(session, inputs_root=str(tmp_path / "inputs"))
        handler.store_many([(k, b"d") for k in _keys(paths[:3])], {})
        got = handler.load_many(_keys(paths))                  # 3 hits, 2 misses, nothing stale
        assert len(got) == 3
        assert not session.in_transaction()

    def test_a_step_writes_in_one_commit(self, engine, tmp_path):
        paths = _staged(tmp_path / "inputs", 300)
        handler = UniversalCacheHandler(sessionmaker(bind=engine)(), inputs_root=str(tmp_path / "inputs"))
        seen = _count_cache_queries(engine)
        assert handler.store_many([(k, b"d") for k in _keys(paths)], {}) == 300
        assert seen["commit"] == 1 and seen["select"] == 1

    def test_the_l4_reuses_what_this_pc_cached(self, engine, tmp_path):
        """Same photos, different input folders (~/.sim_bench/jobs/_inputs here, /inputs on the L4), different
        mtimes (an upload makes a new file): every key hits."""
        pc = _staged(tmp_path / "pc_inputs", 20)
        UniversalCacheHandler(sessionmaker(bind=engine)(), inputs_root=str(tmp_path / "pc_inputs")).store_many(
            [(k, f"emb-{i}".encode()) for i, k in enumerate(_keys(pc))], {"model_version": "v2"})
        time.sleep(0.05)
        l4 = _staged(tmp_path / "l4_inputs", 20)          # written later: other mtimes
        got = UniversalCacheHandler(sessionmaker(bind=engine)(), inputs_root=str(tmp_path / "l4_inputs")).load_many(
            _keys(l4))
        assert len(got) == 20
        key = f"{os.path.normpath(l4[3])}:insightface_faces:buffalo_l"
        assert got[key][0] == b"emb-3" and got[key][1]["model_version"] == "v2"
        with sessionmaker(bind=engine)() as s:
            stored = {r.image_path for r in s.query(UniversalCache).all()}
        assert all(p.startswith("staged:") for p in stored) and "staged:0000000000000003/IMG_0003.jpg" in stored

    def test_face_suffix_keys_are_content_keys_too(self, engine, tmp_path):
        p = _staged(tmp_path / "inputs", 1)[0]
        h = UniversalCacheHandler(sessionmaker(bind=engine)(), inputs_root=str(tmp_path / "inputs"))
        assert h.db_key(os.path.normpath(p) + ":face_2") == ("staged:0000000000000000/IMG_0000.jpg:face_2", True)

    def test_an_older_run_s_path_keyed_row_still_hits(self, engine, tmp_path):
        """Rows written before spec-117 (full staged path, real mtime) are found by the same query."""
        paths = _staged(tmp_path / "inputs", 3)
        UniversalCacheHandler(sessionmaker(bind=engine)()).store_many([(k, b"old") for k in _keys(paths)], {})
        got = UniversalCacheHandler(sessionmaker(bind=engine)(), inputs_root=str(tmp_path / "inputs")).load_many(
            _keys(paths))
        assert sorted(v[0] for v in got.values()) == [b"old"] * 3

    def test_a_photo_outside_the_input_store_keeps_its_mtime_check(self, engine, tmp_path):
        photo = tmp_path / "trip" / "a.jpg"
        photo.parent.mkdir()
        photo.write_bytes(b"x")
        h = UniversalCacheHandler(sessionmaker(bind=engine)(), inputs_root=str(tmp_path / "inputs"))
        h.store_many([(k, b"d") for k in _keys([str(photo)])], {})
        assert len(h.load_many(_keys([str(photo)]))) == 1
        os.utime(photo, (time.time() + 10, time.time() + 10))     # the photo changed
        assert h.load_many(_keys([str(photo)])) == {}
        with sessionmaker(bind=engine)() as s:
            assert s.query(UniversalCache).count() == 0            # invalidated row deleted

    def test_a_rewrite_updates_in_place(self, engine, tmp_path):
        paths = _staged(tmp_path / "inputs", 2)
        h = UniversalCacheHandler(sessionmaker(bind=engine)(), inputs_root=str(tmp_path / "inputs"))
        h.store_many([(k, b"v1") for k in _keys(paths)], {"model_version": "v1"})
        h.store_many([(k, b"v2") for k in _keys(paths)], {"model_version": "v2"})
        with sessionmaker(bind=engine)() as s:
            rows = s.query(UniversalCache).all()
        assert len(rows) == 2 and {r.data_blob for r in rows} == {b"v2"} and {r.model_version for r in rows} == {"v2"}


def test_the_worker_cache_follows_database_url(monkeypatch, tmp_path):
    """R12: with DATABASE_URL set the pipeline's cache is that database; unset it is sim_bench.db as before."""
    from sim_bench.serving.curate_job import FEATURE_CACHE_ENV, feature_cache_config

    monkeypatch.delenv(FEATURE_CACHE_ENV, raising=False)
    monkeypatch.setenv("SIM_BENCH_DB", str(tmp_path / "sim_bench.db"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert feature_cache_config().url == f"sqlite:///{(tmp_path / 'sim_bench.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@ep-x.neon.tech/neondb")
    assert feature_cache_config().url == "postgresql://u:p@ep-x.neon.tech/neondb"
    monkeypatch.setenv(FEATURE_CACHE_ENV, str(tmp_path / "isolated.db"))
    assert feature_cache_config().url.endswith("isolated.db")
