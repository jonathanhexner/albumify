"""T009 — Tests for face_cluster.run_naming."""
from pathlib import Path

from face_cluster.run_naming import RunDirSpec, allocate_run_dir


def test_sequential_calls_increment(tmp_path):
    """First call → kind_1, second call → kind_2."""
    db = tmp_path / "test.db"
    spec = RunDirSpec(source_album="Noa2_5", kind="remerge")
    root = tmp_path / "results"

    p1 = allocate_run_dir(spec, root, db_path=db)
    p2 = allocate_run_dir(spec, root, db_path=db)

    assert p1.name == "remerge_1"
    assert p2.name == "remerge_2"
    assert p1 != p2


def test_existing_directory_causes_skip(tmp_path):
    """If remerge_1 already exists on disk, allocation must return remerge_2."""
    db = tmp_path / "test.db"
    spec = RunDirSpec(source_album="Noa2_5", kind="remerge")
    root = tmp_path / "results"
    (root / "Noa2_5" / "remerge_1").mkdir(parents=True)

    p = allocate_run_dir(spec, root, db_path=db)
    assert p.name == "remerge_2"


def test_rapid_calls_return_different_paths(tmp_path):
    """Rapid successive allocations must all be unique."""
    db = tmp_path / "test.db"
    spec = RunDirSpec(source_album="album_a", kind="recluster")
    root = tmp_path / "results"

    paths = [allocate_run_dir(spec, root, db_path=db) for _ in range(5)]
    assert len(set(paths)) == 5, "All 5 allocations must be distinct"


def test_different_albums_independent(tmp_path):
    """Two different source_albums must not interfere with each other's counters."""
    db = tmp_path / "test.db"
    root = tmp_path / "results"

    p_a = allocate_run_dir(RunDirSpec("albumA", "remerge"), root, db_path=db)
    p_b = allocate_run_dir(RunDirSpec("albumB", "remerge"), root, db_path=db)

    assert p_a.name == "remerge_1"
    assert p_b.name == "remerge_1"
    assert p_a.parent.name == "albumA"
    assert p_b.parent.name == "albumB"
