"""T011 — Tests for face_cluster.config_diff."""
from face_cluster.config_diff import compute, ConfigDelta


def test_identical_configs_produce_empty_list():
    cfg = {"distance_threshold": 0.35, "min_cluster_size": 3}
    assert compute(cfg, cfg) == []


def test_one_changed_field():
    parent = {"distance_threshold": 0.35}
    child  = {"distance_threshold": 0.50}
    result = compute(parent, child)
    assert result == [ConfigDelta("distance_threshold", 0.35, 0.50)]


def test_key_only_in_child():
    parent = {"a": 1}
    child  = {"a": 1, "b": 99}
    result = compute(parent, child)
    assert result == [ConfigDelta("b", None, 99)]


def test_key_only_in_parent():
    parent = {"a": 1, "b": 99}
    child  = {"a": 1}
    result = compute(parent, child)
    assert result == [ConfigDelta("b", 99, None)]


def test_multiple_changes():
    parent = {"x": 1, "y": 2, "z": 3}
    child  = {"x": 1, "y": 99, "z": 0}
    result = compute(parent, child)
    fields = {d.field for d in result}
    assert fields == {"y", "z"}


def test_empty_dicts():
    assert compute({}, {}) == []


def test_both_none_values_not_reported():
    """If both sides are absent (None), it should NOT appear as a delta."""
    parent = {"a": 1}
    child  = {"a": 1}
    # Neither dict has key "b" — effectively None == None, no delta
    result = compute(parent, child)
    assert all(d.field != "b" for d in result)
