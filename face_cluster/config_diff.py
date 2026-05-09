"""Compute a shallow diff between two pipeline config dicts.

Usage:
    from face_cluster.config_diff import compute, ConfigDelta
    deltas = compute(parent_config, child_config)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfigDelta:
    field: str
    parent_value: Any
    child_value: Any


def compute(parent: dict, child: dict) -> list[ConfigDelta]:
    """Return entries that differ between parent and child configs.

    Only keys present in *either* dict are compared.
    A key present in one but absent in the other is treated as None on the
    missing side — not as a change if both resolve to the same effective value.
    """
    all_keys = set(parent) | set(child)
    return [
        ConfigDelta(k, parent.get(k), child.get(k))
        for k in sorted(all_keys)
        if parent.get(k) != child.get(k)
    ]
