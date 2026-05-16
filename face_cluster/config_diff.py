"""Compute a shallow diff between two pipeline config dicts.

Usage (existing API):
    from face_cluster.config_diff import compute, ConfigDelta
    deltas = compute(parent_config, child_config)

spec-033 P-F: canonical "effective config" representation lets FC App and
Albumify runs be compared directly even though their configs originate in
different shapes (PipelineConfig dataclass vs. nested step_configs dict).

CLI:
    python -m face_cluster.config_diff <run_dir_a> <run_dir_b>
"""
from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


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


# ---------------------------------------------------------------------------
# spec-033 P-F: effective-config canonicalization
# ---------------------------------------------------------------------------

# Fields that should not participate in diffs (paths, timestamps, callbacks).
# Two runs with different output_dir are not "different configs."
_NOISE_FIELDS = frozenset({
    "source_dir", "output_dir", "stages", "on_progress",
})


def effective_config_from_fc_config(cfg) -> Dict[str, Any]:
    """Return the canonical effective-config dict from an FC App PipelineConfig.

    Drops ``_NOISE_FIELDS`` so a diff focuses on knobs that actually affect
    the algorithm, not the run wiring.
    """
    raw = dataclasses.asdict(cfg)
    return {k: v for k, v in raw.items() if k not in _NOISE_FIELDS}


def effective_config_from_albumify(step_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return the same canonical shape from an Albumify step_configs nested dict.

    Reads the ``cluster_people`` slice (where face_cluster_knn parameters
    live) and constructs a PipelineConfig — defaults from the dataclass fill
    in for missing keys. Aligns with the FC App side so ``compute(...)``
    against two effective configs is meaningful.
    """
    from face_cluster.config import PipelineConfig

    cluster_people = step_configs.get("cluster_people", {}) or {}
    known = {f.name for f in dataclasses.fields(PipelineConfig)}
    overrides = {k: v for k, v in cluster_people.items() if k in known}
    cfg = PipelineConfig(**overrides)
    return effective_config_from_fc_config(cfg)


def _load_effective_from_run_dir(run_dir: Path) -> Dict[str, Any]:
    """Load the effective config from a run dir's ``pipeline_run.json``."""
    from face_cluster.config import PipelineConfig

    run_json = run_dir / "pipeline_run.json"
    if not run_json.exists():
        raise FileNotFoundError(f"No pipeline_run.json in {run_dir}")
    data = json.loads(run_json.read_text(encoding="utf-8"))
    cfg = data.get("config") or {}
    # Heuristic: if the config dict has a "cluster_people" sub-key it's an
    # Albumify-shaped step_configs payload; otherwise treat it as a flat
    # PipelineConfig serialization.
    if isinstance(cfg, dict) and isinstance(cfg.get("cluster_people"), dict):
        return effective_config_from_albumify(cfg)
    known = {f.name for f in dataclasses.fields(PipelineConfig)}
    overrides = {k: v for k, v in (cfg or {}).items() if k in known}
    return effective_config_from_fc_config(PipelineConfig(**overrides))


def main(argv: list[str] | None = None) -> int:
    """CLI: print field-level differences between two run dirs.

    Returns 0 on identical, 1 on differing, 2 on usage error.
    """
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print("Usage: python -m face_cluster.config_diff <run_dir_a> <run_dir_b>",
              file=sys.stderr)
        return 2
    a_eff = _load_effective_from_run_dir(Path(argv[0]))
    b_eff = _load_effective_from_run_dir(Path(argv[1]))
    deltas = compute(a_eff, b_eff)
    if not deltas:
        print("IDENTICAL — no field differences in the effective config.")
        return 0
    print(f"{len(deltas)} field(s) differ:")
    for d in deltas:
        print(f"  {d.field}: {d.parent_value!r} -> {d.child_value!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
