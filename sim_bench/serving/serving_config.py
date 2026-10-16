"""Load ``configs/serving.yaml`` + env overrides (spec-112 TA4.1)."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import yaml

__all__ = ["ServingConfig", "RUNNERS", "load_serving_config"]

RUNNERS = ("local", "modal_gpu", "modal_cpu")
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "serving.yaml"
_ENV = {"runner": "ALBUMIFY_RUNNER", "pipeline": "ALBUMIFY_PIPELINE",
        "device": "ALBUMIFY_DEVICE", "jobs_root": "ALBUMIFY_JOBS_ROOT"}


@dataclass(frozen=True, slots=True)
class ServingConfig:
    runner: str = "local"
    pipeline: str = "default"
    device: str = "cpu"
    jobs_root: Path = Path.home() / ".sim_bench" / "jobs"

    def __post_init__(self):
        if self.runner not in RUNNERS:
            raise ValueError(f"runner must be one of {RUNNERS}, got {self.runner!r}")

    def with_runner(self, runner: Optional[str]) -> "ServingConfig":
        return replace(self, runner=runner) if runner else self


def load_serving_config(path: Optional[Path] = None) -> ServingConfig:
    path = Path(path) if path else _DEFAULT_PATH
    raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}
    for key, env in _ENV.items():
        if os.environ.get(env):
            raw[key] = os.environ[env]
    if "jobs_root" in raw:
        raw["jobs_root"] = Path(os.path.expanduser(str(raw["jobs_root"])))
    unknown = set(raw) - set(_ENV)
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    return ServingConfig(**raw)
