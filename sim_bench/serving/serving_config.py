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
_NUMBERS = ("max_photos_l4", "max_gpu_minutes", "l4_fixed_seconds", "l4_seconds_per_photo",
            "l4_dollars_per_hour")
_ENV = {"runner": "ALBUMIFY_RUNNER", "pipeline": "ALBUMIFY_PIPELINE",
        "device": "ALBUMIFY_DEVICE", "jobs_root": "ALBUMIFY_JOBS_ROOT"}


@dataclass(frozen=True, slots=True)
class ServingConfig:
    runner: str = "local"
    pipeline: str = "default"
    device: str = "cpu"
    jobs_root: Path = Path.home() / ".sim_bench" / "jobs"
    # spec-115 R2: L4 estimate and spend cap. Rate from the 2026-10-11 Slovakia26 run (411 s for 715
    # photos + container start and model load).
    max_photos_l4: int = 1000
    max_gpu_minutes: float = 30.0
    l4_fixed_seconds: float = 150.0
    l4_seconds_per_photo: float = 0.6
    l4_dollars_per_hour: float = 0.80

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
    unknown = set(raw) - set(_ENV) - set(_NUMBERS)
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    return ServingConfig(**raw)
