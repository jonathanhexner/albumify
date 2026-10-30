"""What a run ran with (spec-115 W4, R12): written into ``run.json`` by the worker, shown and compared in the app.

The worker records it (it is the only side that knows its own library versions); the control plane passes the
code commit through ``ALBUMIFY_COMMIT``, because a Modal container has no git checkout.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["RunSettings", "COMMIT_ENV", "PACKAGES", "current_commit"]

COMMIT_ENV = "ALBUMIFY_COMMIT"
PACKAGES = ("torch", "onnxruntime-gpu", "onnxruntime", "insightface", "numpy", "transformers", "pillow")
_REPO = Path(__file__).resolve().parents[2]


def current_commit() -> Optional[str]:
    """This checkout's commit (control plane side), or the one handed to a worker via the env."""
    if os.environ.get(COMMIT_ENV):
        return os.environ[COMMIT_ENV]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> Dict[str, str]:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


@dataclass
class RunSettings:
    pipeline: str
    steps: List[Dict[str, Any]] = field(default_factory=list)   # [{"step": name, "params": {...}}] in run order
    commit: Optional[str] = None
    runner: str = "local"
    device: str = "cpu"
    packages: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def collect(cls, plan, *, pipeline: str, runner: str, device: str) -> "RunSettings":
        """``plan`` is the resolved :class:`~sim_bench.serving.pipeline_runner.SpecResult`."""
        steps = [{"step": s, "params": dict(plan.step_configs.get(s) or {})} for s in plan.steps]
        return cls(pipeline=pipeline, steps=steps, commit=current_commit(), runner=runner, device=device,
                   packages=_versions())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["RunSettings"]:
        if not d:
            return None
        return cls(**{k: d[k] for k in ("pipeline", "steps", "commit", "runner", "device", "packages") if k in d})

    def diff(self, other: "RunSettings") -> Dict[str, Any]:
        """What differs, as ``{name: [mine, theirs]}``; step parameters as ``step.param``; steps only one run
        has as ``steps``."""
        out: Dict[str, Any] = {}
        for name in ("pipeline", "commit", "runner", "device"):
            if getattr(self, name) != getattr(other, name):
                out[name] = [getattr(self, name), getattr(other, name)]
        for pkg in sorted(set(self.packages) | set(other.packages)):
            a, b = self.packages.get(pkg), other.packages.get(pkg)
            if a != b:
                out[f"package {pkg}"] = [a, b]
        mine = {s["step"]: s["params"] for s in self.steps}
        theirs = {s["step"]: s["params"] for s in other.steps}
        if [s["step"] for s in self.steps] != [s["step"] for s in other.steps]:
            only = sorted(set(mine) ^ set(theirs))
            out["steps"] = [sorted(set(mine) - set(theirs)), sorted(set(theirs) - set(mine))] if only \
                else ["same steps, different order", ""]
        for step in sorted(set(mine) & set(theirs)):
            for key in sorted(set(mine[step]) | set(theirs[step])):
                a, b = mine[step].get(key), theirs[step].get(key)
                if a != b:
                    out[f"{step}.{key}"] = [a, b]
        return out
