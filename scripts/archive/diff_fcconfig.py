"""Diff the FCConfig that each app builds from the SAME profile.

FC v2:    FCParams.load(profile).to_fc_config()           (1:1, all knobs)
Albumify: build_step_configs() overlay -> bridge.build_fc_config()  (lossy)

No pipeline run needed — pure config construction. Prints every field where
the two FCConfigs disagree, i.e. the exact reason the cluster output differs.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "scripts"))
sys.path.insert(0, str(_repo / "tests"))

import _budapest_baseline as anchor  # noqa: E402
import capture_albumify_baseline as cap  # noqa: E402
from face_cluster.fc_params import FCParams  # noqa: E402
from sim_bench.pipeline.steps.face_cluster_bridge import build_fc_config  # noqa: E402

SKIP = {"stages", "source_dir", "output_dir", "on_progress"}


def main() -> int:
    # Optional argv[1] = profile path override (default: the anchor profile).
    prof = Path(sys.argv[1]) if len(sys.argv) > 1 else anchor.PROFILE_PATH
    fc_v2 = FCParams.load(prof).to_fc_config()
    alb_step = cap.build_step_configs(profile_path=prof)["cluster_people"]
    alb = build_fc_config(alb_step)

    names = [f.name for f in dataclasses.fields(fc_v2) if f.name not in SKIP]
    print(f"profile: {prof.name}\n")
    print(f"{'FIELD':<36}{'FC v2':<14}{'Albumify':<14}")
    print("-" * 64)
    ndiff = 0
    for name in names:
        a = getattr(fc_v2, name)
        b = getattr(alb, name, "<MISSING>")
        if a != b:
            ndiff += 1
            print(f"{name:<36}{str(a):<14}{str(b):<14}  <-- DIFF")
    print("-" * 64)
    print(f"{ndiff} differing fields of {len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
