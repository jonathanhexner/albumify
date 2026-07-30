"""spec-079 Stage 0c — why does the producer chain feed different faces?

Diffs the two producer-output face sets (FC v2 run dir vs Albumify _v4 export)
to pin what drives the core-set gap (133 vs 186). Reports, per image: faces
detected, faces with embeddings, and faces selected as core by the SAME quality
gate (profile_5_nogates). Then shows where the two sets disagree.

Run:  .venv/Scripts/python scripts/diff_face_sets.py <fc_v2_run_dir> <albumify_v4_dir>
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from face_cluster.fc_params import FCParams  # noqa: E402
from face_cluster.quality import QualityGater  # noqa: E402
from sim_bench.run_db.store import RunStore  # noqa: E402

PROFILE = REPO / "specs" / "079-albumify-shared-core" / "profile_5_nogates.json"
DEFAULT_FC = Path.home() / ".sim_bench" / "runs" / "2f8d1db8057d432eb12bb720402a29ff"


def _name(p: str) -> str:
    return Path(p).name


def analyze(run_dir: Path, fc_cfg) -> dict:
    faces = RunStore(run_dir).faces()
    core, _holdout, _verdicts = QualityGater(fc_cfg).select_core_set(faces)
    core_set = set(core)
    per_img_faces = defaultdict(int)
    per_img_core = defaultdict(int)
    n_emb = 0
    for i, f in enumerate(faces):
        per_img_faces[_name(f.image_path)] += 1
        if getattr(f, "embedding_normalized", None) is not None:
            n_emb += 1
        if i in core_set:
            per_img_core[_name(f.image_path)] += 1
    return {
        "n_faces": len(faces), "n_emb": n_emb, "n_core": len(core),
        "per_img_faces": dict(per_img_faces), "per_img_core": dict(per_img_core),
    }


def main() -> int:
    fc_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FC
    alb_dir = Path(sys.argv[2])
    fc_cfg = FCParams.load(PROFILE).to_fc_config()

    fc = analyze(fc_dir, fc_cfg)
    alb = analyze(alb_dir, fc_cfg)

    print(f"{'':<16}{'FC v2':>10}{'Albumify':>10}")
    print("-" * 36)
    print(f"{'faces':<16}{fc['n_faces']:>10}{alb['n_faces']:>10}")
    print(f"{'with embedding':<16}{fc['n_emb']:>10}{alb['n_emb']:>10}")
    print(f"{'core (gated)':<16}{fc['n_core']:>10}{alb['n_core']:>10}")
    print(f"{'images w/ faces':<16}{len(fc['per_img_faces']):>10}{len(alb['per_img_faces']):>10}")

    all_imgs = set(fc["per_img_faces"]) | set(alb["per_img_faces"])
    only_fc = set(fc["per_img_faces"]) - set(alb["per_img_faces"])
    only_alb = set(alb["per_img_faces"]) - set(fc["per_img_faces"])
    print(f"\nimages only in FC v2 : {len(only_fc)}")
    print(f"images only in Albumify: {len(only_alb)}")

    # Where does the core difference come from, per image?
    rows = []
    for img in all_imgs:
        fcc = fc["per_img_core"].get(img, 0)
        alc = alb["per_img_core"].get(img, 0)
        if fcc != alc:
            rows.append((alc - fcc, img, fc["per_img_faces"].get(img, 0),
                         alb["per_img_faces"].get(img, 0), fcc, alc))
    rows.sort(key=lambda r: -abs(r[0]))
    total_delta = sum(r[0] for r in rows)
    print(f"\ncore delta (Albumify - FC v2) = {total_delta}  across {len(rows)} images")
    print(f"{'image':<34}{'fc_faces':>9}{'alb_faces':>10}{'fc_core':>8}{'alb_core':>9}")
    print("-" * 70)
    for delta, img, fcf, alf, fcc, alc in rows[:20]:
        print(f"{img[:33]:<34}{fcf:>9}{alf:>10}{fcc:>8}{alc:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
