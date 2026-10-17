"""spec-079 Stage 0 diagnostic — is identity_refinement (the step Albumify runs
and FC v2 doesn't) the reason Albumify finds 20 identities vs FC v2's 15?

Runs the Albumify pipeline ONCE (profile_4 knobs overlaid on cluster_people) and
reads BOTH groupings the single run holds in memory:

  * context.people_clusters          = RAW (cluster_people output, pre-refinement)
  * context.refined_people_clusters  = AFTER identity_refinement (what the people
                                        table actually stores)

So we isolate the refinement effect without a second run. Also emits:
  * CLUSTER_COMPARISON.html  — sizes: Albumify raw vs refined vs FC v2
  * CLUSTER_GALLERY.html     — face thumbnails per cluster (Albumify raw + FC v2),
                               so you can SEE the clusters.

Run:  .venv/Scripts/python scripts/experiment_clustering_ab.py
"""
from __future__ import annotations

import html
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import _budapest_baseline as anchor  # noqa: E402
import capture_albumify_baseline as cap  # noqa: E402
from sim_bench.api.database.session import get_session_direct  # noqa: E402
from sim_bench.api.services.album_service import AlbumService  # noqa: E402
from sim_bench.api.services.pipeline_service import PipelineService, _jobs  # noqa: E402

OUT = REPO / "specs" / "079-albumify-shared-core"


def sizes_of(clusters: dict) -> list[int]:
    return sorted((len(v) for v in (clusters or {}).values()), reverse=True)


def rundir_cluster_crops(run_dir: Path) -> dict[int, list[Path]]:
    """cluster_id -> [absolute crop paths] from a run-dir face_clustering.db (final iter)."""
    db = run_dir / "face_clustering.db"
    if not db.is_file():
        return {}
    c = sqlite3.connect(str(db))
    mx = c.execute("SELECT MAX(iteration) FROM cluster_assignments").fetchone()[0]
    rows = c.execute(
        "SELECT ca.cluster_id, f.crop_path FROM cluster_assignments ca "
        "JOIN faces f ON f.face_id = ca.face_id "
        "WHERE ca.iteration=? AND ca.cluster_id>=0 ORDER BY ca.cluster_id", (mx,),
    ).fetchall()
    out: dict[int, list[Path]] = {}
    for cid, crop in rows:
        out.setdefault(int(cid), []).append((run_dir / crop).resolve())
    return out


def _img(p: Path) -> str:
    return f'<img src="file:///{str(p).replace(chr(92), "/")}" loading="lazy">'


def gallery_html(title: str, clusters: dict[int, list[Path]]) -> str:
    blocks = []
    for cid in sorted(clusters, key=lambda k: -len(clusters[k])):
        crops = clusters[cid]
        thumbs = "".join(_img(p) for p in crops if p.exists())
        blocks.append(
            f'<div class="cl"><div class="hd">cluster {cid} · {len(crops)} faces</div>'
            f'<div class="row">{thumbs}</div></div>'
        )
    return f'<section><h2>{html.escape(title)} — {len(clusters)} clusters</h2>{"".join(blocks)}</section>'


def main() -> int:
    session = get_session_direct()
    albums = AlbumService(session)
    pipeline = PipelineService(session)

    for a in albums.list_all():
        if a.name == "spec079_ab_experiment":
            albums.delete(a.id)
    album = albums.create("spec079_ab_experiment", str(anchor.SOURCE_DIR))
    step_configs = cap.build_step_configs()
    job_id = pipeline.start_pipeline(album.id, steps=None, step_configs=step_configs, fail_fast=True)
    print(f"[exp] running {job_id} ...")
    pipeline.execute_pipeline(job_id)
    run = pipeline.get_status(job_id)
    if run.status != "completed":
        print(f"[exp] FAILED: {run.error_message}")
        return 2

    ctx = _jobs[job_id].context
    raw = sizes_of(ctx.people_clusters)
    refined = sizes_of(getattr(ctx, "refined_people_clusters", None))
    ref_sizes = anchor.EXPECTED_CLUSTER_SIZES

    print("\n========== A/B: identity_refinement effect ==========")
    print(f"  RAW  cluster_people      : {len(raw):2d} clusters  sizes={raw}")
    print(f"  REFINED (+refinement)    : {len(refined):2d} clusters  sizes={refined}")
    print(f"  FC v2 reference          : {len(ref_sizes):2d} clusters  sizes={ref_sizes}")
    if len(raw) == len(refined):
        verdict = ("identity_refinement does NOT change the count "
                   f"({len(raw)}). The 20-vs-15 gap is the CONFIG (exemplar knobs), not refinement.")
    elif abs(len(raw) - len(ref_sizes)) < abs(len(refined) - len(ref_sizes)):
        verdict = (f"identity_refinement moves the count AWAY from FC v2 "
                   f"(raw {len(raw)} -> refined {len(refined)} vs FC v2 {len(ref_sizes)}). "
                   "It is a contributor.")
    else:
        verdict = (f"identity_refinement moves the count TOWARD FC v2 "
                   f"(raw {len(raw)} -> refined {len(refined)}). Config is the main driver.")
    print(f"  VERDICT: {verdict}")
    print("=====================================================")

    # --- comparison html ---
    def col(name, s):
        rows = "".join(f"<tr><td>{i+1}</td><td>{v}</td></tr>" for i, v in enumerate(s))
        return (f'<div class="c"><h3>{name}</h3><div class="n">{len(s)} clusters</div>'
                f'<table><tr><th>#</th><th>faces</th></tr>{rows}</table></div>')
    comp = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Cluster comparison</title>
<style>body{{background:#0f1220;color:#e8eaf2;font-family:Segoe UI,Arial;max-width:900px;margin:0 auto;padding:40px 24px}}
h1{{font-size:26px}}.v{{background:#171a2b;border:1px solid #2c3150;border-left:4px solid #6ea8fe;border-radius:10px;padding:16px 20px;margin:18px 0}}
.cols{{display:flex;gap:16px}}.c{{flex:1;background:#171a2b;border:1px solid #2c3150;border-radius:10px;padding:14px}}
.c h3{{margin:.2em 0}}.n{{color:#9aa0bd;font-size:13px;margin-bottom:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #2c3150;padding:3px 8px;text-align:left}}
th{{background:#1e2238}}</style></head><body>
<h1>Budapest clusters: Albumify (raw / refined) vs FC v2</h1>
<div class="v"><b>A/B verdict:</b> {html.escape(verdict)}</div>
<div class="cols">{col('Albumify RAW<br>(cluster_people)', raw)}{col('Albumify REFINED<br>(+identity_refinement)', refined)}{col('FC v2 reference', ref_sizes)}</div>
<p style="color:#9aa0bd">RAW = cluster_people output before identity_refinement. REFINED = what the people table stores. FC v2 ran no refinement step.</p>
</body></html>"""
    (OUT / "CLUSTER_COMPARISON.html").write_text(comp, encoding="utf-8")

    # --- gallery html (raw albumify + fc v2, both have crops) ---
    export_dir = None
    if getattr(ctx, "fc_export_dir", None):
        p = Path(ctx.fc_export_dir)
        export_dir = p if p.is_absolute() else (REPO / p)
    alb_crops = rundir_cluster_crops(export_dir) if export_dir else {}
    fcv2_crops = rundir_cluster_crops(anchor.REFERENCE_RUN_DIR)
    gal = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Cluster gallery</title>
<style>body{{background:#0f1220;color:#e8eaf2;font-family:Segoe UI,Arial;max-width:1200px;margin:0 auto;padding:36px 22px}}
h1{{font-size:25px}}h2{{font-size:19px;border-top:1px solid #2c3150;padding-top:14px;margin-top:32px}}
.cl{{background:#171a2b;border:1px solid #2c3150;border-radius:9px;padding:10px 12px;margin:10px 0}}
.hd{{font-weight:700;color:#cdd2ee;margin-bottom:7px;font-size:14px}}
.row{{display:flex;flex-wrap:wrap;gap:4px}}img{{width:58px;height:58px;object-fit:cover;border-radius:5px;border:1px solid #2c3150}}</style></head><body>
<h1>See the clusters — face thumbnails</h1>
<p style="color:#9aa0bd">Left-to-right within a row = faces grouped as the same person. Compare Albumify's grouping to FC v2's.</p>
{gallery_html('Albumify (raw cluster_people)', alb_crops)}
{gallery_html('FC v2 reference', fcv2_crops)}
</body></html>"""
    (OUT / "CLUSTER_GALLERY.html").write_text(gal, encoding="utf-8")
    print(f"[exp] wrote CLUSTER_COMPARISON.html and CLUSTER_GALLERY.html to {OUT}")
    print(f"[exp] albumify export dir: {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
