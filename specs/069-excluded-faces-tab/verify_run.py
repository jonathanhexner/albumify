"""spec-069 — verification artifact generator (Phase 1, human golden baseline).

Reads a completed v2 run dir and produces a standalone, **sortable** HTML
table: one row per face with its thumbnail + every per-face metric (blur,
area, det_score, yaw/pitch/roll) + its status (assigned cluster / noise /
gate-rejected reason). Click any column header to sort — so you can find the
highest blur, the smallest face, the lowest det_score, etc., and judge whether
the exclusions look right BEFORE we build the tab or freeze any criteria.

Throwaway verification tool — reuses existing repository/RunStore reads only.

Usage:
    .venv/Scripts/python specs/069-excluded-faces-tab/verify_run.py <run_dir>
"""
from __future__ import annotations

import base64
import sys
from collections import defaultdict
from pathlib import Path

from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisCriteria,
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
    FilterDecisionCriteria,
)
from sim_bench.pipeline.clustering_labels import NOISE_LABEL, is_noise
from sim_bench.run_db.store import RunStore


def _crop_b64(run_dir: Path, face_id: int) -> str:
    crops = run_dir / "crops"
    cand = None
    for pat in (f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg",
                f"face_{face_id}_aligned.jpg"):
        p = crops / pat
        if p.is_file():
            cand = p
            break
    if cand is None and crops.is_dir():
        hits = list(crops.glob(f"face_{face_id:04d}*.jpg"))
        cand = hits[0] if hits else None
    if cand is None:
        return ""
    return base64.b64encode(cand.read_bytes()).decode("ascii")


def _num(v, nd=1):
    return "" if v is None else f"{float(v):.{nd}f}"


def main(run_dir: Path) -> int:
    repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    faces = RunStore(run_dir).faces()

    # status maps
    assign = {a.face_id: a.cluster_id
              for a in repo.find_assignments(ClusterAnalysisCriteria(include_noise=True))}
    rejects_by_face: dict[int, list[str]] = defaultdict(list)
    for r in repo.list_filter_decisions(FilterDecisionCriteria(rejected=True)):
        if str(r.item_id).lstrip("-").isdigit():
            rejects_by_face[int(r.item_id)].append(r.filter_name)

    def status(fid: int) -> tuple[str, str]:
        if fid in rejects_by_face:
            return ("gate", "reject:" + "+".join(sorted(set(rejects_by_face[fid]))))
        cid = assign.get(fid)
        if cid is None:
            return ("none", "—")
        if is_noise(cid):
            return ("noise", "noise")
        return ("ok", f"cluster {cid}")

    # funnel
    n_detected = len(faces)
    n_gate = len(rejects_by_face)
    n_noise = sum(1 for c in assign.values() if is_noise(c))
    n_assigned = sum(1 for c in assign.values() if not is_noise(c))
    n_pose = sum(1 for f in faces if f.pose is not None)

    # rows
    rows = []
    for f in faces:
        cls, lab = status(f.face_id)
        yaw, pitch, roll = (f.pose if f.pose else (None, None, None))
        b64 = _crop_b64(run_dir, f.face_id)
        img = (f"<img loading='lazy' src='data:image/jpeg;base64,{b64}'>"
               if b64 else "<div class='noimg'>—</div>")
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{img}</td>"
            f"<td data-v='{f.face_id}'>{f.face_id}</td>"
            f"<td data-v='{cls}'><span class='badge {cls}'>{lab}</span></td>"
            f"<td data-v='{f.blur_score or 0:.4f}'>{_num(f.blur_score)}</td>"
            f"<td data-v='{f.area or 0:.1f}'>{_num(f.area, 0)}</td>"
            f"<td data-v='{f.det_score or 0:.4f}'>{_num(f.det_score, 3)}</td>"
            f"<td data-v='{(yaw if yaw is not None else -999)}'>{_num(yaw)}</td>"
            f"<td data-v='{(pitch if pitch is not None else -999)}'>{_num(pitch)}</td>"
            f"<td data-v='{(roll if roll is not None else -999)}'>{_num(roll)}</td>"
            f"</tr>"
        )

    pose_note = ("" if n_pose else
                 "<span class='warn'>pose not computed this run "
                 "(use_pose_estimation off) — yaw/pitch/roll are blank</span>")

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>spec-069 verify · {run_dir.name}</title><style>
body{{font:13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e9ef}}
header{{padding:16px 22px;border-bottom:1px solid #2a2f3a}}
h1{{margin:0;font-size:18px}} .sub{{color:#9aa3b2;font-size:12px}}
.funnel{{display:flex;gap:18px;margin:10px 22px;flex-wrap:wrap;font-size:13px}}
.funnel b{{font-size:17px}} .det b{{color:#e6e9ef}} .gate b{{color:#e0a458}}
.noise b{{color:#c977d6}} .ok b{{color:#5fd29b}}
.warn{{color:#efb36f;font-size:12px}}
table{{border-collapse:collapse;width:calc(100% - 44px);margin:10px 22px 60px;font-size:12px}}
th,td{{border-bottom:1px solid #232833;padding:5px 8px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(3),td:nth-child(3){{text-align:left}}
th{{position:sticky;top:0;background:#1a1e27;cursor:pointer;user-select:none;color:#cdd6e4}}
th:hover{{color:#fff}} th.sorted::after{{content:' ▾';color:#5b9dff}} th.asc::after{{content:' ▴';color:#5b9dff}}
td img{{width:48px;height:48px;object-fit:cover;border-radius:4px;display:block}}
.noimg{{width:48px;height:48px;display:flex;align-items:center;justify-content:center;color:#555;background:#11141a;border-radius:4px}}
.badge{{padding:1px 7px;border-radius:5px;font-size:11px;font-weight:600}}
.badge.gate{{background:#241d10;color:#e0a458}} .badge.noise{{background:#231325;color:#c977d6}}
.badge.ok{{background:#10251a;color:#5fd29b}} .badge.none{{background:#222;color:#888}}
tr.gate{{background:#1a1610}} tr.noise{{background:#180f1a}}
</style></head><body>
<header><h1>Excluded Faces — per-face metrics (sortable)</h1>
<div class='sub'>run: {run_dir.name} · click a column header to sort · {pose_note}</div></header>
<div class='funnel'>
<span class='det'>detected <b>{n_detected}</b></span>
<span class='gate'>gate-rejected <b>{n_gate}</b></span>
<span class='noise'>unassigned <b>{n_noise}</b></span>
<span class='ok'>assigned <b>{n_assigned}</b></span>
<span style='color:#9aa3b2'>with-pose <b>{n_pose}</b></span></div>
<table id='t'><thead><tr>
<th>face</th><th data-t='n'>id</th><th data-t='s'>status</th>
<th data-t='n'>blur</th><th data-t='n'>area(px²)</th><th data-t='n'>det_score</th>
<th data-t='n'>yaw</th><th data-t='n'>pitch</th><th data-t='n'>roll</th>
</tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<script>
const tb=document.querySelector('#t tbody');
document.querySelectorAll('#t th').forEach((th,ci)=>{{
  let asc=true;
  th.addEventListener('click',()=>{{
    document.querySelectorAll('#t th').forEach(h=>h.classList.remove('sorted','asc'));
    const num=th.dataset.t==='n';
    const rows=[...tb.rows];
    rows.sort((a,b)=>{{
      let x=a.cells[ci].dataset.v??a.cells[ci].innerText;
      let y=b.cells[ci].dataset.v??b.cells[ci].innerText;
      if(num){{x=parseFloat(x)||0;y=parseFloat(y)||0;return asc?x-y:y-x;}}
      return asc?(''+x).localeCompare(y):(''+y).localeCompare(x);
    }});
    rows.forEach(r=>tb.appendChild(r));
    th.classList.add(asc?'asc':'sorted');asc=!asc;
  }});
}});
</script></body></html>"""

    out_html = Path(__file__).parent / f"VERIFY_{run_dir.name}.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"detected={n_detected} gate_rejected={n_gate} unassigned={n_noise} "
          f"assigned={n_assigned} with_pose={n_pose}")
    print(f"wrote {out_html}")
    return 0


if __name__ == "__main__":
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path.home() / ".sim_bench" / "runs" / "v2_budapest_20260605")
    if not rd.is_dir():
        print(f"run dir not found: {rd}")
        sys.exit(2)
    sys.exit(main(rd))
