# REVIEW — spec-073 Area-% quality gate

**Reviewed**: 2026-06-05 · **Scope**: spec-073 diff in isolation. **Verdict**: ✅ no High findings.

## Files
- `face_cluster/fc_params.py` — `min_face_area_pct: Optional[float]` (0–100, None=off)
- `face_cluster/config.py` — `PipelineConfig.min_face_area_pct` (parity)
- `app/face_clustering_v2/ui_spec.py` — quality-group entry (auto-renders in Run tab)
- `face_cluster/quality.py` — `_add_area_pct_gate` + wired into both verdict paths + rejection priority
- `tests/face_clustering/test_area_pct_gate.py` — NEW (4)

## Checklist
| § | Finding |
|---|---|
| Correctness | `area_ratio*100 ≥ thr`; None thr → gate skipped; None area_ratio → permissive pass (legacy-safe). `area_ratio` is set at detection (`insightface_detect_faces.py:187`) so it's present at gate time. ✅ |
| Reuse | mirrors `min_face_area`; no new plumbing — `model_dump()` carries it to the step config + `QualityGater.config`. ✅ |
| Tests | 4 gate units (reject/pass/off/permissive) + parity + ui_spec + config_parity arch tests green; AppTest Run-tab widget present, 0 exc. ✅ |
| Layering | gate logic in `quality.py` (domain); UI knob via the registry, not hardcoded. ✅ |
| Contracts | FCParams↔FCConfig parity held (test green); verdict gate `area_pct` rides the spec-093 G1 writer → persists to `filter_decisions`. ✅ |
| Risk | **default None → fully inert**; budapest baseline unaffected (verified: default flows None to the step). Both pixel + % gates can be set independently. ✅ |

**No High findings → Implemented.**
