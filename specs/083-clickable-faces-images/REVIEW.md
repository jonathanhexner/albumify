# REVIEW — spec-083 Clickable faces/images + useful Images + pass-filter boxes

**2026-06-06** · scope: spec-083 diff · ✅ no High findings.

## Files
- `sim_bench/run_db/store.py` — `ImageRow.n_passed`; `list_images()` grouped
  count of faces with `rejection_reason IS NULL`.
- `face_cluster/views/image_metrics.py` — `ColumnSpec("n_passed","Passed")` + default.
- NEW `components/face_pick_grid.py` — clickable face grid (Open → Face Analysis).
- NEW `components/image_pick_grid.py` — clickable image grid (Open → in-tab detail).
- `tabs/face_metrics_tab.py` — Grid(default)/Table toggle; Grid uses real buttons.
- `tabs/images_tab.py` — Grid(default)/Table toggle; master-detail (`selected_image_path`
  + Back); thumbnails cached as bytes (grid) with data-URL derived for the table.
- `components/image_analysis.py` — `faces_to_box` (pure); default passing-only boxes
  + "show filtered" toggle + identity list + legend.
- NEW `tests/.../test_spec083_images_faces.py` (7); NEW e2e `test_scenario_j_*`,
  `test_scenario_k_*`; README matrix rows J/K.

## Root cause (the recurring "I can't click")
Reproduced live: `st.dataframe` row-select fires **only on the ~20px checkbox
column** — clicking a face/image cell does nothing (glide-grid text-select). The
drill-in "worked" only on an invisible target. Fixed by switching to the proven
`face_grid` pattern: a real `st.button("Open")` per thumbnail.

## Checklist
| § | Finding |
|---|---|
| Correctness | Grids: Open → `selected_face_id`/`selected_image_path` + nav/rerun (AppTest fid=0→Face Analysis; img→detail+Back, 0 exc). `n_passed` exact vs Face table (sum 186==186 on ref run). `faces_to_box` hides filtered by default, reveals on toggle, drops invalid bboxes. ✅ |
| Useful info | Images now shows per-image **faces · passed · W×H · gate**; analysis adds **identities present (C0,C3…)**; boxes default to faces that *passed filtration* (the explicit ask). ✅ |
| Reuse | face_pick_grid mirrors the validated face_grid Open path; image detail view reused; thumbnail threading + cache reused from spec-082. ✅ |
| Layering | grids are render-only components; tabs orchestrate; `n_passed` computed in the store query (not the tab). `populated_columns`/services Streamlit-free. Arch + views suite **248 passed**. ✅ |
| Tests | 7 unit + AppTest (both grids + detail/back) + **budapest J, K, D green** (real buttons are finally Playwright-addressable → permanent coverage for the click path that confused the user 3×). Screenshots: images grid, image detail (green box on passing face), FM grid. ✅ |
| Pre-existing | `tests/run_db/test_split_equivalence.py` golden `db:run_metadata` mismatch — reproduced with my `store.py` reverted (same hashes) → **not spec-083**; filed **SIGHTING-095**. ✅ |

## Deferred
- Per-image identity *column* in the list (needs the cluster-assignment join;
  shown in the analysis instead).
- Tab LOC (face_metrics 146 / images 163) over the spec-053 thin-tab ideal but
  unguarded; bodies are thin, the bulk is presentation helpers (thumbnail
  encode, table render). Extract if a guard is added.

**No High → Implemented.**
