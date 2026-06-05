# spec-071 — Tasks

**Spec**: [spec.md](spec.md) · **Design**: [DESIGN.html](DESIGN.html) · **Status**: Draft (awaiting go-ahead to build)

Bring V1 Merge Analysis's production value into V2 Merged Clusters:
A = gate/threshold review, B = visual pair crops. Approval(C)/ML(D)/remerge(E)
out. All data already persisted — no pipeline/DB change.

## Build  (T01–T05 DONE 2026-06-05; verified on real run `v2_budapest_20260605b`)

- [x] T01 — Extend `face_cluster/views/merged_clusters.py`:
      `summary() -> MergeReviewSummary`, `gate_badges(row) -> list[GateBadge]`
      (pure mapping of `passes_*` + value/threshold detail), `pair_faces(row)
      -> PairFaces` (via `find_assignments(iteration=row.iteration)` filtered to
      cluster_a / cluster_b + `crop_path`). New dataclasses. Streamlit-free.

- [x] T02 — Components: `merge_gate_badges.py` (✓/✗ chips) +
      `cluster_pair_crops.py` (two thumbnail strips). DONE.

- [x] T03 — Enriched `merged_clusters_tab.py`: summary strip + filter + table;
      on select → gate badges + numbers + pair crops (replaced raw `st.json`).
      Telemetry kept; tab 90 LOC (at budget). DONE.

- [x] T04 — +3 synthetic tests (gate mapping, summary+top gate, pair-face
      resolution matches repo assignments). 12 green incl. arch LOC. DONE.

- [x] T05 — Real-run check (`v2_budapest_20260605b`): summary n_rejected=2
      top=cross; pair crops resolved (cluster 0 = 23, cluster 6 = 2);
      AppTest 0 exceptions + `tab.done name=merged_clusters`. DONE.

- [x] T05b — **Playwright browser test (option a)**: added `?selected_merge_pair=a,b`
      seed (`main.py`) + tab consumption; new e2e **Scenario I**
      (`test_scenario_i_merged_clusters_detail.py`) seeds a real pair and asserts
      the gate badges + numbers caption + pair-crop captions + a visible `<img>`
      paint in a real browser (bypasses the canvas row-pick, SIGHTING-091).
      **PASSES.** README matrix row added. Regression: 142 passed.

- [ ] T06 — **User visual sign-off**: open tab, pick a rejected pair, confirm.

- [ ] T07 — Close-out: CHANGES_LOG (done); docs/architecture classes.html for
      the new dataclasses; `/code-review` → REVIEW.md; flip Status → Implemented.

## Notes
- Reuses existing repo reads only (AC5) — arch test should confirm no new
  `face_clustering` per-run DB path / no SQL in the tab.
- Iteration nuance: `cluster_a/cluster_b` are PRE-merge ids at `row.iteration`;
  resolve faces with `find_assignments(iteration=row.iteration)`, not "final".
