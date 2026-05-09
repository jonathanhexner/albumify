# Tasks — Spec 019: Merge Cross-Distance OR Gate

## Design Notes

Gate A becomes `p25_exemplar_dist <= T_exemplar OR p25_cross_dist <= T_cross`.
Both metrics are always computed and logged. Unique-pair support is an opt-in
replacement for raw support in Gate B.

Key implementation sites:
- `face_cluster/config.py` — add 3 config fields
- `face_cluster/merge.py` — `_evaluate_merge_evidence`, new helpers
- `face_cluster/views/merge_view.py` — expose new fields in analysis view
- `app/face_clustering/_merge_decisions_panel.py` — show cross-dist in gate badges

## Task Checklist

- [x] T1: Add config fields (`merge_cross_threshold`, `merge_use_cross_gate`, `merge_support_unique`) to `PipelineConfig`
- [x] T2: Implement `_p25_cross_distance(nodes_a, nodes_b, distance_matrix) -> float` in `ConservativeMerger`
- [x] T3: Implement `_count_unique_support(nodes_a, nodes_b, distance_matrix, threshold) -> int` in `ConservativeMerger`
- [x] T4: Wire OR gate into `_evaluate_merge_evidence` — Gate A becomes `passes_exemplar OR passes_cross`
- [x] T5: Add `p25_cross_dist`, `passes_cross`, and `unique_support` to the evidence dict / merge_log entries
- [x] T6: Wire `merge_support_unique` flag into Gate B evaluation
- [x] T7: Unit tests for `_p25_cross_distance` (basic, comparison with exemplar dist)
- [x] T8: Unit tests for `_count_unique_support` (node reuse prevented, greedy ordering)
- [x] T9: Unit test for OR gate: pair fails exemplar but passes cross -> valid
- [x] T10: Unit test for config flag off: old behavior preserved
- [x] T11: Update gate badge rendering to show cross-dist and unique support
- [x] T12: Update `MergeDecisionRow` / analysis view to carry new fields
- [x] T13: Update CHANGES_LOG.md
