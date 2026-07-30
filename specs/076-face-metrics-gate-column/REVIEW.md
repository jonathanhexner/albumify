# REVIEW — spec-076 Face Metrics gate/reason column

**2026-06-05** · scope: spec-076 diff · ✅ no High findings.

- `face_metrics.py`: `FaceMetricRow.rejection_reason` + service populates from the faces table.
- `face_metrics_tab.py`: "reason" structural column.

| § | Finding |
|---|---|
| Correctness | 154/340 reference faces show `top_k_per_image`; None → "". ✅ |
| Drill-in | already works (spec-070 row-click → `selected_face_id` → Face Analysis bbox+pose); confirmed, no change needed. ✅ |
| Tests | registry tests updated for the new field, 6 green; AppTest 0 exc; e2e Scenario D. ✅ |
| Note | thumbnail-click not natively possible in Streamlit — row-click is the supported equivalent. |

No High → Implemented.
