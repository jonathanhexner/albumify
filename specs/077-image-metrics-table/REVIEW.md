# REVIEW — spec-077 Per-image metrics table

**2026-06-05** · scope: spec-077 diff · ✅ no High findings.

- `store.py`: `ImageRow` + `RunStore.list_images()` (ORM read of the images table).
- `image_metrics.py`: `ImageMetricsService` + `IMAGE_METRIC_COLUMNS` + `DEFAULT_IMAGE_COLUMNS`.
- `images_tab.py` (NEW, 76 LOC) + wired into `main.py` (11th tab "Images").

| § | Finding |
|---|---|
| Correctness | 122 images load (n_faces + filter_passed real; iqa/ava None for face-only runs — expected, columns are opt-in). ✅ |
| Reuse | ColumnSpec registry + `resolve_run_dir`; column multiselect drives the shown set. ✅ |
| Tests | registry + real-fixture smoke (2) green; AppTest all 11 tabs 0 exc. ✅ |
| Layering | service Streamlit-free; tab ≤80 LOC, no SQL/FS; ORM read in RunStore. ✅ |

No High → Implemented.
