"""spec-090: resolve which run to view — picked-if-valid else latest."""

from app.streamlit.components.run_selector import resolve_run_id

_RESULTS = [  # API order: newest first
    {"job_id": "run-c", "num_selected": 7},
    {"job_id": "run-b", "num_selected": 5},
    {"id": "run-a", "num_selected": 3},  # older shape uses "id"
]


class ut_ResolveRunId:

    def test_empty_results_is_none(self):
        assert resolve_run_id([], "anything") is None

    def test_none_current_picks_latest(self):
        assert resolve_run_id(_RESULTS, None) == "run-c"

    def test_valid_current_is_kept(self):
        assert resolve_run_id(_RESULTS, "run-b") == "run-b"

    def test_valid_current_via_id_key(self):
        assert resolve_run_id(_RESULTS, "run-a") == "run-a"

    def test_stale_current_falls_back_to_latest(self):
        assert resolve_run_id(_RESULTS, "deleted-run") == "run-c"
