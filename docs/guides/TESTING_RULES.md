# Testing Rules

## Naming Conventions
- Unit test classes: `ut_<Subject>` (e.g., `ut_QualityGater`). Methods: `test_<scenario>`.
- E2E/integration: `test_*` module-level functions.
- pytest configured in `pyproject.toml` to discover `ut_*` classes and `test_*` functions.

## Requirements
- Tests must use production-default config, not relaxed values. If relaxed config is needed, add a comment explaining why and add a separate test with default config.
- Pipeline integration tests must exercise the holdout path (core_indices is a proper subset).
- Tests must run on Windows. ASCII-only in CLI output (no Unicode box chars).

## Anti-Patterns — Never Acceptable
- **No `skipif` on missing fixtures.** Use `pytest.fail()` if fixture is absent.
- **No `sys.path` manipulation.** Fix with `pip install -e .`.
- **Clustering tests must assert both purity AND completeness.** Purity alone passes with every-face-in-own-cluster.
- **No vacuously true assertions.** Assert on concrete expected values.
- **File format contracts must be tested end-to-end.** Writer → reader → validate types.
