# Tests Directory

This directory contains all test scripts for the sim-bench project.

## Directory Structure

### `unit/`
**Unit tests** - Fast, isolated tests with no I/O or external dependencies.

- Test individual functions and classes
- Use mocked dependencies
- Run in milliseconds
- No file system or network access

### `integration/`
**Integration tests** - Tests that verify component interactions.

- Test with real file I/O
- May require dataset access
- Test actual workflows
- Slower but more comprehensive

### Root Test Files
Current root-level test files (to be organized):

- **test_comparison_functions.py** - Multi-experiment comparison tests
- **test_export.py** - Notebook export functionality tests
- **test_multi_experiment.py** - Multi-experiment analysis loader tests
- **test_quality_assessment.py** - Quality assessment method tests
- **test_quality_benchmark.py** - Quality benchmark framework tests
- **test_clip_aesthetic.py** - CLIP aesthetic assessment tests
- **test_clip_integration.py** - CLIP integration tests
- **test_photo_analysis.py** - Photo analysis tests

### Subdirectories
- `agent/` - Agent-related tests
- `quality_assessment/` - Quality assessment specific tests

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test categories:
```bash
# Unit tests (fast)
pytest tests/unit/

# Integration tests (slower)
pytest tests/integration/

# Specific test file
pytest tests/test_quality_assessment.py
```

### Run with coverage:
```bash
pytest --cov=sim_bench tests/
```

## Test Guidelines

### Unit Tests
- Must run in < 100ms per test
- No file I/O, network, or database access
- Mock all external dependencies
- Test ONE thing per test
- Use `tests/unit/` directory

### Integration Tests
- Can use real file system
- May require test datasets
- Test component interactions
- Longer running time acceptable
- Use `tests/integration/` directory

### Debugging Scripts vs Tests
**Debugging scripts** (in `scripts/debug/`) are NOT tests:
- No assertions or test framework
- One-off verification tools
- Ad-hoc debugging utilities
- Examples: `verify_file_determinism.py`, `check_batch_files.py`

**Real tests** (in `tests/`) use pytest and have assertions:
- Use pytest framework
- Have clear assertions
- Part of CI/CD pipeline
- Examples: `test_quality_assessment.py`

## Requirements

Integration tests may require:
- Experiment results in `outputs/` directory
- Full dataset access (where applicable)
- All dependencies from `requirements.txt`

## Contributing

When adding tests:
1. Choose appropriate directory (`unit/` or `integration/`)
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures for setup/teardown
4. Add docstrings explaining what's being tested
5. Keep unit tests fast (< 100ms)

## See Also

- [`../scripts/debug/`](../scripts/debug/) - Debugging and verification scripts
- [`../examples/`](../examples/) - Example usage scripts
