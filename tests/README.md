# Tests Directory

This directory contains test scripts for the sim-bench project.

## Test Files

- **test_comparison_functions.py** - Tests for multi-experiment comparison functions
- **test_export.py** - Tests for notebook export functionality
- **test_multi_experiment.py** - Tests for multi-experiment analysis loader
- **test_quality_assessment.py** - Tests for quality assessment methods
- **test_quality_benchmark.py** - Tests for quality benchmark framework

## Running Tests

Run individual test files:

```bash
python tests/test_comparison_functions.py
python tests/test_export.py
python tests/test_multi_experiment.py
python tests/test_quality_assessment.py
python tests/test_quality_benchmark.py
```

## Note

These are integration tests that work with actual experiment data. They require:
- Experiment results in `outputs/` directory
- Full dataset access (where applicable)
- All dependencies installed from `requirements.txt`
