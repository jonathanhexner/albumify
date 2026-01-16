# Scripts Directory

All utility and executable scripts organized by purpose.

## Directory Structure

### `datasets/`
**Dataset-specific scripts** - Tools for working with specific datasets.

- `datasets/phototriage/` - PhotoTriage dataset scripts
  - Data preparation and labeling
  - Analysis and visualization
  - Dataset conversion utilities
- `datasets/ukbench/` - UKBench dataset scripts (future)
- `datasets/holidays/` - Holidays dataset scripts (future)

### `debug/`
**Debugging and verification scripts** - One-off tools for troubleshooting and validation.

- Determinism checks
- Training pipeline verification
- Dataloader debugging
- Model inspection utilities

**Note:** These are NOT unit tests - they're ad-hoc verification tools.

### `analysis/`
**Analysis and visualization scripts** - Tools for analyzing results and creating visualizations.

- Benchmark result analysis
- Quality assessment visualization
- Performance comparison tools

### `data_preparation/`
**General dataset preparation scripts** - Cross-dataset data processing utilities.

- Data conversion tools
- Feature precomputation
- General preprocessing utilities

### `utilities/`
**General utility scripts** - Miscellaneous tools.

- Kaggle notebook generation
- Training monitoring
- Model evaluation
- Integration verification

## Script Organization Guidelines

**Dataset-Specific Scripts** → `datasets/{dataset_name}/`
- If it's specific to PhotoTriage, UKBench, or Holidays, put it in the appropriate dataset folder

**Debugging Scripts** → `debug/`
- One-off verification scripts
- Determinism checks
- Pipeline validation
- Named `verify_*`, `check_*`, or `quick_*`

**Analysis Scripts** → `analysis/`
- Result visualization
- Benchmark analysis
- Performance comparison

**General Tools** → `utilities/`
- Cross-cutting utilities
- Monitoring tools
- Generic helpers

## Running Scripts

Most scripts can be run from the project root:

```bash
# Dataset-specific
python scripts/datasets/phototriage/analyze_results.py

# Debugging
python scripts/debug/verify_file_determinism.py

# Analysis
python scripts/analysis/visualize_quality_benchmark.py
```

## Contributing

When adding new scripts:
1. Choose the appropriate subdirectory
2. Use clear, descriptive names
3. Add a docstring at the top of the file
4. Include usage examples in the docstring
5. Update this README if adding a new category

## See Also

- [`../tests/`](../tests/) - Actual unit and integration tests
- [`../examples/`](../examples/) - Example usage scripts
