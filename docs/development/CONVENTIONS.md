# Repository Organization Conventions

This document defines the organizational structure and conventions for the sim-bench repository.

## Directory Structure

```
sim-bench/
├── README.md                    # Main project readme
├── LICENSE                      # License file
├── docs/                        # All documentation
│   ├── guides/                  # User-facing guides
│   ├── architecture/            # System architecture docs
│   ├── development/             # Development logs & notes
│   │   ├── session_logs/        # AI session notes
│   │   ├── refactoring/         # Refactoring logs
│   │   ├── summaries/           # Implementation summaries
│   │   └── experiments/         # Experiment documentation
│   └── quality_assessment/      # Quality assessment docs
├── scripts/                     # All executable scripts
│   ├── datasets/                # Dataset-specific scripts
│   ├── debug/                   # Debugging tools
│   ├── analysis/                # Analysis scripts
│   ├── data_preparation/        # Data prep scripts
│   └── utilities/               # General utilities
├── tests/                       # All tests
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── notebooks/                   # Jupyter notebooks
│   ├── analysis/                # Analysis notebooks
│   ├── experiments/             # Experiment notebooks
│   └── examples/                # Tutorial notebooks
├── examples/                    # Example scripts
├── configs/                     # Configuration files
├── sim_bench/                   # Main package
├── app/                         # Streamlit app
├── data/                        # Small data files
└── outputs/                     # Training outputs (not docs)
```

## File Placement Rules

### Root Directory
**Allowed:**
- `README.md` - Main project readme
- `LICENSE` - License file
- Configuration files: `requirements*.txt`, `pyproject.toml`, `setup.cfg`
- Entry point scripts: `app.py`, `app_agent*.py`, `run_*.py`

**NOT Allowed:**
- Markdown files (except README.md)
- Test files (`test_*.py`)
- Utility scripts
- Jupyter notebooks

### Documentation (`docs/`)

#### User-Facing Guides → `docs/guides/`
- Quickstart guides: `*_quickstart.md`
- Setup guides: `*_setup.md`, `*_guide.md`
- How-to documentation

#### Development Documentation → `docs/development/`
- **Session logs** → `docs/development/session_logs/`
  - AI-assisted development notes
  - Debugging session logs
  - Implementation Q&A

- **Refactoring logs** → `docs/development/refactoring/`
  - Major refactoring documentation
  - Superseded docs → `refactoring/archive/`

- **Implementation summaries** → `docs/development/summaries/`
  - Feature implementation summaries
  - Feature-specific → `summaries/features/`

- **Experiments** → `docs/development/experiments/`
  - Experiment documentation
  - Analysis reports
  - Investigation logs

#### Architecture Documentation → `docs/architecture/`
- System architecture documents
- Component architecture
- Design decisions

### Scripts (`scripts/`)

#### Dataset-Specific → `scripts/datasets/{dataset_name}/`
**Examples:**
- `scripts/datasets/phototriage/` - PhotoTriage tools
- `scripts/datasets/ukbench/` - UKBench tools
- `scripts/datasets/holidays/` - Holidays tools

**Contents:**
- Data preparation for specific dataset
- Dataset analysis
- Dataset conversion

#### Debugging Tools → `scripts/debug/`
**Purpose:** One-off verification and debugging scripts

**Naming:** `verify_*`, `check_*`, `quick_*`

**Examples:**
- `verify_file_determinism.py`
- `check_batch_files.py`
- `quick_determinism_check.py`

**Note:** These are NOT tests - no test framework, just ad-hoc tools.

#### Analysis Scripts → `scripts/analysis/`
**Purpose:** Result analysis and visualization

**Examples:**
- `visualize_quality_benchmark.py`
- `analyze_results.py`

#### Data Preparation → `scripts/data_preparation/`
**Purpose:** General data processing (not dataset-specific)

**Examples:**
- Feature precomputation
- Data conversion utilities
- General preprocessing

#### Utilities → `scripts/utilities/`
**Purpose:** General-purpose utilities

**Examples:**
- `monitor_training.py`
- `create_kaggle_notebook.py`
- `evaluate_best_model.py`

### Tests (`tests/`)

#### Unit Tests → `tests/unit/`
**Characteristics:**
- Fast (< 100ms per test)
- No I/O, network, or database
- Mocked dependencies
- Test ONE thing

#### Integration Tests → `tests/integration/`
**Characteristics:**
- Real file I/O allowed
- May require datasets
- Test component interactions
- Slower but comprehensive

**NOT Tests:**
- Scripts without pytest/assertions go to `scripts/debug/`

### Notebooks (`notebooks/`)

#### Analysis → `notebooks/analysis/`
- Data analysis notebooks
- Result exploration

#### Experiments → `notebooks/experiments/`
- Experimental work
- Kaggle notebooks
- Feature prototyping

#### Examples → `notebooks/examples/`
- Tutorial notebooks
- Usage demonstrations

## Naming Conventions

### Documentation
- **Guides:** `{topic}_quickstart.md`, `{topic}_guide.md`
- **Session logs:** Descriptive names with context
- **Summaries:** `{feature}_SUMMARY.md`
- **Architecture:** `{component}_architecture.md`

### Scripts
- **Debugging:** `verify_*`, `check_*`, `quick_*`
- **Analysis:** `analyze_*`, `visualize_*`
- **Utilities:** Descriptive verb-noun format

### Tests
- **Test files:** `test_*.py`
- **Test functions:** `test_{what_is_tested}()`
- **Fixtures:** Descriptive names

### Notebooks
- **Descriptive names:** `analyze_pairwise_benchmark.ipynb`
- **No version numbers:** Use git for versioning
- **Lowercase with underscores:** `my_notebook.ipynb`

## Workflow Guidelines

### Adding New Documentation
1. Determine if it's a guide, development log, or architecture doc
2. Place in appropriate `docs/` subdirectory
3. Update relevant README.md
4. Link from main README if user-facing

### Adding New Scripts
1. Determine category (dataset, debug, analysis, utility)
2. Place in appropriate `scripts/` subdirectory
3. Add docstring with usage example
4. Update `scripts/README.md` if introducing new pattern

### Adding Tests
1. Determine if unit or integration test
2. Place in `tests/unit/` or `tests/integration/`
3. Use pytest framework
4. Include clear assertions
5. Add docstrings

### Adding Notebooks
1. Determine category (analysis, experiment, example)
2. Place in appropriate `notebooks/` subdirectory
3. Clear markdown cells explaining each step
4. Clean output before committing

## Migration Checklist

When reorganizing files:
- [ ] Move files to correct directories
- [ ] Update import statements
- [ ] Update documentation links
- [ ] Update CI/CD paths
- [ ] Run test suite
- [ ] Update README files
- [ ] Verify all links work

## Benefits of This Structure

**Discoverability:** Easy to find files by purpose
**Scalability:** Clear place for new additions
**Professionalism:** Clean, organized repository
**Maintainability:** Logical grouping reduces confusion
**Onboarding:** New contributors understand structure quickly

## Enforcement

- Code reviews should check file placement
- CI/CD should warn on files in wrong locations
- This document is the single source of truth
- Update this document when patterns change

---

Last Updated: January 2026
