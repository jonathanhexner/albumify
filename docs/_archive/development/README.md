# Development Documentation

This directory contains development logs, implementation notes, and historical documentation from the project's evolution.

## Directory Structure

### `session_logs/`
AI-assisted development session notes and debugging guides created during active development sessions.

**Contents:**
- Session-specific debugging guides (e.g., `DETERMINISM_DEBUG_GUIDE.md`, `BATCH_COMPARISON_GUIDE.md`)
- Implementation Q&A documents (e.g., `ANSWER_TO_KAGGLE_NOTEBOOK_QUESTION.md`)
- Configuration and setup notes

### `refactoring/`
Documentation of major refactoring efforts throughout the project.

**Contents:**
- `REFACTORING_SUMMARY.md` - Main refactoring summary
- Specific refactoring logs (dataloader, training scripts, multi-model, etc.)
- Historical refactoring plans and progress notes in `archive/`

### `summaries/`
Implementation summaries and completion reports.

**Contents:**
- High-level implementation summaries
- Feature-specific implementation reports
- `features/` - Summaries for specific features (CLIP, telemetry, hyperparameter search, etc.)

### `experiments/`
Experiment documentation and analysis reports.

**Contents:**
- Feature experiments (CLIP aesthetic, Siamese networks, etc.)
- Benchmark implementation plans
- `dataloader_investigation/` - Analysis reports from dataloader debugging

## Purpose

This directory preserves the development history and decision-making process. It helps:
- Understand why certain architectural decisions were made
- Track the evolution of features
- Learn from past debugging sessions
- Onboard new contributors

## Guidelines

- Session logs are timestamped or clearly dated
- Each major feature should have a summary document
- Experimental work should be documented in `experiments/`
- Superseded documents go to `archive/` subdirectories

## See Also

- [`../guides/`](../guides/) - User-facing guides and quickstarts
- [`../architecture/`](../architecture/) - Current system architecture documentation
