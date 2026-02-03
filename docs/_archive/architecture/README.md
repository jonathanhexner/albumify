# Architecture and Development

System design, implementation details, and development guides.

## Quick Links

- [Overview](overview.md) - System architecture
- [Caching](caching.md) - Feature caching system
- [Logging](logging.md) - Logging configuration
- [Refactoring Notes](refactoring.md) - Code improvements

## System Components

### Core Architecture

- **Dataset Abstraction**: Unified interface for all datasets
- **Method Interface**: Standard API for all similarity methods
- **Result Management**: Consistent result storage and retrieval
- **Experiment Runner**: Orchestrates benchmarks

See [overview.md](overview.md) for details.

### Performance Features

- **Feature Caching**: Cache extracted features to disk for reuse
- **Batch Processing**: Efficient batch feature extraction
- **Sampling**: Quick testing on dataset subsets

See [caching.md](caching.md) for caching details.

### Logging System

- **Detailed Logs**: Per-experiment logging
- **Error Tracking**: Full stack traces in logs
- **Progress Reporting**: Real-time progress bars

See [logging.md](logging.md) and [logging_detailed.md](logging_detailed.md).

### Deep Learning Integration

Deep learning methods (DINOv2, OpenCLIP) are documented in the image_similarity section since they are primarily used for similarity search.

## Development

See [refactoring.md](refactoring.md) for recent code improvements and design decisions.

