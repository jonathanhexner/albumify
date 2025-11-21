# Quality Assessment Module Refactoring Plan

## Issues Identified

1. **Factory Pattern**: Messy if/else chains, hard to maintain
2. **Config Handling**: Inconsistent parameter extraction, should be method's responsibility
3. **Output Directories**: Using `results/` instead of `outputs/`
4. **Device Configuration**: Hardcoded `cuda` in configs, need `cpu` default
5. **Analysis Organization**: Common analysis code scattered, needs sub-package
6. **Folder Structure**: Flat structure, need methods sub-package
7. **Logging**: Missing failure logs from benchmark runs

## Proposed Structure

```
sim_bench/quality_assessment/
├── __init__.py                    # Main exports
├── base.py                        # QualityAssessor base class (with is_available, from_config)
├── registry.py                    # Clean registry-based factory
│
├── methods/                       # All quality methods
│   ├── __init__.py
│   ├── rule_based.py             # @register_method('rule_based')
│   ├── clip_aesthetic.py         # @register_method('clip_aesthetic')
│   ├── cnn_methods.py            # @register_method('nima'), etc.
│   └── transformer_methods.py    # @register_method('musiq'), etc.
│
├── evaluation/                    # Evaluation frameworks
│   ├── __init__.py
│   ├── evaluator.py              # Series selection evaluator (legacy)
│   ├── pairwise_evaluator.py    # Pairwise classification evaluator
│   ├── benchmark.py              # Series benchmark runner
│   └── pairwise_benchmark.py    # Pairwise benchmark runner
│
├── analysis/                      # Analysis and visualization
│   ├── __init__.py
│   ├── correlation.py
│   ├── failure_analysis.py
│   ├── visualization.py
│   ├── load_results.py
│   └── method_wins.py
│
├── factory.py                     # DEPRECATED - redirects to registry
└── visualization.py               # DEPRECATED - moved to analysis/
```

## Changes Required

### 1. Registry-Based Factory

**Base Class** (`base.py`):
```python
class QualityAssessor(ABC):
    @classmethod
    def is_available(cls) -> bool:
        """Check if dependencies available."""
        return True  # Override in subclasses with dependencies

    @classmethod
    def from_config(cls, config: Dict) -> 'QualityAssessor':
        """Create from config - method pulls what it needs."""
        return cls(**config)
```

**Registry** (`registry.py`):
```python
@register_method('rule_based')
class RuleBasedQuality(QualityAssessor):
    @classmethod
    def from_config(cls, config):
        return cls(
            weights=config.get('weights'),
            device=config.get('device', 'cpu')
        )
```

**Usage**:
```python
# Old way (DEPRECATED)
from sim_bench.quality_assessment.factory import create_quality_assessor

# New way
from sim_bench.quality_assessment.registry import create_quality_assessor
# OR
from sim_bench.quality_assessment import create_quality_assessor
```

### 2. Output Directory Convention

**Change**: `results/` → `outputs/`

**Files to update**:
- `configs/pairwise_benchmark.*.yaml`: `base_dir: "outputs/pairwise_benchmark"`
- `configs/quality_benchmark.*.yaml`: `base_dir: "outputs/quality_benchmark"`
- `global_config.yaml`: Update default output directory
- Documentation

### 3. Device Configuration

**Current**: Many configs have `device: "cuda"`
**Fix**: Default to `"cpu"`, add note about GPU

**All benchmark configs**:
```yaml
methods:
  - name: "CLIP-Aesthetic-LAION"
    type: "clip_aesthetic"
    variant: "laion"
    device: "cpu"  # Change to "cuda" if GPU available
```

### 4. Analysis Package Organization

**Move**:
- `quality_assessment/analysis/*.py` → Keep (already organized)
- `quality_assessment/visualization.py` → `analysis/visualization.py`

**Update imports** throughout codebase.

### 5. Methods Sub-Package

**Create** `quality_assessment/methods/` and move:
- `rule_based.py` → `methods/rule_based.py`
- `cnn_methods.py` → `methods/cnn_methods.py`
- `transformer_methods.py` → `methods/transformer_methods.py`
- `clip_aesthetic.py` → `methods/clip_aesthetic.py`

**Update** `__init__.py` to import from new locations.

### 6. Evaluation Sub-Package

**Create** `quality_assessment/evaluation/` and move:
- `evaluator.py` → `evaluation/evaluator.py`
- `benchmark.py` → `evaluation/benchmark.py`
- `pairwise_evaluator.py` → `evaluation/pairwise_evaluator.py`
- `pairwise_benchmark.py` → `evaluation/pairwise_benchmark.py`

### 7. Enhanced Logging

**Benchmark runners** need:
```python
# Add exception logging with full traceback
except Exception as e:
    logger.error(f"Error evaluating {method_name}: {e}", exc_info=True)
    # Also save to file
    with open(output_dir / f'{method_name}_error.log', 'w') as f:
        import traceback
        f.write(traceback.format_exc())
```

**Log file per benchmark run**:
```python
log_file = output_dir / 'benchmark.log'
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
```

## Migration Steps

### Phase 1: Registry (No breaking changes)
1. ✅ Add `is_available()` and `from_config()` to base class
2. ✅ Create `registry.py`
3. ✅ Add `@register_method` to `RuleBasedQuality`
4. Add to other methods (CLIP, NIMA, etc.)
5. Update imports in `__init__.py`

### Phase 2: Reorganize Structure
1. Create `methods/`, `evaluation/` directories
2. Move files to new locations
3. Update all imports
4. Update `__init__.py` exports
5. Mark old `factory.py` as deprecated with redirect

### Phase 3: Fix Configs
1. Change `results/` → `outputs/`
2. Change `cuda` → `cpu` (with comments)
3. Update documentation

### Phase 4: Enhanced Logging
1. Add file logging to benchmarks
2. Add error logs per failed method
3. Test error scenarios

## Backward Compatibility

To avoid breaking existing code:

1. Keep `factory.py` with deprecation warning:
```python
# factory.py
import warnings
from sim_bench.quality_assessment.registry import create_quality_assessor

warnings.warn(
    "factory.py is deprecated, use registry.py",
    DeprecationWarning
)
```

2. Keep old imports working:
```python
# __init__.py
from sim_bench.quality_assessment.registry import create_quality_assessor
from sim_bench.quality_assessment.methods.rule_based import RuleBasedQuality
# ... etc
```

## Testing Plan

1. Run quick benchmark with new registry
2. Verify all methods load correctly
3. Check output directories
4. Test error logging
5. Run full benchmark

## Implementation Priority

**High Priority** (Do Now):
1. ✅ Fix device configs (cuda → cpu) - DONE (pairwise_benchmark.3hour.yaml)
2. ✅ Add error logging to benchmarks - DONE (file logging + per-method error logs)
3. ✅ Fix output directory (results → outputs) - DONE (all pairwise configs updated)
4. ✅ Fix Unicode encoding issues - DONE (replaced ≥ and ≤ with >= and <=)

**Medium Priority** (Next):
5. ✅ Create registry pattern - DONE (registry.py: 117 lines, factory.py deleted - was 220+ lines)
6. ✅ Update all configs to use CPU by default - DONE

**Low Priority** (Later):
7. Reorganize folder structure
8. Full refactoring

## Notes

- Keep backward compatibility during transition
- Document all changes in CHANGELOG
- Update docs/quality_assessment/ after refactoring
