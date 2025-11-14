# Unified Orchestration Architecture - Effort Analysis

## Executive Summary

**Yes, it's absolutely possible** to build a unified architecture that supports all three orchestration strategies (rule-based, agent/tools, LLM) with minimal duplication. The key insight is that **80% of the code is shared infrastructure**, and only the orchestration decision-making logic differs.

## Shared Infrastructure (80% of code)

All three strategies share the same core components:

### 1. Base Abstractions (100% shared)
```python
# These classes are identical across all strategies:
- Signal (abstract feature extractor)
- Pipeline (composed of multiple signals with fusion)
- PipelineSpec (specification for building pipelines)
- DetectionResult (output from scene detection)
- Detector (abstract scene/object detector)
- Orchestrator (abstract interface)
```

**Effort**: ~2 days (one-time implementation)

### 2. Signal Library (100% shared)
```python
# All strategies use the same signal implementations:
- ArcFaceSignal (face recognition)
- DINOv2Signal (semantic embeddings)
- SIFTSignal (geometric features)
- PerceptualHashSignal (duplicate detection)
- ColorHistogramSignal (color features)
- OpenCLIPSignal (multimodal)
```

**Effort**: ~3 days (one-time implementation, can reuse existing extractors)

### 3. Fusion Engine (100% shared)
```python
# All strategies use the same fusion logic:
- WeightedAverageFusion
- CascadeFusion
- VotingFusion
```

**Effort**: ~1 day (one-time implementation)

### 4. Detection Stage (100% shared)
```python
# All strategies use the same detectors:
- CLIPZeroShotDetector
- YOLODetector
```

**Effort**: ~1.5 days (one-time implementation)

### 5. Pipeline Builder (100% shared)
```python
# Pipeline construction logic is identical:
- Pipeline.__init__(spec)
- Pipeline.extract_features()
- Pipeline.compute_similarity_matrix()
```

**Effort**: ~1 day (one-time implementation)

## Unique Per-Strategy Code (20% of code)

Only the orchestration decision-making differs:

### Strategy 1: Rule-Based
```python
class RuleBasedOrchestrator(Orchestrator):
    def build_pipeline(self, detection, config):
        # Simple dictionary lookup + hardcoded rules
        scene_type = detection.scene_type
        registry_entry = self.pipeline_registry[scene_type]

        # Build spec from registry
        spec = PipelineSpec(...)
        return Pipeline(spec)
```

**Unique code**: ~50 lines
**Effort**: ~0.5 days

### Strategy 2: LLM-Based
```python
class LLMOrchestrator(Orchestrator):
    def build_pipeline(self, detection, config):
        # Call LLM with structured output
        prompt = ChatPromptTemplate.from_messages([...])
        structured_llm = self.llm.with_structured_output(PipelineConfiguration)
        pipeline_config = (prompt | structured_llm).invoke({...})

        # Convert to PipelineSpec
        spec = PipelineSpec(...)
        return Pipeline(spec)
```

**Unique code**: ~80 lines
**Effort**: ~1 day (including prompt engineering)

### Strategy 3: Agent/Tools
```python
class AgentBasedOrchestrator(Orchestrator):
    def build_pipeline(self, detection, config):
        # Create agent with tools
        agent = create_react_agent(llm, self.tools, prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools)

        # Invoke agent
        result = executor.invoke({...})

        # Parse output to PipelineSpec
        spec = self._parse_agent_output(result['output'])
        return Pipeline(spec)
```

**Unique code**: ~100 lines
**Effort**: ~1.5 days (most complex parsing logic)

## Unified Configuration Design

All three strategies can use the **same YAML structure** with just one parameter change:

```yaml
# configs/adaptive_experiment.yaml

mode: adaptive

# Detection stage (100% shared)
detection_stage:
  type: clip
  model: openai/clip-vit-base-patch32
  scene_types:
    - a photo of a person face
    - a photo of a famous landmark
    - a photo of a product
    - a general photograph

# Orchestration (only 'strategy' field differs)
orchestration:
  strategy: rule_based  # OR 'llm' OR 'agent_tools'

  # Optional: LLM config (only used if strategy=llm or agent_tools)
  model: gpt-4o-mini
  temperature: 0

  # Pipeline registry (used by rule_based, can inform LLM strategies)
  pipeline_registry:
    face:
      primary: arcface
      secondary: sift
      fusion: weighted_average
      weights: {arcface: 0.7, sift: 0.3}

    landmark:
      primary: dinov2
      secondary: sift
      fusion: weighted_average
      weights: {dinov2: 0.6, sift: 0.4}

    # ... other scene types

# Dataset, output configs (100% shared)
dataset:
  name: holidays
  config_path: configs/dataset.holidays.yaml

output:
  experiment_dir: results/adaptive_experiment
```

**Key insight**: The `pipeline_registry` serves dual purpose:
- **Rule-based**: Direct lookup table
- **LLM/Agent**: Examples to guide decision-making

## Unified Runner Design

Single runner that switches strategies based on config:

```python
# sim_bench/orchestration/adaptive_runner.py

class AdaptiveExperimentRunner:
    """Unified runner supporting all orchestration strategies."""

    def __init__(self, config_path: Path):
        self.config = yaml.safe_load(config_path.read_text())

        # Create detector (same for all strategies)
        self.detector = self._create_detector(self.config['detection_stage'])

        # Create orchestrator (strategy-specific, but same interface)
        self.orchestrator = self._create_orchestrator(self.config['orchestration'])

    def _create_orchestrator(self, config: Dict[str, Any]) -> Orchestrator:
        """Factory method - switch based on strategy."""
        strategy = config.get('strategy', 'rule_based')

        if strategy == 'rule_based':
            from .rule_based import RuleBasedOrchestrator
            return RuleBasedOrchestrator(config)

        elif strategy == 'llm':
            from .llm_based import LLMOrchestrator
            return LLMOrchestrator(config)

        elif strategy == 'agent_tools':
            from .agent_based import AgentBasedOrchestrator
            return AgentBasedOrchestrator(config)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def run(self, image_paths: List[str]) -> Dict[str, Any]:
        """Run experiment - same workflow for all strategies."""

        # Stage 1: Detect scene type (same for all)
        detections = [self.detector.detect(img) for img in sample_images]
        aggregated_detection = self._aggregate_detections(detections)

        # Stage 2: Build pipeline (strategy-specific, but same interface)
        pipeline = self.orchestrator.build_pipeline(
            aggregated_detection,
            self.config
        )

        # Stage 3-4: Extract features & compute similarities (same for all)
        features = pipeline.extract_features(image_paths)
        similarity_matrix = pipeline.compute_similarity_matrix(features)

        return {...}
```

**Effort**: ~1 day (mostly already shown in design doc)

## Code Organization

```
sim_bench/orchestration/
├── __init__.py
│
├── base.py                    # Shared abstractions (80% of code)
│   ├── Signal (ABC)
│   ├── Pipeline
│   ├── PipelineSpec
│   ├── DetectionResult
│   ├── Detector (ABC)
│   └── Orchestrator (ABC)
│
├── signals/                   # Shared signal library
│   ├── __init__.py
│   ├── arcface.py            # ArcFaceSignal
│   ├── dinov2.py             # DINOv2Signal
│   ├── sift.py               # SIFTSignal
│   ├── phash.py              # PerceptualHashSignal
│   ├── color.py              # ColorHistogramSignal
│   └── openclip.py           # OpenCLIPSignal
│
├── detectors.py               # Shared detectors
│   ├── CLIPZeroShotDetector
│   └── YOLODetector
│
├── fusion.py                  # Shared fusion logic
│   ├── FusionEngine
│   ├── WeightedAverageFusion
│   ├── CascadeFusion
│   └── VotingFusion
│
├── orchestrators/             # Strategy-specific (20% of code)
│   ├── __init__.py
│   ├── rule_based.py         # 50 lines - RuleBasedOrchestrator
│   ├── llm_based.py          # 80 lines - LLMOrchestrator
│   └── agent_based.py        # 100 lines - AgentBasedOrchestrator
│
└── adaptive_runner.py         # Unified runner with factory pattern
```

## Effort Breakdown

### Phase 1: Shared Infrastructure (One-time, ~8.5 days)
| Component | Effort | Notes |
|-----------|--------|-------|
| Base abstractions | 2 days | Signal, Pipeline, Detector, Orchestrator |
| Signal library (6 signals) | 3 days | Can reuse existing feature extractors |
| Fusion engine | 1 day | Weighted, cascade, voting |
| Detectors (CLIP + YOLO) | 1.5 days | CLIP faster to implement |
| Pipeline builder | 1 day | Construction + feature extraction |
| **Subtotal** | **8.5 days** | **Supports all strategies** |

### Phase 2: Strategy Implementations (Incremental, ~3 days total)
| Strategy | Unique Code | Effort | Dependencies |
|----------|-------------|--------|--------------|
| Rule-based | 50 lines | 0.5 days | None |
| LLM-based | 80 lines | 1 day | OpenAI API key |
| Agent/Tools | 100 lines | 1.5 days | LangChain + OpenAI |
| **Subtotal** | **230 lines** | **3 days** | **Can implement incrementally** |

### Phase 3: Integration & Testing (2 days)
| Task | Effort | Notes |
|------|--------|-------|
| Unified runner | 0.5 days | Factory pattern + config parsing |
| Configuration examples | 0.5 days | YAML configs for all strategies |
| Unit tests | 0.5 days | Test each strategy independently |
| Integration tests | 0.5 days | End-to-end workflow tests |
| **Subtotal** | **2 days** | **Ensures quality** |

### **Total Effort: ~13.5 days (2.7 weeks)**

## Incremental Implementation Strategy

The beauty of this architecture is that you can implement strategies **incrementally**:

### Week 1: Core Infrastructure + Rule-Based (MVP)
```
Day 1-2: Base abstractions
Day 3-5: Signal library (start with 3 signals: DINOv2, SIFT, ArcFace)
Day 6: Fusion engine (weighted average only)
Day 7: CLIP detector + Rule-based orchestrator
```

**Deliverable**: Working adaptive system with rule-based orchestration

### Week 2: Complete Signals + LLM Strategy
```
Day 8-9: Complete signal library (add remaining 3 signals)
Day 10: Add YOLO detector
Day 11: Implement LLM orchestrator
Day 12: Add cascade + voting fusion
```

**Deliverable**: Full feature set + LLM orchestration option

### Week 3: Agent Strategy + Polish
```
Day 13-14: Implement agent/tools orchestrator
Day 15: Integration testing + bug fixes
Day 16: Documentation + examples
```

**Deliverable**: All three strategies + production-ready code

## Cost Comparison: Development vs Runtime

### Development Cost
| Strategy | Implementation Effort | Incremental Cost |
|----------|----------------------|------------------|
| Rule-based | 8.5 days (shared) + 0.5 days | **9 days total** |
| LLM-based | 8.5 days (shared) + 1 day | **+1 day** (if rule-based exists) |
| Agent/Tools | 8.5 days (shared) + 1.5 days | **+1.5 days** (if rule-based exists) |

**Key insight**: After implementing rule-based, adding LLM or Agent strategies costs only **1-1.5 additional days each**.

### Runtime Cost (per 10K images)
| Strategy | Development Time | Runtime Cost | Latency | Deterministic |
|----------|-----------------|--------------|---------|---------------|
| Rule-based | 9 days | **$0** | ~0ms | ✅ Yes |
| LLM-based | +1 day | **~$0.001** | ~200ms | ❌ No |
| Agent/Tools | +1.5 days | **~$0.01** | ~500ms | ❌ No |

## Recommended Implementation Approach

### Option A: Minimal (Rule-Based Only)
**Effort**: 9 days
**Cost**: $0 runtime
**When to choose**: Production use, deterministic behavior required

### Option B: Dual Strategy (Rule-Based + LLM)
**Effort**: 10 days (+11% vs Option A)
**Cost**: $0 for rule-based, ~$0.001 for LLM experimentation
**When to choose**: Want flexibility for edge cases without agent complexity

### Option C: Full Suite (All Three)
**Effort**: 13.5 days (+50% vs Option A)
**Cost**: Mix of $0 (rule-based) and ~$0.001-0.01 (LLM/agent)
**When to choose**: Research environment, maximum flexibility

## Code Reuse Analysis

Here's the breakdown of code sharing:

```
Total lines of code: ~1,800 lines

Shared infrastructure:
  - base.py: 200 lines
  - signals/: 600 lines (6 signals × 100 lines)
  - fusion.py: 150 lines
  - detectors.py: 200 lines
  - adaptive_runner.py: 200 lines
  Subtotal: 1,350 lines (75%)

Strategy-specific:
  - rule_based.py: 50 lines (2.8%)
  - llm_based.py: 80 lines (4.4%)
  - agent_based.py: 100 lines (5.6%)
  Subtotal: 230 lines (12.8%)

Tests + config:
  - tests/: 150 lines
  - configs/: 70 lines
  Subtotal: 220 lines (12.2%)
```

**Code reuse: 87.2%** when implementing all three strategies!

## Switching Between Strategies

Users can switch strategies with **one line change** in config:

```yaml
# Use rule-based (deterministic, free)
orchestration:
  strategy: rule_based

# Switch to LLM (flexible, ~$0.001 per run)
orchestration:
  strategy: llm
  model: gpt-4o-mini

# Switch to agent (most flexible, ~$0.01 per run)
orchestration:
  strategy: agent_tools
  model: gpt-4o-mini
```

Same command works for all:
```bash
python -m sim_bench.cli \
  --config configs/adaptive_experiment.yaml \
  --dataset holidays
```

## Risk Analysis

### Low Risk ✅
- **Shared infrastructure**: Well-defined abstractions, similar to existing `experiment_runner.py`
- **Rule-based orchestrator**: Simple dictionary lookup, no external dependencies
- **Signal library**: Can reuse existing feature extractors (DINOv2, ResNet50, etc.)

### Medium Risk ⚠️
- **LLM orchestrator**: Depends on OpenAI API stability, but failure mode is graceful (fallback to rule-based)
- **Detection stage**: CLIP/YOLO accuracy for scene classification (can test before committing)

### Higher Risk 🔴
- **Agent/Tools**: Most complex, LangChain API changes frequently, harder to debug
- **Fusion strategies**: Need to validate that weighted/cascade/voting actually improve results vs single-signal baselines

**Mitigation**: Implement in order (rule-based → LLM → agent), validate each before moving to next.

## Comparison to Current Architecture

### Current `experiment_runner.py`
```python
# Single method, static pipeline
class ExperimentRunner:
    def run_single_method(self, method_name: str):
        # 1. Load method config
        # 2. Extract features
        # 3. Compute distances
        # 4. Compute rankings
        # 5. Evaluate metrics
        # 6. Save results
```

**Limitation**: Can only run one method at a time, no multi-signal fusion.

### Proposed `AdaptiveExperimentRunner`
```python
# Multiple signals, dynamic pipeline building
class AdaptiveExperimentRunner:
    def run(self, image_paths: List[str]):
        # 1. Detect scene type (NEW)
        # 2. Build specialized pipeline (NEW)
        # 3. Extract features (multi-signal) (ENHANCED)
        # 4. Fuse similarities (NEW)
        # 5. Evaluate metrics (SAME)
        # 6. Save results (SAME)
```

**Code overlap**: ~40% of existing `experiment_runner.py` logic can be reused (feature extraction, metrics evaluation, result saving).

## Final Recommendation

### Implement in Three Phases:

**Phase 1 (MVP)**: Shared infrastructure + rule-based orchestrator
**Effort**: 9 days
**Delivers**: Working adaptive system with 0 runtime cost
**Risk**: Low ✅

**Phase 2 (Optional)**: Add LLM orchestrator
**Effort**: +1 day
**Delivers**: Flexibility for edge cases at negligible cost (~$0.001/run)
**Risk**: Medium ⚠️

**Phase 3 (Research)**: Add agent/tools orchestrator
**Effort**: +1.5 days
**Delivers**: Maximum flexibility for complex scenarios
**Risk**: Higher 🔴

### Why This Works

1. **Strategy Pattern**: All orchestrators implement the same `Orchestrator` interface
2. **Factory Pattern**: `AdaptiveExperimentRunner` creates the right orchestrator based on config
3. **Dependency Inversion**: High-level runner depends on abstractions, not concrete implementations
4. **Open/Closed Principle**: Easy to add new strategies without modifying existing code

### Conclusion

**Yes, a unified architecture is very feasible.** The key is that:
- **87% of code is shared** across all strategies
- **Only the decision-making logic differs** (50-100 lines per strategy)
- **Same config structure** works for all (just change `strategy` field)
- **Incremental implementation** possible - start with rule-based, add others later
- **Total effort: 9-13.5 days** depending on how many strategies you implement

The architecture naturally supports all three approaches with minimal duplication because the **core problem is the same** (build a pipeline from signals) - only the **method of deciding which signals to use** differs.
