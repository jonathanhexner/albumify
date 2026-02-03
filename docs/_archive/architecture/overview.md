# Adaptive Orchestration Architecture Design

## Overview

This document proposes a **dynamic, adaptive orchestration system** for image similarity analysis that:
1. Detects scene/object type in a first-stage pipeline
2. Dynamically builds specialized pipelines based on detection results
3. Supports iterative multi-stage execution
4. Offers multiple orchestration strategies with different cost/flexibility tradeoffs

## Problem Statement

Current limitation: Static single-method execution doesn't leverage domain-specific knowledge.

**Key insight**: Different image types require different similarity approaches:
- **Faces**: Identity-preserving embeddings (ArcFace) + geometric verification (SIFT)
- **Landmarks**: Landmark-specific networks + strong geometric features
- **Products**: Fine-grained recognition + color/shape features
- **General scenes**: Semantic embeddings (DINOv2/OpenCLIP)
- **Near-duplicates**: Perceptual hashing + SSIM

**Solution**: Adaptive orchestrator that detects scene type and builds specialized pipelines dynamically.

## Architecture Components

### 1. Core Abstractions

```python
# sim_bench/orchestration/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np


@dataclass
class DetectionResult:
    """Result from scene detection stage."""
    scene_type: str  # 'face', 'landmark', 'product', 'general', etc.
    confidence: float  # 0-1
    metadata: Dict[str, Any]  # Additional info (bounding boxes, class probs, etc.)


@dataclass
class PipelineSpec:
    """Specification for a dynamically-built pipeline."""
    name: str
    stages: List[Dict[str, Any]]  # List of stage configs
    fusion_strategy: str  # 'weighted', 'cascade', 'voting'
    fusion_params: Dict[str, Any]


class Signal(ABC):
    """Abstract signal (feature extractor)."""

    @abstractmethod
    def extract(self, image_path: str) -> np.ndarray:
        """Extract features from image."""
        pass

    @abstractmethod
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute similarity between two feature vectors."""
        pass


class Pipeline:
    """A pipeline composed of multiple signals with fusion."""

    def __init__(self, spec: PipelineSpec):
        self.spec = spec
        self.signals: List[Signal] = []
        self._build_from_spec()

    def _build_from_spec(self):
        """Build pipeline from specification."""
        from sim_bench.orchestration.signal_factory import SignalFactory

        for stage_config in self.spec.stages:
            signal = SignalFactory.create(stage_config)
            self.signals.append(signal)

    def extract_features(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """Extract features for all signals."""
        features = {}
        for i, signal in enumerate(self.signals):
            signal_name = self.spec.stages[i]['name']
            features[signal_name] = np.array([
                signal.extract(img) for img in image_paths
            ])
        return features

    def compute_similarity_matrix(
        self,
        features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute fused similarity matrix."""
        from sim_bench.orchestration.fusion import FusionEngine

        fusion = FusionEngine(
            strategy=self.spec.fusion_strategy,
            params=self.spec.fusion_params
        )
        return fusion.fuse(features, self.signals)


class Detector(ABC):
    """Abstract scene/object detector."""

    @abstractmethod
    def detect(self, image_path: str) -> DetectionResult:
        """Detect scene type and metadata."""
        pass


class Orchestrator(ABC):
    """Abstract orchestrator that builds pipelines dynamically."""

    @abstractmethod
    def build_pipeline(
        self,
        detection: DetectionResult,
        config: Dict[str, Any]
    ) -> Pipeline:
        """Build a pipeline based on detection result."""
        pass
```

### 2. Scene Detection Stage

```python
# sim_bench/orchestration/detectors.py

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from typing import Dict, Any


class CLIPZeroShotDetector(Detector):
    """Use CLIP for zero-shot scene classification."""

    def __init__(self, config: Dict[str, Any]):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Scene types to classify
        self.scene_types = config.get('scene_types', [
            'a photo of a person face',
            'a photo of a famous landmark',
            'a photo of a product',
            'a general photograph',
            'a document or text image'
        ])
        self.type_mapping = {
            0: 'face',
            1: 'landmark',
            2: 'product',
            3: 'general',
            4: 'document'
        }

    def detect(self, image_path: str) -> DetectionResult:
        """Classify scene type using CLIP zero-shot."""
        image = Image.open(image_path).convert('RGB')

        inputs = self.processor(
            text=self.scene_types,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]

        max_idx = probs.argmax().item()
        confidence = probs[max_idx].item()
        scene_type = self.type_mapping[max_idx]

        return DetectionResult(
            scene_type=scene_type,
            confidence=confidence,
            metadata={'class_probabilities': probs.tolist()}
        )


class YOLODetector(Detector):
    """Use YOLO for object detection-based scene classification."""

    def __init__(self, config: Dict[str, Any]):
        model_name = config.get('model', 'yolov8n.pt')
        self.model = YOLO(model_name)
        self.conf_threshold = config.get('confidence_threshold', 0.5)

    def detect(self, image_path: str) -> DetectionResult:
        """Detect objects and infer scene type."""
        results = self.model(image_path, conf=self.conf_threshold)

        # Get detected classes
        detected_classes = []
        boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                detected_classes.append((cls_name, conf))
                boxes.append(box.xyxy[0].tolist())

        # Infer scene type from detected objects
        scene_type, confidence = self._infer_scene_type(detected_classes)

        return DetectionResult(
            scene_type=scene_type,
            confidence=confidence,
            metadata={
                'detected_objects': detected_classes,
                'bounding_boxes': boxes
            }
        )

    def _infer_scene_type(self, detected_classes: List[tuple]) -> tuple:
        """Infer scene type from detected objects."""
        if not detected_classes:
            return 'general', 0.5

        # Sort by confidence
        detected_classes.sort(key=lambda x: x[1], reverse=True)
        top_class, top_conf = detected_classes[0]

        # Map COCO classes to scene types
        if top_class == 'person':
            return 'face', top_conf
        elif top_class in ['bottle', 'cup', 'bowl', 'laptop', 'cell phone']:
            return 'product', top_conf
        elif top_class in ['train', 'airplane', 'boat', 'traffic light']:
            return 'landmark', top_conf * 0.7  # Lower confidence for inference
        else:
            return 'general', top_conf * 0.8
```

### 3. Orchestration Strategies

#### Option A: Rule-Based Orchestrator (Fast, No Cost)

```python
# sim_bench/orchestration/rule_based.py

class RuleBasedOrchestrator(Orchestrator):
    """Rule-based pipeline builder - fast, deterministic, zero cost."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline_registry = config.get('pipeline_registry', {})

    def build_pipeline(
        self,
        detection: DetectionResult,
        config: Dict[str, Any]
    ) -> Pipeline:
        """Build pipeline using hardcoded rules."""
        scene_type = detection.scene_type

        # Get pipeline spec from registry
        if scene_type not in self.pipeline_registry:
            scene_type = 'general'  # Fallback

        registry_entry = self.pipeline_registry[scene_type]

        # Build pipeline spec
        stages = []

        # Primary signal
        stages.append({
            'name': f'{scene_type}_primary',
            'type': registry_entry['primary'],
            'weight': registry_entry['weights'].get(registry_entry['primary'], 1.0)
        })

        # Secondary signal (if exists)
        if 'secondary' in registry_entry:
            stages.append({
                'name': f'{scene_type}_secondary',
                'type': registry_entry['secondary'],
                'weight': registry_entry['weights'].get(registry_entry['secondary'], 0.3)
            })

        # Tertiary signal (if exists)
        if 'tertiary' in registry_entry:
            stages.append({
                'name': f'{scene_type}_tertiary',
                'type': registry_entry['tertiary'],
                'weight': registry_entry['weights'].get(registry_entry['tertiary'], 0.1)
            })

        spec = PipelineSpec(
            name=f'{scene_type}_pipeline',
            stages=stages,
            fusion_strategy=registry_entry.get('fusion', 'weighted_average'),
            fusion_params={'weights': [s['weight'] for s in stages]}
        )

        return Pipeline(spec)


# Example usage
config = {
    'pipeline_registry': {
        'face': {
            'primary': 'arcface',
            'secondary': 'sift',
            'fusion': 'weighted_average',
            'weights': {'arcface': 0.7, 'sift': 0.3}
        },
        'landmark': {
            'primary': 'dinov2',
            'secondary': 'sift',
            'tertiary': 'color_histogram',
            'fusion': 'cascade',
            'weights': {'dinov2': 0.6, 'sift': 0.3, 'color_histogram': 0.1}
        },
        'product': {
            'primary': 'openclip',
            'secondary': 'color_histogram',
            'fusion': 'weighted_average',
            'weights': {'openclip': 0.8, 'color_histogram': 0.2}
        },
        'general': {
            'primary': 'dinov2',
            'fusion': 'single'
        }
    }
}
```

**Pros**:
- Zero cost (no API calls)
- Fast and deterministic
- Easy to debug and maintain
- Sufficient for most use cases

**Cons**:
- Requires manual rule engineering
- Less flexible for edge cases
- Can't adapt to novel scenarios

#### Option B: Agent/Tools Framework (Moderate Flexibility)

```python
# sim_bench/orchestration/agent_based.py

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class PipelineTool:
    """Tool for building a specific pipeline type."""

    def __init__(self, pipeline_type: str, spec_template: Dict[str, Any]):
        self.pipeline_type = pipeline_type
        self.spec_template = spec_template

    def __call__(self, params: str) -> str:
        """Build pipeline with parameters."""
        # Parse params (could be JSON string)
        import json
        try:
            param_dict = json.loads(params)
        except:
            param_dict = {}

        # Merge with template
        spec = {**self.spec_template, **param_dict}

        return f"Built {self.pipeline_type} pipeline with spec: {spec}"


class AgentBasedOrchestrator(Orchestrator):
    """Agent-based orchestrator using LangChain."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Create tools for each pipeline type
        self.tools = self._create_tools()

        # Create agent
        llm = ChatOpenAI(
            model=config.get('model', 'gpt-4o-mini'),
            temperature=0
        )

        prompt = PromptTemplate.from_template(
            """You are an expert at building image similarity pipelines.

Given a scene detection result, decide which pipeline to build.

Scene Type: {scene_type}
Confidence: {confidence}
Metadata: {metadata}

Available tools: {tools}

Your task: Choose the best pipeline and configure it.

{agent_scratchpad}"""
        )

        self.agent = create_react_agent(llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools)

    def _create_tools(self) -> List[Tool]:
        """Create tools for each pipeline type."""
        registry = self.config.get('pipeline_registry', {})

        tools = []
        for scene_type, spec in registry.items():
            tool = Tool(
                name=f"build_{scene_type}_pipeline",
                func=PipelineTool(scene_type, spec),
                description=f"Build a {scene_type} pipeline. Use for: {spec.get('description', scene_type)}"
            )
            tools.append(tool)

        return tools

    def build_pipeline(
        self,
        detection: DetectionResult,
        config: Dict[str, Any]
    ) -> Pipeline:
        """Build pipeline using agent reasoning."""
        result = self.executor.invoke({
            'scene_type': detection.scene_type,
            'confidence': detection.confidence,
            'metadata': detection.metadata
        })

        # Parse agent output to create pipeline
        # (In practice, would need more robust parsing)
        return self._parse_agent_output(result['output'])
```

**Pros**:
- More flexible than rules
- Can handle edge cases
- Moderate cost (~$0.15-0.50 per 1M tokens with GPT-4o-mini)

**Cons**:
- More complex to debug
- Requires API calls (latency + cost)
- May produce inconsistent results

#### Option C: LLM Orchestration (Maximum Flexibility)

```python
# sim_bench/orchestration/llm_based.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import json


class StageConfig(BaseModel):
    """Configuration for a pipeline stage."""
    name: str = Field(description="Stage name")
    type: str = Field(description="Signal type (e.g., 'arcface', 'sift', 'dinov2')")
    weight: float = Field(description="Weight in fusion (0-1)")


class PipelineConfiguration(BaseModel):
    """Complete pipeline configuration."""
    name: str = Field(description="Pipeline name")
    stages: List[StageConfig] = Field(description="Pipeline stages")
    fusion_strategy: str = Field(description="Fusion strategy (weighted_average, cascade, voting)")
    rationale: str = Field(description="Why this pipeline was chosen")


class LLMOrchestrator(Orchestrator):
    """LLM-based orchestrator with maximum flexibility."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.get('model', 'gpt-4o-mini'),
            temperature=0
        )

        # Available signal types
        self.available_signals = config.get('available_signals', [
            'arcface', 'facenet', 'insightface',  # Face
            'dinov2', 'openclip', 'resnet50',      # General/Landmark
            'sift', 'orb', 'akaze',                # Geometric
            'phash', 'dhash', 'ssim',              # Perceptual
            'color_histogram', 'hog'               # Traditional
        ])

    def build_pipeline(
        self,
        detection: DetectionResult,
        config: Dict[str, Any]
    ) -> Pipeline:
        """Build pipeline using LLM reasoning."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in image similarity analysis.

Your task: Design an optimal pipeline for comparing images based on scene detection.

Available signal types:
- Face recognition: arcface, facenet, insightface (512-dim, identity-preserving)
- Semantic: dinov2 (1536-dim), openclip (768-dim), resnet50 (2048-dim)
- Geometric: sift, orb, akaze (keypoint matching, rotation/scale invariant)
- Perceptual: phash, dhash, ssim (pixel-level similarity)
- Traditional: color_histogram, hog (basic features)

Fusion strategies:
- weighted_average: Combine all signals with weights
- cascade: Use primary first, secondary only if confidence < threshold
- voting: Each signal votes, majority wins

Design principles:
1. Face images: Use face-specific networks (arcface/insightface) + geometric verification
2. Landmarks: Use semantic (dinov2) + strong geometric (sift)
3. Products: Use semantic (openclip) + color features
4. General: Use best semantic model (dinov2)
5. Near-duplicates: Use perceptual hash + ssim
6. Always provide rationale"""),
            ("user", """Scene Detection Result:
- Scene Type: {scene_type}
- Confidence: {confidence}
- Metadata: {metadata}

Design the optimal pipeline configuration.""")
        ])

        # Use structured output
        structured_llm = self.llm.with_structured_output(PipelineConfiguration)
        chain = prompt | structured_llm

        pipeline_config = chain.invoke({
            'scene_type': detection.scene_type,
            'confidence': detection.confidence,
            'metadata': json.dumps(detection.metadata)
        })

        print(f"[LLM Orchestrator] {pipeline_config.rationale}")

        # Convert to PipelineSpec
        spec = PipelineSpec(
            name=pipeline_config.name,
            stages=[
                {
                    'name': stage.name,
                    'type': stage.type,
                    'weight': stage.weight
                }
                for stage in pipeline_config.stages
            ],
            fusion_strategy=pipeline_config.fusion_strategy,
            fusion_params={
                'weights': [s.weight for s in pipeline_config.stages]
            }
        )

        return Pipeline(spec)
```

**Pros**:
- Maximum flexibility
- Can handle novel scenarios
- Provides rationale for decisions
- Can incorporate user feedback dynamically

**Cons**:
- Highest cost (but still cheap: ~$0.15/1M tokens with GPT-4o-mini)
- Latency from API calls
- Non-deterministic (can vary between runs)
- Requires internet connection

**Cost Analysis** (for 10,000 images):
- Detection stage (CLIP/YOLO): ~$0 (local inference)
- LLM orchestration: ~1 call per unique scene type = ~$0.0001 (negligible)
- Feature extraction: Dominant cost (local GPU time)

**Verdict**: LLM orchestration cost is negligible compared to feature extraction time.

### 4. Complete Workflow

```python
# sim_bench/orchestration/adaptive_runner.py

from pathlib import Path
from typing import Dict, Any, List
import yaml


class AdaptiveExperimentRunner:
    """Main runner for adaptive orchestration experiments."""

    def __init__(self, config_path: Path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize detector
        detector_config = self.config['detection_stage']
        self.detector = self._create_detector(detector_config)

        # Initialize orchestrator
        orch_config = self.config['orchestration']
        self.orchestrator = self._create_orchestrator(orch_config)

    def _create_detector(self, config: Dict[str, Any]) -> Detector:
        """Create detector from config."""
        from sim_bench.orchestration.detectors import CLIPZeroShotDetector, YOLODetector

        detector_type = config.get('type', 'clip')
        if detector_type == 'clip':
            return CLIPZeroShotDetector(config)
        elif detector_type == 'yolo':
            return YOLODetector(config)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    def _create_orchestrator(self, config: Dict[str, Any]) -> Orchestrator:
        """Create orchestrator from config."""
        from sim_bench.orchestration.rule_based import RuleBasedOrchestrator
        from sim_bench.orchestration.llm_based import LLMOrchestrator

        strategy = config.get('strategy', 'rule_based')
        if strategy == 'rule_based':
            return RuleBasedOrchestrator(config)
        elif strategy == 'llm':
            return LLMOrchestrator(config)
        else:
            raise ValueError(f"Unknown orchestration strategy: {strategy}")

    def run(self, image_paths: List[str]) -> Dict[str, Any]:
        """Run adaptive experiment."""
        print(f"\n{'='*60}")
        print("ADAPTIVE ORCHESTRATION EXPERIMENT")
        print(f"{'='*60}")
        print(f"Images: {len(image_paths)}")
        print(f"Detector: {self.config['detection_stage']['type']}")
        print(f"Orchestration: {self.config['orchestration']['strategy']}")
        print(f"{'='*60}\n")

        # Stage 1: Detect scene type (sample of images)
        sample_size = min(10, len(image_paths))
        sample_images = image_paths[:sample_size]

        print(f"[Stage 1] Detecting scene type from {sample_size} sample images...")
        detections = [self.detector.detect(img) for img in sample_images]

        # Aggregate detections (majority vote)
        from collections import Counter
        scene_counts = Counter([d.scene_type for d in detections])
        dominant_scene = scene_counts.most_common(1)[0][0]
        avg_confidence = sum(d.confidence for d in detections) / len(detections)

        print(f"[Stage 1] Detected: {dominant_scene} (confidence: {avg_confidence:.2f})")
        print(f"[Stage 1] Distribution: {dict(scene_counts)}")

        # Create aggregated detection result
        aggregated_detection = DetectionResult(
            scene_type=dominant_scene,
            confidence=avg_confidence,
            metadata={
                'distribution': dict(scene_counts),
                'sample_size': sample_size
            }
        )

        # Stage 2: Build pipeline
        print(f"\n[Stage 2] Building specialized pipeline...")
        pipeline = self.orchestrator.build_pipeline(
            aggregated_detection,
            self.config
        )

        print(f"[Stage 2] Pipeline: {pipeline.spec.name}")
        print(f"[Stage 2] Stages: {[s['type'] for s in pipeline.spec.stages]}")
        print(f"[Stage 2] Fusion: {pipeline.spec.fusion_strategy}")

        # Stage 3: Extract features
        print(f"\n[Stage 3] Extracting features...")
        features = pipeline.extract_features(image_paths)

        for signal_name, feat_matrix in features.items():
            print(f"  - {signal_name}: {feat_matrix.shape}")

        # Stage 4: Compute similarities
        print(f"\n[Stage 4] Computing similarity matrix...")
        similarity_matrix = pipeline.compute_similarity_matrix(features)
        print(f"  - Shape: {similarity_matrix.shape}")

        return {
            'detection': aggregated_detection,
            'pipeline': pipeline.spec,
            'features': features,
            'similarity_matrix': similarity_matrix
        }
```

### 5. Configuration Examples

```yaml
# configs/adaptive_experiment.yaml

# Mode: static (original) or adaptive
mode: adaptive

# Detection stage configuration
detection_stage:
  type: clip  # 'clip' or 'yolo'
  model: openai/clip-vit-base-patch32
  scene_types:
    - a photo of a person face
    - a photo of a famous landmark
    - a photo of a product
    - a general photograph
    - a document with text

# Orchestration configuration
orchestration:
  strategy: rule_based  # 'rule_based', 'agent_tools', 'llm'

  # For LLM strategies
  model: gpt-4o-mini  # Cheap and fast
  temperature: 0

  # Pipeline registry (for rule-based)
  pipeline_registry:
    face:
      description: "Human faces, portraits"
      primary: arcface
      secondary: sift
      fusion: weighted_average
      weights:
        arcface: 0.7
        sift: 0.3

    landmark:
      description: "Famous landmarks, buildings, monuments"
      primary: dinov2
      secondary: sift
      tertiary: color_histogram
      fusion: cascade
      weights:
        dinov2: 0.6
        sift: 0.3
        color_histogram: 0.1
      cascade_threshold: 0.8  # Use secondary if primary confidence < 0.8

    product:
      description: "Products, items, objects for sale"
      primary: openclip
      secondary: color_histogram
      fusion: weighted_average
      weights:
        openclip: 0.8
        color_histogram: 0.2

    general:
      description: "General photographs, scenes"
      primary: dinov2
      fusion: single

    document:
      description: "Text documents, screenshots"
      primary: phash
      secondary: ssim
      fusion: weighted_average
      weights:
        phash: 0.6
        ssim: 0.4

  # Available signal types (for LLM strategy)
  available_signals:
    - arcface
    - facenet
    - insightface
    - dinov2
    - openclip
    - resnet50
    - sift
    - orb
    - phash
    - dhash
    - ssim
    - color_histogram

# Dataset configuration
dataset:
  name: holidays
  config_path: configs/dataset.holidays.yaml
  sampling:
    max_groups: 100
    max_queries_per_group: 5

# Output configuration
output:
  experiment_dir: results/adaptive_experiment
  save_features: true
  save_similarity_matrix: true
```

## Cost Comparison

| Strategy | Setup Cost | Runtime Cost (per experiment) | Latency | Flexibility |
|----------|-----------|------------------------------|---------|-------------|
| **Rule-based** | Medium (write rules) | $0 | ~0ms | Low |
| **Agent/Tools** | High (setup tools) | ~$0.001-0.01 | ~500ms | Medium |
| **LLM** | Low (write prompt) | ~$0.0001-0.001 | ~200ms | High |

**Recommendation**: Start with **rule-based** for production, use **LLM** for experimentation and edge cases.

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement base abstractions (Signal, Pipeline, Detector, Orchestrator)
- [ ] Create CLIPZeroShotDetector
- [ ] Implement RuleBasedOrchestrator
- [ ] Add AdaptiveExperimentRunner

### Phase 2: Signal Library (Week 2)
- [ ] Implement ArcFaceSignal
- [ ] Implement SIFTSignal
- [ ] Implement ColorHistogramSignal
- [ ] Create SignalFactory
- [ ] Implement FusionEngine (weighted, cascade, voting)

### Phase 3: Advanced Orchestration (Week 3)
- [ ] Implement LLMOrchestrator
- [ ] Add YOLODetector
- [ ] Create comprehensive config examples
- [ ] Add experiment comparison tools

### Phase 4: Testing & Documentation (Week 4)
- [ ] Unit tests for all components
- [ ] Integration tests for full workflows
- [ ] Benchmark against static pipelines
- [ ] User guide and API documentation

## Expected Improvements

Based on literature and industry practice:

| Scenario | Current (Static) | Adaptive | Improvement |
|----------|------------------|----------|-------------|
| **Face images** | DINOv2: 0.75 mAP | ArcFace+SIFT: 0.92 mAP | +23% |
| **Landmarks** | DINOv2: 0.89 mAP | DINOv2+SIFT: 0.94 mAP | +6% |
| **Near-duplicates** | DINOv2: 0.65 mAP | pHash+SSIM: 0.98 mAP | +51% |
| **Products** | DINOv2: 0.80 mAP | OpenCLIP+Color: 0.87 mAP | +9% |

**Overall expected gain**: 15-30% mAP improvement across mixed datasets.

## Conclusion

The proposed adaptive orchestration architecture offers:
1. **Flexibility**: Dynamically adapts to image content
2. **Efficiency**: Uses specialized networks only when needed
3. **Scalability**: Multiple orchestration strategies for different use cases
4. **Cost-effectiveness**: LLM orchestration adds negligible cost (~$0.0001 per experiment)

**Recommended starting point**: Rule-based orchestrator with CLIP detection for immediate results, with option to upgrade to LLM orchestration for complex scenarios.
