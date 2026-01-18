# Image Quality Models - Unified Interface

Unified interface for image quality assessment models supporting Siamese, AVA, IQA, and future models.

## Architecture

All models implement `BaseQualityModel` interface with two core methods:
1. `score_image(image_path)` → single image quality score
2. `compare_images(img1, img2)` → pairwise comparison

Models are created via factory pattern from YAML configuration.

## Available Models

### Learned Models

**Siamese E2E** (`siamese`)
- Pairwise ranking model trained on PhotoTriage
- Only implements `compare_images()` (native pairwise)
- Returns prediction (0 or 1) and confidence

**AVA ResNet** (`ava`)
- Aesthetic scoring model (1-10 scale)
- Implements `score_image()` 
- `compare_images()` auto-implemented via score difference

### IQA Baselines

**Rule-Based IQA** (`rule_based_iqa`)
- Combined hand-crafted features
- Overall quality score (0-1 range)

**Individual Metrics**:
- `sharpness_iqa`: Laplacian variance (blur detection)
- `exposure_iqa`: Exposure quality
- `colorfulness_iqa`: Color saturation
- `contrast_iqa`: Contrast measure

## Usage

### Factory Pattern

```python
from sim_bench.image_quality_models.model_factory import create_model

# Create from config
config = {
    'type': 'ava',
    'checkpoint': 'path/to/model.pt',
    'device': 'cpu'
}
model = create_model(config)

# Score single image
score = model.score_image(Path('image.jpg'))

# Compare two images
result = model.compare_images(Path('img1.jpg'), Path('img2.jpg'))
# Returns: {'prediction': 1, 'confidence': 0.85, 'score_img1': 7.2, 'score_img2': 5.8}
```

### Adding New Models

1. Create wrapper class inheriting from `BaseQualityModel`
2. Implement `score_image()` and optionally override `compare_images()`
3. Implement `from_config()` classmethod
4. Register in `MODEL_REGISTRY` in `model_factory.py`

Example:

```python
from sim_bench.image_quality_models.base_model import BaseQualityModel

class MyCustomModel(BaseQualityModel):
    def score_image(self, image_path: Path) -> float:
        # Your scoring logic
        return score
    
    @classmethod
    def from_config(cls, config: Dict):
        return cls(config['checkpoint'], config.get('device', 'cpu'))

# Register
from sim_bench.image_quality_models.model_factory import register_model
register_model('my_model', MyCustomModel)
```

## Benchmark Integration

Models are benchmarked via `scripts/image_quality_utilities/test_model_degradations.py`:

```bash
python scripts/image_quality_utilities/test_model_degradations.py \
    --config configs/image_quality_benchmarks/degradation_test.yaml
```

See `configs/image_quality_benchmarks/README.md` for details.

## Design Principles

1. **Unified Interface**: All models look the same to benchmark code
2. **Config-Driven**: Easy to test different model combinations
3. **Extensible**: Add new models without changing benchmark code
4. **Auto-Implementation**: Models with `score_image()` automatically support `compare_images()`
5. **No Dependencies**: Base interface has minimal dependencies
