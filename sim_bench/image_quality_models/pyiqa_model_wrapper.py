"""PyIQA model wrapper - no-reference IQA/aesthetic metrics via the pyiqa library.

One class fronts every pyiqa metric we expose (MANIQA, MUSIQ, HyperIQA, BRISQUE,
NIQE, CLIP-IQA). The metric name comes from the registry ``type`` key, so all of
them register against this single class in ``model_factory.MODEL_REGISTRY``.

Direction is normalized to the ``BaseQualityModel`` contract (higher = better):
pyiqa metrics expose ``lower_better`` (True for BRISQUE / NIQE), and
``score_image`` negates the raw value when that flag is set. ``raw_score`` keeps
the un-flipped value for display (spec-094's comparison table).
"""

import logging
from pathlib import Path
from typing import Dict, List

from sim_bench.image_quality_models.base_model import BaseQualityModel

logger = logging.getLogger(__name__)

# The pyiqa metrics spec-093 exposes. Each registers against PyIQAModel.
PYIQA_METRICS: List[str] = ["maniqa", "musiq", "hyperiqa", "brisque", "niqe", "clipiqa"]


class PyIQAModel(BaseQualityModel):
    """Wraps a single pyiqa metric as a unified quality model."""

    def __init__(self, metric_name: str, device: str = "cpu"):
        """Load a pyiqa metric.

        Args:
            metric_name: pyiqa metric id (e.g. 'maniqa', 'brisque').
            device: 'cpu' or 'cuda'.

        Raises:
            ImportError: if pyiqa is not installed.
            ValueError: if metric_name is not a known pyiqa metric.
        """
        super().__init__(name=f"pyiqa-{metric_name}", device=device)
        try:
            import pyiqa
        except ImportError as e:  # pragma: no cover - exercised via is_available
            raise ImportError(
                "pyiqa is required for PyIQAModel. Install with: pip install pyiqa "
                '(hold numpy: pip install pyiqa "numpy<2")'
            ) from e

        self.metric_name = metric_name
        self._metric = pyiqa.create_metric(metric_name, device=device)
        # pyiqa metrics expose lower_better; True for distortion metrics (BRISQUE/NIQE).
        self.lower_better = bool(getattr(self._metric, "lower_better", False))
        logger.info(
            "Loaded pyiqa metric '%s' (device=%s, lower_better=%s)",
            metric_name, device, self.lower_better,
        )

    def raw_score(self, image_path: Path) -> float:
        """Raw pyiqa score (un-normalized; direction depends on the metric)."""
        return float(self._metric(str(image_path)).item())

    def score_image(self, image_path: Path) -> float:
        """Quality score honoring the higher=better contract.

        For lower_better metrics (BRISQUE/NIQE) the raw value is negated so that
        ranking is consistent with every other model.
        """
        raw = self.raw_score(image_path)
        return -raw if self.lower_better else raw

    @classmethod
    def is_available(cls) -> bool:
        """True if pyiqa can be imported (so callers can grey-out the method)."""
        try:
            import pyiqa  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def from_config(cls, config: Dict) -> "PyIQAModel":
        """Create from a config dict.

        The metric name comes from ``config['metric']`` or, failing that, the
        registry key ``config['type']`` (create_model passes the full config).
        """
        metric_name = config.get("metric") or config.get("type")
        if not metric_name:
            raise ValueError("PyIQAModel config needs 'metric' or 'type'")
        return cls(metric_name, device=config.get("device", "cpu"))
