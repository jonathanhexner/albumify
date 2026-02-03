"""SixDRepNet head pose estimator wrapper."""

import logging
from typing import Dict, Any

import numpy as np
from PIL import Image

from sixdrepnet import SixDRepNet

from sim_bench.face_pipeline.types import PoseEstimate

logger = logging.getLogger(__name__)


class SixDRepNetEstimator:
    """Estimate head pose using SixDRepNet."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Configuration dict with 'device' key
        """
        self._device = config.get('device', 'cpu')
        self._model = None
        logger.info(f"SixDRepNetEstimator initialized (device={self._device})")

    def _load_model(self):
        """Lazy load SixDRepNet model."""
        if self._model is not None:
            return
        # gpu_id=-1 = CPU, gpu_id>=0 = CUDA device
        # SixDRepNet_Detector handles device placement and transforms internally
        gpu_id = -1 if self._device == 'cpu' else 0
        self._model = SixDRepNet(gpu_id=gpu_id)
        logger.info(f"SixDRepNet model loaded (gpu_id={gpu_id})")

    def estimate_pose(self, image: Image.Image) -> PoseEstimate:
        """
        Estimate head pose from a face image.

        Args:
            image: Cropped face image (PIL)

        Returns:
            PoseEstimate with yaw, pitch, roll in degrees
        """
        self._load_model()

        # Convert PIL to numpy BGR (what SixDRepNet.predict expects)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_rgb = np.array(image)
        img_bgr = img_rgb[:, :, ::-1]  # RGB -> BGR

        # predict() returns (pitch, yaw, roll) as numpy arrays
        pitch, yaw, roll = self._model.predict(img_bgr)

        return PoseEstimate(
            yaw=float(yaw[0]),
            pitch=float(pitch[0]),
            roll=float(roll[0])
        )
