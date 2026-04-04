"""
Pedestrian pose estimation using OpenPifPaf.

Returns 17 COCO keypoints (pixel coordinates + confidence scores) for every
person visible in a frame.  The keypoints are then lifted to 3D world
coordinates by the fusion module using the depth map.

COCO keypoint order (indices 0–16):
    0  nose
    1  left_eye     2  right_eye
    3  left_ear     4  right_ear
    5  left_shoulder 6  right_shoulder
    7  left_elbow   8  right_elbow
    9  left_wrist   10 right_wrist
    11 left_hip     12 right_hip
    13 left_knee    14 right_knee
    15 left_ankle   16 right_ankle

COCO skeleton connectivity (used in Blender rendering):
    (0,1),(0,2),(1,3),(2,4),          # head
    (5,6),(5,7),(7,9),(6,8),(8,10),   # arms
    (5,11),(6,12),(11,12),            # torso
    (11,13),(13,15),(12,14),(14,16)   # legs

Usage
-----
    from pose_estimator import PifPafPoseEstimator

    estimator = PifPafPoseEstimator()
    poses = estimator.predict(frame_bgr)
    for p in poses:
        print(p.person_track_id, p.keypoints_xy.shape)  # (17, 2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# COCO 17-point skeleton edges (index pairs)
COCO_SKELETON: List[tuple] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

NUM_KEYPOINTS = 17


@dataclass
class PoseResult:
    """Raw 2D pose for one person (image coordinates)."""
    person_track_id: str
    keypoints_xy: np.ndarray   # (17, 2) float32, pixel coords
    scores: np.ndarray         # (17,)   float32, per-keypoint confidence
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2] if available


class PifPafPoseEstimator:
    """
    Wraps OpenPifPaf to return 17-point COCO skeleton predictions.

    Falls back to dummy zero keypoints if OpenPifPaf is not installed,
    so the rest of the pipeline can still run.

    Args:
        model_name: OpenPifPaf checkpoint name (default "shufflenetv2k30").
        device:     "cuda", "cpu", or "mps".
        min_score:  Minimum pose confidence to include in results.
    """

    def __init__(
        self,
        model_name: str = "shufflenetv2k30",
        device: str = "cpu",
        min_score: float = 0.2,
    ) -> None:
        self._device = device
        self._model_name = model_name
        self._min_score = min_score
        self._predictor = None
        self._available = False
        self._load()

    def _load(self) -> None:
        try:
            import openpifpaf
            import torch

            self._predictor = openpifpaf.Predictor(
                checkpoint=self._model_name,
                device=self._device,
            )
            self._available = True
            print(f"[PifPafPoseEstimator] Loaded '{self._model_name}' on {self._device}")
        except ImportError:
            print(
                "[PifPafPoseEstimator] OpenPifPaf not installed. "
                "Install with: pip install openpifpaf\n"
                "Pose estimation will return empty results."
            )
        except Exception as e:
            print(f"[PifPafPoseEstimator] Failed to load model: {e}\n"
                  "Pose estimation will return empty results.")

    def predict(self, frame_bgr: np.ndarray) -> List[PoseResult]:
        """
        Run pose estimation on a single BGR frame.

        Args:
            frame_bgr: H×W×3 BGR image.

        Returns:
            List of PoseResult, one per detected person.
        """
        if not self._available or self._predictor is None:
            return []

        try:
            import PIL.Image
            frame_rgb = frame_bgr[:, :, ::-1]
            pil_img = PIL.Image.fromarray(frame_rgb)

            predictions, _, _ = self._predictor.numpy_image(pil_img)

            results = []
            for i, ann in enumerate(predictions):
                if hasattr(ann, "score") and ann.score < self._min_score:
                    continue

                kps = ann.data  # (17, 3) — x, y, score
                if kps.shape[0] < NUM_KEYPOINTS:
                    # Pad if fewer keypoints
                    pad = np.zeros((NUM_KEYPOINTS - kps.shape[0], 3), dtype=np.float32)
                    kps = np.vstack([kps, pad])

                xy = kps[:NUM_KEYPOINTS, :2].astype(np.float32)
                sc = kps[:NUM_KEYPOINTS, 2].astype(np.float32)

                # Bounding box from visible keypoints
                visible = xy[sc > 0.1]
                bbox = None
                if visible.shape[0] >= 2:
                    x1, y1 = visible.min(axis=0).astype(int).tolist()
                    x2, y2 = visible.max(axis=0).astype(int).tolist()
                    bbox = [x1, y1, x2, y2]

                results.append(
                    PoseResult(
                        person_track_id=f"pose_{i}",
                        keypoints_xy=xy,
                        scores=sc,
                        bbox=bbox,
                    )
                )
            return results

        except Exception as e:
            print(f"[PifPafPoseEstimator] Inference error: {e}")
            return []

    def is_available(self) -> bool:
        """Return True if OpenPifPaf loaded successfully."""
        return self._available
