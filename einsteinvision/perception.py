"""Perception pipeline orchestration for frame-level scene understanding.

This module defines lightweight data contracts (dataclasses) and protocol
interfaces for the perception stack, plus a few concrete model wrapper stubs
and a config-driven factory to assemble a :class:`PerceptionPipeline`.

Deep learning model internals are intentionally left unimplemented; you can
plug in YOLOv12, RF-DETR, Depth Anything V2, etc., inside the provided
wrappers without changing the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml


@dataclass(slots=True)
class BoundingBox2D:
    """Axis-aligned pixel bounding box in image coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(slots=True)
class DetectedObject2D:
    """2D object detection output for a single tracked entity."""

    track_id: str
    class_name: str
    confidence: float
    bbox: BoundingBox2D
    orientation_yaw_rad: float | None = None
    # Phase 2 fields
    sub_class: str | None = None          # e.g. "sedan", "suv", "hatchback", "pickup"
    arrow_direction: str | None = None    # "left", "straight", "right", "off" (traffic lights)
    speed_limit_value: int | None = None  # OCR-extracted speed limit integer


@dataclass(slots=True)
class LaneSegmentationOutput:
    """Lane segmentation result for one frame."""

    lane_mask: np.ndarray
    confidence_map: np.ndarray | None = None


@dataclass(slots=True)
class DepthEstimationOutput:
    """Monocular depth estimation output for one frame."""

    depth_map_m: np.ndarray
    scale_hint: float | None = None


@dataclass(slots=True)
class OpticalFlowOutput:
    """Optical flow output between consecutive frames."""

    flow_uv: np.ndarray
    moving_mask: np.ndarray | None = None


@dataclass(slots=True)
class PoseKeypoints:
    """17-point COCO skeleton keypoints for one person (image coordinates)."""

    person_track_id: str
    keypoints_xy: np.ndarray   # shape (17, 2) — pixel coords
    scores: np.ndarray | None = None  # shape (17,) — per-keypoint confidence


@dataclass(slots=True)
class PoseKeypoints3D:
    """17-point COCO skeleton keypoints in 3D world coordinates."""

    person_track_id: str
    keypoints_xyz: np.ndarray  # shape (17, 3) — world coords in meters


@dataclass(slots=True)
class PerceptionFrameOutput:
    """Unified output of all perception stages for a single frame."""

    frame_index: int
    timestamp_s: float
    lanes: LaneSegmentationOutput | None
    objects: list[DetectedObject2D] = field(default_factory=list)
    depth: DepthEstimationOutput | None = None
    optical_flow: OpticalFlowOutput | None = None
    poses: list[PoseKeypoints] = field(default_factory=list)


class LaneSegmenter(Protocol):
    """Protocol for pluggable lane segmentation backends."""

    def predict(self, frame_bgr: np.ndarray) -> LaneSegmentationOutput:
        """Runs lane segmentation on a frame."""


class ObjectDetector(Protocol):
    """Protocol for pluggable object detector backends."""

    def predict(self, frame_bgr: np.ndarray) -> list[DetectedObject2D]:
        """Runs object detection/tracking on a frame."""


class DepthEstimator(Protocol):
    """Protocol for pluggable monocular depth estimators."""

    def predict(self, frame_bgr: np.ndarray) -> DepthEstimationOutput:
        """Runs depth estimation on a frame."""


class OpticalFlowEstimator(Protocol):
    """Protocol for pluggable optical flow estimators."""

    def predict(self, prev_frame_bgr: np.ndarray, frame_bgr: np.ndarray) -> OpticalFlowOutput:
        """Computes optical flow from previous frame to current frame."""


class PerceptionPipeline:
    """Orchestrates perception model execution and aggregates outputs."""

    def __init__(
        self,
        lane_segmenter: LaneSegmenter | None = None,
        object_detector: ObjectDetector | None = None,
        depth_estimator: DepthEstimator | None = None,
        optical_flow_estimator: OpticalFlowEstimator | None = None,
    ) -> None:
        self.lane_segmenter = lane_segmenter
        self.object_detector = object_detector
        self.depth_estimator = depth_estimator
        self.optical_flow_estimator = optical_flow_estimator
        self._previous_frame: np.ndarray | None = None

    def run_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_s: float,
    ) -> PerceptionFrameOutput:
        """Runs all configured perception modules on a single frame."""

        lanes = self.segment_lanes(frame_bgr) if self.lane_segmenter else None
        objects = self.detect_objects(frame_bgr) if self.object_detector else []
        depth = self.estimate_depth(frame_bgr) if self.depth_estimator else None

        optical_flow: OpticalFlowOutput | None = None
        if self.optical_flow_estimator and self._previous_frame is not None:
            optical_flow = self.estimate_optical_flow(self._previous_frame, frame_bgr)

        self._previous_frame = frame_bgr
        return PerceptionFrameOutput(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            lanes=lanes,
            objects=objects,
            depth=depth,
            optical_flow=optical_flow,
        )

    def segment_lanes(self, frame_bgr: np.ndarray) -> LaneSegmentationOutput:
        """Delegates lane segmentation to the configured lane model."""

        if self.lane_segmenter is None:
            raise RuntimeError("No lane segmenter configured.")
        return self.lane_segmenter.predict(frame_bgr)

    def detect_objects(self, frame_bgr: np.ndarray) -> list[DetectedObject2D]:
        """Delegates object detection to the configured detector model."""

        if self.object_detector is None:
            raise RuntimeError("No object detector configured.")
        return self.object_detector.predict(frame_bgr)

    def estimate_depth(self, frame_bgr: np.ndarray) -> DepthEstimationOutput:
        """Delegates depth estimation to the configured depth model."""

        if self.depth_estimator is None:
            raise RuntimeError("No depth estimator configured.")
        return self.depth_estimator.predict(frame_bgr)

    def estimate_optical_flow(
        self,
        prev_frame_bgr: np.ndarray,
        frame_bgr: np.ndarray,
    ) -> OpticalFlowOutput:
        """Delegates optical flow computation to the configured flow model."""

        if self.optical_flow_estimator is None:
            raise RuntimeError("No optical flow estimator configured.")
        return self.optical_flow_estimator.predict(prev_frame_bgr, frame_bgr)


# ---------------------------------------------------------------------------
# Concrete model wrapper stubs
# ---------------------------------------------------------------------------


class Yolo12ObjectDetector(ObjectDetector):
    """YOLOv12-based object detector wrapper (stub).

    This class adapts a YOLOv12 checkpoint to the :class:`ObjectDetector`
    protocol. Implement the internal model loading and inference in
    :meth:`_load_model` and :meth:`predict`.
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self._model = None  # type: ignore[assignment]

    def _load_model(self) -> None:
        """Loads YOLOv12 weights into memory.

        Replace this stub with actual library-specific loading, e.g.:

        - from ultralytics import YOLO
        - self._model = YOLO(self.weights_path)
        """

        raise NotImplementedError("YOLOv12 model loading is not implemented yet.")

    def predict(self, frame_bgr: np.ndarray) -> list[DetectedObject2D]:
        """Runs YOLOv12 inference and maps results to :class:`DetectedObject2D`.

        Args:
            frame_bgr: Input frame in BGR channel order (OpenCV convention).

        Returns:
            A list of :class:`DetectedObject2D` instances.
        """

        raise NotImplementedError("YOLOv12 inference is not implemented yet.")


class RFDetrObjectDetector(ObjectDetector):
    """RF-DETR-based object detector wrapper (stub).

    Use this on the Turing GPU profile where VRAM is plentiful and you want
    higher mAP and better small-object recall.
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        max_detections: int = 500,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self._model = None  # type: ignore[assignment]

    def _load_model(self) -> None:
        """Loads RF-DETR weights.

        Replace this stub with the official RF-DETR repo loading code.
        """

        raise NotImplementedError("RF-DETR model loading is not implemented yet.")

    def predict(self, frame_bgr: np.ndarray) -> list[DetectedObject2D]:
        """Runs RF-DETR inference and returns :class:`DetectedObject2D` detections."""

        raise NotImplementedError("RF-DETR inference is not implemented yet.")


class DepthAnythingEstimator(DepthEstimator):
    """Depth Anything V2-based monocular depth estimator (stub)."""

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cuda",
        target_size: tuple[int, int] = (640, 384),
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = device
        self.target_size = target_size
        self._model = None  # type: ignore[assignment]

    def _load_model(self) -> None:
        """Loads Depth Anything V2 weights.

        Replace with the official Depth Anything V2 initialization code.
        """

        raise NotImplementedError("Depth Anything V2 model loading is not implemented yet.")

    def predict(self, frame_bgr: np.ndarray) -> DepthEstimationOutput:
        """Runs depth estimation and returns a metric depth map.

        Args:
            frame_bgr: Input frame in BGR format.

        Returns:
            :class:`DepthEstimationOutput` with depth_map_m in meters.
        """

        raise NotImplementedError("Depth Anything V2 inference is not implemented yet.")


# ---------------------------------------------------------------------------
# Config-driven factory
# ---------------------------------------------------------------------------


def build_perception_pipeline(
    profile: str,
    config_path: str | Path = "configs/model_config.yaml",
) -> PerceptionPipeline:
    """Builds a :class:`PerceptionPipeline` from a YAML model config.

    Args:
        profile: Name of the profile to load (e.g. "laptop", "turing").
        config_path: Path to ``model_config.yaml``.

    Returns:
        An instance of :class:`PerceptionPipeline` with detector and depth
        estimator wired according to the config.
    """

    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Model config not found at {cfg_path}")

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    profiles_cfg = config.get("profiles", {})
    if profile not in profiles_cfg:
        raise KeyError(f"Profile '{profile}' not found in model_config.yaml")

    prof_cfg = profiles_cfg[profile]

    # Detector
    det_cfg = prof_cfg.get("detector", {})
    det_type = str(det_cfg.get("type", "yolo12")).lower()
    det_weights = det_cfg.get("weights", "")
    det_device = det_cfg.get("device", "cuda")
    det_conf = float(det_cfg.get("conf_threshold", 0.25))
    det_iou = float(det_cfg.get("iou_threshold", 0.45))
    det_max = int(det_cfg.get("max_detections", 300))

    if det_type == "yolo12":
        object_detector: ObjectDetector = Yolo12ObjectDetector(
            weights_path=det_weights,
            device=det_device,
            conf_threshold=det_conf,
            iou_threshold=det_iou,
            max_detections=det_max,
        )
    elif det_type == "rfdetr":
        object_detector = RFDetrObjectDetector(
            weights_path=det_weights,
            device=det_device,
            conf_threshold=det_conf,
            max_detections=det_max,
        )
    else:
        raise ValueError(f"Unsupported detector type '{det_type}' in profile '{profile}'")

    # Depth estimator
    depth_cfg = prof_cfg.get("depth", {})
    depth_type = str(depth_cfg.get("type", "depth_anything_v2")).lower()
    depth_weights = depth_cfg.get("weights", "")
    depth_device = depth_cfg.get("device", "cuda")
    target_size = depth_cfg.get("target_size", [640, 384])
    depth_target_size = (int(target_size[0]), int(target_size[1]))

    if depth_type == "depth_anything_v2":
        depth_estimator: DepthEstimator = DepthAnythingEstimator(
            weights_path=depth_weights,
            device=depth_device,
            target_size=depth_target_size,
        )
    else:
        raise ValueError(f"Unsupported depth type '{depth_type}' in profile '{profile}'")

    return PerceptionPipeline(
        lane_segmenter=None,
        object_detector=object_detector,
        depth_estimator=depth_estimator,
        optical_flow_estimator=None,
    )

