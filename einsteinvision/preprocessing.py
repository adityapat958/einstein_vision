"""Preprocessing components for camera calibration and frame undistortion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(slots=True)
class CalibrationConfig:
    """Configuration for checkerboard-based camera calibration.

    Attributes:
        checkerboard_size: Number of inner corners (columns, rows).
        square_size_m: Physical square size in meters.
        max_frames: Optional cap on processed frames for calibration.
        refine_subpix: Whether to refine corner estimates to sub-pixel precision.
    """

    checkerboard_size: tuple[int, int]
    square_size_m: float
    max_frames: Optional[int] = None
    refine_subpix: bool = True


@dataclass(slots=True)
class CalibrationResult:
    """Result bundle for calibrated camera intrinsics."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    reprojection_error: float


class CameraCalibrator:
    """Calibrates camera intrinsics from a checkerboard video sequence."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        self._result: Optional[CalibrationResult] = None

    @property
    def result(self) -> Optional[CalibrationResult]:
        """Returns the latest calibration result, if available."""

        return self._result

    def calibrate_from_video(self, video_path: str | Path) -> CalibrationResult:
        """Calibrates intrinsics and distortion coefficients from a checkerboard video.

        Args:
            video_path: Path to checkerboard calibration video.

        Returns:
            CalibrationResult containing intrinsic matrix and distortion parameters.

        Raises:
            NotImplementedError: Calibration internals are not implemented yet.
        """

        raise NotImplementedError("Checkerboard calibration implementation pending.")

    def save(self, output_path: str | Path) -> None:
        """Serializes calibration result to disk.

        Args:
            output_path: Target `.npz` or `.json` path.

        Raises:
            ValueError: If calibration has not been computed yet.
            NotImplementedError: Serialization logic is pending.
        """

        if self._result is None:
            raise ValueError("No calibration result available. Run calibrate_from_video first.")
        raise NotImplementedError("Calibration serialization implementation pending.")

    @classmethod
    def load(cls, file_path: str | Path, config: CalibrationConfig) -> "CameraCalibrator":
        """Creates a calibrator instance populated from saved intrinsics.

        Args:
            file_path: Path to saved calibration data.
            config: Calibration config used to initialize the object.

        Returns:
            CameraCalibrator initialized with loaded result.

        Raises:
            NotImplementedError: Deserialization logic is pending.
        """

        raise NotImplementedError("Calibration deserialization implementation pending.")


class FrameUndistorter:
    """Applies lens undistortion to frames using calibrated intrinsics."""

    def __init__(self, calibration: CalibrationResult) -> None:
        self.calibration = calibration

    def undistort_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Undistorts a single BGR frame.

        Args:
            frame_bgr: Input frame as OpenCV BGR image.

        Returns:
            Undistorted frame.

        Raises:
            NotImplementedError: Undistortion internals are not implemented yet.
        """

        raise NotImplementedError("Frame undistortion implementation pending.")

    def undistort_stream(self, frames: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
        """Undistorts an iterable stream of frames.

        Args:
            frames: Iterable of BGR frames.

        Yields:
            Undistorted BGR frames.
        """

        for frame in frames:
            yield self.undistort_frame(frame)
