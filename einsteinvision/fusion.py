"""Fusion module for lifting 2D detections to 3D world coordinates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .perception import DetectedObject2D, PerceptionFrameOutput


@dataclass(slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics used for 2D->3D projection."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(slots=True)
class ObjectPose3D:
    """Fused 3D object pose for a single frame."""

    object_id: str
    class_name: str
    position_xyz_m: tuple[float, float, float]
    orientation_yaw_rad: float | None


@dataclass(slots=True)
class FrameFusionRecord:
    """Frame-level fusion output ready for serialization."""

    frame_index: int
    timestamp_s: float
    objects: list[ObjectPose3D] = field(default_factory=list)


class SceneFusion:
    """Fuses perception outputs into world-space 3D trajectories."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        camera_to_world: np.ndarray | None = None,
    ) -> None:
        self.intrinsics = intrinsics
        self.camera_to_world = camera_to_world

    def pixel_to_camera(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """Projects a pixel coordinate and depth into camera coordinates.

        Args:
            u: Pixel x-coordinate.
            v: Pixel y-coordinate.
            depth_m: Depth value in meters.

        Returns:
            A 3-vector `[X, Y, Z]` in camera coordinates (meters).
        """

        x = (u - self.intrinsics.cx) * depth_m / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * depth_m / self.intrinsics.fy
        z = depth_m
        return np.array([x, y, z], dtype=np.float32)

    def camera_to_world_point(self, xyz_camera: np.ndarray) -> np.ndarray:
        """Transforms a camera-space point into world coordinates.

        If no extrinsics are provided, the input point is returned unchanged.
        """

        if self.camera_to_world is None:
            return xyz_camera

        point_h = np.array([xyz_camera[0], xyz_camera[1], xyz_camera[2], 1.0], dtype=np.float32)
        world_h = self.camera_to_world @ point_h
        return world_h[:3]

    def fuse_frame(self, perception_output: PerceptionFrameOutput) -> FrameFusionRecord:
        """Fuses one perception output frame into 3D object poses.

        Notes:
            Requires `perception_output.depth` to be populated with depth in meters.
            Current implementation samples depth at each bounding-box center.
        """

        if perception_output.depth is None:
            raise ValueError("Depth output is required for 3D fusion.")

        depth_map = perception_output.depth.depth_map_m
        fused_objects: list[ObjectPose3D] = []

        for detection in perception_output.objects:
            center_u, center_v = self._bbox_center(detection)
            depth_m = float(depth_map[int(center_v), int(center_u)])
            xyz_cam = self.pixel_to_camera(center_u, center_v, depth_m)
            xyz_world = self.camera_to_world_point(xyz_cam)
            fused_objects.append(
                ObjectPose3D(
                    object_id=detection.track_id,
                    class_name=detection.class_name,
                    position_xyz_m=(float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2])),
                    orientation_yaw_rad=detection.orientation_yaw_rad,
                )
            )

        return FrameFusionRecord(
            frame_index=perception_output.frame_index,
            timestamp_s=perception_output.timestamp_s,
            objects=fused_objects,
        )

    def export_json(self, records: list[FrameFusionRecord], output_path: str | Path) -> None:
        """Exports frame-by-frame fused scene records to JSON."""

        payload = {
            "schema_version": "1.0",
            "frames": [self._record_to_dict(record) for record in records],
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _bbox_center(detection: DetectedObject2D) -> tuple[float, float]:
        """Computes bbox center in pixel coordinates."""

        center_u = (detection.bbox.x_min + detection.bbox.x_max) / 2.0
        center_v = (detection.bbox.y_min + detection.bbox.y_max) / 2.0
        return center_u, center_v

    @staticmethod
    def _record_to_dict(record: FrameFusionRecord) -> dict:
        """Converts a frame fusion dataclass into a JSON-friendly dict."""

        return {
            "frame_index": record.frame_index,
            "timestamp_s": record.timestamp_s,
            "objects": [asdict(obj) for obj in record.objects],
        }
