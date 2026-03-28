"""Blender-side renderer for EinsteinVision fused trajectory JSON data.

Run from Blender, for example:
    blender --python src/einsteinvision/blender_render.py -- --json data/output/fused_scene.json

Lane rendering:
    blender --python src/einsteinvision/blender_render.py -- \
        --json data/fused_scene.json --lane-json data/lane_report_style.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except ImportError:  # pragma: no cover - expected outside Blender runtime
    bpy = None
    Vector = None


@dataclass(slots=True)
class BlenderRenderConfig:
    """Settings for mapping fused objects to Blender assets."""

    assets_dir: Path
    class_to_asset: dict[str, str]
    collection_name: str = "EinsteinVisionActors"


class BlenderSceneRenderer:
    """Loads fused frame records and keyframes object transforms in Blender."""

    def __init__(self, config: BlenderRenderConfig) -> None:
        self.config = config
        self._object_cache: dict[str, Any] = {}

    def render_from_json(self, json_path: str | Path) -> None:
        """Reads fused scene JSON and applies frame-by-frame keyframes."""

        self._require_bpy()

        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        frames = payload.get("frames", [])

        collection = self.ensure_collection(self.config.collection_name)
        for frame in frames:
            frame_index = int(frame["frame_index"])
            for obj in frame.get("objects", []):
                object_ref = self.get_or_create_actor(
                    track_id=str(obj["object_id"]),
                    class_name=str(obj["class_name"]),
                    collection=collection,
                )
                self.apply_keyframe(
                    blender_object=object_ref,
                    frame_index=frame_index,
                    position_xyz=obj["position_xyz_m"],
                    yaw_rad=obj.get("orientation_yaw_rad"),
                )

    def ensure_collection(self, name: str):
        """Ensures a target collection exists in the active Blender scene."""

        self._require_bpy()
        collection = bpy.data.collections.get(name)
        if collection is None:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
        return collection

    def get_or_create_actor(self, track_id: str, class_name: str, collection):
        """Returns actor object for track ID, creating it from mapped asset if needed."""

        self._require_bpy()

        if track_id in self._object_cache:
            return self._object_cache[track_id]

        actor = self.instantiate_asset(class_name=class_name, actor_name=track_id, collection=collection)
        self._object_cache[track_id] = actor
        return actor

    def instantiate_asset(self, class_name: str, actor_name: str, collection):
        """Instantiates an asset for a semantic class.

        Note:
            Asset import/append internals are intentionally left as a stub.
            You can implement this using `bpy.ops.wm.append`, `bpy.ops.import_scene.*`,
            or Blender's Asset Browser APIs.
        """

        self._require_bpy()

        asset_rel_path = self.config.class_to_asset.get(class_name)
        if asset_rel_path is None:
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            actor = bpy.context.active_object
            actor.name = actor_name
            collection.objects.link(actor)
            bpy.context.scene.collection.objects.unlink(actor)
            return actor

        raise NotImplementedError(
            f"Asset loading for class '{class_name}' mapped to '{asset_rel_path}' is pending."
        )

    def apply_keyframe(
        self,
        blender_object,
        frame_index: int,
        position_xyz: list[float] | tuple[float, float, float],
        yaw_rad: float | None,
    ) -> None:
        """Applies location and yaw keyframes for one frame."""

        self._require_bpy()

        blender_object.location = (float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2]))
        if yaw_rad is not None:
            blender_object.rotation_mode = "XYZ"
            blender_object.rotation_euler[2] = float(yaw_rad)

        blender_object.keyframe_insert(data_path="location", frame=frame_index)
        blender_object.keyframe_insert(data_path="rotation_euler", frame=frame_index)

    @staticmethod
    def _require_bpy() -> None:
        """Ensures script is running inside Blender with bpy available."""

        if bpy is None:
            raise RuntimeError("This module must be executed inside Blender where 'bpy' is available.")


@dataclass(slots=True)
class LaneRenderConfig:
    """Settings for rendering lane detections as 3D spline curves in Blender."""

    collection_name: str = "EinsteinVisionLanes"
    bevel_depth: float = 0.05
    curve_resolution: int = 12
    solid_color: tuple = (1.0, 0.95, 0.7, 1.0)
    dashed_color: tuple = (0.9, 0.9, 0.4, 1.0)
    emission_strength: float = 0.8
    min_score: float = 0.0


class LaneRenderer:
    """Reads lane_report_style.json and builds animated NURBS lane splines in Blender.

    Each (frame, lane) pair becomes one curve object, shown only on its active frame
    via hide_viewport/hide_render keyframes.

    Coordinate mapping (vehicle → Blender world):
        vehicle X (right)    → Blender X
        vehicle Y (up/height) → Blender Z
        vehicle Z (forward)  → Blender Y
    """

    def __init__(self, config: LaneRenderConfig | None = None) -> None:
        self.config = config or LaneRenderConfig()
        self._materials: dict[str, Any] = {}

    def render_lanes_from_json(self, json_path: str | Path) -> None:
        """Reads lane JSON and creates one NURBS curve object per (frame, lane)."""

        self._require_bpy()
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        frames = payload.get("frames", [])

        collection = self._ensure_collection(self.config.collection_name)
        self._ensure_materials()

        max_frame = 1
        for frame in frames:
            frame_idx = int(frame["frame_idx"])
            max_frame = max(max_frame, frame_idx + 1)
            for lane_i, lane in enumerate(frame.get("lanes", [])):
                if float(lane.get("score", 1.0)) < self.config.min_score:
                    continue

                pts_gv = lane.get("points_ground_vehicle")
                if not pts_gv:
                    # Fallback 1: camera frame (X right, Y down, Z forward) → vehicle frame
                    pts_cam = lane.get("points_3d_camera")
                    if pts_cam:
                        pts_gv = [
                            [p[0], -p[1], p[2]] for p in pts_cam if p is not None
                        ]
                if not pts_gv:
                    # Fallback 2: project 2D image points to ground plane
                    pts_2d = lane.get("points_2d")
                    if pts_2d:
                        pts_gv = self._project_2d_to_ground(pts_2d)
                if not pts_gv or len(pts_gv) < 2:
                    continue

                world_pts = [self._vehicle_to_blender(p) for p in pts_gv]
                label = str(lane.get("label_name", "")).lower()
                obj = self._make_lane_curve(
                    name=f"lane_f{frame_idx:06d}_{lane_i:02d}",
                    world_pts=world_pts,
                    label_name=label,
                    collection=collection,
                )
                self._apply_visibility_keyframes(obj, blender_frame=frame_idx + 1)

        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = max_frame

    @staticmethod
    def _vehicle_to_blender(p: list[float]):
        """Converts vehicle-frame point to Blender world coords.

        Vehicle: X=right, Y=up(height), Z=forward
        Blender: X=right, Y=forward,    Z=up
        """
        return Vector((float(p[0]), float(p[2]), float(p[1])))

    @staticmethod
    def _project_2d_to_ground(
        pts_2d: list[list[float]],
        fx: float = 1594.7, fy: float = 1607.7,
        cx: float = 654.3, cy: float = 413.4,
        cam_height: float = 1.45,
        cam_pitch_deg: float = 5.0,
    ) -> list[list[float]]:
        """Project 2D image points onto a flat ground plane in vehicle frame.

        Uses the same math as lane_export.py:image_point_to_ground_vehicle().
        Vehicle frame: X=right, Y=up, Z=forward.
        """
        import math

        pitch = math.radians(cam_pitch_deg)
        cp, sp = math.cos(pitch), math.sin(pitch)
        results = []
        for u, v in pts_2d:
            # Normalised ray in OpenCV camera frame (x right, y down, z forward)
            rx = (u - cx) / fx
            ry = (v - cy) / fy
            # Flip Y: OpenCV (y-down) → vehicle (Y-up)
            ry_veh = -ry
            rz_veh = 1.0
            # Rotate by camera pitch around X axis
            ry_rot = cp * ry_veh - sp * rz_veh
            rz_rot = sp * ry_veh + cp * rz_veh
            if abs(ry_rot) < 1e-8:
                continue
            t = -cam_height / ry_rot
            if t <= 0:
                continue
            results.append([rx * t, cam_height + t * ry_rot, rz_rot * t])
        return results

    def _make_lane_curve(self, name: str, world_pts: list, label_name: str, collection) -> Any:
        """Creates a NURBS curve object from world-space points."""

        curve_data = bpy.data.curves.new(name=f"{name}_data", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.resolution_u = self.config.curve_resolution
        curve_data.bevel_depth = self.config.bevel_depth
        curve_data.bevel_resolution = 4
        curve_data.fill_mode = "FULL"

        spline = curve_data.splines.new("NURBS")
        spline.points.add(len(world_pts) - 1)  # splines.new() already gives 1 point
        for i, v in enumerate(world_pts):
            spline.points[i].co = (v.x, v.y, v.z, 1.0)
        spline.use_endpoint_u = True

        obj = bpy.data.objects.new(name, curve_data)
        collection.objects.link(obj)

        mat_key = "dashed" if any(kw in label_name for kw in ("dash", "dot")) else "solid"
        obj.data.materials.append(self._materials[mat_key])
        return obj

    def _apply_visibility_keyframes(self, obj: Any, blender_frame: int) -> None:
        """Hides obj on all frames except blender_frame."""

        def set_hidden(frame: int, hidden: bool) -> None:
            obj.hide_viewport = hidden
            obj.hide_render = hidden
            obj.keyframe_insert(data_path="hide_viewport", frame=frame)
            obj.keyframe_insert(data_path="hide_render", frame=frame)

        if blender_frame > 1:
            set_hidden(blender_frame - 1, True)
        set_hidden(blender_frame, False)
        set_hidden(blender_frame + 1, True)

    def _ensure_materials(self) -> None:
        """Creates or retrieves emissive lane materials."""

        def make_mat(name: str, rgba: tuple) -> Any:
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name=name)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = rgba
                    # "Emission Color" in Blender ≥4.0; "Emission" in ≤3.x
                    emit_key = "Emission Color" if "Emission Color" in bsdf.inputs else "Emission"
                    bsdf.inputs[emit_key].default_value = rgba
                    bsdf.inputs["Emission Strength"].default_value = self.config.emission_strength
                    bsdf.inputs["Roughness"].default_value = 0.3
            return mat

        self._materials["solid"] = make_mat("EV_LaneSolid", self.config.solid_color)
        self._materials["dashed"] = make_mat("EV_LaneDashed", self.config.dashed_color)

    def _ensure_collection(self, name: str) -> Any:
        """Creates or retrieves a Blender scene collection."""

        collection = bpy.data.collections.get(name)
        if collection is None:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
        return collection

    @staticmethod
    def _require_bpy() -> None:
        if bpy is None:
            raise RuntimeError("This module must be executed inside Blender where 'bpy' is available.")


def _build_cli_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for Blender script execution."""

    parser = argparse.ArgumentParser(description="EinsteinVision Blender renderer")
    parser.add_argument("--json", dest="json_path", required=False, default=None, help="Path to fused scene JSON")
    parser.add_argument("--lane-json", dest="lane_json_path", default=None, help="Path to lane_report_style.json")
    parser.add_argument("--assets-dir", dest="assets_dir", default="assets", help="Directory with 3D assets")
    parser.add_argument("--lane-bevel-depth", dest="lane_bevel_depth", type=float, default=0.05, help="Lane tube radius in metres")
    parser.add_argument("--lane-min-score", dest="lane_min_score", type=float, default=0.0, help="Min lane confidence to render")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point when executed as a Blender Python script."""

    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.json_path:
        config = BlenderRenderConfig(
            assets_dir=Path(args.assets_dir),
            class_to_asset={
                "vehicle": "vehicles/generic_car.blend",
                "pedestrian": "pedestrians/generic_pedestrian.blend",
                "traffic_light": "traffic/traffic_light.blend",
                "sign": "traffic/sign.blend",
            },
        )
        renderer = BlenderSceneRenderer(config=config)
        renderer.render_from_json(args.json_path)

    if args.lane_json_path:
        lane_config = LaneRenderConfig(
            bevel_depth=args.lane_bevel_depth,
            min_score=args.lane_min_score,
        )
        lane_renderer = LaneRenderer(config=lane_config)
        lane_renderer.render_lanes_from_json(args.lane_json_path)


if __name__ == "__main__":
    main()
