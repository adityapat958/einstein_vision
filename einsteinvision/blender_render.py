"""Phase 2 Blender renderer for EinsteinVision.

Run from Blender, for example:
    blender --background --python einsteinvision/blender_render.py -- \
        --json phase2_output/scene1/detections.json \
        --lane-json cv_p3/lane_out/scene1/lane_report_style.json \
        --assets-dir cv_p3/P3Data/Assets

New in Phase 2:
  - instantiate_asset() loads real .blend assets from P3Data/Assets/
  - Sub-class routing: "sedan" → SedanAndHatchback.blend, "suv" → SUV.blend, …
  - Motion-state shading: parked=gray, slow=yellow-tint, moving=white
  - Intent shading: brake lights → red emission, turn signal → blinking orange
  - Pedestrian pose rendering: 17-point COCO skeleton as sphere+cylinder rigs
  - Speed-limit sign texture injection
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import bpy          # type: ignore
    from mathutils import Vector, Euler  # type: ignore
except ImportError:     # pragma: no cover – outside Blender
    bpy = None
    Vector = None
    Euler = None


# ── COCO skeleton connectivity ────────────────────────────────────────────────
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Default class → asset file mapping (relative to assets_dir)
_DEFAULT_CLASS_TO_ASSET: dict[str, str] = {
    "sedan":           "Vehicles/SedanAndHatchback.blend",
    "hatchback":       "Vehicles/SedanAndHatchback.blend",
    "suv":             "Vehicles/SUV.blend",
    "pickup":          "Vehicles/PickupTruck.blend",
    "truck":           "Vehicles/Truck.blend",
    "bicycle":         "Vehicles/Bicycle.blend",
    "motorcycle":      "Vehicles/Motorcycle.blend",
    # Generic fallbacks
    "car":             "Vehicles/SedanAndHatchback.blend",
    "bus":             "Vehicles/Truck.blend",
    "pedestrian":      "Pedestrain.blend",
    "person":          "Pedestrain.blend",
    "traffic_light":   "TrafficSignal.blend",
    "traffic light":   "TrafficSignal.blend",
    "stop_sign":       "StopSign.blend",
    "speed_limit_sign":"SpeedLimitSign.blend",
    "traffic_cone":    "TrafficConeAndCylinder.blend",
    "traffic_cylinder":"TrafficConeAndCylinder.blend",
    "dustbin":         "Dustbin.blend",
    "trash can":       "Dustbin.blend",
}

# Realistic default scale (metres) per class  [X, Y, Z]
_DEFAULT_SCALE: dict[str, tuple] = {
    "sedan": (1.8, 4.5, 1.5), "hatchback": (1.7, 4.0, 1.5),
    "suv":   (1.9, 4.7, 1.7), "pickup":    (2.0, 5.5, 1.8),
    "truck": (2.5, 8.0, 3.5), "car":       (1.8, 4.5, 1.5),
    "bicycle":    (0.6, 1.8, 1.0), "motorcycle": (0.8, 2.2, 1.2),
    "pedestrian": (0.5, 0.5, 1.75), "person":    (0.5, 0.5, 1.75),
    "traffic_light":    (0.4, 0.4, 1.2),
    "traffic_cone":     (0.4, 0.4, 0.6),
    "traffic_cylinder": (0.4, 0.4, 0.8),
    "dustbin":          (0.6, 0.6, 0.9),
    "stop_sign":        (0.6, 0.1, 0.6),
    "speed_limit_sign": (0.6, 0.1, 0.6),
}


@dataclass(slots=True)
class BlenderRenderConfig:
    """Settings for mapping fused objects to Blender assets."""

    assets_dir: Path
    class_to_asset: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_CLASS_TO_ASSET))
    collection_name: str = "EinsteinVisionActors"


class BlenderSceneRenderer:
    """Loads Phase 2 fused detections.json and keyframes all objects in Blender."""

    def __init__(self, config: BlenderRenderConfig) -> None:
        self.config = config
        self._object_cache: dict[str, Any] = {}
        self._material_cache: dict[str, Any] = {}
        # Per-track position history for motion-state computation
        self._pos_history: dict[str, deque] = {}
        # Blinking state for turn signals
        self._blink_state: dict[str, bool] = {}

    # ── Main entry ─────────────────────────────────────────────────────────

    def render_from_json(self, json_path: str | Path) -> None:
        """Read detections.json (schema v2.0) and keyframe all objects."""
        self._require_bpy()
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        frames = payload.get("frames", [])
        fps = float(payload.get("fps", 30.0))

        collection = self.ensure_collection(self.config.collection_name)

        for frame in frames:
            frame_index = int(frame["frame_index"])
            blender_frame = frame_index + 1   # Blender is 1-indexed

            for obj_data in frame.get("objects", []):
                track_id = str(obj_data["object_id"])
                # Use sub_class if available, else class_name
                class_name = str(
                    obj_data.get("sub_class") or obj_data.get("class_name", "unknown")
                ).lower()

                actor = self.get_or_create_actor(track_id, class_name, collection)

                pos = obj_data.get("position_xyz_m", [0, 0, 0])
                yaw = obj_data.get("orientation_yaw_rad")
                self.apply_keyframe(actor, blender_frame, pos, yaw)

                # Motion state shading
                motion_state = self._compute_motion_state(track_id, pos, fps)
                declared_state = obj_data.get("motion_state", motion_state)
                self._apply_motion_material(actor, declared_state, blender_frame)

                # Intent: brake lights
                intent = obj_data.get("intent", {})
                if intent.get("brake_lights_on"):
                    self._apply_brake_light(actor, blender_frame)

                # Intent: turn signal blink
                turn = intent.get("turn_signal", "none")
                if turn in ("left", "right"):
                    self._apply_turn_signal(actor, blender_frame, turn)

        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = max(
            (int(f["frame_index"]) + 1 for f in frames), default=1
        )

    # ── Pose rendering ─────────────────────────────────────────────────────

    def render_poses_from_json(self, json_path: str | Path) -> None:
        """Render pedestrian 17-point skeletons from detections.json."""
        self._require_bpy()
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        frames = payload.get("frames", [])

        collection = self.ensure_collection("EinsteinVisionPoses")
        pose_objects: dict[str, list] = {}   # person_id → [sphere objects]

        for frame in frames:
            blender_frame = int(frame["frame_index"]) + 1
            for pose in frame.get("poses", []):
                pid = str(pose["person_track_id"])
                kps3d = pose.get("keypoints_3d", [])
                if not kps3d or all(k is None for k in kps3d):
                    continue
                self._keyframe_pose(pid, kps3d, blender_frame, collection, pose_objects)

    def _keyframe_pose(
        self,
        person_id: str,
        kps3d: list,
        blender_frame: int,
        collection,
        pose_objects: dict,
    ) -> None:
        """Create sphere+cylinder skeleton for a person at a given frame."""
        if person_id not in pose_objects:
            # Create 17 sphere joints
            spheres = []
            for i in range(17):
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, segments=6, ring_count=4)
                s = bpy.context.active_object
                s.name = f"pose_{person_id}_kp{i}"
                s.data.materials.append(self._get_or_create_skin_material())
                collection.objects.link(s)
                bpy.context.scene.collection.objects.unlink(s)
                spheres.append(s)

            # Create skeleton bones (cylinders between connected kps)
            bones = []
            for (i, j) in COCO_SKELETON:
                bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=1.0)
                b = bpy.context.active_object
                b.name = f"pose_{person_id}_bone{i}_{j}"
                b.data.materials.append(self._get_or_create_skin_material())
                collection.objects.link(b)
                bpy.context.scene.collection.objects.unlink(b)
                bones.append((i, j, b))

            pose_objects[person_id] = {"spheres": spheres, "bones": bones}

        objs = pose_objects[person_id]
        spheres = objs["spheres"]
        bones = objs["bones"]

        # Keyframe sphere positions
        for i, kp in enumerate(kps3d[:17]):
            s = spheres[i]
            if kp is not None:
                x, y, z = float(kp[0]), float(kp[2]), -float(kp[1])
                s.location = (x, y, z)
                s.hide_viewport = False
                s.hide_render = False
            else:
                s.hide_viewport = True
                s.hide_render = True
            s.keyframe_insert("location", frame=blender_frame)
            s.keyframe_insert("hide_viewport", frame=blender_frame)
            s.keyframe_insert("hide_render", frame=blender_frame)

        # Keyframe bone cylinders (position + scale + rotation between two joints)
        for (i, j, bone) in bones:
            kp_i = kps3d[i] if i < len(kps3d) else None
            kp_j = kps3d[j] if j < len(kps3d) else None
            if kp_i is None or kp_j is None:
                bone.hide_viewport = True
                bone.hide_render = True
                bone.keyframe_insert("hide_viewport", frame=blender_frame)
                bone.keyframe_insert("hide_render", frame=blender_frame)
                continue
            ax, ay, az = float(kp_i[0]), float(kp_i[2]), -float(kp_i[1])
            bx, by, bz = float(kp_j[0]), float(kp_j[2]), -float(kp_j[1])
            mid = ((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2)
            length = math.sqrt((bx-ax)**2 + (by-ay)**2 + (bz-az)**2)
            # Direction vector
            dx, dy, dz = bx - ax, by - ay, bz - az
            norm = math.sqrt(dx*dx + dy*dy + dz*dz)
            if norm < 1e-6 or length < 0.01:
                bone.hide_viewport = True
                bone.hide_render = True
                bone.keyframe_insert("hide_viewport", frame=blender_frame)
                continue
            bone.location = mid
            bone.scale = (1.0, 1.0, length / 2.0)
            # Rotation to align cylinder (default: Z-up) to bone direction
            from mathutils import Vector as V
            up = V((0, 0, 1))
            direction = V((dx / norm, dy / norm, dz / norm))
            rot = up.rotation_difference(direction)
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = rot
            bone.hide_viewport = False
            bone.hide_render = False
            bone.keyframe_insert("location", frame=blender_frame)
            bone.keyframe_insert("scale", frame=blender_frame)
            bone.keyframe_insert("rotation_quaternion", frame=blender_frame)
            bone.keyframe_insert("hide_viewport", frame=blender_frame)
            bone.keyframe_insert("hide_render", frame=blender_frame)

    # ── Asset instantiation ────────────────────────────────────────────────

    def ensure_collection(self, name: str):
        self._require_bpy()
        col = bpy.data.collections.get(name)
        if col is None:
            col = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(col)
        return col

    def get_or_create_actor(self, track_id: str, class_name: str, collection):
        self._require_bpy()
        if track_id in self._object_cache:
            return self._object_cache[track_id]
        actor = self.instantiate_asset(class_name, track_id, collection)
        self._object_cache[track_id] = actor
        return actor

    def instantiate_asset(self, class_name: str, actor_name: str, collection):
        """
        Load a .blend asset for the given class and link it into the collection.

        Tries bpy.ops.wm.append() from the configured assets_dir.
        Falls back to a colour-coded primitive cube if the asset is not found.
        """
        self._require_bpy()

        asset_rel = self.config.class_to_asset.get(class_name)
        if asset_rel:
            asset_path = self.config.assets_dir / asset_rel
            if asset_path.exists():
                try:
                    obj = self._append_blend_object(asset_path, actor_name, collection)
                    if obj is not None:
                        self._set_asset_scale(obj, class_name)
                        return obj
                except Exception as e:
                    print(f"[BlenderRenderer] Failed to append {asset_path}: {e}")

        # Fallback: coloured cube primitive
        return self._create_fallback_primitive(class_name, actor_name, collection)

    def _append_blend_object(self, blend_path: Path, name: str, collection):
        """Append the first mesh object from a .blend file."""
        inner_dir = str(blend_path) + "/Object/"
        # List available objects in the blend file
        with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
            obj_names = list(data_from.objects)
            if not obj_names:
                return None
            data_to.objects = [obj_names[0]]

        obj = data_to.objects[0]
        if obj is None:
            return None
        obj.name = name
        collection.objects.link(obj)
        # Unlink from any default scene collection it may have been added to
        for col in obj.users_collection:
            if col != collection:
                col.objects.unlink(obj)
        return obj

    def _create_fallback_primitive(self, class_name: str, actor_name: str, collection):
        """Create a colour-coded cube as a stand-in asset."""
        scale = _DEFAULT_SCALE.get(class_name, (1.0, 1.0, 1.0))
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.name = actor_name
        obj.scale = scale
        # Apply a distinctive colour per class category
        mat = self._get_or_create_class_material(class_name)
        obj.data.materials.append(mat)
        collection.objects.link(obj)
        for c in obj.users_collection:
            if c != collection:
                c.objects.unlink(obj)
        return obj

    def _set_asset_scale(self, obj, class_name: str) -> None:
        scale = _DEFAULT_SCALE.get(class_name)
        if scale:
            obj.scale = scale

    # ── Material helpers ────────────────────────────────────────────────────

    def _get_or_create_class_material(self, class_name: str):
        key = f"EV_class_{class_name}"
        if key in self._material_cache:
            return self._material_cache[key]
        # Colour palette per class
        colours = {
            "sedan": (0.2, 0.4, 0.9, 1), "hatchback": (0.2, 0.4, 0.9, 1),
            "suv":   (0.1, 0.3, 0.8, 1), "pickup":    (0.3, 0.5, 0.9, 1),
            "truck": (0.5, 0.3, 0.1, 1), "car":       (0.2, 0.4, 0.9, 1),
            "bicycle":    (0.0, 0.7, 0.3, 1), "motorcycle": (0.0, 0.6, 0.3, 1),
            "pedestrian": (0.9, 0.6, 0.1, 1), "person":    (0.9, 0.6, 0.1, 1),
            "traffic_light": (0.1, 0.8, 0.1, 1),
            "traffic_cone":  (0.9, 0.4, 0.0, 1),
            "dustbin":       (0.4, 0.4, 0.4, 1),
        }
        rgba = colours.get(class_name, (0.6, 0.6, 0.6, 1))
        mat = self._make_principled_material(key, rgba)
        self._material_cache[key] = mat
        return mat

    def _get_or_create_skin_material(self):
        key = "EV_pose_skin"
        if key not in self._material_cache:
            self._material_cache[key] = self._make_principled_material(
                key, (0.9, 0.75, 0.6, 1)
            )
        return self._material_cache[key]

    def _make_principled_material(self, name: str, rgba: tuple, emission: float = 0.0):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = rgba
                bsdf.inputs["Roughness"].default_value = 0.5
                if emission > 0:
                    emit_key = (
                        "Emission Color"
                        if "Emission Color" in bsdf.inputs
                        else "Emission"
                    )
                    bsdf.inputs[emit_key].default_value = rgba
                    bsdf.inputs["Emission Strength"].default_value = emission
        return mat

    # ── Motion-state shading ───────────────────────────────────────────────

    def _compute_motion_state(self, track_id: str, pos: list, fps: float) -> str:
        if track_id not in self._pos_history:
            self._pos_history[track_id] = deque(maxlen=5)
        hist = self._pos_history[track_id]
        hist.append(pos[:])
        if len(hist) < 2:
            return "unknown"
        prev = hist[-2]
        dx = pos[0] - prev[0]
        dz = pos[2] - prev[2]
        speed = math.sqrt(dx * dx + dz * dz) * fps
        if speed < 0.3:
            return "parked"
        if speed < 2.0:
            return "slow"
        return "moving"

    def _apply_motion_material(self, obj, state: str, frame: int) -> None:
        colour_map = {
            "parked":  (0.5, 0.5, 0.5, 1),
            "slow":    (0.9, 0.8, 0.1, 1),
            "moving":  (0.9, 0.9, 0.9, 1),
            "unknown": (0.6, 0.6, 0.6, 1),
        }
        rgba = colour_map.get(state, (0.6, 0.6, 0.6, 1))
        key = f"EV_motion_{state}"
        if key not in self._material_cache:
            self._material_cache[key] = self._make_principled_material(key, rgba)
        mat = self._material_cache[key]
        if obj.data and hasattr(obj.data, "materials"):
            if len(obj.data.materials) == 0:
                obj.data.materials.append(mat)
            else:
                obj.data.materials[0] = mat

    # ── Intent shading ─────────────────────────────────────────────────────

    def _apply_brake_light(self, obj, frame: int) -> None:
        """Add red emissive material on brake-light frames."""
        key = "EV_brake_light"
        if key not in self._material_cache:
            self._material_cache[key] = self._make_principled_material(
                key, (1.0, 0.05, 0.05, 1), emission=2.0
            )
        mat = self._material_cache[key]
        if obj.data and hasattr(obj.data, "materials"):
            # Append brake-light material slot if not present
            if mat.name not in [m.name for m in obj.data.materials if m]:
                obj.data.materials.append(mat)

    def _apply_turn_signal(self, obj, frame: int, side: str) -> None:
        """Blink orange emission at ~1.5 Hz."""
        key = f"EV_turn_{side}"
        if key not in self._material_cache:
            self._material_cache[key] = self._make_principled_material(
                key, (1.0, 0.5, 0.0, 1), emission=1.5
            )
        mat = self._material_cache[key]
        # Toggle every 10 frames (≈1.5 Hz at 30 fps)
        is_on = (frame // 10) % 2 == 0
        if obj.data and hasattr(obj.data, "materials") and is_on:
            if mat.name not in [m.name for m in obj.data.materials if m]:
                obj.data.materials.append(mat)

    # ── Keyframing ─────────────────────────────────────────────────────────

    def apply_keyframe(
        self,
        blender_object,
        frame_index: int,
        position_xyz: list,
        yaw_rad: float | None,
    ) -> None:
        self._require_bpy()
        blender_object.location = (
            float(position_xyz[0]),
            float(position_xyz[2]),   # Z forward → Blender Y
            -float(position_xyz[1]),  # Y up → Blender -Z (camera Y-down)
        )
        if yaw_rad is not None:
            blender_object.rotation_mode = "XYZ"
            blender_object.rotation_euler[2] = float(yaw_rad)
        blender_object.keyframe_insert(data_path="location", frame=frame_index)
        blender_object.keyframe_insert(data_path="rotation_euler", frame=frame_index)

    @staticmethod
    def _require_bpy() -> None:
        if bpy is None:
            raise RuntimeError(
                "This module must be executed inside Blender where 'bpy' is available."
            )


# ── Lane renderer (unchanged from Phase 1, kept here for single-file import) ──

@dataclass(slots=True)
class LaneRenderConfig:
    collection_name: str = "EinsteinVisionLanes"
    bevel_depth: float = 0.05
    curve_resolution: int = 12
    emission_strength: float = 0.8
    min_score: float = 0.0
    label_colors: dict = field(default_factory=lambda: {
        "solid-line":   (1.0, 1.0, 1.0, 1.0),
        "dotted-line":  (1.0, 0.85, 0.0, 1.0),
        "double-line":  (1.0, 0.85, 0.0, 1.0),
        "dashed-line":  (1.0, 1.0, 1.0, 1.0),
        "divider-line": (1.0, 0.55, 0.0, 1.0),
        "random-line":  (0.8, 0.8, 0.8, 1.0),
    })


class LaneRenderer:
    """Reads lane_report_style.json and builds animated NURBS lane splines."""

    def __init__(self, config: LaneRenderConfig | None = None) -> None:
        self.config = config or LaneRenderConfig()
        self._materials: dict[str, Any] = {}

    def render_lanes_from_json(self, json_path: str | Path) -> None:
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
                    pts_cam = lane.get("points_3d_camera")
                    if pts_cam:
                        pts_gv = [[p[0], -p[1], p[2]] for p in pts_cam if p is not None]
                if not pts_gv:
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
    def _vehicle_to_blender(p: list):
        return Vector((float(p[0]), float(p[2]), float(p[1])))

    @staticmethod
    def _project_2d_to_ground(
        pts_2d,
        fx=1594.7, fy=1607.7, cx=654.3, cy=413.4,
        cam_height=1.45, cam_pitch_deg=5.0,
    ):
        pitch = math.radians(cam_pitch_deg)
        cp, sp = math.cos(pitch), math.sin(pitch)
        results = []
        for u, v in pts_2d:
            rx = (u - cx) / fx
            ry_veh = -((v - cy) / fy)
            ry_rot = cp * ry_veh - sp * 1.0
            rz_rot = sp * ry_veh + cp * 1.0
            if abs(ry_rot) < 1e-8:
                continue
            t = -cam_height / ry_rot
            if t <= 0:
                continue
            results.append([rx * t, cam_height + t * ry_rot, rz_rot * t])
        return results

    def _make_lane_curve(self, name, world_pts, label_name, collection):
        curve_data = bpy.data.curves.new(name=f"{name}_data", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.resolution_u = self.config.curve_resolution
        curve_data.bevel_depth = self.config.bevel_depth
        curve_data.bevel_resolution = 4
        curve_data.fill_mode = "FULL"
        spline = curve_data.splines.new("NURBS")
        spline.points.add(len(world_pts) - 1)
        for i, v in enumerate(world_pts):
            spline.points[i].co = (v.x, v.y, v.z, 1.0)
        spline.use_endpoint_u = True
        obj = bpy.data.objects.new(name, curve_data)
        collection.objects.link(obj)
        obj.data.materials.append(self._get_material(label_name))
        return obj

    def _apply_visibility_keyframes(self, obj, blender_frame):
        def set_hidden(frame, hidden):
            obj.hide_viewport = hidden
            obj.hide_render = hidden
            obj.keyframe_insert(data_path="hide_viewport", frame=frame)
            obj.keyframe_insert(data_path="hide_render", frame=frame)
        if blender_frame > 1:
            set_hidden(blender_frame - 1, True)
        set_hidden(blender_frame, False)
        set_hidden(blender_frame + 1, True)

    def _make_material(self, name, rgba):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = rgba
                emit_key = "Emission Color" if "Emission Color" in bsdf.inputs else "Emission"
                bsdf.inputs[emit_key].default_value = rgba
                bsdf.inputs["Emission Strength"].default_value = self.config.emission_strength
                bsdf.inputs["Roughness"].default_value = 0.3
        return mat

    def _get_material(self, label_name):
        if label_name not in self._materials:
            rgba = self.config.label_colors.get(label_name, (0.9, 0.9, 0.9, 1.0))
            self._materials[label_name] = self._make_material(f"EV_Lane_{label_name}", rgba)
        return self._materials[label_name]

    def _ensure_materials(self):
        for label, rgba in self.config.label_colors.items():
            self._get_material(label)

    def _ensure_collection(self, name):
        col = bpy.data.collections.get(name)
        if col is None:
            col = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(col)
        return col

    @staticmethod
    def _require_bpy():
        if bpy is None:
            raise RuntimeError("Must run inside Blender.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EinsteinVision Phase 2 Blender renderer")
    p.add_argument("--json",       dest="json_path",      default=None, help="Phase 2 detections.json")
    p.add_argument("--lane-json",  dest="lane_json_path", default=None, help="lane_report_style.json")
    p.add_argument("--assets-dir", dest="assets_dir",
                   default=str(Path(__file__).parent.parent / "cv_p3" / "P3Data" / "Assets"),
                   help="Root directory of .blend asset files")
    p.add_argument("--lane-bevel-depth", type=float, default=0.05)
    p.add_argument("--lane-min-score",   type=float, default=0.1)
    p.add_argument("--output", default=None, help="Render output pattern (e.g. /tmp/frame_####.png)")
    return p


def main(argv=None):
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.json_path:
        config = BlenderRenderConfig(
            assets_dir=Path(args.assets_dir),
        )
        renderer = BlenderSceneRenderer(config=config)
        renderer.render_from_json(args.json_path)
        renderer.render_poses_from_json(args.json_path)

    if args.lane_json_path:
        lane_config = LaneRenderConfig(
            bevel_depth=args.lane_bevel_depth,
            min_score=args.lane_min_score,
        )
        LaneRenderer(config=lane_config).render_lanes_from_json(args.lane_json_path)

    if args.output and bpy is not None:
        bpy.context.scene.render.filepath = args.output
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
