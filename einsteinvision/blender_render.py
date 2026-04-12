"""EinsteinVision Blender World Renderer.

Builds an incremental 3D world from ego-vehicle perspective using
detection JSONs, lane JSONs, and .blend assets.

Usage:
    # World only (debug scene setup):
    blender --python einsteinvision/blender_render.py

    # World + lanes:
    blender --python einsteinvision/blender_render.py -- \
        --lane-json cv_p3/lane_out/scene1/lane_report_style.json

    # World + detections + lanes:
    blender --python einsteinvision/blender_render.py -- \
        --json phase2_output/scene1/detections.json \
        --lane-json cv_p3/lane_out/scene1/lane_report_style.json \
        --assets-dir P3Data/Assets

    # Headless render to PNG sequence:
    blender --background --python einsteinvision/blender_render.py -- \
        --json phase2_output/scene1/detections.json \
        --lane-json cv_p3/lane_out/scene1/lane_report_style.json \
        --assets-dir P3Data/Assets \
        --output renders/scene1/frame_####.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except ImportError:  # pragma: no cover
    bpy = None
    Vector = None

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# 1:1 metre mapping. Vehicle X->Blender X, Vehicle Z (forward)->Blender Y, Vehicle Y (up)->Blender Z.
WORLD_STRETCH = 1.0

# Class name → asset .blend file (relative to --assets-dir)
CLASS_TO_ASSET: dict[str, str] = {
    "car":             "Vehicles/SedanAndHatchback.blend",
    "sedan":           "Vehicles/SedanAndHatchback.blend",
    "hatchback":       "Vehicles/SedanAndHatchback.blend",
    "suv":             "Vehicles/SUV.blend",
    "pickup":          "Vehicles/PickupTruck.blend",
    "truck":           "Vehicles/Truck.blend",
    "bus":             "Vehicles/Truck.blend",
    "bicycle":         "Vehicles/Bicycle.blend",
    "motorcycle":      "Vehicles/Motorcycle.blend",
    "person":          "Pedestrain.blend",
    "pedestrian":      "Pedestrain.blend",
    "traffic light":   "TrafficSignal.blend",
    "stop sign":       "StopSign.blend",
    "traffic_cone":    "TrafficConeAndCylinder.blend",
    "dustbin":         "Dustbin.blend",
    "trash can":       "Dustbin.blend",
}

# Uniform scale factor to normalize each asset's raw mesh to real-world metres.
# Computed from: target_dimension / raw_mesh_dimension.
# Scale factors to normalize each asset to real-world metres.
# Sedan 0.18 confirmed by user. Others scaled proportionally from their raw mesh dims.
ASSET_NORMALIZE_SCALE: dict[str, float] = {
    "Vehicles/SedanAndHatchback.blend": 0.18,    # raw ~103m -> ~18.5m... TODO: user to verify others
    "Vehicles/SUV.blend":              2.9,       # raw 0.66m -> 1.9m wide
    "Vehicles/PickupTruck.blend":      0.21,      # raw 9.47m -> ~2m wide
    "Vehicles/Truck.blend":            0.00064,   # raw 3951m -> ~2.5m wide
    "Vehicles/Bicycle.blend":          0.16,      # raw 3.67m -> ~0.6m wide
    "Vehicles/Motorcycle.blend":       0.0023,    # raw 351m -> ~0.8m wide
    "Pedestrain.blend":                0.0028,    # raw 178m -> ~0.5m wide
    "TrafficSignal.blend":             1.0,
    "StopSign.blend":                  1.0,
    "TrafficConeAndCylinder.blend":    1.0,
    "Dustbin.blend":                   1.0,
}

# Fallback cube dimensions (metres) for classes without a .blend asset
FALLBACK_SCALE: dict[str, tuple[float, float, float]] = {
    "car":             (1.8, 4.5, 1.5),
    "truck":           (2.5, 8.0, 3.5),
    "bus":             (2.5, 8.0, 3.5),
    "person":          (0.5, 0.5, 1.75),
    "pedestrian":      (0.5, 0.5, 1.75),
    "traffic light":   (0.4, 0.4, 1.2),
    "stop sign":       (0.6, 0.1, 0.6),
}

# YOLO classes that are false positives in driving scenes — skip entirely
SKIP_CLASSES: set[str] = {
    "airplane", "boat", "train", "clock", "cell phone", "kite",
    "bench", "parking meter", "fire hydrant", "potted plant",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Milestone 1 — World Setup
# ═══════════════════════════════════════════════════════════════════════════════

def _load_ego_vehicle(collection) -> Any | None:
    """Load SedanAndHatchback.blend as the ego car, normalized to real-world scale."""
    assets_dir = Path("P3Data/Assets")
    blend_path = assets_dir / "Vehicles/SedanAndHatchback.blend"
    if not blend_path.exists():
        print(f"[EV] Ego asset not found: {blend_path}")
        return None

    try:
        with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)

        meshes = [o for o in data_to.objects if o is not None and o.type not in {"CAMERA", "LIGHT"}]
        if not meshes:
            return None

        parent = bpy.data.objects.new("EgoVehicle", None)
        parent.empty_display_type = "ARROWS"
        parent.empty_display_size = 0.1
        collection.objects.link(parent)

        for obj in meshes:
            obj.name = f"EgoVehicle_{obj.name}"
            collection.objects.link(obj)
            for c in obj.users_collection:
                if c != collection:
                    c.objects.unlink(obj)
            obj.parent = parent

        # 0.018 = normalizes raw ~102m mesh to real ~1.85m sedan width
        s = 0.018
        parent.scale = (s, s, s)
        parent.rotation_euler = (0.0, 0.0, math.radians(90))
        # Place ego behind the detection origin so it doesn't overlap detected cars
        parent.location = (0.0, -3.0, 0.0)
        return parent

    except Exception as e:
        print(f"[EV] Failed to load ego vehicle: {e}")
        return None


def _setup_world() -> None:
    """Clear default scene and build: ground, sky, lighting, camera, ego car."""
    if bpy is None:
        return

    scene = bpy.context.scene

    # ── Clear defaults ────────────────────────────────────────────────────
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # ── Render engine ─────────────────────────────────────────────────────
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.image_settings.file_format = "PNG"

    # ── World collection ──────────────────────────────────────────────────
    col = bpy.data.collections.get("EinsteinVisionWorld")
    if col is None:
        col = bpy.data.collections.new("EinsteinVisionWorld")
        scene.collection.children.link(col)

    # ── Ground plane ──────────────────────────────────────────────────────
    bpy.ops.mesh.primitive_plane_add(size=1000.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.08, 0.08, 0.08, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.85
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.1
    ground.data.materials.append(mat)

    col.objects.link(ground)
    for c in ground.users_collection:
        if c != col:
            c.objects.unlink(ground)

    # ── Sky / world background ────────────────────────────────────────────
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.45, 0.58, 0.72, 1.0)
        bg.inputs["Strength"].default_value = 0.8

    # ── Sun light ─────────────────────────────────────────────────────────
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun_data.angle = math.radians(5.0)
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    col.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(45), math.radians(30), 0.0)

    # ── Fill light ────────────────────────────────────────────────────────
    fill_data = bpy.data.lights.new("Fill", type="SUN")
    fill_data.energy = 0.8
    fill_data.angle = math.radians(15.0)
    fill_obj = bpy.data.objects.new("Fill", fill_data)
    col.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(120), 0.0, math.radians(-45))

    # ── Chase camera ──────────────────────────────────────────────────────
    cam_data = bpy.data.cameras.new("ChaseCam")
    cam_data.lens = 35
    cam_data.clip_end = 2000.0
    cam_obj = bpy.data.objects.new("ChaseCam", cam_data)
    col.objects.link(cam_obj)
    # Further back, higher up, looking forward+down at the ego car
    # Camera transform from user (converted degrees to radians)
    cam_obj.location = (-37.6, 1.46, 11.88)
    cam_obj.rotation_euler = (math.radians(77), math.radians(0), math.radians(-92))
    scene.camera = cam_obj

    # ── Ego vehicle — load real SedanAndHatchback asset ───────────────────
    ego = _load_ego_vehicle(col)
    if ego is None:
        # Fallback cube if asset missing
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        ego = bpy.context.active_object
        ego.name = "EgoVehicle"
        ego.scale = (1.8, 4.5, 1.5)
        ego.location = (0.0, 2.25, 0.75)
        ego_mat = bpy.data.materials.new("EgoMat")
        ego_mat.use_nodes = True
        bsdf = ego_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.05, 0.05, 0.05, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.3
        ego.data.materials.append(ego_mat)
        col.objects.link(ego)
        for c in ego.users_collection:
            if c != col:
                c.objects.unlink(ego)


# ═══════════════════════════════════════════════════════════════════════════════
# Milestone 2 — Lane Renderer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class LaneRenderConfig:
    collection_name: str = "EinsteinVisionLanes"
    bevel_depth: float = 0.025  # metres, will be multiplied by WORLD_STRETCH in Blender units
    curve_resolution: int = 12
    solid_color: tuple = (1.0, 0.95, 0.7, 1.0)
    dashed_color: tuple = (0.9, 0.9, 0.4, 1.0)
    emission_strength: float = 0.8
    min_score: float = 0.0


class LaneRenderer:
    """Renders lane detections as animated NURBS curves, one per (frame, lane).

    Coordinate mapping (vehicle -> Blender, with WORLD_STRETCH):
        vehicle X (right)   * WORLD_STRETCH -> Blender X
        vehicle Z (forward) * WORLD_STRETCH -> Blender Y
        vehicle Y (height)  * WORLD_STRETCH -> Blender Z
    """

    def __init__(self, config: LaneRenderConfig | None = None) -> None:
        self.config = config or LaneRenderConfig()
        self._materials: dict[str, Any] = {}

    def render_lanes_from_json(self, json_path: str | Path) -> None:
        if bpy is None:
            return
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
        bpy.context.scene.frame_end = max(bpy.context.scene.frame_end, max_frame)

    @staticmethod
    def _vehicle_to_blender(p: list[float]):
        """Vehicle (X right, Y up, Z forward) -> Blender (X right, Y forward, Z up)."""
        return Vector((
            float(p[0]),   # lateral X -> Blender X
            float(p[2]),   # forward Z -> Blender Y
            float(p[1]),   # height Y -> Blender Z
        ))

    def _make_lane_curve(self, name: str, world_pts: list, label_name: str, collection) -> Any:
        curve_data = bpy.data.curves.new(name=f"{name}_data", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.resolution_u = self.config.curve_resolution
        curve_data.bevel_depth = self.config.bevel_depth  # in metres, 1:1 with world
        curve_data.bevel_resolution = 4
        curve_data.fill_mode = "FULL"

        spline = curve_data.splines.new("NURBS")
        spline.points.add(len(world_pts) - 1)
        for i, v in enumerate(world_pts):
            spline.points[i].co = (v.x, v.y, v.z, 1.0)
        spline.use_endpoint_u = True

        obj = bpy.data.objects.new(name, curve_data)
        collection.objects.link(obj)

        mat_key = "dashed" if any(kw in label_name for kw in ("dash", "dot")) else "solid"
        obj.data.materials.append(self._materials[mat_key])
        return obj

    def _apply_visibility_keyframes(self, obj: Any, blender_frame: int) -> None:
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
        def make_mat(name: str, rgba: tuple) -> Any:
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

        self._materials["solid"] = make_mat("EV_LaneSolid", self.config.solid_color)
        self._materials["dashed"] = make_mat("EV_LaneDashed", self.config.dashed_color)

    def _ensure_collection(self, name: str) -> Any:
        collection = bpy.data.collections.get(name)
        if collection is None:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
        return collection


# ═══════════════════════════════════════════════════════════════════════════════
# Milestone 3 — Object Detection Renderer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class BlenderRenderConfig:
    assets_dir: Path = field(default_factory=lambda: Path("P3Data/Assets"))
    collection_name: str = "EinsteinVisionActors"


class BlenderSceneRenderer:
    """Loads fused detection JSON and keyframes tracked objects in Blender.

    Features:
        - Lazy asset loading (objects created on first appearance)
        - Track healing (merges broken YOLO IDs by proximity)
        - Scale-based visibility (0,0,0 when hidden, real scale when visible)
        - CONSTANT interpolation for instant pop in/out
    """

    def __init__(self, config: BlenderRenderConfig) -> None:
        self.config = config
        self._object_cache: dict[str, Any] = {}

    def render_from_json(self, json_path: str | Path) -> None:
        if bpy is None:
            return
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        frames = payload.get("frames", [])

        collection = self._ensure_collection(self.config.collection_name)

        # Track healer state: {track_id: (position, class_name)}
        active_tracks: dict[str, tuple[list, str]] = {}

        for frame in frames:
            frame_index = int(frame["frame_index"])
            current_tracks: dict[str, tuple[list, str]] = {}

            for obj_data in frame.get("objects", []):
                class_name = str(obj_data.get("class_name", "unknown")).lower()

                # Skip bogus YOLO detections
                if class_name in SKIP_CLASSES:
                    continue

                # Use sub_class for finer asset selection (sedan, suv, pickup, etc.)
                sub_class = obj_data.get("sub_class")
                lookup_class = str(sub_class).lower() if sub_class else class_name

                raw_id = str(obj_data["object_id"])
                pos = obj_data["position_xyz_m"]

                # ── Track healing ─────────────────────────────────────────
                matched_id = raw_id
                best_dist = 2.5  # max metres a car can jump in one frame
                for active_id, (active_pos, active_class) in active_tracks.items():
                    if active_class == class_name:
                        dist = math.dist(
                            [pos[0], pos[2]],
                            [active_pos[0], active_pos[2]],
                        )
                        if dist < best_dist:
                            best_dist = dist
                            matched_id = active_id

                current_tracks[matched_id] = (pos, class_name)

                # Use lookup_class (sub_class if available) for asset selection
                actor = self._get_or_create_actor(matched_id, lookup_class, collection)
                self._apply_keyframe(
                    actor, frame_index,
                    pos, obj_data.get("orientation_yaw_rad"),
                )

            active_tracks = current_tracks

        # Set timeline range
        if frames:
            max_frame = max(int(f["frame_index"]) for f in frames) + 1
            bpy.context.scene.frame_end = max(bpy.context.scene.frame_end, max_frame)

    # ── Actor management ──────────────────────────────────────────────────

    def _get_or_create_actor(self, track_id: str, class_name: str, collection) -> Any:
        if track_id in self._object_cache:
            return self._object_cache[track_id]

        actor = self._instantiate_asset(class_name, track_id, collection)

        # Start hidden on frame 1
        actor.scale = (0.0, 0.0, 0.0)
        actor.keyframe_insert(data_path="scale", frame=1)

        self._object_cache[track_id] = actor
        return actor

    def _instantiate_asset(self, class_name: str, actor_name: str, collection) -> Any:
        asset_rel = CLASS_TO_ASSET.get(class_name)
        asset_path = self.config.assets_dir / asset_rel if asset_rel else None

        # Try loading real .blend asset
        if asset_path and asset_path.exists():
            try:
                return self._load_blend_asset(asset_path, asset_rel, actor_name, collection)
            except Exception as e:
                print(f"[EV] Failed to load {asset_path}: {e}")

        # Fallback: coloured cube
        return self._create_fallback_cube(class_name, actor_name, collection)

    def _load_blend_asset(self, blend_path: Path, asset_rel: str, name: str, collection) -> Any:
        """Load all mesh/curve objects from a .blend, parent to an empty."""
        with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)

        meshes = [o for o in data_to.objects if o is not None and o.type not in {"CAMERA", "LIGHT"}]
        if not meshes:
            raise ValueError(f"No mesh objects in {blend_path}")

        parent = bpy.data.objects.new(name, None)
        parent.empty_display_type = "ARROWS"
        parent.empty_display_size = 0.5
        collection.objects.link(parent)

        for obj in meshes:
            obj.name = f"{name}_{obj.name}"
            collection.objects.link(obj)
            for c in obj.users_collection:
                if c != collection:
                    c.objects.unlink(obj)
            obj.parent = parent

        # Normalize scale
        normalize = ASSET_NORMALIZE_SCALE.get(asset_rel, 1.0)
        parent["visible_scale"] = (normalize, normalize, normalize)

        # Signs and lights face wrong direction — rotate 180° on Z
        ROTATE_180 = {"TrafficSignal.blend", "StopSign.blend"}
        if asset_rel in ROTATE_180:
            parent.rotation_euler = (0.0, 0.0, math.radians(180))

        return parent

    def _create_fallback_cube(self, class_name: str, actor_name: str, collection) -> Any:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        actor = bpy.context.active_object
        actor.name = actor_name

        dims = FALLBACK_SCALE.get(class_name, (1.0, 1.0, 1.0))
        actor["visible_scale"] = (dims[0], dims[1], dims[2])

        # Give it a distinct colour
        mat = bpy.data.materials.new(f"mat_{class_name}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Colour by class
            colors = {
                "car": (0.2, 0.4, 0.8, 1.0),
                "truck": (0.8, 0.3, 0.1, 1.0),
                "bus": (0.8, 0.5, 0.0, 1.0),
                "person": (0.1, 0.8, 0.3, 1.0),
                "pedestrian": (0.1, 0.8, 0.3, 1.0),
            }
            bsdf.inputs["Base Color"].default_value = colors.get(class_name, (0.5, 0.5, 0.5, 1.0))
        actor.data.materials.append(mat)

        collection.objects.link(actor)
        for c in actor.users_collection:
            if c != collection:
                c.objects.unlink(actor)

        return actor

    # ── Keyframing ────────────────────────────────────────────────────────

    def _apply_keyframe(
        self,
        blender_object: Any,
        frame_index: int,
        position_xyz: list[float],
        yaw_rad: float | None,
    ) -> None:
        # Vehicle (X right, Y up, Z forward) -> Blender (X right, Y forward, Z up)
        blender_object.location = (
            float(position_xyz[0]),   # lateral X -> X
            float(position_xyz[2]),   # forward Z -> Blender Y
            float(position_xyz[1]),   # height Y -> Blender Z
        )
        if yaw_rad is not None:
            blender_object.rotation_mode = "XYZ"
            blender_object.rotation_euler[2] = float(yaw_rad)

        blender_object.keyframe_insert(data_path="location", frame=frame_index)
        blender_object.keyframe_insert(data_path="rotation_euler", frame=frame_index)

        # Pop visible on this frame
        target_scale = blender_object.get("visible_scale", (1.0, 1.0, 1.0))
        blender_object.scale = target_scale
        blender_object.keyframe_insert(data_path="scale", frame=frame_index)

        # Pre-emptively vanish next frame (overridden if object persists)
        blender_object.scale = (0.0, 0.0, 0.0)
        blender_object.keyframe_insert(data_path="scale", frame=frame_index + 1)

        # Force instant transitions (no easing between visible/hidden)
        if blender_object.animation_data and blender_object.animation_data.action:
            for fc in blender_object.animation_data.action.fcurves:
                if fc.data_path == "scale":
                    for kf in fc.keyframe_points:
                        kf.interpolation = "CONSTANT"

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ensure_collection(self, name: str) -> Any:
        collection = bpy.data.collections.get(name)
        if collection is None:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
        return collection


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EinsteinVision Blender World Renderer")
    p.add_argument("--json", dest="json_path", default=None,
                   help="Path to detections JSON (phase2_output/sceneN/detections.json)")
    p.add_argument("--lane-json", dest="lane_json_path", default=None,
                   help="Path to lane_report_style.json")
    p.add_argument("--assets-dir", dest="assets_dir", default="P3Data/Assets",
                   help="Directory containing .blend asset files")
    p.add_argument("--lane-bevel-depth", type=float, default=0.05,
                   help="Lane tube radius in metres (before stretch)")
    p.add_argument("--lane-min-score", type=float, default=0.0,
                   help="Minimum lane confidence to render")
    p.add_argument("--output", default=None,
                   help="Render output path/pattern (e.g. renders/frame_####.png)")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    # Always set up the world
    _setup_world()

    # M3: Detected objects
    if args.json_path:
        config = BlenderRenderConfig(assets_dir=Path(args.assets_dir))
        renderer = BlenderSceneRenderer(config=config)
        renderer.render_from_json(args.json_path)

    # M2: Lanes
    if args.lane_json_path:
        lane_config = LaneRenderConfig(
            bevel_depth=args.lane_bevel_depth,
            min_score=args.lane_min_score,
        )
        LaneRenderer(config=lane_config).render_lanes_from_json(args.lane_json_path)

    # Headless render (only with --output)
    if args.output and bpy is not None:
        bpy.context.scene.render.filepath = args.output
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    import sys
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
