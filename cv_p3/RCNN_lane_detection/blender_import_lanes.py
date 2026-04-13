import bpy
import json
from mathutils import Vector

# =========================================================
# CONFIG
# =========================================================

JSON_PATH = "/home/alien/cv_p3/lane_out/lane_report_style.json"
CAMERA_NAME = "Camera"
COLLECTION_NAME = "LaneCurves"
CURVE_BEVEL_DEPTH = 0.03
CURVE_RESOLUTION = 12
MATERIAL_SOLID = "LaneSolidMaterial"
MATERIAL_DASHED = "LaneDashedMaterial"

# If no 3D points are available in the JSON, these are used
FALLBACK_Z = 15.0
FALLBACK_SCALE = 0.01


# =========================================================
# Helpers
# =========================================================

def ensure_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def get_or_create_material(name, rgba=(1, 1, 1, 1)):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = rgba
            bsdf.inputs["Roughness"].default_value = 0.4
            bsdf.inputs["Emission Color"].default_value = rgba
            bsdf.inputs["Emission Strength"].default_value = 0.5
    return mat


def cv_camera_to_blender_camera_local(p):
    """
    CV camera coordinates:
      X right, Y down, Z forward
    Blender camera local:
      X right, Y up, Z backward (camera looks along -Z)
    """
    X, Y, Z = p
    return Vector((X, -Y, -Z))


def camera_local_to_world(camera_obj, p_local):
    return camera_obj.matrix_world @ p_local


def make_curve_object(name, world_points, bevel_depth=0.02, resolution=12):
    curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = resolution
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 4

    spline = curve_data.splines.new('POLY')
    spline.points.add(len(world_points) - 1)

    for i, p in enumerate(world_points):
        spline.points[i].co = (p.x, p.y, p.z, 1.0)

    obj = bpy.data.objects.new(name, curve_data)
    return obj


def set_visibility_for_frame(obj, frame_idx):
    """
    Visible only on frame_idx+1 because Blender frames start at 1 typically.
    """
    show_frame = frame_idx + 1
    hide_before = max(1, show_frame - 1)
    hide_after = show_frame + 1

    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=hide_before)
    obj.keyframe_insert(data_path="hide_render", frame=hide_before)

    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_viewport", frame=show_frame)
    obj.keyframe_insert(data_path="hide_render", frame=show_frame)

    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=hide_after)
    obj.keyframe_insert(data_path="hide_render", frame=hide_after)


# =========================================================
# Main
# =========================================================

with open(JSON_PATH, "r") as f:
    data = json.load(f)

scene = bpy.context.scene
lane_col = ensure_collection(COLLECTION_NAME)

camera_obj = bpy.data.objects.get(CAMERA_NAME)
if camera_obj is None:
    raise RuntimeError(f"Camera '{CAMERA_NAME}' not found in Blender scene.")

mat_solid = get_or_create_material(MATERIAL_SOLID, rgba=(1.0, 1.0, 1.0, 1.0))
mat_dashed = get_or_create_material(MATERIAL_DASHED, rgba=(1.0, 1.0, 1.0, 1.0))

max_frame = 1

for frame_data in data["frames"]:
    frame_idx = frame_data["frame_idx"]
    max_frame = max(max_frame, frame_idx + 1)

    for lane_i, lane in enumerate(frame_data["lanes"]):
        label_name = str(lane.get("label_name", "lane")).lower()
        points_3d_camera = lane.get("points_3d_camera", None)
        points_2d = lane.get("points_2d", [])

        world_points = []

        if points_3d_camera is not None:
            for p in points_3d_camera:
                if p is None:
                    continue
                p_local = cv_camera_to_blender_camera_local(p)
                p_world = camera_local_to_world(camera_obj, p_local)
                world_points.append(p_world)
        else:
            # Fallback: camera-facing approximate placement if no depth was exported
            for p in points_2d:
                x_px, y_px = p
                x = (x_px - 960.0) * FALLBACK_SCALE
                y = -(y_px - 540.0) * FALLBACK_SCALE
                z = -FALLBACK_Z
                p_local = Vector((x, y, z))
                p_world = camera_local_to_world(camera_obj, p_local)
                world_points.append(p_world)

        if len(world_points) < 2:
            continue

        obj_name = f"lane_f{frame_idx:06d}_{lane_i:02d}"
        curve_obj = make_curve_object(
            name=obj_name,
            world_points=world_points,
            bevel_depth=CURVE_BEVEL_DEPTH,
            resolution=CURVE_RESOLUTION,
        )
        lane_col.objects.link(curve_obj)

        if "dash" in label_name or "dot" in label_name:
            curve_obj.data.materials.append(mat_dashed)
        else:
            curve_obj.data.materials.append(mat_solid)

        set_visibility_for_frame(curve_obj, frame_idx)

scene.frame_start = 1
scene.frame_end = max_frame

print(f"[DONE] Imported lane curves into collection '{COLLECTION_NAME}'")
print(f"[DONE] Scene frame range set to 1..{max_frame}")