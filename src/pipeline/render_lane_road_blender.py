import bpy
import os
import json
import math
from mathutils import Vector

# =========================================================
# CONFIG
# =========================================================

LANE_JSON_PATH = "/home/alien/cv_p3/lane_out_scene1/lane_road_for_blender.json"
CAR_BLEND_PATH = "/home/alien/cv_p3/P3Data/Assets/Audi_R8_2017.blend"

OUTPUT_FRAMES_DIR = "/home/alien/cv_p3/lane_out/tesla_lane_frames"
OUTPUT_VIDEO_PATH = "/home/alien/cv_p3/lane_out/tesla_lane_dashboard.mp4"

RENDER_RES_X = 1280
RENDER_RES_Y = 720
RENDER_FPS = 20

# Dashboard look
EGO_CAR_LOCATION = (0.0, -6.5, 0.18)
EGO_CAR_ROT_Z_DEG = 0.0
EGO_CAR_SCALE = 1.10

# Camera placed behind and above the ego car, looking forward
CAMERA_LOCATION = (0.0, -14.5, 7.2)
CAMERA_ROT_X_DEG = 73.0
CAMERA_ROT_Y_DEG = 0.0
CAMERA_ROT_Z_DEG = 0.0
CAMERA_LENS = 34.0

# Mapping from lane-export vehicle frame to Blender world
# lane exporter gives:
#   X = right
#   Y = up
#   Z = forward
#
# Blender world we use:
#   X = right
#   Y = forward
#   Z = up
#
# So: (X, Y_up, Z_forward) -> (x, y, z) = (X, Z_forward + offset, lane_height)
LANE_FORWARD_OFFSET = -1.2
LANE_HEIGHT = 0.025

ROAD_LENGTH = 90.0
ROAD_WIDTH = 16.0
ROAD_LOCATION = (0.0, 20.0, 0.0)

LANE_BEVEL_DEPTH = 0.055
LANE_EMISSION_STRENGTH = 12.0

BLUE_LANE_COLOR = (0.08, 0.55, 1.0, 1.0)
YELLOW_LANE_COLOR = (1.00, 0.80, 0.10, 1.0)
ROAD_COLOR = (0.03, 0.03, 0.035, 1.0)
GROUND_COLOR = (0.015, 0.015, 0.018, 1.0)

MAX_RENDER_FRAMES = -1   # -1 means all frames from json


# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for _ in range(3):
        try:
            bpy.ops.outliner.orphans_purge(do_recursive=True)
        except Exception:
            pass


def set_if_exists(obj, attr, value):
    if hasattr(obj, attr):
        try:
            setattr(obj, attr, value)
            print(f"[INFO] Set {obj.__class__.__name__}.{attr} = {value}")
        except Exception as e:
            print(f"[WARN] Could not set {attr}: {e}")
    else:
        print(f"[WARN] {obj.__class__.__name__} has no attribute '{attr}', skipping")


def set_render_settings(fps=20):
    scene = bpy.context.scene

    # Engine
    scene.render.engine = 'BLENDER_EEVEE'

    # Eevee settings guarded for Blender build differences
    eevee = scene.eevee
    set_if_exists(eevee, "taa_render_samples", 64)
    set_if_exists(eevee, "taa_samples", 16)
    set_if_exists(eevee, "use_bloom", True)
    set_if_exists(eevee, "bloom_intensity", 0.12)
    set_if_exists(eevee, "bloom_radius", 6.5)
    set_if_exists(eevee, "bloom_threshold", 0.8)
    set_if_exists(eevee, "use_gtao", True)
    set_if_exists(eevee, "gtao_factor", 1.3)
    set_if_exists(eevee, "gtao_quality", 0.25)
    set_if_exists(eevee, "use_ssr", True)
    set_if_exists(eevee, "use_ssr_refraction", True)
    set_if_exists(eevee, "ssr_quality", 0.35)
    set_if_exists(eevee, "ssr_max_roughness", 0.7)
    set_if_exists(eevee, "use_soft_shadows", True)
    set_if_exists(eevee, "use_taa_reprojection", True)

    # Resolution / fps
    scene.render.resolution_x = RENDER_RES_X
    scene.render.resolution_y = RENDER_RES_Y
    scene.render.resolution_percentage = 100
    scene.render.fps = fps

    # IMPORTANT:
    # This Blender build does not support file_format='FFMPEG'.
    # So render PNG frames instead.
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.compression = 15

    # Use a frame sequence path
    scene.render.filepath = os.path.join(OUTPUT_FRAMES_DIR, "frame_")

    # Color management
    try:
        scene.view_settings.look = 'None'
    except Exception:
        pass

    try:
        scene.view_settings.exposure = 0.8
        scene.view_settings.gamma = 1.0
    except Exception as e:
        print(f"[WARN] Could not set color management options: {e}")


def setup_world():
    if bpy.context.scene.world is not None:
        world = bpy.context.scene.world
        world.name = "TeslaWorld"
    else:
        world = bpy.data.worlds.new("TeslaWorld")
        bpy.context.scene.world = world

    world.use_nodes = True

    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    bg = nodes.new(type='ShaderNodeBackground')
    bg.inputs["Color"].default_value = (0.008, 0.01, 0.015, 1.0)
    bg.inputs["Strength"].default_value = 1.3

    out = nodes.new(type='ShaderNodeOutputWorld')
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def look_at(obj, target: Vector):
    direction = target - obj.location
    quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = quat.to_euler()


def create_camera():
    cam_data = bpy.data.cameras.new("DashboardCamera")
    cam = bpy.data.objects.new("DashboardCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    cam.location = Vector(CAMERA_LOCATION)
    cam.rotation_euler = (
        math.radians(CAMERA_ROT_X_DEG),
        math.radians(CAMERA_ROT_Y_DEG),
        math.radians(CAMERA_ROT_Z_DEG),
    )
    cam.data.lens = CAMERA_LENS
    cam.data.clip_start = 0.01
    cam.data.clip_end = 500.0
    return cam


def add_light_sun():
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 4.0
    light = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.scene.collection.objects.link(light)
    light.rotation_euler = (
        math.radians(52.0),
        math.radians(0.0),
        math.radians(-28.0),
    )
    return light


def add_light_area(name, location, rotation_deg_xyz, energy=2500.0, size=10.0, color=(1, 1, 1)):
    light_data = bpy.data.lights.new(name=name, type='AREA')
    light_data.energy = energy
    light_data.shape = 'RECTANGLE'
    light_data.size = size
    light_data.size_y = size * 0.6
    light_data.color = color

    light = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = Vector(location)
    light.rotation_euler = tuple(math.radians(v) for v in rotation_deg_xyz)
    return light


def make_principled_material(name, base_color=(1, 1, 1, 1), roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return mat


def make_emission_material(name, color=(0.1, 0.5, 1.0, 1.0), strength=8.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength

    out = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def create_road():
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=ROAD_LOCATION)
    road = bpy.context.active_object
    road.name = "RoadPlane"
    road.scale = (ROAD_WIDTH / 2.0, ROAD_LENGTH / 2.0, 1.0)

    road_mat = make_principled_material(
        "RoadMaterial",
        base_color=ROAD_COLOR,
        roughness=0.92,
        metallic=0.0,
    )
    road.data.materials.clear()
    road.data.materials.append(road_mat)
    return road


def create_background_ground():
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 20.0, -0.03))
    bg = bpy.context.active_object
    bg.name = "BackgroundGround"
    bg.scale = (140.0, 140.0, 1.0)

    mat = make_principled_material(
        "GroundMaterial",
        base_color=GROUND_COLOR,
        roughness=1.0,
        metallic=0.0,
    )
    bg.data.materials.clear()
    bg.data.materials.append(mat)
    return bg


def get_collection_root_objects(coll):
    roots = []
    for obj in coll.objects:
        if obj.parent is None:
            roots.append(obj)
    if not roots:
        roots = list(coll.objects)
    return roots


def append_best_car_asset(blend_path):
    if not os.path.exists(blend_path):
        raise FileNotFoundError(f"Car blend file not found: {blend_path}")

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        if data_from.collections:
            data_to.collections = [data_from.collections[0]]
            data_to.objects = []
        elif data_from.objects:
            data_to.collections = []
            data_to.objects = data_from.objects
        else:
            raise RuntimeError(f"No collections or objects found in {blend_path}")

    roots = []

    if data_to.collections:
        for coll in data_to.collections:
            if coll is None:
                continue
            bpy.context.scene.collection.children.link(coll)
            roots.extend(get_collection_root_objects(coll))
    else:
        for obj in data_to.objects:
            if obj is None:
                continue
            bpy.context.scene.collection.objects.link(obj)
            if obj.parent is None:
                roots.append(obj)

    if not roots:
        raise RuntimeError("No root objects imported from the car blend file")

    ctrl = bpy.data.objects.new("EgoCar_CTRL", None)
    ctrl.empty_display_type = 'PLAIN_AXES'
    ctrl.empty_display_size = 1.0
    bpy.context.scene.collection.objects.link(ctrl)

    for obj in roots:
        if obj != ctrl:
            obj.parent = ctrl

    ctrl.location = Vector(EGO_CAR_LOCATION)
    ctrl.rotation_euler = (0.0, 0.0, math.radians(EGO_CAR_ROT_Z_DEG))
    ctrl.scale = (EGO_CAR_SCALE, EGO_CAR_SCALE, EGO_CAR_SCALE)

    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    return ctrl


def create_lane_materials():
    mats = {
        "solid": make_emission_material(
            "LaneBlue",
            color=BLUE_LANE_COLOR,
            strength=LANE_EMISSION_STRENGTH
        ),
        "dashed": make_emission_material(
            "LaneYellow",
            color=YELLOW_LANE_COLOR,
            strength=LANE_EMISSION_STRENGTH * 0.95
        )
    }
    return mats


def label_to_lane_style(label_name: str):
    n = (label_name or "").lower()
    if "dash" in n:
        return "dashed"
    return "solid"


def vehicle_point_to_blender(pt):
    if pt is None or len(pt) < 3:
        return None

    x = float(pt[0])
    z_forward = float(pt[2])

    return Vector((x, z_forward + LANE_FORWARD_OFFSET, LANE_HEIGHT))


def create_lane_curve_object(name, material):
    curve_data = bpy.data.curves.new(name=name + "_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 24
    curve_data.bevel_depth = LANE_BEVEL_DEPTH
    curve_data.bevel_resolution = 8
    curve_data.fill_mode = 'FULL'

    spline = curve_data.splines.new('POLY')
    spline.points.add(1)

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.scene.collection.objects.link(obj)

    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def set_curve_points(curve_obj, points):
    curve = curve_obj.data

    while len(curve.splines) > 0:
        curve.splines.remove(curve.splines[0])

    if len(points) < 2:
        spline = curve.splines.new('POLY')
        spline.points.add(1)
        spline.points[0].co = (0.0, 0.0, -1000.0, 1.0)
        spline.points[1].co = (0.0, 0.0, -1000.0, 1.0)
        return

    spline = curve.splines.new('POLY')
    spline.points.add(len(points) - 1)

    for i, p in enumerate(points):
        spline.points[i].co = (p.x, p.y, p.z, 1.0)


def add_dashboard_backdrop():
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, -18.0, 9.0))
    back = bpy.context.active_object
    back.name = "Backdrop"
    back.scale = (60.0, 1.0, 20.0)
    back.rotation_euler = (math.radians(90.0), 0.0, 0.0)

    mat = make_principled_material(
        "BackdropMat",
        base_color=(0.01, 0.012, 0.018, 1.0),
        roughness=1.0,
        metallic=0.0,
    )
    back.data.materials.clear()
    back.data.materials.append(mat)
    return back


def add_fake_dashboard_panel():
    bpy.ops.mesh.primitive_cube_add(location=(0.0, -10.2, 0.55))
    panel = bpy.context.active_object
    panel.name = "DashboardPanel"
    panel.scale = (5.6, 0.45, 0.65)

    mat = make_principled_material(
        "DashboardPanelMat",
        base_color=(0.018, 0.02, 0.025, 1.0),
        roughness=0.9,
        metallic=0.0,
    )
    panel.data.materials.clear()
    panel.data.materials.append(mat)
    return panel


def load_lane_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lane JSON not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        frames = data.get("frames", [])
        meta = data.get("meta", {})
    elif isinstance(data, list):
        frames = data
        meta = {}
    else:
        raise RuntimeError("Unsupported lane JSON structure")

    return meta, frames


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_dir(OUTPUT_FRAMES_DIR)

    clear_scene()
    set_render_settings()
    setup_world()

    create_background_ground()
    create_road()
    add_dashboard_backdrop()
    add_fake_dashboard_panel()

    cam = create_camera()

    add_light_sun()
    add_light_area(
        "FrontFill",
        location=(0.0, -2.0, 11.5),
        rotation_deg_xyz=(68.0, 0.0, 0.0),
        energy=4200.0,
        size=16.0,
        color=(1.0, 0.98, 0.95),
    )
    add_light_area(
        "LeftRim",
        location=(-8.0, -9.0, 4.5),
        rotation_deg_xyz=(78.0, 0.0, 58.0),
        energy=1600.0,
        size=7.0,
        color=(0.75, 0.85, 1.0),
    )
    add_light_area(
        "RightRim",
        location=(8.0, -9.0, 4.5),
        rotation_deg_xyz=(78.0, 0.0, -58.0),
        energy=1600.0,
        size=7.0,
        color=(0.75, 0.85, 1.0),
    )

    append_best_car_asset(CAR_BLEND_PATH)

    look_at(cam, Vector((0.0, 15.0, 0.7)))

    meta, frames = load_lane_json(LANE_JSON_PATH)

    fps = int(round(meta.get("fps", RENDER_FPS))) if isinstance(meta, dict) else RENDER_FPS
    if fps <= 0:
        fps = RENDER_FPS
    bpy.context.scene.render.fps = fps

    if MAX_RENDER_FRAMES > 0:
        frames = frames[:MAX_RENDER_FRAMES]

    if len(frames) == 0:
        raise RuntimeError("No frames found in lane JSON")

    max_lanes = 0
    for fr in frames:
        lanes = fr.get("lanes", []) if isinstance(fr, dict) else []
        max_lanes = max(max_lanes, len(lanes))

    if max_lanes == 0:
        raise RuntimeError("No lane entries found in the JSON")

    lane_mats = create_lane_materials()

    lane_objs = []
    for i in range(max_lanes):
        lane_obj = create_lane_curve_object(f"Lane_{i:02d}", lane_mats["solid"])
        lane_objs.append(lane_obj)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(frames)

    for frame_number, fr in enumerate(frames, start=1):
        scene.frame_set(frame_number)
        lanes = fr.get("lanes", []) if isinstance(fr, dict) else []

        for i, lane in enumerate(lanes):
            lane_obj = lane_objs[i]
            label_name = lane.get("label_name", "lane")
            style = label_to_lane_style(label_name)

            pts_ground = lane.get("points_ground_vehicle", [])
            blender_pts = []

            for p in pts_ground:
                bp = vehicle_point_to_blender(p)
                if bp is not None:
                    blender_pts.append(bp)

            set_curve_points(lane_obj, blender_pts)

            lane_obj.hide_viewport = False
            lane_obj.hide_render = False
            lane_obj.data.materials.clear()
            lane_obj.data.materials.append(lane_mats[style])

            lane_obj.keyframe_insert(data_path="hide_viewport", frame=frame_number)
            lane_obj.keyframe_insert(data_path="hide_render", frame=frame_number)

        for i in range(len(lanes), len(lane_objs)):
            lane_obj = lane_objs[i]
            lane_obj.hide_viewport = True
            lane_obj.hide_render = True
            lane_obj.keyframe_insert(data_path="hide_viewport", frame=frame_number)
            lane_obj.keyframe_insert(data_path="hide_render", frame=frame_number)

    print(f"[INFO] Frames in JSON: {len(frames)}")
    print(f"[INFO] Max lanes per frame: {max_lanes}")
    print(f"[INFO] Output frames dir: {OUTPUT_FRAMES_DIR}")
    print(f"[INFO] Planned video path: {OUTPUT_VIDEO_PATH}")

    bpy.ops.render.render(animation=True)

    print("[DONE] PNG frame render complete.")
    print("[DONE] Convert to MP4 with:")
    print(f'ffmpeg -framerate {scene.render.fps} -i "{OUTPUT_FRAMES_DIR}/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p "{OUTPUT_VIDEO_PATH}"')


if __name__ == "__main__":
    main()