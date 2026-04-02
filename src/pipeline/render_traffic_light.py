import bpy
import os
import json
import math
import gc
from mathutils import Vector

# =========================================================
# CONFIG
# =========================================================
TRACKS_JSON = "/home/alien/cv_p3/pipeline_out/scene_tracks_with_lights.json"

CAR_BLEND_PATH = "/home/alien/cv_p3/P3Data/Assets/Audi_R8_2017.blend"
CAR_OBJECT_NAME = ""
CAR_COLLECTION_NAME = ""

OUTPUT_FRAMES_DIR = "/home/alien/cv_p3/pipeline_out/blender_frames_traffic"
OUTPUT_VIDEO_PATH = "/home/alien/cv_p3/pipeline_out/blender_render_traffic.mp4"

GROUND_Z = 0.0
WORLD_SCALE = 1.0

EGO_CAR = True
EGO_CAR_LOCATION = (0.0, 0.0, 0.55)
EGO_CAR_SCALE = (1.15, 1.15, 1.15)

EGO_SPEED_MPS = 8.0

CAM_BACK = 10.0
CAM_HEIGHT = 5.5
CAM_FORWARD_LOOK = 18.0

RENDER_RES_X = 1280
RENDER_RES_Y = 720
RENDER_ENGINE = "BLENDER_EEVEE"
RENDER_SAMPLES = 32

OUTPUT_MODE = "PNG_SEQUENCE"

MAX_FRAMES_TO_RENDER = None
PROGRESS_EVERY_N_FRAMES = 10

# Asset normalization
TARGET_CAR_LENGTH = 4.6
ASSET_Z_OFFSET = 0.0
ASSET_YAW_OFFSET_DEG = 180.0

FORCE_SIMPLE_MATERIALS = True
ASSET_BASE_COLOR = (0.08, 0.08, 0.08, 1.0)
ASSET_GLASS_COLOR = (0.06, 0.06, 0.08, 1.0)
ASSET_TYRE_COLOR = (0.03, 0.03, 0.03, 1.0)
ASSET_METAL_COLOR = (0.45, 0.45, 0.45, 1.0)

# -------------------------
# Vehicle de-overlap tuning
# -------------------------
MAX_VISIBLE_VEHICLES = 12
MERGE_DIST_X = 1.8
MERGE_DIST_Y = 4.5
MIN_FORWARD_GAP = 6.0
LANE_CENTERS = [-3.5, 0.0, 3.5]
MAX_SIDE_OFFSET = 5.5
MIN_FORWARD_DIST = 4.0
MAX_FORWARD_DIST = 80.0
FORCE_LANE_ALIGNMENT = True

# -------------------------
# Traffic light rendering
# -------------------------
TRAFFIC_LIGHT_MIN_Z = 8.0
TRAFFIC_LIGHT_MAX_Z = 120.0
TRAFFIC_LIGHT_MAX_SIDE_OFFSET = 30.0

TRAFFIC_LIGHT_RIGHT_X = 8.5
TRAFFIC_LIGHT_LEFT_X = -8.5

TRAFFIC_LIGHT_MERGE_DIST_X = 3.0
TRAFFIC_LIGHT_MERGE_DIST_Y = 12.0
TRAFFIC_LIGHT_MIN_GAP_Y = 20.0

TRAFFIC_LIGHT_POLE_HEIGHT = 6.5
TRAFFIC_LIGHT_HEAD_Z = 5.5
TRAFFIC_LIGHT_FORWARD_OFFSET = 0.0
TRAFFIC_LIGHT_FACING_YAW_DEG = -90.0

# =========================================================
# camera: X right, Y down, Z forward
# blender: X right, Y forward, Z up
# =========================================================
def cam_to_blender_relative(p):
    X, Y, Z = p
    bx = X * WORLD_SCALE
    by = Z * WORLD_SCALE
    bz = 0.0
    return (bx, by, bz)

# =========================================================
# Utilities
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clear_scene():
    print("[INFO] Clearing existing Blender scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)
    for block in list(bpy.data.fonts):
        if block.users == 0:
            bpy.data.fonts.remove(block)
    for block in list(bpy.data.collections):
        if block.users == 0 and block.name != "Scene Collection":
            bpy.data.collections.remove(block)
    print("[INFO] Scene cleared.")

def make_material(name, rgba, metallic=0.0, roughness=0.5, emission_strength=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        if "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = rgba
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat

def setup_world():
    print("[INFO] Setting up world...")
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.03, 0.03, 0.03, 1.0)
        bg.inputs[1].default_value = 1.2
    print("[INFO] World ready.")

def create_ground():
    print("[INFO] Creating ground plane...")
    bpy.ops.mesh.primitive_plane_add(size=2000, location=(0, 500, GROUND_Z))
    plane = bpy.context.active_object
    plane.name = "Ground"
    mat = make_material("GroundMat", (0.12, 0.12, 0.12, 1.0))
    plane.data.materials.append(mat)
    print("[INFO] Ground plane created.")
    return plane

def create_lane_markings():
    print("[INFO] Creating long lane markings...")
    lane_mat = make_material("LaneMat", (0.95, 0.95, 0.85, 1.0), roughness=0.4)
    count = 0
    for lane_x in [-3.5, 0.0, 3.5]:
        for y in range(-100, 2000, 8):
            bpy.ops.mesh.primitive_cube_add(location=(lane_x, y, GROUND_Z + 0.01))
            dash = bpy.context.active_object
            dash.scale = (0.08, 1.4, 0.01)
            dash.name = f"Lane_{lane_x}_{y}"
            dash.data.materials.append(lane_mat)
            if hasattr(dash, "visible_shadow"):
                dash.visible_shadow = False
            count += 1
    print(f"[INFO] Lane markings created: {count}")

def create_side_environment():
    print("[INFO] Creating long side environment...")
    pole_mat = make_material("PoleMat", (0.25, 0.25, 0.25, 1.0))
    tree_mat = make_material("TreeMat", (0.12, 0.35, 0.12, 1.0))

    count_poles = 0
    count_trees = 0
    for y in range(-50, 2000, 18):
        for x in (-10, 10):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=4.0, location=(x, y, 2.0))
            pole = bpy.context.active_object
            pole.data.materials.append(pole_mat)
            if hasattr(pole, "visible_shadow"):
                pole.visible_shadow = False
            count_poles += 1

            bpy.ops.mesh.primitive_uv_sphere_add(radius=1.2, location=(x, y, 4.8))
            tree = bpy.context.active_object
            tree.data.materials.append(tree_mat)
            if hasattr(tree, "visible_shadow"):
                tree.visible_shadow = False
            count_trees += 1

    print(f"[INFO] Side environment created: poles={count_poles}, trees={count_trees}")

def setup_lights():
    print("[INFO] Setting up lights...")
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 20))
    sun = bpy.context.active_object
    sun.data.energy = 4.5
    sun.rotation_euler = (math.radians(35), 0.0, math.radians(20))

    bpy.ops.object.light_add(type='AREA', location=(0, -8, 8))
    area = bpy.context.active_object
    area.data.energy = 1500
    area.data.shape = 'RECTANGLE'
    area.data.size = 12
    area.data.size_y = 8
    print("[INFO] Lights ready.")

def create_camera_rig():
    print("[INFO] Creating camera rig...")
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.2))
    cam_target = bpy.context.active_object
    cam_target.name = "CameraTarget"

    bpy.ops.object.camera_add(location=(0, -CAM_BACK, CAM_HEIGHT))
    cam = bpy.context.active_object
    cam.name = "TPP_Camera"
    cam.data.lens = 35
    cam.data.clip_end = 5000

    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = cam_target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    bpy.context.scene.camera = cam
    print("[INFO] Camera rig created.")
    return cam, cam_target

def make_placeholder_vehicle(name="vehicle_proto", color=(0.7, 0.7, 0.75, 1.0)):
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.5))
    body = bpy.context.active_object
    body.name = name
    body.scale = (0.95, 2.1, 0.45)

    mat = make_material(f"{name}_mat", color)
    body.data.materials.append(mat)
    if hasattr(body, "visible_shadow"):
        body.visible_shadow = False

    bpy.ops.mesh.primitive_cube_add(location=(0, 0.1, 1.0))
    roof = bpy.context.active_object
    roof.scale = (0.65, 0.95, 0.32)
    roof.parent = body
    roof.data.materials.append(mat)
    if hasattr(roof, "visible_shadow"):
        roof.visible_shadow = False
    return body

# =========================================================
# Traffic light creation
# =========================================================
def set_hide_recursive(obj, hidden):
    obj.hide_viewport = hidden
    obj.hide_render = hidden
    for ch in obj.children_recursive:
        ch.hide_viewport = hidden
        ch.hide_render = hidden

def normalize_signal_state(state):
    s = str(state).strip().lower()
    if s == "orange":
        return "yellow"
    if s in {"red", "yellow", "green"}:
        return s
    return "unknown"

def make_traffic_light_materials(prefix):
    mats = {
        "pole": make_material(f"{prefix}_pole", (0.22, 0.22, 0.22, 1.0), metallic=0.4, roughness=0.5),
        "head": make_material(f"{prefix}_head", (0.08, 0.08, 0.08, 1.0), metallic=0.1, roughness=0.65),
        "red_on": make_material(f"{prefix}_red_on", (1.0, 0.12, 0.12, 1.0), roughness=0.2, emission_strength=22.0),
        "red_off": make_material(f"{prefix}_red_off", (0.18, 0.03, 0.03, 1.0), roughness=0.55, emission_strength=0.0),
        "yellow_on": make_material(f"{prefix}_yellow_on", (1.0, 0.55, 0.05, 1.0), roughness=0.2, emission_strength=22.0),
        "yellow_off": make_material(f"{prefix}_yellow_off", (0.18, 0.10, 0.02, 1.0), roughness=0.55, emission_strength=0.0),
        "green_on": make_material(f"{prefix}_green_on", (0.05, 1.0, 0.12, 1.0), roughness=0.2, emission_strength=22.0),
        "green_off": make_material(f"{prefix}_green_off", (0.02, 0.18, 0.03, 1.0), roughness=0.55, emission_strength=0.0),
    }
    return mats

def create_traffic_light_rig(name):
    mats = make_traffic_light_materials(name)

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    root = bpy.context.active_object
    root.name = name

    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.08,
        depth=TRAFFIC_LIGHT_POLE_HEIGHT,
        location=(0, 0, TRAFFIC_LIGHT_POLE_HEIGHT * 0.5)
    )
    pole = bpy.context.active_object
    pole.name = f"{name}_pole"
    pole.parent = root
    pole.data.materials.append(mats["pole"])

    bpy.ops.mesh.primitive_cube_add(location=(0, 0, TRAFFIC_LIGHT_HEAD_Z))
    head = bpy.context.active_object
    head.name = f"{name}_head"
    head.scale = (0.22, 0.18, 0.6)
    head.parent = root
    head.data.materials.append(mats["head"])

    lamp_offsets = {
        "red": 0.35,
        "yellow": 0.0,
        "green": -0.35,
    }

    lamps = {}
    for state, zoff in lamp_offsets.items():
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.10,
            location=(0.23, 0, TRAFFIC_LIGHT_HEAD_Z + zoff)
        )
        lamp = bpy.context.active_object
        lamp.name = f"{name}_{state}"
        lamp.scale = (1.0, 0.55, 1.0)
        lamp.parent = root
        lamp.data.materials.append(mats[f"{state}_off"])
        lamps[state] = lamp

    root["lamp_red"] = lamps["red"].name
    root["lamp_yellow"] = lamps["yellow"].name
    root["lamp_green"] = lamps["green"].name
    return root

def set_traffic_light_state(root, state):
    state = normalize_signal_state(state)
    name = root.name
    mats = make_traffic_light_materials(name)

    red_obj = bpy.data.objects.get(root.get("lamp_red", ""))
    yellow_obj = bpy.data.objects.get(root.get("lamp_yellow", ""))
    green_obj = bpy.data.objects.get(root.get("lamp_green", ""))

    if red_obj is not None:
        red_obj.data.materials.clear()
        red_obj.data.materials.append(mats["red_on"] if state == "red" else mats["red_off"])
    if yellow_obj is not None:
        yellow_obj.data.materials.clear()
        yellow_obj.data.materials.append(mats["yellow_on"] if state == "yellow" else mats["yellow_off"])
    if green_obj is not None:
        green_obj.data.materials.clear()
        green_obj.data.materials.append(mats["green_on"] if state == "green" else mats["green_off"])

# =========================================================
# Asset loading helpers
# =========================================================
def resolve_blend_path(path):
    if os.path.isfile(path):
        return path
    if path.endswith(".blend1"):
        alt = path[:-1]
        if os.path.isfile(alt):
            return alt
    raise FileNotFoundError(f"Blend asset not found: {path}")

def get_object_bbox_world(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return min_v, max_v

def all_descendants(root):
    objs = [root]
    for c in root.children_recursive:
        objs.append(c)
    return objs

def hierarchy_bbox(root):
    objs = all_descendants(root)
    mins = []
    maxs = []
    for obj in objs:
        if obj.type == "MESH":
            mn, mx = get_object_bbox_world(obj)
            mins.append(mn)
            maxs.append(mx)

    if not mins:
        return None, None

    min_v = Vector((
        min(v.x for v in mins),
        min(v.y for v in mins),
        min(v.z for v in mins)
    ))
    max_v = Vector((
        max(v.x for v in maxs),
        max(v.y for v in maxs),
        max(v.z for v in maxs)
    ))
    return min_v, max_v

def choose_material_style(obj_name):
    low = obj_name.lower()
    if any(k in low for k in ["glass", "windscreen", "window"]):
        return ASSET_GLASS_COLOR, 0.0, 0.08
    if any(k in low for k in ["tyre", "tire", "wheel", "rubber"]):
        return ASSET_TYRE_COLOR, 0.0, 0.9
    if any(k in low for k in ["mirror", "chrome", "metal", "rim", "centre"]):
        return ASSET_METAL_COLOR, 0.6, 0.25
    return ASSET_BASE_COLOR, 0.2, 0.45

def apply_simple_materials(root):
    for obj in all_descendants(root):
        if obj.type != "MESH":
            continue
        rgba, metallic, roughness = choose_material_style(obj.name)
        mat = make_material(f"{obj.name}_simple", rgba, metallic=metallic, roughness=roughness)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if hasattr(obj, "visible_shadow"):
            obj.visible_shadow = False

def normalize_asset_root(root):
    bpy.context.view_layer.update()
    mn, mx = hierarchy_bbox(root)
    if mn is None:
        return

    size = mx - mn
    car_len = max(size.x, size.y, 1e-6)
    s = TARGET_CAR_LENGTH / car_len
    root.scale = root.scale * s
    bpy.context.view_layer.update()

    mn, mx = hierarchy_bbox(root)
    if mn is None:
        return

    center = (mn + mx) * 0.5
    root.location = root.location - Vector((center.x, center.y, mn.z))
    bpy.context.view_layer.update()

def is_skippable_asset_name(name):
    low = name.lower()
    skip_tokens = [
        "sun", "camera", "light", "empty",
        "numberplate", "numbreplate", "font",
        "ground", "text"
    ]
    return any(tok in low for tok in skip_tokens)

def append_full_car_asset(name):
    blend_path = resolve_blend_path(CAR_BLEND_PATH)

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        available_objects = list(data_from.objects)
        available_collections = list(data_from.collections)

    print(f"[INFO] Asset file: {blend_path}")
    print(f"[INFO] Requested object: {CAR_OBJECT_NAME if CAR_OBJECT_NAME else '(auto full asset)'}")
    print(f"[INFO] Available objects: {available_objects[:40]}")
    print(f"[INFO] Available collections: {available_collections[:20]}")

    imported = []

    if CAR_COLLECTION_NAME and CAR_COLLECTION_NAME in available_collections:
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
            data_to.collections = [CAR_COLLECTION_NAME]

        for coll in data_to.collections:
            if coll is not None:
                bpy.context.scene.collection.children.link(coll)
                for obj in coll.objects:
                    if obj.type == "MESH" and not is_skippable_asset_name(obj.name):
                        imported.append(obj)

    elif CAR_OBJECT_NAME and CAR_OBJECT_NAME in available_objects:
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
            data_to.objects = [CAR_OBJECT_NAME]

        for obj in data_to.objects:
            if obj is not None and obj.type == "MESH":
                bpy.context.collection.objects.link(obj)
                imported.append(obj)

    else:
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)

        for obj in data_to.objects:
            if obj is None:
                continue
            if obj.type != "MESH":
                continue
            if is_skippable_asset_name(obj.name):
                continue
            bpy.context.collection.objects.link(obj)
            imported.append(obj)

    if not imported:
        print("[WARN] No valid mesh asset objects imported. Using placeholder.")
        return make_placeholder_vehicle(name)

    print(f"[INFO] Imported mesh parts for '{name}': {len(imported)}")

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    root = bpy.context.active_object
    root.name = name

    for obj in imported:
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()

    normalize_asset_root(root)

    if FORCE_SIMPLE_MATERIALS:
        apply_simple_materials(root)

    print(f"[INFO] Asset root '{name}' normalized and ready.")
    return root

def duplicate_hierarchy(root, new_name):
    obj_map = {}

    root_copy = root.copy()
    root_copy.name = new_name
    bpy.context.collection.objects.link(root_copy)
    obj_map[root] = root_copy

    for child in root.children_recursive:
        child_copy = child.copy()
        if child.type == "MESH" and child.data is not None:
            child_copy.data = child.data.copy()
        child_copy.name = f"{new_name}_{child.name}"
        bpy.context.collection.objects.link(child_copy)
        obj_map[child] = child_copy

    for old_obj, new_obj in obj_map.items():
        if old_obj.parent is not None and old_obj.parent in obj_map:
            new_obj.parent = obj_map[old_obj.parent]
            new_obj.matrix_parent_inverse = old_obj.matrix_parent_inverse.copy()

    bpy.context.view_layer.update()
    return root_copy

CAR_PROTOTYPE = None

def get_vehicle_prototype(color=(0.7, 0.7, 0.75, 1.0)):
    global CAR_PROTOTYPE
    if CAR_PROTOTYPE is not None:
        return CAR_PROTOTYPE

    print("[INFO] Creating car prototype...")
    if CAR_BLEND_PATH:
        try:
            proto = append_full_car_asset("__car_prototype__")
            proto.hide_viewport = True
            proto.hide_render = True
            for ch in proto.children_recursive:
                ch.hide_viewport = True
                ch.hide_render = True
            CAR_PROTOTYPE = proto
            print("[INFO] Created car prototype successfully.")
            return CAR_PROTOTYPE
        except Exception as e:
            print(f"[WARN] Asset load failed for prototype: {e}")
            print("[WARN] Falling back to placeholder prototype.")

    proto = make_placeholder_vehicle("__car_prototype__", color)
    proto.hide_viewport = True
    proto.hide_render = True
    CAR_PROTOTYPE = proto
    print("[INFO] Placeholder prototype created.")
    return CAR_PROTOTYPE

def load_or_create_vehicle(name="vehicle_proto", color=(0.7, 0.7, 0.75, 1.0)):
    proto = get_vehicle_prototype(color=color)
    obj = duplicate_hierarchy(proto, name)
    obj.hide_viewport = False
    obj.hide_render = False
    for ch in obj.children_recursive:
        ch.hide_viewport = False
        ch.hide_render = False
    print(f"[INFO] Vehicle instance created: {name}")
    return obj

# =========================================================
# World placement helpers
# =========================================================
def set_ego_transform(obj, ego_world_y):
    obj.location = (
        EGO_CAR_LOCATION[0],
        EGO_CAR_LOCATION[1] + ego_world_y,
        EGO_CAR_LOCATION[2]
    )
    obj.rotation_euler = (0.0, 0.0, math.radians(ASSET_YAW_OFFSET_DEG))
    obj.scale = EGO_CAR_SCALE

def set_vehicle_transform_world(obj, ego_world_y, xyz_cam, yaw_rad, scale):
    rel_x, rel_y, _ = cam_to_blender_relative(xyz_cam)

    world_x = EGO_CAR_LOCATION[0] + rel_x
    world_y = EGO_CAR_LOCATION[1] + ego_world_y + rel_y
    world_z = GROUND_Z + ASSET_Z_OFFSET

    obj.location = (world_x, world_y, world_z)
    obj.rotation_euler = (0.0, 0.0, -yaw_rad + math.radians(ASSET_YAW_OFFSET_DEG))
    obj.scale = (scale, scale, scale)

def set_traffic_light_transform_world(obj, ego_world_y, xyz_cam, state):
    rel_x, rel_y, _ = cam_to_blender_relative(xyz_cam)

    side_right = rel_x >= 0.0
    world_x = TRAFFIC_LIGHT_RIGHT_X if side_right else TRAFFIC_LIGHT_LEFT_X
    world_y = EGO_CAR_LOCATION[1] + ego_world_y + rel_y + TRAFFIC_LIGHT_FORWARD_OFFSET
    world_z = GROUND_Z

    obj.location = (world_x, world_y, world_z)
    obj.rotation_euler = (0.0, 0.0, math.radians(TRAFFIC_LIGHT_FACING_YAW_DEG))
    set_traffic_light_state(obj, state)

def set_camera_target_from_ego(cam_target, ego_obj):
    ego_loc = ego_obj.location.copy()
    cam_target.location = Vector((
        ego_loc.x,
        ego_loc.y + CAM_FORWARD_LOOK,
        ego_loc.z + 1.5
    ))

def set_camera_position_relative_to_ego(cam_obj, ego_obj):
    ego_loc = ego_obj.location.copy()
    cam_obj.location = Vector((
        ego_loc.x,
        ego_loc.y - CAM_BACK,
        ego_loc.z + CAM_HEIGHT
    ))

def render_frame_still(scene, frame_number, filepath):
    scene.frame_set(frame_number)
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

# =========================================================
# Vehicle helpers
# =========================================================
def nearest_lane_center(x):
    return min(LANE_CENTERS, key=lambda c: abs(x - c))

def sanitize_track(tr):
    pos = tr.get("position_cam_xyz", [0.0, 0.0, 0.0])
    if len(pos) < 3:
        return None

    x = float(pos[0])
    z = float(pos[2])
    score = float(tr.get("score", 0.0))
    yaw = float(tr.get("yaw_rad", 0.0))
    scale = float(tr.get("scale", 1.0))
    tid = int(tr.get("track_id", -1))
    cls_name = tr.get("class", "car")

    if z < MIN_FORWARD_DIST or z > MAX_FORWARD_DIST:
        return None
    if abs(x) > MAX_SIDE_OFFSET:
        return None

    if FORCE_LANE_ALIGNMENT:
        x = nearest_lane_center(x)

    return {
        "track_id": tid,
        "class": cls_name,
        "score": score,
        "yaw_rad": yaw,
        "scale": max(0.9, min(scale, 1.25)),
        "position_cam_xyz": [x, 0.0, z],
        "lane_x": x,
    }

def merge_close_tracks(tracks):
    cleaned = []
    used = [False] * len(tracks)

    for i in range(len(tracks)):
        if used[i]:
            continue
        a = tracks[i]
        best = a
        used[i] = True

        ax = a["position_cam_xyz"][0]
        az = a["position_cam_xyz"][2]

        for j in range(i + 1, len(tracks)):
            if used[j]:
                continue
            b = tracks[j]
            bx = b["position_cam_xyz"][0]
            bz = b["position_cam_xyz"][2]

            if abs(ax - bx) < MERGE_DIST_X and abs(az - bz) < MERGE_DIST_Y:
                used[j] = True
                if b["score"] > best["score"]:
                    best = b

        cleaned.append(best)

    return cleaned

def enforce_lane_spacing(tracks):
    by_lane = {}
    for tr in tracks:
        lane = tr["lane_x"]
        by_lane.setdefault(lane, []).append(tr)

    final_tracks = []
    for lane, lane_tracks in by_lane.items():
        lane_tracks.sort(key=lambda t: t["position_cam_xyz"][2])
        adjusted = []
        last_z = None

        for tr in lane_tracks:
            z = tr["position_cam_xyz"][2]
            if last_z is not None and (z - last_z) < MIN_FORWARD_GAP:
                z = last_z + MIN_FORWARD_GAP
            tr["position_cam_xyz"][2] = z
            last_z = z
            adjusted.append(tr)

        final_tracks.extend(adjusted)

    return final_tracks

def prepare_frame_vehicle_tracks(frame_tracks):
    valid = []
    for tr in frame_tracks:
        cls_name = str(tr.get("class", "")).strip().lower()
        if cls_name == "traffic light":
            continue
        s = sanitize_track(tr)
        if s is not None:
            valid.append(s)

    valid.sort(key=lambda t: t["position_cam_xyz"][2])
    valid = valid[:MAX_VISIBLE_VEHICLES]
    valid = merge_close_tracks(valid)
    valid = enforce_lane_spacing(valid)
    valid.sort(key=lambda t: (t["position_cam_xyz"][2], t["position_cam_xyz"][0]))
    return valid

# =========================================================
# Single traffic light selection
# =========================================================
def sanitize_traffic_light_track(tr):
    pos = tr.get("position_cam_xyz", [0.0, 0.0, 0.0])
    if len(pos) < 3:
        return None

    x = float(pos[0])
    z = float(pos[2])
    score = float(tr.get("score", 0.0))
    state = normalize_signal_state(tr.get("traffic_light_state", "unknown"))

    if z < TRAFFIC_LIGHT_MIN_Z or z > TRAFFIC_LIGHT_MAX_Z:
        return None
    if abs(x) > TRAFFIC_LIGHT_MAX_SIDE_OFFSET:
        return None

    return {
        "track_id": int(tr.get("track_id", -1)),
        "class": "traffic light",
        "score": score,
        "position_cam_xyz": [x, 0.0, z],
        "traffic_light_state": state,
    }

def merge_close_traffic_lights(tracks):
    if not tracks:
        return []

    used = [False] * len(tracks)
    merged = []

    for i in range(len(tracks)):
        if used[i]:
            continue

        a = tracks[i]
        ax = a["position_cam_xyz"][0]
        az = a["position_cam_xyz"][2]

        cluster = [a]
        used[i] = True

        for j in range(i + 1, len(tracks)):
            if used[j]:
                continue
            b = tracks[j]
            bx = b["position_cam_xyz"][0]
            bz = b["position_cam_xyz"][2]

            if abs(ax - bx) <= TRAFFIC_LIGHT_MERGE_DIST_X and abs(az - bz) <= TRAFFIC_LIGHT_MERGE_DIST_Y:
                used[j] = True
                cluster.append(b)

        best = max(cluster, key=lambda t: t["score"])
        merged.append(best)

    return merged

def choose_single_traffic_light(frame_tracks):
    candidates = []
    for tr in frame_tracks:
        cls_name = str(tr.get("class", "")).strip().lower()
        if cls_name != "traffic light":
            continue
        s = sanitize_traffic_light_track(tr)
        if s is not None:
            candidates.append(s)

    if not candidates:
        return None

    candidates.sort(key=lambda t: t["position_cam_xyz"][2])
    candidates = merge_close_traffic_lights(candidates)

    # Prefer known state, then nearer light, then better score
    candidates.sort(
        key=lambda t: (
            0 if t["traffic_light_state"] != "unknown" else 1,
            t["position_cam_xyz"][2],
            -t["score"],
        )
    )
    return candidates[0]

# =========================================================
# Main
# =========================================================
if not os.path.isfile(TRACKS_JSON):
    raise FileNotFoundError(
        f"TRACKS_JSON not found: {TRACKS_JSON}\n"
        "Run extract_vehicle_pose_video.py first so scene_tracks_with_lights.json exists."
    )

ensure_dir(OUTPUT_FRAMES_DIR)
ensure_dir(os.path.dirname(OUTPUT_VIDEO_PATH))

with open(TRACKS_JSON, "r") as f:
    data = json.load(f)

fps = data.get("fps", 20.0)
frames = data["frames"]

if MAX_FRAMES_TO_RENDER is not None:
    frames = frames[:MAX_FRAMES_TO_RENDER]

print(f"[INFO] TRACKS_JSON: {TRACKS_JSON}")
print(f"[INFO] OUTPUT_FRAMES_DIR: {OUTPUT_FRAMES_DIR}")
print(f"[INFO] OUTPUT_VIDEO_PATH: {OUTPUT_VIDEO_PATH}")
print(f"[INFO] Number of frames to process: {len(frames)}")
print(f"[INFO] EGO_SPEED_MPS: {EGO_SPEED_MPS}")

clear_scene()
setup_world()
create_ground()
create_lane_markings()
create_side_environment()
setup_lights()
cam, cam_target = create_camera_rig()
print("[INFO] Static scene setup completed.")

scene = bpy.context.scene
scene.render.engine = RENDER_ENGINE
if scene.render.engine == "BLENDER_EEVEE" and hasattr(scene, "eevee"):
    if hasattr(scene.eevee, "use_shadows"):
        scene.eevee.use_shadows = False
    if hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = True

scene.render.resolution_x = RENDER_RES_X
scene.render.resolution_y = RENDER_RES_Y
scene.render.resolution_percentage = 100
scene.render.fps = int(round(fps)) if fps > 0 else 20
scene.render.use_file_extension = True
scene.render.use_overwrite = True

if scene.render.engine == "CYCLES":
    scene.cycles.samples = RENDER_SAMPLES
else:
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = RENDER_SAMPLES

scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

palette = [
    (0.78, 0.20, 0.20, 1.0),
    (0.18, 0.42, 0.82, 1.0),
    (0.20, 0.65, 0.32, 1.0),
    (0.85, 0.65, 0.18, 1.0),
    (0.62, 0.30, 0.82, 1.0),
    (0.25, 0.70, 0.72, 1.0),
    (0.82, 0.42, 0.20, 1.0),
    (0.70, 0.70, 0.70, 1.0),
]

ego = None
if EGO_CAR:
    print("[INFO] Creating ego vehicle...")
    ego = load_or_create_vehicle("ego_car", color=(0.15, 0.15, 0.15, 1.0))
    set_hide_recursive(ego, False)
    print("[INFO] Ego vehicle ready.")

vehicle_objects = {}
traffic_light_object = None
total_vehicle_tracks_created = 0

print("[INFO] Starting per-frame rendering with ego forward motion...")

for f_idx, f in enumerate(frames):
    frame_idx = f["frame_idx"]
    scene_frame = f_idx + 1

    t_sec = f_idx / max(fps, 1e-6)
    ego_world_y = EGO_SPEED_MPS * t_sec

    if (f_idx % PROGRESS_EVERY_N_FRAMES) == 0:
        print(
            f"[INFO] Processing frame {f_idx + 1}/{len(frames)} "
            f"(source frame_idx={frame_idx}) | ego_world_y={ego_world_y:.2f} "
            f"| vehicles={len(vehicle_objects)} | single_light={'yes' if traffic_light_object else 'no'}"
        )

    if ego is not None:
        set_ego_transform(ego, ego_world_y)
        set_hide_recursive(ego, False)
        set_camera_position_relative_to_ego(cam, ego)
        set_camera_target_from_ego(cam_target, ego)

    frame_vehicle_tracks = prepare_frame_vehicle_tracks(f.get("tracks", []))
    single_light = choose_single_traffic_light(f.get("tracks", []))

    active_vehicle_ids = set()

    for tr in frame_vehicle_tracks:
        tid = tr["track_id"]
        active_vehicle_ids.add(tid)

        if tid not in vehicle_objects:
            print(f"[INFO]   New vehicle track -> creating object for track_id={tid}")
            obj = load_or_create_vehicle(f"track_{tid}", color=palette[tid % len(palette)])
            vehicle_objects[tid] = obj
            total_vehicle_tracks_created += 1
            print(f"[INFO]   Total created vehicle objects: {total_vehicle_tracks_created}")

        obj = vehicle_objects[tid]
        set_vehicle_transform_world(
            obj,
            ego_world_y,
            tr["position_cam_xyz"],
            tr.get("yaw_rad", 0.0),
            tr.get("scale", 1.0)
        )
        set_hide_recursive(obj, False)

    if traffic_light_object is None:
        traffic_light_object = create_traffic_light_rig("traffic_light_main")
        set_hide_recursive(traffic_light_object, True)

    if single_light is not None:
        set_traffic_light_transform_world(
            traffic_light_object,
            ego_world_y,
            single_light["position_cam_xyz"],
            single_light.get("traffic_light_state", "unknown"),
        )
        set_hide_recursive(traffic_light_object, False)
    else:
        set_hide_recursive(traffic_light_object, True)

    for tid, obj in vehicle_objects.items():
        if tid not in active_vehicle_ids:
            set_hide_recursive(obj, True)

    bpy.context.view_layer.update()

    out_path = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{scene_frame:04d}.png")
    print(f"[INFO]   Rendering current frame to: {out_path}")
    render_frame_still(scene, scene_frame, out_path)

    if os.path.isfile(out_path):
        print(f"[INFO]   Saved: {out_path}")
    else:
        print(f"[WARN]   Expected frame missing after render: {out_path}")

    if ((f_idx + 1) % PROGRESS_EVERY_N_FRAMES) == 0:
        try:
            saved_count = len([x for x in os.listdir(OUTPUT_FRAMES_DIR) if x.lower().endswith(".png")])
            print(f"[INFO]   PNG count so far: {saved_count}")
        except Exception as e:
            print(f"[WARN]   Could not count PNG files: {e}")

    gc.collect()

print("[INFO] Per-frame render finished.")

try:
    saved = sorted([x for x in os.listdir(OUTPUT_FRAMES_DIR) if x.lower().endswith(".png")])
    print(f"[INFO] PNG files found after render: {len(saved)}")
    if saved:
        print(f"[INFO] First saved frame: {os.path.join(OUTPUT_FRAMES_DIR, saved[0])}")
        print(f"[INFO] Last saved frame:  {os.path.join(OUTPUT_FRAMES_DIR, saved[-1])}")
    else:
        print("[WARN] No PNG frames found in output directory after render.")
except Exception as e:
    print(f"[WARN] Could not inspect output dir: {e}")

print(f"[INFO] To convert PNGs to video later, write to: {OUTPUT_VIDEO_PATH}")