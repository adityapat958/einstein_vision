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

PEDESTRIAN_BLEND_PATH = "/home/alien/cv_p3/P3Data/Assets/Pedestrain.blend"
PEDESTRIAN_OBJECT_NAME = ""
PEDESTRIAN_COLLECTION_NAME = ""

OUTPUT_FRAMES_DIR = "/home/alien/cv_p3/pipeline_out/blender_frames"
OUTPUT_VIDEO_PATH = "/home/alien/cv_p3/pipeline_out/blender_render.mp4"

WORLD_SCALE = 1.0
GROUND_Z = 0.0
ROAD_WIDTH = 11.5
ROAD_LENGTH = 2400.0
ROAD_CENTER_X = 0.0
ROAD_CENTER_Y = 900.0

EGO_CAR = True
EGO_CAR_LOCATION = (0.0, 0.0, 0.55)
EGO_CAR_SCALE = (1.10, 1.10, 1.10)
EGO_SPEED_MPS = 8.0

# Tesla-like trailing view
CAM_BACK = 8.0
CAM_HEIGHT = 3.2
CAM_FORWARD_LOOK = 26.0
CAM_LENS = 38

RENDER_RES_X = 1280
RENDER_RES_Y = 720
RENDER_ENGINE = "BLENDER_EEVEE"
RENDER_SAMPLES = 32

MAX_FRAMES_TO_RENDER = None
PROGRESS_EVERY_N_FRAMES = 10

# -------------------------
# Asset normalization
# -------------------------
TARGET_CAR_LENGTH = 4.6
TARGET_PEDESTRIAN_HEIGHT = 1.72

ASSET_Z_OFFSET = 0.0
ASSET_YAW_OFFSET_DEG = 180.0

# Pedestrian tuning
PEDESTRIAN_GLOBAL_SCALE = 0.34
PEDESTRIAN_LEFT_INWARD_YAW_DEG = 0.0
PEDESTRIAN_RIGHT_INWARD_YAW_DEG = 180.0

FORCE_SIMPLE_MATERIALS = True
ASSET_BASE_COLOR = (0.10, 0.10, 0.10, 1.0)
ASSET_GLASS_COLOR = (0.05, 0.06, 0.08, 1.0)
ASSET_TYRE_COLOR = (0.02, 0.02, 0.02, 1.0)
ASSET_METAL_COLOR = (0.45, 0.45, 0.45, 1.0)

PEDESTRIAN_BASE_COLOR = (0.72, 0.72, 0.76, 1.0)

# -------------------------
# Filtering / overlap tuning
# -------------------------
MAX_VISIBLE_VEHICLES = 12
MAX_VISIBLE_PEDESTRIANS = 14

MERGE_DIST_X = 1.5
MERGE_DIST_Y = 4.0
MIN_FORWARD_GAP = 6.0

LANE_CENTERS = [-3.5, 0.0, 3.5]
MAX_SIDE_OFFSET = 8.0
MIN_FORWARD_DIST = 4.0
MAX_FORWARD_DIST = 75.0
FORCE_LANE_ALIGNMENT = True

MAX_PEDESTRIAN_SIDE_ABS = 8.5

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
# Labels
# =========================================================
def normalize_label(name):
    return str(name).strip().lower().replace("_", " ")

def is_vehicle_label(name):
    n = normalize_label(name)
    vehicle_aliases = {
        "car", "truck", "bus", "motorcycle", "bicycle",
        "van", "pickup truck", "pickup", "automobile",
        "suv", "minivan"
    }
    if n in vehicle_aliases:
        return True
    if n == "car" or n.endswith(" car"):
        return True
    if "truck" in n:
        return True
    if "bus" in n:
        return True
    if "motorcycle" in n:
        return True
    if "bicycle" in n or n == "bike":
        return True
    return False

def is_person_label(name):
    n = normalize_label(name)
    person_aliases = {"person", "pedestrian", "man", "woman", "boy", "girl", "people"}
    if n in person_aliases:
        return True
    if "pedestrian" in n:
        return True
    return False

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
    for block in list(bpy.data.curves):
        if block.users == 0:
            bpy.data.curves.remove(block)
    for block in list(bpy.data.collections):
        if block.users == 0 and block.name != "Scene Collection":
            bpy.data.collections.remove(block)
    print("[INFO] Scene cleared.")

def make_material(name, rgba, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
    return mat

def setup_world():
    print("[INFO] Setting up world...")
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.015, 0.017, 0.022, 1.0)
        bg.inputs[1].default_value = 0.9
    print("[INFO] World ready.")

def create_ground_and_road():
    print("[INFO] Creating Tesla-like road layout...")

    # outer ground
    bpy.ops.mesh.primitive_plane_add(size=4000, location=(0, 1000, GROUND_Z - 0.002))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground_mat = make_material("GroundMat", (0.22, 0.22, 0.22, 1.0), roughness=1.0)
    ground.data.materials.append(ground_mat)

    # road slab
    bpy.ops.mesh.primitive_plane_add(size=2, location=(ROAD_CENTER_X, ROAD_CENTER_Y, GROUND_Z))
    road = bpy.context.active_object
    road.name = "Road"
    road.scale = (ROAD_WIDTH * 0.5, ROAD_LENGTH * 0.5, 1.0)
    road_mat = make_material("RoadMat", (0.08, 0.08, 0.08, 1.0), roughness=0.95)
    road.data.materials.append(road_mat)

    # shoulders
    shoulder_mat = make_material("ShoulderMat", (0.12, 0.12, 0.12, 1.0), roughness=1.0)
    for sx in (-ROAD_WIDTH * 0.5 - 0.7, ROAD_WIDTH * 0.5 + 0.7):
        bpy.ops.mesh.primitive_plane_add(size=2, location=(sx, ROAD_CENTER_Y, GROUND_Z + 0.001))
        s = bpy.context.active_object
        s.scale = (0.7, ROAD_LENGTH * 0.5, 1.0)
        s.data.materials.append(shoulder_mat)

    print("[INFO] Road layout created.")
    return road

def create_lane_markings():
    print("[INFO] Creating lane markings...")

    lane_mat = make_material("LaneMat", (0.95, 0.95, 0.92, 1.0), roughness=0.35)

    # dashed center separators
    dashed_x = [-1.75, 1.75]
    for lane_x in dashed_x:
        for y in range(-40, 2200, 10):
            bpy.ops.mesh.primitive_cube_add(location=(lane_x, y, GROUND_Z + 0.01))
            dash = bpy.context.active_object
            dash.scale = (0.06, 1.3, 0.01)
            dash.data.materials.append(lane_mat)

    # outer lane boundaries
    solid_x = [-5.25, 5.25]
    for lane_x in solid_x:
        for y in range(-40, 2200, 4):
            bpy.ops.mesh.primitive_cube_add(location=(lane_x, y, GROUND_Z + 0.011))
            line = bpy.context.active_object
            line.scale = (0.05, 1.8, 0.01)
            line.data.materials.append(lane_mat)

    print("[INFO] Lane markings created.")

def create_side_environment():
    print("[INFO] Creating minimal side environment...")
    tree_trunk_mat = make_material("TreeTrunkMat", (0.22, 0.22, 0.22, 1.0), roughness=0.9)
    tree_leaf_mat = make_material("TreeLeafMat", (0.18, 0.34, 0.18, 1.0), roughness=0.95)

    for y in range(-20, 2200, 28):
        for x in (-13.5, 13.5):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=3.8, location=(x, y, 1.9))
            trunk = bpy.context.active_object
            trunk.data.materials.append(tree_trunk_mat)

            bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(x, y, 4.4))
            crown = bpy.context.active_object
            crown.data.materials.append(tree_leaf_mat)

    print("[INFO] Minimal side environment created.")

def setup_lights():
    print("[INFO] Setting up lights...")
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 25))
    sun = bpy.context.active_object
    sun.data.energy = 3.8
    sun.rotation_euler = (math.radians(52), 0.0, math.radians(18))

    bpy.ops.object.light_add(type='AREA', location=(0, -6, 7))
    area = bpy.context.active_object
    area.data.energy = 900
    area.data.shape = 'RECTANGLE'
    area.data.size = 8
    area.data.size_y = 6
    print("[INFO] Lights ready.")

def create_camera_rig():
    print("[INFO] Creating camera rig...")
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.6))
    cam_target = bpy.context.active_object
    cam_target.name = "CameraTarget"

    bpy.ops.object.camera_add(location=(0, -CAM_BACK, CAM_HEIGHT))
    cam = bpy.context.active_object
    cam.name = "TPP_Camera"
    cam.data.lens = CAM_LENS
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

    bpy.ops.mesh.primitive_cube_add(location=(0, 0.1, 1.0))
    roof = bpy.context.active_object
    roof.scale = (0.65, 0.95, 0.32)
    roof.parent = body
    roof.data.materials.append(mat)
    return body

def make_placeholder_pedestrian(name="ped_proto", color=(0.65, 0.65, 0.72, 1.0)):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    root = bpy.context.active_object
    root.name = name

    body_mat = make_material(f"{name}_mat", color, roughness=0.7)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.16, location=(0, 0, 1.65))
    head = bpy.context.active_object
    head.parent = root
    head.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.18, depth=0.70, location=(0, 0, 1.15))
    torso = bpy.context.active_object
    torso.parent = root
    torso.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.80, location=(-0.08, 0, 0.45))
    leg_l = bpy.context.active_object
    leg_l.parent = root
    leg_l.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.80, location=(0.08, 0, 0.45))
    leg_r = bpy.context.active_object
    leg_r.parent = root
    leg_r.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.65, location=(-0.22, 0, 1.18))
    arm_l = bpy.context.active_object
    arm_l.rotation_euler[1] = math.radians(20)
    arm_l.parent = root
    arm_l.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.65, location=(0.22, 0, 1.18))
    arm_r = bpy.context.active_object
    arm_r.rotation_euler[1] = math.radians(-20)
    arm_r.parent = root
    arm_r.data.materials.append(body_mat)

    return root

# =========================================================
# Asset helpers
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

def choose_pedestrian_material_style(obj_name):
    low = obj_name.lower()
    if any(k in low for k in ["shoe", "boot"]):
        return (0.08, 0.08, 0.08, 1.0), 0.0, 0.85
    if any(k in low for k in ["skin", "face", "head", "hand"]):
        return (0.72, 0.60, 0.52, 1.0), 0.0, 0.6
    return PEDESTRIAN_BASE_COLOR, 0.0, 0.65

def apply_simple_materials(root):
    for obj in all_descendants(root):
        if obj.type != "MESH":
            continue
        rgba, metallic, roughness = choose_material_style(obj.name)
        mat = make_material(f"{obj.name}_simple", rgba, metallic=metallic, roughness=roughness)
        obj.data.materials.clear()
        obj.data.materials.append(mat)

def apply_pedestrian_materials(root):
    for obj in all_descendants(root):
        if obj.type != "MESH":
            continue
        rgba, metallic, roughness = choose_pedestrian_material_style(obj.name)
        mat = make_material(f"{obj.name}_ped_simple", rgba, metallic=metallic, roughness=roughness)
        obj.data.materials.clear()
        obj.data.materials.append(mat)

def normalize_asset_root_to_length(root, target_length):
    bpy.context.view_layer.update()
    mn, mx = hierarchy_bbox(root)
    if mn is None:
        return
    size = mx - mn
    obj_len = max(size.x, size.y, 1e-6)
    s = target_length / obj_len
    root.scale = root.scale * s
    bpy.context.view_layer.update()

    mn, mx = hierarchy_bbox(root)
    if mn is None:
        return
    center = (mn + mx) * 0.5
    root.location = root.location - Vector((center.x, center.y, mn.z))
    bpy.context.view_layer.update()

def normalize_asset_root_to_height(root, target_height):
    bpy.context.view_layer.update()
    mn, mx = hierarchy_bbox(root)
    if mn is None:
        return
    size = mx - mn
    obj_h = max(size.z, 1e-6)
    s = target_height / obj_h
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

def append_full_asset(name, blend_path, object_name="", collection_name="", kind="car"):
    blend_path = resolve_blend_path(blend_path)

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        available_objects = list(data_from.objects)
        available_collections = list(data_from.collections)

    print(f"[INFO] Asset file: {blend_path}")
    print(f"[INFO] Requested object: {object_name if object_name else '(auto full asset)'}")
    print(f"[INFO] Available objects: {available_objects[:40]}")
    print(f"[INFO] Available collections: {available_collections[:20]}")

    imported = []

    if collection_name and collection_name in available_collections:
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
            data_to.collections = [collection_name]

        for coll in data_to.collections:
            if coll is not None:
                bpy.context.scene.collection.children.link(coll)
                for obj in coll.objects:
                    if obj.type == "MESH" and not is_skippable_asset_name(obj.name):
                        imported.append(obj)

    elif object_name and object_name in available_objects:
        with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
            data_to.objects = [object_name]

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
        print(f"[WARN] No valid mesh asset objects imported for {kind}.")
        return None

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    root = bpy.context.active_object
    root.name = name

    for obj in imported:
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()

    if kind == "car":
        normalize_asset_root_to_length(root, TARGET_CAR_LENGTH)
        if FORCE_SIMPLE_MATERIALS:
            apply_simple_materials(root)
    elif kind == "pedestrian":
        normalize_asset_root_to_height(root, TARGET_PEDESTRIAN_HEIGHT)
        if FORCE_SIMPLE_MATERIALS:
            apply_pedestrian_materials(root)

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
PEDESTRIAN_PROTOTYPE = None

def get_vehicle_prototype(color=(0.7, 0.7, 0.75, 1.0)):
    global CAR_PROTOTYPE
    if CAR_PROTOTYPE is not None:
        return CAR_PROTOTYPE

    print("[INFO] Creating car prototype...")
    if CAR_BLEND_PATH:
        try:
            proto = append_full_asset(
                "__car_prototype__",
                CAR_BLEND_PATH,
                object_name=CAR_OBJECT_NAME,
                collection_name=CAR_COLLECTION_NAME,
                kind="car",
            )
            if proto is not None:
                proto.hide_viewport = True
                proto.hide_render = True
                for ch in proto.children_recursive:
                    ch.hide_viewport = True
                    ch.hide_render = True
                CAR_PROTOTYPE = proto
                return CAR_PROTOTYPE
        except Exception as e:
            print(f"[WARN] Vehicle asset load failed: {e}")

    proto = make_placeholder_vehicle("__car_prototype__", color)
    proto.hide_viewport = True
    proto.hide_render = True
    CAR_PROTOTYPE = proto
    return CAR_PROTOTYPE

def get_pedestrian_prototype():
    global PEDESTRIAN_PROTOTYPE
    if PEDESTRIAN_PROTOTYPE is not None:
        return PEDESTRIAN_PROTOTYPE

    print("[INFO] Creating pedestrian prototype...")
    if PEDESTRIAN_BLEND_PATH:
        try:
            proto = append_full_asset(
                "__pedestrian_prototype__",
                PEDESTRIAN_BLEND_PATH,
                object_name=PEDESTRIAN_OBJECT_NAME,
                collection_name=PEDESTRIAN_COLLECTION_NAME,
                kind="pedestrian",
            )
            if proto is not None:
                proto.hide_viewport = True
                proto.hide_render = True
                for ch in proto.children_recursive:
                    ch.hide_viewport = True
                    ch.hide_render = True
                PEDESTRIAN_PROTOTYPE = proto
                return PEDESTRIAN_PROTOTYPE
        except Exception as e:
            print(f"[WARN] Pedestrian asset load failed: {e}")

    proto = make_placeholder_pedestrian("__pedestrian_prototype__", PEDESTRIAN_BASE_COLOR)
    proto.hide_viewport = True
    proto.hide_render = True
    PEDESTRIAN_PROTOTYPE = proto
    return PEDESTRIAN_PROTOTYPE

def load_or_create_vehicle(name="vehicle_proto", color=(0.7, 0.7, 0.75, 1.0)):
    proto = get_vehicle_prototype(color=color)
    obj = duplicate_hierarchy(proto, name)
    obj.hide_viewport = False
    obj.hide_render = False
    for ch in obj.children_recursive:
        ch.hide_viewport = False
        ch.hide_render = False
    return obj

def load_or_create_pedestrian(name="ped_proto"):
    proto = get_pedestrian_prototype()
    obj = duplicate_hierarchy(proto, name)
    obj.hide_viewport = False
    obj.hide_render = False
    for ch in obj.children_recursive:
        ch.hide_viewport = False
        ch.hide_render = False
    return obj

# =========================================================
# World placement
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

def set_pedestrian_transform_world(obj, ego_world_y, xyz_cam, yaw_rad=0.0, scale=1.0):
    rel_x, rel_y, _ = cam_to_blender_relative(xyz_cam)
    world_x = EGO_CAR_LOCATION[0] + rel_x
    world_y = EGO_CAR_LOCATION[1] + ego_world_y + rel_y

    # Always keep on ground
    world_z = GROUND_Z

    obj.location = (world_x, world_y, world_z)

    if world_x >= 0.0:
        yaw_deg = PEDESTRIAN_RIGHT_INWARD_YAW_DEG
    else:
        yaw_deg = PEDESTRIAN_LEFT_INWARD_YAW_DEG

    obj.rotation_euler = (0.0, 0.0, math.radians(yaw_deg))

    base_s = max(0.82, min(scale, 1.0))
    s = base_s * PEDESTRIAN_GLOBAL_SCALE
    obj.scale = (s, s, s)

def set_camera_target_from_ego(cam_target, ego_obj):
    ego_loc = ego_obj.location.copy()
    cam_target.location = Vector((
        ego_loc.x,
        ego_loc.y + CAM_FORWARD_LOOK,
        ego_loc.z + 1.35
    ))

def set_camera_position_relative_to_ego(cam_obj, ego_obj):
    ego_loc = ego_obj.location.copy()
    cam_obj.location = Vector((
        ego_loc.x,
        ego_loc.y - CAM_BACK,
        ego_loc.z + CAM_HEIGHT
    ))

def set_hide_recursive(obj, hidden):
    obj.hide_viewport = hidden
    obj.hide_render = hidden
    for ch in obj.children_recursive:
        ch.hide_viewport = hidden
        ch.hide_render = hidden

def render_frame_still(scene, frame_number, filepath):
    scene.frame_set(frame_number)
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

# =========================================================
# Track prep
# =========================================================
def nearest_lane_center(x):
    return min(LANE_CENTERS, key=lambda c: abs(x - c))

def sanitize_vehicle_track(tr):
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
        "group": "vehicle",
        "lane_x": x,
    }

def sanitize_pedestrian_track(tr):
    pos = tr.get("position_cam_xyz", [0.0, 0.0, 0.0])
    if len(pos) < 3:
        return None

    x = float(pos[0])
    z = float(pos[2])
    score = float(tr.get("score", 0.0))
    yaw = float(tr.get("yaw_rad", 0.0))
    scale = float(tr.get("scale", 1.0))
    tid = int(tr.get("track_id", -1))
    cls_name = tr.get("class", "person")

    if z < MIN_FORWARD_DIST or z > MAX_FORWARD_DIST:
        return None
    if abs(x) > MAX_PEDESTRIAN_SIDE_ABS:
        return None

    return {
        "track_id": tid,
        "class": cls_name,
        "score": score,
        "yaw_rad": yaw,
        "scale": max(0.8, min(scale, 1.0)),
        "position_cam_xyz": [x, 0.0, z],
        "group": "pedestrian",
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

            if abs(ax - bx) < MERGE_DIST_X and abs(az - bz) < MERGE_DIST_Y and a["group"] == b["group"]:
                used[j] = True
                if b["score"] > best["score"]:
                    best = b

        cleaned.append(best)

    return cleaned

def enforce_spacing(tracks, min_gap):
    by_band = {}
    for tr in tracks:
        band = round(tr["lane_x"], 1)
        by_band.setdefault(band, []).append(tr)

    final_tracks = []
    for band, band_tracks in by_band.items():
        band_tracks.sort(key=lambda t: t["position_cam_xyz"][2])
        adjusted = []
        last_z = None
        for tr in band_tracks:
            z = tr["position_cam_xyz"][2]
            if last_z is not None and (z - last_z) < min_gap:
                z = last_z + min_gap
            tr["position_cam_xyz"][2] = z
            last_z = z
            adjusted.append(tr)
        final_tracks.extend(adjusted)
    return final_tracks

def prepare_frame_tracks(frame_tracks):
    vehicles = []
    pedestrians = []

    for tr in frame_tracks:
        cls_name = tr.get("class", "")
        if is_vehicle_label(cls_name):
            s = sanitize_vehicle_track(tr)
            if s is not None:
                vehicles.append(s)
        elif is_person_label(cls_name):
            s = sanitize_pedestrian_track(tr)
            if s is not None:
                pedestrians.append(s)

    vehicles.sort(key=lambda t: t["position_cam_xyz"][2])
    pedestrians.sort(key=lambda t: t["position_cam_xyz"][2])

    vehicles = vehicles[:MAX_VISIBLE_VEHICLES]
    pedestrians = pedestrians[:MAX_VISIBLE_PEDESTRIANS]

    vehicles = merge_close_tracks(vehicles)
    pedestrians = merge_close_tracks(pedestrians)

    vehicles = enforce_spacing(vehicles, MIN_FORWARD_GAP)
    pedestrians = enforce_spacing(pedestrians, 3.0)

    final_tracks = vehicles + pedestrians
    final_tracks.sort(key=lambda t: (t["position_cam_xyz"][2], t["position_cam_xyz"][0]))
    return final_tracks

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
create_ground_and_road()
create_lane_markings()
create_side_environment()
setup_lights()
cam, cam_target = create_camera_rig()
print("[INFO] Static scene setup completed.")

scene = bpy.context.scene
scene.render.engine = RENDER_ENGINE
if scene.render.engine == "BLENDER_EEVEE" and hasattr(scene, "eevee"):
    if hasattr(scene.eevee, "use_shadows"):
        scene.eevee.use_shadows = True

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

track_objects = {}
track_types = {}
total_tracks_created = 0

print("[INFO] Starting per-frame Tesla-style rendering...")

for f_idx, f in enumerate(frames):
    frame_idx = f["frame_idx"]
    scene_frame = f_idx + 1

    t_sec = f_idx / max(fps, 1e-6)
    ego_world_y = EGO_SPEED_MPS * t_sec

    if (f_idx % PROGRESS_EVERY_N_FRAMES) == 0:
        print(
            f"[INFO] Processing frame {f_idx + 1}/{len(frames)} "
            f"(source frame_idx={frame_idx}) | ego_world_y={ego_world_y:.2f} | tracks={len(track_objects)}"
        )

    if ego is not None:
        set_ego_transform(ego, ego_world_y)
        set_hide_recursive(ego, False)
        set_camera_position_relative_to_ego(cam, ego)
        set_camera_target_from_ego(cam_target, ego)

    frame_tracks = prepare_frame_tracks(f.get("tracks", []))
    active_ids = set()

    for tr in frame_tracks:
        tid = tr["track_id"]
        active_ids.add(tid)
        cls_name = tr["class"]

        if tid not in track_objects:
            print(f"[INFO]   New track -> track_id={tid}, class={cls_name}")

            if is_vehicle_label(cls_name):
                obj = load_or_create_vehicle(f"track_{tid}", color=palette[tid % len(palette)])
                track_types[tid] = "vehicle"
            elif is_person_label(cls_name):
                obj = load_or_create_pedestrian(f"track_{tid}")
                track_types[tid] = "pedestrian"
            else:
                continue

            track_objects[tid] = obj
            total_tracks_created += 1

        obj = track_objects[tid]
        obj_type = track_types.get(tid, "vehicle")

        if obj_type == "vehicle":
            set_vehicle_transform_world(
                obj,
                ego_world_y,
                tr["position_cam_xyz"],
                tr.get("yaw_rad", 0.0),
                tr.get("scale", 1.0)
            )
        elif obj_type == "pedestrian":
            set_pedestrian_transform_world(
                obj,
                ego_world_y,
                tr["position_cam_xyz"],
                tr.get("yaw_rad", 0.0),
                tr.get("scale", 1.0)
            )

        set_hide_recursive(obj, False)

    for tid, obj in track_objects.items():
        if tid not in active_ids:
            set_hide_recursive(obj, True)

    bpy.context.view_layer.update()

    out_path = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{scene_frame:04d}.png")
    print(f"[INFO]   Rendering: {out_path}")
    render_frame_still(scene, scene_frame, out_path)

    gc.collect()

print("[INFO] Per-frame render finished.")

try:
    saved = sorted([x for x in os.listdir(OUTPUT_FRAMES_DIR) if x.lower().endswith(".png")])
    print(f"[INFO] PNG files found after render: {len(saved)}")
    if saved:
        print(f"[INFO] First saved frame: {os.path.join(OUTPUT_FRAMES_DIR, saved[0])}")
        print(f"[INFO] Last saved frame:  {os.path.join(OUTPUT_FRAMES_DIR, saved[-1])}")
except Exception as e:
    print(f"[WARN] Could not inspect output dir: {e}")

print(f"[INFO] To convert PNGs to video later, write to: {OUTPUT_VIDEO_PATH}")