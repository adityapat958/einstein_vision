import bpy
import os
import json
import math
from mathutils import Vector, Euler

# =========================================================
# CONFIG
# =========================================================

SCENE_JSON_PATH = "/home/alien/cv_p3/scene11/tesla_dashboard_phase1/scene_states.json"
OUTPUT_FRAMES_DIR = "/home/alien/cv_p3/scene11/tesla_dashboard_phase1/blender_frames"

ASSET_LIBRARY = {
    "car": [
        {
            "blend_path": "/home/alien/cv_p3/P3Data/Assets/Vehicles/SedanAndHatchback.blend",
            "name": "Car",
        },
        {
            "blend_path": "/home/alien/cv_p3/P3Data/Assets/Vehicles/SUV.blend",
            "name": "Jeep_3_",
        },
    ],
    "truck": [
        {
            "blend_path": "/home/alien/cv_p3/P3Data/Assets/Vehicles/Truck.blend",
            "name": "Truck",
        }
    ],
    "motorcycle": [
        {
            "blend_path": "/home/alien/cv_p3/P3Data/Assets/Vehicles/Motorcycle.blend",
            "name": None,
        }
    ],
    "bicycle": [
        {
            "blend_path": "/home/alien/cv_p3/P3Data/Assets/Vehicles/Bicycle.blend",
            "name": None,
        }
    ],
    "pedestrian": [
        {
            "blend_path": "/home/alien/cv_p3/P3Data/Assets/Pedestrain.blend",
            "name": None,
        }
    ],
}

DEBUG_NUM_FRAMES = 100

# Render
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
RENDER_PERCENT = 100
USE_EEVEE = True

# Camera
CAMERA_LOCATION = (0.0, -16.0, 5.5)
CAMERA_TARGET_LOCATION = (0.0, 18.0, 1.2)
CAMERA_LENS_MM = 35

# World / road
ADD_ROAD_PLANE = True
ROAD_PLANE_SIZE = 800.0
ROAD_Y_CENTER = 70.0

# Placement
LATERAL_SPREAD = 8.5
FORWARD_MIN = 8.0
FORWARD_MAX = 48.0
GROUND_Z = 0.0

# =========================================================
# ORIENTATION FIX
# =========================================================
# For dashboard / rear-view traffic scenes, detections usually do not carry
# reliable 3D heading. So default is: all vehicles follow the road direction.
USE_DETECTION_YAW = False
ROAD_HEADING_DEG = 0.0   # vehicles ahead move toward +Y in this scene

# If you want to use noisy yaw from JSON, set True.
DETECTION_YAW_SCALE = 1.0

# Per-class final world yaw offsets
CLASS_WORLD_YAW_OFFSET_DEG = {
    "car": 0.0,
    "truck": 0.0,
    "motorcycle": 0.0,
    "bicycle": 0.0,
    "pedestrian": 0.0,
}

# Some assets need local correction even after automatic long-axis alignment.
# Adjust these if one asset still faces the wrong way.
EXACT_ASSET_LOCAL_Z_FIX_DEG = {
    ("/home/alien/cv_p3/P3Data/Assets/Vehicles/SedanAndHatchback.blend", "Car"): 0.0,
    ("/home/alien/cv_p3/P3Data/Assets/Vehicles/SUV.blend", "Jeep_3_"): 180.0,
    ("/home/alien/cv_p3/P3Data/Assets/Vehicles/Truck.blend", "Truck"): 180.0,
}

# Target sizes
TARGET_LENGTH_M = {
    "car": 4.4,
    "truck": 7.5,
    "motorcycle": 2.2,
    "bicycle": 1.8,
    "pedestrian": 1.7,
}

STATIC_COLLECTION_NAME = "StaticScene"
INSTANCES_COLLECTION_NAME = "SpawnedInstances"
ASSET_STASH_COLLECTION_NAME = "AssetStash"

PRINT_DEBUG_FIRST_N_FRAMES = 10

# =========================================================
# GLOBALS
# =========================================================

LOADED_SOURCE_ROOTS = {}
TRACK_INSTANCES = {}

# =========================================================
# BASIC HELPERS
# =========================================================

def deg2rad(x):
    return math.radians(x)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ensure_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for datablock_list in [
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.objects,
    ]:
        for block in list(datablock_list):
            if block.users == 0:
                datablock_list.remove(block)

def setup_render(fps, total_frames):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = total_frames
    scene.render.fps = int(round(fps))
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = RENDER_PERCENT
    scene.render.film_transparent = False

    if scene.view_settings is not None:
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    if scene.display_settings is not None:
        try:
            scene.display_settings.display_device = 'sRGB'
        except Exception:
            pass

    if USE_EEVEE:
        scene.render.engine = 'BLENDER_EEVEE'
        if hasattr(scene, "eevee"):
            eevee = scene.eevee
            if hasattr(eevee, "taa_render_samples"):
                eevee.taa_render_samples = 64
            if hasattr(eevee, "taa_samples"):
                eevee.taa_samples = 32
            if hasattr(eevee, "use_bloom"):
                eevee.use_bloom = False
            if hasattr(eevee, "use_gtao"):
                eevee.use_gtao = True
            if hasattr(eevee, "gtao_quality"):
                eevee.gtao_quality = 0.25
    else:
        scene.render.engine = 'CYCLES'

    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.filepath = os.path.join(OUTPUT_FRAMES_DIR, "frame_")

def make_material(name, base_rgba=(0.8, 0.1, 0.1, 1.0), roughness=0.45):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)

    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    bsdf.inputs["Base Color"].default_value = base_rgba
    bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.0

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def setup_world_background():
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links

    for n in list(nodes):
        nodes.remove(n)

    bg = nodes.new(type="ShaderNodeBackground")
    out = nodes.new(type="ShaderNodeOutputWorld")

    bg.inputs["Color"].default_value = (0.82, 0.82, 0.82, 1.0)
    bg.inputs["Strength"].default_value = 0.18

    links.new(bg.outputs["Background"], out.inputs["Surface"])

def look_at(obj, target):
    direction = target - obj.location
    quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = quat.to_euler()

def setup_static_scene():
    static_col = ensure_collection(STATIC_COLLECTION_NAME)
    setup_world_background()

    if ADD_ROAD_PLANE:
        bpy.ops.mesh.primitive_plane_add(size=ROAD_PLANE_SIZE, location=(0, ROAD_Y_CENTER, 0))
        road = bpy.context.active_object
        road.name = "RoadPlane"

        if road.users_collection:
            for c in list(road.users_collection):
                c.objects.unlink(road)
        static_col.objects.link(road)

        road.scale = (1.0, 3.0, 1.0)
        road_mat = make_material("RoadMaterial", (0.32, 0.32, 0.32, 1.0), 0.95)
        road.data.materials.clear()
        road.data.materials.append(road_mat)

    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    cam = bpy.context.active_object
    cam.name = "DashboardCamera"
    cam.data.clip_start = 0.1
    cam.data.clip_end = 1000.0
    cam.data.lens = CAMERA_LENS_MM
    look_at(cam, Vector(CAMERA_TARGET_LOCATION))

    if cam.users_collection:
        for c in list(cam.users_collection):
            c.objects.unlink(cam)
    static_col.objects.link(cam)
    bpy.context.scene.camera = cam

    bpy.ops.object.light_add(type='SUN', location=(0, -10, 20))
    sun = bpy.context.active_object
    sun.name = "MainSun"
    sun.data.energy = 1.0
    sun.rotation_euler = (deg2rad(42), 0.0, deg2rad(20))
    if sun.users_collection:
        for c in list(sun.users_collection):
            c.objects.unlink(sun)
    static_col.objects.link(sun)

    bpy.ops.object.light_add(type='AREA', location=(0, -8, 8))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 800
    fill.data.shape = 'RECTANGLE'
    fill.data.size = 18
    fill.data.size_y = 10
    fill.rotation_euler = (deg2rad(75), 0.0, 0.0)
    if fill.users_collection:
        for c in list(fill.users_collection):
            c.objects.unlink(fill)
    static_col.objects.link(fill)

# =========================================================
# ASSET HELPERS
# =========================================================

def list_blend_objects(blend_path):
    if not os.path.isfile(blend_path):
        raise FileNotFoundError(f"Blend file not found: {blend_path}")
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        return list(data_from.objects)

def resolve_object_name(asset_def):
    blend_path = asset_def["blend_path"]
    requested_name = asset_def.get("name", None)
    objects = list_blend_objects(blend_path)

    if requested_name is not None and requested_name in objects:
        return requested_name

    if requested_name is not None and requested_name not in objects:
        print(f"[WARN] Requested object '{requested_name}' not found in {blend_path}")
        print(f"[WARN] Available objects: {objects}")

    bad_tokens = ["camera", "light", "lamp", "sun", "plane", "cube", "world"]
    candidates = []
    for name in objects:
        lname = name.lower()
        if any(tok in lname for tok in bad_tokens):
            continue
        candidates.append(name)

    preferred_tokens = ["car", "truck", "jeep", "suv", "bike", "motor", "cycle", "bicycle", "ped", "human", "man", "woman"]

    for name in candidates:
        lname = name.lower()
        if any(tok in lname for tok in preferred_tokens):
            print(f"[INFO] Auto-using object '{name}' from {blend_path}")
            return name

    if candidates:
        print(f"[INFO] Fallback object '{candidates[0]}' from {blend_path}")
        return candidates[0]

    raise RuntimeError(f"No usable object found in {blend_path}")

def append_root_object(blend_path, object_name):
    key = (blend_path, object_name)
    if key in LOADED_SOURCE_ROOTS:
        return LOADED_SOURCE_ROOTS[key]

    directory = os.path.join(blend_path, "Object")
    before = set(bpy.data.objects.keys())

    bpy.ops.wm.append(
        filepath=os.path.join(directory, object_name),
        directory=directory + os.sep,
        filename=object_name,
        link=False,
        autoselect=False,
    )

    after = set(bpy.data.objects.keys())
    new_names = list(after - before)

    obj = bpy.data.objects.get(object_name)
    if obj is None and new_names:
        for n in new_names:
            cand = bpy.data.objects.get(n)
            if cand is not None:
                obj = cand
                break

    if obj is None:
        raise RuntimeError(f"Could not append object '{object_name}' from {blend_path}")

    stash_col = ensure_collection(ASSET_STASH_COLLECTION_NAME)
    if len(obj.users_collection) == 0:
        stash_col.objects.link(obj)
    else:
        for c in list(obj.users_collection):
            if c != stash_col:
                try:
                    c.objects.unlink(obj)
                except Exception:
                    pass
        if stash_col not in obj.users_collection:
            stash_col.objects.link(obj)

    obj.hide_render = True
    obj.hide_viewport = True

    LOADED_SOURCE_ROOTS[key] = obj
    return obj

def choose_variant(asset_class, track_id):
    variants = ASSET_LIBRARY.get(asset_class, [])
    if not variants:
        return None
    return variants[abs(int(track_id)) % len(variants)]

# =========================================================
# OBJECT / HIERARCHY HELPERS
# =========================================================

def gather_hierarchy_objects(root):
    objs = [root]
    for child in root.children:
        objs.extend(gather_hierarchy_objects(child))
    return objs

def gather_mesh_descendants(root):
    return [o for o in gather_hierarchy_objects(root) if o.type == "MESH"]

def add_object_to_instances(obj):
    instances_col = ensure_collection(INSTANCES_COLLECTION_NAME)
    if obj.users_collection:
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
    instances_col.objects.link(obj)

def create_controller_empty(instance_name):
    instances_col = ensure_collection(INSTANCES_COLLECTION_NAME)
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    ctrl = bpy.context.active_object
    ctrl.name = instance_name
    if ctrl.users_collection:
        for c in list(ctrl.users_collection):
            c.objects.unlink(ctrl)
    instances_col.objects.link(ctrl)
    return ctrl

def duplicate_hierarchy(root_obj, name_prefix):
    original_nodes = gather_hierarchy_objects(root_obj)
    orig_to_dup = {}

    for src in original_nodes:
        dup = src.copy()
        if src.data is not None:
            try:
                dup.data = src.data.copy()
            except Exception:
                pass
        dup.animation_data_clear()
        dup.name = f"{name_prefix}__{src.name}"
        add_object_to_instances(dup)
        orig_to_dup[src] = dup

    for src in original_nodes:
        dup = orig_to_dup[src]
        if src.parent in orig_to_dup:
            dup.parent = orig_to_dup[src.parent]
            dup.matrix_parent_inverse = src.matrix_parent_inverse.copy()
        else:
            dup.parent = None
        dup.matrix_basis = src.matrix_basis.copy()

    dup_root = orig_to_dup[root_obj]
    return dup_root, list(orig_to_dup.values())

def unhide_recursive(obj):
    for o in gather_hierarchy_objects(obj):
        o.hide_render = False
        o.hide_viewport = False
        if hasattr(o, "hide_set"):
            try:
                o.hide_set(False)
            except Exception:
                pass

def set_simple_asset_material(root, asset_class):
    if asset_class == "car":
        mat = make_material("CarSimpleMat", (0.82, 0.12, 0.12, 1.0), 0.5)
    elif asset_class == "truck":
        mat = make_material("TruckSimpleMat", (0.10, 0.35, 0.82, 1.0), 0.55)
    elif asset_class == "motorcycle":
        mat = make_material("MotoSimpleMat", (0.90, 0.25, 0.10, 1.0), 0.45)
    elif asset_class == "bicycle":
        mat = make_material("BikeSimpleMat", (0.12, 0.70, 0.15, 1.0), 0.45)
    else:
        mat = make_material("PedSimpleMat", (0.65, 0.15, 0.65, 1.0), 0.65)

    for obj in gather_mesh_descendants(root):
        if obj.data is not None:
            obj.data.materials.clear()
            obj.data.materials.append(mat)

def combined_local_bbox(root):
    meshes = gather_mesh_descendants(root)
    if not meshes:
        return None

    points = []
    root_inv = root.matrix_world.inverted()

    for m in meshes:
        for corner in m.bound_box:
            pw = m.matrix_world @ Vector(corner)
            pl = root_inv @ pw
            points.append(pl)

    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_v, max_v

def combined_world_bbox(root):
    meshes = gather_mesh_descendants(root)
    if not meshes:
        return None

    points = []
    for m in meshes:
        for corner in m.bound_box:
            pw = m.matrix_world @ Vector(corner)
            points.append(pw)

    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_v, max_v

def normalize_asset_size(root, asset_class):
    bbox = combined_local_bbox(root)
    if bbox is None:
        print(f"[WARN] No mesh bbox found for asset {root.name}")
        return

    min_v, max_v = bbox
    dims = max_v - min_v
    current_length = max(dims.x, dims.y)

    if current_length < 1e-5:
        print(f"[WARN] current_length too small for {root.name}")
        return

    target_length = TARGET_LENGTH_M.get(asset_class, 4.0)
    scale_factor = target_length / current_length
    root.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()

def ground_align_root(root, desired_z=0.0):
    bbox = combined_world_bbox(root)
    if bbox is None:
        return
    min_v, _ = bbox
    root.location.z += (desired_z - min_v.z)
    bpy.context.view_layer.update()

# =========================================================
# AUTO ASSET ORIENTATION FIX
# =========================================================

def bbox_dims_local(root):
    bbox = combined_local_bbox(root)
    if bbox is None:
        return None
    min_v, max_v = bbox
    return max_v - min_v

def auto_align_asset_long_axis_to_y(root):
    """
    Rotate local Z by 0/90/180/270 and choose the one where
    the object's longer footprint axis aligns with local Y.
    """
    original_rot = root.rotation_euler.copy()
    best_deg = 0.0
    best_score = -1e18

    for deg in [0.0, 90.0, 180.0, 270.0]:
        root.rotation_euler = Euler((original_rot.x, original_rot.y, deg2rad(deg)), 'XYZ')
        bpy.context.view_layer.update()

        dims = bbox_dims_local(root)
        if dims is None:
            continue

        # Prefer long dimension along Y and compact along X
        score = 3.0 * dims.y - 1.0 * dims.x

        if score > best_score:
            best_score = score
            best_deg = deg

    root.rotation_euler = Euler((original_rot.x, original_rot.y, deg2rad(best_deg)), 'XYZ')
    bpy.context.view_layer.update()
    return best_deg

def apply_exact_asset_local_fix(root, blend_path, object_name):
    fix_deg = EXACT_ASSET_LOCAL_Z_FIX_DEG.get((blend_path, object_name), 0.0)
    root.rotation_euler.z += deg2rad(fix_deg)
    bpy.context.view_layer.update()
    return fix_deg

# =========================================================
# PLACEMENT
# =========================================================

def dashboard_position_from_detection(obj_info, image_w, image_h):
    bbox = obj_info.get("bbox_2d", None)

    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        u = 0.5 * (x1 + x2)
        h = max(1.0, float(y2 - y1))
        bottom = float(y2)

        cx = 0.5 * image_w
        lateral_norm = (u - cx) / max(cx, 1.0)
        x = lateral_norm * LATERAL_SPREAD

        # Two cues:
        # 1) taller box => closer
        # 2) lower bottom edge => closer
        forward_from_height = 900.0 / (h + 18.0)
        bottom_norm = clamp(bottom / max(image_h, 1.0), 0.0, 1.0)
        forward_from_bottom = FORWARD_MAX - bottom_norm * (FORWARD_MAX - FORWARD_MIN)

        forward = 0.65 * forward_from_height + 0.35 * forward_from_bottom
        forward = clamp(forward, FORWARD_MIN, FORWARD_MAX)

        return Vector((x, forward, GROUND_Z))

    x = float(obj_info.get("x_world", 0.0)) * 4.0
    y = clamp(float(obj_info.get("y_world", 8.0)) * 4.0, FORWARD_MIN, FORWARD_MAX)
    return Vector((x, y, GROUND_Z))

def compute_final_world_yaw(obj_info, asset_class):
    base_yaw = deg2rad(ROAD_HEADING_DEG + CLASS_WORLD_YAW_OFFSET_DEG.get(asset_class, 0.0))

    if USE_DETECTION_YAW:
        det_yaw = float(obj_info.get("yaw", 0.0))
        return base_yaw + DETECTION_YAW_SCALE * det_yaw
    else:
        return base_yaw

# =========================================================
# INSTANCE MANAGEMENT
# =========================================================

def create_instance_from_asset(asset_class, track_id):
    asset_def = choose_variant(asset_class, track_id)
    if asset_def is None:
        return None

    blend_path = asset_def["blend_path"]
    object_name = resolve_object_name(asset_def)
    source_root = append_root_object(blend_path, object_name)

    print(f"[INFO] track={track_id} class={asset_class} source='{source_root.name}' type={source_root.type}")

    dup_root, dup_nodes = duplicate_hierarchy(source_root, f"{asset_class}_{track_id}")
    controller = create_controller_empty(f"{asset_class}_{track_id}_CTRL")

    dup_root.parent = controller
    dup_root.matrix_parent_inverse = controller.matrix_world.inverted()

    unhide_recursive(controller)
    set_simple_asset_material(dup_root, asset_class)

    auto_deg = auto_align_asset_long_axis_to_y(dup_root)
    fix_deg = apply_exact_asset_local_fix(dup_root, blend_path, object_name)

    normalize_asset_size(dup_root, asset_class)
    ground_align_root(controller, desired_z=GROUND_Z)

    bbox = combined_world_bbox(controller)
    if bbox is not None:
        min_v, max_v = bbox
        dims = max_v - min_v
        print(f"[INFO] final dims for {asset_class}_{track_id}: ({dims.x:.3f}, {dims.y:.3f}, {dims.z:.3f})")
    else:
        print(f"[WARN] No visible mesh found after duplication for {asset_class}_{track_id}")

    print(
        f"[INFO] asset local alignment: class={asset_class} track={track_id} "
        f"auto_align_z={auto_deg:.1f} deg, exact_fix_z={fix_deg:.1f} deg"
    )

    return {
        "controller": controller,
        "root_obj": dup_root,
        "asset_class": asset_class,
        "blend_path": blend_path,
        "object_name": object_name,
    }

def ensure_track_instance(track_id, asset_class):
    key = int(track_id)
    existing = TRACK_INSTANCES.get(key)

    if existing is not None:
        if existing["asset_class"] == asset_class:
            return existing
        try:
            for o in reversed(gather_hierarchy_objects(existing["controller"])):
                bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass
        del TRACK_INSTANCES[key]

    entry = create_instance_from_asset(asset_class, key)
    if entry is None:
        return None

    for o in gather_hierarchy_objects(entry["controller"]):
        o.hide_render = True
        o.hide_viewport = True

    TRACK_INSTANCES[key] = entry
    return entry

def insert_visibility_key_recursive(obj, visible, frame_idx):
    for o in gather_hierarchy_objects(obj):
        o.hide_render = not visible
        o.hide_viewport = not visible
        o.keyframe_insert(data_path="hide_render", frame=frame_idx)
        o.keyframe_insert(data_path="hide_viewport", frame=frame_idx)

def keyframe_controller_transform(controller, location, yaw_rad, frame_idx):
    controller.location = location
    controller.rotation_euler = (0.0, 0.0, yaw_rad)
    bpy.context.view_layer.update()
    ground_align_root(controller, desired_z=GROUND_Z)

    controller.keyframe_insert(data_path="location", frame=frame_idx)
    controller.keyframe_insert(data_path="rotation_euler", frame=frame_idx)

def _get_action_fcurves(action):
    if action is None:
        return []
    if hasattr(action, "fcurves"):
        try:
            return list(action.fcurves)
        except Exception:
            pass
    if hasattr(action, "channels"):
        try:
            return list(action.channels)
        except Exception:
            pass
    return []

def set_interpolation(obj):
    for o in gather_hierarchy_objects(obj):
        ad = getattr(o, "animation_data", None)
        if ad is None:
            continue
        action = getattr(ad, "action", None)
        if action is None:
            continue

        for fcurve in _get_action_fcurves(action):
            kps = getattr(fcurve, "keyframe_points", None)
            if kps is None:
                continue

            data_path = getattr(fcurve, "data_path", "")
            for kp in kps:
                if data_path in {"hide_render", "hide_viewport"}:
                    kp.interpolation = 'CONSTANT'
                else:
                    kp.interpolation = 'LINEAR'

# =========================================================
# MAIN
# =========================================================

def main():
    if not os.path.isfile(SCENE_JSON_PATH):
        raise FileNotFoundError(f"Scene JSON not found: {SCENE_JSON_PATH}")

    with open(SCENE_JSON_PATH, "r") as f:
        data = json.load(f)

    fps = float(data["fps"])
    all_frames = data["frames"][:DEBUG_NUM_FRAMES]
    total_frames = len(all_frames)
    image_w = int(data.get("width", 1920))
    image_h = int(data.get("height", 1080))

    print(f"[INFO] Frames: {total_frames}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Image size: {image_w} x {image_h}")

    clear_scene()
    setup_render(fps, total_frames)
    setup_static_scene()

    for local_idx, frame_state in enumerate(all_frames, start=1):
        frame_idx = local_idx
        bpy.context.scene.frame_set(frame_idx)

        active_track_ids = set()
        objects = frame_state.get("objects", [])

        for obj_info in objects:
            if not bool(obj_info.get("visible", True)):
                continue

            asset_class = obj_info.get("asset_class", None)
            track_id = int(obj_info.get("track_id", -1))

            if asset_class not in {"car", "truck", "motorcycle", "bicycle", "pedestrian"}:
                continue
            if track_id < 0:
                continue

            track_entry = ensure_track_instance(track_id, asset_class)
            if track_entry is None:
                continue

            controller = track_entry["controller"]
            location = dashboard_position_from_detection(obj_info, image_w, image_h)
            yaw = compute_final_world_yaw(obj_info, asset_class)

            if frame_idx <= PRINT_DEBUG_FIRST_N_FRAMES:
                print(
                    f"[DEBUG] frame={frame_idx} class={asset_class} track={track_id} "
                    f"bbox={obj_info.get('bbox_2d', None)} "
                    f"loc=({location.x:.2f},{location.y:.2f},{location.z:.2f}) "
                    f"world_yaw_deg={math.degrees(yaw):.2f}"
                )

            insert_visibility_key_recursive(controller, True, frame_idx)
            keyframe_controller_transform(controller, location, yaw, frame_idx)
            active_track_ids.add(track_id)

        for tid, entry in TRACK_INSTANCES.items():
            if tid not in active_track_ids:
                insert_visibility_key_recursive(entry["controller"], False, frame_idx)

    for entry in TRACK_INSTANCES.values():
        set_interpolation(entry["controller"])

    bpy.context.scene.frame_set(1)
    print(f"[INFO] Rendering to: {OUTPUT_FRAMES_DIR}")
    bpy.ops.render.render(animation=True)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()