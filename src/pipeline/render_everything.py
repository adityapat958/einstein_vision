import bpy
import json
import math
import os

# =========================================================
# CONFIGURATION
# =========================================================
JSON_FILE_PATH = "/home/adipat/Documents/Spring_26/CV/p3/src/pipeline_out/scene6.json"
OUTPUT_RENDER_DIR = "/home/adipat/Documents/Spring_26/CV/p3/src/pipeline_out/scene6_blender_renders/"
ASSET_BASE_DIR = "/home/adipat/Documents/Spring_26/CV/p3/P3Data/Assets/"

# Set to True to render only the first 50 frames for a quick test. 
# Set to False to render the entire video.
RENDER_PREVIEW_ONLY = True

ASSET_MAPPING = {
    "bicycle":          os.path.join(ASSET_BASE_DIR, "Vehicles/Bicycle.blend"),
    "motorcycle":       os.path.join(ASSET_BASE_DIR, "Vehicles/Motorcycle.blend"),
    "pickup_truck":     os.path.join(ASSET_BASE_DIR, "Vehicles/PickupTruck.blend"),
    "sedan":            os.path.join(ASSET_BASE_DIR, "Vehicles/SedanAndHatchback.blend"),
    "hatchback":        os.path.join(ASSET_BASE_DIR, "Vehicles/SedanAndHatchback.blend"),
    "suv":              os.path.join(ASSET_BASE_DIR, "Vehicles/SUV.blend"),
    "truck":            os.path.join(ASSET_BASE_DIR, "Vehicles/Truck.blend"),
    "dustbin":          os.path.join(ASSET_BASE_DIR, "Dustbin.blend"),
    "pedestrian":       os.path.join(ASSET_BASE_DIR, "Pedestrain.blend"),
    "speed_limit":      os.path.join(ASSET_BASE_DIR, "SpeedLimitSign.blend"),
    "stop_sign":        os.path.join(ASSET_BASE_DIR, "StopSign.blend"),
    "traffic_cone":     os.path.join(ASSET_BASE_DIR, "TrafficConeAndCylinder.blend"),
    "traffic_cylinder": os.path.join(ASSET_BASE_DIR, "TrafficConeAndCylinder.blend"),
    "traffic_pole":     os.path.join(ASSET_BASE_DIR, "TrafficAssets.blend"),
    "traffic_light":    os.path.join(ASSET_BASE_DIR, "TrafficSignal.blend"),
    "generic_sign":     os.path.join(ASSET_BASE_DIR, "StopSign.blend")
}

ASSET_YAW_OFFSET = math.radians(180)

# Traffic light state → emissive RGB colour (linear)
TL_STATE_COLOR = {
    "red":     (1.0, 0.02, 0.02),
    "yellow":  (1.0, 0.6,  0.0),
    "green":   (0.0, 1.0,  0.1),
    "unknown": (0.1, 0.1,  0.1),
}
TL_EMISSION_STRENGTH = 8.0

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.materials: bpy.data.materials.remove(block)

def load_asset(filepath, track_id):
    if not os.path.exists(filepath):
        bpy.ops.mesh.primitive_cube_add(size=1)
        cube = bpy.context.active_object
        cube.name = f"Track_{track_id}_MISSING"
        return cube

    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = [
            name for name in data_from.objects 
            if not name.startswith("Camera") and not name.startswith("Light")
        ]

    bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1.0)
    root_empty = bpy.context.active_object
    root_empty.name = f"Track_{track_id}"

    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            if not obj.parent:
                obj.parent = root_empty

    return root_empty

def set_visibility_keyframe(obj, frame, is_visible):
    obj.hide_viewport = not is_visible
    obj.hide_render = not is_visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)
    
    for child in obj.children_recursive:
        child.hide_viewport = not is_visible
        child.hide_render = not is_visible
        child.keyframe_insert(data_path="hide_viewport", frame=frame)
        child.keyframe_insert(data_path="hide_render", frame=frame)

def get_or_create_tl_material(state: str):
    """Return a cached emissive material for the given traffic light state."""
    mat_name = f"TL_State_{state}"
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat

    rgb = TL_STATE_COLOR.get(state, TL_STATE_COLOR["unknown"])
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out  = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value    = (*rgb, 1.0)
    emit.inputs["Strength"].default_value = TL_EMISSION_STRENGTH
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


# Arrow direction → arrow mesh name cached per direction

def spawn_tl_text(track_id: int, collection):
    """Spawn two text objects for a traffic light: state label + arrow label.

    Both start hidden. Returns (state_text_obj, arrow_text_obj).
    """
    # State text  e.g. "RED"
    state_curve = bpy.data.curves.new(f"TL_StateText_{track_id}", type='FONT')
    state_curve.body = "?"
    state_curve.size = 0.8
    state_curve.align_x = 'CENTER'
    state_obj = bpy.data.objects.new(f"TL_State_{track_id}", state_curve)
    state_obj.rotation_euler = (math.radians(90), 0, 0)
    collection.objects.link(state_obj)

    # Arrow text  e.g. "↑" / "←" / "→"
    arrow_curve = bpy.data.curves.new(f"TL_ArrowText_{track_id}", type='FONT')
    arrow_curve.body = ""
    arrow_curve.size = 1.2
    arrow_curve.align_x = 'CENTER'
    arrow_obj = bpy.data.objects.new(f"TL_Arrow_{track_id}", arrow_curve)
    arrow_obj.rotation_euler = (math.radians(90), 0, 0)
    collection.objects.link(arrow_obj)

    # Start both hidden
    for obj in (state_obj, arrow_obj):
        obj.hide_viewport = True
        obj.hide_render   = True
        obj.keyframe_insert(data_path="hide_viewport", frame=0)
        obj.keyframe_insert(data_path="hide_render",   frame=0)

    return state_obj, arrow_obj


_ARROW_CHAR = {"left": "←", "right": "→", "straight": "↑", "none": "", "off": ""}


def keyframe_tl_text(state_obj, arrow_obj, state: str, arrow: str,
                     bx: float, by: float, bz: float, frame: int):
    """Update text content, colour, position, and visibility for this frame."""
    mat = get_or_create_tl_material(state)

    # ── State text ────────────────────────────────────────────────────────
    state_obj.data.body = state.upper()
    # Position above the asset
    state_obj.location = (bx, by, bz + 4.5)
    state_obj.keyframe_insert(data_path="location", frame=frame)

    if not state_obj.data.materials:
        state_obj.data.materials.append(mat)
    else:
        state_obj.data.materials[0] = mat

    state_obj.hide_viewport = False
    state_obj.hide_render   = False
    state_obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    state_obj.keyframe_insert(data_path="hide_render",   frame=frame)

    # Hide on next frame (will be re-shown if detection continues)
    state_obj.hide_viewport = True
    state_obj.hide_render   = True
    state_obj.keyframe_insert(data_path="hide_viewport", frame=frame + 1)
    state_obj.keyframe_insert(data_path="hide_render",   frame=frame + 1)

    # Force CONSTANT interpolation so text snaps rather than fades
    if state_obj.animation_data and state_obj.animation_data.action:
        for fc in state_obj.animation_data.action.fcurves:
            if "hide" in fc.data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'CONSTANT'

    # ── Arrow text ────────────────────────────────────────────────────────
    arrow_char = _ARROW_CHAR.get(arrow or "none", "")
    arrow_obj.data.body = arrow_char
    arrow_obj.location  = (bx, by, bz + 3.5)
    arrow_obj.keyframe_insert(data_path="location", frame=frame)

    if not arrow_obj.data.materials:
        arrow_obj.data.materials.append(mat)
    else:
        arrow_obj.data.materials[0] = mat

    show = bool(arrow_char)
    arrow_obj.hide_viewport = not show
    arrow_obj.hide_render   = not show
    arrow_obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    arrow_obj.keyframe_insert(data_path="hide_render",   frame=frame)

    arrow_obj.hide_viewport = True
    arrow_obj.hide_render   = True
    arrow_obj.keyframe_insert(data_path="hide_viewport", frame=frame + 1)
    arrow_obj.keyframe_insert(data_path="hide_render",   frame=frame + 1)

    if arrow_obj.animation_data and arrow_obj.animation_data.action:
        for fc in arrow_obj.animation_data.action.fcurves:
            if "hide" in fc.data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'CONSTANT'

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    print("Starting Scene Generation...")
    clear_scene()

    if not os.path.exists(JSON_FILE_PATH):
        raise FileNotFoundError(f"Cannot find JSON file: {JSON_FILE_PATH}")
    
    with open(JSON_FILE_PATH, 'r') as f:
        data = json.load(f)

    frames_data = data.get("frames", [])
    if not frames_data:
        print("[WARN] No frames found in JSON.")
        return

    # Camera Setup
    cam_intrinsics = data.get("camera_intrinsics", {})
    fx = cam_intrinsics.get("fx", 1594.7)
    img_w = data.get("width", 1920)
    img_h = data.get("height", 1080)
    
    bpy.ops.object.camera_add(location=(0, 0, 1.5), rotation=(math.radians(90), 0, 0))
    cam = bpy.context.active_object
    cam.name = "MainCamera"
    bpy.context.scene.camera = cam
    
    fov = 2 * math.atan(img_w / (2 * fx))
    cam.data.angle = fov
    bpy.context.scene.render.resolution_x = img_w
    bpy.context.scene.render.resolution_y = img_h

    # Environment Setup
    bpy.ops.object.light_add(type='SUN', rotation=(math.radians(45), math.radians(45), 0))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "GroundPlane"
    mat = bpy.data.materials.new(name="Asphalt")
    mat.use_nodes = True
    if "Principled BSDF" in mat.node_tree.nodes:
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.1, 0.1, 0.1, 1)
    ground.data.materials.append(mat)

    # Frame Range Setup
    start_f = frames_data[0]["frame_idx"]
    end_f = frames_data[-1]["frame_idx"]
    
    if RENDER_PREVIEW_ONLY:
        end_f = min(start_f + 600, end_f)
        print(f"PREVIEW MODE: Only rendering frames {start_f} to {end_f}.")

    bpy.context.scene.frame_start = start_f
    bpy.context.scene.frame_end = end_f
    bpy.context.scene.render.fps = int(data.get("fps", 30))

    spawned_objects = {} 
    last_seen_frames = {}

    print("Baking tracking data into 3D space...")
    for frame_info in frames_data:
        frame_idx = frame_info["frame_idx"]
        if frame_idx > end_f:
            break # Stop processing if we hit our preview limit

        detections = frame_info.get("detections", [])
        current_frame_tracks = set()

        for det in detections:
            track_id = det["track_id"]
            subclass = det["subclass"]
            pos_3d = det["position_3d"]
            yaw_rad = det.get("yaw_rad", 0.0)
            
            current_frame_tracks.add(track_id)

            b_x = pos_3d[0]
            b_y = pos_3d[2]   
            b_z = 0.0         

            if track_id not in spawned_objects:
                asset_path = ASSET_MAPPING.get(subclass, ASSET_MAPPING.get("sedan"))
                root_obj = load_asset(asset_path, track_id)
                spawned_objects[track_id] = root_obj
                set_visibility_keyframe(root_obj, frame_idx - 1, is_visible=False)

            root_obj = spawned_objects[track_id]

            if last_seen_frames.get(track_id, -1) < frame_idx - 1:
                set_visibility_keyframe(root_obj, frame_idx, is_visible=True)

            root_obj.location = (b_x, b_y, b_z)
            root_obj.keyframe_insert(data_path="location", frame=frame_idx)
            root_obj.rotation_euler = (0, 0, -yaw_rad + ASSET_YAW_OFFSET)
            root_obj.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
            last_seen_frames[track_id] = frame_idx

            # Traffic light: show state + arrow as text above the asset
            if subclass == "traffic_light":
                tl_info  = det.get("traffic_light_info", {})
                tl_state = tl_info.get("state", "unknown")
                tl_arrow = tl_info.get("arrow", "none")
                ind_key  = f"tl_ind_{track_id}"
                if ind_key not in spawned_objects:
                    col = bpy.context.scene.collection
                    state_txt, arrow_txt = spawn_tl_text(track_id, col)
                    spawned_objects[ind_key] = (state_txt, arrow_txt)
                state_txt, arrow_txt = spawned_objects[ind_key]
                keyframe_tl_text(state_txt, arrow_txt, tl_state, tl_arrow,
                                 b_x, b_y, b_z, frame_idx)

        for track_id, root_obj in spawned_objects.items():
            if track_id not in current_frame_tracks and last_seen_frames.get(track_id) == frame_idx - 1:
                set_visibility_keyframe(root_obj, frame_idx, is_visible=False)

    # Rendering Setup
    os.makedirs(OUTPUT_RENDER_DIR, exist_ok=True)
    bpy.context.scene.render.filepath = os.path.join(OUTPUT_RENDER_DIR, "frame_")
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    print(f"\n==========================================")
    print(f"STARTING EXPORT TO PNG: {OUTPUT_RENDER_DIR}")
    print(f"==========================================\n")
    
    # THIS LINE FORCES THE SCRIPT TO RENDER THE IMAGES
    bpy.ops.render.render(animation=True)
    
    print("\n[SUCCESS] Rendering Complete!")

if __name__ == "__main__":
    main()