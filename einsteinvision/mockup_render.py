"""EinsteinVision — style mockup renderer (front chase view, single frame).

Renders one detection frame in one of three visual styles so we can pick a
look before rewriting the full-sequence renderer.

    blender -b --factory-startup --python einsteinvision/mockup_render.py -- \
        --json phase2_output/scene1/detections.json --frame 2131 \
        --assets P3Data/Assets --style A --out mockups/style_A.png

Styles:
    A  "FSD Dark"      — Tesla-like: dark world, clay vehicles, glowing lanes, EEVEE
    B  "Light Studio"  — bright clay product render, class-tinted vehicles, EEVEE
    C  "Cinematic"     — Cycles GPU, Nishita sky, asphalt, original asset materials

Geometry notes (why this differs from blender_render.py):
  * detections.json position_xyz_m is ~30x compressed (bad metric-depth scale).
    Here depth is re-derived from bbox width + known class width (pinhole).
  * Assets have inconsistent up/forward axes and units (0.66 .. 8937).
    ASSET_SPEC gives per-file rotation + real-world target size; each asset
    is normalised by its bounding box, not a hand-tuned scale constant.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# ── Camera intrinsics (front camera, from lane_report_style.json meta) ────────
FX, CX = 1594.7, 654.3

# class → (real width m for depth-from-width, asset file)
CLASS_INFO = {
    "car":        (1.85, None),     # resolved by sub_class
    "truck":      (2.50, "Vehicles/Truck.blend"),
    "bus":        (2.55, "Vehicles/Truck.blend"),
    "person":     (0.55, "Pedestrain.blend"),
    "bicycle":    (0.60, "Vehicles/Bicycle.blend"),
    "motorcycle": (0.80, "Vehicles/Motorcycle.blend"),
    "stop sign":  (0.75, "StopSign.blend"),
    "traffic light": (0.35, "TrafficSignal.blend"),
}
SUBCLASS_ASSET = {
    "sedan": "Vehicles/SedanAndHatchback.blend",
    "hatchback": "Vehicles/SedanAndHatchback.blend",
    "suv": "Vehicles/SUV.blend",
    "pickup": "Vehicles/PickupTruck.blend",
    "pickup_truck": "Vehicles/PickupTruck.blend",
    "truck": "Vehicles/Truck.blend",
}
SKIP = {"airplane", "boat", "train", "clock", "cell phone", "kite", "bench"}

# file → (rotation to Z-up/Y-forward as euler XYZ deg, size axis, target m)
#   size axis: 1 = length along Y, 2 = height along Z
ASSET_SPEC = {
    "Vehicles/SedanAndHatchback.blend": ((0, 0, 0),   1, 4.7),
    "Vehicles/SUV.blend":               ((0, 0, 0),   1, 4.8),
    "Vehicles/Truck.blend":             ((0, 0, 0),   1, 9.5),
    "Vehicles/PickupTruck.blend":       ((0, 0, 90),  1, 5.6),
    "Vehicles/Motorcycle.blend":        ((90, 0, 90), 1, 2.1),
    "Vehicles/Bicycle.blend":           ((90, 0, 0),  1, 1.75),
    "Pedestrain.blend":                 ((90, 0, 0),  2, 1.75),
    "StopSign.blend":                   ((90, 0, 0),  2, 2.6),
    "TrafficSignal.blend":              ((90, 0, 0),  2, 1.2),
    "TrafficConeAndCylinder.blend":     ((90, 0, 0),  2, 0.7),
    "Dustbin.blend":                    ((90, 0, 0),  2, 1.0),
}

LANE_W = 3.7
# Road layout for scene1 frame 2131 (multi-lane highway, ego lane index 0).
LANE_CENTERS = [LANE_W * k for k in range(-5, 2)]      # -18.5 .. 3.7
ROAD_LEFT = LANE_CENTERS[0] - LANE_W / 2               # -20.35
ROAD_RIGHT = LANE_CENTERS[-1] + LANE_W / 2             # 5.55


# ═════════════════════════════════════════════════════════════════════════════
# Detection → world layout
# ═════════════════════════════════════════════════════════════════════════════
def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def layout_objects(frame: dict, max_depth: float = 95.0, snap: float = 0.7):
    objs = [o for o in frame["objects"] if o["class_name"] not in SKIP]
    objs.sort(key=lambda o: -o.get("confidence", 0.5))
    kept = []
    for o in objs:                                   # NMS across classes
        if all(_iou(o["bbox_2d"], k["bbox_2d"]) < 0.6 for k in kept):
            kept.append(o)

    placed = []
    for o in kept:
        cls = o["class_name"]
        width, asset = CLASS_INFO.get(cls, (1.85, None))
        if cls == "car":
            asset = SUBCLASS_ASSET.get(o.get("sub_class") or "sedan",
                                       "Vehicles/SedanAndHatchback.blend")
        if asset is None:
            continue
        x1, y1, x2, y2 = o["bbox_2d"]
        z = FX * width / max(x2 - x1, 1)
        if z > max_depth:
            continue
        x = ((x1 + x2) / 2 - CX) * z / FX
        if asset.startswith("Vehicles/"):              # soft snap to lane centre
            lc = min(LANE_CENTERS, key=lambda c: abs(c - x))
            x = x + snap * (lc - x)
        placed.append(dict(cls=cls, sub=o.get("sub_class"), asset=asset,
                           x=x, y=z, yaw=float(o.get("orientation_yaw_rad") or 0.0),
                           moving=o.get("motion_state") == "moving",
                           intent=o.get("intent") or {}))
    return placed


# ═════════════════════════════════════════════════════════════════════════════
# Blender helpers
# ═════════════════════════════════════════════════════════════════════════════
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def mat_principled(name, color, rough=0.5, metal=0.0, emit=None, emit_strength=0.0,
                   coat=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if coat:
        b.inputs["Coat Weight"].default_value = coat
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1.0)
        b.inputs["Emission Strength"].default_value = emit_strength
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        try:
            m.surface_render_method = "BLENDED"
        except AttributeError:
            m.blend_method = "BLEND"
    return m


def mesh_obj(name, verts, faces, mat, col):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    if mat:
        me.materials.append(mat)
    col.objects.link(ob)
    return ob


def quad_strips(rects, z=0.0):
    """rects: list of (x0, x1, y0, y1) → verts, faces (one mesh)."""
    v, f = [], []
    for x0, x1, y0, y1 in rects:
        i = len(v)
        v += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        f.append((i, i + 1, i + 2, i + 3))
    return v, f


_TEMPLATES: dict[str, bpy.types.Collection] = {}


def asset_template(assets_dir: Path, rel: str, override=None, fill=None):
    """Load .blend once into an unlinked collection, normalised to metres,
    Z-up, Y-forward, base on z=0, centred in XY."""
    key = rel
    if key in _TEMPLATES:
        return _TEMPLATES[key]
    path = assets_dir / rel
    with bpy.data.libraries.load(str(path), link=False) as (src, dst):
        dst.objects = list(src.objects)
    meshes = [o for o in dst.objects
              if o is not None and o.type == "MESH" and not o.name.startswith("Cube")]
    rot_deg, axis, target = ASSET_SPEC[rel]
    R = (Matrix.Rotation(math.radians(rot_deg[2]), 4, "Z")
         @ Matrix.Rotation(math.radians(rot_deg[1]), 4, "Y")
         @ Matrix.Rotation(math.radians(rot_deg[0]), 4, "X"))
    pts = [R @ o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    mn = Vector([min(p[i] for p in pts) for i in range(3)])
    mx = Vector([max(p[i] for p in pts) for i in range(3)])
    s = target / max(mx[axis] - mn[axis], 1e-9)
    T = Matrix.Translation((-(mn.x + mx.x) / 2, -(mn.y + mx.y) / 2, -mn.z))
    N = Matrix.Scale(s, 4) @ T @ R

    col = bpy.data.collections.new(f"TPL_{Path(rel).stem}")
    for o in meshes:
        o.parent = None
        o.matrix_world = N @ o.matrix_world
        col.objects.link(o)
        if override is not None:
            o.data.materials.clear()
            o.data.materials.append(override)
        elif fill is not None and not any(o.data.materials):
            o.data.materials.clear()
            o.data.materials.append(fill)
    _TEMPLATES[key] = col
    return col


def instance(col_tpl, name, x, y, yaw, scene_col, heading_flip=True):
    e = bpy.data.objects.new(name, None)
    e.instance_type = "COLLECTION"
    e.instance_collection = col_tpl
    e.location = (x, y, 0.0)
    # Vehicles drive along +Y; assets face -Y by convention → flip 180°.
    e.rotation_euler = (0, 0, (math.pi if heading_flip else 0.0) - yaw)
    scene_col.objects.link(e)
    return e


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


# ═════════════════════════════════════════════════════════════════════════════
# World building
# ═════════════════════════════════════════════════════════════════════════════
def build_road(col, road_mat, shoulder_mat, paint_white, paint_yellow, y0=-40, y1=320):
    # shoulder / ground
    v, f = quad_strips([(-400, 400, -400, 800)], z=-0.02)
    mesh_obj("Ground", v, f, shoulder_mat, col)
    v, f = quad_strips([(ROAD_LEFT - 1.2, ROAD_RIGHT + 1.2, y0, y1)], z=0.0)
    mesh_obj("Road", v, f, road_mat, col)

    lw, dash, period = 0.15, 3.0, 12.0
    dashes = []
    for k in range(1, len(LANE_CENTERS)):
        xl = LANE_CENTERS[k] - LANE_W / 2
        y = y0
        while y < y1:
            dashes.append((xl - lw / 2, xl + lw / 2, y, y + dash))
            y += period
    v, f = quad_strips(dashes, z=0.012)
    mesh_obj("LaneDashed", v, f, paint_white, col)

    v, f = quad_strips([(ROAD_RIGHT - lw / 2, ROAD_RIGHT + lw / 2, y0, y1)], z=0.012)
    mesh_obj("EdgeRight", v, f, paint_white, col)
    v, f = quad_strips([(ROAD_LEFT - 0.12 - lw / 2, ROAD_LEFT - 0.12 + lw / 2, y0, y1),
                        (ROAD_LEFT + 0.12 - lw / 2, ROAD_LEFT + 0.12 + lw / 2, y0, y1)], z=0.012)
    mesh_obj("DoubleYellow", v, f, paint_yellow, col)


def build_path_and_arrows(col, placed, path_mat, arrow_mat):
    # Planned ego path (Tesla-style translucent ribbon in ego lane)
    v, f = quad_strips([(-0.9, 0.9, 2.5, 38.0)], z=0.02)
    mesh_obj("EgoPath", v, f, path_mat, col)
    # Motion chevrons in front of moving vehicles (Phase 3 direction cue)
    for i, p in enumerate(placed):
        if not p["moving"] or not p["asset"].startswith("Vehicles/"):
            continue
        L = 5.5 if "Truck" not in p["asset"] else 6.5
        cx, cy = p["x"], p["y"] + L
        verts = [(-0.7, 0.0, 0.03), (0.0, 1.1, 0.03), (0.7, 0.0, 0.03),
                 (0.0, 0.45, 0.03)]
        ob = mesh_obj(f"Chevron_{i}", verts, [(0, 3, 1), (3, 2, 1)], arrow_mat, col)
        ob.location = (cx, cy, 0)
        ob.rotation_euler = (0, 0, -p["yaw"])


def setup_camera(scene, col):
    cam_d = bpy.data.cameras.new("ChaseCam")
    cam_d.lens = 26
    cam_d.clip_end = 600
    cam = bpy.data.objects.new("ChaseCam", cam_d)
    col.objects.link(cam)
    cam.location = (0.0, -11.5, 5.2)
    look_at(cam, (0.0, 24.0, 0.0))
    scene.camera = cam
    return cam


def world_color(scene, color, strength=1.0):
    w = bpy.data.worlds.new("World")
    scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = strength
    return w


def world_sky(scene, sun_elev_deg=22.0, sun_rot_deg=150.0, strength=0.35):
    w = bpy.data.worlds.new("World")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(sun_elev_deg)
    sky.sun_rotation = math.radians(sun_rot_deg)
    sky.air_density = 1.2
    sky.dust_density = 2.0
    sky.sun_disc = False
    nt.links.new(sky.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = strength


def add_sun(col, energy, angle_deg, rot=(50, 0, 150), color=(1, 1, 1)):
    d = bpy.data.lights.new("Sun", "SUN")
    d.energy = energy
    d.angle = math.radians(angle_deg)
    d.color = color
    ob = bpy.data.objects.new("Sun", d)
    ob.rotation_euler = tuple(math.radians(a) for a in rot)
    col.objects.link(ob)
    return ob


def add_area(col, energy, size, loc, rot=(0, 0, 0), color=(1, 1, 1)):
    d = bpy.data.lights.new("Area", "AREA")
    d.energy = energy
    d.size = size
    d.color = color
    ob = bpy.data.objects.new("Area", d)
    ob.location = loc
    ob.rotation_euler = tuple(math.radians(a) for a in rot)
    col.objects.link(ob)


def color_mgmt(scene, look="AgX - Medium High Contrast", exposure=0.0):
    vs = scene.view_settings
    try:
        vs.view_transform = "AgX"
        vs.look = look
    except TypeError:
        vs.view_transform = "Filmic"
    vs.exposure = exposure


def eevee(scene, samples=64):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    ee = scene.eevee
    ee.taa_render_samples = samples
    for attr, val in (("use_raytracing", True), ("use_shadows", True),
                      ("use_gtao", True), ("gtao_distance", 1.5)):
        try:
            setattr(ee, attr, val)
        except (AttributeError, TypeError):
            pass


def cycles_gpu(scene, samples=128):
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == dev_type]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == dev_type
                scene.cycles.device = "GPU"
                print(f"[mockup] Cycles on {dev_type}: {[d.name for d in gpus]}")
                break
        except TypeError:
            continue
    else:
        print("[mockup] no GPU found — Cycles CPU")
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPTIX" if scene.cycles.device == "GPU" else "OPENIMAGEDENOISE"
    except TypeError:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Styles
# ═════════════════════════════════════════════════════════════════════════════
def style_A(scene, col, assets, placed):
    """FSD Dark — Tesla-inspired dark UI."""
    eevee(scene, 64)
    color_mgmt(scene, "AgX - Base Contrast", 0.2)
    world_color(scene, (0.010, 0.012, 0.016), 1.0)
    road = mat_principled("Road", (0.028, 0.030, 0.034), rough=0.55)
    ground = mat_principled("Ground", (0.012, 0.013, 0.016), rough=0.9)
    white = mat_principled("PaintW", (0.8, 0.8, 0.8), emit=(0.85, 0.88, 0.95), emit_strength=1.6)
    yellow = mat_principled("PaintY", (0.9, 0.6, 0.1), emit=(1.0, 0.62, 0.12), emit_strength=1.4)
    build_road(scene.collection, road, ground, white, yellow)
    path = mat_principled("Path", (0.1, 0.35, 1.0), emit=(0.15, 0.45, 1.0),
                          emit_strength=1.2, alpha=0.35)
    arrow = mat_principled("Arrow", (0.2, 0.5, 1.0), emit=(0.25, 0.55, 1.0), emit_strength=3.0)
    build_path_and_arrows(scene.collection, placed, path, arrow)

    clay = mat_principled("Clay", (0.30, 0.31, 0.33), rough=0.35, coat=0.3)
    ego_m = mat_principled("Ego", (0.85, 0.86, 0.88), rough=0.2, coat=0.8)
    add_area(scene.collection, 6000, 60, (0, 25, 40), color=(0.85, 0.9, 1.0))
    add_sun(scene.collection, 1.2, 8, rot=(40, 0, 160), color=(0.8, 0.85, 1.0))
    return clay, ego_m, None


def style_B(scene, col, assets, placed):
    """Light Studio — bright clay product render, class-tinted."""
    eevee(scene, 64)
    color_mgmt(scene, "AgX - Medium High Contrast", 0.0)
    world_color(scene, (0.70, 0.74, 0.80), 0.8)
    road = mat_principled("Road", (0.16, 0.17, 0.19), rough=0.8)
    ground = mat_principled("Ground", (0.70, 0.72, 0.75), rough=0.9)
    white = mat_principled("PaintW", (0.95, 0.95, 0.95), rough=0.6)
    yellow = mat_principled("PaintY", (0.95, 0.68, 0.12), rough=0.6)
    build_road(scene.collection, road, ground, white, yellow)
    path = mat_principled("Path", (0.15, 0.45, 1.0), emit=(0.2, 0.5, 1.0),
                          emit_strength=0.4, alpha=0.45)
    arrow = mat_principled("Arrow", (0.1, 0.4, 1.0), emit=(0.15, 0.45, 1.0), emit_strength=1.0)
    build_path_and_arrows(scene.collection, placed, path, arrow)
    add_sun(scene.collection, 3.5, 12, rot=(45, 0, 145))
    ego_m = mat_principled("Ego", (0.93, 0.93, 0.95), rough=0.25, coat=0.6)
    per_asset = {
        "Vehicles/SedanAndHatchback.blend": mat_principled("Sedan", (0.55, 0.68, 0.85), 0.35, coat=0.4),
        "Vehicles/SUV.blend":               mat_principled("SUV",   (0.55, 0.80, 0.62), 0.35, coat=0.4),
        "Vehicles/PickupTruck.blend":       mat_principled("Pickup",(0.88, 0.70, 0.45), 0.35, coat=0.4),
        "Vehicles/Truck.blend":             mat_principled("Truck", (0.92, 0.58, 0.40), 0.40, coat=0.3),
    }
    return None, ego_m, per_asset


def style_C(scene, col, assets, placed):
    """Cinematic — Cycles, sky, asphalt, original materials."""
    cycles_gpu(scene, 128)
    color_mgmt(scene, "AgX - Medium High Contrast", 0.0)
    world_sky(scene, 24, 200, 0.25)

    # procedural asphalt
    road = bpy.data.materials.new("Asphalt")
    road.use_nodes = True
    nt = road.node_tree
    b = nt.nodes["Principled BSDF"]
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 180.0
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.035, 0.035, 0.037, 1)
    ramp.color_ramp.elements[1].color = (0.09, 0.09, 0.092, 1)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Roughness"].default_value = 0.85

    grass = bpy.data.materials.new("Grass")
    grass.use_nodes = True
    g = grass.node_tree
    gb = g.nodes["Principled BSDF"]
    gn = g.nodes.new("ShaderNodeTexNoise")
    gn.inputs["Scale"].default_value = 6.0
    gr = g.nodes.new("ShaderNodeValToRGB")
    gr.color_ramp.elements[0].color = (0.035, 0.06, 0.02, 1)
    gr.color_ramp.elements[1].color = (0.10, 0.13, 0.05, 1)
    g.links.new(gn.outputs["Fac"], gr.inputs["Fac"])
    g.links.new(gr.outputs["Color"], gb.inputs["Base Color"])
    gb.inputs["Roughness"].default_value = 0.95

    white = mat_principled("PaintW", (0.82, 0.82, 0.8), rough=0.55)
    yellow = mat_principled("PaintY", (0.85, 0.6, 0.08), rough=0.55)
    build_road(scene.collection, road, grass, white, yellow)
    add_sun(scene.collection, 3.2, 1.5, rot=(62, 0, 200), color=(1.0, 0.95, 0.88))
    fill = mat_principled("CarPaint", (0.25, 0.27, 0.3), rough=0.25, metal=0.6, coat=1.0)
    ego_m = mat_principled("Ego", (0.8, 0.02, 0.02), rough=0.2, metal=0.5, coat=1.0)
    return None, ego_m, {"__fill__": fill}


STYLES = {"A": style_A, "B": style_B, "C": style_C}


# ═════════════════════════════════════════════════════════════════════════════
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--frame", type=int, default=2131)
    ap.add_argument("--assets", default="P3Data/Assets")
    ap.add_argument("--style", choices=list(STYLES), default="A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, nargs=2, default=(1920, 1080))
    ap.add_argument("--no-flip", action="store_true", help="don't rotate assets 180°")
    a = ap.parse_args(argv)

    data = json.loads(Path(a.json).read_text())
    frame = next(f for f in data["frames"] if f["frame_index"] == a.frame)
    placed = layout_objects(frame)
    print(f"[mockup] frame {a.frame}: {len(placed)} objects after NMS")
    for p in placed:
        print(f"   {p['cls']:8s} {str(p['sub']):10s} x={p['x']:6.1f} y={p['y']:6.1f} yaw={p['yaw']:+.2f}")

    scene = reset_scene()
    scene.render.resolution_x, scene.render.resolution_y = a.res
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    col = scene.collection
    assets = Path(a.assets)

    override, ego_mat, per_asset = STYLES[a.style](scene, col, assets, placed)
    fill = (per_asset or {}).get("__fill__")

    # Ego vehicle (separate template so it can take its own material)
    ego_tpl = asset_template_copy(assets, "Vehicles/SedanAndHatchback.blend", ego_mat)
    instance(ego_tpl, "Ego", 0.0, 0.0, 0.0, col, heading_flip=not a.no_flip)

    for i, p in enumerate(placed):
        ov = override
        if per_asset and p["asset"] in per_asset:
            ov = per_asset[p["asset"]]
        tpl = asset_template(assets, p["asset"], override=ov, fill=fill)
        instance(tpl, f"Obj_{i}_{p['cls']}", p["x"], p["y"], p["yaw"], col,
                 heading_flip=not a.no_flip)

    setup_camera(scene, col)
    scene.render.filepath = str(Path(a.out).resolve())
    bpy.ops.render.render(write_still=True)
    print(f"[mockup] wrote {scene.render.filepath}")


def asset_template_copy(assets_dir, rel, material):
    """Independent (non-shared) template, used for the ego car."""
    saved = _TEMPLATES.pop(rel, None)
    tpl = asset_template(assets_dir, rel, override=material)
    tpl.name = "TPL_Ego"
    _TEMPLATES.pop(rel, None)
    if saved is not None:
        _TEMPLATES[rel] = saved
    return tpl


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
