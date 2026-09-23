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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import road_model as rm   # noqa: E402
import infra              # noqa: E402

JUNCTION = None      # intersection ahead (infra.find_junction) or None
DASH_PHASE = 0.0     # ego distance travelled (m) → world-fixed dashes / barrier joints scroll

ROAD = None          # derived road (road_model.derive) for the current frame, or None
SIGNALS = []         # traffic lights for the current frame (tl_state.py)

# ── Camera (front, undistorted 1280x960) ─────────────────────────────────────
FX, CX, FY, CY = 1594.7, 654.3, 1607.7, 413.4
IMG_W, IMG_H = 1280, 960
CAM_H = 1.45          # m, windshield camera height
V_HORIZON = 444.0     # px — median over 2485 scene1 cars (IQR 438–452); ≈1.1° pitch up

# Real-world vehicle dims (w, l, h) m. Assets are scaled PER AXIS to these,
# so every model has correct proportions regardless of how it was modelled.
VEH_DIMS = {
    "Vehicles/SedanAndHatchback.blend": (1.85, 4.70, 1.45),
    "Vehicles/SUV.blend":               (1.95, 4.70, 1.80),
    "Vehicles/PickupTruck.blend":       (2.00, 5.60, 1.90),
    "Vehicles/Truck.blend":             (2.55, 8.50, 3.60),
    "Vehicles/Motorcycle.blend":        (0.80, 2.10, 1.20),
    "Vehicles/Bicycle.blend":           (0.60, 1.75, 1.10),
}
MIN_GAP = 0.8         # m, bumper-to-bumper minimum after overlap resolution

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
    "Vehicles/PickupTruck.blend":       ((90, 0, 90), 1, 5.6),   # model lies on its side: +Y is up
    "Vehicles/Motorcycle.blend":        ((90, 0, 90), 1, 2.1),
    "Vehicles/Bicycle.blend":           ((90, 0, 0),  1, 1.75),
    "Pedestrain.blend":                 ((90, 0, 0),  2, 1.75),
    "StopSign.blend":                   ((90, 0, 0),  2, 2.6),
    "TrafficSignal.blend":              ((90, 0, 0),  2, 1.2),
    "TrafficConeAndCylinder.blend":     ((90, 0, 0),  2, 0.7),
    "Dustbin.blend":                    ((90, 0, 0),  2, 1.0),
}

# Assets whose front faces -Y (Blender "front" convention) → need 180° so they drive +Y.
# Measured from height profiles (asset_front.py, 2026-09-23): sedan & pickup face -Y;
# SUV (jeep), truck, motorcycle face +Y — matches user: "car forward, truck & jeep opposite".
ASSET_FLIP = {"Vehicles/SedanAndHatchback.blend", "Vehicles/PickupTruck.blend"}

EGO_SPEED = 25.0       # m/s assumed ego speed (scene1 highway) for absolute heading
MAX_YAW = math.radians(8)      # highway lane change ≈ 5–8°; more = depth noise
VX_DEADBAND = 0.6              # m/s lateral; below → lane-aligned

LANE_W = 3.7
# Road layout for scene1 frame 2131 (multi-lane highway, ego lane index 0).
LANE_CENTERS = [LANE_W * k for k in range(-5, 2)]      # -18.5 .. 3.7
ROAD_LEFT = LANE_CENTERS[0] - LANE_W / 2               # -20.35
ROAD_RIGHT = LANE_CENTERS[-1] + LANE_W / 2             # 5.55
BARRIER_X = ROAD_LEFT - 0.9                            # median Jersey barrier
ONC_RIGHT = BARRIER_X - 0.9                            # oncoming carriageway (4 lanes)
ONC_LEFT = ONC_RIGHT - 4 * LANE_W


# ═════════════════════════════════════════════════════════════════════════════
# Detection → world layout
# ═════════════════════════════════════════════════════════════════════════════
def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _aspect(o):
    x1, y1, x2, y2 = o["bbox_2d"]
    return (y2 - y1) / max(x2 - x1, 1)


def asset_for(o):
    cls = o["class_name"]
    if cls in ("truck", "bus") and _aspect(o) < 0.95:
        return "Vehicles/PickupTruck.blend"          # rear view too flat for a box truck
    if cls == "car":
        return SUBCLASS_ASSET.get(o.get("sub_class") or "sedan", "Vehicles/SedanAndHatchback.blend")
    return CLASS_INFO.get(cls, (1.85, None))[1]


def obj_dims(o):
    a = asset_for(o)
    if a in VEH_DIMS:
        return VEH_DIMS[a]
    w = CLASS_INFO.get(o["class_name"], (0.6, None))[0]
    return (w, w, 1.7)


def _bottom_occluded(o, others):
    x1, _, x2, y2 = o["bbox_2d"]
    for b in others:
        if b is o:
            continue
        bx1, by1, bx2, by2 = b["bbox_2d"]
        ov = min(x2, bx2) - max(x1, bx1)
        if ov > 0.3 * (x2 - x1) and by2 > y2 + 3 and by1 < y2:   # nearer box covers our bottom
            return True
    return False


def depth_of(o, others=()):
    """Distance (m) to the nearest face of the object, fused from
    ground contact (bbox bottom on road plane) and side-aware width."""
    x1, y1, x2, y2 = o["bbox_2d"]
    W, L, _ = obj_dims(o)
    t = abs((x1 + x2) / 2 - CX) / FX                 # tan(viewing angle)
    zw = FX * (W + L * t / (1 + t)) / max(x2 - x1, 1)  # rear face + visible flank
    if x1 <= 2 or x2 >= IMG_W - 2:
        zw = None                                     # truncated at image side
    zg = None
    if y2 < IMG_H - 4 and y2 - V_HORIZON > 12 and not _bottom_occluded(o, others):
        zg = FY * CAM_H / (y2 - V_HORIZON)
    if zg is None and zw is None:
        return FX * W / max(x2 - x1, 1)
    if zg is None:
        return zw
    if zw is None:
        return zg
    sg = zg * zg / (FY * CAM_H) * 2.0 + 0.03 * zg      # 2 px bottom jitter
    sw = 0.08 * zw + zw * 2.0 / max(x2 - x1, 1)        # model + 2 px width jitter
    if abs(zg - zw) > 0.4 * min(zg, zw):              # disagree → trust the better-conditioned cue
        return zg if y2 - V_HORIZON >= 40 else zw
    return (zg / sg**2 + zw / sw**2) / (1 / sg**2 + 1 / sw**2)


def _depth_xy(o, others=()):
    """(x, y) of the object CENTRE on the ground plane."""
    x1, _, x2, _ = o["bbox_2d"]
    z = depth_of(o, others)
    L = obj_dims(o)[1]
    return ((x1 + x2) / 2 - CX) * z / FX, z + L / 2


def resolve_overlaps(placed, ego_len=4.7, ego_w=1.85, iters=8):
    """Push the farther vehicle back until footprints are ≥ MIN_GAP apart."""
    for _ in range(iters):
        moved = False
        for p in placed:                                         # ego clearance
            W, L = p["dims"][0], p["dims"][1]
            if abs(p["x"]) < (W + ego_w) / 2 + 0.3:
                ymin = ego_len / 2 + MIN_GAP + L / 2
                if p["y"] < ymin:
                    p["y"], moved = ymin, True
        order = sorted(placed, key=lambda p: p["y"])
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                if abs(a["x"] - b["x"]) >= (a["dims"][0] + b["dims"][0]) / 2 + 0.3:
                    continue
                need = (a["dims"][1] + b["dims"][1]) / 2 + MIN_GAP
                if b["y"] - a["y"] < need:
                    b["y"], moved = a["y"] + need, True
        if not moved:
            break
    return placed


def motion_heading(frames, fi, oid, fps, win=18):
    """Yaw (rad, CCW from +Y) of a track from its motion over ±win frames.

    Relative velocity (vx, vy) comes from a least-squares fit of the track's
    bbox-derived positions; absolute forward speed = EGO_SPEED + vy (same
    carriageway traffic). Returns (yaw, rel_speed) or (None, 0)."""
    ts, xs, ys = [], [], []
    for f in frames[max(0, fi - win): fi + win + 1]:
        for o in f["objects"]:
            if str(o["object_id"]) == oid:
                x, y = _depth_xy(o, f["objects"])
                ts.append(f["frame_index"] / fps); xs.append(x); ys.append(y)
    if len(ts) < 5:
        return None, 0.0
    tm = sum(ts) / len(ts)
    den = sum((t - tm) ** 2 for t in ts) or 1e-9
    vx = sum((t - tm) * (x - sum(xs) / len(xs)) for t, x in zip(ts, xs)) / den
    vy = sum((t - tm) * (y - sum(ys) / len(ys)) for t, y in zip(ts, ys)) / den
    fwd = EGO_SPEED + vy                          # absolute forward speed (negative = oncoming)
    # lateral error of x = (u-cx)·Z/fx grows with Z (bbox-width depth jitter,
    # side faces entering the bbox) → distance-scaled dead-band
    db = VX_DEADBAND + 0.03 * (sum(ys) / len(ys)) + 0.05 * abs(sum(xs) / len(xs))
    vx = 0.0 if abs(vx) < db else vx - math.copysign(db, vx)
    if abs(fwd) < 1.0:                            # ~stationary: lane-aligned
        return 0.0, math.hypot(vx, vy)
    yaw = math.atan2(-vx, fwd)
    base = 0.0 if abs(yaw) <= math.pi / 2 else math.copysign(math.pi, yaw)
    cap = math.pi / 2 if abs(vx) > 5.0 and len(ts) >= 15 else MAX_YAW   # crossing traffic
    dev = max(-cap, min(cap, yaw - base))
    return base + dev, math.hypot(vx, vy)


def layout_objects(frames: list, fi: int, fps: float, max_depth: float = 110.0, snap: float = 0.35):
    frame = frames[fi]
    objs = [o for o in frame["objects"] if o["class_name"] not in SKIP]
    objs.sort(key=lambda o: -o.get("confidence", 0.5))
    kept = []
    for o in objs:                                   # NMS across classes
        dup = next((k for k in kept if _iou(o["bbox_2d"], k["bbox_2d"]) >= 0.6), None)
        if dup is None:
            kept.append(o)
        elif {dup["class_name"], o["class_name"]} == {"car", "truck"}:
            # car/truck duplicate: box shape decides (truck rear h/w ≈ 1.2+, car ≈ 0.75)
            want = "truck" if _aspect(o) >= 0.95 else "car"
            if o["class_name"] == want and dup["class_name"] != want:
                kept[kept.index(dup)] = o

    placed = []
    for o in kept:
        cls = o["class_name"]
        asset = asset_for(o)
        if asset is None or cls == "traffic light":   # signals: procedural, from tl_state.py
            continue
        if cls == "stop sign":                        # plate 0.75 m wide, on a post
            x1, y1, x2, y2 = o["bbox_2d"]
            z = FX * 0.75 / max(x2 - x1, 1)
            if z > 70:
                continue
            x = ((x1 + x2) / 2 - CX) * z / FX
            if ROAD is not None:                       # signs stand beside the carriageway
                off = rm.offset_of(ROAD, x, z)
                lo = min([l["a"] for l in ROAD["oncoming_lines"]], default=ROAD["left"])
                if lo - 1.0 < off < ROAD["right"] + 1.0:
                    edge = ROAD["right"] + 1.2 if off > (lo + ROAD["right"]) / 2 - 1.0 else lo - 1.2
                    x = rm.x_at(ROAD, edge, z)
            placed.append(dict(oid=str(o["object_id"]), cls=cls, sub="stop", asset=asset, x=x, y=z,
                               yaw=math.pi / 2 + (rm.heading_at(ROAD, z) if ROAD else 0.0),
                               bbox=list(o["bbox_2d"]), dims=(0.75, 0.1, 2.6), moving=False,
                               intent={}))
            continue
        x, z = _depth_xy(o, kept)
        if z > max_depth:
            continue
        if asset.startswith("Vehicles/"):              # gentle snap, only if close
            if ROAD is not None:
                off = rm.offset_of(ROAD, x, z)
                cs = rm.lane_centers(ROAD)
                lc = min(cs, key=lambda c: abs(c - off)) if cs else off
                if abs(lc - off) < 1.2:
                    x = x + snap * (lc - off)
            else:
                lc = min(LANE_CENTERS, key=lambda c: abs(c - x))
                if abs(lc - x) < 1.0:
                    x = x + snap * (lc - x)
        yaw, _ = motion_heading(frames, fi, str(o["object_id"]), fps)
        if yaw is None or o.get("motion_state") == "parked":
            yaw = 0.0                                 # parked / short track → lane-aligned
        if ROAD is not None:
            onc = rm.is_oncoming(ROAD, x, z)
            if onc and abs(yaw) < math.pi / 2:        # beyond median but motion unclear → oncoming
                yaw = math.pi - yaw
            elif not onc and abs(yaw) > math.pi / 2:  # our side of the median → same direction
                yaw = math.pi - yaw
            yaw += rm.heading_at(ROAD, z)             # follow road curvature
        elif x < BARRIER_X:                           # beyond the median → oncoming traffic
            yaw = math.pi - yaw
        placed.append(dict(oid=str(o["object_id"]), cls=cls, sub=o.get("sub_class"), asset=asset,
                           x=x, y=z, yaw=yaw,
                           bbox=list(o["bbox_2d"]), dims=obj_dims(o),
                           moving=o.get("motion_state") == "moving",
                           intent=o.get("intent") or {}))
    resolve_overlaps([p for p in placed if p["asset"].startswith("Vehicles/")])
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
    T = Matrix.Translation((-(mn.x + mx.x) / 2, -(mn.y + mx.y) / 2, -mn.z))
    if rel in VEH_DIMS:
        w, l, h = VEH_DIMS[rel]
        S = Matrix.Diagonal((w / (mx.x - mn.x), l / (mx.y - mn.y), h / (mx.z - mn.z), 1.0))
    else:
        S = Matrix.Scale(target / max(mx[axis] - mn[axis], 1e-9), 4)
    N = S @ T @ R

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
    col.use_fake_user = True          # survive orphans_purge in the sequence renderer
    _TEMPLATES[key] = col
    return col


def instance(col_tpl, name, x, y, yaw, scene_col, flip=False):
    """yaw: CCW heading from +Y (direction of travel). flip: model faces -Y."""
    e = bpy.data.objects.new(name, None)
    e.instance_type = "COLLECTION"
    e.instance_collection = col_tpl
    e.location = (x, y, 0.0)
    e.rotation_euler = (0, 0, yaw + (math.pi if flip else 0.0))
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
    v, f = quad_strips([(ONC_LEFT - 1.2, ROAD_RIGHT + 1.2, y0, y1)], z=0.0)
    mesh_obj("Road", v, f, road_mat, col)
    lw = 0.15
    onc = []
    for k in range(1, 4):
        xl = ONC_RIGHT - k * LANE_W
        y = y0
        while y < y1:
            onc.append((xl - lw / 2, xl + lw / 2, y, y + 3.0)); y += 12.0
    onc += [(ONC_LEFT - lw / 2, ONC_LEFT + lw / 2, y0, y1)]
    v, f = quad_strips(onc, z=0.012)
    mesh_obj("OncomingLines", v, f, paint_white, col)
    v, f = quad_strips([(ONC_RIGHT - 0.12 - lw / 2, ONC_RIGHT - 0.12 + lw / 2, y0, y1),
                        (ONC_RIGHT + 0.12 - lw / 2, ONC_RIGHT + 0.12 + lw / 2, y0, y1)], z=0.012)
    mesh_obj("OncomingYellow", v, f, paint_yellow, col)

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


def build_divider(col, body_mat, top_mat, x_c, y0=-40, y1=320, seg=6.0, gap=0.04):
    """Concrete Jersey barrier (median divider), segmented every `seg` m."""
    prof = [(-0.30, 0.0), (-0.25, 0.08), (-0.10, 0.33), (-0.075, 0.81),
            (0.075, 0.81), (0.10, 0.33), (0.25, 0.08), (0.30, 0.0)]
    v, f = [], []
    y = y0
    while y < y1:
        ya, yb = y, y + seg - gap
        i0 = len(v)
        v += [(x_c + px, ya, pz) for px, pz in prof] + [(x_c + px, yb, pz) for px, pz in prof]
        n = len(prof)
        for k in range(n - 1):
            f.append((i0 + k, i0 + k + 1, i0 + n + k + 1, i0 + n + k))
        f.append(tuple(i0 + k for k in range(n))[::-1])      # front cap
        f.append(tuple(i0 + n + k for k in range(n)))         # back cap
        y += seg
    ob = mesh_obj("MedianBarrier", v, f, body_mat, col)
    for poly in ob.data.polygons:
        poly.use_smooth = False
    if top_mat is not None:     # thin glowing cap line so it reads on dark UI
        tv, tf = quad_strips([(x_c - 0.06, x_c + 0.06, y0, y1)], z=0.815)
        mesh_obj("MedianBarrierTop", tv, tf, top_mat, col)
    return ob


def build_guardrail(col, mat, x, y0=-40, y1=320, post_every=4.0):
    """W-beam guardrail on posts along the right shoulder."""
    v, f = [], []
    for zb, zt in ((0.55, 0.62), (0.66, 0.78)):                 # two-tone beam faces
        i = len(v)
        v += [(x, y0, zb), (x, y1, zb), (x, y1, zt), (x, y0, zt)]
        f.append((i, i + 1, i + 2, i + 3))
    y = y0
    while y < y1:                                                 # posts (thin boxes)
        i = len(v)
        a, b, c, d = x + 0.02, x + 0.17, y - 0.07, y + 0.07
        for zz in (0.0, 0.75):
            v += [(a, c, zz), (b, c, zz), (b, d, zz), (a, d, zz)]
        f += [(i, i + 1, i + 5, i + 4), (i + 1, i + 2, i + 6, i + 5),
              (i + 2, i + 3, i + 7, i + 6), (i + 3, i, i + 4, i + 7), (i + 4, i + 5, i + 6, i + 7)]
        y += post_every
    return mesh_obj("Guardrail", v, f, mat, col)


# ─── curved road from road_model ────────────────────────────────────────────────
def _ribbon(road, a0, a1, y0, y1, z, step=2.0):
    """quad strip between offsets a0<a1 following the road curve."""
    v, f = [], []
    n = int((y1 - y0) / step)
    for i in range(n + 1):
        y = y0 + i * step
        v += [(rm.x_at(road, a0, y), y, z), (rm.x_at(road, a1, y), y, z)]
        if i:
            k = len(v) - 4
            f.append((k, k + 2, k + 3, k + 1))
    return v, f


def _merge(parts):
    V, F = [], []
    for v, f in parts:
        o = len(V)
        V += v
        F += [tuple(i + o for i in q) for q in f]
    return V, F


def _line_parts(road, L, y0, y1, lw=0.15):
    if JUNCTION is not None and y0 < JUNCTION["y0"] - 4.6 < y1:
        return (_line_parts_span(road, L, y0, JUNCTION["y0"] - 4.6, lw)
                + _line_parts_span(road, L, JUNCTION["y1"] + 3.6, y1, lw))
    return _line_parts_span(road, L, y0, y1, lw)


def _line_parts_span(road, L, y0, y1, lw=0.15):
    parts = []
    offs = [L["a"]]
    if L.get("double"):
        offs = [L["a"] - 0.12, L["a"] + 0.12]
    for a in offs:
        if L["style"] == "solid":
            parts.append(_ribbon(road, a - lw / 2, a + lw / 2, y0, y1, 0.012))
        else:                                              # 3 m dash / 9 m gap (US)
            y = y0 - ((y0 + DASH_PHASE) % 12.0)
            while y < y1:
                ya, yb = max(y, y0), min(y + 3.0, y1)
                if yb > ya:
                    parts.append(_ribbon(road, a - lw / 2, a + lw / 2, ya, yb, 0.012, step=1.0))
                y += 12.0
    return parts


def build_road_curved(col, road, road_mat, ground_mat, white, yellow, y0=-40.0, y1=260.0):
    v, f = quad_strips([(-600, 600, -400, 900)], z=-0.02)
    mesh_obj("Ground", v, f, ground_mat, col)
    onc = road["oncoming_lines"]
    a_lo = min([l["a"] for l in onc], default=road["left"]) - 1.2
    a_hi = road["right"] + 1.5
    v, f = _ribbon(road, a_lo, a_hi, y0, y1, 0.0)
    mesh_obj("Road", v, f, road_mat, col)
    W, Y = [], []
    for L in road["lines"] + onc:
        (Y if L["color"] == "yellow" else W).extend(_line_parts(road, L, y0, y1))
    if W:
        mesh_obj("LinesWhite", *_merge(W), white, col)
    if Y:
        mesh_obj("LinesYellow", *_merge(Y), yellow, col)


def build_divider_curved(col, road, body_mat, a, y0=-40.0, y1=260.0, seg=6.0, gap=0.04):
    prof = [(-0.30, 0.0), (-0.25, 0.08), (-0.10, 0.33), (-0.075, 0.81),
            (0.075, 0.81), (0.10, 0.33), (0.25, 0.08), (0.30, 0.0)]
    n = len(prof)
    v, f = [], []
    y = y0 - ((y0 + DASH_PHASE) % seg)
    while y < y1:
        for ya, yb in ((y, y + seg - gap),):
            i0 = len(v)
            for yy in (ya, yb):
                xc = rm.x_at(road, a, yy)
                v += [(xc + px, yy, pz) for px, pz in prof]
            for k in range(n - 1):
                f.append((i0 + k, i0 + k + 1, i0 + n + k + 1, i0 + n + k))
            f.append(tuple(i0 + k for k in range(n))[::-1])
            f.append(tuple(i0 + n + k for k in range(n)))
        y += seg
    return mesh_obj("MedianBarrier", v, f, body_mat, col)


def build_guardrail_curved(col, road, mat, a, y0=-40.0, y1=260.0, post_every=4.0):
    parts = []
    for zb, zt in ((0.55, 0.62), (0.66, 0.78)):
        vv, ff = [], []
        n = int((y1 - y0) / 2.0)
        for i in range(n + 1):
            y = y0 + i * 2.0
            x = rm.x_at(road, a, y)
            vv += [(x, y, zb), (x, y, zt)]
            if i:
                k = len(vv) - 4
                ff.append((k, k + 2, k + 3, k + 1))
        parts.append((vv, ff))
    y = y0 - ((y0 + DASH_PHASE) % post_every)
    while y < y1:
        x = rm.x_at(road, a, y)
        vv, ff = [], []
        A, B, C, D = x + 0.02, x + 0.17, y - 0.07, y + 0.07
        for zz in (0.0, 0.75):
            vv += [(A, C, zz), (B, C, zz), (B, D, zz), (A, D, zz)]
        ff = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
        parts.append((vv, ff))
        y += post_every
    return mesh_obj("Guardrail", *_merge(parts), mat, col)


# ─── traffic signals / signs ──────────────────────────────────────────────────
_LAMP = {"red": (1.0, 0.03, 0.02), "yellow": (1.0, 0.55, 0.0), "green": (0.05, 1.0, 0.35)}


def _box(cx, cy, cz, sx, sy, sz):
    v = [(cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2)
         for dz in (-1, 1) for dy in (-1, 1) for dx in (-1, 1)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v, f


def build_signals(col, signals, road):
    """Procedural signal heads (housing + 3 lamps, lit lamp emissive) on mast arms."""
    if not signals:
        return
    housing = mat_principled("SignalHousing", (0.02, 0.02, 0.022), rough=0.6)
    pole = mat_principled("SignalPole", (0.35, 0.36, 0.37), rough=0.4, metal=0.8)
    off = mat_principled("LampOff", (0.04, 0.04, 0.04), rough=0.3)
    lit = {k: mat_principled(f"Lamp_{k}", c, rough=0.2, emit=c, emit_strength=40.0) for k, c in _LAMP.items()}
    right_a = road["right"] + 2.0 if road else 7.0
    arm_parts, pole_parts, house_parts = [], [], []
    lamp_parts = {k: [] for k in ["off", "red", "yellow", "green"]}
    seen_y = {}
    for s in signals:
        x, y, z = s["x"], s["y"], max(s["z"], 4.8)
        if y < 8 or y > 140:
            continue
        key = round(y / 6.0)                  # heads at similar distance share one mast
        house_parts.append(_box(x, y, z, 0.36, 0.3, 1.05))
        for i, st in enumerate(("red", "yellow", "green")):
            on = s.get("state") == st
            lamp_parts[st if on else "off"].append(_box(x, y - 0.16, z + 0.33 - i * 0.33, 0.24, 0.02, 0.24))
        px = rm.x_at(road, right_a, y) if road else right_a
        arm_parts.append(_box((x + px) / 2, y, z + 0.6, abs(px - x) + 0.3, 0.14, 0.14))
        if key not in seen_y:
            seen_y[key] = True
            pole_parts.append(_box(px, y, (z + 0.7) / 2, 0.25, 0.25, z + 0.7))
    mesh_obj("SignalHousings", *_merge(house_parts), housing, col)
    mesh_obj("SignalArms", *_merge(arm_parts + pole_parts), pole, col)
    for k, parts in lamp_parts.items():
        if parts:
            mesh_obj(f"Lamps_{k}", *_merge(parts), off if k == "off" else lit[k], col)
    for s in signals:                          # glow onto the scene
        if s.get("state") in _LAMP and 8 <= s["y"] <= 80:
            d = bpy.data.lights.new("SigGlow", "POINT")
            d.energy = 60.0
            d.color = _LAMP[s["state"]]
            d.shadow_soft_size = 0.2
            ob = bpy.data.objects.new("SigGlow", d)
            ob.location = (s["x"], s["y"] - 0.6, max(s["z"], 4.8))
            col.objects.link(ob)


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
        ob.rotation_euler = (0, 0, p["yaw"])


def setup_camera(scene, col, cinematic=False):
    cam_d = bpy.data.cameras.new("ChaseCam")
    cam_d.lens = 32 if cinematic else 26
    cam_d.clip_end = 600
    cam = bpy.data.objects.new("ChaseCam", cam_d)
    col.objects.link(cam)
    if cinematic:
        cam.location = (0.6, -10.0, 3.4)
        look_at(cam, (-0.8, 30.0, 0.6))
        cam_d.dof.use_dof = True
        cam_d.dof.focus_distance = 25.0
        cam_d.dof.aperture_fstop = 5.6
    else:
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
    build_divider(scene.collection,
                  mat_principled("Barrier", (0.16, 0.17, 0.19), rough=0.45, coat=0.2),
                  mat_principled("BarrierTop", (0.6, 0.7, 0.8), emit=(0.55, 0.7, 0.9), emit_strength=1.2),
                  BARRIER_X)
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
    build_divider(scene.collection, mat_principled("Barrier", (0.62, 0.62, 0.6), rough=0.8), None, BARRIER_X)
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
    cycles_gpu(scene, 192)
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
    concrete = mat_principled("Concrete", (0.45, 0.44, 0.42), rough=0.85)
    steel = mat_principled("Galvanised", (0.55, 0.56, 0.58), rough=0.35, metal=0.9)
    road = infra.pbr_material("AsphaltPBR", infra.EXTRA / "textures/asphalt_02", 0.25) or road
    walk = infra.pbr_material("SidewalkPBR", infra.EXTRA / "textures/concrete_floor_02", 0.4) or \
        mat_principled("Sidewalk", (0.5, 0.5, 0.48), rough=0.8)
    if ROAD is not None:
        build_road_curved(scene.collection, ROAD, road, grass, white, yellow)
        if JUNCTION is not None:
            infra.build_junction(scene.collection, ROAD, JUNCTION, road, white, yellow)
        if ROAD["median"] != "barrier":
            infra.build_sidewalks(scene.collection, ROAD, JUNCTION, walk, concrete)
        infra.scatter_furniture(scene.collection, ROAD, JUNCTION)
        if ROAD["median"] == "barrier":
            build_divider_curved(scene.collection, ROAD, concrete, ROAD["barrier_a"])
        if ROAD["median"] == "barrier":
            build_guardrail_curved(scene.collection, ROAD, steel, ROAD["right"] + 1.0)
    else:
        build_road(scene.collection, road, grass, white, yellow)
        build_divider(scene.collection, concrete, None, BARRIER_X)
        build_guardrail(scene.collection, steel, ROAD_RIGHT + 1.0)
    build_signals(scene.collection, SIGNALS, ROAD)
    add_sun(scene.collection, 3.6, 1.2, rot=(58, 0, 215), color=(1.0, 0.90, 0.78))
    fill = mat_principled("CarPaint", (0.25, 0.27, 0.3), rough=0.25, metal=0.6, coat=1.0)
    ego_m = mat_principled("Ego", (0.8, 0.02, 0.02), rough=0.2, metal=0.5, coat=1.0)
    return None, ego_m, {"__fill__": fill}


STYLES = {"A": style_A, "B": style_B, "C": style_C}


# ═════════════════════════════════════════════════════════════════════════════
def main(argv):
    global EGO_SPEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--frame", type=int, default=2131)
    ap.add_argument("--assets", default="P3Data/Assets")
    ap.add_argument("--style", choices=list(STYLES), default="A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, nargs=2, default=(1920, 1080))
    ap.add_argument("--flip-all", action="store_true", help="invert every asset's facing (debug)")
    ap.add_argument("--ego-speed", type=float, default=EGO_SPEED)
    ap.add_argument("--road", default=None, help="road/sceneN/road_model.json (lane_bev.py)")
    ap.add_argument("--tl", default=None, help="road/sceneN/traffic_lights.json (tl_state.py)")
    a = ap.parse_args(argv)

    data = json.loads(Path(a.json).read_text())
    EGO_SPEED = a.ego_speed
    frames = data["frames"]
    fi = next(i for i, f in enumerate(frames) if f["frame_index"] == a.frame)
    global ROAD, SIGNALS
    if a.road and Path(a.road).exists():
        ROAD = rm.derive(rm.load(a.road).get(a.frame))
        print(f"[mockup] road: median={ROAD['median']} W={ROAD['W']:.2f} b={ROAD['b']:+.3f} "
              f"c={ROAD['c']:+.6f} lines={[(round(l['a'],1), l['color'][0]+l['style'][0], 'syn' if l.get('synth') else 'det') for l in ROAD['lines']]}")
    if a.tl and Path(a.tl).exists():
        SIGNALS = json.loads(Path(a.tl).read_text()).get(str(a.frame), [])
        print(f"[mockup] signals: {[(s['state'], s['x'], s['y'], s['z']) for s in SIGNALS]}")
    placed = layout_objects(frames, fi, float(data.get("fps", 30.0)))
    print(f"[mockup] frame {a.frame}: {len(placed)} objects after NMS")
    for p in placed:
        print(f"   {p['cls']:8s} {str(p['sub']):10s} x={p['x']:6.1f} y={p['y']:6.1f} "
              f"heading={math.degrees(p['yaw']):+5.1f}deg flip={p['asset'] in ASSET_FLIP}")

    global JUNCTION
    JUNCTION = infra.find_junction(placed, SIGNALS, ROAD) if ROAD is not None else None
    if JUNCTION:
        print(f"[mockup] junction {JUNCTION['kind']} y={JUNCTION['y0']:.1f}-{JUNCTION['y1']:.1f} cues={JUNCTION['why']}")
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
    instance(ego_tpl, "Ego", 0.0, 0.0, 0.0, col,
             flip=("Vehicles/SedanAndHatchback.blend" in ASSET_FLIP) != a.flip_all)

    for i, p in enumerate(placed):
        ov = override
        if per_asset and p["asset"] in per_asset:
            ov = per_asset[p["asset"]]
        tpl = asset_template(assets, p["asset"], override=ov, fill=fill)
        if p["asset"] == "StopSign.blend" and not tpl.get("textured"):
            infra.texture_plate(tpl, assets / "StopSignImage.png", "StopSign")
            tpl["textured"] = True
        instance(tpl, f"Obj_{i}_{p['cls']}", p["x"], p["y"], p["yaw"], col,
                 flip=(p["asset"] in ASSET_FLIP) != a.flip_all)

    setup_camera(scene, col, cinematic=(a.style == "C"))
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
