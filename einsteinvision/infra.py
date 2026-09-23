"""EinsteinVision — road infrastructure for the Blender mockups.

* find_junction(): decide whether an intersection is ahead and where (from
  traffic-light heads, stop signs, the crosswalk flag, crossing traffic)
* build_junction(): perpendicular cross street (4-way), stop line, zebra
  crossings, cross-street lane paint — the ego road's lines are cut there
* build_sidewalks(): curb + concrete sidewalk for city roads
* scatter_furniture(): street lamps, poles, trees/shrubs, hydrants, bins
  (Poly Haven CC0 models in P3Data/ExtraAssets, falls back to nothing)
* texture_plate(): project an image onto a sign's plate faces (stop / speed)
* speed_limit_png(): draw a US R2-1 "SPEED LIMIT NN" texture

All geometry follows the curved road model (road_model.x_at).
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import bpy
from mathutils import Vector

import road_model as rm

EXTRA = Path("P3Data/ExtraAssets")


# ─── tiny helpers (self-contained; no import of mockup_render) ─────────────────
def _mat(name, color, rough=0.5, metal=0.0, emit=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = strength
    return m


def _obj(name, v, f, mat, col):
    me = bpy.data.meshes.new(name)
    me.from_pydata(v, [], f)
    me.update()
    ob = bpy.data.objects.new(name, me)
    if mat:
        me.materials.append(mat)
    col.objects.link(ob)
    return ob


def _quad(x0, x1, y0, y1, z):
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [(0, 1, 2, 3)]


def _merge(parts):
    V, F = [], []
    for v, f in parts:
        o = len(V)
        V += v
        F += [tuple(i + o for i in q) for q in f]
    return V, F


def _box(cx, cy, cz, sx, sy, sz):
    v = [(cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2)
         for dz in (-1, 1) for dy in (-1, 1) for dx in (-1, 1)]
    return v, [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def _ribbon(road, a0, a1, y0, y1, z, step=2.0):
    v, f = [], []
    n = max(1, int((y1 - y0) / step))
    for i in range(n + 1):
        y = y0 + (y1 - y0) * i / n
        v += [(rm.x_at(road, a0, y), y, z), (rm.x_at(road, a1, y), y, z)]
        if i:
            k = len(v) - 4
            f.append((k, k + 2, k + 3, k + 1))
    return v, f


# ─── image textures ───────────────────────────────────────────────────────────
def image_material(name, img_path, rough=0.4, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(str(Path(img_path).resolve()), check_existing=True)
    t.extension = "CLIP"
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    if emit:                                       # retro-reflective pop
        nt.links.new(t.outputs["Color"], b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = emit
    b.inputs["Roughness"].default_value = rough
    return m


def pbr_material(name, tex_dir: Path, scale=0.25):
    """Poly Haven texture set (diffuse/normal/rough) with world-space XY mapping."""
    files = {p.name.lower(): p for p in tex_dir.glob("*")} if tex_dir.exists() else {}
    diff = next((p for k, p in files.items() if "diff" in k), None)
    if diff is None:
        return None
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def tex(p, non_color=False):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(str(p.resolve()), check_existing=True)
        if non_color:
            t.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
        return t
    nt.links.new(tex(diff).outputs["Color"], b.inputs["Base Color"])
    rough = next((p for k, p in files.items() if "rough" in k), None)
    if rough:
        nt.links.new(tex(rough, True).outputs["Color"], b.inputs["Roughness"])
    nor = next((p for k, p in files.items() if "nor_gl" in k), None)
    if nor:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 0.6
        nt.links.new(tex(nor, True).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    return m


def speed_limit_png(value: int, out: Path) -> Path:
    """US MUTCD R2-1 sign face, 600×750 px."""
    from PIL import Image, ImageDraw, ImageFont
    out.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (600, 750), "white")
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((12, 12, 588, 738), 36, outline="black", width=14)
    try:
        f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 118)
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 330)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    for text, y, f in (("SPEED", 60, f1), ("LIMIT", 190, f1), (str(value), 330, f2)):
        w = d.textlength(text, font=f)
        d.text(((600 - w) / 2, y), text, font=f, fill="black")
    im.save(out)
    return out


def texture_plate(template_col, img_path, name, back_color=(0.55, 0.56, 0.57)):
    """Assign an image to the sign plate faces (largest flat faces facing ±X or ±Y
    in world space, above 1.2 m), with our own planar UVs over the plate bbox."""
    face_mat = image_material(f"{name}_Face", img_path, emit=0.15)
    back = _mat(f"{name}_Metal", back_color, rough=0.35, metal=0.8)
    for o in template_col.objects:
        if o.type != "MESH":
            continue
        me = o.data
        mw = o.matrix_world
        rot = mw.to_3x3()
        me.materials.clear()
        me.materials.append(back)
        me.materials.append(face_mat)
        # plate normal axis: the thin horizontal axis of the whole object
        pts = [mw @ v.co for v in me.vertices]
        ext = [max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(2)]
        ax = 0 if ext[0] < ext[1] else 1
        zmax = max(p.z for p in pts)
        plate = []
        for pl in me.polygons:
            n = (rot @ pl.normal).normalized()
            c = mw @ pl.center
            if abs(n[ax]) > 0.85 and c.z > 0.55 * zmax and n[ax] < 0:   # front side (−axis)
                plate.append(pl)
        if not plate:
            continue
        uax = 1 - ax
        vs = [mw @ me.vertices[i].co for pl in plate for i in pl.vertices]
        u0, u1 = min(p[uax] for p in vs), max(p[uax] for p in vs)
        v0, v1 = min(p.z for p in vs), max(p.z for p in vs)
        uv = me.uv_layers.new(name="PlateUV") if "PlateUV" not in me.uv_layers else me.uv_layers["PlateUV"]
        me.uv_layers.active = uv
        for pl in plate:
            pl.material_index = 1
            for li in pl.loop_indices:
                p = mw @ me.vertices[me.loops[li].vertex_index].co
                u = (p[uax] - u0) / max(u1 - u0, 1e-6)
                uv.data[li].uv = ((1 - u) if ax == 0 else u, (p.z - v0) / max(v1 - v0, 1e-6))
    return template_col


# ─── junction ────────────────────────────────────────────────────────────────
def find_junction(placed, signals, road, frames=None):
    """→ dict(y0, y1, kind, why) or None.  y0/y1 = near/far edge of the cross street."""
    cues = []
    heads = [s for s in (signals or []) if 10 < s["y"] < 90]
    over = [s for s in heads if road is None or
            road["left"] - 2.0 < rm.offset_of(road, s["x"], s["y"]) < road["right"] + 2.0]
    use = over or heads                             # heads above our lanes hang over the FAR side
    if use:
        ys = sorted(s["y"] for s in use)
        cues.append(("signals", ys[len(ys) // 2] - 16.0))
    for p in placed:
        if p["cls"] == "stop sign" and p["y"] < 45:
            cues.append(("stop sign", p["y"] + 2.0))
    if road and road.get("crosswalk") is not None:
        cues.append(("crosswalk", 9.0))
    for p in placed:                                 # crossing traffic
        if p["asset"].startswith("Vehicles/") and abs(abs(p["yaw"]) - math.pi / 2) < 0.6 and p["y"] < 70:
            cues.append(("cross traffic", p["y"] - 7.0))
    if not cues:
        return None
    y0 = max(6.0, sorted(c[1] for c in cues)[len(cues) // 2])
    if road is not None:
        queue = [p["y"] + p["dims"][1] / 2 for p in placed if p["asset"].startswith("Vehicles/")
                 and abs(p["yaw"]) < 0.5 and not rm.is_oncoming(road, p["x"], p["y"])
                 and road["left"] - 0.5 < rm.offset_of(road, p["x"], p["y"]) < road["right"] + 0.5
                 and p["y"] < y0 + 14.0]
        if queue and max(queue) > y0 - 4.6:        # stop line sits behind the queue's head
            y0 = max(queue) + 5.0
    return dict(y0=y0, y1=y0 + 14.0, kind="4way", why=[c[0] for c in cues])


def build_junction(col, road, J, road_mat, white, yellow, extent=90.0):
    y0, y1 = J["y0"], J["y1"]
    xc = rm.x_at(road, 0.0, (y0 + y1) / 2)
    # cross street surface (slightly above main road to avoid z-fighting)
    _obj("CrossStreet", *_quad(xc - extent, xc + extent, y0, y1, 0.004), road_mat, col)
    lw = 0.15
    ym = (y0 + y1) / 2
    road_l = min([l["a"] for l in road["oncoming_lines"]], default=road["left"])
    road_r = road["right"]
    xl, xr = rm.x_at(road, road_l, ym) - 0.5, rm.x_at(road, road_r, ym) + 0.5
    parts_w, parts_y = [], []
    for xa, xb in ((xc - extent, xl), (xr, xc + extent)):          # outside the box only
        parts_y += [_quad(xa, xb, ym - 0.12 - lw / 2, ym - 0.12 + lw / 2, 0.016),
                    _quad(xa, xb, ym + 0.12 - lw / 2, ym + 0.12 + lw / 2, 0.016)]
        for yy in (y0 + 3.5, y1 - 3.5):                              # lane dashes
            x = xa
            while x < xb:
                parts_w.append(_quad(x, min(x + 3.0, xb), yy - lw / 2, yy + lw / 2, 0.016))
                x += 12.0
    # stop line across our approach lanes (0.6 m, 1.2 m before the crosswalk)
    a_lo = road["left"] if road["median"] != "none" else 0.0
    parts_w.append(_ribbon(road, a_lo, road_r, y0 - 4.6, y0 - 4.0, 0.016, step=0.6))
    # zebra crossings on all four arms (continental, 0.6 m bars / 0.6 m gaps)
    zebra = []
    for yy0 in (y0 - 3.4, y1 + 0.4):
        x = xl + 0.3
        while x < xr - 0.3:
            zebra.append(_quad(x, x + 0.6, yy0, yy0 + 3.0, 0.016))
            x += 1.2
    for xx in (xl - 3.4, xr + 0.4):
        y = y0 + 0.3
        while y < y1 - 0.3:
            zebra.append(_quad(xx, xx + 3.0, y, y + 0.6, 0.016))
            y += 1.2
    _obj("JunctionPaintW", *_merge(parts_w + zebra), white, col)
    _obj("JunctionPaintY", *_merge(parts_y), yellow, col)


# ─── sidewalks & furniture ───────────────────────────────────────────────────
def build_sidewalks(col, road, J, concrete, curb_mat, y0=-40.0, y1=260.0, width=3.0):
    left_a = min([l["a"] for l in road["oncoming_lines"]], default=road["left"])
    right_a = road["right"]
    spans = [(y0, y1)] if J is None else [(y0, J["y0"] - 3.6), (J["y1"] + 3.6, y1)]
    parts_s, parts_c = [], []
    for a0, sign in ((right_a + 0.4, 1), (left_a - 0.4, -1)):
        for s0, s1 in spans:
            if s1 - s0 < 2:
                continue
            ain, aout = (a0, a0 + sign * width) if sign > 0 else (a0 + sign * width, a0)
            v, f = _ribbon(road, min(ain, aout), max(ain, aout), s0, s1, 0.15)
            parts_s.append((v, f))
            ac = a0 if sign > 0 else a0                          # curb face
            v2, f2 = [], []
            n = max(1, int((s1 - s0) / 2))
            for i in range(n + 1):
                y = s0 + (s1 - s0) * i / n
                x = rm.x_at(road, ac, y)
                v2 += [(x, y, 0.0), (x, y, 0.15)]
                if i:
                    k = len(v2) - 4
                    f2.append((k, k + 2, k + 3, k + 1))
            parts_c.append((v2, f2))
    if parts_s:
        _obj("Sidewalks", *_merge(parts_s), concrete, col)
        _obj("Curbs", *_merge(parts_c), curb_mat, col)


_PH: dict[str, bpy.types.Collection] = {}


def ph_template(name):
    """Poly Haven model (real-world metres, Z-up) → unlinked collection, base at z=0."""
    if name in _PH:
        return _PH[name]
    files = sorted((EXTRA / name).glob("*.blend"))
    if not files:
        _PH[name] = None
        return None
    with bpy.data.libraries.load(str(files[0].resolve()), link=False) as (src, dst):
        dst.objects = list(src.objects)
    objs = [o for o in dst.objects if o is not None and o.type in ("MESH", "EMPTY")]
    colT = bpy.data.collections.new(f"PH_{name}")
    for o in objs:
        colT.objects.link(o)
    meshes = [o for o in objs if o.type == "MESH"]
    if meshes:
        zmin = min((o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box)
        for o in objs:
            if o.parent is None:
                o.location.z -= zmin
    _PH[name] = colT
    return colT


def _inst(tpl, name, x, y, yaw, col, scale=1.0):
    e = bpy.data.objects.new(name, None)
    e.instance_type = "COLLECTION"
    e.instance_collection = tpl
    e.location = (x, y, 0.0)
    e.rotation_euler = (0, 0, yaw)
    e.scale = (scale, scale, scale)
    col.objects.link(e)
    return e


def scatter_furniture(col, road, J, seed=7, y0=-10.0, y1=200.0):
    rnd = random.Random(seed)
    city = road["median"] != "barrier"
    left_a = min([l["a"] for l in road["oncoming_lines"]], default=road["left"])
    right_a = road["right"]
    placed = 0

    def free(y):
        return J is None or not (J["y0"] - 6 < y < J["y1"] + 6)

    lamp = ph_template("street_lamp_01") or ph_template("street_lamp_02")
    if lamp:
        step = 32.0 if city else 45.0
        y = y0 + 8
        while y < y1:
            if free(y):
                _inst(lamp, "Lamp", rm.x_at(road, right_a + (0.9 if city else 2.2), y), y,
                      rm.heading_at(road, y) + math.pi / 2, col)
                if city:
                    _inst(lamp, "LampL", rm.x_at(road, left_a - 0.9, y + step / 2), y + step / 2,
                          rm.heading_at(road, y) - math.pi / 2, col)
                placed += 1
            y += step
    if not city:
        pole = ph_template("modular_electricity_poles")
        if pole:
            y = y0 + 20
            while y < y1:
                _inst(pole, "Pole", rm.x_at(road, right_a + 9.0, y), y, rm.heading_at(road, y), col)
                y += 55.0
    greens = [t for t in (ph_template(n) for n in ("tree_small_02", "shrub_01", "shrub_02", "shrub_04",
                                                   "jacaranda_tree")) if t]
    if greens:
        for side_a, sgn in ((right_a, 1), (left_a, -1)):
            y = y0 + rnd.uniform(0, 10)
            while y < y1:
                if free(y):
                    off = (3.8 if city else 7.0) + rnd.uniform(0, 4.0 if city else 12.0)
                    t = rnd.choice(greens)
                    _inst(t, "Green", rm.x_at(road, side_a + sgn * off, y), y, rnd.uniform(0, 6.28), col,
                          scale=rnd.uniform(0.8, 1.2))
                y += rnd.uniform(9, 18) if city else rnd.uniform(14, 30)
    if city:
        for name, every in (("fire_hydrant", 70.0), ("metal_trash_can", 55.0), ("utility_box_01", 90.0)):
            t = ph_template(name)
            if not t:
                continue
            y = y0 + rnd.uniform(5, every)
            while y < y1:
                if free(y):
                    _inst(t, name, rm.x_at(road, right_a + 1.2, y), y, rm.heading_at(road, y), col)
                y += every
    return placed
