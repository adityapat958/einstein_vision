"""Look-dev layer for the cinematic renders: per-scene environment (sky / time of day / wetness),
vehicle car-paint shader, roadside ground.  Imported by mockup_render.style_C.

Scene table is from eyeballing the 13 front videos (contact sheet 2026-09-25):
  kind   : highway (no sidewalks / street furniture, guardrails) | city
  tod    : day | overcast | dusk | night
  wet    : glossy road (scene 6 is raining)
"""
import hashlib
import math
from pathlib import Path

import bpy

EXTRA = Path("P3Data/ExtraAssets")

SCENE_ENV = {
    1: dict(kind="highway", tod="day"),
    2: dict(kind="highway", tod="overcast"),
    3: dict(kind="city", tod="overcast"),
    4: dict(kind="city", tod="overcast"),
    5: dict(kind="city", tod="day"),
    6: dict(kind="city", tod="overcast", wet=True),
    7: dict(kind="city", tod="overcast"),
    8: dict(kind="city", tod="overcast"),
    9: dict(kind="city", tod="dusk"),
    10: dict(kind="city", tod="night"),
    11: dict(kind="city", tod="dusk"),
    12: dict(kind="highway", tod="night"),
    13: dict(kind="highway", tod="dusk"),
}
ENV = dict(kind="city", tod="day")          # current scene (set by sequence_render / mockup_render)


def set_scene(n: int | None):
    global ENV
    ENV = dict(kind="city", tod="day", wet=False, **{})
    ENV.update(SCENE_ENV.get(n, {}))
    return ENV


def is_night():
    return ENV.get("tod") == "night"


# ── sky / lighting ────────────────────────────────────────────────────────────
def _world(scene):
    w = bpy.data.worlds.new("World")
    scene.world = w
    w.use_nodes = True
    return w, w.node_tree


def world_hdri(scene, name, strength=1.0, rot_deg=0.0):
    p = EXTRA / "hdri" / f"{name}.hdr"
    if not p.exists():
        return False
    w, nt = _world(scene)
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(p.resolve()), check_existing=True)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rot_deg))
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = strength
    return True


def world_nishita(scene, elev, rot, strength, air=1.0, dust=1.0, ozone=1.0):
    w, nt = _world(scene)
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(elev)
    sky.sun_rotation = math.radians(rot)
    sky.air_density, sky.dust_density, sky.ozone_density = air, dust, ozone
    sky.sun_disc = False
    nt.links.new(sky.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = strength


def world_night(scene):
    """Deep blue-violet gradient: sodium-lit urban sky glow near the horizon, dark zenith."""
    w, nt = _world(scene)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.030, 0.018, 0.012, 1)     # horizon: warm light pollution
    ramp.color_ramp.elements[1].position = 0.25
    ramp.color_ramp.elements[1].color = (0.002, 0.003, 0.008, 1)     # zenith
    nt.links.new(ramp.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = 1.0


def sun(col, energy, elev, azim, color=(1, 1, 1), angle=1.0):
    d = bpy.data.lights.new("Sun", "SUN")
    d.energy, d.color, d.angle = energy, color, math.radians(angle)
    ob = bpy.data.objects.new("Sun", d)
    ob.rotation_euler = (math.radians(90 - elev), 0.0, math.radians(azim))
    col.objects.link(ob)
    return ob


def lighting(scene, col):
    """World + key light for the current scene's time of day. Returns exposure tweak."""
    tod = ENV.get("tod", "day")
    if tod == "day":
        if not world_hdri(scene, "kloofendal_48d_partly_cloudy_puresky", 0.9, 30):
            world_nishita(scene, 30, 200, 0.3)
        sun(col, 3.2, 38, 215, (1.0, 0.93, 0.84), 1.5)
        return 0.0
    if tod == "overcast":
        if not world_hdri(scene, "kloofendal_overcast_puresky", 1.3, 0):
            world_nishita(scene, 25, 200, 0.4, dust=4)
        sun(col, 0.9, 50, 215, (0.95, 0.97, 1.0), 12.0)              # very soft
        return 0.1
    if tod == "dusk":                     # late-afternoon low sun (scenes 9/11/13): warm key, cool sky
        if not world_hdri(scene, "kloofendal_48d_partly_cloudy_puresky", 0.55, 30):
            world_nishita(scene, 8, 250, 0.4)
        sun(col, 2.6, 9.0, 250, (1.0, 0.62, 0.36), 1.0)                # long warm shadows
        return 0.15
    world_night(scene)                                                # night
    sun(col, 0.06, 40, 120, (0.6, 0.7, 1.0), 2.0)                     # moon / sky-glow fill
    return 0.5


def street_lights(col, road, phase, y0=-20.0, y1=200.0, every=34.0):
    """Night only: overhead light pools (no posts — user asked for no lamp posts), world-anchored."""
    import road_model as rm
    left = min([l["a"] for l in road["oncoming_lines"]], default=road["left"])
    xs = [road["right"] - 2.5, left + 2.5]          # lamp heads on arms reaching over the outer lanes
    n0 = math.floor((y0 + phase) / every)
    n1 = math.ceil((y1 + phase) / every)
    for n in range(n0, n1 + 1):
        y = n * every - phase
        for i, a in enumerate(xs):
            yy = y + (every / 2 if i else 0.0)
            d = bpy.data.lights.new("StreetL", "SPOT")
            d.energy = 5000.0
            d.color = (1.0, 0.70, 0.40)                                # sodium / warm LED
            d.spot_size = math.radians(140)
            d.spot_blend = 0.7
            d.shadow_soft_size = 0.4
            ob = bpy.data.objects.new("StreetL", d)
            ob.location = (rm.x_at(road, a, yy), yy, 10.0)
            col.objects.link(ob)


# ── materials ────────────────────────────────────────────────────────────────
def pbr(name, tex_dir: Path, scale=0.25, nor_strength=0.6):
    """Poly Haven texture set with world-space XY mapping (any resolution suffix)."""
    files = {p.name.lower(): p for p in tex_dir.glob("*")} if tex_dir.exists() else {}
    diff = next((p for k, p in files.items() if "diff" in k), None)
    if diff is None:
        return None, None
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
    dt = tex(diff)
    nt.links.new(dt.outputs["Color"], b.inputs["Base Color"])
    rough = next((p for k, p in files.items() if "rough" in k), None)
    if rough:
        nt.links.new(tex(rough, True).outputs["Color"], b.inputs["Roughness"])
    nor = next((p for k, p in files.items() if "nor_gl" in k), None)
    if nor:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = nor_strength
        nt.links.new(tex(nor, True).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    return m, dt


def grass_material():
    """Roadside grass: PBR texture at two scales mixed by a large noise (kills visible tiling),
    tinted towards late-winter New England (the footage is March: brown-green, not lawn green)."""
    m, dt = pbr("GrassPBR", EXTRA / "textures/aerial_grass_rock", 0.12, 0.8)
    if m is None:
        return None
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp2 = nt.nodes.new("ShaderNodeMapping")
    mp2.inputs["Scale"].default_value = (0.031, 0.031, 0.031)
    nt.links.new(tc.outputs["Object"], mp2.inputs["Vector"])
    t2 = nt.nodes.new("ShaderNodeTexImage")
    t2.image = dt.image
    nt.links.new(mp2.outputs["Vector"], t2.inputs["Vector"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.02
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(noise.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(dt.outputs["Color"], mix.inputs[6])
    nt.links.new(t2.outputs["Color"], mix.inputs[7])
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Hue"].default_value = 0.55          # push the straw texture towards olive green
    hsv.inputs["Saturation"].default_value = 0.85
    hsv.inputs["Value"].default_value = 0.55
    nt.links.new(mix.outputs[2], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], b.inputs["Base Color"])
    add_haze(m)
    return m


HAZE = {  # tod → (linear colour, emission strength) ≈ sky radiance just above the horizon
    "day": ((0.62, 0.68, 0.78), 0.9), "overcast": ((0.66, 0.67, 0.70), 1.0),
    "dusk": ((0.62, 0.62, 0.66), 0.55), "night": ((0.03, 0.018, 0.012), 1.0),
}


def add_haze(m, near=60.0, far=650.0, maxf=0.92):
    """Aerial perspective on the ground plane: blend towards the horizon sky colour with view
    distance, so the infinite ground does not end in a hard line against the sky."""
    nt = m.node_tree
    outn = nt.nodes["Material Output"]
    src = outn.inputs["Surface"].links[0].from_socket
    cam = nt.nodes.new("ShaderNodeCameraData")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = near, far
    mr.inputs["To Max"].default_value = maxf
    nt.links.new(cam.outputs["View Distance"], mr.inputs["Value"])
    col, stren = HAZE.get(ENV.get("tod", "day"), HAZE["day"])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*col, 1)
    em.inputs["Strength"].default_value = stren
    ms = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mr.outputs["Result"], ms.inputs["Fac"])
    nt.links.new(src, ms.inputs[1])
    nt.links.new(em.outputs["Emission"], ms.inputs[2])
    nt.links.new(ms.outputs["Shader"], outn.inputs["Surface"])


def wet_road(road_mat):
    """Rain: darker, glossy asphalt (reflections of lamps / cars)."""
    b = road_mat.node_tree.nodes.get("Principled BSDF")
    if b is None:
        return
    r = b.inputs["Roughness"]
    for l in list(r.links):
        road_mat.node_tree.links.remove(l)
    r.default_value = 0.12
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = 0.6
        b.inputs["Coat Roughness"].default_value = 0.05


# Real-world US car colour mix (PPG/Axalta ≈2023), linear RGB
PALETTE = [
    ("white", (0.80, 0.80, 0.78), 0.25), ("black", (0.012, 0.012, 0.014), 0.22),
    ("grey", (0.12, 0.125, 0.13), 0.18), ("silver", (0.42, 0.43, 0.45), 0.12),
    ("blue", (0.02, 0.05, 0.16), 0.09), ("red", (0.32, 0.012, 0.012), 0.09),
    ("brown", (0.10, 0.06, 0.035), 0.03), ("green", (0.03, 0.07, 0.04), 0.02),
]
EGO_COLOR = (0.34, 0.012, 0.016)          # Tesla "Ultra Red"-ish: keeps the approved red ego, less plastic
COLORS = {}                                # oid → linear RGB (car_colors.json), else hashed palette


def load_colors(path):
    import json
    COLORS.clear()
    p = Path(path)
    if p.exists():
        COLORS.update({k: tuple(v) for k, v in json.loads(p.read_text()).items()})
    return len(COLORS)


def color_for(oid: str, asset: str = ""):
    if oid in COLORS:
        return COLORS[oid]
    if asset.endswith("Truck.blend") and "Pickup" not in asset:
        return (0.72, 0.72, 0.70)                     # box trucks: white box
    h = int(hashlib.md5(oid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for _, c, w in PALETTE:
        acc += w
        if h <= acc:
            return c
    return PALETTE[0][1]


def car_paint(name="CarPaint", glass=True):
    """Body colour from the instancer's object colour (one shared material for all cars).
    Generated-Z bands (mesh is baked to Z-up metres): glass greenhouse above the beltline on
    non-horizontal faces, black rubber/trim below the sills."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.use_fake_user = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "INSTANCER"
    attr.attribute_name = "color"
    body = attr.outputs["Color"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sepn = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], sepn.inputs["Vector"])

    def math_node(op, a, bb=None, clamp=False):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, v in enumerate((a, bb)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                nt.links.new(v, n.inputs[i])
        return n.outputs[0]

    above = math_node("GREATER_THAN", sep.outputs["Z"], 0.60)
    steep = math_node("LESS_THAN", math_node("ABSOLUTE", sepn.outputs["Z"]), 0.78)
    glass_m = math_node("MULTIPLY", above, steep) if glass else None
    below = math_node("LESS_THAN", sep.outputs["Z"], 0.16)

    def mixc(fac, a, c):
        mx = nt.nodes.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        nt.links.new(fac, mx.inputs["Factor"])
        for sock, v in ((mx.inputs[6], a), (mx.inputs[7], c)):
            if isinstance(v, tuple):
                sock.default_value = (*v, 1.0)
            else:
                nt.links.new(v, sock)
        return mx.outputs[2]

    def mixf(fac, a, c):
        mx = nt.nodes.new("ShaderNodeMix")
        mx.data_type = "FLOAT"
        nt.links.new(fac, mx.inputs["Factor"])
        for sock, v in ((mx.inputs[2], a), (mx.inputs[3], c)):
            if isinstance(v, (int, float)):
                sock.default_value = v
            else:
                nt.links.new(v, sock)
        return mx.outputs[0]

    col = mixc(below, body, (0.012, 0.012, 0.012))      # tyres / sills / trim
    rough = mixf(below, 0.32, 0.7)
    metal = mixf(below, 0.45, 0.0)
    if glass:
        col = mixc(glass_m, col, (0.006, 0.008, 0.01))
        rough = mixf(glass_m, rough, 0.03)
        metal = mixf(glass_m, metal, 0.0)
    nt.links.new(col, b.inputs["Base Color"])
    nt.links.new(rough, b.inputs["Roughness"])
    nt.links.new(metal, b.inputs["Metallic"])
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = 0.8
        b.inputs["Coat Roughness"].default_value = 0.04
    return m


def box_truck_paint():
    """Box trucks: white box, dark cab glass handled by the generic shader is wrong (box is tall) →
    plain off-white body with the lower trim band only."""
    return car_paint("TruckPaint", glass=False)
