"""Phase-3 vehicle-state visualisation helpers for the Blender renderers.

Reads road/sceneN/vehicle_state.json (vehicle_state.py) and turns it into geometry:
  * brake      → both rear lamps emissive red (bright), otherwise dim tail-lamp red
  * indicator  → amber lamp on the indicated side blinking at BLINK_HZ (hazard = both)
  * parked/stopped → car drawn with a desaturated, slightly darker copy of its materials

Lamp placement is in the car's own frame: yaw is the CCW heading from +Y (direction of
travel, as in mockup_render.instance), so forward f = (-sin yaw, cos yaw), right r = (cos yaw, sin yaw).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import bpy

BLINK_HZ = 1.5
LAMP_H = {  # rear-lamp centre height (m) per asset
    "Vehicles/SedanAndHatchback.blend": 0.85,
    "Vehicles/SUV.blend": 1.00,
    "Vehicles/PickupTruck.blend": 1.00,
    "Vehicles/Truck.blend": 1.10,
    "Vehicles/Motorcycle.blend": 0.80,
    "Vehicles/Bicycle.blend": 0.80,
}


class VehicleState:
    """Lookup of per-track / per-frame state; missing data → moving, lamps off."""

    def __init__(self, path):
        p = Path(path)
        self.tracks = json.loads(p.read_text()).get("tracks", {}) if str(path) and p.is_file() else {}

    def __bool__(self):
        return bool(self.tracks)

    def get(self, oid, k):
        t = self.tracks.get(str(oid))
        if t is None:
            return dict(moving=True, parked=False, rear_view=False, brake=False, indicator="none")
        st = t["frames"].get(str(k))
        moving = bool(st["moving"]) if st is not None and "moving" in st else bool(t["moving"])
        return dict(moving=moving, parked=not moving, rear_view=bool(t.get("rear_view")),
                    brake=bool(st and st.get("brake")),
                    indicator=(st or {}).get("indicator", "none"))


def blink_on(k, fps):
    """On for the first half of each period; +1e-6 guards k*hz/fps landing a hair under an integer."""
    return ((k * BLINK_HZ / fps) + 1e-6) % 1.0 < 0.5


# ── materials / meshes (fake-user so orphans_purge in the frame loop keeps them) ──
_CACHE: dict = {}


def _emit_mat(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Emission Color"].default_value = (*color, 1)
    b.inputs["Emission Strength"].default_value = strength
    b.inputs["Roughness"].default_value = 0.3
    m.use_fake_user = True
    return m


def _box_mesh(name, mat):
    me = bpy.data.meshes.new(name)
    v = [(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    me.from_pydata(v, [], f)
    me.materials.append(mat)
    me.use_fake_user = True
    return me


def lamp_meshes():
    if not _CACHE.get("lamps"):
        _CACHE["lamps"] = {
            "tail": _box_mesh("LampTail", _emit_mat("TailDim", (0.10, 0.004, 0.004), 0.15)),
            "brake": _box_mesh("LampBrake", _emit_mat("BrakeOn", (1.0, 0.02, 0.01), 80.0)),
            "amber_off": _box_mesh("LampAmberOff", _emit_mat("AmberOff", (0.10, 0.05, 0.01), 0.0)),
            "amber_on": _box_mesh("LampAmberOn", _emit_mat("AmberOn", (1.0, 0.35, 0.0), 50.0)),
        }
    return _CACHE["lamps"]


def _desat_material(m, sat=0.25, val=0.6):
    m2 = m.copy()
    m2.name = f"{m.name}_parked"
    m2.use_fake_user = True
    if not m2.use_nodes:
        c = m2.diffuse_color
        g = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
        m2.diffuse_color = tuple(val * (g + sat * (ch - g)) for ch in c[:3]) + (c[3],)
        return m2
    nt = m2.node_tree
    for b in [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"]:
        inp = b.inputs["Base Color"]
        hsv = nt.nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Saturation"].default_value = sat
        hsv.inputs["Value"].default_value = val
        if inp.is_linked:
            src = inp.links[0].from_socket
            nt.links.new(src, hsv.inputs["Color"])
        else:
            hsv.inputs["Color"].default_value = inp.default_value[:]
        nt.links.new(hsv.outputs["Color"], inp)
        if "Coat Weight" in b.inputs:
            b.inputs["Coat Weight"].default_value = 0.0      # matte, "static" look
        b.inputs["Roughness"].default_value = max(0.6, b.inputs["Roughness"].default_value)
    return m2


def parked_template(tpl):
    """Desaturated copy of a template collection (cached per template)."""
    key = ("parked", tpl.name)
    if key in _CACHE:
        return _CACHE[key]
    col = bpy.data.collections.new(f"{tpl.name}_parked")
    mats = {}
    for o in tpl.objects:
        o2 = o.copy()
        if o.data is not None:
            o2.data = o.data.copy()
            for i, m in enumerate(o2.data.materials):
                if m is None:
                    continue
                if m.name not in mats:
                    mats[m.name] = _desat_material(m)
                o2.data.materials[i] = mats[m.name]
        col.objects.link(o2)
    col.use_fake_user = True
    _CACHE[key] = col
    return col


def add_lamps(col, name, p, st, k, fps):
    """Rear lamp boxes for placed vehicle p (dict with asset, x, y, yaw, dims)."""
    if not p["asset"].startswith("Vehicles/"):
        return 0
    L = lamp_meshes()
    W, Ln, _ = p["dims"]
    yaw = p["yaw"]
    f = (-math.sin(yaw), math.cos(yaw))
    r = (math.cos(yaw), math.sin(yaw))
    h = LAMP_H.get(p["asset"], 0.9)
    moto = p["asset"] in ("Vehicles/Motorcycle.blend", "Vehicles/Bicycle.blend")
    red_off = 0.0 if moto else W / 2 - 0.25
    amb_off = 0.22 if moto else W / 2 - 0.25          # directly under the tail lamp (body corners are curved)
    back = -(Ln / 2 + 0.05)
    on = blink_on(k, fps)
    ind = st["indicator"]
    specs = []
    for side in ((0,) if moto else (-1, 1)):
        specs.append(("brake" if st["brake"] else "tail", side * red_off, h,
                      (0.34 if not moto else 0.14, 0.06, 0.13)))
    for side, nm in ((-1, "left"), (1, "right")):
        lit = on and (ind == nm or ind == "hazard")
        specs.append(("amber_on" if lit else "amber_off", side * amb_off, h - 0.16, (0.30, 0.10, 0.12)))
    for i, (kind, lat, z, sc) in enumerate(specs):
        ob = bpy.data.objects.new(f"{name}_lamp{i}", L[kind])
        ob.location = (p["x"] + back * f[0] + lat * r[0], p["y"] + back * f[1] + lat * r[1], z)
        ob.rotation_euler = (0, 0, yaw)
        ob.scale = sc
        col.objects.link(ob)
    return len(specs)
