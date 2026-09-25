"""EinsteinVision — temporally smoothed sequence renderer (style C; --view front|top|chase).

    blender -b --factory-startup --python einsteinvision/sequence_render.py -- \
        --scene 1 --start 1900 --end 2140 --out renders/scene1_1900 [--samples 48 --res 1280 720]

1. layout every frame (mockup_render.layout_objects, road per frame)
2. temporal smoothing
     objects : per track Gaussian (σ = --sigma frames) on x, y, heading (sin/cos),
               gaps ≤ 4 frames interpolated, tracks seen < 6 frames dropped
     road    : lane structure = modal signature over ±10 frames (no flicker),
               b, c already EMA'd; dashes/barrier joints scroll by odometry s
     junction: world-anchored (s + y0), median over ±30 frames
     signals : per id Gaussian
3. render all frames in ONE Blender session (templates cached)
Prints raw-vs-smoothed jitter (RMS 2nd difference, cm/frame) → jitter.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mockup_render as mr   # noqa: E402
import road_model as rm      # noqa: E402
import infra                 # noqa: E402
import vstate_render as vsr  # noqa: E402


def gauss_smooth(series: dict[int, dict], sigma: float, keys=("x", "y")):
    """series: frame → obj dict. Returns frame → smoothed copy (incl. gap fill)."""
    fr = sorted(series)
    if len(fr) < 6:
        return {}
    out = {}
    lo, hi = fr[0], fr[-1]
    ymed = sorted(series[f]["y"] for f in fr)[len(fr) // 2]
    sg = sigma * min(max(ymed / 30.0, 1.0), 2.6)      # depth noise ∝ z² → wider window far away
    W = int(3 * sg)
    for k in range(lo, hi + 1):
        near0 = [f for f in fr if abs(f - k) <= 6]
        if not near0 or min(abs(f - k) for f in near0) > 4:   # gap too long → hidden
            continue
        near = [f for f in fr if abs(f - k) <= W]
        ws = [math.exp(-0.5 * ((f - k) / sg) ** 2) for f in near]
        sw = sum(ws)
        base = series[min(near, key=lambda f: abs(f - k))]
        o = dict(base)
        for key in keys:
            o[key] = sum(w * series[f][key] for w, f in zip(ws, near)) / sw
        sy = sum(w * math.sin(series[f]["yaw"]) for w, f in zip(ws, near))
        cy = sum(w * math.cos(series[f]["yaw"]) for w, f in zip(ws, near))
        o["yaw"] = math.atan2(sy, cy)
        out[k] = o
    return out


def jitter_cm(tracks: dict[str, dict[int, dict]]):
    acc = []
    for s in tracks.values():
        fr = sorted(s)
        for a, b, c in zip(fr, fr[1:], fr[2:]):
            if c - a == 2:
                for key in ("x", "y"):
                    acc.append(s[a][key] - 2 * s[b][key] + s[c][key])
    return 100 * math.sqrt(sum(v * v for v in acc) / len(acc)) if acc else 0.0


def road_signature(r):
    return (r["median"], tuple(round(l["a"] / 0.6) for l in r["lines"]),
            tuple(l["color"][0] + l["style"][0] for l in r["lines"]))


def dump_boxes(scene, cam, placed, VS, k, path):
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    bpy.context.view_layer.update()
    rx, ry = scene.render.resolution_x, scene.render.resolution_y
    rows = []
    for p in placed:
        if not p["asset"].startswith("Vehicles/"):
            continue
        W, L, H = p["dims"]
        c, s_ = math.cos(p["yaw"]), math.sin(p["yaw"])
        pts = []
        for dx in (-W / 2, W / 2):
            for dy in (-L / 2, L / 2):
                for z in (0.0, H):
                    v = world_to_camera_view(scene, cam, Vector((p["x"] + c * dx - s_ * dy, p["y"] + s_ * dx + c * dy, z)))
                    pts.append((v.x * rx, (1 - v.y) * ry, v.z))
        if min(q[2] for q in pts) <= 0:
            continue
        rows.append(dict(oid=p["oid"], asset=p["asset"], x=round(p["x"], 2), y=round(p["y"], 2),
                         yaw=round(p["yaw"], 3), state=VS.get(p["oid"], k),
                         bbox=[round(min(q[0] for q in pts)), round(min(q[1] for q in pts)),
                               round(max(q[0] for q in pts)), round(max(q[1] for q in pts))]))
    Path(path).write_text(json.dumps(rows, indent=1))


VIEWS = ("front", "top", "chase")


def setup_view(scene, col, view):
    """Camera for one view. front = the approved cinematic camera (mockup_render)."""
    if view == "front":
        return mr.setup_camera(scene, col, cinematic=True)
    cam_d = bpy.data.cameras.new(f"Cam_{view}")
    cam_d.clip_end = 800
    cam = bpy.data.objects.new(f"Cam_{view}", cam_d)
    col.objects.link(cam)
    if view == "top":        # orthographic bird's-eye, ego near bottom, forward = image up
        cam_d.type = "ORTHO"
        cam_d.ortho_scale = 72.0
        cam.location = (0.0, 14.5, 120.0)
        cam.rotation_euler = (0.0, 0.0, 0.0)          # looks down −Z, image-up = +Y (forward)
    else:                    # chase: well behind and above the ego car, ego fully in frame
        cam_d.lens = 28
        cam.location = (0.0, -17.0, 8.5)
        mr.look_at(cam, (0.0, 14.0, 0.0))
    scene.camera = cam
    return cam


def use_eevee(scene, samples):
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            continue
    ee = scene.eevee
    ee.taa_render_samples = max(4, samples)
    for attr, val in (("use_shadows", True), ("use_raytracing", False), ("use_gtao", True)):
        if hasattr(ee, attr):
            setattr(ee, attr, val)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--assets", default="P3Data/Assets")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--res", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("--sigma", type=float, default=3.5)
    ap.add_argument("--raw", action="store_true", help="no smoothing (for comparison)")
    ap.add_argument("--analyze", action="store_true", help="print jitter by distance, no render")
    ap.add_argument("--stills", type=int, nargs="*", default=None,
                    help="lay out/smooth the whole range but render only these frames")
    ap.add_argument("--step", type=int, default=1, help="render every Nth frame (video holds frames)")
    ap.add_argument("--jpeg", action="store_true", help="write JPEG (q90) instead of PNG to save disk")
    ap.add_argument("--no-vstate", action="store_true", help="ignore road/sceneN/vehicle_state.json")
    ap.add_argument("--view", nargs="+", choices=VIEWS, default=["front"],
                    help="camera(s): front (cinematic, default) | top (bird's-eye following ego) | chase "
                         "(behind+above ego). Several views → out/<view>/frame_N from ONE scene build per frame")
    ap.add_argument("--engine", choices=("cycles", "eevee"), default="cycles",
                    help="eevee = cheap EEVEE Next pass (for top/chase composite cells)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(a.assets)

    data = json.loads(Path(f"phase2_output/scene{a.scene}/detections.json").read_text())
    frames, fps = data["frames"], float(data.get("fps", 30.0))
    idx = {f["frame_index"]: i for i, f in enumerate(frames)}
    R = rm.load(f"road/scene{a.scene}/road_model.json")
    tlp = Path(f"road/scene{a.scene}/traffic_lights.json")
    TL = json.loads(tlp.read_text()) if tlp.exists() else {}
    ks = [k for k in range(a.start, a.end + 1) if k in idx and k in R]
    # Phase-3 semantics: parked/moving, brake lamps, indicators (vehicle_state.py sidecar)
    VS = vsr.VehicleState(f"road/scene{a.scene}/vehicle_state.json") if not a.no_vstate else vsr.VehicleState("")
    print(f"[seq] vehicle_state: {len(VS.tracks)} tracks" if VS else "[seq] vehicle_state: none")

    # ── 1. per-frame layout ──────────────────────────────────────────────────
    t0 = time.time()
    roads, raw_tracks, speeds = {}, defaultdict(dict), {}
    # ego speed: fused vehicle_state ego (dash + BEV flow + divergence stop gate) when
    # present — raw dash ds_s is bin-quantised / zero-locks in city scenes
    vsp = Path(f"road/scene{a.scene}/vehicle_state.json")
    ego_v = {}
    if vsp.is_file() and not a.no_vstate:
        ego_v = {int(k): float(v) for k, v in json.loads(vsp.read_text()).get("ego", {}).items()}
    for k in ks:
        rf = R[k]
        if k in ego_v:
            speeds[k] = min(max(ego_v[k], 0.0), 45.0)
        else:
            speeds[k] = min(max(rf.get("ds_s", 0.7) * fps, 3.0), 45.0)
        mr.EGO_SPEED = speeds[k]
        mr.ROAD = roads[k] = rm.derive(rf)
        for p in mr.layout_objects(frames, idx[k], fps):
            raw_tracks[p["oid"]][k] = p
    print(f"[seq] layout {len(ks)} frames in {time.time() - t0:.0f}s, {len(raw_tracks)} tracks")

    # ── 2. smoothing ────────────────────────────────────────────────────────
    if a.raw:
        sm_tracks = {o: s for o, s in raw_tracks.items()}
    else:
        sm_tracks = {o: gauss_smooth(s, a.sigma) for o, s in raw_tracks.items()}
    jr, js = jitter_cm(raw_tracks), jitter_cm(sm_tracks)
    if a.analyze:
        for lo, hi in ((0, 20), (20, 40), (40, 70), (70, 120)):
            band = lambda T: {o: {k: p for k, p in s_.items() if lo <= p["y"] < hi} for o, s_ in T.items()}
            n = sum(len(v) for v in band(raw_tracks).values())
            print(f"[seq] y {lo:3d}-{hi:3d} m: n={n:4d}  raw {jitter_cm(band(raw_tracks)):6.1f}  "
                  f"smooth {jitter_cm(band(sm_tracks)):6.1f} cm/frame")
        return
    print(f"[seq] object jitter (RMS 2nd diff): raw {jr:.1f} cm/frame → smoothed {js:.1f} cm/frame")
    per_frame = defaultdict(list)
    for o, s in sm_tracks.items():
        for k, p in s.items():
            per_frame[k].append(p)

    sigs = {k: road_signature(roads[k]) for k in ks}
    road_s = {}
    changes_raw = sum(1 for k0, k1 in zip(ks, ks[1:]) if sigs[k0] != sigs[k1])
    for k in ks:
        win = [j for j in ks if abs(j - k) <= 20]
        mode = Counter(sigs[j] for j in win).most_common(1)[0][0]
        same = [j for j in win if sigs[j] == mode]
        rep = min(same, key=lambda j: abs(j - k))
        r = dict(roads[rep])
        r["lines"] = [dict(l) for l in roads[rep]["lines"]]
        for i, l in enumerate(r["lines"]):                 # average offsets over the mode frames
            l["a"] = sum(roads[j]["lines"][i]["a"] for j in same) / len(same)
        r["left"], r["right"] = r["lines"][0]["a"], r["lines"][-1]["a"]
        if r.get("barrier_a") is not None:
            r["barrier_a"] = r["left"] - 0.9
        r["b"], r["c"] = roads[k]["b"], roads[k]["c"]
        road_s[k] = r if not a.raw else roads[k]
    changes_sm = sum(1 for k0, k1 in zip(ks, ks[1:]) if road_signature(road_s[k0]) != road_signature(road_s[k1]))
    print(f"[seq] road structure changes: raw {changes_raw} → smoothed {changes_sm}")

    if ego_v:                                      # distance for dash scrolling / junction anchoring
        S, acc = {}, 0.0
        for k in range(0, max(ks) + 1):            # absolute from frame 0: chunk-consistent
            acc += ego_v.get(k, 0.0) / fps
            S[k] = acc
        S = {k: S[k] for k in ks}
        print(f"[seq] ego speed from vehicle_state: median {3.6 * sorted(speeds.values())[len(speeds) // 2]:.0f} km/h, "
              f"{S[ks[-1]] - S[ks[0]]:.0f} m")
    else:
        S = {k: R[k].get("s", 0.0) for k in ks}
    Jw = {}
    for k in ks:
        mr.ROAD = road_s[k]
        J = infra.find_junction(per_frame[k], TL.get(str(k), []), road_s[k])
        if J:
            Jw[k] = S[k] + J["y0"]
    junc = {}
    for k in ks:
        win = [Jw[j] for j in ks if abs(j - k) <= 30 and j in Jw]
        n = sum(1 for j in ks if abs(j - k) <= 30)
        if win and len(win) >= 0.3 * n:
            y0 = sorted(win)[len(win) // 2] - S[k]
            if -20 < y0 < 120:
                junc[k] = dict(y0=y0, y1=y0 + 14.0, kind="4way", why=["smoothed"])
    sig_tracks = defaultdict(dict)
    for k in ks:
        for sgl in TL.get(str(k), []):
            sig_tracks[sgl["id"]][k] = dict(sgl, yaw=0.0)
    sig_s = defaultdict(list)
    for sid, s in sig_tracks.items():
        for k, v in (gauss_smooth(s, a.sigma, keys=("x", "y", "z")) if not a.raw else s).items():
            sig_s[k].append(v)
    Path(out / "jitter.json").write_text(json.dumps(dict(raw_cm=jr, smooth_cm=js, road_changes_raw=changes_raw,
                                                         road_changes_smooth=changes_sm,
                                                         junction_frames=len(junc))))

    # ── 3. render ───────────────────────────────────────────────────────────
    scene = mr.reset_scene()
    scene.render.resolution_x, scene.render.resolution_y = a.res
    ext = "jpg" if a.jpeg else "png"
    scene.render.image_settings.file_format = "JPEG" if a.jpeg else "PNG"
    if a.jpeg:
        scene.render.image_settings.quality = 90
    col = scene.collection
    ego_tpl = None
    t0 = time.time()
    rks = [k for k in ks if a.stills is None or k in set(a.stills)]
    if a.stills is None and a.step > 1:
        rks = [k for k in rks if (k - a.start) % a.step == 0]
    nstat = Counter()
    for n, k in enumerate(rks):
        for ob in list(col.all_objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        for c in list(col.children):
            col.children.unlink(c)
        if n % 25 == 0:
            bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=False, do_recursive=True)
        mr.ROAD, mr.SIGNALS, mr.JUNCTION = road_s[k], sig_s.get(k, []), junc.get(k)
        mr.DASH_PHASE = S[k]
        placed = per_frame.get(k, [])
        override, ego_mat, per_asset = mr.style_C(scene, col, assets, placed)
        scene.cycles.samples = a.samples
        fill = (per_asset or {}).get("__fill__")
        if ego_tpl is None:
            ego_tpl = mr.asset_template_copy(assets, "Vehicles/SedanAndHatchback.blend", ego_mat)
        mr.instance(ego_tpl, "Ego", 0.0, 0.0, 0.0, col,
                    flip="Vehicles/SedanAndHatchback.blend" in mr.ASSET_FLIP)
        for i, p in enumerate(placed):
            tpl = mr.asset_template(assets, p["asset"], override=override, fill=fill)
            if VS and p["asset"].startswith("Vehicles/"):
                st = VS.get(p["oid"], k)
                if st["parked"]:
                    tpl = vsr.parked_template(tpl)
                    if st["rear_view"] and math.cos(p["yaw"] - rm.heading_at(road_s[k], p["y"])) < 0:
                        p = dict(p, yaw=p["yaw"] - math.pi)   # no motion cue; we saw its tail → same direction
                    nstat["parked"] += 1
                vsr.add_lamps(col, f"Obj_{i}", p, st, k, fps)
                nstat["brake"] += st["brake"]
                nstat["ind"] += st["indicator"] != "none"
            if p["asset"] == "StopSign.blend" and not tpl.get("textured"):
                infra.texture_plate(tpl, assets / "StopSignImage.png", "StopSign")
                tpl["textured"] = True
            mr.instance(tpl, f"Obj_{i}", p["x"], p["y"], p["yaw"], col, flip=p["asset"] in mr.ASSET_FLIP)
        if a.engine == "eevee":
            use_eevee(scene, a.samples)
        for view in a.view:
            cam = setup_view(scene, col, view)
            vdir = out if a.view == ["front"] else out / view
            vdir.mkdir(parents=True, exist_ok=True)
            if a.stills is not None and view == "front":   # projected boxes → inset crops / vstate debugging
                dump_boxes(scene, cam, placed, VS, k, out / f"frame_{k:05d}_objs.json")
            scene.render.filepath = str((vdir / f"frame_{k:05d}.{ext}").resolve())
            bpy.ops.render.render(write_still=True)
        if n % 10 == 0:
            el = time.time() - t0
            print(f"[seq] {n + 1}/{len(rks)} frame {k}  {el / (n + 1):.1f}s/frame  eta {el / (n + 1) * (len(rks) - n - 1) / 60:.0f} min",
                  flush=True)
    print(f"[seq] vstate instances: {dict(nstat)}")
    print(f"[seq] done {len(rks)} frames in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
