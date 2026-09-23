"""EinsteinVision — temporally smoothed sequence renderer (style C, front chase view).

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


def gauss_smooth(series: dict[int, dict], sigma: float, keys=("x", "y")):
    """series: frame → obj dict. Returns frame → smoothed copy (incl. gap fill)."""
    fr = sorted(series)
    if len(fr) < 6:
        return {}
    out = {}
    W = int(3 * sigma)
    lo, hi = fr[0], fr[-1]
    for k in range(lo, hi + 1):
        near = [f for f in fr if abs(f - k) <= W]
        if not near or min(abs(f - k) for f in near) > 4:     # gap too long → hidden
            continue
        ws = [math.exp(-0.5 * ((f - k) / sigma) ** 2) for f in near]
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

    # ── 1. per-frame layout ──────────────────────────────────────────────────
    t0 = time.time()
    roads, raw_tracks, speeds = {}, defaultdict(dict), {}
    for k in ks:
        rf = R[k]
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
    print(f"[seq] object jitter (RMS 2nd diff): raw {jr:.1f} cm/frame → smoothed {js:.1f} cm/frame")
    per_frame = defaultdict(list)
    for o, s in sm_tracks.items():
        for k, p in s.items():
            per_frame[k].append(p)

    sigs = {k: road_signature(roads[k]) for k in ks}
    road_s = {}
    changes_raw = sum(1 for k0, k1 in zip(ks, ks[1:]) if sigs[k0] != sigs[k1])
    for k in ks:
        win = [j for j in ks if abs(j - k) <= 10]
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
    scene.render.image_settings.file_format = "PNG"
    col = scene.collection
    ego_tpl = None
    t0 = time.time()
    for n, k in enumerate(ks):
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
            if p["asset"] == "StopSign.blend" and not tpl.get("textured"):
                infra.texture_plate(tpl, assets / "StopSignImage.png", "StopSign")
                tpl["textured"] = True
            mr.instance(tpl, f"Obj_{i}", p["x"], p["y"], p["yaw"], col, flip=p["asset"] in mr.ASSET_FLIP)
        mr.setup_camera(scene, col, cinematic=True)
        scene.render.filepath = str((out / f"frame_{k:05d}.png").resolve())
        bpy.ops.render.render(write_still=True)
        if n % 10 == 0:
            el = time.time() - t0
            print(f"[seq] {n + 1}/{len(ks)} frame {k}  {el / (n + 1):.1f}s/frame  eta {el / (n + 1) * (len(ks) - n - 1) / 60:.0f} min",
                  flush=True)
    print(f"[seq] done {len(ks)} frames in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
