"""EinsteinVision Phase 3 — vehicle semantics: parked/moving, brake lights, indicators.

    python3 einsteinvision/vehicle_state.py --scene 1 [--sheet]      (plain python + cv2, no bpy)

Output: road/sceneN/vehicle_state.json   (sidecar; detections.json is not mutated)
    {meta, ego:{frame: speed m/s},
     tracks:{oid: {class, moving, motion_state, world_speed_mps, rear_view, lamp_frames,
                   frames:{frame: {moving, speed, brake, indicator, readable}}}}}
        road/sceneN/vehicle_state_check.png  (contact sheet of real crops, labelled)

1. Ego speed: dash odometry (lane_bev, ds/ds_conf) where confident; gaps < 5 s
   interpolated; elsewhere ground-plane BEV phase-correlation flow (ego_flow.py:
   city streets without dashed lines, stops at lights).
2. Parked / moving: per-frame object ground position (mockup_render depth
   model) → per-track Gaussian smoothing (sequence_render.gauss_smooth) → local
   1 s linear fit → relative velocity + ego speed = world velocity. Hysteresis:
   stationary after < STOP_V for ≥ 1 s, moving again after > GO_V for ≥ 0.5 s.
   Track label = majority; "parked" = (almost) never moves while ego drives by
   or for ≥ 5 s, "stopped" = partly stationary (queued at a light). Far (> 60 m)
   or short (< 1.5 s) tracks default to moving (depth too noisy to call it).
3. Lamps (same-direction cars seen from behind, bbox ≥ MIN_W px only):
   ROIs = left / right third of the bbox, rows 30–70 % from the top (tail-lamp band).
     brake: red-lamp score (Lab: brightness of reddish pixels + bright-red fraction)
            vs. the track's own rolling low-percentile baseline; both sides must
            rise ≥ k·noise for ≥ 0.5 s (hysteresis on / off). Deceleration relaxes k.
     indicator: warm (amber / red) bright-pixel fraction per side; 1.5 s window
            FFT: dominant 0.9–2.3 Hz peak + autocorrelation at the blink period +
            amplitude well above the other side; both sides blinking = hazard.
   Unreadable (far, truncated, oncoming, side view) → brake False, indicator none.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _stub_bpy():
    """mockup_render / sequence_render import bpy at module level; the geometry we
    need (depth model, smoothing) is pure python → stub Blender modules."""
    for name in ("bpy", "mathutils"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__getattr__ = lambda attr: type(attr, (), {})       # noqa: E731
            sys.modules[name] = m


_stub_bpy()
import mockup_render as mr     # noqa: E402
import road_model as rm        # noqa: E402
import sequence_render as sr   # noqa: E402
import ego_flow                # noqa: E402

import os
DIAG = bool(os.environ.get("VS_DIAG"))
DIAG_OUT = []
VEH = {"car", "truck", "bus", "motorcycle"}
STOP_V, GO_V = 1.2, 2.5          # m/s hysteresis thresholds
MIN_W = 42                       # px bbox width for readable lamps
LAMP_ROWS = (0.25, 0.60)         # fraction of bbox height from the top
BLINK_BAND = (1.1, 2.3)          # Hz


# ═════════════════════════════════════════════════════════════════════════════
# lamp features
# ═════════════════════════════════════════════════════════════════════════════
def lamp_rois(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    r0, r1 = int(y1 + LAMP_ROWS[0] * h), int(y1 + LAMP_ROWS[1] * h)
    t = w / 3.0
    return (int(x1), r0, int(x1 + t), r1), (int(x2 - t), r0, int(x2), r1)


def lamp_feats(img, bbox):
    """→ (red_L, red_R, lum_L, lum_R) or None.

    The Tesla front camera is nearly achromatic (frame-wide a* p99.9 ≈ 4): lit
    tail lamps show as a faint a* tint (+3…+8) on bbox rows 25–60 %, amber and
    red are indistinguishable. Float Lab (sub-integer a*):
      red = p95 of a* (reddest lamp pixels), lum = p95 of L (brightest pixels)."""
    H, W = img.shape[:2]
    out = []
    for (a, b, c, d) in lamp_rois(bbox):
        a, c = max(0, a), min(W, c)
        b, d = max(0, b), min(H, d)
        if c - a < 6 or d - b < 4:
            return None
        roi = img[b:d, a:c].astype(np.float32) / 255.0
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        out.append((float(np.percentile(lab[:, 1], 95)), float(np.percentile(lab[:, 0], 95))))
    return out[0][0], out[1][0], out[0][1], out[1][1]


# ═════════════════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════════════════
def periodic(x, fps, band=None):
    """Blink test on a ~2 s lamp-brightness window.
    → (ratio, ac@period, ac@half-period, sd, score, ok, ok_strict)
    ratio: spectral energy fraction at the 1.1–2.3 Hz peak; a true on/off blink is
    positively correlated one period apart and anti-correlated half a period
    apart, with regular zero crossings (slow drift fails the half-period test)."""
    band = band or BLINK_BAND
    W = len(x)
    tt = np.arange(W)
    x = x - np.polyval(np.polyfit(tt, x, 2), tt)
    sd = float(x.std())
    if sd < 1.0:
        return (0.0, 0.0, 0.0, sd, 0.0, False, False)
    freqs = np.fft.rfftfreq(256, 1.0 / fps)
    inb = (freqs >= band[0]) & (freqs <= band[1])
    P = np.abs(np.fft.rfft(x * np.hanning(W), 256)) ** 2
    P[0] = 0
    pk = int(np.argmax(np.where(inb, P, 0)))
    ratio = float(P[max(0, pk - 3): pk + 4].sum() / (P.sum() + 1e-12))
    per = fps / max(freqs[pk], 1e-3)
    lag, half = int(round(per)), int(round(per / 2))
    ac = float(np.corrcoef(x[:-lag], x[lag:])[0, 1]) if 3 < lag < W - 8 else 0.0
    ah = float(np.corrcoef(x[:-half], x[half:])[0, 1]) if half >= 2 else 0.0
    zc = np.where(np.diff(np.sign(x)) != 0)[0]
    iv = np.diff(zc)
    regular = len(zc) >= 4 and iv.std() < 0.4 * iv.mean()
    score = ratio * max(ac, 0) * max(-ah, 0)
    ok = sd >= 1.5 and ratio > 0.55 and ac > 0.5 and ah < -0.4 and regular
    strict = sd >= 3.0 and ratio > 0.65 and ac > 0.65 and ah < -0.55 and regular
    return (ratio, ac, ah, sd, score, ok, strict)


def runs(mask):
    """[(start, end_exclusive)] of True runs."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def hysteresis(sig, on_thr, off_thr, on_len, off_len, init=False):
    """sig high → state True after ≥ on_len frames > on_thr; back to False after
    ≥ off_len frames < off_thr. Returns bool array (retro-applied to run start)."""
    n = len(sig)
    st = np.zeros(n, bool)
    cur, cnt, start = init, 0, 0
    for i, v in enumerate(sig):
        if not cur:
            if v > on_thr:
                if cnt == 0:
                    start = i
                cnt += 1
                if cnt >= on_len:
                    cur, cnt = True, 0
                    st[start:i + 1] = True
                    continue
            else:
                cnt = 0
        else:
            if v < off_thr:
                if cnt == 0:
                    start = i
                cnt += 1
                if cnt >= off_len:
                    cur, cnt = False, 0
                    st[start:i + 1] = False
                    continue
            else:
                cnt = 0
        st[i] = cur
    return st


def local_slope(t, y, half):
    """Least-squares slope of y(t) over ±half samples (edges shrink)."""
    n = len(y)
    out = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        tt, yy = t[a:b], y[a:b]
        if len(tt) < 3:
            continue
        tm = tt.mean()
        den = ((tt - tm) ** 2).sum()
        out[i] = ((tt - tm) * (yy - yy.mean())).sum() / den if den > 0 else 0.0
    return out


def _nanmed_filter(x, w):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w): i + w + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 3:
            out[i] = np.median(seg)
    return out


def ego_speed(scene, fps, n, flow, div, log=print):
    """Fused ego speed (m/s).
      metric : dash odometry (confident, gaps < 5 s interpolated) else BEV flow
      gate   : divergence / flow magnitude of static structure (Divergence) —
               stopped ⇒ 0; moving but metric < 3 m/s (dash zero-lock, texture-less
               asphalt) ⇒ α·divergence with α fitted where metric is trusted."""
    Rj = json.loads(Path(f"road/scene{scene}/road_model.json").read_text())["frames"]
    ds = np.full(n, np.nan)
    for f in Rj:
        k = f["frame_index"]
        if k < n and f.get("ds") is not None and f.get("ds_conf", 0) > 0.6:
            ds[k] = f["ds"] * fps
    dash = _nanmed_filter(ds, int(0.5 * fps))
    ok = np.isfinite(dash)
    if ok.any():                                          # fill gaps < 5 s by interpolation
        idx = np.arange(n)
        interp = np.interp(idx, idx[ok], dash[ok])
        for a, b in runs(~ok):
            if b - a < 5 * fps and a > 0 and b < n:
                dash[a:b] = interp[a:b]
    fl = ego_flow.smooth_speed(flow, fps)
    metric = np.where(np.isfinite(dash), dash, fl)
    src = np.where(np.isfinite(dash), "dash", "flow").astype(object)
    # divergence gate (1 s median)
    rad = _nanmed_filter(div[:, 0], int(0.5 * fps))
    mag = _nanmed_filter(div[:, 1], int(0.5 * fps))
    idx = np.arange(n)
    for arr in (rad, mag):
        g = np.isfinite(arr)
        if g.any():
            arr[:] = np.interp(idx, idx[g], arr[g])
        else:
            arr[:] = np.nan
    if np.isfinite(rad).all():
        stopped = (np.abs(rad) < 4.0) & (mag < 2.0)
        moving = (rad > 8.0) | (mag > 4.0)
        trust = moving & (metric > 5.0) & (rad > 8.0)
        alpha = float(np.median(metric[trust] / rad[trust])) if trust.sum() > fps else 0.5
        alpha = min(max(alpha, 0.2), 1.2)
        low = moving & (metric < np.maximum(3.0, 0.35 * alpha * rad))
        metric = np.where(low, np.maximum(alpha * np.maximum(rad, 0), 3.0), metric)
        src[low] = "divergence"
        metric = np.where(stopped, 0.0, metric)
        src[stopped] = "stopped"
        log(f"[vs] ego gate: stopped {stopped.mean():.0%}, div-substituted {low.mean():.0%}, alpha {alpha:.2f}")
    v = ego_flow.smooth_speed(metric, fps)
    return v, src


# ═════════════════════════════════════════════════════════════════════════════
# main analysis
# ═════════════════════════════════════════════════════════════════════════════
def analyse(scene, sheet=True, log=print):
    data = json.loads(Path(f"phase2_output/scene{scene}/detections.json").read_text())
    frames, fps = data["frames"], float(data.get("fps", 30.0))
    vid = sorted(glob.glob(f"P3Data/Sequences/scene{scene}/Undist/*-front_undistort.mp4"))[0]
    R = rm.load(f"road/scene{scene}/road_model.json")
    n = max(f["frame_index"] for f in frames) + 1
    byk = {f["frame_index"]: f for f in frames}

    # ── pass 1: video (ego flow + lamp features) ────────────────────────────
    cap = cv2.VideoCapture(vid)
    E = ego_flow.EgoFlow(fps)
    Dv = ego_flow.Divergence()
    flow = np.full(n, np.nan)
    div = np.full((n, 2), np.nan)
    feats = defaultdict(dict)                 # oid → frame → (rl, rr, wl, wr)
    keep_crops = defaultdict(dict)            # oid → frame → small crop (for the sheet)
    k = 0
    while k < n:
        ok, img = cap.read()
        if not ok:
            break
        objs = byk.get(k, {}).get("objects", [])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bxs = [o["bbox_2d"] for o in objs]
        flow[k] = E.step(gray, bxs)
        div[k] = Dv.step(gray, bxs)
        for o in objs:
            if o["class_name"] not in VEH:
                continue
            x1, y1, x2, y2 = o["bbox_2d"]
            if x2 - x1 < MIN_W or x1 <= 2 or x2 >= img.shape[1] - 2:
                continue
            f4 = lamp_feats(img, o["bbox_2d"])
            if f4 is None:
                continue
            oid = str(o["object_id"])
            feats[oid][k] = f4
            if sheet and k % 6 == 0:
                mx, my = int(0.12 * (x2 - x1)), int(0.12 * (y2 - y1))
                c = img[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx]
                sc = 150.0 / max(c.shape[0], 1)
                keep_crops[oid][k] = (cv2.resize(c, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA),
                                      (x1 - max(0, x1 - mx), y1 - max(0, y1 - my), x2 - x1, y2 - y1, sc))
        k += 1
    cap.release()
    n = k
    v_ego, v_src = ego_speed(scene, fps, n, flow[:n], div[:n], log)
    log(f"[vs] scene{scene}: {n} frames, ego speed median {np.median(v_ego):.1f} m/s "
        f"(p10 {np.percentile(v_ego, 10):.1f}, p90 {np.percentile(v_ego, 90):.1f}); "
        f"sources {Counter(v_src.tolist())}")

    # ── pass 2: world tracks ────────────────────────────────────────────────
    raw = defaultdict(dict)
    cls = defaultdict(Counter)
    onc = defaultdict(list)
    for k in range(n):
        f = byk.get(k)
        if not f:
            continue
        objs = [o for o in f["objects"] if o["class_name"] in VEH]
        road = rm.derive(R.get(k))
        for o in objs:
            x, y = mr._depth_xy(o, objs)
            if y > 110:
                continue
            oid = str(o["object_id"])
            raw[oid][k] = dict(x=x, y=y, yaw=0.0, bbox=list(o["bbox_2d"]), off=rm.offset_of(road, x, y))
            cls[oid][o["class_name"]] += 1
            onc[oid].append(rm.is_oncoming(road, x, y))
    tracks = {}
    t_all = np.arange(n) / fps
    for oid, ser in raw.items():
        sm = sr.gauss_smooth(ser, 3.5, keys=("x", "y", "off"))
        if len(sm) < int(0.5 * fps):
            continue
        ks = np.array(sorted(sm))
        # contiguous segments only (gauss_smooth drops long gaps)
        segs = np.split(ks, np.where(np.diff(ks) > 1)[0] + 1)
        spd = {}
        vfw = {}
        for sg in segs:
            if len(sg) < 5:
                continue
            tt = t_all[sg]
            ys = np.array([sm[j]["y"] for j in sg])
            xs = np.array([sm[j]["x"] for j in sg])
            half = int(0.5 * fps)
            vy = local_slope(tt, ys, half)
            vx = local_slope(tt, xs, half)
            fw = v_ego[sg] + vy                       # world forward speed (neg. = oncoming)
            for j, a, b in zip(sg, fw, vx):
                spd[j] = float(math.hypot(a, b))
                vfw[j] = float(a)
        if len(spd) < int(0.5 * fps):
            continue
        tracks[oid] = dict(sm=sm, spd=spd, vfw=vfw)

    # ── motion state with hysteresis ────────────────────────────────────────
    out_tracks = {}
    for oid, T in tracks.items():
        ks = sorted(T["spd"])
        sp = np.array([T["spd"][j] for j in ks])
        # depth noise ∝ z² → far tracks need a wider tolerance before "stationary"
        ymed = float(np.median([T["sm"][j]["y"] for j in ks]))
        tol = min(1.0 + max(0.0, (ymed - 25.0) * 0.05), 2.0)
        reliable = ymed <= 60.0 and len(ks) >= 1.5 * fps     # depth good enough to call "stationary"
        init_moving = float(np.median(sp[: int(fps)])) > STOP_V * tol
        moving = ~hysteresis(-sp, -STOP_V * tol, -GO_V * tol, int(1.0 * fps), int(0.5 * fps),
                             init=not init_moving)
        if not reliable and float(np.percentile(sp, 90)) > STOP_V:
            moving[:] = True                             # far / short: default moving (precision)
        frac_moving = float(moving.mean())
        long_stop = any(b - a >= 5 * fps for a, b in runs(~moving))
        ego_mv = float(np.median(v_ego[ks])) > 2.0
        rear = (np.mean(onc[oid]) < 0.5 and float(np.median([T["vfw"][j] for j in ks])) > -2.0)
        if frac_moving >= 0.5:
            mstate = "moving"
        elif frac_moving < 0.1 and (ego_mv or long_stop):
            mstate = "parked"                            # never moves while we drive past / long stop
        else:
            mstate = "stopped"                           # queued / waiting at a light
        out_tracks[oid] = dict(cls=cls[oid].most_common(1)[0][0], ks=ks, moving=moving, sp=sp,
                               rear=bool(rear), mstate=mstate, frac_moving=frac_moving)

    # ── lamps ───────────────────────────────────────────────────────────────
    lamp_stats = Counter()
    for oid, T in out_tracks.items():
        ks = T["ks"]
        brake = np.zeros(len(ks), bool)
        ind = np.array(["none"] * len(ks), dtype=object)
        readable = np.array([j in feats.get(oid, {}) for j in ks])
        T["brake"], T["ind"], T["readable"] = brake, ind, readable
        if not T["rear"] or readable.sum() < fps or T["cls"] == "motorcycle":
            continue
        pos = {j: i for i, j in enumerate(ks)}
        F = np.full((len(ks), 4), np.nan)
        for j, f4 in feats[oid].items():
            if j in pos:
                F[pos[j]] = f4
        # fill short unreadable gaps (≤ 4 frames) for the time series
        for c in range(4):
            col = F[:, c]
            okc = np.isfinite(col)
            if okc.sum() >= 2:
                F[:, c] = np.interp(np.arange(len(col)), np.where(okc)[0], col[okc])
        # --- brake
        rise = np.zeros((len(ks), 2))
        noise = np.zeros(2)
        for side in (0, 1):
            s = F[:, side]
            s = np.array([np.median(s[max(0, i - 2): i + 3]) for i in range(len(s))])
            w0, w1 = int(4 * fps), int(1 * fps)
            base = np.array([np.percentile(s[max(0, i - w0): i + w1 + 1], 20) for i in range(len(s))])
            d = np.diff(s)
            noise[side] = max(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2) * 3.0, 0.8)
            rise[:, side] = s - base
        # deceleration (world forward) relaxes the threshold
        vf = np.array([tracks[oid]["vfw"][j] for j in ks])
        acc = local_slope(np.array(ks) / fps, vf, int(0.5 * fps))
        k_on = np.where(acc < -1.0, 3.0, 4.5)
        z = np.min(rise / noise, axis=1)                # both sides must rise
        z_rel = z / k_on
        br = hysteresis(z_rel, 1.0, 0.45, int(0.5 * fps), int(0.3 * fps))
        br &= readable | (np.convolve(readable.astype(float), np.ones(9), "same") > 0)
        brake[:] = br
        lamp_stats["acc_brake"] += float(acc[br].sum())
        lamp_stats["acc_nobrake"] += float(acc[~br & readable].sum())
        lamp_stats["n_nobrake"] += int((~br & readable).sum())
        # --- indicators (periodic warm-lamp signal per side)
        W = int(2.0 * fps)
        blink = np.zeros((len(ks), 2), bool)
        blink_best = []
        off = np.array([tracks[oid]["sm"][j].get("off", tracks[oid]["sm"][j]["x"]) for j in ks])
        if len(ks) >= W:
            for i0 in range(0, len(ks) - W + 1, 3):
                if readable[i0:i0 + W].mean() < 0.85:
                    continue
                # lamp signal = brightness + red tint (US rear turn signals are often red)
                xl = F[i0:i0 + W, 2] + 2.5 * F[i0:i0 + W, 0]
                xr = F[i0:i0 + W, 3] + 2.5 * F[i0:i0 + W, 1]
                pl, pr, pdiff = (periodic(x, fps) for x in (xl, xr, xl - xr))
                blink_best.append((round(pdiff[4], 3), int(ks[i0]), [round(v, 2) for v in pdiff[:4]],
                                   round(pl[3], 2), round(pr[3], 2)))
                sl = slice(i0 + W // 4, i0 + 3 * W // 4)
                ratio_d, ac_d, ah_d, sd_d = pdiff[:4]
                loose = sd_d >= 2.0 and ratio_d > 0.55 and ac_d > 0.4 and ah_d < -0.5
                for side, (pa, pb) in enumerate(((pl, pr), (pr, pl))):
                    if not (pa[3] > 1.8 * pb[3]):      # one-sided: this lamp carries the blink
                        continue
                    # corroboration: lateral move toward the signalled side within 4 s
                    j0 = i0 + W // 2
                    fut = [off[j] for j in range(j0, min(len(ks), j0 + int(4 * fps)))]
                    moved = (max(fut) - off[j0] if side == 1 else off[j0] - min(fut)) if fut else 0.0
                    if pdiff[6] or (loose and moved >= 1.5):
                        blink[sl, side] = True
                if not blink[sl].any() and pl[6] and pr[6] and not pdiff[5] \
                        and min(pl[3], pr[3]) > 0.6 * max(pl[3], pr[3]):
                    blink[sl, :] = True            # hazard: both strong, in phase
            blink_best = sorted(blink_best, reverse=True)[:2]
        for side in (0, 1):                           # bridge gaps < 1 s between blink windows
            for a_, b_ in runs(~blink[:, side]):
                if 0 < a_ and b_ < len(ks) and b_ - a_ < int(1.0 * fps):
                    blink[a_:b_, side] = True
        for side in (0, 1):                           # keep only sustained blinking (≥ 1.2 s)
            for a, b in runs(blink[:, side]):
                if b - a < int(1.2 * fps):
                    blink[a:b, side] = False
        ind[blink[:, 0] & blink[:, 1]] = "hazard"
        ind[blink[:, 0] & ~blink[:, 1]] = "left"
        ind[~blink[:, 0] & blink[:, 1]] = "right"
        lamp_stats["tracks_lamp"] += 1
        if DIAG:
            DIAG_OUT.append(dict(oid=oid, n=len(ks), noise=noise.round(3).tolist(),
                                 z_p99=float(np.percentile(z, 99)), zrel_max=float(z_rel.max()),
                                 brake=int(brake.sum()), blink_best=blink_best))
        lamp_stats["brake_frames"] += int(brake.sum())
        lamp_stats["ind_frames"] += int((ind != "none").sum())

    # ── write ───────────────────────────────────────────────────────────────
    res = dict(meta=dict(scene=scene, fps=fps, frames=n, method=__doc__.split("\n")[0],
                         stop_v=STOP_V, go_v=GO_V, min_lamp_px=MIN_W,
                         ego_source={k_: round(float(np.mean(v_src == k_)), 3) for k_ in ("dash", "flow", "divergence", "stopped")}),
               ego={str(k): round(float(v_ego[k]), 2) for k in range(n)}, tracks={})
    for oid, T in out_tracks.items():
        ks = T["ks"]
        res["tracks"][oid] = dict(
            **{"class": T["cls"]}, moving=T["mstate"] == "moving", motion_state=T["mstate"],
            world_speed_mps=round(float(np.median(T["sp"])), 2), rear_view=T["rear"],
            lamp_frames=int(T["readable"].sum()),
            frames={str(j): dict(moving=bool(T["moving"][i]), speed=round(float(T["sp"][i]), 2),
                                 brake=bool(T["brake"][i]), indicator=str(T["ind"][i]),
                                 readable=bool(T["readable"][i]))
                    for i, j in enumerate(ks)})
    out = Path(f"road/scene{scene}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "vehicle_state.json").write_text(json.dumps(res))
    nb = sum(T["brake"].sum() for T in out_tracks.values())
    nf = sum(len(T["ks"]) for T in out_tracks.values())
    ms = Counter(T["mstate"] for T in out_tracks.values())
    inds = Counter(x for T in out_tracks.values() for x in T["ind"] if x != "none")
    log(f"[vs] mean world accel: braking {lamp_stats['acc_brake'] / max(lamp_stats['brake_frames'], 1):+.2f} m/s² "
        f"vs not {lamp_stats['acc_nobrake'] / max(lamp_stats['n_nobrake'], 1):+.2f} m/s²")
    log(f"[vs] scene{scene}: {len(out_tracks)} tracks {dict(ms)}; lamp-readable rear tracks "
        f"{lamp_stats['tracks_lamp']}; brake frames {nb}/{nf} ({nb / max(nf, 1):.1%}); indicators {dict(inds)}")
    for oid, T in sorted(out_tracks.items(), key=lambda kv: -len(kv[1]["ks"]))[:12]:
        log(f"[vs]   t{oid:>4} {T['cls']:10s} {T['mstate']:8s} v={np.median(T['sp']):5.1f} "
            f"rear={int(T['rear'])} frames={len(T['ks'])} readable={int(T['readable'].sum())} "
            f"brake={int(T['brake'].sum())} ind={dict(Counter(x for x in T['ind'] if x != 'none'))}")
    if DIAG:
        Path(f"/tmp/vs_diag_{scene}.json").write_text(json.dumps(DIAG_OUT))
        DIAG_OUT.clear()
    if sheet:
        contact_sheet(scene, out_tracks, keep_crops, out / "vehicle_state_check.png")
    return res


# ═════════════════════════════════════════════════════════════════════════════
# contact sheet
# ═════════════════════════════════════════════════════════════════════════════
def contact_sheet(scene, T, crops, path, per_row=8):
    cats = [("BRAKE", lambda t, i: t["brake"][i]),
            ("IND LEFT", lambda t, i: t["ind"][i] == "left"),
            ("IND RIGHT", lambda t, i: t["ind"][i] == "right"),
            ("HAZARD", lambda t, i: t["ind"][i] == "hazard"),
            ("no lamps (rear, readable)", lambda t, i: t["rear"] and t["readable"][i]
             and not t["brake"][i] and t["ind"][i] == "none"),
            ("PARKED", lambda t, i: t["mstate"] == "parked"),
            ("STOPPED", lambda t, i: t["mstate"] == "stopped"),
            ("MOVING", lambda t, i: t["mstate"] == "moving")]
    rows = []
    for name, pred in cats:
        picks = []
        cand = []
        for oid, t in T.items():
            idx = {j: i for i, j in enumerate(t["ks"])}
            hits = [j for j in crops.get(oid, {}) if j in idx and pred(t, idx[j])]
            if hits:
                cand.append((oid, hits, idx))
        cand.sort(key=lambda c: -len(c[1]))
        # spread: ≤ 2 crops per track (start / middle of its run), different tracks first
        for oid, hits, idx in cand:
            hits = sorted(hits)
            for j in {hits[len(hits) // 2], hits[0]}:
                picks.append((oid, j, idx[j]))
        picks = picks[:per_row]
        tiles = []
        for oid, j, i in picks:
            im, (ox, oy, bw, bh, sc) = crops[oid][j]
            im = im.copy()
            t = T[oid]
            for (a, b, c, d) in lamp_rois((0, 0, bw, bh)):
                cv2.rectangle(im, (int((a + ox) * sc), int((b + oy) * sc)),
                              (int((c + ox) * sc), int((d + oy) * sc)), (255, 255, 0), 1)
            lab = []
            lab.append(t["mstate"][:4].upper())
            if t["brake"][i]:
                lab.append("BRAKE")
            if t["ind"][i] != "none":
                lab.append(str(t["ind"][i]).upper())
            if not t["readable"][i] or not t["rear"]:
                lab.append("lamps:n/a")
            tile = np.zeros((190, max(im.shape[1], 150), 3), np.uint8)
            tile[:im.shape[0], :im.shape[1]] = im[:150]
            cv2.putText(tile, f"t{oid} f{j} v={t['sp'][i]:.1f}", (3, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            col = (0, 0, 255) if t["brake"][i] else ((0, 190, 255) if t["ind"][i] != "none" else (200, 200, 200))
            cv2.putText(tile, " ".join(lab), (3, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            tiles.append(tile)
        hdr = np.zeros((26, 10, 3), np.uint8)
        rows.append((name, tiles))
    Wmax = 8 * 230
    canvas = []
    for name, tiles in rows:
        band = np.full((26, Wmax, 3), 40, np.uint8)
        cv2.putText(band, f"{name}  ({len(tiles)} shown)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        canvas.append(band)
        row = np.zeros((190, Wmax, 3), np.uint8)
        x = 0
        for tl in tiles:
            w = min(tl.shape[1], Wmax - x)
            if w <= 0:
                break
            row[:, x:x + w] = tl[:, :w]
            x += w + 6
        if not tiles:
            cv2.putText(row, "(none detected)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        canvas.append(row)
    title = np.full((34, Wmax, 3), 20, np.uint8)
    cv2.putText(title, f"scene{scene} vehicle_state check - real crops, cyan = lamp ROIs, label = prediction",
                (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([title] + canvas))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, nargs="+", required=True)
    ap.add_argument("--no-sheet", action="store_true")
    a = ap.parse_args()
    for s in a.scene:
        analyse(s, sheet=not a.no_sheet)
