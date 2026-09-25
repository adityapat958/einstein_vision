"""Classical lane-marking extractor in bird's-eye view (BEV).

Replaces the Mask R-CNN lane JSON (no colour, <22 m range, frequent misses).

Per frame of the undistorted front video:
  1. inverse-perspective warp to a metric ground grid (camera h=1.45 m,
     horizon v=444 px measured from 2485 cars in scene1)
  2. marking mask = lateral top-hat (thin bright stripes) ∧ (white | yellow)
  3. sliding-window line tracing seeded by the column histogram
  4. joint fit: every line shares heading b and curvature c, own offset a
         x_i(z) = a_i + b·z + c·z²        (x right, z forward, metres)
  5. per line: colour (yellow/white), style (solid/dashed) from fill ratio,
     double (twin stripe 0.2–0.45 m apart)
  6. temporal tracking: EMA on (b, c), line association by offset,
     lines survive short dropouts (dash gaps, occlusion)

Output: <out>/road_model.json   {meta, frames:[{frame_index, b, c, lines:[...]}]}
        <out>/debug/*.png       camera | BEV overlays for a few frames

    python3 einsteinvision/lane_bev.py --video .../front_undistort.mp4 --out road/scene1 \
        [--start 0 --end -1 --debug-frames 600 1530 2131]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

FX, FY, CX, CY = 1594.7, 1607.7, 654.3, 413.4
CAM_H = 1.45
V_HORIZON = 444.0

# BEV grid
X_MIN, X_MAX, DX = -14.0, 14.0, 0.05
Z_MIN, Z_MAX, DZ = 5.0, 50.0, 0.10
BW = int(round((X_MAX - X_MIN) / DX))
BH = int(round((Z_MAX - Z_MIN) / DZ))


def bev_maps():
    xs = X_MIN + (np.arange(BW) + 0.5) * DX
    zs = Z_MAX - (np.arange(BH) + 0.5) * DZ           # row 0 = far
    X, Z = np.meshgrid(xs, zs)
    u = CX + FX * X / Z
    v = V_HORIZON + FY * CAM_H / Z
    return u.astype(np.float32), v.astype(np.float32), xs, zs


def marking_masks(bev_bgr):
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))        # 0.75 m lateral
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    s_top = cv2.morphologyEx(hsv[..., 1], cv2.MORPH_TOPHAT, k)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lab_b = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2LAB)[..., 2]
    b_top = cv2.morphologyEx(lab_b, cv2.MORPH_TOPHAT, k)              # local yellowness
    # washed-out daylight yellow: sat only ~45-55, so use local contrast, not absolute sat
    yellow = (h >= 14) & (h <= 34) & (s >= 38) & (v >= 80) & ((s_top >= 14) | (b_top >= 6))
    white = (s <= 60) & (v >= 150) & (top >= 28)
    clean = lambda m: cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    return clean(white).astype(bool), clean(yellow).astype(bool)


def trace_lines(mask, xs, zs, max_lines=8):
    """Sliding windows near→far from histogram peaks. Returns list of (xpts, zpts)."""
    near = mask[BH // 2:, :].sum(axis=0).astype(np.float32)
    near = np.convolve(near, np.ones(9) / 9, mode="same")
    peaks = []
    order = np.argsort(near)[::-1]
    for c in order:
        if near[c] < 12:
            break
        if all(abs(c - p) * DX > 1.2 for p in peaks):
            peaks.append(c)
        if len(peaks) >= max_lines:
            break
    ys, xs_idx = np.nonzero(mask)
    lines = []
    nwin, half = 15, int(0.45 / DX)
    win_h = BH // nwin
    for p in peaks:
        cx = p
        px, pz = [], []
        misses = 0
        for w in range(nwin):
            r1 = BH - w * win_h
            r0 = r1 - win_h
            sel = (ys >= r0) & (ys < r1) & (xs_idx >= cx - half) & (xs_idx < cx + half)
            if sel.sum() >= 6:
                cx = int(np.median(xs_idx[sel]))
                px.append(xs[xs_idx[sel]]); pz.append(zs[ys[sel]])
                misses = 0
            else:
                misses += 1
                if misses > 4:
                    break
        if px:
            px, pz = np.concatenate(px), np.concatenate(pz)
            if np.ptp(pz) >= 6.0 and len(px) >= 40:
                lines.append((px, pz))
    return lines


def joint_fit(lines, b0=0.0, c0=0.0, reg=50.0):
    """x = a_i + b z + c z²; shared b,c; ridge prior toward previous (b0,c0)."""
    n = len(lines)
    rows, rhs = [], []
    for i, (px, pz) in enumerate(lines):
        w = 1.0 / math.sqrt(len(px))                  # equal weight per line
        A = np.zeros((len(px), n + 2))
        A[:, i] = 1
        A[:, n] = pz
        A[:, n + 1] = pz ** 2
        rows.append(A * w); rhs.append(px * w)
    A = np.vstack(rows); y = np.concatenate(rhs)
    P = np.zeros((2, n + 2)); P[0, n] = reg * 1.0; P[1, n + 1] = reg * 40.0
    A = np.vstack([A, P]); y = np.concatenate([y, [reg * b0, reg * 40.0 * c0]])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = sol[:n]; b, c = sol[n], sol[n + 1]
    res = [float(np.median(np.abs(px - (a[i] + b * pz + c * pz ** 2)))) for i, (px, pz) in enumerate(lines)]
    return a, float(b), float(c), res


def describe(a, b, c, white, yellow, zs):
    """colour / style / double for a fitted line."""
    fill, ycount, wcount, twin = [], 0, 0, 0
    bins = np.arange(6.0, 22.0, 0.5)     # near band: far yellow fades in daylight washout
    for z in bins:
        r = int((Z_MAX - z) / DZ)
        if r < 0 or r >= BH:
            continue
        xc = a + b * z + c * z * z
        c0 = int((xc - X_MIN) / DX)
        lo, hi = max(c0 - 6, 0), min(c0 + 7, BW)
        wv, yv = white[r, lo:hi].any(), yellow[r, lo:hi].any()
        fill.append(wv or yv)
        wcount += int(wv); ycount += int(yv)
        for off in (-1, 1):                             # twin 0.2–0.45 m to either side
            t0 = c0 + off * int(0.2 / DX); t1 = c0 + off * int(0.45 / DX)
            lo2, hi2 = sorted((t0, t1))
            lo2, hi2 = max(lo2, 0), min(hi2 + 1, BW)
            if yellow[r, lo2:hi2].any() and yv:
                twin += 1
                break
    f = float(np.mean(fill)) if fill else 0.0
    color = "yellow" if ycount > 0.5 * max(wcount, 1) and ycount >= 3 else "white"
    return dict(color=color, fill=round(f, 2),
                double=bool(color == "yellow" and twin >= 0.4 * max(ycount, 1)))


class Tracker:
    def __init__(self):
        self.b = 0.0; self.c = 0.0
        self.lines: list[dict] = []
        self.next_id = 0

    def update(self, dets, b, c, alpha=0.25):
        if dets:
            self.b += alpha * (b - self.b); self.c += alpha * (c - self.c)
        for L in self.lines:
            L["age"] += 1
        for d in dets:
            best = min(self.lines, key=lambda L: abs(L["a"] - d["a"]), default=None)
            if best is not None and abs(best["a"] - d["a"]) < 0.9 and best["age"] > 0:
                best["a"] += 0.35 * (d["a"] - best["a"])
                best["age"] = 0
                best["hits"] += 1
                best["fill_hist"] = (best["fill_hist"] + [d["fill"]])[-40:]
                best["y_votes"] += 1 if d["color"] == "yellow" else -1
                best["y_votes"] = max(-15, min(15, best["y_votes"]))
                best["dbl_votes"] = max(-15, min(15, best["dbl_votes"] + (1 if d["double"] else -1)))
            else:
                self.lines.append(dict(id=self.next_id, a=d["a"], age=0, hits=1, fill_hist=[d["fill"]],
                                       y_votes=1 if d["color"] == "yellow" else -1,
                                       dbl_votes=1 if d["double"] else -1))
                self.next_id += 1
        self.lines = [L for L in self.lines if L["age"] <= 45]
        # merge lines that converged
        self.lines.sort(key=lambda L: L["a"])
        merged = []
        for L in self.lines:
            if merged and abs(L["a"] - merged[-1]["a"]) < 0.6:
                if L["hits"] > merged[-1]["hits"]:
                    merged[-1] = L
            else:
                merged.append(L)
        self.lines = merged

    def state(self):
        out = []
        for L in self.lines:
            if L["hits"] < 4:
                continue
            f = float(np.mean(L["fill_hist"]))
            yellow = L["y_votes"] > 0
            out.append(dict(id=L["id"], a=round(L["a"], 3), color="yellow" if yellow else "white",
                            style="solid" if f >= 0.70 else "dashed",
                            double=bool(yellow and L["dbl_votes"] > 0), fill=round(f, 2),
                            stale=L["age"]))
        return out


ODO_Z0, ODO_Z1 = 6.0, 36.0


def dash_profile(mask, lines, b, c):
    """Mean paint occupancy along z for all tracked dashed lines (fresh ones)."""
    zs = np.arange(ODO_Z0, ODO_Z1, DZ)
    rows = ((Z_MAX - zs) / DZ).astype(int)
    profs = []
    for L in lines:
        if L["style"] != "dashed" or L["stale"] > 2:
            continue
        xc = L["a"] + b * zs + c * zs * zs
        cols = ((xc - X_MIN) / DX).astype(int)
        pr = np.zeros(len(zs), np.float32)
        for k, (r, cc) in enumerate(zip(rows, cols)):
            if 0 <= r < BH and 6 <= cc < BW - 6:
                pr[k] = mask[r, cc - 6:cc + 7].any()
        if pr.sum() >= 15 and pr.mean() < 0.7:          # real dashes: some paint, some gaps
            profs.append(pr)
    if not profs:
        return None
    return np.mean(profs, axis=0)


def match_shift(prev, cur, max_ds=2.2):
    """ds (m) maximising NCC of cur(z) vs prev(z + ds); returns (ds, score) or (None, score).
    Sub-bin parabola on the NCC curve (DZ = 0.1 m is 13 km/h per bin at 36 fps, so
    integer bins alone quantise city speeds to 0 / 13 / 26 km/h)."""
    n = len(cur)
    K = int(max_ds / DZ) + 1
    sc = np.full(K + 1, -1.0)
    for k in range(0, K + 1):
        a = cur[: n - k]
        b_ = prev[k:]
        if len(a) < 20 or a.std() < 1e-3 or b_.std() < 1e-3:
            continue
        sc[k] = float(np.corrcoef(a, b_)[0, 1])
    best = int(np.argmax(sc))
    bs = float(sc[best])
    if bs < 0.55:
        return None, bs
    d = float(best)
    if 0 < best < K and sc[best - 1] > -1 and sc[best + 1] > -1:
        y0, y1, y2 = sc[best - 1], sc[best], sc[best + 1]
        den = y0 - 2 * y1 + y2
        if den < -1e-9:
            d += float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return d * DZ, bs


ODO_LAG = 4          # compare against the profile 4 frames back: 3.2 km/h per bin, ≤ 4.4 m shift


def match_shift_lagged(hist, cur):
    """Per-frame ds from the longest available lag (≤ ODO_LAG) whose match is confident."""
    for lag in range(min(ODO_LAG, len(hist)), 0, -1):
        p = hist[-lag]
        if p is None:
            continue
        ds, sc = match_shift(p, cur, max_ds=1.1 * lag)
        if ds is not None:
            return ds / lag, sc
    return None, 0.0


def draw_debug(frame, bev, white, yellow, lines_fit, trk_state, b, c, out_png):
    vis = bev.copy()
    vis[white] = (255, 255, 255); vis[yellow] = (0, 220, 255)
    cam = frame.copy()
    for L in trk_state:
        col = (0, 220, 255) if L["color"] == "yellow" else (255, 255, 255)
        if L["style"] == "dashed":
            col = tuple(int(0.6 * v) for v in col)
        zs = np.linspace(Z_MIN, Z_MAX, 60)
        xs = L["a"] + b * zs + c * zs * zs
        pb = np.stack([(xs - X_MIN) / DX, (Z_MAX - zs) / DZ], 1).astype(np.int32)
        cv2.polylines(vis, [pb], False, (0, 0, 255), 2)
        pc = np.stack([CX + FX * xs / zs, V_HORIZON + FY * CAM_H / zs], 1).astype(np.int32)
        cv2.polylines(cam, [pc], False, (0, 0, 255), 3)
        tag = f"{L['color'][0].upper()}{'2' if L['double'] else ''}-{L['style'][0]}"
        cv2.putText(cam, tag, tuple(pc[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(cam, f"curv c={c:+.5f} (R={1/(2*abs(c)) if abs(c)>1e-6 else 1e9:.0f} m) b={b:+.3f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    vis = cv2.resize(vis, (int(BW * cam.shape[0] / BH), cam.shape[0]))
    cv2.imwrite(str(out_png), np.hstack([cam, vis]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--debug-frames", type=int, nargs="*", default=[])
    ap.add_argument("--post", action="store_true", help="post debug images via vizjob_hook")
    a = ap.parse_args()
    out = Path(a.out); (out / "debug").mkdir(parents=True, exist_ok=True)
    mu, mv, xs, zs = bev_maps()
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = n if a.end < 0 else min(a.end, n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    trk = Tracker()
    frames_out = []
    prof_hist = []          # last ODO_LAG dash profiles (None = no dashes)
    r0, r1 = int((Z_MAX - 30.0) / DZ), int((Z_MAX - 6.0) / DZ)      # z 6–30 m
    c0, c1 = int((-7.0 - X_MIN) / DX), int((7.0 - X_MIN) / DX)        # |x| < 7 m
    win = cv2.createHanningWindow((c1 - c0, r1 - r0), cv2.CV_32F)
    dbg = set(a.debug_frames)
    for fi in range(a.start, end):
        ok, frame = cap.read()
        if not ok:
            break
        bev = cv2.remap(frame, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        white, yellow = marking_masks(bev)

        lines = trace_lines(white | yellow, xs, zs)
        dets, b, c = [], trk.b, trk.c
        if lines:
            av, b, c, res = joint_fit(lines, trk.b, trk.c)
            if abs(c) > 0.004 or abs(b) > 0.25:        # implausible (R < 125 m / 14°) → skip
                av, b, c = [], trk.b, trk.c
            for i, ai in enumerate(av):
                if res[i] > 0.25:
                    continue
                d = describe(ai, b, c, white, yellow, zs)
                d["a"] = float(ai)
                dets.append(d)
        trk.update(dets, b, c)
        st = trk.state()
        # ego odometry from dashed paint: profile along each dashed line (z 6–36 m),
        # marks move toward the camera → cur(z) ≈ prev(z + ds)
        prof = dash_profile(white | yellow, st, trk.b, trk.c)
        ds, dconf = None, 0.0
        if prof is not None and prof_hist:
            ds, dconf = match_shift_lagged(prof_hist, prof)
        prof_hist.append(prof)
        if len(prof_hist) > ODO_LAG:
            prof_hist.pop(0)
        frames_out.append(dict(frame_index=fi, b=round(trk.b, 5), c=round(trk.c, 7), lines=st,
                               n_raw=len(dets), ds=None if ds is None else round(ds, 4),
                               ds_conf=round(float(dconf), 3)))
        if fi in dbg:
            p = out / "debug" / f"lanes_f{fi}.png"
            draw_debug(frame, bev, white, yellow, lines, st, trk.b, trk.c, p)
            if a.post:
                try:
                    import vizjob_hook as vj
                    vj.post_image(p, f"lane BEV debug frame {fi}")
                except Exception as e:
                    print("post failed", e)
        if fi % 300 == 0:
            print(f"[lanes] {fi}/{end} lines={[(L['color'][0]+L['style'][0], L['a']) for L in st]} "
                  f"c={trk.c:+.5f}", flush=True)
    # fill / smooth odometry: median over ±9 frames of valid ds, then cumulative distance s
    dsv = np.array([f["ds"] if f["ds"] is not None else np.nan for f in frames_out], float)
    sm = np.array([np.nanmedian(dsv[max(0, i - 9): i + 10]) if np.isfinite(dsv[max(0, i - 9): i + 10]).any()
                   else np.nan for i in range(len(dsv))])
    sm = np.where(np.isfinite(sm), sm, np.nanmedian(dsv) if np.isfinite(dsv).any() else 0.0)
    S = np.cumsum(sm)
    for f, d, s_ in zip(frames_out, sm, S):
        f["ds_s"], f["s"] = round(float(d), 4), round(float(s_), 3)
    print(f"[lanes] odometry: valid {np.isfinite(dsv).mean():.0%} of frames; median speed {np.median(sm) * fps * 3.6:.0f} km/h, distance {S[-1]:.0f} m")
    meta = dict(video=a.video, fps=fps, fx=FX, fy=FY, cx=CX, cy=CY, cam_h=CAM_H, v_horizon=V_HORIZON,
                model="x = a + b z + c z^2 (ego frame, x right, z fwd, m)")
    (out / "road_model.json").write_text(json.dumps(dict(meta=meta, frames=frames_out)))
    have = sum(1 for f in frames_out if f["lines"])
    print(f"[lanes] wrote {out/'road_model.json'}  frames={len(frames_out)} with_lines={have}")


if __name__ == "__main__":
    main()
