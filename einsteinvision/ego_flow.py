"""Ego forward speed from ground-plane optical flow (pure cv2/numpy, no bpy).

Road-surface corners between the horizon and the hood are tracked frame to
frame (pyramidal LK + forward-backward check). A ground point on image row v
lies at depth Z = fy·h / (v − v_h); its forward displacement between frames is
Z_prev − Z_next, and the ego speed is the robust median × fps.
Pixels inside detected objects are masked, so a lead car cannot bias it.

Works where dash odometry (lane_bev) fails: city streets without dashed
lines, stopped at lights (→ ≈0), night (few corners → NaN, interpolated).
"""
from __future__ import annotations

import cv2
import numpy as np

FY, CAM_H, V_H = 1607.7, 1.45, 444.0


FX, CX = 1594.7, 654.3
BX0, BX1, BZ0, BZ1, BD = -5.0, 5.0, 6.5, 26.0, 0.05     # BEV road patch (m), 5 cm cells


class EgoFlow:
    """Ground-plane BEV + phase correlation. Ego forward motion d (m) is a pure
    row shift of the metric BEV between consecutive frames (lane lines along
    the motion carry no signal but cannot bias it either — no aperture problem)."""

    def __init__(self, fps, img_h=960, img_w=1280):
        self.fps = fps
        xs = BX0 + (np.arange(int((BX1 - BX0) / BD)) + 0.5) * BD
        zs = BZ1 - (np.arange(int((BZ1 - BZ0) / BD)) + 0.5) * BD      # row 0 = far
        X, Z = np.meshgrid(xs, zs)
        self.mu = (CX + FX * X / Z).astype(np.float32)
        self.mv = (V_H + FY * CAM_H / Z).astype(np.float32)
        self.win = cv2.createHanningWindow((len(xs), len(zs)), cv2.CV_32F)
        self.hist = []
        self.resp = 0.0

    def _bev(self, gray, boxes):
        b = cv2.remap(gray, self.mu, self.mv, cv2.INTER_LINEAR).astype(np.float32)
        b = b - cv2.GaussianBlur(b, (0, 0), 6)                # high-pass: texture, not shading
        if boxes:
            m = np.zeros(gray.shape, np.uint8)
            for x1, y1, x2, y2 in boxes:
                m[max(0, int(y1)):int(y2) + 20, max(0, int(x1) - 8):int(x2) + 8] = 1
            mb = cv2.remap(m, self.mu, self.mv, cv2.INTER_NEAREST)
            b[mb > 0] = 0.0
        return b

    GAP = 3          # frames between correlated BEVs: 0.6 m/s .. 43 m/s at 36 fps

    def _corr(self, a, b):
        """Phase-correlation surface of b vs a, fftshifted (centre = zero shift)."""
        A = np.fft.rfft2(a * self.win)
        B = np.fft.rfft2(b * self.win)
        X = B * np.conj(A)
        X /= np.abs(X) + 1e-6
        return np.fft.fftshift(np.fft.irfft2(X, s=a.shape))

    def step(self, gray, boxes):
        """gray: full-res uint8 frame, boxes: list of [x1,y1,x2,y2]. → m/s or nan.

        stopped  : BEV(t) ≈ BEV(t−GAP) (NCC > 0.8)
        otherwise: strongest phase-correlation peak with ≥ 1 row forward shift,
                   |lateral| ≤ 0.6 m (static peak from hood/reflections excluded)."""
        cur = self._bev(gray, boxes)
        self.hist.append(cur)
        if len(self.hist) > self.GAP + 1:
            self.hist.pop(0)
        if len(self.hist) <= self.GAP or cur.std() < 0.5:
            return float("nan")
        prev = self.hist[0]
        ncc = float(np.corrcoef(prev.ravel(), cur.ravel())[0, 1])
        if ncc > 0.8:
            self.resp = ncc
            return 0.0
        S = self._corr(prev, cur)
        H, W = S.shape
        cy, cx = H // 2, W // 2
        lat = int(0.6 / BD)
        band = S[:, cx - lat: cx + lat + 1].copy()
        band[cy - 1: cy + 2, :] = -1.0          # |dy| < 1.5 rows excluded
        band[:cy, :] = -1.0                      # forward motion only (rows move toward camera)
        r, c = np.unravel_index(int(np.argmax(band)), band.shape)
        self.resp = float(band[r, c])
        if self.resp < 0.02:
            return float("nan")
        dy = float(r - cy)
        if 0 < r < H - 1:                        # sub-row parabola
            y0, y1, y2 = S[r - 1, cx - lat + c], S[r, cx - lat + c], S[r + 1, cx - lat + c]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-9:
                dy += 0.5 * (y0 - y2) / den
        return dy * BD * self.fps / self.GAP


class Divergence:
    """Moving / stopped evidence from static structure above the horizon.

    Forward ego motion makes world-static features expand radially from the FOE;
    turning moves them sideways. Median over LK features outside detected boxes
    (other traffic is a minority) → (radial rate ×1e3 per GAP frames, median |flow| px).
    Stopped ⇔ both ≈ 0. Non-metric; scaled per scene against dash/BEV speed."""
    GAP = 6

    def __init__(self, foe=(654.3, V_H)):
        self.foe = np.array(foe, np.float32)
        self.hist = []

    def step(self, gray, boxes):
        self.hist.append(gray)
        if len(self.hist) > self.GAP + 1:
            self.hist.pop(0)
        if len(self.hist) <= self.GAP:
            return float("nan"), float("nan")
        A, B = self.hist[0], gray
        m = np.zeros_like(A)
        m[60:int(V_H + 76), :] = 255
        for x1, y1, x2, y2 in boxes:
            m[max(0, int(y1) - 10):int(y2) + 10, max(0, int(x1) - 10):int(x2) + 10] = 0
        p0 = cv2.goodFeaturesToTrack(A, 400, 0.01, 10, mask=m)
        if p0 is None or len(p0) < 20:
            return float("nan"), float("nan")
        p1, st, _ = cv2.calcOpticalFlowPyrLK(A, B, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st[:, 0] == 1
        p0, p1 = p0[ok, 0], p1[ok, 0]
        r = p0 - self.foe
        rn = np.linalg.norm(r, axis=1)
        sel = rn > 150
        if sel.sum() < 15:
            return float("nan"), float("nan")
        rad = ((p1 - p0)[sel] * r[sel]).sum(1) / rn[sel] ** 2
        return float(np.median(rad)) * 1e3, float(np.median(np.linalg.norm(p1 - p0, axis=1)))


def smooth_speed(raw, fps, fallback=None):
    """NaN-robust: median over 0.5 s, fallback / gap fill, Gaussian σ = 0.25 s."""
    x = np.asarray(raw, float)
    n = len(x)
    w = max(3, int(0.25 * fps))
    med = np.full(n, np.nan)
    for i in range(n):
        seg = x[max(0, i - w): i + w + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 3:
            med[i] = np.median(seg)
    if fallback is not None:
        fb = np.asarray(fallback, float)
        med = np.where(np.isfinite(med), med, fb)
    idx = np.arange(n)
    ok = np.isfinite(med)
    if ok.sum() == 0:
        return np.zeros(n)
    med = np.interp(idx, idx[ok], med[ok])
    sg = 0.25 * fps
    r = 3 * int(sg)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sg) ** 2)
    k /= k.sum()
    pad = np.pad(med, r, mode="edge")
    return np.clip(np.convolve(pad, k, mode="valid")[:n], 0.0, 60.0)


if __name__ == "__main__":      # quick check: python3 ego_flow.py VIDEO [every]
    import sys
    cap = cv2.VideoCapture(sys.argv[1])
    fps = cap.get(cv2.CAP_PROP_FPS)
    E = EgoFlow(fps)
    import json, re
    sc = re.search(r"scene(\d+)", sys.argv[1]).group(1)
    D = json.load(open(f"phase2_output/scene{sc}/detections.json"))["frames"]
    vs, rs = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i = len(vs)
        bx = [o["bbox_2d"] for o in D[i]["objects"]] if i < len(D) else []
        vs.append(E.step(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), bx))
        rs.append(E.resp)
    sm = smooth_speed(vs, fps)
    ev = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    print("nan frac", np.mean(~np.isfinite(vs)), "resp med", np.median(rs))
    print("raw", [round(float(vs[i]), 1) for i in range(1, len(vs), ev)])
    print([round(float(sm[i]), 1) for i in range(0, len(sm), ev)])
