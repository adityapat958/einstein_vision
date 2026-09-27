"""Per-track car body colour sampled from the front camera → road/sceneN/car_colors.json.

For every vehicle track: take its K largest, non-truncated boxes, sample the lower body panel
(x 25–75 %, y 55–85 % of the box: below the rear/side windows, above the tyres/road), median in
CIE-Lab, then snap to the paint palette in look.py (clean, readable colours instead of the camera's
washed-out tones). The Tesla camera is very desaturated, so chroma thresholds are low.

    python3 einsteinvision/car_colors.py --scene 1 [--debug road/scene1/car_colors.png]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

VEH = {"car", "truck", "bus"}
PAL = {  # linear RGB, same values as look.PALETTE
    "white": (0.80, 0.80, 0.78), "black": (0.012, 0.012, 0.014), "grey": (0.12, 0.125, 0.13),
    "silver": (0.42, 0.43, 0.45), "blue": (0.02, 0.05, 0.16), "red": (0.32, 0.012, 0.012),
    "brown": (0.10, 0.06, 0.035), "green": (0.03, 0.07, 0.04), "beige": (0.45, 0.38, 0.28),
    "orange": (0.55, 0.16, 0.02), "yellow": (0.65, 0.48, 0.05),
}


def dominant(px, k=3):
    """Largest k-means cluster in Lab = body paint (windows, tyres, plates, lamps are smaller)."""
    px = px[:: max(1, len(px) // 1500)].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab_, cen = cv2.kmeans(px, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    return cen[np.bincount(lab_.ravel(), minlength=k).argmax()]


def classify(L, a, b, Lref):
    """L in 0..100 relative to the scene's road brightness Lref (exposure-normalised)."""
    C = float(np.hypot(a, b))
    h = float(np.degrees(np.arctan2(b, a))) % 360
    Ln = L * 55.0 / max(Lref, 1.0)          # road asphalt ≈ L 55 in a neutral exposure
    if C > 11:
        if h < 45 or h > 330:
            return "red" if Ln < 70 else "orange"
        if h < 75:
            return "brown" if Ln < 50 else ("orange" if C > 25 else "beige")
        if h < 105:
            return "yellow" if C > 25 else "beige"
        if h < 200:
            return "green"
        if h < 300:
            return "blue"
    if Ln > 66:
        return "white"
    if Ln > 50:
        return "silver"
    if Ln > 32:
        return "grey"
    return "black"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--debug", default=None, help="write a swatch sheet PNG")
    a = ap.parse_args()
    d = json.loads(Path(f"phase2_output/scene{a.scene}/detections.json").read_text())
    vid = sorted(Path(f"P3Data/Sequences/scene{a.scene}/Undist").glob("*-front_undistort.mp4"))[0]
    cap = cv2.VideoCapture(str(vid))
    W, H = int(cap.get(3)), int(cap.get(4))

    dets = defaultdict(list)                  # oid → [(area, frame, bbox)]
    for f in d["frames"]:
        for o in f["objects"]:
            if o["class_name"] not in VEH:
                continue
            x1, y1, x2, y2 = o["bbox_2d"]
            if x1 < 4 or y1 < 4 or x2 > W - 4 or y2 > H - 4 or (x2 - x1) < 40:
                continue
            dets[str(o["object_id"])].append(((x2 - x1) * (y2 - y1), f["frame_index"], (x1, y1, x2, y2)))
    want = defaultdict(list)                  # frame → [(oid, bbox)]
    for oid, L in dets.items():
        for _, fi, bb in sorted(L, reverse=True)[:a.k]:
            want[fi].append((oid, bb))
    samples = defaultdict(list)
    road_L = []
    crops = {}
    fi = -1
    for target in sorted(want):
        while fi < target:
            ok, img = cap.read()
            fi += 1
            if not ok:
                break
        if not ok:
            break
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        road_L.append(np.median(lab[int(0.8 * H):int(0.9 * H), int(0.4 * W):int(0.6 * W), 0]) * 100 / 255)
        for oid, (x1, y1, x2, y2) in want[target]:
            w, h = x2 - x1, y2 - y1
            px = lab[int(y1 + 0.30 * h):int(y1 + 0.75 * h), int(x1 + 0.15 * w):int(x1 + 0.85 * w)].reshape(-1, 3)
            if len(px) < 30:
                continue
            L_, A_, B_ = dominant(px)
            samples[oid].append((L_ * 100 / 255, A_ - 128, B_ - 128))
            if oid not in crops or w * h > crops[oid][0]:
                crops[oid] = (w * h, cv2.resize(img[int(y1):int(y2), int(x1):int(x2)], (96, 72)))
    Lref = float(np.median(road_L)) if road_L else 55.0
    out, names, raw = {}, {}, {}
    for oid, S in samples.items():
        L_, A_, B_ = np.median(np.array(S), axis=0)
        nm = classify(L_, A_, B_, Lref)
        out[oid], names[oid] = list(PAL[nm]), nm
        raw[oid] = L_ * 55.0 / max(Lref, 1.0)
    p = Path(f"road/scene{a.scene}/car_colors.json")
    p.write_text(json.dumps(out))
    hist = defaultdict(int)
    for n in names.values():
        hist[n] += 1
    print(f"[colors] scene{a.scene}: {len(out)} tracks, road L {Lref:.0f}, "
          f"{dict(sorted(hist.items(), key=lambda kv: -kv[1]))} → {p}")
    if a.debug:                               # crop | swatch, biggest tracks first
        top = sorted(crops, key=lambda o: -crops[o][0])[:40]
        tiles = []
        for oid in top:
            if oid not in out:
                continue
            sw = np.zeros((72, 48, 3), np.uint8)
            srgb = [255 * (c ** (1 / 2.2)) for c in out[oid]]
            sw[:] = (srgb[2], srgb[1], srgb[0])
            t = np.hstack([crops[oid][1], sw])
            cv2.putText(t, "%s %.0f" % (names[oid][:6], raw[oid]), (2, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(t)
        rows = [np.hstack(tiles[i:i + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[i:i + 8])))
                for i in range(0, len(tiles), 8)]
        if rows:
            cv2.imwrite(a.debug, np.vstack(rows))


if __name__ == "__main__":
    sys.exit(main())
