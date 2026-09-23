"""Re-classify traffic-light state from video crops (replaces broken HSV field).

For every 'traffic light' detection: crop the bbox from the undistorted front
video, find the lit lamp = bright + saturated pixels, classify by hue AND by
vertical position in the housing (top=red, middle=yellow, bottom=green),
then majority-vote per track over a ±10-frame window.

Also estimates 3D head position (ego frame, m) from bbox height (head ≈ 1.0 m).

    python3 einsteinvision/tl_state.py --scene 3
    → road/scene3/traffic_lights.json  {frame_index: [{id, state, conf, x, y, z, bbox}]}
"""
import argparse, glob, json, collections
from pathlib import Path
import cv2
import numpy as np

FX, FY, CX, CY = 1594.7, 1607.7, 654.3, 413.4
CAM_H, V_HORIZON = 1.45, 444.0
HEAD_H = 1.0


def lamp_state(crop):
    if crop.size == 0 or min(crop.shape[:2]) < 6:
        return "unknown", 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    lit = (v >= max(150, np.percentile(v, 90))) & (s >= 70)
    if lit.sum() < 3:
        return "unknown", 0.0
    hh = h[lit]
    ys = np.nonzero(lit)[0] / max(crop.shape[0] - 1, 1)      # 0 top … 1 bottom
    pos = float(np.median(ys))
    green = int(((hh >= 40) & (hh <= 100)).sum())
    warm = int(((hh <= 34) | (hh >= 160)).sum())
    n = green + warm
    if n < 3:
        return "unknown", 0.0
    if green > warm:
        return "green", green / n
    # red LEDs clip to orange (hue 13-18) on this camera → lamp POSITION decides red vs yellow
    tall = crop.shape[0] > 1.5 * crop.shape[1]
    if not tall:
        return ("red" if ((hh <= 8) | (hh >= 160)).sum() > 0.5 * warm else "yellow"), 0.5
    if pos < 0.42:
        return "red", warm / n
    if pos < 0.68:
        return "yellow", warm / n
    return "green", 0.4                       # warm blob in bottom slot: over-exposed green


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--win", type=int, default=10)
    a = ap.parse_args()
    det = json.load(open(f"phase2_output/scene{a.scene}/detections.json"))
    video = glob.glob(f"P3Data/Sequences/scene{a.scene}/Undist/*front_undistort.mp4")[0]
    want = {f["frame_index"]: [o for o in f["objects"] if o["class_name"] == "traffic light"]
            for f in det["frames"]}
    want = {k: v for k, v in want.items() if v}
    cap = cv2.VideoCapture(video)
    raw = collections.defaultdict(list)                 # track → [(fi, state, conf)]
    out = collections.defaultdict(list)
    fi = -1
    for target in sorted(want):
        while fi < target:
            ok, frame = cap.read(); fi += 1
            if not ok:
                break
        for o in want[target]:
            x1, y1, x2, y2 = map(int, o["bbox_2d"])
            st, cf = lamp_state(frame[max(y1, 0):y2, max(x1, 0):x2])
            tid = str(o["object_id"])
            raw[tid].append((target, st, cf))
            hpx = max(y2 - y1, 1)
            Z = FY * HEAD_H / hpx
            X = ((x1 + x2) / 2 - CX) * Z / FX
            Yup = CAM_H - ((y1 + y2) / 2 - V_HORIZON) * Z / FY
            out[target].append(dict(id=tid, bbox=[x1, y1, x2, y2], x=round(X, 2), y=round(Z, 2),
                                    z=round(Yup, 2)))
    # temporal vote per track
    smooth = {}
    for tid, seq in raw.items():
        for i, (f, _, _) in enumerate(seq):
            c = collections.Counter()
            for f2, st2, cf2 in seq:
                if abs(f2 - f) <= a.win and st2 != "unknown":
                    c[st2] += cf2
            smooth[(tid, f)] = (c.most_common(1)[0][0], round(c.most_common(1)[0][1] / max(sum(c.values()), 1e-9), 2)) if c else ("unknown", 0.0)
    for f, lst in out.items():
        for d in lst:
            d["state"], d["conf"] = smooth[(d["id"], f)]
    Path(f"road/scene{a.scene}").mkdir(parents=True, exist_ok=True)
    json.dump({str(k): v for k, v in out.items()}, open(f"road/scene{a.scene}/traffic_lights.json", "w"))
    cnt = collections.Counter(d["state"] for v in out.values() for d in v)
    print(f"[tl] scene{a.scene}: {sum(cnt.values())} light-detections  {dict(cnt)}")


if __name__ == "__main__":
    main()
