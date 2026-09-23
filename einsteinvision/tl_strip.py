"""Contact sheet of traffic-light crops with their classified state (for human check)."""
import argparse, glob, json, random
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--scene", type=int); ap.add_argument("--out"); ap.add_argument("--n", type=int, default=32)
a = ap.parse_args()
tl = json.load(open(f"road/scene{a.scene}/traffic_lights.json"))
items = [(int(f), d) for f, v in tl.items() for d in v if (d["bbox"][3] - d["bbox"][1]) >= 24]
if not items:
    raise SystemExit("no lights")
random.seed(0); items = sorted(random.sample(items, min(a.n, len(items))), key=lambda t: t[0])
cap = cv2.VideoCapture(glob.glob(f"P3Data/Sequences/scene{a.scene}/Undist/*front_undistort.mp4")[0])
col = {"red": (0, 0, 255), "yellow": (0, 200, 255), "green": (0, 255, 0), "unknown": (128, 128, 128)}
tiles, fi = [], -1
for f, d in items:
    while fi < f:
        ok, fr = cap.read(); fi += 1
    x1, y1, x2, y2 = d["bbox"]; p = 6
    c = fr[max(y1 - p, 0):y2 + p, max(x1 - p, 0):x2 + p]
    c = cv2.resize(c, (90, int(90 * c.shape[0] / max(c.shape[1], 1))))[:200]
    t = np.zeros((230, 100, 3), np.uint8); t[:c.shape[0], 5:95] = c
    cv2.putText(t, d["state"][:3], (5, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col[d["state"]], 2)
    cv2.putText(t, str(f), (55, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    tiles.append(t)
while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
print("[tl] strip", a.out)
