"""
Traffic light overlay renderer.

Reads a detections JSON (scene3.json / scene6.json format), re-classifies
every traffic_light bounding box against the source video using
TrafficLightClassifier, and writes an annotated video with:
  - Coloured bounding boxes for all detections
  - Traffic light state label + arrow direction
  - State-coloured circle indicator

Usage:
    python src/pipeline/render_traffic_light_overlay.py \
        --json  src/pipeline_out/scene3.json \
        --out   src/pipeline_out/scene3_tl_overlay.mp4

    # Skip re-classification and use state already in the JSON:
    python src/pipeline/render_traffic_light_overlay.py \
        --json  src/pipeline_out/scene3.json \
        --out   src/pipeline_out/scene3_tl_overlay.mp4 \
        --use-stored-state
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow importing from src/Code without installing the package
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "Code"))
from traffic_light_classifier import TrafficLightClassifier  # noqa: E402

# ── Visual config ─────────────────────────────────────────────────────────────

# Bounding-box colour per subclass (BGR)
_CLASS_COLOR: dict[str, tuple[int, int, int]] = {
    "traffic_light":    (0, 255, 255),   # cyan
    "traffic_pole":     (128, 128, 128), # grey
    "stop_sign":        (0, 0, 255),     # red
    "traffic_cone":     (0, 165, 255),   # orange
    "traffic_cylinder": (0, 165, 255),
    "suv":              (255, 200, 0),   # blue-ish
    "sedan":            (255, 200, 0),
    "hatchback":        (255, 200, 0),
    "pickup_truck":     (255, 200, 0),
    "truck":            (0, 80, 255),
    "pedestrian":       (0, 255, 80),
    "bicycle":          (180, 0, 255),
    "motorcycle":       (180, 0, 255),
    "dustbin":          (100, 100, 100),
}
_DEFAULT_COLOR = (200, 200, 200)

# Traffic light state → indicator colour (BGR)
_STATE_COLOR: dict[str, tuple[int, int, int]] = {
    "red":     (0, 0, 255),
    "yellow":  (0, 200, 255),
    "green":   (0, 220, 0),
    "unknown": (100, 100, 100),
}

_FONT            = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE      = 0.55
_THICKNESS       = 2
_ARROW_FONT_SCALE = 1.4   # bigger arrow/state label on traffic lights
_ARROW_THICKNESS  = 3


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
               color: tuple, label: str) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, _THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _THICKNESS)
    # Label background
    bg_y1 = max(0, y1 - th - 6)
    bg_y2 = y1
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 4, bg_y2), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), _FONT, _FONT_SCALE,
                (0, 0, 0), _THICKNESS, cv2.LINE_AA)


def _draw_traffic_light_info(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                              state: str, arrow: str | None) -> None:
    """Draw state circle + arrow text below the bounding box."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Coloured circle in the centre of the bbox
    radius = max(6, min(18, (x2 - x1) // 4))
    state_color = _STATE_COLOR.get(state, _STATE_COLOR["unknown"])
    cv2.circle(frame, (cx, cy), radius, state_color, -1)
    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)

    # State + arrow label below bbox — larger text
    arrow_str = arrow if arrow and arrow not in ("off", "none", None) else ""
    info = state.upper() + (f"  {arrow_str.upper()}" if arrow_str else "")
    (tw, th), _ = cv2.getTextSize(info, _FONT, _ARROW_FONT_SCALE, _ARROW_THICKNESS)
    tx = max(0, cx - tw // 2)
    ty = min(frame.shape[0] - 4, y2 + th + 8)
    cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4),
                  (0, 0, 0), -1)
    cv2.putText(frame, info, (tx, ty), _FONT, _ARROW_FONT_SCALE,
                state_color, _ARROW_THICKNESS, cv2.LINE_AA)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(json_path: Path, out_path: Path, use_stored_state: bool) -> None:
    with open(json_path) as f:
        data = json.load(f)

    video_path = data["video_path"]
    if not Path(video_path).exists():
        # Repo root is 3 levels up from this file (src/pipeline/render_traffic_light_overlay.py)
        repo_root = Path(__file__).resolve().parents[2]
        scene_name = json_path.stem  # e.g. "scene3"
        scene_num  = "".join(filter(str.isdigit, scene_name))
        # Search for front undistorted video for this scene
        patterns = list(repo_root.glob(
            f"P3Data/Sequences/scene{scene_num}/Undist/*front*.mp4"
        ))
        if patterns:
            video_path = str(patterns[0])
        else:
            sys.exit(f"[ERROR] Video not found: {data['video_path']}")

    print(f"[INFO] Video:  {video_path}")
    print(f"[INFO] JSON:   {json_path}")
    print(f"[INFO] Output: {out_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    fps    = data.get("fps", cap.get(cv2.CAP_PROP_FPS))
    width  = int(data.get("width",  cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(data.get("height", cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    clf = TrafficLightClassifier() if not use_stored_state else None

    # Build frame_idx → detections lookup
    frame_map: dict[int, list] = {}
    for fr in data.get("frames", []):
        frame_map[int(fr["frame_idx"])] = fr.get("detections", [])

    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Processing {total} frames …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = frame_map.get(frame_idx, [])

        for det in detections:
            subclass = det.get("subclass", "unknown")
            bbox     = det.get("bbox_xyxy", [])
            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = (int(c) for c in bbox[:4])
            color = _CLASS_COLOR.get(subclass, _DEFAULT_COLOR)
            score = det.get("score", 0.0)
            label = f"{subclass} {score:.2f}"
            _draw_bbox(frame, x1, y1, x2, y2, color, label)

            # Traffic light — classify and draw state
            if subclass == "traffic_light":
                if use_stored_state:
                    tl_info = det.get("traffic_light_info", {})
                    state  = tl_info.get("state", "unknown")
                    arrow  = tl_info.get("arrow", "off")
                else:
                    result = clf.predict(frame, [x1, y1, x2, y2])
                    state  = result["state"]
                    arrow  = result["arrow"]

                _draw_traffic_light_info(frame, x1, y1, x2, y2, state, arrow)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f"  {frame_idx}/{total} frames done")

    cap.release()
    writer.release()
    print(f"[DONE] Written to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic light overlay renderer")
    parser.add_argument("--json",  required=True,  type=Path,
                        help="Path to scene detections JSON")
    parser.add_argument("--out",   required=True,  type=Path,
                        help="Output video path (.mp4)")
    parser.add_argument("--use-stored-state", action="store_true",
                        help="Use traffic_light_info already in JSON (skip re-classification)")
    args = parser.parse_args()
    run(args.json, args.out, args.use_stored_state)


if __name__ == "__main__":
    main()
