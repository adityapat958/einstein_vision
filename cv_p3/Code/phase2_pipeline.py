#!/usr/bin/env python3
"""
Phase 2 unified inference pipeline.

For each scene (1–13) this script runs:
  1. YOLO-3D detection + depth estimation → 3D object positions
  2. Vehicle sub-classification (sedan / SUV / hatchback / pickup / …)
  3. Traffic-light arrow & state classification
  4. Ground arrow detection (road markings)
  5. Speed limit sign OCR
  6. Pedestrian pose estimation (17-point COCO skeleton)
  7. Brake light & turn-signal intent detection
  8. Export per-scene detections.json  (schema v2.0)

Outputs go to:
    phase2_output/sceneN/detections.json

Lane detection JSONs are produced separately by run_lane_export.sbatch and
live at:
    cv_p3/lane_out/sceneN/lane_report_style.json

Usage:
    python phase2_pipeline.py --scene 1
    python phase2_pipeline.py --scene all

    # With explicit paths:
    python phase2_pipeline.py \
        --scene 1 \
        --weights /path/to/yolo11n.pt \
        --calib-file /path/to/calibration.mat \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── Repo root on sys.path ───────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent          # …/cv/p3
YOLO3D_DIR = REPO / "cv_p3" / "YOLO-3D"
CODE_DIR = REPO / "cv_p3" / "Code"
for d in (str(YOLO3D_DIR), str(CODE_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator
from load_camera_params import load_camera_params, apply_camera_params_to_estimator

from vehicle_subclassifier import VehicleSubClassifier
from traffic_light_classifier import TrafficLightClassifier
from road_sign_detector import GroundArrowDetector, SpeedLimitOCR
from pose_estimator import PifPafPoseEstimator


# ── Scene video map ─────────────────────────────────────────────────────────
SCENE_VIDEOS = {
    1:  "2023-02-14_11-04-07-front_undistort.mp4",
    2:  "2023-03-03_10-31-11-front_undistort.mp4",
    3:  "2023-02-14_11-49-54-front_undistort.mp4",
    4:  "2023-02-14_11-51-54-front_undistort.mp4",
    5:  "2023-02-14_11-56-56-front_undistort.mp4",
    6:  "2023-03-03_15-31-56-front_undistort.mp4",
    7:  "2023-03-03_11-21-43-front_undistort.mp4",
    8:  "2023-03-03_11-40-47-front_undistort.mp4",
    9:  "2023-03-04_17-20-36-front_undistort.mp4",
    10: "2023-03-06_19-48-30-front_undistort.mp4",
    11: "2023-03-11_17-19-53-front_undistort.mp4",
    12: "2023-03-13_06-00-16-front_undistort.mp4",
    13: "2023-03-03_06-59-50-front_undistort.mp4",
}

# ── Intent helpers ───────────────────────────────────────────────────────────
# HSV ranges for brake-light red  (bright red in rear crop)
_BRAKE_RED_RANGES = [
    (np.array([0, 120, 120]), np.array([10, 255, 255])),
    (np.array([170, 120, 120]), np.array([180, 255, 255])),
]
_TURN_ORANGE_LO = np.array([10, 100, 120])
_TURN_ORANGE_HI = np.array([25, 255, 255])


def _detect_brake_light(frame_bgr: np.ndarray, bbox: list) -> bool:
    """True if rear crop of the bbox has significant red pixels."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    split = y1 + int((y2 - y1) * 0.70)   # bottom 30 %
    crop = frame_bgr[split:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total = crop.shape[0] * crop.shape[1]
    red = sum(
        int(np.count_nonzero(cv2.inRange(hsv, lo, hi)))
        for lo, hi in _BRAKE_RED_RANGES
    )
    return (red / total) > 0.15


def _detect_turn_signal(
    frame_bgr: np.ndarray,
    bbox: list,
    history: deque,
) -> str:
    """
    Detect turn signal by checking for orange on left/right thirds and
    watching for a blinking pattern in the history deque.

    history: deque of recent per-side orange ratios for blink detection.
    Returns "left", "right", or "none".
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]
    w = x2 - x1
    if w < 10:
        return "none"
    third = w // 3
    left_crop  = frame_bgr[y1:y2, x1:x1 + third]
    right_crop = frame_bgr[y1:y2, x2 - third:x2]

    def orange_ratio(crop):
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _TURN_ORANGE_LO, _TURN_ORANGE_HI)
        return float(np.count_nonzero(mask)) / mask.size

    lr = orange_ratio(left_crop)
    rr = orange_ratio(right_crop)

    history.append((lr, rr))
    if len(history) < 3:
        return "none"

    # Blink: variance across recent frames > threshold
    lefts  = [h[0] for h in history]
    rights = [h[1] for h in history]
    if np.var(lefts)  > 0.002 and np.mean(lefts)  > 0.05:
        return "left"
    if np.var(rights) > 0.002 and np.mean(rights) > 0.05:
        return "right"
    return "none"


# ── Motion state ─────────────────────────────────────────────────────────────

def _compute_motion_state(
    position_xyz: list,
    history: deque,
    fps: float,
) -> str:
    history.append(position_xyz[:])
    if len(history) < 2:
        return "unknown"
    prev = history[-2]
    dx = position_xyz[0] - prev[0]
    dz = position_xyz[2] - prev[2]
    speed_ms = math.sqrt(dx * dx + dz * dz) * fps
    if speed_ms < 0.3:
        return "parked"
    if speed_ms < 2.0:
        return "slow"
    return "moving"


# ── Core scene processing ─────────────────────────────────────────────────────

def process_scene(scene_num: int, args) -> None:
    data_dir = REPO / "cv_p3" / "P3Data" / "Sequences" / f"scene{scene_num}" / "Undist"
    video_name = SCENE_VIDEOS.get(scene_num)
    if not video_name:
        print(f"[ERROR] Unknown scene number: {scene_num}")
        return
    video_path = data_dir / video_name
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    output_dir = REPO / "phase2_output" / f"scene{scene_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "detections.json"

    print(f"\n{'='*60}")
    print(f"[Phase 2] Scene {scene_num}: {video_path.name}")
    print(f"{'='*60}")

    # ── Initialise models ──────────────────────────────────────────────────
    device = args.device or "cpu"

    detector = ObjectDetector(
        model_size="nano",
        model_path=args.weights,
        conf_thres=0.25,
        iou_thres=0.45,
        device=device,
    )
    class_names = detector.get_class_names()

    depth_estimator = DepthEstimator(model_size=args.depth_model_size, device=device)
    bbox3d = BBox3DEstimator()

    if args.calib_file and Path(args.calib_file).exists():
        cam_params = load_camera_params(args.calib_file)
        if cam_params:
            apply_camera_params_to_estimator(bbox3d, cam_params)
    K = bbox3d.K  # 3×3 intrinsic matrix

    sub_clf     = VehicleSubClassifier(weights_path=args.sub_classifier_weights, device=device)
    tl_clf      = TrafficLightClassifier()
    arrow_det   = GroundArrowDetector()
    speed_ocr   = SpeedLimitOCR()
    pose_est    = PifPafPoseEstimator(device=device) if not args.no_pose else None

    # Per-track state history
    pos_history:   dict[str, deque] = {}
    turn_history:  dict[str, deque] = {}

    # ── Open video ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {w_vid}x{h_vid} @ {fps:.1f}fps, {total} frames")

    frame_records = []
    depth_calibrated = False
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # 1) Detection ─────────────────────────────────────────────────
            _, detections = detector.detect(frame.copy(), track=True)

            # 2) Depth ─────────────────────────────────────────────────────
            depth_map = depth_estimator.estimate_depth(frame)
            if not depth_calibrated:
                depth_estimator.calibrate_metric_scale(
                    depth_map,
                    camera_height_m=args.camera_height,
                    camera_pitch_deg=args.camera_pitch_deg,
                )
                depth_calibrated = True

            # 3) Pose ──────────────────────────────────────────────────────
            poses = pose_est.predict(frame) if pose_est else []

            # 4) Ground arrows ─────────────────────────────────────────────
            ground_arrows = arrow_det.detect(frame)

            # 5) Per-detection enrichment ──────────────────────────────────
            objects = []
            active_ids = []

            for det in detections:
                bbox, score, class_id, obj_id = det
                x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
                x2 = min(w_vid - 1, x2)
                y2 = min(h_vid - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                cname = class_names[class_id] if class_id < len(class_names) else str(class_id)
                track_id = str(obj_id) if obj_id is not None else f"{cname}_{frame_idx}"

                # Depth → 3D position
                if cname.lower() == "person":
                    rel_d = depth_estimator.get_depth_at_point(
                        depth_map, int((x1 + x2) / 2), int((y1 + y2) / 2)
                    )
                else:
                    rel_d = depth_estimator.get_depth_in_region(
                        depth_map, [x1, y1, x2, y2], method="median"
                    )
                depth_m = depth_estimator.to_metric(rel_d or 0.0)
                cx_img = (x1 + x2) / 2.0
                cy_img = (y1 + y2) / 2.0
                X = (cx_img - K[0, 2]) * depth_m / K[0, 0]
                Y = (cy_img - K[1, 2]) * depth_m / K[1, 1]
                Z = depth_m
                yaw = math.atan2(X, Z)

                # Sub-classification
                sub_class = None
                if cname.lower() in ("car", "truck", "bus", "bicycle", "motorcycle"):
                    sub_class = sub_clf.predict(frame, [x1, y1, x2, y2], cname)

                # Traffic light arrow
                arrow_dir = None
                tl_state = None
                if "traffic light" in cname.lower():
                    tl_result = tl_clf.predict(frame, [x1, y1, x2, y2])
                    tl_state = tl_result["state"]
                    arrow_dir = tl_result["arrow"]

                # Speed limit OCR
                speed_val = None
                if "speed" in cname.lower() or "sign" in cname.lower():
                    speed_val = speed_ocr.extract_number(frame, [x1, y1, x2, y2])

                # Brake lights
                brake = _detect_brake_light(frame, [x1, y1, x2, y2])

                # Turn signal
                if track_id not in turn_history:
                    turn_history[track_id] = deque(maxlen=10)
                turn = _detect_turn_signal(frame, [x1, y1, x2, y2], turn_history[track_id])

                # Motion state
                pos = [X, Y, Z]
                if track_id not in pos_history:
                    pos_history[track_id] = deque(maxlen=10)
                motion_state = _compute_motion_state(pos, pos_history[track_id], fps)

                if obj_id is not None:
                    active_ids.append(obj_id)

                objects.append({
                    "object_id": track_id,
                    "class_name": cname,
                    "sub_class": sub_class,
                    "confidence": round(float(score), 4),
                    "bbox_2d": [x1, y1, x2, y2],
                    "position_xyz_m": [round(X, 3), round(Y, 3), round(Z, 3)],
                    "orientation_yaw_rad": round(yaw, 4),
                    "dimensions_m": None,
                    "motion_state": motion_state,
                    "intent": {
                        "brake_lights_on": brake,
                        "turn_signal": turn,
                    },
                    "traffic_light_state": tl_state,
                    "arrow_direction": arrow_dir,
                    "speed_limit_value": speed_val,
                })

            try:
                bbox3d.cleanup_trackers(active_ids)
            except Exception:
                pass

            # 6) Pose keypoints (image coords + 3D lift) ───────────────────
            pose_records = []
            for pose in poses:
                kps_3d = []
                for (u, v), sc in zip(pose.keypoints_xy, pose.scores):
                    if sc < 0.1:
                        kps_3d.append(None)
                        continue
                    rel_d = depth_estimator.get_depth_at_point(depth_map, int(u), int(v))
                    dm = depth_estimator.to_metric(rel_d or 0.0)
                    kx = (u - K[0, 2]) * dm / K[0, 0]
                    ky = (v - K[1, 2]) * dm / K[1, 1]
                    kps_3d.append([round(kx, 3), round(ky, 3), round(dm, 3)])
                pose_records.append({
                    "person_track_id": pose.person_track_id,
                    "keypoints_2d": pose.keypoints_xy.tolist(),
                    "keypoints_scores": pose.scores.tolist(),
                    "keypoints_3d": kps_3d,
                    "bbox": pose.bbox,
                })

            frame_records.append({
                "frame_index": frame_idx,
                "timestamp_s": round(frame_idx / fps, 4),
                "objects": objects,
                "poses": pose_records,
                "ground_arrows": ground_arrows,
            })

        except Exception as e:
            print(f"[WARN] Frame {frame_idx}: {e}")

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"[INFO]  {frame_idx}/{total} frames  ({frame_idx/elapsed:.1f} fps)")

    cap.release()
    elapsed = time.time() - t0
    print(f"[INFO] Scene {scene_num} done: {frame_idx} frames in {elapsed:.1f}s")

    # ── Export JSON ────────────────────────────────────────────────────────
    scene_data = {
        "schema_version": "2.0",
        "scene": scene_num,
        "input_video": str(video_path),
        "fps": fps,
        "total_frames": frame_idx,
        "resolution": [w_vid, h_vid],
        "frames": frame_records,
    }
    with open(json_path, "w") as f:
        json.dump(scene_data, f, indent=2)
    print(f"[INFO] Saved: {json_path}  ({json_path.stat().st_size / 1e6:.1f} MB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EinsteinVision Phase 2 pipeline")
    p.add_argument(
        "--scene", required=True,
        help='Scene number (1–13) or "all"',
    )
    p.add_argument(
        "--weights",
        default=str(REPO / "cv_p3" / "ext_models" / "yolo11n.pt"),
        help="YOLO weights path",
    )
    p.add_argument(
        "--sub-classifier-weights",
        default=None,
        help="Vehicle sub-class EfficientNet weights (.pth); omit for heuristic mode",
    )
    p.add_argument(
        "--calib-file",
        default=str(REPO / "cv_p3" / "calibration.mat"),
        help="Camera calibration file (.mat or .json)",
    )
    p.add_argument("--depth-model-size", default="small", choices=["small", "base", "large"])
    p.add_argument("--device", default=None)
    p.add_argument("--camera-height", type=float, default=1.45)
    p.add_argument("--camera-pitch-deg", type=float, default=5.0)
    p.add_argument("--no-pose", action="store_true", help="Skip pose estimation")
    return p.parse_args()


def main():
    args = parse_args()

    if args.scene.lower() == "all":
        scenes = list(range(1, 14))
    else:
        scenes = [int(args.scene)]

    for s in scenes:
        process_scene(s, args)

    print("\n[Phase 2] All scenes complete.")


if __name__ == "__main__":
    main()
