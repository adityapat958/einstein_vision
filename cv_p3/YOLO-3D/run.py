#!/usr/bin/env python3
"""
Phase 2 YOLO-3D inference script.

Runs YOLOv11 detection + Depth Anything V2 depth estimation + optional
3D bbox projection on a single video and exports:
  - An annotated overlay MP4 (--output-video)
  - A per-frame detections JSON (--output-json) for Blender rendering

Usage:
    python run.py \
        --input  /path/to/video.mp4 \
        --weights /path/to/yolo11n.pt \
        --output-video /path/to/out.mp4 \
        --output-json  /path/to/detections.json \
        --calib-file   /path/to/camera_params.json \
        --depth-model-size small \
        --device cuda
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

if (
    hasattr(torch, "backends")
    and hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator, export_params_to_json


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 YOLO-3D inference")
    p.add_argument("--input", required=True, help="Path to input video")
    p.add_argument(
        "--weights",
        default=str(
            Path(__file__).parent.parent / "ext_models" / "yolo11n.pt"
        ),
        help="Path to YOLO weights file",
    )
    p.add_argument("--output-video", default=None, help="Path for overlay MP4 output")
    p.add_argument("--output-json", default=None, help="Path for per-frame detections JSON")
    p.add_argument(
        "--calib-file",
        default=None,
        help="Camera parameters file (.json or .mat). If a .mat is given and "
             "--export-calib-json is set, the parsed params are also written "
             "to that JSON path for reuse.",
    )
    p.add_argument(
        "--export-calib-json",
        default=None,
        help="If set and --calib-file is a .mat, export parsed params to this JSON path.",
    )
    p.add_argument(
        "--depth-model-size",
        default="small",
        choices=["small", "base", "large"],
        help="Depth Anything V2 model size",
    )
    p.add_argument("--conf-thresh", type=float, default=0.25)
    p.add_argument("--iou-thresh", type=float, default=0.45)
    p.add_argument("--device", default=None, help="cuda / cpu / mps (auto-detected if omitted)")
    p.add_argument("--camera-height", type=float, default=1.45, help="Camera height above ground in metres")
    p.add_argument("--camera-pitch-deg", type=float, default=5.0, help="Camera downward pitch in degrees")
    p.add_argument("--no-bev", action="store_true", help="Disable Bird's Eye View overlay")
    p.add_argument("--no-tracking", action="store_true", help="Disable object tracking")
    return p.parse_args()


def _derive_output_paths(input_path: str, args):
    """Fill in default output paths if the user didn't specify them."""
    stem = Path(input_path).stem
    parent = Path(input_path).parent
    video_out = args.output_video or str(parent / f"{stem}_overlay.mp4")
    json_out = args.output_json or str(parent / f"{stem}_detections.json")
    return video_out, json_out


def main():
    args = parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps"
        else:
            device = "cpu"

    output_path, json_path = _derive_output_paths(args.input, args)

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"YOLO weights not found: {args.weights}")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Input:  {args.input}")
    print(f"[INFO] Output video: {output_path}")
    print(f"[INFO] Output JSON:  {json_path}")

    # ── Models ──────────────────────────────────────────────────────────────
    detector = ObjectDetector(
        model_size="nano",
        model_path=args.weights,
        conf_thres=args.conf_thresh,
        iou_thres=args.iou_thresh,
        device=device,
    )

    depth_estimator = DepthEstimator(model_size=args.depth_model_size, device=device)

    bbox3d_estimator = BBox3DEstimator()

    # Camera calibration
    if args.calib_file and os.path.isfile(args.calib_file):
        cam_params = load_camera_params(args.calib_file)
        if cam_params:
            apply_camera_params_to_estimator(bbox3d_estimator, cam_params)
            # Export to JSON sidecar if requested (useful when input is .mat)
            if args.export_calib_json:
                export_params_to_json(cam_params, args.export_calib_json)
    else:
        cam_params = None
        print("[INFO] No camera params file — using defaults.")

    enable_bev = not args.no_bev
    enable_tracking = not args.no_tracking
    bev = BirdEyeView(scale=60, size=(300, 300)) if enable_bev else None

    # ── Video I/O ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 1280)
    height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)

    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps  ({total_frames} frames)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ── Metric depth scale calibration (once on first frame) ────────────────
    depth_scale_calibrated = False

    # ── Main loop ────────────────────────────────────────────────────────────
    frame_records = []
    frame_count = 0
    proc_start = time.time()
    fps_display = "FPS: --"
    class_names = detector.get_class_names()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        try:
            result_frame = frame.copy()

            # 1) Detection
            try:
                _, detections = detector.detect(frame.copy(), track=enable_tracking)
            except Exception as e:
                print(f"[WARN] Frame {frame_count}: detection failed: {e}")
                detections = []

            # 2) Depth
            try:
                depth_map = depth_estimator.estimate_depth(frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)

                # Calibrate metric scale once on the first successful frame
                if not depth_scale_calibrated:
                    depth_estimator.calibrate_metric_scale(
                        depth_map,
                        camera_height_m=args.camera_height,
                        camera_pitch_deg=args.camera_pitch_deg,
                    )
                    depth_scale_calibrated = True

            except Exception as e:
                print(f"[WARN] Frame {frame_count}: depth failed: {e}")
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)

            # 3) 3D estimation + JSON accumulation
            boxes_3d = []
            active_ids = []
            frame_objects = []

            for det in detections:
                try:
                    bbox, score, class_id, obj_id = det
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    cname = class_names[class_id] if class_id < len(class_names) else str(class_id)

                    # Depth sampling
                    if cname.lower() in ("person", "cat", "dog"):
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        rel_depth = depth_estimator.get_depth_at_point(depth_map, cx, cy)
                    else:
                        rel_depth = depth_estimator.get_depth_in_region(
                            depth_map, [x1, y1, x2, y2], method="median"
                        )
                    if rel_depth is None or np.isnan(rel_depth):
                        rel_depth = 0.0

                    # Convert to metric depth
                    depth_m = depth_estimator.to_metric(rel_depth)

                    box_3d = {
                        "bbox_2d": [x1, y1, x2, y2],
                        "depth_value": float(rel_depth),
                        "depth_method": "center" if cname.lower() == "person" else "median",
                        "class_name": cname,
                        "object_id": obj_id,
                        "score": float(score),
                    }
                    boxes_3d.append(box_3d)
                    if obj_id is not None:
                        active_ids.append(obj_id)

                    # Estimate 3D position via inverse projection
                    K = bbox3d_estimator.K
                    cx_img = (x1 + x2) / 2.0
                    cy_img = (y1 + y2) / 2.0
                    # Back-project: X = (u - cx) * Z / fx
                    X = (cx_img - K[0, 2]) * depth_m / K[0, 0]
                    Y = (cy_img - K[1, 2]) * depth_m / K[1, 1]
                    Z = depth_m

                    # Yaw: coarse estimate from horizontal position relative to image centre
                    yaw = math.atan2(X, Z)

                    frame_objects.append({
                        "object_id": str(obj_id) if obj_id is not None else f"{cname}_{frame_count}",
                        "class_name": cname,
                        "sub_class": None,   # filled by vehicle_subclassifier in phase2_pipeline
                        "confidence": float(score),
                        "bbox_2d": [x1, y1, x2, y2],
                        "position_xyz_m": [round(X, 3), round(Y, 3), round(Z, 3)],
                        "orientation_yaw_rad": round(yaw, 4),
                        "dimensions_m": None,
                        "arrow_direction": None,
                        "speed_limit_value": None,
                    })

                except Exception as e:
                    print(f"[WARN] Frame {frame_count}: detection post-process: {e}")

            try:
                bbox3d_estimator.cleanup_trackers(active_ids)
            except Exception:
                pass

            frame_records.append({
                "frame_index": frame_count - 1,
                "timestamp_s": round((frame_count - 1) / fps, 4),
                "objects": frame_objects,
            })

            # 4) Draw overlays
            for box in boxes_3d:
                try:
                    cn = box["class_name"].lower()
                    if any(k in cn for k in ("car", "vehicle", "truck", "bus")):
                        color = (0, 0, 255)
                    elif "person" in cn:
                        color = (0, 255, 0)
                    elif "bicycle" in cn or "motorcycle" in cn:
                        color = (255, 0, 0)
                    elif "traffic light" in cn:
                        color = (0, 255, 255)
                    elif "sign" in cn:
                        color = (255, 255, 0)
                    else:
                        color = (255, 255, 255)
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box, color=color)
                except Exception:
                    pass

            # 5) BEV overlay
            if enable_bev and bev is not None:
                try:
                    bev.reset()
                    for box in boxes_3d:
                        bev.draw_box(box)
                    bev_img = bev.get_image()
                    bh = height // 4
                    bw = bh
                    result_frame[height - bh:height, 0:bw] = cv2.resize(bev_img, (bw, bh))
                except Exception:
                    pass

            # 6) Depth overlay (top-left)
            try:
                dh = height // 4
                dw = max(1, dh * width // height)
                result_frame[0:dh, 0:dw] = cv2.resize(depth_colored, (dw, dh))
            except Exception:
                pass

            # 7) FPS text
            if frame_count % 10 == 0:
                elapsed = time.time() - proc_start
                if elapsed > 0:
                    fps_display = f"FPS: {frame_count / elapsed:.1f}"
            cv2.putText(
                result_frame,
                f"{fps_display} | {device} | Frame {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            out.write(result_frame)

            if total_frames > 0 and frame_count % 25 == 0:
                print(f"[INFO] {frame_count}/{total_frames} frames")
            elif total_frames == 0 and frame_count % 25 == 0:
                print(f"[INFO] {frame_count} frames processed")

        except Exception as e:
            print(f"[WARN] Frame {frame_count}: unexpected error: {e}")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - proc_start
    print(f"[INFO] Done. {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.2f} fps avg)")
    print(f"[INFO] Video saved: {output_path}")

    # ── Write JSON ────────────────────────────────────────────────────────────
    scene_json = {
        "schema_version": "2.0",
        "input_video": args.input,
        "fps": fps,
        "total_frames": frame_count,
        "depth_model": args.depth_model_size,
        "depth_metric_scale": getattr(depth_estimator, "_metric_scale", None),
        "frames": frame_records,
    }
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(scene_json, f, indent=2)
    print(f"[INFO] Detections JSON saved: {json_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
        cv2.destroyAllWindows()
