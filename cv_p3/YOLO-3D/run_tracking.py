#!/usr/bin/env python3
import os
import time
import json
import cv2
import numpy as np
import torch
from pathlib import Path

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator
from load_camera_params import load_camera_params, apply_camera_params_to_estimator


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(prev, cur, alpha):
    return alpha * cur + (1.0 - alpha) * prev


def map_class_to_asset(class_name: str):
    c = class_name.lower()
    if "car" in c:
        return "car"
    if "truck" in c or "bus" in c:
        return "truck"
    if "motorcycle" in c:
        return "motorcycle"
    if "bicycle" in c:
        return "bicycle"
    if "person" in c:
        return "pedestrian"
    if "traffic light" in c:
        return "traffic_light"
    if "stop sign" in c or c == "sign":
        return "stop_sign"
    return None


def is_phase1_relevant(class_name: str):
    c = class_name.lower()
    allowed = [
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "person",
        "traffic light",
        "stop sign",
    ]
    return any(a in c for a in allowed)


def estimate_world_coordinates(x1, y1, x2, y2, depth_value, width, height, fx=None, fy=None, cx=None, cy=None):
    """
    Approximate ego-centric coordinates for Tesla-style dashboard rendering.
    x_world: lateral left/right
    y_world: forward distance
    z_world: height above road (kept 0 for vehicles)
    """
    if cx is None:
        cx = width * 0.5
    if cy is None:
        cy = height * 0.5
    if fx is None:
        fx = width * 0.9
    if fy is None:
        fy = height * 0.9

    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)

    Z = float(depth_value)
    if not np.isfinite(Z) or Z <= 0:
        Z = 0.0

    X = ((u - cx) / fx) * Z
    Y = Z

    return float(X), float(Y), 0.0


def estimate_scale_from_bbox(x1, y1, x2, y2, class_name):
    h = max(1.0, float(y2 - y1))
    c = class_name.lower()

    if "truck" in c or "bus" in c:
        base = 1.4
    elif "motorcycle" in c:
        base = 0.7
    elif "bicycle" in c:
        base = 0.6
    elif "person" in c:
        base = 0.8
    else:
        base = 1.0

    return float(base)


def main():
    # =========================================================
    # CONFIG
    # =========================================================
    source = "/home/alien/cv_p3/scene11/Undist/pedestrian_presence.mp4"

    output_dir = "/home/alien/cv_p3/scene11/tesla_dashboard_phase1"
    debug_video_path = os.path.join(output_dir, "debug_tracking.mp4")
    scene_json_path = os.path.join(output_dir, "scene_states.json")

    yolo_weights_path = "/home/alien/cv_p3/ext_models/yolo11n.pt"

    yolo_model_size = "Large"
    depth_model_size = "small"
    device = "cpu"

    conf_threshold = 0.25
    iou_threshold = 0.45
    classes = None

    enable_tracking = True
    show_window = False
    save_debug_video = True

    camera_params_file = None

    # smoothing
    alpha_pos = 0.25
    alpha_scale = 0.20
    alpha_yaw = 0.20
    max_missing_frames = 5

    # keep only these first for clean Tesla-like Phase 1
    spawn_only_vehicle_like = True
    # =========================================================

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(yolo_weights_path):
        raise FileNotFoundError(f"YOLO weights file not found: {yolo_weights_path}")

    print(f"[INFO] Using device: {device}")
    print("[INFO] Initializing models...")

    # detector
    try:
        detector = ObjectDetector(
            model_size=yolo_model_size,
            model_path=yolo_weights_path,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device,
        )
    except Exception as e:
        print(f"[WARN] Detector init failed on {device}: {e}")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            model_path=yolo_weights_path,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device="cpu",
        )

    # depth
    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device,
        )
    except Exception as e:
        print(f"[WARN] Depth init failed on {device}: {e}")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device="cpu",
        )

    bbox3d_estimator = BBox3DEstimator()

    cam_params = None
    if camera_params_file is not None and os.path.isfile(camera_params_file):
        try:
            cam_params = load_camera_params(camera_params_file)
            apply_camera_params_to_estimator(bbox3d_estimator, cam_params)
            print(f"[INFO] Loaded camera parameters from: {camera_params_file}")
        except Exception as e:
            print(f"[WARN] Failed to load camera parameters: {e}")
            cam_params = None
    else:
        print("[INFO] No camera parameter file provided. Using approximate intrinsics.")

    print(f"[INFO] Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 1280)
    height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)

    print(f"[INFO] Video size: {width}x{height}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Total frames: {total_frames}")

    debug_writer = None
    if save_debug_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(debug_video_path, fourcc, fps, (width, height))

    frame_count = 0
    proc_start_time = time.time()

    # track memory for smoothing and persistence
    track_memory = {}
    all_scene_states = []

    class_names = detector.get_class_names()

    fx = cam_params["fx"] if cam_params and "fx" in cam_params else width * 0.9
    fy = cam_params["fy"] if cam_params and "fy" in cam_params else height * 0.9
    cx = cam_params["cx"] if cam_params and "cx" in cam_params else width * 0.5
    cy = cam_params["cy"] if cam_params and "cy" in cam_params else height * 0.5

    print("[INFO] Starting processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp_sec = (frame_count - 1) / fps
        debug_frame = frame.copy()

        try:
            # 1) detect
            try:
                _, detections = detector.detect(frame.copy(), track=enable_tracking)
            except Exception as e:
                print(f"[WARN] Frame {frame_count}: detection failed: {e}")
                detections = []

            # 2) depth
            try:
                depth_map = depth_estimator.estimate_depth(frame)
            except Exception as e:
                print(f"[WARN] Frame {frame_count}: depth failed: {e}")
                depth_map = np.zeros((height, width), dtype=np.float32)

            frame_objects = []
            seen_track_ids = set()

            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    if obj_id is None:
                        continue

                    x1, y1, x2, y2 = map(int, bbox)
                    x1 = clamp(x1, 0, width - 1)
                    y1 = clamp(y1, 0, height - 1)
                    x2 = clamp(x2, 0, width - 1)
                    y2 = clamp(y2, 0, height - 1)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
                    if not is_phase1_relevant(class_name):
                        continue

                    asset_class = map_class_to_asset(class_name)
                    if asset_class is None:
                        continue

                    if spawn_only_vehicle_like and asset_class not in ["car", "truck", "motorcycle", "bicycle"]:
                        continue

                    center_x = int((x1 + x2) * 0.5)
                    center_y = int((y1 + y2) * 0.5)

                    try:
                        depth_value = depth_estimator.get_depth_in_region(
                            depth_map, [x1, y1, x2, y2], method="median"
                        )
                    except Exception:
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)

                    if depth_value is None or not np.isfinite(depth_value):
                        continue

                    x_world, y_world, z_world = estimate_world_coordinates(
                        x1, y1, x2, y2, depth_value, width, height, fx, fy, cx, cy
                    )

                    scale = estimate_scale_from_bbox(x1, y1, x2, y2, class_name)

                    # simple yaw approximation from horizontal offset
                    yaw = float(np.clip((center_x - cx) / max(cx, 1.0), -1.0, 1.0) * 0.15)

                    # smoothing
                    if obj_id in track_memory:
                        prev = track_memory[obj_id]
                        x_world = lerp(prev["x_world"], x_world, alpha_pos)
                        y_world = lerp(prev["y_world"], y_world, alpha_pos)
                        z_world = lerp(prev["z_world"], z_world, alpha_pos)
                        scale = lerp(prev["scale"], scale, alpha_scale)
                        yaw = lerp(prev["yaw"], yaw, alpha_yaw)
                    else:
                        prev = None

                    track_memory[obj_id] = {
                        "x_world": x_world,
                        "y_world": y_world,
                        "z_world": z_world,
                        "scale": scale,
                        "yaw": yaw,
                        "class_name": class_name,
                        "asset_class": asset_class,
                        "score": float(score),
                        "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                        "last_seen": frame_count,
                        "missing_count": 0,
                    }

                    seen_track_ids.add(obj_id)

                    frame_objects.append({
                        "track_id": int(obj_id),
                        "class_name": class_name,
                        "asset_class": asset_class,
                        "score": float(score),
                        "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                        "depth_value": float(depth_value),
                        "x_world": float(x_world),
                        "y_world": float(y_world),
                        "z_world": float(z_world),
                        "yaw": float(yaw),
                        "scale": float(scale),
                        "visible": True,
                    })

                except Exception as e:
                    print(f"[WARN] Frame {frame_count}: post-process failed: {e}")
                    continue

            # persistence for briefly missing tracks
            to_delete = []
            for obj_id, state in track_memory.items():
                if obj_id not in seen_track_ids:
                    state["missing_count"] += 1
                    if state["missing_count"] <= max_missing_frames:
                        frame_objects.append({
                            "track_id": int(obj_id),
                            "class_name": state["class_name"],
                            "asset_class": state["asset_class"],
                            "score": float(state["score"]),
                            "bbox_2d": state["bbox_2d"],
                            "depth_value": float(state["y_world"]),
                            "x_world": float(state["x_world"]),
                            "y_world": float(state["y_world"]),
                            "z_world": float(state["z_world"]),
                            "yaw": float(state["yaw"]),
                            "scale": float(state["scale"]),
                            "visible": True,
                        })
                    else:
                        to_delete.append(obj_id)
                else:
                    state["missing_count"] = 0

            for obj_id in to_delete:
                del track_memory[obj_id]

            # sort near-to-far if useful for blender or debug
            frame_objects.sort(key=lambda o: o["y_world"])

            frame_state = {
                "frame_id": frame_count,
                "timestamp_sec": float(timestamp_sec),
                "image_width": int(width),
                "image_height": int(height),
                "objects": frame_objects,
            }
            all_scene_states.append(frame_state)

            # optional debug view
            if debug_writer is not None or show_window:
                for obj in frame_objects:
                    x1, y1, x2, y2 = obj["bbox_2d"]
                    label = f'{obj["asset_class"]} ID:{obj["track_id"]} D:{obj["y_world"]:.1f}m'
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        debug_frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    debug_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                if debug_writer is not None:
                    debug_writer.write(debug_frame)

                if show_window:
                    cv2.imshow("Debug Tracking", debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [27, ord("q")]:
                        break

            if frame_count % 25 == 0:
                print(f"[INFO] Processed {frame_count}/{total_frames}")

        except Exception as e:
            print(f"[WARN] Frame {frame_count}: unexpected error: {e}")
            continue

    cap.release()
    if debug_writer is not None:
        debug_writer.release()
    cv2.destroyAllWindows()

    with open(scene_json_path, "w") as f:
        json.dump({
            "source_video": source,
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "total_frames": int(frame_count),
            "frames": all_scene_states,
        }, f, indent=2)

    total_time = time.time() - proc_start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0.0

    print("[INFO] Done.")
    print(f"[INFO] Scene JSON saved to: {scene_json_path}")
    if save_debug_video:
        print(f"[INFO] Debug video saved to: {debug_video_path}")
    print(f"[INFO] Avg FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()