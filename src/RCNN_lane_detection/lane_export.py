#!/usr/bin/env python3
import os
import json
import math
import argparse
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm


# =========================================================
# Utility
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_front_camera_intrinsics() -> Dict[str, float]:
    K = {
        "fx": 1594.7,
        "fy": 1607.7,
        "cx": 654.3,
        "cy": 413.4,
    }
    print("[INFO] Using FRONT camera intrinsics")
    print(f"[INFO] fx={K['fx']}, fy={K['fy']}, cx={K['cx']}, cy={K['cy']}")
    return K


# =========================================================
# Model
# =========================================================

def build_model(num_classes: int, checkpoint_path: str, device: torch.device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[INFO] Missing keys: {len(missing)}")
    print(f"[INFO] Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


# =========================================================
# ROI / mask cleanup
# =========================================================

def make_road_roi_mask(h: int, w: int) -> np.ndarray:
    """
    Slightly wider, lower-half road trapezoid for front camera.
    """
    poly = np.array([
        [int(0.03 * w), h - 1],
        [int(0.38 * w), int(0.56 * h)],
        [int(0.62 * w), int(0.56 * h)],
        [int(0.98 * w), h - 1],
    ], dtype=np.int32)

    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi, [poly], 1)
    return roi


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_overlap_ratio(mask: np.ndarray, roi_mask: np.ndarray) -> float:
    denom = float(mask.sum())
    if denom < 1:
        return 0.0
    return float((mask * roi_mask).sum()) / denom


def keep_vertical_components(
    mask: np.ndarray,
    frame_h: int,
    frame_w: int,
    min_component_area: int = 40,
    max_component_width_ratio: float = 0.18,
    min_component_height: int = 30,
) -> np.ndarray:
    """
    Keep multiple plausible lane-like components instead of only the largest one.
    This is much better for dashed lanes.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)

    out = np.zeros_like(mask, dtype=np.uint8)

    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])

        if area < min_component_area:
            continue
        if bh < min_component_height:
            continue
        if (bw / max(1, frame_w)) > max_component_width_ratio:
            continue
        if bh < bw:  # reject horizontal blobs
            continue
        if (y + bh) < int(0.55 * frame_h):
            continue

        out[labels == lab] = 1

    return out


def clean_lane_mask(mask: np.ndarray, roi_mask: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    """
    Better cleanup for lane masks:
    - keep only road ROI
    - light morphology so dashed lanes survive
    - keep multiple vertical components
    """
    m = (mask > 0).astype(np.uint8)
    m = m * roi_mask

    if m.sum() == 0:
        return m

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close_v = np.ones((7, 3), np.uint8)   # vertical preference
    kernel_close_small = np.ones((5, 5), np.uint8)

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close_v)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close_small)

    m = keep_vertical_components(
        m,
        frame_h=frame_h,
        frame_w=frame_w,
        min_component_area=35,
        max_component_width_ratio=0.18,
        min_component_height=28,
    )

    return m


def keep_lane_candidate(
    lane_mask: np.ndarray,
    frame_h: int,
    frame_w: int,
    roi_mask: np.ndarray,
    min_area: int = 80,
    min_height: int = 60,
    max_width_ratio: float = 0.18,
    min_roi_overlap: float = 0.55,
) -> bool:
    area = int(lane_mask.sum())
    if area < min_area:
        return False

    overlap = mask_overlap_ratio(lane_mask, roi_mask)
    if overlap < min_roi_overlap:
        return False

    x1, y1, x2, y2 = mask_to_bbox(lane_mask)
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)

    if bh < min_height:
        return False

    if (bw / frame_w) > max_width_ratio:
        return False

    if y2 < int(0.65 * frame_h):
        return False

    if bh < 1.2 * bw:
        return False

    return True


# =========================================================
# Lane fitting
# =========================================================

def fit_lane_polynomial_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit x = a*y^2 + b*y + c using row-wise robust centers.
    Uses medians per row and emphasizes lower rows.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return None

    y_min = int(np.percentile(ys, 15))
    y_max = int(np.percentile(ys, 100))

    row_y = []
    row_x = []
    row_w = []

    for y in range(y_min, y_max + 1):
        x_row = xs[ys == y]
        if x_row.size >= 2:
            # robust center instead of mean
            x_center = float(np.median(x_row))
            row_y.append(float(y))
            row_x.append(x_center)

            # heavier weight for lower image region
            weight = 1.0 + 2.0 * ((y - y_min) / max(1.0, (y_max - y_min)))
            row_w.append(weight)

    if len(row_y) < 12:
        return None

    y_fit = np.array(row_y, dtype=np.float32)
    x_fit = np.array(row_x, dtype=np.float32)
    w_fit = np.array(row_w, dtype=np.float32)

    try:
        coeffs = np.polyfit(y_fit, x_fit, deg=2, w=w_fit)
    except Exception:
        return None

    return coeffs.astype(np.float32)


def sample_equidistant_points_on_curve(
    coeffs: np.ndarray,
    y_min: float,
    y_max: float,
    num_points: int = 10,
    dense_samples: int = 500,
) -> List[Tuple[float, float]]:
    a, b, c = coeffs.tolist()
    ys = np.linspace(y_min, y_max, dense_samples, dtype=np.float32)
    xs = a * ys * ys + b * ys + c

    diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    s = np.concatenate([[0.0], np.cumsum(diffs)])

    total_len = float(s[-1])
    if total_len < 1e-6:
        return [(float(xs[0]), float(ys[0]))] * num_points

    target_s = np.linspace(0.0, total_len, num_points, dtype=np.float32)
    sampled = []

    for ts in target_s:
        idx = int(np.searchsorted(s, ts))
        idx = int(np.clip(idx, 1, len(s) - 1))

        s0, s1 = s[idx - 1], s[idx]
        alpha = 0.0 if abs(s1 - s0) < 1e-8 else (ts - s0) / (s1 - s0)

        y = ys[idx - 1] * (1.0 - alpha) + ys[idx] * alpha
        x = xs[idx - 1] * (1.0 - alpha) + xs[idx] * alpha
        sampled.append((float(x), float(y)))

    return sampled


def clip_points_to_image(points: List[Tuple[float, float]], w: int, h: int) -> List[Tuple[float, float]]:
    return [
        (float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1)))
        for x, y in points
    ]


# =========================================================
# Ground-plane projection
# =========================================================

def rot_x(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ], dtype=np.float32)


def image_point_to_ground_vehicle(
    u: float,
    v: float,
    K: Dict[str, float],
    cam_height_m: float,
    cam_pitch_deg: float,
) -> Optional[List[float]]:
    """
    Vehicle frame:
      X = right
      Y = up
      Z = forward

    IMPORTANT:
    Positive camera_pitch_deg means the camera is pitched downward.
    The previous sign was wrong and can make projected lanes look bad.
    """
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]

    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_cam = np.array([x, y, 1.0], dtype=np.float32)

    # OpenCV camera frame: x right, y down, z forward
    # Vehicle frame:      X right, Y up,   Z forward
    ray_vehicle0 = np.array([ray_cam[0], -ray_cam[1], ray_cam[2]], dtype=np.float32)

    # FIX:
    # If camera is pitched downward by +pitch_deg, rotate the ray by +pitch around X.
    pitch_rad = math.radians(cam_pitch_deg)
    R_pitch = rot_x(pitch_rad)
    ray_world = R_pitch @ ray_vehicle0

    C = np.array([0.0, cam_height_m, 0.0], dtype=np.float32)

    denom = float(ray_world[1])
    if abs(denom) < 1e-8:
        return None

    t = -C[1] / denom
    if t <= 0:
        return None

    P = C + t * ray_world
    return [float(P[0]), float(P[1]), float(P[2])]


# =========================================================
# Visualization
# =========================================================

def color_for_label(label_id: int) -> Tuple[int, int, int]:
    palette = [
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 128, 0),
    ]
    return palette[label_id % len(palette)]


def draw_lane_overlay(
    frame: np.ndarray,
    lane_entries: List[Dict[str, Any]],
    roi_mask: Optional[np.ndarray] = None,
    alpha_mask: float = 0.18,
    draw_boxes: bool = False,
    draw_roi: bool = False,
) -> np.ndarray:
    out = frame.copy()

    if draw_roi and roi_mask is not None:
        roi_vis = np.zeros_like(out)
        roi_vis[:, :, 1] = roi_mask * 120
        out = cv2.addWeighted(out, 1.0, roi_vis, 0.22, 0)

    for lane in lane_entries:
        label_id = lane["label_id"]
        color = color_for_label(label_id)

        if "mask_debug" in lane:
            mask = np.array(lane["mask_debug"], dtype=np.uint8)
            colored = np.zeros_like(out)
            colored[:, :] = color
            idx = mask > 0
            out[idx] = cv2.addWeighted(out[idx], 1.0 - alpha_mask, colored[idx], alpha_mask, 0)

        if draw_boxes:
            x1, y1, x2, y2 = map(int, lane["bbox_xyxy"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        pts_int = [(int(round(p[0])), int(round(p[1]))) for p in lane["points_2d"]]

        for i in range(len(pts_int) - 1):
            cv2.line(out, pts_int[i], pts_int[i + 1], (255, 0, 0), 3)

        for p in pts_int:
            cv2.circle(out, p, 4, (0, 0, 255), -1)

        if pts_int:
            txt = f'{lane["label_name"]} {lane["score"]:.2f}'
            px, py = pts_int[0]
            cv2.putText(
                out,
                txt,
                (px + 6, max(20, py - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    return out


# =========================================================
# Helpers
# =========================================================

def tensor_from_bgr(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float() / 255.0
    t = t.permute(2, 0, 1)
    return t


def is_lane_class(label_id: int, label_name: str, allowed_lane_ids: Optional[set]) -> bool:
    if label_id == 0:
        return False

    if allowed_lane_ids is not None:
        return label_id in allowed_lane_ids

    name = label_name.lower()
    keywords = ["lane", "roadline", "road_line", "divider", "marking", "line"]
    return any(k in name for k in keywords)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)

    # Lower defaults so detections do not disappear
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--mask-threshold", type=float, default=0.35)

    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--class-names", nargs="+", required=True)
    parser.add_argument("--lane-class-ids", nargs="*", type=int, default=None)

    parser.add_argument("--max-frames", type=int, default=-1)

    parser.add_argument("--camera-height", type=float, default=1.45)
    parser.add_argument("--camera-pitch-deg", type=float, default=5.0)

    parser.add_argument("--draw-boxes", action="store_true")
    parser.add_argument("--draw-roi", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--min-lane-area", type=int, default=80)
    parser.add_argument("--min-lane-height", type=int, default=60)
    parser.add_argument("--max-lane-width-ratio", type=float, default=0.18)
    parser.add_argument("--min-roi-overlap", type=float, default=0.55)

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    overlay_path = os.path.join(args.output_dir, "lane_overlay_report_style.mp4")
    json_path = os.path.join(args.output_dir, "lane_report_style.json")

    if len(args.class_names) != args.num_classes:
        raise ValueError("Length of --class-names must match --num-classes")

    K = load_front_camera_intrinsics()
    label_map = {i: name for i, name in enumerate(args.class_names)}
    allowed_lane_ids = set(args.lane_class_ids) if args.lane_class_ids is not None and len(args.lane_class_ids) > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = build_model(args.num_classes, args.weights, device)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    roi_mask = make_road_roi_mask(h, w)

    writer = cv2.VideoWriter(
        overlay_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    result = {
        "meta": {
            "input_video": args.input,
            "weights": args.weights,
            "fps": fps,
            "frame_width": w,
            "frame_height": h,
            "threshold": args.threshold,
            "mask_threshold": args.mask_threshold,
            "num_classes": args.num_classes,
            "class_names": args.class_names,
            "lane_class_ids": list(allowed_lane_ids) if allowed_lane_ids is not None else None,
            "intrinsics": K,
            "camera_height": args.camera_height,
            "camera_pitch_deg": args.camera_pitch_deg,
            "description": "Mask R-CNN lanes -> ROI cleanup -> robust quadratic fit -> 10 equidistant points -> corrected ground projection"
        },
        "frames": []
    }

    pbar = tqdm(total=total_frames, desc="Processing lanes")
    frame_idx = 0

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            inp = tensor_from_bgr(frame).to(device)
            pred = model([inp])[0]

            boxes = pred.get("boxes", torch.empty((0, 4))).detach().cpu().numpy()
            scores = pred.get("scores", torch.empty((0,))).detach().cpu().numpy()
            labels = pred.get("labels", torch.empty((0,), dtype=torch.int64)).detach().cpu().numpy()
            masks = pred.get("masks", None)
            if masks is not None:
                masks = masks.detach().cpu().numpy()

            frame_entry = {"frame_idx": frame_idx, "lanes": []}
            overlay_entries = []

            if args.debug and len(scores) > 0 and frame_idx % 20 == 0:
                topn = min(8, len(scores))
                print(f"\n[DEBUG] frame={frame_idx} top detections:")
                for k in range(topn):
                    lid = int(labels[k])
                    lname = label_map.get(lid, f"class_{lid}")
                    print(f"  det{k}: label={lid} ({lname}), score={scores[k]:.4f}")

            for i in range(len(scores)):
                score = float(scores[i])

                if score < args.threshold:
                    continue
                if masks is None:
                    continue

                label_id = int(labels[i])
                label_name = label_map.get(label_id, f"class_{label_id}")

                if not is_lane_class(label_id, label_name, allowed_lane_ids):
                    continue

                raw_mask = (masks[i, 0] >= args.mask_threshold).astype(np.uint8)

                if raw_mask.sum() == 0:
                    continue

                lane_mask = clean_lane_mask(raw_mask, roi_mask, frame_h=h, frame_w=w)

                if not keep_lane_candidate(
                    lane_mask,
                    frame_h=h,
                    frame_w=w,
                    roi_mask=roi_mask,
                    min_area=args.min_lane_area,
                    min_height=args.min_lane_height,
                    max_width_ratio=args.max_lane_width_ratio,
                    min_roi_overlap=args.min_roi_overlap,
                ):
                    continue

                coeffs = fit_lane_polynomial_from_mask(lane_mask)
                if coeffs is None:
                    continue

                ys, xs = np.where(lane_mask > 0)
                if len(ys) < 10:
                    continue

                # extend down to road bottom for better visible alignment
                y_min = float(max(int(np.percentile(ys, 10)), int(0.56 * h)))
                y_max = float(min(h - 1, max(np.max(ys), int(0.98 * h))))

                pts_2d = sample_equidistant_points_on_curve(
                    coeffs=coeffs,
                    y_min=y_min,
                    y_max=y_max,
                    num_points=10,
                    dense_samples=500,
                )
                pts_2d = clip_points_to_image(pts_2d, w, h)

                pts_ground = []
                for x, y in pts_2d:
                    P = image_point_to_ground_vehicle(
                        u=x,
                        v=y,
                        K=K,
                        cam_height_m=args.camera_height,
                        cam_pitch_deg=args.camera_pitch_deg,
                    )
                    if P is not None:
                        pts_ground.append(P)

                if len(pts_ground) < 2:
                    if args.debug:
                        print(f"[DEBUG] frame={frame_idx} det={i} rejected: projection failed")
                    continue

                lane_entry = {
                    "det_idx": i,
                    "label_id": label_id,
                    "label_name": label_name,
                    "score": score,
                    "bbox_xyxy": [float(v) for v in boxes[i].tolist()],
                    "poly_coeffs_xy_as_fn_of_y": [float(v) for v in coeffs.tolist()],
                    "y_range": [float(y_min), float(y_max)],
                    "points_2d": [[float(x), float(y)] for x, y in pts_2d],
                    "points_ground_vehicle": pts_ground,
                }

                frame_entry["lanes"].append(lane_entry)
                overlay_entries.append({**lane_entry, "mask_debug": lane_mask.tolist()})

            if args.debug and frame_idx % 20 == 0:
                print(f"[DEBUG] frame={frame_idx} final_lanes={len(frame_entry['lanes'])}")

            overlay = draw_lane_overlay(
                frame,
                overlay_entries,
                roi_mask=roi_mask,
                alpha_mask=0.18,
                draw_boxes=args.draw_boxes,
                draw_roi=args.draw_roi,
            )
            writer.write(overlay)
            result["frames"].append(frame_entry)

            frame_idx += 1
            pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] Overlay video: {overlay_path}")
    print(f"[DONE] JSON: {json_path}")


if __name__ == "__main__":
    main()  