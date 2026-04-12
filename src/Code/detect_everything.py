#!/usr/bin/env python3
"""
Detect Everything Pipeline
============================
Single script to detect ALL objects for self-driving visualization:
  - Vehicles: sedan, SUV, hatchback, pickup, truck, bicycle, motorcycle
  - Traffic lights: red/yellow/green + arrow direction (left/right/straight)
  - Road signs: stop sign, speed limit (with OCR), yield
  - Road markings: arrows painted on road
  - Objects: dustbin, traffic pole, traffic cone, traffic cylinder
  - Pedestrians: with YOLOv8-Pose keypoints

Uses: GroundingDINO (detection) + MiDaS (depth) + HSV (traffic light) +
      EasyOCR (speed limit) + YOLOv8-Pose (pedestrian pose)

Usage:
    python detect_everything.py \
        --video-input /path/to/video.mp4 \
        --output-dir ./output \
        --debug-video
"""

import os
import cv2
import json
import math
import time
import argparse
import numpy as np
import torch
import torchvision.transforms as T

from groundingdino.util.inference import load_model, predict

# YOLOv8-Pose for pedestrians
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARN] ultralytics not installed. pip install ultralytics")
    print("[WARN] Pedestrian pose will be disabled.")
    YOLO_AVAILABLE = False

# EasyOCR for speed limits
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    print("[WARN] easyocr not installed. pip install easyocr")
    OCR_AVAILABLE = False


# =========================================================
# GroundingDINO caption — one big prompt for everything
# =========================================================
# Using periods to separate classes (GroundingDINO convention)
DEFAULT_CAPTION = (
    "sedan car . suv . hatchback car . pickup truck . truck . bicycle . motorcycle . "
    "traffic light . "
    "stop sign . speed limit sign . "
    "dustbin . traffic cone . traffic cylinder . traffic pole . "
    "person . pedestrian"
)


# =========================================================
# Classification
# =========================================================
def classify_detection(phrase: str):
    """
    Map GroundingDINO phrase -> (group, subclass)
    Returns: (group, subclass) tuple
    Groups: vehicle, traffic_light, sign, object, pedestrian, marking
    """
    p = phrase.lower().strip()

    # ---- Vehicles ----
    if "sedan" in p:
        return "vehicle", "sedan"
    if "suv" in p:
        return "vehicle", "suv"
    if "hatchback" in p:
        return "vehicle", "hatchback"
    if "pickup" in p:
        return "vehicle", "pickup_truck"
    if "truck" in p and "pickup" not in p:
        return "vehicle", "truck"
    if "motorcycle" in p or "motorbike" in p:
        return "vehicle", "motorcycle"
    if "bicycle" in p or "bike" in p:
        return "vehicle", "bicycle"
    if "car" in p or "vehicle" in p or "van" in p:
        # Generic car — try to sub-classify later
        return "vehicle", "sedan"

    # ---- Traffic lights ----
    if "traffic light" in p or "signal" in p:
        return "traffic_light", "traffic_light"

    # ---- Signs ----
    if "stop" in p and "sign" in p:
        return "sign", "stop_sign"
    if "speed" in p or "limit" in p:
        return "sign", "speed_limit"
    if "yield" in p:
        return "sign", "yield_sign"
    if "sign" in p:
        return "sign", "generic_sign"

    # ---- Objects ----
    if "dustbin" in p or "trash" in p or "garbage" in p or "bin" in p:
        return "object", "dustbin"
    if "cone" in p:
        return "object", "traffic_cone"
    if "cylinder" in p or "barrel" in p or "bollard" in p:
        return "object", "traffic_cylinder"
    if "pole" in p or "post" in p:
        return "object", "traffic_pole"

    # ---- Pedestrians ----
    if "person" in p or "pedestrian" in p or "people" in p:
        return "pedestrian", "pedestrian"
    if "cyclist" in p:
        return "pedestrian", "pedestrian"

    return "unknown", "unknown"


# =========================================================
# Vehicle orientation heuristic
# =========================================================
def estimate_vehicle_yaw(bbox, frame_w):
    """
    Estimate vehicle yaw from bbox shape and position.
    - Wide bbox (aspect > 1.4) = side view = ~90 deg
    - Tall/square bbox = front/rear view = ~0 deg
    - Position in frame hints left vs right facing
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    aspect = w / max(h, 1.0)

    if aspect > 1.4:
        # Side view: left half of frame = facing right, right half = facing left
        if cx < frame_w / 2:
            return math.radians(90)   # facing right
        else:
            return math.radians(-90)  # facing left
    elif aspect > 1.1:
        # Slight angle
        if cx < frame_w / 2:
            return math.radians(45)
        else:
            return math.radians(-45)
    else:
        # Front/rear view
        return 0.0


# =========================================================
# Vehicle sub-classification from crop
# =========================================================
class VehicleSubClassifier:
    """
    Second-pass GroundingDINO on vehicle crops to sub-classify.
    Only runs when initial detection is generic ('car', 'vehicle').
    """
    VEHICLE_CAPTION = "sedan . suv . hatchback . pickup truck . truck . van . minivan"

    def __init__(self, gdino_model, device):
        self.model = gdino_model
        self.device = device
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((256, 256), antialias=True) # Forces enough tokens for GroundingDINO
        ])

    def classify(self, frame_bgr, bbox_xyxy):
        h, w = frame_bgr.shape[:2]
        x1 = max(0, int(bbox_xyxy[0]))
        y1 = max(0, int(bbox_xyxy[1]))
        x2 = min(w, int(bbox_xyxy[2]))
        y2 = min(h, int(bbox_xyxy[3]))

        if x2 - x1 < 30 or y2 - y1 < 30:
            return "sedan"

        crop = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).to(self.device)

        try:
            boxes, logits, phrases = predict(
                model=self.model, image=tensor,
                caption=self.VEHICLE_CAPTION,
                box_threshold=0.25, text_threshold=0.20,
                device=self.device,
            )
            if len(phrases) > 0:
                best_idx = logits.argmax()
                best_phrase = phrases[best_idx].lower().strip()
                if "suv" in best_phrase:
                    return "suv"
                if "hatchback" in best_phrase:
                    return "hatchback"
                if "pickup" in best_phrase:
                    return "pickup_truck"
                if "truck" in best_phrase:
                    return "truck"
                if "van" in best_phrase or "minivan" in best_phrase:
                    return "suv"
        except Exception:
            pass
        return "sedan"


# =========================================================
# Traffic light color + arrow detection
# =========================================================
def classify_traffic_light(frame_bgr, bbox_xyxy):
    """
    Analyze traffic light crop to determine:
    - Color: red, yellow, green, unknown
    - Arrow: left, right, straight, none
    """
    h, w = frame_bgr.shape[:2]
    x1 = max(0, int(bbox_xyxy[0]))
    y1 = max(0, int(bbox_xyxy[1]))
    x2 = min(w, int(bbox_xyxy[2]))
    y2 = min(h, int(bbox_xyxy[3]))

    if x2 - x1 < 6 or y2 - y1 < 10:
        return {"color": "unknown", "arrow": "none", "state": "unknown"}

    crop = frame_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]

    # Less aggressive crop (just 10% to remove bounding box line artifacts)
    sx1, sx2 = int(0.10 * cw), int(0.90 * cw)
    sy1, sy2 = int(0.10 * ch), int(0.90 * ch)
    roi = crop[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return {"color": "unknown", "arrow": "none", "state": "unknown"}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Lowered saturation threshold to account for bright/blown-out lights
    bright_sat = (S > 40) & (V > 120)

    # Hue masks
    red_mask = (((H <= 10) | (H >= 160)) & bright_sat).astype(np.uint8)
    yellow_mask = (((H >= 15) & (H <= 40)) & bright_sat).astype(np.uint8)
    green_mask = (((H >= 45) & (H <= 95)) & bright_sat).astype(np.uint8)

    def get_score(mask):
        count = int(np.count_nonzero(mask))
        if count == 0:
            return 0.0, 0
        mean_v = float(V[mask > 0].mean())
        return count * (mean_v / 255.0), count

    # Evaluate the WHOLE bounding box, ignoring vertical position
    red_score, red_count = get_score(red_mask)
    yel_score, yel_count = get_score(yellow_mask)
    grn_score, grn_count = get_score(green_mask)

    scores = {"red": red_score, "yellow": yel_score, "green": grn_score}
    counts = {"red": red_count, "yellow": yel_count, "green": grn_count}

    best = max(scores, key=scores.get)
    
    # Require a minimum pixel count to prevent noise from triggering a state
    min_pixels = max(4, int((roi.shape[0] * roi.shape[1]) * 0.02))
    
    if scores[best] <= 0 or counts[best] < min_pixels:
        color = "unknown"
        active_mask = None
    else:
        color = best
        if color == "red": active_mask = red_mask
        elif color == "yellow": active_mask = yellow_mask
        else: active_mask = green_mask

    # ---- Arrow detection ----
    arrow = detect_arrow_in_light(roi, color, active_mask)

    state = f"{color}"
    if arrow != "none":
        state = f"{color}_{arrow}"

    return {"color": color, "arrow": arrow, "state": state}


def detect_arrow_in_light(roi, color, active_mask):
    """
    Detect arrow direction directly from the isolated color mask.
    """
    if color == "unknown" or active_mask is None:
        return "none"

    mask = (active_mask * 255).astype(np.uint8)

    # Morphology to clean noise and connect broken parts of blown-out lights
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "none"

    # Find the biggest bright blob
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)

    # Analyze shape: is it circular (no arrow) or elongated (arrow)?
    hull = cv2.convexHull(biggest)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1:
        return "none"

    solidity = area / hull_area
    bx, by, bw, bh = cv2.boundingRect(biggest)
    aspect = bw / max(bh, 1)

    # Circular lights have high solidity (>0.80) and aspect ratio near 1.0
    if solidity > 0.80 and 0.7 < aspect < 1.4:
        return "none"  # solid circle, no arrow

    # Determine direction from mass distribution
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return "none"

    # Check left-right mass balance
    mid_x = bx + bw // 2
    left_pts = sum(1 for pt in biggest if pt[0][0] < mid_x)
    right_pts = sum(1 for pt in biggest if pt[0][0] >= mid_x)
    total_pts = left_pts + right_pts

    if total_pts < 5:
        return "none"

    lr_ratio = (right_pts - left_pts) / total_pts

    # Check vertical elongation for straight arrows
    if bh > bw * 1.5 and abs(lr_ratio) < 0.2:
        return "straight"

    if lr_ratio > 0.25:
        return "right"
    elif lr_ratio < -0.25:
        return "left"

    # Advanced shape check if mass balance is ambiguous
    epsilon = 0.04 * cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, epsilon, True)
    if 4 <= len(approx) <= 8:
        cx_c = M["m10"] / M["m00"]
        pts = approx.reshape(-1, 2)
        dists = np.sqrt((pts[:, 0] - cx_c)**2 + (pts[:, 1] - M["m01"]/M["m00"])**2)
        tip_idx = int(np.argmax(dists))
        tip_x = pts[tip_idx, 0]

        if tip_x < cx_c - 3:
            return "left"
        elif tip_x > cx_c + 3:
            return "right"
        else:
            return "straight"

    return "none"

# =========================================================
# Vehicle light analysis (brake lights + turn indicators)
# YCbCr approach: Cr channel captures red-difference chrominance,
# robust across lighting conditions. Dynamic threshold adapts per crop.
# =========================================================
_BRAKE_THRESH  = 0.04   # mean Cr-red score both sides → braking
_MARGIN_RATIO  = 0.25   # |L-R|/(L+R) > this → indicator (asymmetric)
_VAR_THRESH    = 0.0002 # temporal variance in red score → blink detected


def analyze_vehicle_lights(frame_bgr, bbox_xyxy):
    """
    YCbCr-based brake light and indicator detection.
      1. Crop bottom 50% of vehicle bbox
      2. Gaussian blur to reduce noise
      3. Convert to YCbCr, extract Cr channel (red-difference)
      4. Dynamic threshold: mean(Cr) + 1.5 * std(Cr)
      5. Quantify red area in left vs right half
      6. Return normalized scores for each side
    """
    h, w = frame_bgr.shape[:2]
    x1 = max(0, int(bbox_xyxy[0]))
    y1 = max(0, int(bbox_xyxy[1]))
    x2 = min(w, int(bbox_xyxy[2]))
    y2 = min(h, int(bbox_xyxy[3]))

    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return {"red_score_left": 0.0, "red_score_right": 0.0}

    # Bottom 50% of bbox — rear lights are in the lower portion
    roi_top = y1 + int(0.50 * bh)
    roi = frame_bgr[roi_top:y2, x1:x2]
    if roi.size == 0:
        return {"red_score_left": 0.0, "red_score_right": 0.0}

    # Step 2: Gaussian blur
    roi = cv2.GaussianBlur(roi, (5, 5), 0)

    # Step 3: Convert to YCbCr, extract Cr (index 1 in OpenCV's YCrCb)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Cr = ycrcb[:, :, 1].astype(np.float32)

    # Step 4: Dynamic threshold
    cr_mean = Cr.mean()
    cr_std  = Cr.std()
    threshold = cr_mean + 1.5 * cr_std
    red_mask = (Cr > threshold).astype(np.uint8)

    # Step 5: Quantify red area left vs right
    mid_x = max(1, red_mask.shape[1] // 2)
    left_area  = int(red_mask[:, :mid_x].sum())
    right_area = int(red_mask[:, mid_x:].sum())

    # Normalize by half-roi pixel count
    half_pixels = roi.shape[0] * mid_x
    if half_pixels == 0:
        return {"red_score_left": 0.0, "red_score_right": 0.0}

    return {
        "red_score_left":  left_area  / half_pixels,
        "red_score_right": right_area / half_pixels,
    }


def _compute_vehicle_signals(history):
    """
    Derive brake/indicator state from Cr-score history.
    - Asymmetry (one side dominant) → indicator
    - Both sides elevated + symmetric + steady → braking
    - Temporal variance → blinking indicator
    """
    if not history:
        return {"braking": False, "indicator": "none"}

    bl = [h["red_score_left"]  for h in history]
    br = [h["red_score_right"] for h in history]

    mean_l = float(np.mean(bl))
    mean_r = float(np.mean(br))
    total  = mean_l + mean_r

    asymmetry = abs(mean_l - mean_r) / total if total > 0.01 else 0.0

    var_l  = float(np.var(bl)) if len(bl) > 1 else 0.0
    var_r  = float(np.var(br)) if len(br) > 1 else 0.0
    flash_l = var_l > _VAR_THRESH
    flash_r = var_r > _VAR_THRESH

    braking = (
        mean_l > _BRAKE_THRESH and
        mean_r > _BRAKE_THRESH and
        asymmetry < _MARGIN_RATIO and
        not (flash_l or flash_r)
    )

    indicator = "none"
    if flash_l and flash_r:
        indicator = "hazard"
    elif flash_l:
        indicator = "left"
    elif flash_r:
        indicator = "right"
    elif asymmetry > _MARGIN_RATIO and total > 0.01:
        indicator = "left" if mean_l > mean_r else "right"

    return {"braking": braking, "indicator": indicator}

# =========================================================
# Speed limit OCR
# =========================================================
class SpeedLimitReader:
    def __init__(self):
        if OCR_AVAILABLE:
            self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        else:
            self.reader = None

    def read_speed(self, frame_bgr, bbox_xyxy):
        if self.reader is None:
            return "unknown"
        h, w = frame_bgr.shape[:2]
        x1 = max(0, int(bbox_xyxy[0]))
        y1 = max(0, int(bbox_xyxy[1]))
        x2 = min(w, int(bbox_xyxy[2]))
        y2 = min(h, int(bbox_xyxy[3]))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return "unknown"

        crop = frame_bgr[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        region = crop[int(0.35*ch):int(0.85*ch), int(0.15*cw):int(0.85*cw)]
        if region.size == 0:
            return "unknown"

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 50 or gray.shape[1] < 50:
            s = max(50/gray.shape[0], 50/gray.shape[1], 2.0)
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        try:
            results = self.reader.readtext(binary, allowlist="0123456789")
            if not results:
                results = self.reader.readtext(gray, allowlist="0123456789")
            for (_, text, conf) in results:
                text = text.strip()
                if conf > 0.3 and text.isdigit():
                    val = int(text)
                    if 5 <= val <= 85:
                        return str(val)
        except:
            pass
        return "unknown"


# =========================================================
# MiDaS Depth
# =========================================================
class DepthEstimator:
    def __init__(self, model_name="DPT_Hybrid", device="cuda"):
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        print(f"[INFO] Loading MiDaS: {model_name} on {self.device}")
        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_name in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        elif model_name == "MiDaS_small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.default_transform

    def predict(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin > 1e-8:
            depth = (depth - dmin) / (dmax - dmin)
        else:
            depth = np.zeros_like(depth)
        return np.clip(1.0 - depth, 1e-3, 1.0)


# =========================================================
# Utility
# =========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def backproject(u, v, z, fx, fy, cx, cy):
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return [float(x), float(y), float(z)]

def robust_patch_depth(depth_map, u, v, patch=5, z_near=3.0, z_far=45.0):
    h, w = depth_map.shape[:2]
    uu, vv = int(round(u)), int(round(v))
    x1 = clamp(uu - patch, 0, w - 1)
    x2 = clamp(uu + patch, 0, w - 1)
    y1 = clamp(vv - patch, 0, h - 1)
    y2 = clamp(vv + patch, 0, h - 1)
    vals = depth_map[y1:y2+1, x1:x2+1].reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > 1e-6)]
    if len(vals) == 0:
        return None
    return z_near + float(np.median(vals)) * (z_far - z_near)


# =========================================================
# Tracker
# =========================================================
class Tracker:
    def __init__(self, dist_thresh=100, max_age=10):
        self.next_id = 1
        self.tracks = {}
        self.dist_thresh = dist_thresh
        self.max_age = max_age

    def update(self, detections):
        used = set()
        outputs = []
        for det in detections:
            cx = (det["bbox_xyxy"][0] + det["bbox_xyxy"][2]) / 2
            cy = (det["bbox_xyxy"][1] + det["bbox_xyxy"][3]) / 2
            best_tid, best_d = None, 1e18
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                if tr["group"] != det["group"]:
                    continue
                tcx = (tr["bbox_xyxy"][0] + tr["bbox_xyxy"][2]) / 2
                tcy = (tr["bbox_xyxy"][1] + tr["bbox_xyxy"][3]) / 2
                d = math.sqrt((cx-tcx)**2 + (cy-tcy)**2)
                if d < best_d and d < self.dist_thresh:
                    best_d = d
                    best_tid = tid

            if best_tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {**det, "age": 0, "hits": 1, "track_id": tid, "light_history": []}
            else:
                tid = best_tid
                old = self.tracks[tid]
                alpha = 0.4
                op = old.get("position_3d", det["position_3d"])
                np_ = det["position_3d"]
                det["position_3d"] = [(1-alpha)*op[i]+alpha*np_[i] for i in range(3)]
                old_history = old.get("light_history", [])
                self.tracks[tid].update(det)
                self.tracks[tid]["light_history"] = old_history
                self.tracks[tid]["hits"] = old["hits"] + 1
                self.tracks[tid]["age"] = 0
                self.tracks[tid]["track_id"] = tid

            # Update light history and derive signal states
            raw = det.get("_raw_lights")
            if raw is not None:
                hist = self.tracks[tid]["light_history"]
                hist.append(raw)
                if len(hist) > 15:
                    hist.pop(0)
            self.tracks[tid]["vehicle_signals"] = _compute_vehicle_signals(
                self.tracks[tid]["light_history"]
            )

            used.add(tid)
            outputs.append(dict(self.tracks[tid]))

        for tid in list(self.tracks.keys()):
            if tid not in used:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        return outputs


# =========================================================
# Debug drawing
# =========================================================
GROUP_COLORS = {
    "vehicle":       (255, 100, 0),    # blue
    "traffic_light": (0, 255, 255),    # yellow
    "sign":          (0, 165, 255),    # orange
    "object":        (200, 100, 200),  # purple
    "pedestrian":    (0, 255, 0),      # green
    "marking":       (255, 255, 0),    # cyan
    "unknown":       (128, 128, 128),  # gray
}

COCO_POSE_CONNECTIONS = [
    (0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]


def draw_debug(vis, tracked, pose_results=None):
    """Draw all tracked detections on debug frame."""
    for tr in tracked:
        x1, y1, x2, y2 = [int(v) for v in tr["bbox_xyxy"]]
        group = tr.get("group", "unknown")
        subclass = tr.get("subclass", "?")
        tid = tr.get("track_id", 0)
        z = tr["position_3d"][2]
        color = GROUP_COLORS.get(group, (200, 200, 200))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label
        parts = [f"ID:{tid}", subclass]
        if group == "traffic_light":
            tl = tr.get("traffic_light_info", {})
            parts.append(tl.get("state", "?"))
        if subclass == "speed_limit":
            parts.append(f"SPD:{tr.get('speed_value', '?')}")
        if group == "vehicle":
            sigs = tr.get("vehicle_signals", {})
            if sigs.get("braking"):
                parts.append("BRAKE")
            ind = sigs.get("indicator", "none")
            if ind == "left":
                parts.append("<-TURN")
            elif ind == "right":
                parts.append("TURN->")
            elif ind == "hazard":
                parts.append("HAZARD")
        parts.append(f"z:{z:.1f}")

        label = " ".join(parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(18, y1 - 5)
        cv2.rectangle(vis, (x1, ty-th-4), (x1+tw+4, ty+4), color, -1)
        text_color = (0,0,0) if sum(color) > 400 else (255,255,255)
        cv2.putText(vis, label, (x1+2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # Draw pedestrian poses if available
    if pose_results is not None and len(pose_results) > 0:
        r = pose_results[0]
        if r.keypoints is not None:
            kps_all = r.keypoints.data.cpu().numpy()
            for person_kps in kps_all:
                for (i, j) in COCO_POSE_CONNECTIONS:
                    if i < len(person_kps) and j < len(person_kps):
                        a, b = person_kps[i], person_kps[j]
                        if a[2] > 0.4 and b[2] > 0.4:
                            cv2.line(vis,
                                     (int(a[0]), int(a[1])),
                                     (int(b[0]), int(b[1])),
                                     (255, 0, 255), 2, cv2.LINE_AA)
                for kp in person_kps:
                    if kp[2] > 0.4:
                        cv2.circle(vis, (int(kp[0]), int(kp[1])), 3,
                                   (0, 0, 255), -1, cv2.LINE_AA)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Detect Everything")
    parser.add_argument("--video-input", default=(
        "/home/adipat/Documents/Spring_26/CV/p3/P3Data/Sequences/scene3/Undist/2023-02-14_11-49-54-front_undistort.mp4"
    ))
    parser.add_argument("--output-dir",
                        default="/home/adipat/Documents/Spring_26/CV/p3/src/pipeline_out/scene3")
    parser.add_argument("--gdino-config", default=(
        "/home/adipat/Documents/Spring_26/CV/p3/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    ))
    parser.add_argument("--gdino-weights", default=(
        "/home/adipat/Documents/Spring_26/CV/p3/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    ))
    parser.add_argument("--yolo-pose-model", default="yolov8x-pose.pt",
                        help="YOLOv8 pose model for pedestrians")

    parser.add_argument("--fx", type=float, default=1594.7)
    parser.add_argument("--fy", type=float, default=1607.7)
    parser.add_argument("--cx", type=float, default=654.3)
    parser.add_argument("--cy", type=float, default=413.4)

    parser.add_argument("--caption", default=DEFAULT_CAPTION)
    parser.add_argument("--box-threshold", type=float, default=0.28)
    parser.add_argument("--text-threshold", type=float, default=0.22)
    parser.add_argument("--depth-model", default="DPT_Hybrid")
    parser.add_argument("--z-near", type=float, default=3.0)
    parser.add_argument("--z-far", type=float, default=45.0)

    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--debug-video", action="store_true", default=True)
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--skip-vehicle-subclass", action="store_true",
                        help="Skip second-pass vehicle sub-classification")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load models ----
    print(f"[INFO] Loading GroundingDINO on {device.upper()}...")
    gdino = load_model(args.gdino_config, args.gdino_weights, device=device)
    img_transform = T.Compose([T.ToTensor()])

    depth_est = None if args.skip_depth else DepthEstimator(args.depth_model, device)
    speed_reader = SpeedLimitReader()

    vehicle_classifier = None
    if not args.skip_vehicle_subclass:
        vehicle_classifier = VehicleSubClassifier(gdino, device)

    yolo_pose = None
    if YOLO_AVAILABLE and not args.skip_pose:
        print(f"[INFO] Loading YOLOv8-Pose: {args.yolo_pose_model}")
        yolo_pose = YOLO(args.yolo_pose_model)

    tracker = Tracker()

    # ---- Video ----
    cap = cv2.VideoCapture(args.video_input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video_input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {W}x{H} @ {fps:.1f}fps, {total} frames")

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    writer = None
    if args.debug_video:
        out_vid = os.path.join(args.output_dir, "debug_everything.mp4")
        writer = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    all_frames = []
    frame_idx = args.start_frame
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if 0 < args.max_frames <= processed:
                break

            t0 = time.time()

            # ---- Depth ----
            depth_map = depth_est.predict(frame) if depth_est else None

            # ---- GroundingDINO: detect everything ----
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = img_transform(frame_rgb).to(device)
            boxes, logits, phrases = predict(
                model=gdino, image=tensor,
                caption=args.caption,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=device,
            )

            # ---- YOLOv8-Pose: pedestrian keypoints ----
            pose_results = None
            pose_kps_by_person = {}  # will map person bbox -> keypoints
            if yolo_pose is not None:
                pose_results = yolo_pose.predict(frame, conf=0.25,
                                                  device=device, verbose=False)
                if pose_results and pose_results[0].keypoints is not None:
                    r = pose_results[0]
                    for pi in range(len(r.boxes)):
                        if int(r.boxes.cls[pi]) != 0:
                            continue
                        pb = r.boxes.xyxy[pi].cpu().numpy()
                        kps = r.keypoints.data[pi].cpu().numpy()  # (17,3)
                        # Store by centroid for matching
                        pcx = (pb[0] + pb[2]) / 2
                        pcy = (pb[1] + pb[3]) / 2
                        pose_kps_by_person[(pcx, pcy)] = kps.tolist()

            # ---- Process detections ----
            frame_dets = []

            if len(boxes) > 0:
                boxes_np = boxes.cpu().numpy()
                for i in range(len(boxes_np)):
                    bcx, bcy, bw, bh = boxes_np[i]
                    x1 = (bcx - bw/2) * W
                    y1 = (bcy - bh/2) * H
                    x2 = (bcx + bw/2) * W
                    y2 = (bcy + bh/2) * H
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                    score = float(logits[i])
                    phrase = phrases[i] if i < len(phrases) else "unknown"

                    group, subclass = classify_detection(phrase)
                    if group == "unknown":
                        continue

                    # ---- Depth & 3D position ----
                    cu = (x1 + x2) / 2
                    cv_pt = (y1 + y2) / 2
                    if group == "pedestrian":
                        # Use feet position for pedestrians
                        cv_pt = y1 + 0.9 * (y2 - y1)
                    elif group == "vehicle":
                        # Use bottom center for vehicles
                        cv_pt = y2 - 4

                    cu = clamp(cu, 0, W-1)
                    cv_pt = clamp(cv_pt, 0, H-1)

                    z = None
                    if depth_map is not None:
                        z = robust_patch_depth(depth_map, cu, cv_pt,
                                               patch=6, z_near=args.z_near,
                                               z_far=args.z_far)
                    if z is None:
                        z = 15.0

                    pos_3d = backproject(cu, cv_pt, z,
                                         args.fx, args.fy, args.cx, args.cy)

                    # ---- Group-specific processing ----
                    yaw = 0.0
                    speed_value = "unknown"
                    traffic_light_info = {}
                    pose_keypoints_2d = None
                    pose_keypoints_3d = None

                    raw_lights = None
                    if group == "vehicle":
                        yaw = estimate_vehicle_yaw(bbox, W)
                        # Sub-classify if needed
                        if vehicle_classifier and subclass == "sedan":
                            subclass = vehicle_classifier.classify(frame, bbox)
                        raw_lights = analyze_vehicle_lights(frame, bbox)

                    elif group == "traffic_light":
                        traffic_light_info = classify_traffic_light(frame, bbox)
                        subclass = f"traffic_light"

                    elif group == "sign" and subclass == "speed_limit":
                        speed_value = speed_reader.read_speed(frame, bbox)

                    elif group == "pedestrian":
                        # Match with YOLO pose by nearest centroid
                        pcx = (x1 + x2) / 2
                        pcy = (y1 + y2) / 2
                        best_match = None
                        best_dist = 80  # max matching distance
                        for (kcx, kcy), kps in pose_kps_by_person.items():
                            d = math.sqrt((pcx-kcx)**2 + (pcy-kcy)**2)
                            if d < best_dist:
                                best_dist = d
                                best_match = kps

                        if best_match:
                            pose_keypoints_2d = best_match
                            # Back-project keypoints to 3D
                            pose_keypoints_3d = []
                            for kp in best_match:
                                ku, kv, kvis = kp
                                if kvis > 0.4:
                                    ku_c = clamp(ku, 0, W-1)
                                    kv_c = clamp(kv, 0, H-1)
                                    kz = z
                                    if depth_map is not None:
                                        kz_est = robust_patch_depth(
                                            depth_map, ku_c, kv_c,
                                            patch=3, z_near=args.z_near,
                                            z_far=args.z_far
                                        )
                                        if kz_est is not None:
                                            kz = kz_est
                                    kp3 = backproject(ku_c, kv_c, kz,
                                                       args.fx, args.fy,
                                                       args.cx, args.cy)
                                    pose_keypoints_3d.append(kp3 + [float(kvis)])
                                else:
                                    pose_keypoints_3d.append([0, 0, 0, 0])

                            # Yaw from shoulders
                            ls = best_match[5]  # left shoulder
                            rs = best_match[6]  # right shoulder
                            if ls[2] > 0.4 and rs[2] > 0.4:
                                dx = rs[0] - ls[0]
                                dy = rs[1] - ls[1]
                                yaw = math.atan2(-dy, dx) - math.pi/2

                    det = {
                        "group": group,
                        "subclass": subclass,
                        "phrase": phrase,
                        "bbox_xyxy": bbox,
                        "score": score,
                        "position_3d": pos_3d,
                        "yaw_rad": float(yaw),
                        "scale": 1.0,
                        "speed_value": speed_value,
                        "traffic_light_info": traffic_light_info,
                        "pose_keypoints_2d": pose_keypoints_2d,
                        "pose_keypoints_3d": pose_keypoints_3d,
                        "has_pose": pose_keypoints_2d is not None,
                        "_raw_lights": raw_lights,
                    }
                    frame_dets.append(det)

            # ---- Track ----
            tracked = tracker.update(frame_dets)

            # ---- Debug draw ----
            if writer is not None:
                vis = frame.copy()
                draw_debug(vis, tracked, pose_results)
                writer.write(vis)

            # ---- Store ----
            all_frames.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps,
                "detections": tracked,
            })

            dt = time.time() - t0
            if processed % 5 == 0 or processed == 0:
                counts = {}
                for t in tracked:
                    g = t["group"]
                    counts[g] = counts.get(g, 0) + 1
                summary = " ".join(f"{g}={c}" for g, c in sorted(counts.items()))
                print(f"frame={frame_idx}/{total} total={len(tracked)} "
                      f"[{summary}] time={dt:.2f}s")

            frame_idx += 1
            processed += 1

    finally:
        cap.release()
        if writer:
            writer.release()

    # ---- JSON output ----
    output = {
        "video_path": args.video_input,
        "fps": fps,
        "width": W,
        "height": H,
        "camera_intrinsics": {
            "fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy
        },
        "depth_mode": "pseudo_metric_midas" if depth_est else "dummy",
        "detection_models": {
            "detector": "GroundingDINO",
            "depth": "MiDaS",
            "pose": "YOLOv8-Pose" if yolo_pose else "none",
            "ocr": "EasyOCR" if OCR_AVAILABLE else "none",
        },
        "asset_mapping": {
            "sedan": "SedanAndHatchback.blend",
            "hatchback": "SedanAndHatchback.blend",
            "suv": "SUV.blend",
            "pickup_truck": "PickupTruck.blend",
            "truck": "Truck.blend",
            "bicycle": "Bicycle.blend",
            "motorcycle": "Motorcycle.blend",
            "pedestrian": "Pedestrain.blend",
            "stop_sign": "StopSign.blend",
            "speed_limit": "SpeedLimitSign.blend",
            "traffic_light": "TrafficSignal.blend",
            "dustbin": "Dustbin.blend",
            "traffic_cone": "TrafficConeAndCylinder.blend",
            "traffic_cylinder": "TrafficConeAndCylinder.blend",
            "traffic_pole": "TrafficAssets.blend",
        },
        "pose_keypoint_names": [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ],
        "pose_connections": COCO_POSE_CONNECTIONS,
        "frames": all_frames,
    }

    # Strip internal tracking fields before serializing
    for fr in all_frames:
        for d in fr["detections"]:
            d.pop("_raw_lights", None)
            d.pop("light_history", None)

    out_json = os.path.join(args.output_dir, "detections_everything.json")
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[DONE] JSON: {out_json}")
    print(f"[DONE] Processed {processed} frames")
    if writer:
        print(f"[DONE] Debug: {os.path.join(args.output_dir, 'debug_everything.mp4')}")


if __name__ == "__main__":
    main()