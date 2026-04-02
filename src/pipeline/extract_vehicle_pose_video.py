#!/usr/bin/env python3
import os
import sys
import cv2
import json
import math
import time
import glob
import shutil
import argparse
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

# =========================================================
# Pre-parse Detic root before imports
# =========================================================
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--detic-root", default=os.path.expanduser("~/Detic"))
_pre_args, _ = _pre_parser.parse_known_args()

DETIC_ROOT = os.path.abspath(os.path.expanduser(_pre_args.detic_root))
CENTERNET2_ROOT = os.path.join(DETIC_ROOT, "third_party", "CenterNet2")

if DETIC_ROOT not in sys.path:
    sys.path.insert(0, DETIC_ROOT)
if CENTERNET2_ROOT not in sys.path:
    sys.path.insert(0, CENTERNET2_ROOT)

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo


# =========================================================
# Data classes
# =========================================================
@dataclass
class SceneObs:
    frame_idx: int
    timestamp: float
    det_idx: int
    cls_name: str
    score: float
    bbox_xyxy: List[float]
    detic_centroid_uv: List[float]
    keypoints_2d: List[List[float]]
    keypoints_3d: List[List[float]]
    position_cam_xyz: List[float]
    yaw_rad: float
    scale: float
    yaw_valid: bool
    pose_method: str
    traffic_light_state: str = "unknown"
    sign_state: str = "unknown"


@dataclass
class TrackState:
    track_id: int
    cls_name: str
    last_frame_idx: int
    xyz: np.ndarray
    vel: np.ndarray
    yaw: float
    scale: float
    age: int
    hits: int
    yaw_valid: bool
    pose_method: str
    traffic_light_state: str = "unknown"
    sign_state: str = "unknown"


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def bbox_centroid_xyxy(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def backproject(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float):
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def robust_patch_depth(
    depth_map: np.ndarray,
    u: float,
    v: float,
    patch: int = 5,
    z_near: float = 3.0,
    z_far: float = 45.0,
):
    """
    Convert MiDaS normalized depth to a pseudo-metric depth range.
    IMPORTANT: this is NOT true metric depth. It is only a heuristic mapping.
    """
    h, w = depth_map.shape[:2]
    uu = int(round(u))
    vv = int(round(v))
    x1 = clamp(uu - patch, 0, w - 1)
    x2 = clamp(uu + patch, 0, w - 1)
    y1 = clamp(vv - patch, 0, h - 1)
    y2 = clamp(vv + patch, 0, h - 1)

    patch_vals = depth_map[y1:y2 + 1, x1:x2 + 1].reshape(-1)
    patch_vals = patch_vals[np.isfinite(patch_vals)]
    patch_vals = patch_vals[patch_vals > 1e-6]
    if len(patch_vals) == 0:
        return None

    d = float(np.median(patch_vals))
    z = z_near + d * (z_far - z_near)
    return z


def similarity_transform_2d(template_pts: np.ndarray, observed_pts: np.ndarray):
    assert template_pts.shape == observed_pts.shape
    n = template_pts.shape[0]
    if n < 3:
        return None

    mu_t = template_pts.mean(axis=0)
    mu_o = observed_pts.mean(axis=0)

    Xt = template_pts - mu_t
    Xo = observed_pts - mu_o

    cov = Xt.T @ Xo / n
    U, D, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_t = np.sum(Xt ** 2) / n
    if var_t < 1e-12:
        return None

    s = np.sum(D) / var_t
    t = mu_o - s * (R @ mu_t)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return s, R, t, yaw


def smooth_angle(prev: float, new: float, alpha: float = 0.3):
    d = math.atan2(math.sin(new - prev), math.cos(new - prev))
    return prev + alpha * d


def normalize_label(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def is_vehicle_label(name: str) -> bool:
    n = normalize_label(name)
    vehicle_aliases = {
        "car", "truck", "bus", "motorcycle", "bicycle",
        "van", "pickup truck", "pickup", "automobile",
        "suv", "minivan"
    }
    if n in vehicle_aliases:
        return True
    if n == "car" or n.endswith(" car"):
        return True
    if "truck" in n:
        return True
    if "bus" in n:
        return True
    if "motorcycle" in n:
        return True
    if "bicycle" in n or n == "bike":
        return True
    return False


def is_person_label(name: str) -> bool:
    n = normalize_label(name)
    person_aliases = {
        "person",
        "pedestrian",
        "man",
        "woman",
        "boy",
        "girl",
        "people",
    }
    if n in person_aliases:
        return True
    if "pedestrian" in n:
        return True
    return False


def is_traffic_light_label(name: str) -> bool:
    n = normalize_label(name)
    return n == "traffic light" or ("traffic" in n and "light" in n)


def is_road_sign_label(name: str) -> bool:
    n = normalize_label(name)
    sign_aliases = {
        "stop sign",
        "yield sign",
        "speed limit sign",
        "street sign",
        "road sign",
        "traffic sign",
        "warning sign",
    }
    if n in sign_aliases:
        return True
    if "sign" in n:
        return True
    return False


def is_supported_target_label(name: str) -> bool:
    return (
        is_vehicle_label(name)
        or is_person_label(name)
        or is_traffic_light_label(name)
        or is_road_sign_label(name)
    )


def looks_like_placeholder(path: str) -> bool:
    if not path:
        return True
    lowered = path.lower()
    return ("actual/path" in lowered) or ("whatever_model" in lowered) or ("placeholder" in lowered)


def resolve_detic_weights(args) -> str:
    requested = os.path.abspath(os.path.expanduser(args.detic_weights)) if args.detic_weights else ""
    requested_base = os.path.basename(requested) if requested else ""

    if requested and (not looks_like_placeholder(requested)) and os.path.isfile(requested):
        return requested

    models_dir = os.path.join(args.detic_root, "models")
    all_pths = sorted(glob.glob(os.path.join(models_dir, "*.pth")))
    if not all_pths:
        raise FileNotFoundError(f"No .pth files found in {models_dir}")

    cfg_base = os.path.splitext(os.path.basename(args.detic_config))[0]
    exact_stem = os.path.join(models_dir, cfg_base + ".pth")
    if os.path.isfile(exact_stem):
        return exact_stem

    scored = []
    for p in all_pths:
        base = os.path.splitext(os.path.basename(p))[0]
        score = 0
        for token in ["LCOCOI21k", "LI21k", "LbaseI", "SwinB", "R5021k", "R18", "CXT21k", "max-size"]:
            if token in cfg_base and token in base:
                score += 3
        cfg_parts = cfg_base.replace("-", "_").split("_")
        base_parts = base.replace("-", "_").split("_")
        score += len(set(cfg_parts) & set(base_parts))
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = scored[0]

    if requested_base:
        basename_candidate = os.path.join(models_dir, requested_base)
        if os.path.isfile(basename_candidate):
            return basename_candidate

    if best_score <= 0:
        raise FileNotFoundError(
            f"Detic checkpoint not found for config {args.detic_config}. "
            f"Available checkpoints: {all_pths}"
        )

    return best_path


# =========================================================
# Traffic light color classification
# =========================================================
def classify_traffic_light_color(
    frame_bgr: np.ndarray,
    bbox_xyxy: List[float],
    min_color_pixels: int = 20,
) -> Dict[str, Any]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return {"state": "unknown", "scores": {}}

    crop = frame_bgr[y1:y2 + 1, x1:x2 + 1]
    ch, cw = crop.shape[:2]
    if ch < 6 or cw < 4:
        return {"state": "unknown", "scores": {}}

    sx1 = int(0.25 * cw)
    sx2 = int(0.75 * cw)
    sy1 = int(0.10 * ch)
    sy2 = int(0.90 * ch)
    roi = crop[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return {"state": "unknown", "scores": {}}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    bright_sat = (S > 80) & (V > 100)

    red_mask = (((H <= 12) | (H >= 170)) & bright_sat).astype(np.uint8)
    yellow_mask = (((H >= 15) & (H <= 40)) & bright_sat).astype(np.uint8)
    green_mask = (((H >= 40) & (H <= 95)) & bright_sat).astype(np.uint8)

    thirds = np.array_split(np.arange(roi.shape[0]), 3)

    def region_score(mask, rows):
        if len(rows) == 0:
            return 0.0, 0
        sub = mask[rows[0]:rows[-1] + 1, :]
        vsub = V[rows[0]:rows[-1] + 1, :]
        count = int(np.count_nonzero(sub))
        if count == 0:
            return 0.0, 0
        mean_v = float(vsub[sub > 0].mean()) if np.count_nonzero(sub) > 0 else 0.0
        score = count * (mean_v / 255.0)
        return score, count

    red_score, red_count = region_score(red_mask, thirds[0])
    yellow_score, yellow_count = region_score(yellow_mask, thirds[1])
    green_score, green_count = region_score(green_mask, thirds[2])

    scores = {
        "red": red_score,
        "yellow": yellow_score,
        "green": green_score,
    }
    counts = {
        "red": red_count,
        "yellow": yellow_count,
        "green": green_count,
    }

    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]

    if best_score <= 0.0 or counts[best_state] < min_color_pixels:
        return {"state": "unknown", "scores": scores}

    return {"state": best_state, "scores": scores}


# =========================================================
# Detic setup
# =========================================================
def setup_cfg(args):
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.detic_config)
    cfg.merge_from_list(args.detic_opts)

    cfg.MODEL.DEVICE = "cpu" if args.cpu else "cuda"
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.detic_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.detic_thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.detic_thresh
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    cfg.freeze()
    return cfg


# =========================================================
# Depth with MiDaS
# =========================================================
class DepthEstimator:
    def __init__(self, model_name: str = "DPT_Hybrid", device: str = "cuda"):
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"[INFO] Loading MiDaS model: {model_name} on {self.device}")
        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_name in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        elif model_name == "MiDaS_small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.default_transform

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.detach().cpu().numpy().astype(np.float32)

        dmin = float(depth.min())
        dmax = float(depth.max())
        if dmax - dmin > 1e-8:
            depth = (depth - dmin) / (dmax - dmin)
        else:
            depth = np.zeros_like(depth, dtype=np.float32)

        depth = 1.0 - depth
        depth = np.clip(depth, 1e-3, 1.0)
        return depth


# =========================================================
# OpenPifPaf via CLI
# =========================================================
class OpenPifPafCLI:
    def __init__(
        self,
        checkpoint: str,
        python_exec: str = "python",
        instance_threshold: float = 0.05,
        seed_threshold: float = 0.05,
    ):
        self.checkpoint = checkpoint
        self.python_exec = python_exec
        self.instance_threshold = instance_threshold
        self.seed_threshold = seed_threshold
        self._tmpdir = tempfile.mkdtemp(prefix="opifpaf_tmp_")
        self._counter = 0

    def close(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _collect_json_candidates(self, base_dir: str) -> List[str]:
        candidates = []
        direct = os.path.join(base_dir, "out.json")
        if os.path.isfile(direct):
            candidates.append(direct)
        candidates.extend(sorted(glob.glob(os.path.join(base_dir, "*.json"))))
        seen = set()
        uniq = []
        for p in candidates:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq

    def infer_crop(self, crop_bgr: np.ndarray) -> List[Dict[str, Any]]:
        run_dir = os.path.join(self._tmpdir, f"run_{self._counter:06d}")
        self._counter += 1
        os.makedirs(run_dir, exist_ok=True)

        img_path = os.path.join(run_dir, "crop.png")
        json_target = os.path.join(run_dir, "out.json")
        ok = cv2.imwrite(img_path, crop_bgr)
        if not ok:
            return []

        cmd = [
            self.python_exec, "-m", "openpifpaf.predict",
            img_path,
            "--checkpoint", self.checkpoint,
            "--json-output", json_target,
            "--instance-threshold", str(self.instance_threshold),
            "--seed-threshold", str(self.seed_threshold),
            "--line-width", "2",
            "--font-size", "0",
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            print("[WARN] OpenPifPaf timed out")
            return []

        if proc.returncode != 0:
            print("[WARN] OpenPifPaf failed:\n", proc.stderr)
            return []

        candidate_jsons = self._collect_json_candidates(run_dir)
        if not candidate_jsons:
            return []

        merged = []
        for jp in candidate_jsons:
            try:
                with open(jp, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    merged.append(data)
                elif isinstance(data, list):
                    merged.extend(data)
            except Exception as e:
                print(f"[WARN] Failed reading OpenPifPaf JSON {jp}: {e}")

        return merged


# =========================================================
# Parsing helpers
# =========================================================
def parse_pifpaf_keypoints(pred: Dict[str, Any]) -> np.ndarray:
    kp = pred.get("keypoints", [])
    kp = np.array(kp, dtype=np.float32)
    if kp.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    kp = kp.reshape(-1, 3)
    return kp


def parse_pifpaf_bbox(pred: Dict[str, Any]) -> Optional[List[float]]:
    bbox = pred.get("bbox", None)
    if bbox is None:
        return None
    return xywh_to_xyxy(bbox)


def get_metadata_class_names(demo, args) -> List[str]:
    if hasattr(demo, "metadata"):
        md = demo.metadata
        if hasattr(md, "thing_classes") and md.thing_classes:
            return list(md.thing_classes)
        if hasattr(md, "classes") and md.classes:
            return list(md.classes)

    if args.detic_vocabulary == "custom" and args.detic_custom_vocabulary.strip():
        return [c.strip() for c in args.detic_custom_vocabulary.split(",")]

    return []


def write_tracks_snapshot(path, video_path, fps, width, height, fx, fy, cx, cy, frames):
    out = {
        "video_path": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "camera_intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "depth_mode": "pseudo_metric_midas_scaled",
        "frames": frames,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def summarize_raw_instances(instances, class_names: List[str], topk: int = 10) -> str:
    if len(instances) == 0:
        return "raw=0"
    pred_classes = instances.pred_classes.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    pairs = []
    for i in range(min(len(pred_classes), topk)):
        cls_idx = int(pred_classes[i])
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        pairs.append(f"{cls_name}:{scores[i]:.2f}")
    return f"raw={len(pred_classes)} top={pairs}"


def detic_target_instances(instances, class_names: List[str]):
    pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    pred_classes = instances.pred_classes.cpu().numpy()

    masks = None
    if instances.has("pred_masks"):
        masks = instances.pred_masks.cpu().numpy()

    out = []
    for i in range(len(pred_boxes)):
        cls_idx = int(pred_classes[i])
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        if not is_supported_target_label(cls_name):
            continue
        out.append({
            "det_idx": i,
            "bbox_xyxy": pred_boxes[i].tolist(),
            "score": float(scores[i]),
            "cls_name": cls_name,
            "mask": masks[i] if masks is not None else None,
        })
    return out


# =========================================================
# Template fitting
# =========================================================
def load_template_points(path: str) -> Dict[int, np.ndarray]:
    raw = load_json(path)
    out = {}
    for k, v in raw.items():
        out[int(k)] = np.array(v, dtype=np.float32)
    return out


def estimate_pose_from_keypoints(
    keypoints_2d: np.ndarray,
    depth_map: np.ndarray,
    template_points: Dict[int, np.ndarray],
    fx: float, fy: float, cx: float, cy: float,
    kp_conf_thresh: float = 0.15,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], List[List[float]], List[List[float]]]:
    obs_template_2d = []
    obs_scene_2d = []
    keypoints_2d_out = []
    keypoints_3d_out = []

    for i, (u, v, c) in enumerate(keypoints_2d):
        if c < kp_conf_thresh:
            continue
        if i not in template_points:
            continue

        z = robust_patch_depth(depth_map, u, v, patch=4)
        if z is None:
            continue

        p3 = backproject(u, v, z, fx, fy, cx, cy)
        keypoints_2d_out.append([float(u), float(v), float(c)])
        keypoints_3d_out.append(p3.tolist())

        t_local = template_points[i]
        obs_template_2d.append([float(t_local[0]), float(t_local[2])])
        obs_scene_2d.append([float(p3[0]), float(p3[2])])

    if len(obs_scene_2d) < 3:
        return None, None, None, keypoints_2d_out, keypoints_3d_out

    template_xz = np.array(obs_template_2d, dtype=np.float32)
    scene_xz = np.array(obs_scene_2d, dtype=np.float32)

    fit = similarity_transform_2d(template_xz, scene_xz)
    if fit is None:
        return None, None, None, keypoints_2d_out, keypoints_3d_out

    s, R, t, yaw = fit

    ys = []
    for i, (u, v, c) in enumerate(keypoints_2d):
        if c < kp_conf_thresh or i not in template_points:
            continue
        z = robust_patch_depth(depth_map, u, v, patch=4)
        if z is None:
            continue
        p3 = backproject(u, v, z, fx, fy, cx, cy)
        ys.append(float(p3[1] - s * template_points[i][1]))

    ty = 0.0 if len(ys) == 0 else float(np.median(np.array(ys, dtype=np.float32)))
    translation = np.array([t[0], ty, t[1]], dtype=np.float32)
    return translation, float(yaw), float(s), keypoints_2d_out, keypoints_3d_out


# =========================================================
# Association
# =========================================================
def associate_pifpaf_to_detic(
    crop_preds: List[Dict[str, Any]],
    x1: int,
    y1: int,
    detic_bbox_xyxy: List[float],
):
    if len(crop_preds) == 0:
        return None

    det_c = bbox_centroid_xyxy(detic_bbox_xyxy)
    best = None
    best_d = 1e18

    for pred in crop_preds:
        bbox = parse_pifpaf_bbox(pred)
        kp = parse_pifpaf_keypoints(pred)
        if kp.shape[0] == 0:
            continue

        if bbox is not None:
            bbox_full = [bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1]
            c = bbox_centroid_xyxy(bbox_full)
        else:
            valid = kp[:, 2] > 0.05
            if np.sum(valid) == 0:
                continue
            c = np.array(
                [np.mean(kp[valid, 0]) + x1, np.mean(kp[valid, 1]) + y1],
                dtype=np.float32,
            )

        d = np.linalg.norm(c - det_c)
        if d < best_d:
            best_d = d
            best = pred

    if best is None:
        return None

    kp = parse_pifpaf_keypoints(best)
    if kp.shape[0] > 0:
        kp[:, 0] += x1
        kp[:, 1] += y1

    best = dict(best)
    best["_keypoints_full"] = kp.tolist()
    return best


# =========================================================
# Tracker
# =========================================================
class SimpleTracker:
    def __init__(self, dist_thresh=6.0, alpha_pos=0.35, alpha_yaw=0.25, max_age=12):
        self.next_id = 1
        self.tracks: Dict[int, TrackState] = {}
        self.dist_thresh = dist_thresh
        self.alpha_pos = alpha_pos
        self.alpha_yaw = alpha_yaw
        self.max_age = max_age

    def update(self, frame_idx: int, observations: List[SceneObs]) -> List[Dict[str, Any]]:
        used_tracks = set()
        outputs = []

        for obs in observations:
            p = np.array(obs.position_cam_xyz, dtype=np.float32)
            best_tid = None
            best_d = 1e18

            for tid, tr in self.tracks.items():
                if normalize_label(tr.cls_name) != normalize_label(obs.cls_name):
                    continue
                if tid in used_tracks:
                    continue

                pred = tr.xyz + tr.vel
                d = np.linalg.norm(pred[[0, 2]] - p[[0, 2]])
                if d < best_d and d < self.dist_thresh:
                    best_d = d
                    best_tid = tid

            if best_tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = TrackState(
                    track_id=tid,
                    cls_name=obs.cls_name,
                    last_frame_idx=frame_idx,
                    xyz=p.copy(),
                    vel=np.zeros(3, dtype=np.float32),
                    yaw=obs.yaw_rad,
                    scale=obs.scale,
                    age=0,
                    hits=1,
                    yaw_valid=obs.yaw_valid,
                    pose_method=obs.pose_method,
                    traffic_light_state=obs.traffic_light_state,
                    sign_state=obs.sign_state,
                )
            else:
                tid = best_tid
                tr = self.tracks[tid]
                new_xyz = (1.0 - self.alpha_pos) * tr.xyz + self.alpha_pos * p
                tr.vel = new_xyz - tr.xyz
                tr.xyz = new_xyz

                if obs.yaw_valid:
                    if tr.yaw_valid:
                        tr.yaw = smooth_angle(tr.yaw, obs.yaw_rad, self.alpha_yaw)
                    else:
                        tr.yaw = obs.yaw_rad
                        tr.yaw_valid = True

                tr.scale = 0.8 * tr.scale + 0.2 * obs.scale
                tr.last_frame_idx = frame_idx
                tr.hits += 1
                tr.traffic_light_state = obs.traffic_light_state
                tr.sign_state = obs.sign_state
                if obs.pose_method != "fallback_bottom_center":
                    tr.pose_method = obs.pose_method

            used_tracks.add(tid)
            tr = self.tracks[tid]
            outputs.append({
                "track_id": tid,
                "class": obs.cls_name,
                "position_cam_xyz": tr.xyz.tolist(),
                "yaw_rad": float(tr.yaw),
                "yaw_valid": bool(tr.yaw_valid),
                "scale": float(tr.scale),
                "pose_method": tr.pose_method,
                "bbox_xyxy": obs.bbox_xyxy,
                "score": obs.score,
                "keypoints_2d": obs.keypoints_2d,
                "keypoints_3d": obs.keypoints_3d,
                "traffic_light_state": tr.traffic_light_state,
                "sign_state": tr.sign_state,
            })

        to_delete = []
        for tid, tr in self.tracks.items():
            if tid not in used_tracks:
                tr.age += 1
            else:
                tr.age = 0
            if tr.age > self.max_age:
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return outputs


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-input", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)

    parser.add_argument("--detic-root", default=os.path.expanduser("~/Detic"))
    parser.add_argument("--detic-config", required=True)
    parser.add_argument("--detic-weights", default="")
    parser.add_argument("--detic-thresh", type=float, default=0.18)
    parser.add_argument(
        "--detic-vocabulary",
        default="coco",
        choices=["custom", "coco", "lvis", "openimages", "objects365"],
    )
    parser.add_argument(
        "--detic-custom-vocabulary",
        default="car,truck,bus,motorcycle,bicycle,person,traffic light,stop sign,traffic cone",
    )
    parser.add_argument("--detic-opts", nargs=argparse.REMAINDER, default=[])

    parser.add_argument("--pifpaf-checkpoint", default="shufflenetv2k16-apollo-24")
    parser.add_argument("--pifpaf-python", default="python")
    parser.add_argument("--pifpaf-instance-threshold", type=float, default=0.05)
    parser.add_argument("--pifpaf-seed-threshold", type=float, default=0.05)

    parser.add_argument("--depth-model", default="DPT_Hybrid")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--car-template-json", default="")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--debug-video", action="store_true")
    parser.add_argument("--log-raw-classes", action="store_true")
    parser.add_argument("--write-every-frame", action="store_true")

    args = parser.parse_args()

    args.detic_root = os.path.abspath(os.path.expanduser(args.detic_root))
    args.detic_config = os.path.abspath(os.path.expanduser(args.detic_config))
    if args.car_template_json:
        args.car_template_json = os.path.abspath(os.path.expanduser(args.car_template_json))

    if not os.path.isdir(args.detic_root):
        raise FileNotFoundError(f"Detic root directory not found: {args.detic_root}")
    if not os.path.isfile(args.detic_config):
        raise FileNotFoundError(f"Detic config not found: {args.detic_config}")
    if not os.path.isfile(args.video_input):
        raise FileNotFoundError(f"Video not found: {args.video_input}")

    detic_metadata_file = os.path.join(
        args.detic_root, "datasets", "metadata", "lvis_v1_train_cat_info.json"
    )
    if not os.path.isfile(detic_metadata_file):
        raise FileNotFoundError(f"Missing Detic metadata file: {detic_metadata_file}")

    if "MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH" not in args.detic_opts:
        args.detic_opts += ["MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH", detic_metadata_file]

    args.detic_weights = resolve_detic_weights(args)

    ensure_dir(args.output_dir)
    setup_logger(name="fvcore")
    logger = setup_logger()

    logger.info(f"Using Detic root: {args.detic_root}")
    logger.info(f"Using Detic config: {args.detic_config}")
    logger.info(f"Using Detic weights: {args.detic_weights}")

    cfg = setup_cfg(args)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.detic_weights
    cfg.freeze()

    args.vocabulary = args.detic_vocabulary
    args.custom_vocabulary = args.detic_custom_vocabulary
    args.pred_all_class = False
    args.confidence_threshold = args.detic_thresh
    args.input = None
    args.output = None
    args.webcam = False

    prev_cwd = os.getcwd()
    os.chdir(args.detic_root)
    try:
        demo = VisualizationDemo(cfg, args)
    finally:
        os.chdir(prev_cwd)

    class_names = get_metadata_class_names(demo, args)
    logger.info(f"Detic metadata classes loaded: {len(class_names)}")
    if class_names:
        logger.info(f"First few classes: {class_names[:20]}")
    else:
        logger.warning("Could not read Detic class names from metadata; class IDs may appear numerically.")

    device = "cpu" if args.cpu else "cuda"
    depth_estimator = DepthEstimator(args.depth_model, device=device)

    template_points = None
    use_template_pose = False
    if args.car_template_json:
        if os.path.isfile(args.car_template_json):
            template_points = load_template_points(args.car_template_json)
            use_template_pose = True
            logger.info(f"Loaded template points from: {args.car_template_json}")
        else:
            logger.warning(f"--car-template-json provided but file not found: {args.car_template_json}")

    pifpaf = None
    if use_template_pose:
        pifpaf = OpenPifPafCLI(
            checkpoint=args.pifpaf_checkpoint,
            python_exec=args.pifpaf_python,
            instance_threshold=args.pifpaf_instance_threshold,
            seed_threshold=args.pifpaf_seed_threshold,
        )
        logger.info("OpenPifPaf enabled because template pose fitting is available.")
    else:
        logger.info("OpenPifPaf disabled because no --car-template-json was provided.")

    tracker = SimpleTracker()

    cap = cv2.VideoCapture(args.video_input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    writer = None
    if args.debug_video:
        out_path = os.path.join(args.output_dir, "debug_pose_overlay_with_lights.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    all_frames = []
    frame_idx = args.start_frame
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.max_frames > 0 and processed >= args.max_frames:
                break

            t0 = time.time()
            depth = depth_estimator.predict(frame)

            predictions, _ = demo.run_on_image(frame)
            if "instances" not in predictions:
                all_frames.append({
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / max(fps, 1.0),
                    "tracks": [],
                })
                frame_idx += 1
                processed += 1
                continue

            inst = predictions["instances"].to("cpu")

            if args.log_raw_classes and frame_idx < args.start_frame + 10:
                logger.info(f"frame={frame_idx} {summarize_raw_instances(inst, class_names, topk=12)}")

            detections = detic_target_instances(inst, class_names)

            obs_list = []
            vis = frame.copy()

            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
                x1 = clamp(x1, 0, width - 1)
                x2 = clamp(x2, 0, width - 1)
                y1 = clamp(y1, 0, height - 1)
                y2 = clamp(y2, 0, height - 1)
                if x2 <= x1 or y2 <= y1:
                    continue

                translation = None
                yaw = 0.0
                scale = 1.0
                yaw_valid = False
                pose_method = "fallback_bottom_center"
                k2d, k3d = [], []
                traffic_light_state = "unknown"
                sign_state = "unknown"

                cls_norm = normalize_label(det["cls_name"])

                if is_vehicle_label(cls_norm):
                    if pifpaf is not None and template_points is not None:
                        crop = frame[y1:y2, x1:x2]
                        crop_preds = pifpaf.infer_crop(crop)
                        match = associate_pifpaf_to_detic(crop_preds, x1, y1, det["bbox_xyxy"])

                        if match is not None:
                            kp_full = np.array(match["_keypoints_full"], dtype=np.float32)
                            translation, yaw, scale, k2d, k3d = estimate_pose_from_keypoints(
                                kp_full,
                                depth,
                                template_points,
                                args.fx, args.fy, args.cx, args.cy,
                            )
                            if translation is not None:
                                yaw_valid = True
                                pose_method = "template_keypoints"

                    if translation is None:
                        x1b, y1b, x2b, y2b = det["bbox_xyxy"]
                        cu = 0.5 * (x1b + x2b)
                        cv = min(y2b - 4.0, height - 1.0)
                        z = robust_patch_depth(depth, cu, cv, patch=8)
                        if z is None:
                            continue
                        translation = backproject(cu, cv, z, args.fx, args.fy, args.cx, args.cy)
                        yaw = 0.0
                        scale = 1.0
                        yaw_valid = False
                        pose_method = "fallback_bottom_center"
                        k2d, k3d = [], []

                elif is_person_label(cls_norm):
                    x1b, y1b, x2b, y2b = det["bbox_xyxy"]
                    cu = 0.5 * (x1b + x2b)
                    cv = y1b + 0.92 * (y2b - y1b)
                    cv = min(cv, height - 1.0)

                    z = robust_patch_depth(depth, cu, cv, patch=6)
                    if z is None:
                        cu = 0.5 * (x1b + x2b)
                        cv = 0.5 * (y1b + y2b)
                        z = robust_patch_depth(depth, cu, cv, patch=6)
                    if z is None:
                        continue

                    translation = backproject(cu, cv, z, args.fx, args.fy, args.cx, args.cy)
                    yaw = 0.0
                    scale = max(0.3, float(y2b - y1b) / 180.0)
                    yaw_valid = False
                    pose_method = "person_bottom_center"
                    k2d, k3d = [], []

                elif is_traffic_light_label(cls_norm):
                    color_info = classify_traffic_light_color(frame, det["bbox_xyxy"])
                    traffic_light_state = color_info["state"]

                    cu = 0.5 * (det["bbox_xyxy"][0] + det["bbox_xyxy"][2])
                    cv = 0.5 * (det["bbox_xyxy"][1] + det["bbox_xyxy"][3])
                    z = robust_patch_depth(depth, cu, cv, patch=4)
                    if z is None:
                        continue
                    translation = backproject(cu, cv, z, args.fx, args.fy, args.cx, args.cy)
                    yaw = 0.0
                    scale = 1.0
                    yaw_valid = False
                    pose_method = "traffic_light_bbox_center"

                elif is_road_sign_label(cls_norm):
                    sign_state = cls_norm
                    cu = 0.5 * (det["bbox_xyxy"][0] + det["bbox_xyxy"][2])
                    cv = 0.5 * (det["bbox_xyxy"][1] + det["bbox_xyxy"][3])

                    z = robust_patch_depth(depth, cu, cv, patch=5)
                    if z is None:
                        continue

                    translation = backproject(cu, cv, z, args.fx, args.fy, args.cx, args.cy)
                    yaw = 0.0
                    scale = 1.0
                    yaw_valid = False
                    pose_method = "road_sign_bbox_center"

                else:
                    continue

                obs = SceneObs(
                    frame_idx=frame_idx,
                    timestamp=frame_idx / max(fps, 1.0),
                    det_idx=det["det_idx"],
                    cls_name=det["cls_name"],
                    score=det["score"],
                    bbox_xyxy=det["bbox_xyxy"],
                    detic_centroid_uv=bbox_centroid_xyxy(det["bbox_xyxy"]).tolist(),
                    keypoints_2d=k2d,
                    keypoints_3d=k3d,
                    position_cam_xyz=translation.tolist(),
                    yaw_rad=float(yaw),
                    scale=float(scale),
                    yaw_valid=bool(yaw_valid),
                    pose_method=pose_method,
                    traffic_light_state=traffic_light_state,
                    sign_state=sign_state,
                )
                obs_list.append(obs)

            tracks = tracker.update(frame_idx, obs_list)

            if writer is not None:
                for tr in tracks:
                    x1, y1, x2, y2 = [int(v) for v in tr["bbox_xyxy"]]

                    if is_traffic_light_label(tr["class"]):
                        state = tr.get("traffic_light_state", "unknown")
                        if state == "red":
                            color = (0, 0, 255)
                        elif state == "yellow":
                            color = (0, 255, 255)
                        elif state == "green":
                            color = (0, 255, 0)
                        else:
                            color = (255, 255, 255)

                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        txt = (
                            f"id={tr['track_id']} {tr['class']} "
                            f"{state} z={tr['position_cam_xyz'][2]:.2f}"
                        )
                        cv2.putText(
                            vis, txt, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )

                    elif is_person_label(tr["class"]):
                        color = (255, 0, 255)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        txt = f"id={tr['track_id']} {tr['class']} z={tr['position_cam_xyz'][2]:.2f}"
                        cv2.putText(
                            vis, txt, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )

                    elif is_road_sign_label(tr["class"]):
                        color = (255, 165, 0)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        txt = (
                            f"id={tr['track_id']} {tr['class']} "
                            f"z={tr['position_cam_xyz'][2]:.2f}"
                        )
                        cv2.putText(
                            vis, txt, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )

                    else:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        yaw_text = f"{math.degrees(tr['yaw_rad']):.1f}" if tr["yaw_valid"] else "NA"
                        txt = (
                            f"id={tr['track_id']} {tr['class']} "
                            f"z={tr['position_cam_xyz'][2]:.2f} "
                            f"yaw={yaw_text}"
                        )
                        cv2.putText(
                            vis, txt, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                        )
                        for kp in tr["keypoints_2d"]:
                            u, v, c = kp
                            cv2.circle(vis, (int(u), int(v)), 2, (0, 0, 255), -1)

                writer.write(vis)

            all_frames.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / max(fps, 1.0),
                "tracks": tracks,
            })

            if args.write_every_frame:
                tracks_json_path = os.path.join(args.output_dir, "scene_tracks_with_lights.json")
                write_tracks_snapshot(
                    tracks_json_path,
                    args.video_input,
                    fps,
                    width,
                    height,
                    args.fx,
                    args.fy,
                    args.cx,
                    args.cy,
                    all_frames,
                )

            dt = time.time() - t0
            logger.info(
                f"frame={frame_idx}/{total_frames} raw={len(inst)} "
                f"targets={len(obs_list)} tracks={len(tracks)} time={dt:.2f}s"
            )

            frame_idx += 1
            processed += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if pifpaf is not None:
            pifpaf.close()

    out = {
        "video_path": args.video_input,
        "fps": fps,
        "width": width,
        "height": height,
        "camera_intrinsics": {
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
        },
        "depth_mode": "pseudo_metric_midas_scaled",
        "frames": all_frames,
    }

    out_json_path = os.path.join(args.output_dir, "scene_tracks_with_lights.json")
    save_json(out, out_json_path)
    print(f"[INFO] wrote: {out_json_path}")


if __name__ == "__main__":
    main()