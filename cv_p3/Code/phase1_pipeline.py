#!/usr/bin/env python3
"""
Phase 1 Unified Detection Pipeline for EinsteinVision Project
==============================================================
Combines:
- Vehicle/Pedestrian detection (YOLO + 3D bbox estimation)
- Lane detection (CLRNet)
- Traffic sign/light detection (YOLO)

Outputs:
- Annotated video with all detections overlaid
- JSON file per scene with frame-by-frame detection data for Blender

Usage:
    python phase1_pipeline.py --scene 1
    python phase1_pipeline.py --scene all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# NumPy compatibility for older codebases
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# Project paths
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
CODE_ROOT = FILE.parents[0]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# CLRNET paths
CLRNET_ROOT = PROJECT_ROOT / "CLRNet"
if str(CLRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(CLRNET_ROOT))

# PyTorch 2.6+ compatibility patch
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Scene video mapping
SCENE_VIDEOS = {
    1: "2023-02-14_11-04-07-front_undistort.mp4",
    2: "2023-03-03_10-31-11-front_undistort.mp4",
    3: "2023-02-14_11-49-54-front_undistort.mp4",
    4: "2023-02-14_11-51-54-front_undistort.mp4",
    5: "2023-02-14_11-56-56-front_undistort.mp4",
    6: "2023-03-03_15-31-56-front_undistort.mp4",
    7: "2023-03-03_11-21-43-front_undistort.mp4",
    8: "2023-03-03_11-40-47-front_undistort.mp4",
    9: "2023-03-04_17-20-36-front_undistort.mp4",
    10: "2023-03-06_19-48-30-front_undistort.mp4",
    11: "2023-03-11_17-19-53-front_undistort.mp4",
    12: "2023-03-13_06-00-16-front_undistort.mp4",
    13: "2023-03-03_06-59-50-front_undistort.mp4",
}


class VehicleDetector:
    """YOLO-based vehicle and pedestrian detector with 3D bbox estimation."""

    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    PEDESTRIAN_CLASSES = [0]  # person

    def __init__(self, weights_path, device="cuda"):
        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        from utils.torch_utils import select_device

        self.device = select_device(device)
        self.model = DetectMultiBackend(
            weights_path,
            device=self.device,
            dnn=False,
            data=str(PROJECT_ROOT / "data" / "coco128.yaml")
        )
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((640, 640), s=stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz), half=False)
        self.stride = stride
        self.names = names
        self.pt = pt

    @torch.no_grad()
    def detect(self, frame, conf_thres=0.35, iou_thres=0.45):
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_coords

        im0 = frame.copy()
        im = letterbox(im0, new_shape=self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im[:, :, ::-1].transpose(2, 0, 1)
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, conf_thres=conf_thres, iou_thres=iou_thres,
            classes=self.VEHICLE_CLASSES + self.PEDESTRIAN_CLASSES,
            agnostic=False, max_det=100
        )

        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    xyxy_ = [int(x.item()) for x in xyxy]
                    c = int(cls.item())
                    label = self.names[c]
                    obj_type = "pedestrian" if c in self.PEDESTRIAN_CLASSES else "vehicle"
                    detections.append({
                        "type": obj_type,
                        "class": label,
                        "confidence": float(conf.item()),
                        "bbox_2d": xyxy_,
                    })
        return detections


class LaneDetector:
    """CLRNet-based lane detector."""

    def __init__(self, config_path, checkpoint_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        from clrnet.utils.config import Config
        from clrnet.models.registry import build_net
        from clrnet.utils.net_utils import load_network

        self.cfg = Config.fromfile(config_path)
        self.cfg.load_from = checkpoint_path
        self.cfg.view = False

        self.model = build_net(self.cfg)
        load_network(self.model, checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.img_w = getattr(self.cfg, "img_w", 800)
        self.img_h = getattr(self.cfg, "img_h", 320)

    def _normalize(self, img_rgb):
        img = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return img.transpose(2, 0, 1)

    @torch.no_grad()
    def detect(self, frame):
        orig_h, orig_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (self.img_w, self.img_h))
        tensor = self._normalize(resized)
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        img_metas = [{
            "img_name": "frame",
            "ori_shape": (orig_h, orig_w, 3),
            "img_shape": (self.img_h, self.img_w, 3),
            "scale_factor": np.array([orig_w / self.img_w, orig_h / self.img_h], dtype=np.float32),
        }]

        try:
            predictions = self.model(img=tensor, img_metas=img_metas, return_loss=False)
        except:
            try:
                predictions = self.model(tensor, img_metas=img_metas, return_loss=False)
            except:
                predictions = None

        lanes = []
        if predictions is not None:
            lane_list = predictions[0] if isinstance(predictions, (list, tuple)) else [predictions]
            for lane in lane_list:
                try:
                    if hasattr(lane, "to_array"):
                        arr = lane.to_array(self.cfg)
                    elif isinstance(lane, np.ndarray):
                        arr = lane
                    elif torch.is_tensor(lane):
                        arr = lane.detach().cpu().numpy()
                    else:
                        continue

                    arr = np.asarray(arr)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        # Scale to original size
                        pts = arr[:, :2].copy().astype(np.float32)
                        sx = float(orig_w) / float(self.img_w)
                        sy = float(orig_h) / float(self.img_h)
                        pts[:, 0] *= sx
                        pts[:, 1] *= sy
                        lanes.append(pts.tolist())
                except:
                    continue

        return lanes


class TrafficSignDetector:
    """YOLO-based traffic sign and light detector."""

    def __init__(self, weights_path, device="cuda"):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.device = device

    def detect(self, frame, conf_thres=0.25):
        results = self.model(frame, conf=conf_thres, verbose=False, device=self.device)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
                cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
                label = self.model.names[cls_id] if cls_id in self.model.names else f"class_{cls_id}"

                detections.append({
                    "type": "traffic_sign",
                    "class": label,
                    "confidence": conf,
                    "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                })
        return detections


def draw_detections(frame, vehicle_dets, lanes, traffic_dets):
    """Draw all detections on frame."""
    vis = frame.copy()

    # Draw lanes
    lane_colors = [
        (0, 255, 0),    # green
        (255, 255, 0),  # cyan
        (255, 0, 255),  # magenta
        (0, 255, 255),  # yellow
    ]
    for i, lane in enumerate(lanes):
        color = lane_colors[i % len(lane_colors)]
        pts = [(int(p[0]), int(p[1])) for p in lane if not (np.isnan(p[0]) or np.isnan(p[1]))]
        for j in range(1, len(pts)):
            cv2.line(vis, pts[j-1], pts[j], color, 3)

    # Draw vehicles/pedestrians
    for det in vehicle_dets:
        x1, y1, x2, y2 = det["bbox_2d"]
        color = (0, 0, 255) if det["type"] == "pedestrian" else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.putText(vis, label, (x1, max(25, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw traffic signs
    for det in traffic_dets:
        x1, y1, x2, y2 = det["bbox_2d"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.putText(vis, label, (x1, max(25, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return vis


def process_scene(scene_num, output_dir, device="cuda"):
    """Process a single scene with all detectors."""

    print(f"\n{'='*60}")
    print(f"Processing Scene {scene_num}")
    print(f"{'='*60}")

    # Paths
    video_name = SCENE_VIDEOS.get(scene_num)
    if video_name is None:
        print(f"ERROR: Scene {scene_num} not found in mapping")
        return False

    video_path = PROJECT_ROOT / "P3Data" / "Sequences" / f"scene{scene_num}" / "Undist" / video_name
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return False

    output_video = Path(output_dir) / f"scene{scene_num}_phase1_detections.mp4"
    output_json = Path(output_dir) / f"scene{scene_num}_detections.json"

    # Initialize detectors
    print("Loading models...")

    vehicle_detector = VehicleDetector(
        weights_path=str(PROJECT_ROOT / "ext_models" / "yolov5s.pt"),
        device=device
    )

    try:
        lane_detector = LaneDetector(
            config_path=str(CLRNET_ROOT / "configs" / "clrnet" / "clr_resnet18_tusimple.py"),
            checkpoint_path=str(PROJECT_ROOT / "ext_models" / "69.pth"),
            device=device
        )
        has_lane_detector = True
    except Exception as e:
        print(f"WARNING: Could not load lane detector: {e}")
        has_lane_detector = False
        lane_detector = None

    try:
        traffic_detector = TrafficSignDetector(
            weights_path=str(PROJECT_ROOT / "ext_models" / "traffic_sign_detector.pt"),
            device=device
        )
        has_traffic_detector = True
    except Exception as e:
        print(f"WARNING: Could not load traffic sign detector: {e}")
        has_traffic_detector = False
        traffic_detector = None

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_frames}")

    # Output writer
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    # Process frames
    all_frame_data = []
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = {
            "frame_index": frame_idx,
            "timestamp_s": frame_idx / fps,
            "vehicles": [],
            "pedestrians": [],
            "lanes": [],
            "traffic_signs": [],
        }

        # Vehicle/pedestrian detection
        vehicle_dets = vehicle_detector.detect(frame)
        for det in vehicle_dets:
            if det["type"] == "pedestrian":
                frame_data["pedestrians"].append(det)
            else:
                frame_data["vehicles"].append(det)

        # Lane detection
        lanes = []
        if has_lane_detector and lane_detector is not None:
            lanes = lane_detector.detect(frame)
            frame_data["lanes"] = lanes

        # Traffic sign detection
        traffic_dets = []
        if has_traffic_detector and traffic_detector is not None:
            traffic_dets = traffic_detector.detect(frame)
            frame_data["traffic_signs"] = traffic_dets

        all_frame_data.append(frame_data)

        # Draw and write
        vis = draw_detections(frame, vehicle_dets, lanes, traffic_dets)
        cv2.putText(vis, f"Frame: {frame_idx}/{total_frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        writer.write(vis)

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed
            eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames} | {fps_actual:.1f} FPS | ETA: {eta:.0f}s")

    cap.release()
    writer.release()

    # Save JSON
    scene_data = {
        "scene": scene_num,
        "video_file": video_name,
        "total_frames": frame_idx,
        "fps": fps,
        "resolution": [width, height],
        "frames": all_frame_data,
    }

    with open(output_json, "w") as f:
        json.dump(scene_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    elapsed = time.time() - start_time
    print(f"\nScene {scene_num} completed in {elapsed:.1f}s")
    print(f"  Output video: {output_video}")
    print(f"  Output JSON: {output_json}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Unified Detection Pipeline")
    parser.add_argument("--scene", type=str, default="all",
                        help="Scene number (1-13) or 'all'")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "phase1_output"),
                        help="Output directory")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device ID or 'cpu'")
    args = parser.parse_args()

    device = args.device
    if device.isdigit():
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Output directory: {args.output_dir}")

    if args.scene.lower() == "all":
        scenes = list(SCENE_VIDEOS.keys())
    else:
        scenes = [int(s.strip()) for s in args.scene.split(",")]

    print(f"Processing scenes: {scenes}")

    success_count = 0
    for scene_num in scenes:
        try:
            if process_scene(scene_num, args.output_dir, device):
                success_count += 1
        except Exception as e:
            print(f"ERROR processing scene {scene_num}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count}/{len(scenes)} scenes successful")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
