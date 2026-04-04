# Phase 1 — What We Built

**EinsteinVision Phase 1**: Baseline perception + lane detection + Blender 3D rendering pipeline.

---

## Goal Recap

Transform 13 Tesla dashcam sequences (~30s each, front-facing monocular) into FSD-style 3D visualizations rendered in Blender. Phase 1 focused on getting real detections out of the videos and getting them into Blender as a rendered scene.

---

## What Phase 1 Delivered

### 1. Integrated Perception Pipeline (`cv_p3/Code/phase1_pipeline.py`)

A unified script that runs three detectors on every frame of every scene:

| Detector | Model | Detects |
|---|---|---|
| `VehicleDetector` | YOLOv5 (COCO pretrained) | Cars, motorcycles, buses, trucks, pedestrians |
| `LaneDetector` | CLRNet | Lane polylines with 2D coordinates |
| `TrafficSignDetector` | YOLOv5 | Traffic lights, stop signs, ground markings |

**Usage:**
```bash
python phase1_pipeline.py --scene 1        # single scene
python phase1_pipeline.py --scene all      # all 13 scenes
```

**Outputs per scene:**
- Annotated video with overlaid bounding boxes and lane lines
- Per-frame JSON: `{ frame_idx, detections: [{ type, class, confidence, bbox_2d }], lanes: [...] }`

---

### 2. Lane Detection & Export (`cv_p3/RCNN_lane_detection/lane_export.py`)

Mask R-CNN–based lane segmentation that goes beyond simple polylines:

- Runs Mask R-CNN to get per-pixel lane segmentation masks
- Fits quadratic curves to each mask blob
- Samples equidistant 2D points along each fitted curve
- Projects 2D lane points onto the ground plane using camera intrinsics + pitch angle → `points_ground_vehicle` (3D vehicle-frame coordinates)
- Exports `lane_report_style.json` (9.3 MB covering all 13 scenes)
- Also renders `lane_overlay_report_style.mp4` (1.6 MB) showing the detections on top of the original video

**lane_report_style.json schema:**
```json
{
  "meta": {
    "fps": 30, "frame_width": 1280, "frame_height": 960,
    "class_names": ["background", "solid", "dashed"],
    "description": "Mask R-CNN lanes → mask → quadratic fit → equidistant points"
  },
  "frames": [{
    "frame_idx": 0,
    "lanes": [{
      "label_name": "solid-line",
      "score": 0.95,
      "points_ground_vehicle": [[x, 0, z], ...],
      "points_2d": [[u, v], ...]
    }]
  }]
}
```

---

### 3. Data Fusion Module (`einsteinvision/fusion.py`) — Complete

Projects 2D detections to 3D world coordinates for Blender consumption.

**Key classes:**
- `CameraIntrinsics(fx, fy, cx, cy)` — pinhole camera model
- `SceneFusion` — per-frame fusion orchestrator
  - `pixel_to_camera(pixel, depth)` → 3D camera-space point
  - `camera_to_world_point(cam_pt)` → world-space point via extrinsics
  - `fuse_frame(perception_output)` → `FrameFusionRecord`
  - `export_json()` → scene JSON (schema v1.0)

**Output JSON schema (v1.0):**
```json
{
  "schema_version": "1.0",
  "frames": [{
    "frame_index": 0,
    "timestamp_s": 0.033,
    "objects": [{
      "object_id": "track_123",
      "class_name": "vehicle",
      "position_xyz_m": [1.5, 0.0, 10.0],
      "orientation_yaw_rad": 0.1
    }]
  }]
}
```

---

### 4. Blender Lane Renderer (`einsteinvision/blender_render.py` — `LaneRenderer`)

Reads `lane_report_style.json` and creates animated NURBS spline curves in Blender, one per detected lane per frame.

**Features:**
- NURBS curves with configurable resolution and bevel depth
- Emissive materials with per-label colors matching real road markings:

| Label | Color |
|---|---|
| solid-line | White |
| dashed-line | White |
| dotted-line | Yellow |
| double-line | Yellow |
| divider-line | Orange |
| random-line | Light gray |

- Visibility keyframes so each lane curve is only visible on its detection frame
- Coordinate system conversion: vehicle frame (X=right, Y=up, Z=fwd) → Blender world (X=right, Y=fwd, Z=up)
- Fallback chain if ground-plane points aren't available:
  1. `points_ground_vehicle` (preferred — from lane_export.py)
  2. `points_3d_camera` + Y-flip
  3. Project `points_2d` onto ground plane using camera model + pitch

**SLURM job:** `run_blender_lanes.sbatch` — batch renders all 13 scenes on the cluster

---

### 5. Blender Actor Renderer (`einsteinvision/blender_render.py` — `BlenderSceneRenderer`)

Framework for placing detected vehicles/pedestrians as keyframed objects in Blender.

- Creates Blender collections per scene
- `get_or_create_actor(track_id, class_name)` — manages per-track object lifecycle
- `apply_keyframe(actor, frame, position, yaw)` — sets location + rotation keyframes
- Graceful fallback: if no 3D asset exists for a class, creates a colored cube primitive
- `render_from_json(fused_json_path)` — reads fused scene JSON and drives the full scene

**Status:** Framework complete; 3D asset library integration pending (Phase 2).

---

### 6. Supporting Scripts

| Script | Purpose |
|---|---|
| `cv_p3/Code/detect3d_video.py` | 3D bounding box estimation wrapper around YOLO detections |
| `cv_p3/Code/Traffic_detection.py` | Dedicated traffic light + sign detector with state extraction |
| `cv_p3/Code/video_clrnet_infer.py` | CLRNet per-frame inference on raw video |
| `einsteinvision/lane_render_standalone.py` | Standalone lane renderer (no full Blender scene setup) |
| `install_blender.sh` | Blender install + bpy setup script for cluster nodes |
| `run_lane_export.sbatch` | SLURM job for lane detection on all sequences |
| `run_blender_lanes.sbatch` | SLURM job for Blender lane rendering |

---

### 7. Module Stubs for Phase 2+ (`einsteinvision/`)

Architectural scaffolding is in place for the remaining pipeline stages. These are typed, Protocol-based stubs — interfaces are defined but inference is not yet implemented.

| Module | Key Stubs | Intended Model |
|---|---|---|
| `preprocessing.py` | `CameraCalibrator`, `FrameUndistorter` | OpenCV checkerboard calibration |
| `perception.py` | `Yolo12ObjectDetector`, `RFDetrObjectDetector`, `DepthAnythingEstimator` | YOLOv12, RF-DETR, Depth Anything V2 |

The stubs define all data contracts (`BoundingBox2D`, `DetectedObject2D`, `LaneSegmentationOutput`, `DepthEstimationOutput`, etc.) and Protocol interfaces (`ObjectDetector`, `DepthEstimator`, `LaneSegmenter`, `OpticalFlowEstimator`) so Phase 2 can drop in real models without changing downstream code.

---

## Submodules Present (Ready for Phase 2 Integration)

| Submodule | Location | Purpose |
|---|---|---|
| CLRNet | `cv_p3/CLRNet/` | Lane detection (integrated in Phase 1) |
| detectron2 | `cv_p3/detectron2/` | Mask R-CNN backbone for lane segmentation |
| Detic | `cv_p3/Detic/` | Open-vocabulary detection |
| OpenPifPaf | `cv_p3/openpifpaf/` | Pedestrian pose estimation (keypoints) |
| YOLO-3D | `cv_p3/YOLO-3D/` | 3D bounding box prediction |
| monoloco | `cv_p3/monoloco/` | Monocular 3D localization |

---

## Actual Outputs Generated

- `cv_p3/lane_out/lane_report_style.json` — 9.3 MB, all 13 scenes, all frames, all lanes
- `cv_p3/lane_out/lane_overlay_report_style.mp4` — 1.6 MB overlay video
- `cv_p3/blender_renders/scene{1–13}/` — per-scene Blender render outputs

---

## What Phase 1 Does NOT Include

These are explicitly deferred to later phases:

- Camera calibration from checkerboard video (Module 1 stub only)
- Frame undistortion (Module 1 stub only)
- Depth estimation — `DepthAnythingEstimator` stub exists but no inference
- Multi-object tracking — no DeepSORT/ByteTrack integration yet
- Intent recognition — no brake light / turn signal detection
- Collision prediction / TTC metrics
- Pedestrian pose keypoint rendering
- Real 3D asset loading in Blender (currently uses cube primitives)
- Dynamic shading (brake lights, collision state coloring)

---

## Coordinate Systems Used Throughout

| Frame | X | Y | Z |
|---|---|---|---|
| Vehicle / navigation | Right | Up | Forward |
| Camera (OpenCV) | Right | Down | Forward |
| Blender world | Right | Forward | Up |

All three transforms are implemented and consistent across `fusion.py` and `blender_render.py`.

---

## Key Design Decisions

1. **Protocol-based perception interfaces** — any detector that satisfies the `ObjectDetector` protocol works without changing downstream modules
2. **YAML-driven model selection** (`configs/model_config.yaml`) — swap models without code changes
3. **Fallback chain in lane renderer** — gracefully handles cases where 3D ground points aren't available, falls back to 2D→ground projection
4. **Per-frame visibility keyframes in Blender** — avoids cluttering the scene; each detected lane is visible only on its detection frame
5. **SLURM-ready scripts** — pipeline was designed to run on a compute cluster from day one

---

## Phase 2 Starting Point

The next phase should build on these foundations:

1. **Implement `DepthAnythingEstimator`** in `perception.py` — this unblocks real 3D object placement
2. **Wire up multi-object tracking** (DeepSORT or ByteTrack) to give objects consistent `track_id` across frames
3. **Load actual 3D assets** in `BlenderSceneRenderer.instantiate_asset()` instead of cube primitives
4. **Camera calibration** — implement `CameraCalibrator.calibrate_from_video()` using calibration videos in `cv_p3/P3Data/Calib/`

See [ARCHITECTURE.md](ARCHITECTURE.md) for full module specs, data tensor shapes, and suggested models for each stage.
