# Phase 2 — Implementation Plan & Status

**Branch**: `phase2` (branched off `origin/phase1`)

---

## What Phase 2 Adds

| Capability | Phase 1 | Phase 2 |
|---|---|---|
| Lane detection | Mask R-CNN → Blender NURBS (scene1 only) | All 13 scenes, multi-lane, min-score fixed |
| Vehicle detection | YOLO 2D bbox | YOLOv11 + depth → real 3D positions |
| Vehicle sub-class | "car" only | sedan / SUV / hatchback / pickup / truck / bicycle / motorcycle |
| 3D placement | None (fusion stub) | Depth Anything V2 + ground-plane scale calibration |
| Object tracking | No persistent IDs | Ultralytics native tracker → stable track_ids |
| Pedestrians | bbox only | OpenPifPaf 17-point COCO skeleton → 3D in Blender |
| Traffic lights | bbox detection | State (red/green/yellow) + arrow direction (left/straight/right/off) |
| Road signs | Stop sign bbox | Speed limit sign + OCR (extract the number) + ground arrows |
| Additional objects | None | Dustbin, traffic cone/cylinder, traffic pole (via Detic) |
| Blender assets | Cube primitives | Real .blend assets from P3Data/Assets/ per sub-class |
| Motion state | None | parked / slow / moving → material shading in Blender |
| Intent | None | Brake lights (red glow), turn signals (orange blink at ~1.5 Hz) |
| Blender rendering | On cluster (broken) | **Locally** via render_phase2_local.sh |

---

## Phase 1 Bugs Fixed

| Bug | Fix | File |
|---|---|---|
| Only driving lane in Blender (JSON path mismatch) | Re-run `run_lane_export.sbatch` for all 13 scenes | `run_lane_export.sbatch` |
| min-score 0.3 filters valid lanes | Changed to 0.1 | `run_blender_lanes.sbatch` |
| YOLO-3D hardcoded `/home/alien/` paths | Added argparse | `cv_p3/YOLO-3D/run.py` |
| Depth is relative [0-1], not metric | Ground-plane scale calibration | `cv_p3/YOLO-3D/depth_model.py` |
| YOLO-3D has no JSON output | Added per-frame JSON export | `cv_p3/YOLO-3D/run.py` |
| `load_camera_params.py` only reads JSON | Added `.mat` loading via scipy | `cv_p3/YOLO-3D/load_camera_params.py` |

---

## Architecture

```
Cluster (Turing)                           Local Machine
─────────────────────────────────          ─────────────────────────
submit run_lane_export.sbatch              render_phase2_local.sh
  → lane_out/sceneN/                         reads JSONs from cluster
    lane_report_style.json (7-class)
                                             Blender
submit run_phase2.sbatch                       LaneRenderer (NURBS curves)
  → phase2_output/sceneN/                     BlenderSceneRenderer (assets)
    detections.json (v2.0)                     PoseRenderer (skeleton)
                                               Motion shading + intent
                                             → cv_p3/blender_renders/sceneN.mp4
```

---

## New Files Created

| File | Purpose |
|---|---|
| `cv_p3/Code/phase2_pipeline.py` | Main Phase 2 orchestrator (all 13 scenes) |
| `cv_p3/Code/vehicle_subclassifier.py` | EfficientNet-B0 car sub-class + heuristic fallback |
| `cv_p3/Code/traffic_light_classifier.py` | HSV state + arrow direction classifier |
| `cv_p3/Code/road_sign_detector.py` | Ground arrow detector + EasyOCR speed limit |
| `cv_p3/Code/pose_estimator.py` | OpenPifPaf 17-point COCO skeleton |
| `run_phase2.sbatch` | SLURM GPU job for phase2_pipeline.py |
| `render_phase2_local.sh` | Local Blender render + ffmpeg encode for all 13 scenes |

## Modified Files

| File | What Changed |
|---|---|
| `cv_p3/YOLO-3D/run.py` | Full rewrite: argparse, JSON export, metric depth |
| `cv_p3/YOLO-3D/depth_model.py` | Added `calibrate_metric_scale()` + `to_metric()` |
| `cv_p3/YOLO-3D/load_camera_params.py` | Added `.mat` loading + `export_params_to_json()` |
| `einsteinvision/perception.py` | Added `sub_class`, `arrow_direction`, `speed_limit_value` to `DetectedObject2D`; added `PoseKeypoints`, `PoseKeypoints3D` |
| `einsteinvision/blender_render.py` | Full Phase 2 rewrite: real asset loading, pose rendering, motion/intent shading |
| `run_blender_lanes.sbatch` | `--min-score 0.3` → `0.1` |

---

## detections.json Schema (v2.0)

```json
{
  "schema_version": "2.0",
  "scene": 1,
  "fps": 30.0,
  "frames": [{
    "frame_index": 0,
    "timestamp_s": 0.0,
    "objects": [{
      "object_id": "42",
      "class_name": "car",
      "sub_class": "sedan",
      "confidence": 0.87,
      "bbox_2d": [120, 340, 280, 450],
      "position_xyz_m": [1.5, 0.0, 18.3],
      "orientation_yaw_rad": 0.05,
      "dimensions_m": null,
      "motion_state": "moving",
      "intent": {"brake_lights_on": false, "turn_signal": "none"},
      "traffic_light_state": null,
      "arrow_direction": null,
      "speed_limit_value": null
    }],
    "poses": [{
      "person_track_id": "pose_0",
      "keypoints_2d": [[u, v], ...],
      "keypoints_scores": [0.9, ...],
      "keypoints_3d": [[x, y, z], ...]
    }],
    "ground_arrows": [{
      "direction": "straight",
      "bbox": [400, 600, 600, 700],
      "score": 0.75
    }]
  }]
}
```

---

## How to Run

### Step 1 — Lane export (cluster, if not already done)
```bash
sbatch run_lane_export.sbatch
# or: bash submit_lane_export.sh
```
Verify: `ls cv_p3/lane_out/scene{1..13}/lane_report_style.json`

### Step 2 — Phase 2 pipeline (cluster)
```bash
sbatch run_phase2.sbatch
# Monitor: tail -f phase2_<jobid>.out
```
Verify: `ls phase2_output/scene{1..13}/detections.json`

### Step 3 — Blender rendering (local machine)
```bash
# Sync JSONs from cluster first:
# rsync -av turing:~/repos/cv/p3/phase2_output ./
# rsync -av turing:~/repos/cv/p3/cv_p3/lane_out ./cv_p3/

bash render_phase2_local.sh           # all 13 scenes
bash render_phase2_local.sh 1         # just scene 1 (test first)
```

---

## Dependencies to Install (on cluster)

```bash
conda activate rl

# Core (likely already installed)
pip install ultralytics torch torchvision opencv-python

# Phase 2 additions
pip install openpifpaf          # pedestrian pose
pip install easyocr             # speed limit OCR
pip install scipy               # calibration.mat loading
pip install transformers        # Depth Anything V2 (via HuggingFace)
pip install timm                # Depth Anything V2 backbone
```

---

## Open Items / Next Steps

1. **Vehicle sub-class weights**: `VehicleSubClassifier` defaults to heuristic mode (aspect ratio). For better accuracy, fine-tune EfficientNet-B0 on Stanford Cars dataset and pass `--sub-classifier-weights` to `phase2_pipeline.py`.

2. **Metric depth validation**: After running on cluster, check that `depth_metric_scale` in `detections.json` is plausible (~10–20). If scale is wrong, verify `camera_height` and `camera_pitch_deg` args.

3. **calibration.mat structure**: The `.mat` loader tries several common field names. If the front-camera K matrix is not parsed correctly, inspect it with `python3 -c "import scipy.io; import pprint; pprint.pprint(scipy.io.loadmat('cv_p3/calibration.mat'))"` and update `_load_from_mat()` accordingly.

4. **Detic integration**: `GroundArrowDetector` currently uses colour+contour analysis. For higher accuracy, swap in Detic with custom vocabulary `["ground arrow", "lane arrow"]`.

5. **Asset inventory**: Check `cv_p3/P3Data/Assets/Vehicles/` for exact mesh object names inside each `.blend` — the `_append_blend_object()` function loads the first mesh object. If a .blend contains multiple objects, name the target one explicitly.
