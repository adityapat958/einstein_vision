# EinsteinVision

Modular computer vision and 3D visualization pipeline for FSD-style autonomous vehicle perception and Blender 3D reconstruction from monocular dashcam video.

## Quick Start

**Goal**: Transform 13 Tesla dashcam videos into FSD-style 3D visualizations with moving vehicles, pedestrians, traffic infrastructure, and dynamic intent indicators.

**Inputs**: 13 monocular videos (front-facing) + camera calibration video  
**Outputs**: 13 rendered 3D scene videos + frame-wise perception JSON

## Key Documents

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Complete system design with 6-module pipeline, data tensor flows, suggested models (YOLOv11, Depth Anything V2, RAFT, DeepSORT, etc.), and phase deliverables.
- **[src/einsteinvision/](src/einsteinvision/)**: Python package with modular stubs for each pipeline stage.

## Architecture Overview

```
Raw Video → Preprocessing → Perception → Localization → Temporal → Fusion → Blender Render
  (13×30s)   (Calibration)   (2D Detect)   (Depth, 3D)  (Tracking)  (JSON)      (MP4)
```

## Core Modules

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `preprocessing.py` | Camera calibration, lens undistortion | `CameraCalibrator`, `FrameUndistorter` |
| `perception.py` | 2D detection, tracking, pose | `PerceptionPipeline`, `DetectedObject2D`, `BoundingBox2D` |
| `fusion.py` | 2D→3D projection, world-space fusion | `SceneFusion`, `CameraIntrinsics`, `FrameFusionRecord` |
| `blender_render.py` | Blender scene setup, keyframing, dynamic shading | `BlenderSceneRenderer`, `BlenderRenderConfig` |

## Implementation Phases

- **Phase 1**: Baseline perception + static 3D placement
- **Phase 2**: Multi-object tracking + temporal animation
- **Phase 3**: Intent recognition + collision prediction + dynamic shading
- **Extra Credit**: Pedestrian pose, lane topology, trajectory prediction, custom assets

## Dependencies (TBD)

- **CV/DL**: OpenCV, PyTorch, YOLOv11, Depth Anything V2, RAFT, DeepSORT
- **Rendering**: Blender 4.0+, bpy
- **Utilities**: NumPy, SciPy, EasyOCR, Pandas
