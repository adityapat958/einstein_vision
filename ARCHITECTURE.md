# EinsteinVision System Architecture
## Modular 3D Autonomous Vehicle Perception & Visualization Pipeline

**Project Goal**: Reconstruct FSD-style 3D dashboards from monocular Tesla dashcam video using multi-stage perception, temporal fusion, and procedural Blender rendering.

**Inputs**: 13 monocular videos (front-facing) + calibration videos  
**Outputs**: 13 rendered 3D scene videos with ego-car, dynamic actors, infrastructure, and motion state

---

## 1. System Overview & Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAW VIDEO INGESTION (13 sequences)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULE 1: Data Preprocessing & Calibration                         │
│  • Camera intrinsics extraction (checkerboard cal videos)            │
│  • Lens distortion removal (per-frame undistortion)                  │
│  • Frame normalization (size, color space, temporal alignment)       │
│  Output: Undistorted RGB frames + camera matrix (K, dist_coeffs)    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULE 2: 2D Perception & Classification                           │
│  • Dynamic actors detection & subclassification (RF-DETR / YOLOv12 / Mask R-CNN)│
│  • Pedestrian detection & pose estimation (RTMPose / MediaPipe)     │
│  • Infrastructure recognition (traffic lights, signs, arrows)        │
│  • Road object detection (cones, poles, signs)                      │
│  Output: Per-frame 2D bboxes, keypoints, class labels, confidence   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULE 3: 3D Localization & Geometry                               │
│  • Monocular depth estimation (Depth Anything V2)                   │
│  • 6D pose recovery (perspective-n-point, iterative refinement)     │
│  • Lane boundary detection & semantic classification (PolyLaneNet)  │
│  Output: Per-object 3D positions, orientations; lane curves (splines)│
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULE 4: Temporal & Motion Analysis                               │
│  • Dense optical flow (RAFT) for motion segmentation                │
│  • Multi-object tracking (DeepSORT / ByteTrack)                     │
│  • Intent recognition (brake lights, turn signals, epipolar geo)    │
│  • Predictive analytics (TTC, collision forecast)                   │
│  Output: Tracking IDs, motion state, intent flags, collision risk   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULE 5: Data Serialization & Fusion                              │
│  • Aggregate 2D, 3D, semantic, temporal data per frame              │
│  • Serialize into Frame Records (JSON)                              │
│  • Export scene graph with object trajectories                      │
│  Output: Frame-by-frame JSON with all detections, poses, motion flags│
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODULE 6: Procedural 3D Rendering (Blender)                        │
│  • Parse JSON + load 3D asset library                               │
│  • Place objects in world space (inverse extrinsics)                │
│  • Procedurally generate lane meshes (Geometry Nodes)               │
│  • Apply dynamic shading (brake lights, collision state)            │
│  • Keyframe animation across frames                                 │
│  Output: 13 rendered 3D videos (MP4 / EXR sequence)                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Details & Data Contracts

### Module 1: Data Preprocessing & Calibration

**Responsibility**: Extract camera intrinsics and prepare normalized video frames.

**Inputs**:
- `calibration_video.mp4` (checkerboard sequence, 1080p, 30 fps)
- Raw video sequences (13 × ~30 second clips)

**Processing Steps**:
1. **Camera Calibration** (from calibration video)
   - Extract checkerboard corners from ~30–50 calibration frames
   - Compute intrinsic matrix $\mathbf{K}$ using OpenCV's `calibrateCamera()`
   - Extract radial/tangential distortion coefficients ($k_1, k_2, p_1, p_2$)
   - Report reprojection error (<0.5 pixel target)
   
   **Output Tensors**:
   - Camera matrix $\mathbf{K}$ ∈ ℝ^{3×3}$
   - Distortion coefficients $\mathbf{d}$ ∈ ℝ^4$
   - Image size $(W, H)$

2. **Frame Undistortion**
   - Apply `cv2.undistort()` to each raw frame using $\mathbf{K}, \mathbf{d}$
   - Cache undistorted frames for downstream perception modules
   
   **Output Tensors**:
   - Per-frame undistorted image $I_t$ ∈ ℝ^{H×W×3}$ (uint8, RGB/BGR)

**Python Classes**:
- `CameraCalibrator`: Orchestrates checkerboard calibration
- `FrameUndistorter`: Applies lens correction per-frame

**Suggested Implementations**:
- OpenCV 4.8+ for checkerboard detection & calibration
- SciPy for numerical optimization if custom non-linear refinement needed

---

### Module 2: 2D Perception & Classification

**Responsibility**: Detect and classify all visible entities in 2D image space.

**Inputs**:
- Undistorted frame stream $I_t$ ∈ ℝ^{H×W×3}$

**Submodules**:

#### 2.1 Dynamic Actor Detection & Subclassification
**Classes detected**: Sedan, SUV, Hatchback, Pickup, Truck, Bicycle, Motorcycle

**Model Options**:
- **RF-DETR** *(transformer-based DETR variant)*: SOTA >60 COCO mAP, excels at small/far actors, moderate throughput (~12 FPS on A100).
- **YOLOv12 (attention-centric)**: YOLO ecosystem upgrade with area attention to cut redundancy; higher accuracy than YOLOv11 but ~1.5× memory footprint.
- **Mask R-CNN w/ ResNeXt-101** *(two-stage)*: Pixel-perfect masks, best for offline batches where 3–5 FPS is acceptable; boosts infrastructure mask accuracy too.
- **YOLOv11n/m** *(baseline fallback)*: Lightweight option if GPU memory is constrained or for rapid experimentation.
- **MonoCoP / MonoDiff (direct monocular 3D)**: Optional heavy heads that regress 3D boxes directly; integrate when Module 3 depth fusion is replaced with direct 3D nodes.

All detectors consume $I_t$ (typically resized to 1280×768 or 1024×1024) and output bounding boxes, class IDs, and confidences.

**Output Tensors**:
```
Detections ∈ {
  bbox_2d: (N, 4) → [x_min, y_min, x_max, y_max] (pixels)
  class_id: (N,) → {sedan=0, suv=1, hatchback=2, ...}
  confidence: (N,) ∈ [0, 1]
  track_id: (N,) → persistent ID across frames
}
```

#### 2.2 Pedestrian Detection & Pose Estimation
**Suggested Models**:
- **Detector**: RF-DETR (for dense downtown scenes) or YOLOv12-L (when keeping the YOLO toolchain); RT-DETR remains a lighter alternative.
- **Pose Estimator**: RTMPose or MediaPipe Pose
- Extracts 17 keypoints per person (COCO skeleton)

**Output Tensors**:
```
Pedestrians ∈ {
  bbox_2d: (P, 4)
  keypoints_2d: (P, 17, 2) → [x, y] pixel coordinates
  keypoint_confidence: (P, 17) ∈ [0, 1]
  track_id: (P,)
}
```

#### 2.3 Infrastructure Recognition (Traffic Lights, Signs, Ground Markings)
**Subclasses**:
- **Traffic Lights**: Detect bounding box + extract bulb color (red/yellow/green)
- **Directional Arrows**: Detect turn arrow states (left, straight, right)
- **Signs**: Stop signs, speed limit signs (with OCR for digit extraction)
- **Ground Arrows**: Lane direction indicators

**Suggested Models**:
- Custom fine-tuned YOLOv12 or RF-DETR heads on BDD100K traffic light splits for superior small-object recall.
- Mask R-CNN (ResNeXt-101 + FPN) when per-pixel masks for lane arrows/ground text are critical.
- EasyOCR or PaddleOCR for sign digit recognition (speed limits, arrow descriptors).
- Semantic segmentation mask for lane arrows (optional; can use template matching or transformer-based Mask2Former).

**Output Tensors**:
```
Infrastructure ∈ {
  traffic_light_bbox: (TL, 4)
  traffic_light_color: (TL,) → {red=0, yellow=1, green=2, off=3}
  arrow_direction: (TL,) → {off=0, left=1, straight=2, right=3}
  stop_sign_bbox: (SS, 4)
  speed_sign_bbox: (SPD, 4)
  speed_sign_value: (SPD,) → integer (mph/kmh)
  ground_arrow_bbox: (GA, 4)
  ground_arrow_direction: (GA,) → {left=0, straight=1, right=2}
}
```

#### 2.4 Road Object Detection (Cones, Poles, Dustbins, Speed Bumps)
**Suggested Model**: YOLOv12-S/YOLOv12-L with attention blocks or Mask R-CNN when instance masks aid downstream geometry. RF-DETR also works well for densely packed cones.

**Output Tensors**:
```
RoadObjects ∈ {
  bbox_2d: (RO, 4)
  class_id: (RO,) → {cone=0, pole=1, dustbin=2, speed_bump=3}
  confidence: (RO,) ∈ [0, 1]
}
```

**Consolidated Module Output**:
```python
PerceptionFrameOutput = {
  frame_index: int,
  timestamp_s: float,
  detections: {
    dynamic_actors: DetectedObject2D[],
    pedestrians: PedestrianDetection[],
    infrastructure: InfrastructureData,
    road_objects: RoadObjectDetection[]
  }
}
```

#### 2.5 Detector Selection Cheat-Sheet
- **RF-DETR**: Highest mAP, better for crowded urban data; needs transformer-friendly GPUs (24GB+).
- **YOLOv12**: Balanced choice—YOLO ergonomics with attention-centric accuracy bump; expect ~30–40% slower inference than YOLOv11.
- **Mask R-CNN (ResNeXt-101)**: Ideal when pixel masks feed downstream geometry (e.g., skeletonizing ground arrows); run as offline batch.
- **MonoCoP / MonoDiff**: Deploy when Module 3 is re-architected around direct monocular 3D detection (skip depth map fusion entirely).
- **Hybrid Strategy**: Use YOLOv12 for dense 2D detection, then run MonoCoP on the top-N ROIs to refine 3D boxes.

---

### Module 3: 3D Localization & Geometry

**Responsibility**: Lift 2D detections into 3D camera/world space and extract road topology.

**Inputs**:
- Undistorted frames $I_t$ ∈ ℝ^{H×W×3}$
- 2D detection bboxes, keypoints from Module 2
- Camera intrinsics $\mathbf{K}$ from Module 1

**Submodules**:

#### 3.1 Monocular Metric Depth Estimation
**Suggested Model**: Depth Anything V2 (small/base)
- **Input**: $I_t$ (1024×1024 or flexible resolution)
- **Output**: Dense depth map $D_t$ ∈ ℝ^{H×W}$ with metric scale (meters)

**Considerations**:
- Ensure consistency with camera calibration (no mismatch in principal point)
- Post-process for temporal smoothing (Kalman filter or median over time)

**Output Tensors**:
```
DepthMap ∈ {
  depth_m: (H, W) ∈ [0.5, 100.0] meters
  confidence: (H, W) ∈ [0, 1] (optional from model)
}
```

#### 3.2 Monocular 3D Bounding Box Recovery
**Method**:
1. Sample depth at 2D bbox center: $d_{center} = D_t[v_{center}, u_{center}]$
2. Unproject to camera space:
   $$\mathbf{X}_{cam} = \begin{bmatrix} \frac{(u - c_x) \cdot d}{f_x} \\ \frac{(v - c_y) \cdot d}{f_y} \\ d \end{bmatrix}$$
3. Estimate 3D object extent from 2D bbox dimensions using class-specific priors
4. Iterative refinement: Minimize reprojection error to refine 6D pose $(\mathbf{t}, \mathbf{R})$

**Suggested Tools**:
- OpenCV `solvePnP()` with RANSAC for robust pose estimation
- Custom refinement using Gauss-Newton optimization or bundle adjustment

**Output Tensors**:
```
Object3DBBox ∈ {
  position_cam: (3,) → [X, Y, Z] in camera frame
  quaternion: (4,) → rotation quaternion [qx, qy, qz, qw]
  dimensions: (3,) → [length, width, height] in meters
  confidence_6d: float
}
```

#### 3.3 6D Pose Refinement & Tracking-Aware Updates
- Fuse mono 3D estimates with temporal consistency (Kalman filtering)
- Use optical flow residuals to correct pose estimates across frames

**Output Tensors**:
```
Pose6D ∈ {
  translation_cam: (3,)
  rotation_matrix_cam: (3, 3)
  velocity_cam: (3,) → optional, from temporal differencing
  acceleration_cam: (3,) → optional
}
```

#### 3.4 Lane Boundary Detection & Semantic Classification
**Suggested Model**: PolyLaneNet or SCNN (Spatial CNN)
- **Input**: $I_t$
- **Output**: Lane boundary splines + semantic type

**Semantic Classes**:
- Dashed white lane (ego lane boundaries)
- Solid white lane (road boundary)
- Double yellow / solid yellow (center divider, no-pass zone)
- No lane (grass, sidewalk)

**Output Tensors**:
```
LaneData ∈ {
  lane_curves: [Spline[]],  // per lane: sequence of (x, y) points in image
  semantic_type: (NumLanes,) → {dashed_white=0, solid_white=1, ...}
  confidence: (NumLanes,)
}
```

**Convert Lane to 3D Spline**:
1. Unproject lane pixels using depth: $\mathbf{P}_{cam} = D[y, x] \cdot \mathbf{K}^{-1} [x, y, 1]^T$
2. Fit 3D cubic spline (or parametric polynomial) through projected points
3. Export as control points for Geometry Nodes procedural generation

#### 3.5 Direct Monocular 3D Detectors (Optional Track)
Instead of projecting 2D boxes via a depth map, replace Sections 3.1–3.3 with heavy end-to-end 3D detectors:

- **MonoCoP (Chain-of-Prediction)**: Sequentially regresses depth, size, orientation, and velocity. Offers superior consistency on KITTI/nuScenes but consumes ~20GB VRAM and 500 ms/frame.
- **MonoDiff**: Applies reverse diffusion to sample 3D boxes; highly accurate for distant actors and multi-modal predictions. Useful when estimating aleatoric uncertainty for TTC.

**Integration Path**:
1. Feed undistorted frame $I_t$ directly to MonoCoP/MonoDiff.
2. Use intrinsic matrix $\mathbf{K}$ to align their predicted camera coordinates with the fusion schema.
3. Bypass the depth-map lookup in `SceneFusion` by writing `position_3d_cam_m` and `orientation_6d` directly from detector outputs.
4. Optionally keep Depth Anything V2 as a fallback when the 3D detector confidence < τ.

**Trade-offs**:
- Pros: Higher 3D AP, avoids noisy depth scaling, better orientation.
- Cons: Large memory footprint, limited training data, slower inference, more complex fine-tuning.

---

### Module 4: Temporal & Motion Analysis

**Responsibility**: Track entities across frames, infer motion state, and predict intent/collision.

**Inputs**:
- Per-frame 2D detections + 3D poses from Modules 2 & 3
- Sequence of frames $I_{t-1}, I_t, I_{t+1}, \ldots$

**Submodules**:

#### 4.1 Multi-Object Tracking (MOT)
**Suggested Models**:
- **DeepSORT**: Combines Kalman filter + deep appearance embeddings
- **ByteTrack**: Fast, association-by-detection approach
- **YOLO-v11 with native tracker**: Integrated tracking-by-detection

**Output Tensors**:
```
TrackingState ∈ {
  track_id: int → persistent ID
  age: int → frames since birth
  bbox_2d_history: List[(4,)] → recent 2D bboxes
  pos_3d_history: List[(3,)] → recent 3D positions
  velocity_3d: (3,) → m/s in world frame (from temporal differentiation)
  state: {CONFIRMED, TENTATIVE, LOST}
}
```

#### 4.2 Dense Optical Flow Estimation
**Suggested Model**: RAFT (Recurrent All-Pairs Field Transforms)
- **Input**: $I_{t-1}, I_t$ (1024×1024 padded)
- **Output**: Optical flow map $\mathbf{F}_t$ ∈ ℝ^{H×W×2}$

**Output Tensors**:
```
OpticalFlow ∈ {
  flow_uv: (H, W, 2) → [u, v] pixel displacements
  occlusion_mask: (H, W) → binary (0=visible, 1=occluded)
  confidence: (H, W) ∈ [0, 1]
}
```

#### 4.3 Motion Segmentation (Parked vs. Moving)
**Method**:
1. Compute per-object motion magnitude from optical flow within bbox region
2. Use epipolar geometry (Sampson distance) to separate ego-motion from object motion:
   - Estimate fundamental matrix $\mathbf{F}$ using RANSAC on 2D point correspondences
   - Inliers (small Sampson distance) → ego-motion compensated
   - Outliers (large distance) → moving objects
3. Classify as **Parked** (motion < τ) or **Moving** (motion ≥ τ)

**Window-based Filtering**:
- Use 5-frame sliding window to smooth motion state estimates

**Output Tensors**:
```
MotionSegmentation ∈ {
  is_moving: (NumObjects,) → binary
  motion_direction_2d: (NumObjects, 2) → normalized [u, v]
  motion_magnitude_px: (NumObjects,) → pixels/frame
  motion_magnitude_ms: (NumObjects,) → m/s (from depth)
}
```

#### 4.4 Intent Recognition (Brake Lights, Turn Signals, Directional Arrows)
**Brake Light Detection**:
- Monitor bounding box region on vehicle rear (heuristic offset based on 3D pose)
- Template matching or CNN-based light detector on rear patch
- State: ON/OFF, with 3-frame confirmation delay

**Turn Signal Detection**:
- Similar rear-region monitoring for flashing pattern
- State: LEFT_BLINK, RIGHT_BLINK, OFF

**Pedestrian Intent** (optional for Extra Credit):
- Pose skeleton + motion velocity → predict crossing direction
- Combine with gaze estimation (if pose model provides head keypoints)

**Output Tensors**:
```
IntentState ∈ {
  brake_lights_on: bool
  turn_signal: {LEFT=0, RIGHT=1, OFF=2}
  pedestrian_crossing_direction: (2,) → [vx, vy] normalized
  confidence_intent: float
}
```

#### 4.5 Predictive Analytics (Time-to-Collision)
**TTC Computation**:
$$\text{TTC} = \frac{d}{|\mathbf{v}_{rel} \cdot \hat{\mathbf{d}}|} \quad \text{if closing}$$
where:
- $d$ = distance between ego rear bumper and object front
- $\mathbf{v}_{rel}$ = relative velocity (object − ego)
- $\hat{\mathbf{d}}$ = unit direction from ego to object

**Collision Prediction**:
- Threshold TTC < 2 seconds → flag collision risk
- Exponential moving average over 5 frames to smooth spurious alerts

**Output Tensors**:
```
CollisionForecast ∈ {
  ttc_s: float → time-to-collision (inf if no collision)
  is_collision_imminent: bool → TTC < 2s
  collision_confidence: float ∈ [0, 1]
  colliding_object_id: int → optional, if imminent
}
```

---

### Module 5: Data Serialization & Fusion

**Responsibility**: Consolidate all perception, 3D, and temporal data into frame-wise JSON records.

**Inputs**:
- Module 2 outputs (2D detections, classifications)
- Module 3 outputs (3D poses, lane curves)
- Module 4 outputs (tracking, motion, intent, collision)

**JSON Schema**:

```json
{
  "schema_version": "1.0",
  "video_metadata": {
    "video_id": "sequence_001",
    "fps": 30,
    "total_frames": 900,
    "camera_matrix_k": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "distortion_coeffs": [k1, k2, p1, p2]
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_s": 0.0,
      "detections": {
        "dynamic_actors": [
          {
            "track_id": "car_001",
            "class_name": "sedan",
            "confidence_2d": 0.95,
            "bbox_2d_px": [100, 200, 400, 350],
            "position_3d_cam_m": [5.0, -1.2, 10.5],
            "orientation_6d": {
              "rotation_matrix": [[0.99, 0.1, 0.0], [-0.1, 0.99, 0.0], [0.0, 0.0, 1.0]],
              "quaternion_wxyz": [0.995, 0.0, 0.0, 0.1]
            },
            "dimensions_m": [4.7, 1.8, 1.5],
            "velocity_3d_ms": [2.5, 0.0, 0.0],
            "motion_state": "moving",
            "intent": {
              "brake_lights_on": false,
              "turn_signal": "off"
            },
            "collision": {
              "ttc_s": 12.5,
              "is_imminent": false
            }
          }
        ],
        "pedestrians": [
          {
            "track_id": "ped_042",
            "confidence_2d": 0.88,
            "bbox_2d_px": [250, 150, 310, 450],
            "keypoints_2d": [[260, 165], [280, 170], ...],
            "keypoint_conf": [0.95, 0.92, ...],
            "position_3d_cam_m": [2.0, 0.5, 3.0],
            "walking_velocity_ms": [0.8, -0.2, 0.0],
            "collision": {
              "ttc_s": 3.8,
              "is_imminent": true
            }
          }
        ],
        "infrastructure": {
          "traffic_lights": [
            {
              "bbox_2d_px": [500, 100, 530, 160],
              "color": "green",
              "arrow_direction": "straight",
              "position_3d_cam_m": [15.0, 5.0, 20.0]
            }
          ],
          "stop_signs": [
            {
              "bbox_2d_px": [600, 200, 650, 250],
              "position_3d_cam_m": [18.0, 6.0, 22.0]
            }
          ],
          "speed_signs": [
            {
              "bbox_2d_px": [700, 180, 750, 230],
              "speed_value": 35,
              "position_3d_cam_m": [20.0, 7.0, 25.0]
            }
          ]
        },
        "lanes": {
          "lane_curves": [
            {
              "semantic_type": "dashed_white",
              "curve_3d_control_points": [[0, -3.5, 0], [1, -3.5, 1], [2, -3.5, 2], ...],
              "confidence": 0.92
            },
            {
              "semantic_type": "solid_white",
              "curve_3d_control_points": [[0, 3.5, 0], [1, 3.5, 1], [2, 3.5, 2], ...],
              "confidence": 0.90
            }
          ]
        }
      }
    },
    {
      "frame_index": 1,
      "timestamp_s": 0.033,
      "detections": { ... }
    }
  ]
}
```

**Export Process**:
1. Per-frame aggregation of all Module 2, 3, 4 outputs
2. JSON serialization (no numpy arrays; convert to lists)
3. Write one consolidated JSON per video sequence
4. Verify schema compliance before export

**Python Class**:
- `SceneFusion`: Aggregates detections and exports JSON
- Utility functions for dataclass → dict serialization

---

### Module 6: Procedural 3D Rendering (Blender)

**Responsibility**: Parse JSON and render 3D scenes with dynamic shading and animation.

**Inputs**:
- Fused JSON file from Module 5
- 3D asset library (FBX/Blend files):
  - Generic vehicles (sedan, SUV, truck, etc.)
  - Pedestrian mesh with armature
  - Traffic lights with emissive bulbs
  - Signs (stop, speed limit, directional)
  - Cones, poles, dustbins
  - Lane meshes (procedurally generated)

**Blender Implementation Plan**:

#### 6.1 Scene Setup
1. Create world coordinate system aligned with camera extrinsics
2. Place virtual camera at origin with intrinsics matching $\mathbf{K}$
3. Create collection for dynamic actors, infrastructure, lanes

#### 6.2 Object Instantiation & Positioning
**Per-frame loop**:
```python
for frame in json["frames"]:
    for actor in frame["detections"]["dynamic_actors"]:
        obj = get_or_create_asset(actor["class_name"], actor["track_id"])
        pos_world = camera_to_world(actor["position_3d_cam_m"])
        rot_world = camera_to_world_rotation(actor["orientation_6d"])
        obj.location = pos_world
        obj.rotation_quaternion = rot_world
        obj.keyframe_insert(data_path="location", frame=frame["frame_index"])
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame["frame_index"])
```

**Camera Extrinsics**:
- Inverse transformation: $\mathbf{T}_{c2w} = \mathbf{T}_{w2c}^{-1}$
- Applied per-frame to lift camera-space 3D points to world space

#### 6.3 Dynamic Shading
**Brake Lights**:
- Bind traffic light emission strength to `intent.brake_lights_on` boolean
- Material nodes: Image texture + Emission shader (Strength = 0 or 5 when off/on)

**Collision Risk Highlighting** (Extra Credit):
- If `collision.is_imminent == True`, change vehicle material emission or wireframe to red
- Gradual fade based on TTC: $\text{emission\_strength} = \max(0, 1 - \text{TTC} / 5)$

#### 6.4 Procedural Lane Mesh Generation (Geometry Nodes)
**Approach**:
1. Create a base "lane strip" mesh as a quad
2. Parameterize by spline control points from JSON
3. Use Geometry Nodes to:
   - **Curve from points**: Build cubic Bezier from control points
   - **Curve to mesh**: Extrude curve into lane surface (0.3 m width)
   - **Instance on points**: Duplicate for each lane
4. Apply material with semantic color:
   - Dashed white: stippled white texture
   - Solid white: solid white
   - Yellow: double-yellow stripe
   - Grass: green material

**Pseudo-code**:
```python
for lane in frame["detections"]["lanes"]["lane_curves"]:
    control_points = lane["curve_3d_control_points"]
    create_lane_mesh_from_spline(control_points, lane["semantic_type"])
```

#### 6.5 Animation & Playback
- Set frame range (0 to total_frames)
- Enable auto-keyframing if needed
- Render viewport preview or eevee/cycles output sequence (EXR 16-bit)
- Export as MP4 using FFmpeg integration

---

## 3. Data Tensor Flow Summary

```
┌─────────────────────────────┐
│ Raw Video Frames (H×W×3)    │
└──────────────┬──────────────┘
               │
        ╔──────▼──────╗
        │ Preprocess  │  → K, dist_coeffs (3×3, 4), undistorted frames (H×W×3)
        ╚──────┬──────╝
               │
        ╔──────▼──────╗
        │ Perception  │  → Detections: bbox_2d (N×4), class_id (N), conf (N)
        │             │  → Keypoints: pose_2d (P×17×2)
        ╚──────┬──────╝
               │
        ╔──────▼──────╗
        │ Localization│  → Depth (H×W), bbox_3d (N×6), lanes_3d (Splines)
        ╚──────┬──────╝
               │
        ╔──────▼──────╗
        │ Temporal    │  → Tracking IDs, velocity_3d (N×3), motion_state (N)
        │ Motion      │  → Optical flow (H×W×2), TTC (N)
        ╚──────┬──────╝
               │
        ╔──────▼──────╗
        │ Fusion      │  → JSON (frame-wise aggregation)
        ╚──────┬──────╝
               │
        ╔──────▼──────╗
        │ Rendering   │  → 3D video (MP4 or EXR sequence)
        ╚─────────────╘
```

---

## 4. Suggested Open-Source Models

| **Module** | **Component** | **Recommended Model** | **Notes** |
|---|---|---|---|
| **Preprocessing** | Calibration | OpenCV `calibrateCamera()` | Classical, robust checkerboard method |
| **Perception** | Vehicle/Pedestrian Detection | RF-DETR / YOLOv12 / Mask R-CNN | Choose RF-DETR for SOTA accuracy, YOLOv12 for balance, Mask R-CNN for pixel masks |
| **Perception** | Pedestrian Pose | RTMPose or MediaPipe | 17-point skeleton, good metric accuracy |
| **Perception** | Traffic Light Detection | YOLOv12 / RF-DETR + Mask R-CNN | Higher recall on tiny signals; Mask R-CNN adds arrow masks |
| **Perception** | Sign/OCR | EasyOCR or PaddleOCR | Robust for speed sign digit extraction |
| **Perception** | Lane Segmentation | PolyLaneNet or SCNN | Outputs parametric curves directly |
| **Localization** | Monocular Depth | Depth Anything V2 (small) | Recent SOTA, metric scale, fast |
| **Localization** | Direct Mono 3D Detection | MonoCoP / MonoDiff | Skip depth map; direct 6D pose at high accuracy (slower) |
| **Localization** | 6D Pose | OpenCV `solvePnP()` + custom refinement | Classical but effective; refine with Gauss-Newton |
| **Motion** | Optical Flow | RAFT (Recurrent All-Pairs Field) | Dense, sub-pixel accuracy; ~3GB VRAM |
| **Motion** | Multi-Object Tracking | DeepSORT or ByteTrack | Mature, well-integrated with YOLO |
| **Motion** | Epipolar Geometry | OpenCV `findFundamentalMat()` | RANSAC-based F-matrix estimation |
| **Rendering** | Blender Procedural | Geometry Nodes + bpy API | Native Blender parameterization |

---

## 5. Project Phases & Deliverables

### **Phase 1: Foundation (Baseline Perception)**
- ✅ Module 1: Camera calibration + undistortion
- ✅ Module 2: Vehicle + pedestrian detection (YOLOv11)
- ✅ Module 3: Monocular depth (Depth Anything V2)
- ✅ Module 5: JSON serialization
- ✅ Module 6: Static object placement in Blender

**Deliverable**: 13 videos with static detected objects positioned in 3D space, no animation.

---

### **Phase 2: Temporal & Motion (Motion-Aware Rendering)**
- ✅ Module 2: Infrastructure detection (traffic lights, signs)
- ✅ Module 3: 6D pose refinement + lane detection
- ✅ Module 4: Multi-object tracking + optical flow
- ✅ Module 4: Motion segmentation (parked vs. moving)
- ✅ Module 6: Keyframed animation of detected objects

**Deliverable**: 13 videos with animated moving vehicles, pedestrians; static infrastructure; lane meshes visualized.

---

### **Phase 3: Intent & Prediction (Advanced Semantics)**
- ✅ Module 4: Brake light / turn signal detection
- ✅ Module 4: Time-to-collision forecasting
- ✅ Module 2: Sign OCR (speed limit extraction)
- ✅ Module 6: Dynamic shading (brake light emission)
- ✅ Module 6: Collision risk highlighting

**Deliverable**: 13 videos with intent indicators, collision alerts, sign information overlays.

---

### **Extra Credit: Enhanced Perception & Rendering**
1. **Pedestrian Pose Visualization**: Render skeletal overlays or pose-matched humanoid meshes
2. **Lane Topology**: Extract lane connectivity graph (which lanes are adjacent, merging, diverging)
3. **Prediction Module** (Physics-based):
   - Predict future object trajectories using constant velocity or motion history
   - Visualize predicted paths as semi-transparent meshes
4. **Custom Asset Library**: Procedurally generated traffic cones, signs, lights with varied colors
5. **Real-time Inference Optimization**: ONNX export of models + GPU batching for faster perception
6. **Sensor Fusion (Optional)**: If IMU/GPS data available, fuse with vision for ego-motion refinement

---

## 6. Dependencies & Environment Setup

```bash
# Core perception
pip install opencv-python==4.8.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics==8.2.0  # YOLOv12 attention-centric branch
pip install git+https://github.com/IDEA-Research/RF-DETR  # Transformer detector
pip install timm==0.9.8  # Vision transformers for depth/pose

# Depth & Pose
pip install git+https://github.com/DepthAnything/Depth-Anything-V2
pip install rtmpose  # RTMPose for skeleton extraction
pip install mediapipe  # Alternative to RTMPose

# Direct monocular 3D detectors (optional)
pip install git+https://github.com/OpenDriveLab/MonoCoP
pip install git+https://github.com/OpenDriveLab/MonoDiff

# Optical Flow
pip install raft-core  # RAFT optical flow
pip install git+https://github.com/princeton-vl/RAFT

# Tracking
pip install deep-sort-pytorch  # DeepSORT
pip install git+https://github.com/ifzhang/ByteTrack

# Lane Detection
pip install polylanenet  # PolyLaneNet

# Utilities
pip install numpy scipy scikit-image
pip install easyocr  # For sign OCR
pip install paddleocr  # Alternative OCR

# Blender (separate environment or manual installation)
# Use Blender 4.0+ Python API (bundled with Blender)
```

---

## 7. Summary & Workflow

**End-to-End Pipeline**:
1. User provides 13 monocular videos + calibration video
2. **Preprocessing**: Extract intrinsics, undistort frames
3. **Perception**: Run detectors (RF-DETR, YOLOv12, Mask R-CNN, RTMPose, etc.) per frame
4. **Localization**: Depth estimation + 6D pose + lanes
5. **Temporal**: Tracking, optical flow, motion segmentation, TTC
6. **Fusion**: Aggregate to JSON per video
7. **Rendering**: Load JSON into Blender, animate and shade
8. **Export**: Render 13 output videos (MP4 or EXR sequences)

**Quality Gates**:
- Reprojection error on calibration: < 0.5 px
- Detection mAP (evaluated on validation): > 0.7
- Depth MAE (vs. ground-truth, if available): < 10%
- Tracking ID consistency (MOTA): > 0.6
- Render visual inspection: Objects align with image, lanes follow road

---

## 8. File Structure (Updated)

```text
EinsteinVision/
├── README.md
├── ARCHITECTURE.md              # This file
├── src/
│   └── einsteinvision/
│       ├── __init__.py
│       ├── preprocessing.py     # CameraCalibrator, FrameUndistorter
│       ├── perception.py        # PerceptionPipeline, 2D detectors
│       ├── localization.py      # Depth, 6D pose, lane detection
│       ├── temporal.py          # Tracking, optical flow, motion seg
│       ├── fusion.py            # SceneFusion, JSON export
│       └── blender_render.py    # BlenderSceneRenderer
├── configs/
│   ├── model_config.yaml        # Model paths, thresholds
│   ├── perception_config.yaml   # Detection confidence thresholds
│   └── blender_config.yaml      # Asset paths, render settings
├── assets/
│   ├── vehicles/
│   │   ├── sedan.blend
│   │   ├── suv.blend
│   │   └── ...
│   ├── pedestrians/
│   │   └── humanoid.blend
│   ├── traffic/
│   │   ├── traffic_light.blend
│   │   ├── stop_sign.blend
│   │   └── ...
│   └── road_objects/
│       ├── cone.blend
│       └── ...
├── data/
│   ├── raw/
│   │   ├── video_001.mp4
│   │   ├── video_002.mp4
│   │   └── ...
│   ├── calibration/
│   │   └── calibration_video.mp4
│   └── output/
│       ├── video_001_fused.json
│       ├── video_001_rendered.mp4
│       └── ...
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_perception.py
│   ├── run_localization.py
│   ├── run_temporal.py
│   ├── run_fusion.py
│   └── run_blender_render.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_perception.py
│   └── ...
└── requirements.txt
```

---

**Document Version**: 1.0  
**Last Updated**: March 2026  
**Status**: Architecture Complete, Ready for Implementation
