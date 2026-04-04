#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# =========================================================
# USER SETTINGS
# =========================================================
video_path = "/home/alien/cv_p3/scene11/Undist/2023-03-11_17-19-53-front_undistort.mp4"
checkpoint_path = "/home/alien/cv_p3/ext_models/69.pth"

# IMPORTANT: this must match the checkpoint architecture/training config
config_path = "/home/alien/cv_p3/CLRNet/configs/clrnet/clr_resnet18_tusimple.py"

output_path = "/home/alien/cv_p3/scene11/Undist/2023-03-11_17-19-53-front_clrnet_overlay.mp4"

# visual settings
lane_color = (0, 255, 0)
lane_thickness = 3
show_window = False
use_gpu = True
conf_text = True
# =========================================================


def add_repo_to_path(cfg_path: str):
    repo_root = Path(cfg_path).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def build_model_and_cfg(config_path, checkpoint_path, device):
    from clrnet.utils.config import Config
    from clrnet.models.registry import build_net
    from clrnet.utils.net_utils import load_network

    cfg = Config.fromfile(config_path)
    cfg.load_from = checkpoint_path
    cfg.view = False

    model = build_net(cfg)
    load_network(model, checkpoint_path)

    model = model.to(device)
    model.eval()

    return model, cfg


def get_input_size_from_cfg(cfg):
    img_w = None
    img_h = None

    for key in ["img_w", "ori_img_w", "cut_width"]:
        if hasattr(cfg, key):
            pass

    if hasattr(cfg, "img_w"):
        img_w = int(cfg.img_w)
    elif hasattr(cfg, "img_width"):
        img_w = int(cfg.img_width)

    if hasattr(cfg, "img_h"):
        img_h = int(cfg.img_h)
    elif hasattr(cfg, "img_height"):
        img_h = int(cfg.img_height)

    if img_w is None:
        img_w = 800
    if img_h is None:
        img_h = 320

    return img_w, img_h


def normalize_image(img_rgb):
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return img


def preprocess_frame(frame_bgr, cfg, device):
    orig_h, orig_w = frame_bgr.shape[:2]
    net_w, net_h = get_input_size_from_cfg(cfg)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(frame_rgb, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
    tensor = normalize_image(resized_rgb)
    tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)

    img_metas = [{
        "img_name": "video_frame",
        "ori_shape": (orig_h, orig_w, 3),
        "img_shape": (net_h, net_w, 3),
        "scale_factor": np.array([orig_w / net_w, orig_h / net_h], dtype=np.float32),
        "net_input_size": (net_w, net_h),
    }]

    return tensor, img_metas, (orig_w, orig_h), (net_w, net_h)


def try_model_forward(model, img_tensor, img_metas):
    """
    CLRNet forks vary a bit, so try a few common inference call styles.
    """
    with torch.no_grad():
        errors = []

        candidates = [
            lambda: model(img=img_tensor, img_metas=img_metas, return_loss=False),
            lambda: model(img_tensor, img_metas=img_metas, return_loss=False),
            lambda: model({"img": img_tensor, "img_metas": img_metas}),
            lambda: model(img_tensor),
        ]

        for fn in candidates:
            try:
                out = fn()
                return out
            except Exception as e:
                errors.append(str(e))

        raise RuntimeError(
            "All inference call patterns failed.\n"
            + "\n---\n".join(errors)
        )


def to_lane_arrays(predictions, cfg):
    """
    Normalize possible CLRNet outputs to a list of Nx2 arrays.
    """
    lanes_out = []

    if predictions is None:
        return lanes_out

    # some repos return [lanes] for batch size 1
    if isinstance(predictions, (list, tuple)) and len(predictions) == 1:
        first = predictions[0]
    else:
        first = predictions

    # if first itself is the lane list
    lane_list = first if isinstance(first, (list, tuple)) else [first]

    for lane in lane_list:
        try:
            if hasattr(lane, "to_array"):
                arr = lane.to_array(cfg)
            elif isinstance(lane, np.ndarray):
                arr = lane
            elif torch.is_tensor(lane):
                arr = lane.detach().cpu().numpy()
            else:
                continue

            arr = np.asarray(arr)

            if arr.ndim == 2 and arr.shape[1] >= 2:
                lanes_out.append(arr[:, :2])

        except Exception:
            continue

    return lanes_out


def scale_lane_to_original(arr_xy, net_size, orig_size):
    net_w, net_h = net_size
    orig_w, orig_h = orig_size

    arr = arr_xy.copy().astype(np.float32)

    # Heuristic: if coordinates are normalized [0,1], scale directly.
    if np.nanmax(arr[:, 0]) <= 2.0 and np.nanmax(arr[:, 1]) <= 2.0:
        arr[:, 0] *= orig_w
        arr[:, 1] *= orig_h
        return arr

    # Otherwise assume coordinates are in network input space.
    sx = float(orig_w) / float(net_w)
    sy = float(orig_h) / float(net_h)
    arr[:, 0] *= sx
    arr[:, 1] *= sy
    return arr


def draw_lanes(frame, lane_arrays, net_size, orig_size):
    out = frame.copy()

    for lane in lane_arrays:
        pts = scale_lane_to_original(lane, net_size, orig_size)

        valid_pts = []
        for x, y in pts:
            if np.isnan(x) or np.isnan(y):
                continue
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < orig_size[0] and 0 <= yi < orig_size[1]:
                valid_pts.append((xi, yi))

        if len(valid_pts) >= 2:
            for i in range(1, len(valid_pts)):
                cv2.line(out, valid_pts[i - 1], valid_pts[i], lane_color, lane_thickness)

    return out


def main():
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    repo_root = add_repo_to_path(config_path)
    print(f"[INFO] CLRNet repo root: {repo_root}")

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"[INFO] Using device: {device}")

    model, cfg = build_model_and_cfg(config_path, checkpoint_path, device)
    print("[INFO] Model loaded successfully.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {output_path}")

    print(f"[INFO] Input video: {video_path}")
    print(f"[INFO] Output video: {output_path}")
    print(f"[INFO] Resolution: {width}x{height} @ {fps:.2f} FPS")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        try:
            img_tensor, img_metas, orig_size, net_size = preprocess_frame(frame, cfg, device)
            predictions = try_model_forward(model, img_tensor, img_metas)
            lane_arrays = to_lane_arrays(predictions, cfg)
            vis = draw_lanes(frame, lane_arrays, net_size, orig_size)

            if conf_text:
                cv2.putText(
                    vis,
                    f"CLRNet | Frame {frame_idx}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(vis)

            if show_window:
                cv2.imshow("CLRNet Video Inference", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            if frame_idx % 25 == 0:
                if total_frames > 0:
                    print(f"[INFO] Processed {frame_idx}/{total_frames} frames")
                else:
                    print(f"[INFO] Processed {frame_idx} frames")

        except Exception as e:
            print(f"[WARN] Frame {frame_idx}: inference failed: {e}")
            writer.write(frame)
            continue

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("[INFO] Done.")
    print(f"[INFO] Saved overlay video to: {output_path}")


if __name__ == "__main__":
    main()