"""
Diagnostic script: visualize YCbCr Cr-channel light detection.

Usage:
    python src/Code/debug_lights.py \
        --video P3Data/Sequences/scene3/Undist/2023-02-14_11-49-54-front_undistort.mp4 \
        --start-frame 1616 --max-frames 541 \
        --output src/pipeline_out/debug_lights.mp4
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

_BRAKE_THRESH = 0.04
_MARGIN_RATIO = 0.25
_VAR_THRESH   = 0.0002


def analyze_and_visualize(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return None, {}

    roi_top = y1 + int(0.50 * bh)
    roi = frame[roi_top:y2, x1:x2]
    if roi.size == 0:
        return None, {}

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    ycrcb = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2YCrCb)
    Cr = ycrcb[:, :, 1].astype(np.float32)

    cr_mean = Cr.mean()
    cr_std  = Cr.std()
    threshold = cr_mean + 1.5 * cr_std
    red_mask = (Cr > threshold).astype(np.uint8)

    mid_x = max(1, red_mask.shape[1] // 2)
    left_area  = int(red_mask[:, :mid_x].sum())
    right_area = int(red_mask[:, mid_x:].sum())
    half_pixels = roi.shape[0] * mid_x

    scores = {
        "red_l": left_area  / half_pixels if half_pixels else 0.0,
        "red_r": right_area / half_pixels if half_pixels else 0.0,
    }

    # Visualize: Cr heatmap + threshold mask overlay
    cr_norm = cv2.normalize(Cr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cr_color = cv2.applyColorMap(cr_norm, cv2.COLORMAP_HOT)
    cr_color = cv2.resize(cr_color, (roi.shape[1], roi.shape[0]))

    # Highlight thresholded pixels in cyan
    highlight = roi.copy()
    highlight[red_mask > 0] = (255, 255, 0)
    cv2.line(highlight, (mid_x, 0), (mid_x, highlight.shape[0]), (0, 255, 0), 1)

    overlay = cv2.addWeighted(roi, 0.4, highlight, 0.6, 0)
    cv2.line(overlay, (mid_x, 0), (mid_x, overlay.shape[0]), (0, 255, 0), 1)

    return overlay, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-frame", type=int, default=1616)
    parser.add_argument("--max-frames", type=int, default=541)
    parser.add_argument("--output", default="src/pipeline_out/debug_lights.mp4")
    parser.add_argument("--bbox", type=float, nargs=4, default=None,
                        help="Fixed bbox: x1 y1 x2 y2")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    default_bbox = args.bbox if args.bbox else [W*0.2, H*0.0, W*0.8, H*0.65]

    frame_idx = args.start_frame
    processed = 0
    history = []

    while processed < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()
        bbox = default_bbox
        x1, y1, x2, y2 = [int(v) for v in bbox]

        overlay, scores = analyze_and_visualize(frame, bbox)

        if overlay is not None:
            roi_top = y1 + int(0.50 * (y2 - y1))
            target_h = y2 - roi_top
            target_w = x2 - x1
            if overlay.shape[0] > 0 and overlay.shape[1] > 0:
                scaled = cv2.resize(overlay, (target_w, target_h))
                vis[roi_top:y2, x1:x2] = cv2.addWeighted(
                    vis[roi_top:y2, x1:x2], 0.3, scaled, 0.7, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.line(vis, (x1, roi_top), (x2, roi_top), (255, 255, 0), 1)

            history.append(scores)
            if len(history) > 15:
                history.pop(0)

            braking = False
            indicator = "none"
            if len(history) >= 3:
                bl = [h["red_l"] for h in history]
                br = [h["red_r"] for h in history]
                mean_l, mean_r = np.mean(bl), np.mean(br)
                total = mean_l + mean_r
                asymmetry = abs(mean_l - mean_r) / total if total > 0.01 else 0.0
                var_l = np.var(bl) if len(bl) > 1 else 0.0
                var_r = np.var(br) if len(br) > 1 else 0.0
                flash_l, flash_r = var_l > _VAR_THRESH, var_r > _VAR_THRESH

                braking = (mean_l > _BRAKE_THRESH and mean_r > _BRAKE_THRESH
                           and asymmetry < _MARGIN_RATIO and not (flash_l or flash_r))

                if flash_l and flash_r:
                    indicator = "HAZARD"
                elif flash_l:
                    indicator = "<- LEFT"
                elif flash_r:
                    indicator = "RIGHT ->"
                elif asymmetry > _MARGIN_RATIO and total > 0.01:
                    indicator = "<- LEFT" if mean_l > mean_r else "RIGHT ->"

            if processed % 10 == 0:
                print(f"frame={frame_idx}  "
                      f"red_l={scores['red_l']:.4f}  red_r={scores['red_r']:.4f}  "
                      f"| {indicator if indicator != 'none' else ('BRAKE' if braking else '-')}")

            lines = [
                f"f={frame_idx}",
                f"Cr_L:{scores['red_l']:.4f}  Cr_R:{scores['red_r']:.4f}",
                "BRAKING" if braking else "",
                indicator if indicator != "none" else "",
            ]
            for i, line in enumerate(lines):
                if not line:
                    continue
                color = (0, 0, 255) if "BRAKE" in line else \
                        (0, 165, 255) if any(x in line for x in ("LEFT", "RIGHT", "HAZARD")) else \
                        (255, 255, 255)
                cv2.putText(vis, line, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        writer.write(vis)
        frame_idx += 1
        processed += 1

    cap.release()
    writer.release()
    print(f"\n[DONE] {args.output}")


if __name__ == "__main__":
    main()
