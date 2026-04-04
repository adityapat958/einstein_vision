"""
Road sign detection: ground arrows + speed limit sign OCR.

GroundArrowDetector
-------------------
Detects directional arrows painted on the road surface (left / straight /
right).  Uses a two-step approach:
  1. Restrict the search to the road ROI (lower ~55 % of the frame, same
     trapezoid as lane_export.py).
  2. Find bright white/yellow blobs whose aspect ratio and solidity suggest
     an arrow shape.
  3. Classify direction by the centroid position and the tip of the convex
     hull relative to the bounding box.

SpeedLimitOCR
-------------
Given a bounding box for a "speed limit sign", crops the region and runs
EasyOCR (or a regex fallback on the raw image) to extract the numeric value.
Falls back gracefully if EasyOCR is not installed.

Usage
-----
    from road_sign_detector import GroundArrowDetector, SpeedLimitOCR

    arrow_det = GroundArrowDetector()
    arrows = arrow_det.detect(frame_bgr)
    # returns list of dicts: {"direction": "left"|"straight"|"right",
    #                          "bbox": [x1,y1,x2,y2], "score": float}

    ocr = SpeedLimitOCR()
    value = ocr.extract_number(frame_bgr, bbox=[x1, y1, x2, y2])
    # returns int (e.g. 30) or None
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import cv2
import numpy as np


# ── Road ROI (same trapezoid as lane_export.py) ──────────────────────────────

def _make_road_roi_mask(h: int, w: int) -> np.ndarray:
    """Trapezoidal road region covering lower ~44 % of the frame."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([
        [int(w * 0.03), h],
        [int(w * 0.38), int(h * 0.56)],
        [int(w * 0.62), int(h * 0.56)],
        [int(w * 0.98), h],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ── Arrow direction heuristic ─────────────────────────────────────────────────

def _classify_arrow_direction(contour: np.ndarray, bbox: tuple) -> str:
    """
    Classify arrow direction from a contour.

    Uses the convex hull tip relative to the bounding box centroid.
    Returns "left", "straight", or "right".
    """
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    hull = cv2.convexHull(contour)
    if hull is None or len(hull) < 3:
        return "straight"

    # Find the point of the convex hull furthest from the centroid
    pts = hull.reshape(-1, 2).astype(np.float32)
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    tip = pts[np.argmax(dists)]

    dx = tip[0] - cx
    dy = tip[1] - cy   # positive = downward (bottom of image)

    if abs(dx) < 0.15 * w:
        return "straight"
    if dx < 0:
        return "left"
    return "right"


class GroundArrowDetector:
    """
    Detect road-surface directional arrows using colour + shape analysis.

    Args:
        min_area:   Minimum contour area in pixels to consider.
        min_solidity: Minimum solidity (area / convex hull area) threshold.
    """

    def __init__(self, min_area: int = 800, min_solidity: float = 0.50) -> None:
        self._min_area = min_area
        self._min_solidity = min_solidity

    def detect(
        self, frame_bgr: np.ndarray
    ) -> List[Dict]:
        """
        Run arrow detection on a frame.

        Returns:
            List of dicts: {"direction": str, "bbox": [x1,y1,x2,y2], "score": float}
        """
        h, w = frame_bgr.shape[:2]
        roi_mask = _make_road_roi_mask(h, w)

        # Threshold for white/yellow road markings
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # White: high V, low S
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 180], dtype=np.uint8),
            np.array([180, 60, 255], dtype=np.uint8),
        )
        # Yellow: typical road marking hue
        yellow_mask = cv2.inRange(
            hsv,
            np.array([15, 80, 150], dtype=np.uint8),
            np.array([35, 255, 255], dtype=np.uint8),
        )
        marking_mask = cv2.bitwise_or(white_mask, yellow_mask)
        marking_mask = cv2.bitwise_and(marking_mask, roi_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        marking_mask = cv2.morphologyEx(marking_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        marking_mask = cv2.morphologyEx(marking_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(
            marking_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1:
                continue
            solidity = area / hull_area
            if solidity < self._min_solidity:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / (bh + 1e-6)

            # Arrows are roughly 0.5–3× wide relative to height
            if not (0.4 < aspect < 3.5):
                continue

            direction = _classify_arrow_direction(cnt, (x, y, bw, bh))
            score = float(min(1.0, solidity))

            results.append({
                "direction": direction,
                "bbox": [x, y, x + bw, y + bh],
                "score": score,
            })

        return results


# ── Speed limit OCR ───────────────────────────────────────────────────────────

class SpeedLimitOCR:
    """
    Extract the numeric speed limit from a speed-limit sign crop.

    Tries EasyOCR first (if installed); falls back to contour-based digit
    detection as a last resort.
    """

    def __init__(self) -> None:
        self._reader = None
        self._reader_tried = False

    def _get_reader(self):
        if self._reader_tried:
            return self._reader
        self._reader_tried = True
        try:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("[SpeedLimitOCR] EasyOCR loaded.")
        except ImportError:
            print("[SpeedLimitOCR] EasyOCR not installed. "
                  "Install with: pip install easyocr")
        except Exception as e:
            print(f"[SpeedLimitOCR] Could not load EasyOCR: {e}")
        return self._reader

    def extract_number(
        self,
        frame_bgr: np.ndarray,
        bbox: List[int],
    ) -> Optional[int]:
        """
        Extract the speed limit number from a sign bounding box.

        Args:
            frame_bgr: Full BGR frame.
            bbox: [x1, y1, x2, y2] of the speed limit sign.

        Returns:
            Integer speed limit (e.g. 30, 45, 65) or None if not found.
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Upscale small crops for better OCR accuracy
        scale = max(1, int(150 / max(crop.shape[:2])))
        if scale > 1:
            crop = cv2.resize(
                crop,
                (crop.shape[1] * scale, crop.shape[0] * scale),
                interpolation=cv2.INTER_CUBIC,
            )

        reader = self._get_reader()
        if reader is not None:
            try:
                results = reader.readtext(crop, detail=0, paragraph=False)
                for text in results:
                    nums = re.findall(r"\b(\d{1,3})\b", text)
                    for n in nums:
                        val = int(n)
                        if 5 <= val <= 130:   # plausible speed limit range
                            return val
            except Exception as e:
                print(f"[SpeedLimitOCR] EasyOCR inference error: {e}")

        # Fallback: threshold + find large digit-shaped contours
        return self._contour_fallback(crop)

    def _contour_fallback(self, crop_bgr: np.ndarray) -> Optional[int]:
        """Very rough fallback: look for dark digits on white background."""
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert if background is dark
        if np.mean(thresh) < 128:
            thresh = cv2.bitwise_not(thresh)
        contours, _ = cv2.findContours(
            cv2.bitwise_not(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Count large contours as potential digits; very rough proxy
        digit_cnt = sum(
            1 for c in contours
            if 50 < cv2.contourArea(c) < crop_bgr.shape[0] * crop_bgr.shape[1] * 0.3
        )
        # Cannot extract actual number without OCR — return None
        return None
