"""
Traffic-light state and arrow-direction classifier.

Given a full BGR frame and a bounding box around a detected traffic light,
returns:
  - state:  "red" | "green" | "yellow" | "unknown"
  - arrow:  "left" | "straight" | "right" | "off" | None

Both are determined via HSV colour analysis (fast, no extra model needed).
Arrow detection is template-matching based; a small CNN head can be swapped
in later without changing the interface.

Usage
-----
    from traffic_light_classifier import TrafficLightClassifier

    clf = TrafficLightClassifier()
    result = clf.predict(frame_bgr, bbox=[x1, y1, x2, y2])
    print(result)  # {"state": "red", "arrow": "off"}
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


# ── HSV thresholds ────────────────────────────────────────────────────────────
# Each entry: (H_lo, H_hi, S_lo, V_lo)  — all in OpenCV HSV range (H: 0-180)
_RED_RANGES = [
    (0, 10, 120, 120),
    (170, 180, 120, 120),
]
_YELLOW_RANGES = [(15, 35, 120, 120)]
_GREEN_RANGES  = [(40, 90, 80, 80)]

_MIN_COLOUR_RATIO = 0.08   # at least 8 % of the roi pixels must match


def _count_pixels(hsv: np.ndarray, ranges: List[Tuple]) -> int:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (h_lo, h_hi, s_lo, v_lo) in ranges:
        lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        hi = np.array([h_hi, 255, 255], dtype=np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)
    return int(np.count_nonzero(mask))


def _detect_state(crop_bgr: np.ndarray) -> str:
    """Return the dominant lit colour of the traffic light crop."""
    if crop_bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return "unknown"

    n_red    = _count_pixels(hsv, _RED_RANGES)
    n_yellow = _count_pixels(hsv, _YELLOW_RANGES)
    n_green  = _count_pixels(hsv, _GREEN_RANGES)

    counts = {"red": n_red, "yellow": n_yellow, "green": n_green}
    best, best_n = max(counts.items(), key=lambda kv: kv[1])
    if best_n / total >= _MIN_COLOUR_RATIO:
        return best
    return "unknown"


# ── Arrow detection via contour analysis ─────────────────────────────────────

def _detect_arrow(crop_bgr: np.ndarray, state: str) -> str:
    """
    Detect arrow direction in a traffic-light crop.

    Returns "left", "straight", "right", or "off".

    Strategy:
    1. Threshold the lit region (colour matching the state).
    2. Find contours.
    3. For the largest contour, analyse the horizontal centroid offset
       relative to the crop centre and the bounding-box aspect ratio to
       infer direction.  This is a heuristic; a CNN head can be added later.
    """
    if crop_bgr.size == 0 or state == "unknown":
        return "off"

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    if state == "red":
        ranges = _RED_RANGES
    elif state == "yellow":
        ranges = _YELLOW_RANGES
    else:  # green
        ranges = _GREEN_RANGES

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (h_lo, h_hi, s_lo, v_lo) in ranges:
        lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        hi = np.array([h_hi, 255, 255], dtype=np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "off"

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 20:
        return "off"

    x, y, w, h = cv2.boundingRect(largest)
    crop_w = crop_bgr.shape[1]
    crop_h = crop_bgr.shape[0]
    cx = x + w / 2.0
    cy = y + h / 2.0

    # Normalised centroid offset from centre [-1, 1]
    norm_x = (cx - crop_w / 2.0) / (crop_w / 2.0 + 1e-6)
    norm_y = (cy - crop_h / 2.0) / (crop_h / 2.0 + 1e-6)
    aspect = w / (h + 1e-6)

    # Circular light: roughly symmetric, no strong arrow
    if abs(norm_x) < 0.2 and aspect < 1.5:
        return "off"

    # Elongated horizontally → straight or turn arrow
    if aspect > 1.8:
        # Arrow points left or right based on which side the centroid is on
        if norm_x < -0.15:
            return "left"
        if norm_x > 0.15:
            return "right"
        return "straight"

    # Vertically elongated pointing upward → straight
    if aspect < 0.6 and norm_y < -0.1:
        return "straight"

    return "off"


class TrafficLightClassifier:
    """
    Classifies traffic-light crops into state + arrow direction.

    Args:
        use_top_half_for_arrow: If True, only the top 60 % of the crop is
            used for arrow detection (to focus on the actual light housing
            and avoid sign reflections below).
    """

    def __init__(self, use_top_half_for_arrow: bool = True) -> None:
        self._top_half = use_top_half_for_arrow

    def predict(
        self,
        frame_bgr: np.ndarray,
        bbox: List[int],
    ) -> Dict[str, str | None]:
        """
        Classify the traffic light.

        Args:
            frame_bgr: Full BGR frame.
            bbox: [x1, y1, x2, y2] pixel bounding box of the traffic light.

        Returns:
            dict with keys:
                "state":  "red" | "yellow" | "green" | "unknown"
                "arrow":  "left" | "straight" | "right" | "off" | None
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return {"state": "unknown", "arrow": None}

        state = _detect_state(crop)

        if state == "unknown":
            return {"state": "unknown", "arrow": None}

        # Use top portion of the crop for arrow analysis
        arrow_crop = crop
        if self._top_half:
            split = max(1, int(crop.shape[0] * 0.60))
            arrow_crop = crop[:split, :]

        arrow = _detect_arrow(arrow_crop, state)

        return {"state": state, "arrow": arrow}
