"""
Vehicle sub-classification module.

Takes a full frame + a 2D bounding box around a detected vehicle and returns
one of: sedan, hatchback, suv, pickup, truck, bicycle, motorcycle.

Two modes
---------
1. **Model-based** (default when weights are provided):
   A fine-tuned EfficientNet-B0 with a 7-class head is loaded.  The crop is
   pre-processed with ImageNet normalisation and the argmax class is returned.

2. **Heuristic fallback** (when no weights file is given or the model fails):
   Uses bounding-box aspect ratio and normalised height as a rough proxy for
   vehicle type.  Not accurate, but ensures the pipeline always produces a
   sub_class rather than None.

Usage in phase2_pipeline.py
----------------------------
    from vehicle_subclassifier import VehicleSubClassifier

    classifier = VehicleSubClassifier(weights_path="path/to/weights.pth")
    sub_class = classifier.predict(frame_bgr, bbox=[x1, y1, x2, y2],
                                   class_name="car")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import cv2
import numpy as np

CLASSES: List[str] = ["sedan", "hatchback", "suv", "pickup", "truck", "bicycle", "motorcycle"]

# Mapping from YOLO COCO class name → which sub-classes are possible
_COCO_TO_CANDIDATES = {
    "car":        ["sedan", "hatchback", "suv", "pickup"],
    "truck":      ["truck", "pickup"],
    "bus":        ["truck"],
    "bicycle":    ["bicycle"],
    "motorcycle": ["motorcycle"],
}

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INPUT_SIZE = (224, 224)


def _preprocess(crop_bgr: np.ndarray) -> "np.ndarray":
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = cv2.resize(rgb, _INPUT_SIZE)
    normalised = (resized - _MEAN) / _STD
    # HWC → CHW → 1CHW
    return normalised.transpose(2, 0, 1)[None].astype(np.float32)


def _heuristic(bbox: List[int], class_name: str, frame_h: int) -> str:
    """Rule-based fallback using bbox geometry."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return "sedan"

    aspect = w / h                        # width / height
    rel_h  = h / frame_h if frame_h > 0 else 0.1

    cname = class_name.lower()
    if "bicycle" in cname:
        return "bicycle"
    if "motorcycle" in cname:
        return "motorcycle"
    if "truck" in cname or "bus" in cname:
        return "truck"

    # Car heuristics
    if aspect > 2.2:          # very wide → likely pickup truck bed or SUV
        return "pickup" if aspect > 2.8 else "suv"
    if rel_h > 0.22:          # tall in frame → SUV / van
        return "suv"
    if aspect < 1.5:          # squarish → hatchback
        return "hatchback"
    return "sedan"             # default


class VehicleSubClassifier:
    """
    Two-stage vehicle sub-classifier.

    Args:
        weights_path: Path to a PyTorch checkpoint (.pth) with an
            EfficientNet-B0 backbone + 7-class head.  Pass ``None`` or a
            non-existent path to use the heuristic fallback only.
        device: "cuda", "cpu", or "mps".
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self._device = device
        self._model = None

        if weights_path and Path(weights_path).exists():
            self._load_model(Path(weights_path))

    def _load_model(self, weights_path: Path) -> None:
        try:
            import torch
            import torchvision.models as models

            model = models.efficientnet_b0(weights=None)
            # Replace the classifier head for 7 classes
            import torch.nn as nn
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, len(CLASSES))
            ckpt = torch.load(str(weights_path), map_location=self._device)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state, strict=False)
            model.to(self._device)
            model.eval()
            self._model = model
            print(f"[VehicleSubClassifier] Loaded model from {weights_path}")
        except Exception as e:
            print(f"[VehicleSubClassifier] Could not load model ({e}); using heuristics.")
            self._model = None

    def predict(
        self,
        frame_bgr: np.ndarray,
        bbox: List[int],
        class_name: str = "car",
    ) -> str:
        """
        Predict the vehicle sub-class.

        Args:
            frame_bgr: Full BGR frame.
            bbox: [x1, y1, x2, y2] pixel coordinates.
            class_name: COCO class name of the detection ("car", "truck", etc.).

        Returns:
            Sub-class string, e.g. "sedan", "suv".
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h_frame = frame_bgr.shape[0]

        # Use heuristics directly for non-car classes
        cname = class_name.lower()
        if cname in ("bicycle", "motorcycle"):
            return cname
        if cname in ("bus",):
            return "truck"

        if self._model is None:
            return _heuristic([x1, y1, x2, y2], class_name, h_frame)

        # Crop with a small margin
        margin = 4
        x1c = max(0, x1 - margin)
        y1c = max(0, y1 - margin)
        x2c = min(frame_bgr.shape[1], x2 + margin)
        y2c = min(h_frame, y2 + margin)
        crop = frame_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return _heuristic([x1, y1, x2, y2], class_name, h_frame)

        try:
            import torch
            tensor = torch.from_numpy(_preprocess(crop)).to(self._device)
            with torch.no_grad():
                logits = self._model(tensor)
                idx = int(logits.argmax(dim=1).item())

            candidate = CLASSES[idx]

            # Constrain to plausible candidates for the detected class
            candidates = _COCO_TO_CANDIDATES.get(cname, CLASSES)
            if candidate not in candidates:
                # Argmax among valid candidates only
                import torch.nn.functional as F
                probs = F.softmax(logits, dim=1)[0]
                cand_idx = [CLASSES.index(c) for c in candidates if c in CLASSES]
                if cand_idx:
                    best = cand_idx[int(probs[cand_idx].argmax().item())]
                    candidate = CLASSES[best]
                else:
                    candidate = candidates[0]

            return candidate

        except Exception as e:
            print(f"[VehicleSubClassifier] Inference error ({e}); falling back to heuristic.")
            return _heuristic([x1, y1, x2, y2], class_name, h_frame)
