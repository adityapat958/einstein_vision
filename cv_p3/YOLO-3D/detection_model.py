import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque


class ObjectDetector:
    """
    Object detection using YOLOv11 from Ultralytics
    """

    def __init__(
        self,
        model_size='small',
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        device=None,
        model_path=None
    ):
        """
        Initialize the object detector

        Args:
            model_size (str): Model size ('nano', 'small', 'medium', 'large', 'extra')
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            classes (list): List of classes to detect (None for all classes)
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
            model_path (str): Full local path to YOLO weights (.pt). If provided,
                              this is used directly and no auto-download is needed.
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device

        # Set MPS fallback for operations not supported on Apple Silicon
        if self.device == 'mps':
            print("Using MPS device with CPU fallback for unsupported operations")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        print(f"Using device: {self.device} for object detection")

        # Map model size to model filename
        model_map = {
            'nano': 'yolo11n.pt',
            'small': 'yolo11s.pt',
            'medium': 'yolo11m.pt',
            'large': 'yolo11l.pt',
            'extra': 'yolo11x.pt'
        }

        # Decide which model path/name to use
        if model_path is not None:
            model_to_load = model_path
            if not os.path.isfile(model_to_load):
                raise FileNotFoundError(f"YOLO weights file not found: {model_to_load}")
            print(f"Loading YOLO model from local weights: {model_to_load}")
        else:
            model_to_load = model_map.get(model_size.lower(), model_map['small'])
            print(f"Loading YOLO model by name: {model_to_load}")

        # Load model
        try:
            self.model = YOLO(model_to_load)
            print(f"Loaded YOLO model on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Move model to device if possible
        try:
            self.model.to(self.device)
        except Exception as e:
            print(f"[WARN] Could not move YOLO model to {self.device}: {e}")
            print("[INFO] Continuing with Ultralytics default device handling.")

        # Set model parameters
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000

        if classes is not None:
            self.model.overrides['classes'] = classes

        # Initialize tracking trajectories
        self.tracking_trajectories = {}

    def detect(self, image, track=True):
        """
        Detect objects in an image

        Args:
            image (numpy.ndarray): Input image (BGR format)
            track (bool): Whether to track objects across frames

        Returns:
            tuple: (annotated_image, detections)
                - annotated_image (numpy.ndarray): Image with detections drawn
                - detections (list): List of detections [bbox, score, class_id, object_id]
        """
        detections = []

        # Make a copy of the image for annotation
        annotated_image = image.copy()

        try:
            if track:
                # Run inference with tracking
                results = self.model.track(
                    image,
                    verbose=False,
                    device=self.device,
                    persist=True,
                    conf=self.model.overrides.get('conf', 0.25),
                    iou=self.model.overrides.get('iou', 0.45),
                    classes=self.model.overrides.get('classes', None)
                )
            else:
                # Run inference without tracking
                results = self.model.predict(
                    image,
                    verbose=False,
                    device=self.device,
                    conf=self.model.overrides.get('conf', 0.25),
                    iou=self.model.overrides.get('iou', 0.45),
                    classes=self.model.overrides.get('classes', None)
                )

        except RuntimeError as e:
            # Handle potential MPS errors
            if self.device == 'mps' and "not currently implemented for the MPS device" in str(e):
                print(f"MPS error during detection: {e}")
                print("Falling back to CPU for this frame")
                if track:
                    results = self.model.track(
                        image,
                        verbose=False,
                        device='cpu',
                        persist=True,
                        conf=self.model.overrides.get('conf', 0.25),
                        iou=self.model.overrides.get('iou', 0.45),
                        classes=self.model.overrides.get('classes', None)
                    )
                else:
                    results = self.model.predict(
                        image,
                        verbose=False,
                        device='cpu',
                        conf=self.model.overrides.get('conf', 0.25),
                        iou=self.model.overrides.get('iou', 0.45),
                        classes=self.model.overrides.get('classes', None)
                    )
            else:
                raise

        if results is None:
            return annotated_image, detections

        if track:
            # Clean up trajectories for objects that are no longer tracked
            active_ids = []
            for predictions in results:
                if predictions is None or predictions.boxes is None:
                    continue
                for bbox in predictions.boxes:
                    if hasattr(bbox, 'id') and bbox.id is not None:
                        try:
                            active_ids.append(int(bbox.id))
                        except Exception:
                            pass

            for id_ in list(self.tracking_trajectories.keys()):
                if id_ not in active_ids:
                    del self.tracking_trajectories[id_]

            # Process results
            for predictions in results:
                if predictions is None:
                    continue

                if predictions.boxes is None:
                    continue

                # Process boxes
                for bbox in predictions.boxes:
                    scores = bbox.conf
                    classes = bbox.cls
                    bbox_coords = bbox.xyxy

                    # Check if tracking IDs are available
                    if hasattr(bbox, 'id') and bbox.id is not None:
                        ids = bbox.id
                    else:
                        ids = [None] * len(scores)

                    # Process each detection
                    for score, class_id, bbox_coord, id_ in zip(scores, classes, bbox_coords, ids):
                        xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()

                        detections.append([
                            [xmin, ymin, xmax, ymax],
                            float(score),
                            int(class_id),
                            int(id_) if id_ is not None else None
                        ])

                        # Draw bounding box
                        cv2.rectangle(
                            annotated_image,
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (0, 0, 225),
                            2
                        )

                        # Add label
                        class_name = predictions.names[int(class_id)] if int(class_id) in predictions.names else str(int(class_id))
                        label = f"ID: {int(id_) if id_ is not None else 'N/A'} {class_name} {float(score):.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dim, baseline = text_size[0], text_size[1]

                        y_top = max(int(ymin) - dim[1] - baseline, 0)
                        cv2.rectangle(
                            annotated_image,
                            (int(xmin), y_top),
                            (int(xmin) + dim[0], int(ymin)),
                            (30, 30, 30),
                            cv2.FILLED
                        )
                        cv2.putText(
                            annotated_image,
                            label,
                            (int(xmin), max(int(ymin) - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )

                        # Update tracking trajectories
                        if id_ is not None:
                            centroid_x = (xmin + xmax) / 2
                            centroid_y = (ymin + ymax) / 2

                            if int(id_) not in self.tracking_trajectories:
                                self.tracking_trajectories[int(id_)] = deque(maxlen=10)

                            self.tracking_trajectories[int(id_)].append((centroid_x, centroid_y))

            # Draw trajectories
            for id_, trajectory in self.tracking_trajectories.items():
                for i in range(1, len(trajectory)):
                    thickness = int(2 * (i / len(trajectory)) + 1)
                    cv2.line(
                        annotated_image,
                        (int(trajectory[i - 1][0]), int(trajectory[i - 1][1])),
                        (int(trajectory[i][0]), int(trajectory[i][1])),
                        (255, 255, 255),
                        thickness
                    )

        else:
            # Process results for non-tracking mode
            for predictions in results:
                if predictions is None:
                    continue

                if predictions.boxes is None:
                    continue

                for bbox in predictions.boxes:
                    scores = bbox.conf
                    classes = bbox.cls
                    bbox_coords = bbox.xyxy

                    for score, class_id, bbox_coord in zip(scores, classes, bbox_coords):
                        xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()

                        detections.append([
                            [xmin, ymin, xmax, ymax],
                            float(score),
                            int(class_id),
                            None
                        ])

                        cv2.rectangle(
                            annotated_image,
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (0, 0, 225),
                            2
                        )

                        class_name = predictions.names[int(class_id)] if int(class_id) in predictions.names else str(int(class_id))
                        label = f"{class_name} {float(score):.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dim, baseline = text_size[0], text_size[1]

                        y_top = max(int(ymin) - dim[1] - baseline, 0)
                        cv2.rectangle(
                            annotated_image,
                            (int(xmin), y_top),
                            (int(xmin) + dim[0], int(ymin)),
                            (30, 30, 30),
                            cv2.FILLED
                        )
                        cv2.putText(
                            annotated_image,
                            label,
                            (int(xmin), max(int(ymin) - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )

        return annotated_image, detections

    def get_class_names(self):
        """
        Get the names of the classes that the model can detect

        Returns:
            list or dict: Class names from the loaded model
        """
        return self.model.names