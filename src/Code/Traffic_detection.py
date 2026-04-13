from ultralytics import YOLO
import cv2
import os

# --------------------------------------------------
# Paths
# --------------------------------------------------
model_path = "/home/alien/cv_p3/ext_models/traffic_sign_detector.pt"
video_path = "/home/alien/cv_p3/scene11/Undist/2023-03-11_17-19-53-front_undistort.mp4"
output_path = "/home/alien/cv_p3/scene11/Undist/2023-03-11_17-19-53-front_undistort_detected.mp4"

# --------------------------------------------------
# Load model
# --------------------------------------------------
detector = YOLO(model_path)

# --------------------------------------------------
# Open input video
# --------------------------------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Unable to open input video: {video_path}")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30.0  # fallback if metadata is bad

# --------------------------------------------------
# Create output video writer
# --------------------------------------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Unable to create output video: {output_path}")

# --------------------------------------------------
# Inference loop
# --------------------------------------------------
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    # You can tune conf and imgsz if needed
    results = detector(frame, conf=0.25, verbose=False)

    annotated_frame = frame.copy()

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Confidence
            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0

            # Class ID and label
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
            label = detector.names[cls_id] if cls_id in detector.names else f"class_{cls_id}"

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label text
            text = f"{label}: {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Put filled rectangle behind text
            y_text = max(y1 - 10, text_h + 5)
            cv2.rectangle(
                annotated_frame,
                (x1, y_text - text_h - baseline),
                (x1 + text_w, y_text + baseline),
                (0, 255, 0),
                -1
            )

            # Put text
            cv2.putText(
                annotated_frame,
                text,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

    # Show frame
    cv2.imshow("Traffic Sign / Light Detection", annotated_frame)

    # Save frame
    writer.write(annotated_frame)

    frame_count += 1

    # Press q to quit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"[INFO] Done. Processed {frame_count} frames.")
print(f"[INFO] Saved annotated video to: {output_path}")