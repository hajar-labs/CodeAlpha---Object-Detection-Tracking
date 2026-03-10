"""
Object Detection and Tracking — CodeAlpha AI Internship Task 4
=============================================================
Uses YOLOv8 for detection + SORT tracking algorithm.
Run on webcam or any video file.
"""

import cv2
import numpy as np
import argparse
import time
from ultralytics import YOLO
from sort_tracker import SORTTracker


# ─── COCO class names ────────────────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ─── Distinct color palette for tracking IDs ─────────────────────────────────
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)


def get_color(track_id: int) -> tuple:
    return tuple(int(c) for c in COLORS[track_id % len(COLORS)])


# ─── Drawing helpers ──────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, track_id, label, conf):
    color = get_color(track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"ID:{track_id} {label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    # filled label background
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, fps, num_objects, frame_idx):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS   : {fps:5.1f}",       (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {num_objects:4d}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame  : {frame_idx:5d}",   (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


# ─── Main pipeline ────────────────────────────────────────────────────────────
def run(source, model_path, conf_thresh, iou_thresh, classes, show, save_path):
    """
    Parameters
    ----------
    source      : int (webcam index) or str (video / image path)
    model_path  : path to YOLOv8 .pt weights
    conf_thresh : detection confidence threshold
    iou_thresh  : NMS IoU threshold
    classes     : list[int] | None — filter to specific COCO class IDs
    show        : bool — display window
    save_path   : str | None — output video path
    """
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] Opening source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_in, (width, height))
        print(f"[INFO] Saving output to: {save_path}")

    tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)

    frame_idx = 0
    fps_timer = time.time()
    fps_display = 0.0

    print("[INFO] Running… press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. Detect ──────────────────────────────────────────────────────
        results = model(
            frame,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=classes,
            verbose=False,
        )[0]

        # ── 2. Parse detections → [x1,y1,x2,y2,conf,cls] ──────────────────
        detections = []
        det_meta   = []   # (class_id, conf) paired to each detection row
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf])
            det_meta.append((cls, conf))

        dets_np = np.array(detections, dtype=np.float32) if detections \
                  else np.empty((0, 5), dtype=np.float32)

        # ── 3. Track ───────────────────────────────────────────────────────
        tracks = tracker.update(dets_np)   # → [[x1,y1,x2,y2,id], ...]

        # ── 4. Draw ────────────────────────────────────────────────────────
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])

            # Find best-matching detection for label / conf
            best_cls, best_conf = 0, 0.0
            for (cls, conf), det in zip(det_meta, dets_np):
                dx1, dy1, dx2, dy2 = map(int, det[:4])
                overlap_x = max(0, min(x2, dx2) - max(x1, dx1))
                overlap_y = max(0, min(y2, dy2) - max(y1, dy1))
                if overlap_x * overlap_y > 0:
                    if conf > best_conf:
                        best_conf = conf
                        best_cls  = cls

            label = COCO_CLASSES[best_cls] if best_cls < len(COCO_CLASSES) else "object"
            draw_box(frame, x1, y1, x2, y2, track_id, label, best_conf)

        # ── 5. HUD ─────────────────────────────────────────────────────────
        frame_idx += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps_display = frame_idx / (time.time() - (fps_timer - elapsed))
            fps_timer   = time.time()
            frame_idx   = 0

        draw_hud(frame, fps_display, len(tracks), frame_idx)

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("Object Detection & Tracking — CodeAlpha", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 + SORT Object Detection & Tracking"
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: webcam index (0,1…) or path to video file"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLOv8 model weights (yolov8n/s/m/l/x.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Detection confidence threshold (default: 0.4)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--classes", nargs="+", type=int, default=None,
        help="Filter by COCO class IDs, e.g. --classes 0 2 (person, car)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Disable live display window"
    )
    parser.add_argument(
        "--save", default=None,
        help="Path to save output video, e.g. output.mp4"
    )

    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    run(
        source     = source,
        model_path = args.model,
        conf_thresh= args.conf,
        iou_thresh = args.iou,
        classes    = args.classes,
        show       = not args.no_show,
        save_path  = args.save,
    )