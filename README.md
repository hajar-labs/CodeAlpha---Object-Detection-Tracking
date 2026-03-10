<div align="center">

# 🎯 Object Detection & Tracking

### Real-time multi-object detection and tracking using YOLOv8 + SORT

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

*Built as part of the **CodeAlpha AI Internship** — Task 4*

</div>

---

## ✨ Features

- 🔍 **Real-time detection** of 80 COCO object classes using YOLOv8
- 🆔 **Persistent tracking IDs** that survive occlusion and re-entry
- 🎨 **Color-coded bounding boxes** unique to each tracked object
- 📹 **Webcam & video file** support out of the box
- 💾 **Save annotated output** to MP4 with a single flag
- 🎛️ **Class filtering** — track only the objects you care about
- ⚡ **CPU & GPU compatible** — auto-detected via PyTorch

---

## 🏗️ Architecture

```
┌──────────────┐   frames    ┌─────────────────────┐   detections   ┌──────────────────────┐
│              │────────────▶│                     │───────────────▶│                      │
│  Webcam /    │             │   YOLOv8 Detector   │                │    SORT Tracker      │
│  Video File  │             │                     │                │                      │
│  (OpenCV)    │             │  • 80 COCO classes  │                │  • Kalman Filter     │
│              │             │  • NMS filtering    │                │  • Hungarian Match   │
└──────────────┘             │  • Conf threshold   │                │  • Persistent IDs    │
                             └─────────────────────┘                └──────────┬───────────┘
                                                                               │
                                                                    track IDs + boxes
                                                                               │
                                                                    ┌──────────▼───────────┐
                                                                    │   Annotated Output   │
                                                                    │  • Colored boxes     │
                                                                    │  • ID + class label  │
                                                                    │  • Live FPS HUD      │
                                                                    └──────────────────────┘
```

### Detection — YOLOv8
YOLOv8 is a single-stage, anchor-free object detector. Each frame is passed through the network, which simultaneously predicts class probabilities and bounding box coordinates. Non-Maximum Suppression (NMS) then removes redundant overlapping boxes.

### Tracking — SORT
SORT (Simple Online and Realtime Tracking) maintains a set of tracks across frames using two components:

1. **Kalman Filter** — models each object as a constant-velocity box and predicts its next position
2. **Hungarian Algorithm** — optimally assigns new detections to existing predicted tracks by maximising IoU overlap

> **Track lifecycle:** `New detection` → *(3 confirmed hits)* → `Active track` → *(30 frames without update)* → `Deleted`

---

## 📁 Project Structure

```
CodeAlpha_ObjectDetectionTracking/
│
├── main.py             # Entry point — detection + tracking pipeline
├── sort_tracker.py     # Pure-NumPy SORT implementation
├── requirements.txt    # Python dependencies
└── README.md           # You are here
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
```

**2. Create a virtual environment** *(recommended)*
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> YOLOv8 weights (`yolov8n.pt`, ~6 MB) are **downloaded automatically** on the first run.

---

## 🚀 Usage

### Run on webcam
```bash
python main.py
```

### Run on a video file
```bash
python main.py --source path/to/video.mp4
```

### Save the annotated output
```bash
python main.py --source video.mp4 --save output.mp4
```

### Filter specific object classes
```bash
# Track only people (0) and cars (2)
python main.py --classes 0 2
```

### Choose a model by size/accuracy
```bash
python main.py --model yolov8n.pt   # nano  — fastest ⚡
python main.py --model yolov8s.pt   # small
python main.py --model yolov8m.pt   # medium
python main.py --model yolov8l.pt   # large
python main.py --model yolov8x.pt   # extra-large — most accurate 🎯
```

Press **`q`** at any time to quit the live window.

---

## 🎛️ CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Webcam index or path to video file |
| `--model` | `yolov8n.pt` | YOLOv8 weights (auto-downloaded) |
| `--conf` | `0.40` | Minimum detection confidence |
| `--iou` | `0.45` | NMS IoU threshold |
| `--classes` | *(all)* | Space-separated COCO class IDs to keep |
| `--no-show` | `False` | Disable the live display window |
| `--save` | *(none)* | Path for saving annotated output video |

---

## 📊 On-Screen HUD

```
┌──────────────────┐
│ FPS   :   42.3   │  ← Real-time processing speed
│ Objects:    5    │  ← Currently active tracked objects
│ Frame :   318    │  ← Frame counter
└──────────────────┘

 [ID:3 person 94%]       ← Track ID · class · confidence
 ┌─────────────────┐
 │                 │  ← Color-coded bounding box (unique per ID)
 └─────────────────┘
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥ 8.0 | YOLOv8 detection model |
| `opencv-python` | ≥ 4.8 | Video I/O and frame rendering |
| `numpy` | ≥ 1.24 | Array operations |
| `scipy` | ≥ 1.11 | Hungarian algorithm (linear_sum_assignment) |

---

## 🔑 Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **COCO** | Dataset of 80 everyday object classes used to pre-train YOLOv8 |
| **NMS** | Non-Maximum Suppression — removes duplicate overlapping boxes |
| **Kalman Filter** | Probabilistic model that predicts a track's next position from its velocity |
| **Hungarian Algorithm** | Optimal O(n³) bipartite matching between detections and tracks |
| **IoU** | Intersection over Union — measures bounding box overlap (0–1) |
| **Track age** | Number of frames since a track last received a matched detection |

---

## 🤝 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — state-of-the-art object detection
- [SORT Paper](https://arxiv.org/abs/1602.00763) — Bewley et al., 2016
- [CodeAlpha](https://www.codealpha.tech) — AI Internship Program

---

<div align="center">
  Made with ❤️ for the <strong>CodeAlpha AI Internship</strong>
</div>
