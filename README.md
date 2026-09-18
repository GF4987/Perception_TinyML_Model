# Perception TinyML Engine

[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi-red.svg)](https://www.raspberrypi.com/)
[![Computer Vision](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

A lightweight computer vision and gesture recognition suite engineered for embedded Raspberry Pi hardware using native `Picamera2` frame acquisition. Developed to enable fast, real-time spatial detection and gesture interaction at the edge.

> **Project Origin:** Created as part of the award-winning **Perception Innovations** team entry for the Purdue ECE Spark Competition (Holographic Display Interface).

---

## Repository Overview & File Comparison

This repository consists strictly of two standalone Python vision pipelines utilizing `Picamera2` for hardware-accelerated camera capture at $640 \times 480$ resolution (`BGR888`).

### File Breakdowns

* **`mobilenetv2.py`**
  * **Core Focus:** Deep learning-based facial detection engine.
  * **Architecture:** Custom MobileNetV2 backbone ($\alpha = 0.35$, input shape $96 \times 96 \times 3$) paired with Single Shot MultiBox Detector (SSD) prediction heads implemented in TensorFlow/Keras.
  * **Pipeline:** 
    * Captures live frame arrays via `Picamera2.capture_array()`.
    * Preprocesses frames by resizing to $96 \times 96$ and normalizing pixel values to $[0, 1]$.
    * Generates anchor boxes matched to output feature maps to evaluate classification and localization predictions.
    * Decodes predictions and applies Non-Maximum Suppression (NMS) with configurable IoU (`0.4`) and confidence (`0.5`) thresholds.
  * **UI & Controls:** Displays real-time FPS counter, face count, and annotated bounding boxes. Press `'q'` to quit and display the session summary; press `'s'` to export the current frame as `detected_face_N.jpg`.

* **`gesture_detect.py`**
  * **Core Focus:** Low-overhead, heuristic spatial finger tracking and direction detection.
  * **Architecture:** Lightweight OpenCV/NumPy processing pipeline utilizing HSV skin segmentation, morphological filtering, spatial moments, and temporal history queue smoothing.
  * **Pipeline:**
    * Converts `Picamera2` BGR frames to HSV and applies dual-range thresholding to handle red-hue wraparound (checking $170\text{--}179$ when `H low` $\le 10$).
    * Cleans binary masks using $7 \times 7$ ellipse morphological opening (2 iterations) and closing (3 iterations).
    * Filters contours by area (`min_area = 3000`) and aspect ratio via `minAreaRect` (`aspect = 1.8`).
    * Computes spatial centroid using moments ($\frac{M_{10}}{M_{00}}, \frac{M_{01}}{M_{00}}$) and identifies the finger tip as the contour point with maximum Euclidean distance from the centroid.
    * Evaluates vector direction (`LEFT` vs `RIGHT`) and stabilizes outputs across a temporal queue (`smooth = 9` frames) via majority voting.
  * **UI & Controls:** Opens dual windows (`Finger Direction` overlay and `Skin Mask + Trackbars`). Press `'q'` to quit; press `'p'` to dump active trackbar parameters to the terminal.

---

## System Requirements & Setup

### Prerequisites
- Raspberry Pi with Pi Camera Module
- Raspberry Pi OS with `picamera2` configured
- Python 3.9+

### Installation & Environment Setup

```bash
# Clone the repository
git clone [https://github.com/your-username/perception-tinyml-engine.git](https://github.com/your-username/perception-tinyml-engine.git)
cd perception-tinyml-engine

# Create virtual environment inheriting system site-packages for Picamera2 access
python -m venv --system-site-packages venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install tensorflow opencv-python numpy
```

---

## Usage

### 1. Run MobileNetV2 Face Detector
Launch the deep learning face detection engine:

```bash
python mobilenetv2.py
```

* **Interactive Controls:**
  * `q` — Quit detection and display session summary metrics (frames processed, average FPS, saved frames).
  * `s` — Save the active annotated frame snapshot to disk (`detected_face_N.jpg`).

---

### 2. Run Index Finger Gesture Detector
Launch the heuristic finger directional tracker with interactive calibration trackbars:

```bash
python gesture_detect.py
```

* **Calibration & Operational Workflow:**
  1. Use the **Skin Mask + Trackbars** window to adjust `H low/high`, `S low/high`, and `V low/high` until your hand appears as a clean white mask.
  2. Adjust `Min area` and `Aspect x10` to filter out background noise and ensure only elongated finger contours are tracked.
  3. Point your index finger left or right. The primary window displays directional vectors (`< LEFT` or `RIGHT >`), contour bounds, centroid marker, aspect ratio calculations, and real-time FPS.
* **Interactive Controls:**
  * `q` — Quit application.
  * `p` — Print current trackbar parameters (`params` dictionary) directly to stdout.

---

## Performance & Optimization

* **Zero-Copy Memory Access:** Uses `Picamera2.capture_array()` across both `mobilenetv2.py` and `gesture_detect.py` to map BGR frames straight to NumPy arrays, avoiding OpenCV `VideoCapture` overhead on Raspberry Pi hardware.
* **Aggressive Input Downsampling:** Resizes input tensors to $96 \times 96$ within `mobilenetv2.py` to minimize FLOPs and memory overhead during model inference.
* **Efficient Heuristic Filtering:** `gesture_detect.py` avoids heavy neural network compute by using geometric moment calculations ($M_{10}, M_{01}$) and spatial distance metrics.
* **Temporal Stabilization:** Uses a rolling buffer (`smooth = 9`) to eliminate single-frame directional flicker without introducing noticeable latency.
