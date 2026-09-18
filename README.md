# Perception TinyML Engine

[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi-red.svg)](https://www.raspberrypi.com/)
[![Computer Vision](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

A lightweight computer vision and gesture recognition suite engineered for embedded Raspberry Pi hardware using native `Picamera2` integration. Developed to enable fast, real-time spatial detection and interaction at the edge.

> **Project Origin:** Created as part of the award-winning **Perception Innovations** team entry for the Purdue ECE Spark Competition (Holographic Display Interface).

---

## Repository Overview & File Comparison

This repository houses two complementary vision pipelines and a dependencies manifest. Dedicated register-level hardware drivers and firmware for external microcontrollers are managed in a separate subsystem repository.

### File Breakdowns

* **`tinyml_face_detection.py`**
  * **Core Focus:** Neural network-based facial landmark and object detection.
  * **Architecture:** Builds a custom MobileNetV2 backbone with Single Shot MultiBox Detector (SSD) prediction heads using TensorFlow/Keras.
  * **Functionality:** Ingests live frame arrays from `Picamera2`, preprocesses image matrices to 96x96, and runs anchor box decoding with Non-Maximum Suppression (NMS) to draw localized bounding boxes and confidence scores.

* **`index_finger_direction.py`**
  * **Core Focus:** Low-overhead, heuristic-based spatial orientation tracking.
  * **Architecture:** Lightweight OpenCV/NumPy pipeline utilizing skin-color segmentation and geometric contour analysis.
  * **Functionality:** Processes `Picamera2` frames using HSV thresholding, morphological closing/opening, and minimum area bounding rectangles. Computes shape aspect ratios and contour spatial moments to determine pointing vectors (`LEFT` vs `RIGHT`) with runtime trackbar calibration.

* **`requirements.txt`**
  * Contains the Python dependencies required to run both scripts on Raspberry Pi hardware (e.g., `tensorflow`, `opencv-python`, `numpy`, `scipy`, `pillow`).

---

## System Requirements & Setup

### Prerequisites
- Raspberry Pi (with `picamera2` configured)
- Raspberry Pi Camera Module
- Python 3.9+

### Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/perception-tinyml-engine.git](https://github.com/your-username/perception-tinyml-engine.git)
cd perception-tinyml-engine

# Create virtual environment (inherit system site-packages for picamera2 access)
python -m venv --system-site-packages venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Run Face Detection (TinyML)
Launch the neural network pipeline to perform real-time facial and object detection using the custom MobileNetV2-SSD backbone:

```bash
python tinyml_face_detection.py
```
*Press `q` in the video stream window to terminate the process.*

### Run Index Finger Direction Detector
Start the low-overhead heuristic spatial tracking pipeline. This script initializes the camera feed alongside an interactive configuration window to fine-tune HSV tracking thresholds for specific lighting conditions.

```bash
python index_finger_direction.py
```

**Calibration & Operation Steps:**
1. Ensure your hand is well-lit and clearly visible within the camera frame.
2. Adjust the `H_MIN`, `S_MIN`, and `V_MIN` trackbars in the control window until the binary mask successfully isolates your skin tone from the background noise.
3. Point your index finger. The terminal will continuously output `LEFT` or `RIGHT` directional vectors based on the geometric contour spatial moments and aspect ratio calculations.
*Press `q` in the active window to terminate the tracking process.*

---

## Performance & Optimization

To achieve stable, real-time frame rates on resource-constrained embedded hardware, this engine employs several strict optimization strategies:

* **Native `Picamera2` Memory Management:** By utilizing `Picamera2`'s direct NumPy array mapping, the pipeline bypasses costly memory-copy operations between the camera module's Image Signal Processor (ISP) and the Python runtime environment.
* **Algorithmic Triage:** The system splits heavy neural network inferences (`tinyml_face_detection.py`) from lightweight OpenCV heuristics (`index_finger_direction.py`). This allows the hardware to execute the most computationally efficient method required for the immediate interaction context.
* **Aggressive Downsampling:** Incoming camera frames are structurally downsampled (e.g., to 96x96 matrices for the MobileNetV2 pipeline) prior to analysis, heavily reducing the required FLOPs per frame while preserving localized spatial accuracy.
* **TensorFlow Lite Readiness:** The SSD architecture is structured to support seamless export to `.tflite` formats, allowing for future INT8 or Float16 integer quantization to further minimize the memory footprint and execution latency at the edge.
