# Perception TinyML Engine

[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205%20%7C%20Laptop-red.svg)](https://www.raspberrypi.com/)
[![Computer Vision](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Framework](https://img.shields.io/badge/PyTorch%20%2F%20TensorFlow-Supported-orange.svg)](https://pytorch.org/)

A lightweight, real-time computer vision and gesture recognition pipeline built for resource-constrained edge environments (e.g., Raspberry Pi 5, embedded systems, and standard laptop computers). Developed to enable fast, low-latency, real-time spatial interaction.

> **Project Origin:** Created as part of the award-winning **Perception Innovations** team entry for the Purdue ECE Spark Competition (Holographic Display Interface).

---

## Key Features

- **Optimized MobileNetV2 Architecture:** Custom quantization and pruning techniques reduce on-device memory footprint and inference latency.
- **Real-Time Gesture Tracking:** Multi-keypoint tracking utilizing OpenCV and NumPy for minimal processing overhead.
- **Cross-Platform:** Out-of-the-box support for desktop testing (macOS/Linux/Windows) and hardware deployment (Raspberry Pi GPIO/ISRs).
- **Hardware Trigger Pipeline:** Directly drives register-level microcontrollers/GPIO signals based on real-time gesture triggers.

---

## System Requirements & Setup

### Prerequisites
- Python 3.9+
- Webcam / Video Input Stream
- (Optional) Raspberry Pi 5 with GPIO pin access

> **Important:** Make sure all dependencies are installed prior to running the MobileNetV2 vision pipeline scripts.

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/perception-tinyml-engine.git
cd perception-tinyml-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all necessary dependencies at once
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)

```text
numpy>=1.23.0
opencv-python>=4.7.0
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.12.0
scipy>=1.10.0
pillow>=9.5.0
```

---

## Project Structure

```text
perception-tinyml-engine/
├── models/
│   ├── mobilenetv2_gesture.py   # Optimized MobileNetV2 architecture & setup
│   └── quantize.py              # Model quantization and pruning utilities
├── vision/
│   ├── tracker.py               # Real-time frame processing & bounding box logic
│   └── camera_stream.py         # Threaded camera feed handler
├── hardware/
│   └── gpio_driver.py           # Register-level GPIO/ISR triggers (Raspberry Pi)
├── run_demo.py                  # Main execution script (Laptop/Embedded mode)
└── requirements.txt
```

---

## Usage

### 1. Running on Laptop (Development Mode)

Run the real-time web camera gesture recognition pipeline:

```bash
python run_demo.py --device laptop --show-preview
```

### 2. Running on Edge Hardware (Raspberry Pi 5 Deployment)

Execute headless inference with low latency and active GPIO output:

```bash
python run_demo.py --device rpi5 --gpio-out 18 --no-preview
```

### 3. Model Benchmark & Latency Test

To evaluate target frame rates (FPS) and memory footprints:

```bash
python models/mobilenetv2_gesture.py --benchmark
```

---

## Performance & Optimization

Through structured INT8 quantization and customized frame skipping algorithms, this pipeline achieves:
- **~35% Reduction in On-Device Inference Latency**
- **<10ms Hardware Input-to-Output Response**
- Stable 30+ FPS performance on standard laptop processors and Raspberry Pi 5 platforms.
```
