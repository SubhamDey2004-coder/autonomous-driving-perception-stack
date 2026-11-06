# Autonomous Driving Perception Stack

A computer vision project combining vehicle detection, multi-object tracking, and lane detection into a unified video-processing pipeline.

## Features

- **Vehicle detection:** YOLO11n detects objects in traffic videos.
- **Multi-object tracking:** ByteTrack maintains object IDs across frames.
- **Lane detection:** Ultra-Fast Lane Detection v2 identifies road-lane markings.
- **Integrated visualization:** Combines vehicle bounding boxes, tracking IDs, and lane overlays.
- **Performance profiling:** Measures inference, preprocessing, annotation, video-writing time, and effective FPS.

## Tech Stack

- Python
- PyTorch and TorchVision
- OpenCV
- Ultralytics YOLO
- ByteTrack
- Ultra-Fast Lane Detection v2

## Project Structure

```text
autonomous-driving-perception-stack/
â”œâ”€â”€ artifacts/models/       # Model weights (not tracked by Git)
â”œâ”€â”€ configs/                # Configuration files
â”œâ”€â”€ data/samples/           # Sample videos and images
â”œâ”€â”€ docs/                   # Project documentation
â”œâ”€â”€ experiments/            # Experimental and previous implementations
â”œâ”€â”€ external_models/        # External lane-detection repository
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ detection/           # Object detection
â”‚   â”œâ”€â”€ lanes/               # Lane detection
â”‚   â””â”€â”€ tracking/            # Object tracking
â”œâ”€â”€ tests/                   # Automated tests
â”œâ”€â”€ tools/
â”‚   â”œâ”€â”€ integrated_perception.py
â”‚   â””â”€â”€ evaluate_ufld_video.py
â”œâ”€â”€ pytest.ini
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ requirements-dev.txt
â””â”€â”€ README.md
```

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

For development and testing:

```powershell
python -m pip install -r requirements-dev.txt
```

PyTorch installation may require a platform-specific adjustment. Choose a compatible build for your operating system and hardware.

### 3. Obtain model weights

Place the required model weights in `artifacts/models/`:

- `yolo11n.pt` for object detection and tracking
- `culane_res18.pth` for lane detection

Model weights are excluded from Git. Obtain them from their appropriate official sources and verify their licenses before redistribution.

### 4. Obtain the external lane-detection repository

Clone the official Ultra-Fast Lane Detection v2 repository into:

```text
external_models/Ultra-Fast-Lane-Detection-v2/
```

Follow its setup instructions and verify that its model configuration and checkpoint are compatible with this project.

## Run the Integrated Pipeline

From the project root, activate the virtual environment and run:

```powershell
python tools/integrated_perception.py
```

The script processes the configured input video and saves the annotated output under:

```text
outputs/perception/integrated_traffic.mp4
```

Check the paths in the script if your local video or model locations differ.

## Run Tests

Run the automated test suite from the project root:

```powershell
python -m pytest -v
```

The current test suite covers lane-processing functions and ByteTrack object tracking.

## Performance

The integrated pipeline was benchmarked on a CPU-only environment using a 1,260-frame traffic video at 1280 × 720 resolution.

| Metric | Result |
|---|---:|
| Total frames processed | 1,260 |
| Total processing time | 181.00 seconds |
| Effective processing speed | 6.96 FPS |
| Lane inference frames | 158 |
| Lane inference interval | Every 8 frames |
| YOLO detection and tracking | Every frame |

### Component Timing

| Component | Time |
|---|---:|
| YOLO detection and tracking | 81.15 seconds |
| Lane model inference | 56.47 seconds |
| Lane preprocessing | 5.62 seconds |
| Annotation and drawing | 4.19 seconds |
| Video writing | 16.23 seconds |

The lane inference sampling interval was increased from every 5 frames to every 8 frames, reducing the number of lane inference operations while retaining vehicle detection and tracking on every frame.

In the measured baseline, processing time decreased from 237.19 seconds to 181.00 seconds, and effective processing speed improved from 5.31 FPS to 6.96 FPS, an improvement of approximately 31.1%.

Performance varies with hardware, input video, model configuration, and software versions. The reported FPS represents processing throughput, not the output video's playback frame rate. These results are development benchmarks and do not establish real-time or safety-critical readiness.

## Limitations

- Lane predictions may be imperfect, especially around curves or unclear road markings.
- Lane inference is performed less frequently than vehicle tracking to reduce processing time.
- The current pipeline is a prototype and is not validated for real-world autonomous driving or safety-critical use.

## Future Improvements

- Improve lane continuity and handling of curved roads.
- Optimize inference latency.
- Evaluate detection and tracking accuracy using appropriate metrics.
- Improve automated test coverage.
- Compare experimental lane-smoothing approaches.

## Disclaimer

This project is intended for educational and research purposes. It must not be used as a standalone system for vehicle control or safety-critical navigation.