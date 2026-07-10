# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OmniUAV is a PyQt5-based multi-UAV visualization and 3D reconstruction application. It provides:
- Real-time multi-camera feed visualization from multiple UAVs
- TSDF-based 3D scene reconstruction from RGB-D data
- LLM integration for analyzing camera frames
- Object detection and tracking (VisDrone-trained models)
- Cross-camera object tracking

## Running the Application

```bash
# Setup (first time)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run
python app.py

# Alternative: Use the batch script (attempts conda activation)
run_omniuav.bat
```

## Configuration

Settings are managed via `config.json` (auto-generated on first run) and the UI settings dialog (File > Settings). Categories include:

- **LLM**: API key, base URL, model, timeout, auto-analyze on pause
- **Detection**: Model architecture, confidence threshold, device
- **Tracking**: KCF tracker parameters (padding, features, kernel, lambda)
- **Cross-camera**: Similarity threshold, max distance, feature type
- **Multi-frame LLM**: Frame count, interval, analysis type
- **UI**: Theme, default view, show FPS/confidence
- **Data**: Default directories, save options

Environment variables (for LLM):
- `LLM_API_KEY` (required if not set in config)
- `LLM_BASE_URL` (optional, default: `https://api.openai.com/v1`)
- `LLM_MODEL` (optional, default: `gpt-4o-mini`)

## Data Format

### Video Input
Place in a folder with:
- `cam01.mp4`, `cam02.mp4`, `cam03.mp4`, `cam04.mp4` (one per UAV)
- OR a `rgb/` subdirectory with image sequences (`.jpg`, `.jpeg`, `.png`)

### 3D Reconstruction Input
Required structure in data directory:
- `transforms.csv` - Camera poses and intrinsics (format: image_number, 3x4 pose matrix flattened, fx, fy, cx, cy)
- `depth/` - Depth images as 16-bit PNG (millimeters, 65535 = invalid)
- `rgb/` - Color images (`.jpg` or `.png`)

The default data directory is `examples/`.

## Architecture

The codebase follows a modular PyQt5 architecture with separation of concerns:

### Directory Structure

- `app.py` - Main window and application orchestration
- `tabs/` - Main UI tabs (camera feed, point cloud visualization)
- `widgets/` - Custom widgets (camera feed display, video stream)
- `workers/` - Background threads (TSDF reconstruction, LLM API calls)
- `utils/` - Core utilities (detection, tracking, TSDF, LLM client, settings)
- `dialogs/` - UI dialogs (settings)

### Key Components

**Tabs** (`tabs/`):

- `MultiUavCameraTab` (camera_tab.py): Manages 4 UAV camera feeds
  - View modes: single camera or 2x2 grid
  - Video sources: MP4 files (`VideoStream`) or image sequences (`ImageSequenceStream`)
  - Object detection with VisDrone models (Faster R-CNN, FCOS, RetinaNet)
  - KCF-based single-object tracking
  - Cross-camera tracking to match objects across UAVs
  - Pause/resume with automatic LLM analysis
  - Frame buffer for multi-frame temporal analysis

- `PlyMeshTab` (reconstruction_tab.py): 3D visualization and reconstruction
  - PyQtGraph OpenGL viewer
  - PLY mesh loading (ASCII format with vertex colors)
  - TSDF reconstruction via `TsdfReconstructionWorker`
  - Live preview updates during reconstruction

**Workers** (`workers/`):

- `TsdfReconstructionWorker`: Background thread for TSDF fusion
  - Progress signals for live mesh updates
  - Volume bounds estimation from camera frustums
  - Marching cubes mesh extraction

- `LlmWorker`: Async LLM API calls
  - Queue-based request handling to prevent concurrent calls

**Widgets** (`widgets/`):

- `CameraFeedWidget`: Display widget for single camera feed
  - Detection box click handling for tracker initialization
  - Pause/resume button
  - Image scaling and aspect ratio handling

- `VideoStream` / `ImageSequenceStream`: Video and image sequence readers

**Utils** (`utils/`):

- `VisDroneDetector`: Object detection with torchvision models
  - Supports: fasterrcnn_resnet50, fasterrcnn_mobilenet, fcos_resnet50, retinanet_resnet50
  - Custom checkpoint loading
  - Detection visualization

- `ObjectTrackerManager`: KCF tracker management
  - Multi-object tracking per camera
  - Tracker lifecycle management

- `CrossCameraTracker`: Cross-camera object association
  - Feature-based similarity matching (color histogram, deep features)
  - Global track IDs across cameras

- `FrameBuffer`: Circular buffer for temporal frame storage
  - Per-camera frame history
  - Interval-based frame retrieval for multi-frame analysis

- `TSDFVolume`: TSDF volume fusion implementation
  - GPU acceleration via PyTorch (CUDA/MPS/CPU)
  - Marching cubes mesh extraction
  - Camera frustum calculation

- `LlmClient`: OpenAI-compatible API client
  - Multi-modal requests (text + base64 images)

- `SettingsManager`: JSON-based settings persistence

### Key Data Flows

1. **Video playback**: Timer (40ms) → `_advance_streams()` → `Stream.read()` → display
2. **Object detection**: Frame → `VisDroneDetector.detect()` → draw boxes → click to initialize tracker
3. **Object tracking**: Frame → `KCF.update()` → draw tracker boxes
4. **Cross-camera tracking**: Local trackers → `CrossCameraTracker.update()` → global ID assignment
5. **Pause frame**: User clicks pause → save frame → trigger `_handle_pause_frame()` → LLM analysis
6. **TSDF reconstruction**: User clicks "开始重建" → `TsdfReconstructionWorker` → fuse RGB-D frames → extract mesh → save PLY

### Threading Model

- Main thread: UI and video playback timer
- `TsdfReconstructionWorker` (QThread): Background 3D reconstruction with progress signals
- `LlmWorker` (QThread): Async LLM API calls with queue management

## Point Cloud File Support

The PLY loader (`utils/ply_loader.py`) only supports ASCII format. It handles:
- Vertex positions (x, y, z)
- Vertex colors (red, green, blue as 0-255)
- Faces (triangulated if polygon count > 3)

## UI Language

The UI is in Chinese. Key labels:
- "无人机" = UAV
- "镜头" = Camera
- "暂停" = Pause, "继续" = Resume
- "开始重建" = Start Reconstruction
- "在线可视化" = Live Preview
- "大模型交互" = LLM Interaction
- "目标检测" = Object Detection
- "目标跟踪" = Object Tracking
- "跨相机跟踪" = Cross-camera Tracking
