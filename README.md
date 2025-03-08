# Real-Time Object Detection and Tracking System

A comprehensive computer vision system that performs real-time object detection and tracking using YOLOv8 and OpenCV. The system supports multiple scenarios including traffic monitoring, retail analytics, and automotive testing.

## Key Features

- Real-time object detection using YOLOv8
- Multi-object tracking with unique IDs
- Trajectory visualization
- Data export functionality (CSV format)
- Support for both video files and webcam input
- Configurable detection and tracking parameters
- Real-time performance (28-31 FPS on CPU)

## Technical Stack

- Python 3.x
- OpenCV for video processing
- YOLOv8 for object detection
- Custom tracking algorithms
- NumPy for numerical operations

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd object-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python src/main.py --source <video_path_or_camera_id>
```

Advanced options:

```bash
python src/main.py --source data/video.mp4 \
                   --conf-thres 0.3 \
                   --detection-interval 2 \
                   --max-trajectory 30 \
                   --max-lost 20 \
                   --iou-threshold 0.4 \
                   --debug \
                   --export-data
```

### Parameters

- `--source`: Path to video file or camera ID (default: 0)
- `--conf-thres`: Confidence threshold for detection (default: 0.25)
- `--detection-interval`: Run detection every N frames (default: 1)
- `--max-trajectory`: Maximum length of object trajectories (default: 30)
- `--max-lost`: Maximum frames to keep lost tracks (default: 30)
- `--iou-threshold`: IOU threshold for track matching (default: 0.3)
- `--debug`: Enable debug mode with visualizations
- `--export-data`: Export tracking data to CSV

## Project Structure

```plaintext
object_detection/
├── src/
│   ├── detector/     # YOLO detection implementation
│   ├── tracker/      # Object tracking algorithms
│   └── utils/        # Helper functions and data export
├── models/           # Pre-trained YOLO models
├── data/
│   ├── input/        # Input videos
│   └── output/       # Tracking data and processed videos
└── requirements.txt  # Project dependencies
```

## Performance Optimization

The system includes several optimizations:

- Configurable detection intervals
- Efficient track management
- Vectorized operations for bbox calculations
- Automatic resource cleanup

## Tested Scenarios

Successfully tested with:

- Store environment (static object detection)
- Traffic monitoring (vehicle tracking)
- Automotive reviews (mixed object detection)
- Urban scenes (multi-class detection)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
