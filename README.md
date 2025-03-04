# Object Detection and Tracking System

A real-time object detection and tracking system built with OpenCV and YOLO.

## Features
- Real-time video capture from webcam/video files
- Multiple object detection using YOLOv8
- Object tracking across frames
- Bounding box visualization
- Frame rate optimization

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python src/main.py
```

## Project Structure
```
object_detection/
├── requirements.txt
├── README.md
├── src/
│   ├── detector/     # Object detection modules
│   ├── tracker/      # Object tracking modules
│   └── utils/        # Utility functions
├── models/           # Pre-trained models
└── data/
    ├── videos/       # Input videos
    └── output/       # Output data
```
