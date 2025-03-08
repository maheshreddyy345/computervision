import cv2
import argparse
import time
from pathlib import Path
from detector.yolo import YOLODetector
from tracker.trajectory import draw_trajectory
from utils.data_export import DataExporter
import numpy as np
from collections import deque

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    if union <= 0:
        return 0
    return intersection / union

def get_center_point(bbox):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection and Tracking')
    parser.add_argument('--source', type=str, default='0',
                      help='Source (0 for webcam, or path to video file)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                      help='Confidence threshold for detections')
    parser.add_argument('--detection-interval', type=int, default=5,
                      help='Run detection every N frames')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    parser.add_argument('--video', type=str,
                      help='Path to video file (optional)')
    parser.add_argument("--max-trajectory", type=int, default=30,
                      help="Maximum trajectory length")
    parser.add_argument("--export-data", action="store_true",
                      help="Export tracking data and video")
    parser.add_argument("--max-lost", type=int, default=45,
                      help="Maximum number of frames to keep lost tracks")
    parser.add_argument("--iou-threshold", type=float, default=0.4,
                      help="IOU threshold for track matching")
    return parser.parse_args()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                      help="Video source (0 for webcam, or video file path)")
    parser.add_argument("--conf-thres", type=float, default=0.4,
                      help="Confidence threshold for detections")
    parser.add_argument("--detection-interval", type=int, default=5,
                      help="Run detection every N frames")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug output")
    parser.add_argument("--max-trajectory", type=int, default=30,
                      help="Maximum trajectory length")
    parser.add_argument("--export-data", action="store_true",
                      help="Export tracking data and video")
    parser.add_argument("--max-lost", type=int, default=45,
                      help="Maximum number of frames to keep lost tracks")
    parser.add_argument("--iou-threshold", type=float, default=0.4,
                      help="IOU threshold for track matching")
    args = parser.parse_args()

    # Initialize video capture
    print("Opening video capture...")
    try:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
    except Exception as e:
        print(f"Error opening video source: {e}")
        return

    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    try:
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video properties: {width}x{height} @ {fps}fps")
        
        # Initialize detector
        print("Initializing detector...")
        detector = YOLODetector()

        # Initialize tracking variables
        tracks = {}  # id -> {bbox, class_name, lost_count, confidence, trajectory}
        next_id = 0
        max_lost = args.max_lost
        iou_threshold = args.iou_threshold
        conf_threshold = args.conf_thres

        # Performance monitoring
        frame_count = 0
        start_time = time.time()
        fps_update_interval = 30
        fps_history = deque(maxlen=10)
        last_fps_update = 0

        # Initialize data exporter if requested
        data_exporter = None
        if args.export_data:
            data_exporter = DataExporter()
            data_exporter.setup_video_writer(width, height, fps)

        print("Starting main loop...")
        while True:
            try:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("End of video stream")
                    break

                # Print frame info
                if args.debug:
                    print(f"\nFrame {frame_count}:")
                    print(f"Frame shape: {frame.shape}")
                    print(f"Frame type: {frame.dtype}")

                # Run detection every N frames
                if frame_count % args.detection_interval == 0:
                    try:
                        if args.debug:
                            print(f"Running detection on frame {frame_count}...")
                        detections = detector.detect(frame, conf_thres=args.conf_thres)
                        
                        # Filter low confidence detections
                        detections = [det for det in detections if det['confidence'] > conf_threshold]
                        
                        if detections and args.debug:
                            print(f"Found {len(detections)} objects:")
                            for det in detections:
                                print(f"  {det['class_name']}: {det['confidence']:.2f}")

                        # Update tracks
                        matched_track_ids = set()
                        matched_detection_ids = set()

                        # Sort tracks by lost count (update most recently seen first)
                        track_items = sorted(tracks.items(), key=lambda x: x[1]['lost_count'])

                        # Match detections to existing tracks
                        for track_id, track in track_items:
                            if track['lost_count'] > max_lost:
                                continue

                            best_iou = 0
                            best_det_idx = -1
                            for i, det in enumerate(detections):
                                if i in matched_detection_ids:
                                    continue

                                # Only match same class
                                if det['class_name'] != track['class_name']:
                                    continue

                                iou = calculate_iou(track['bbox'], det['bbox'])
                                if iou > best_iou and iou > iou_threshold:
                                    best_iou = iou
                                    best_det_idx = i

                            if best_det_idx >= 0:
                                # Update track with new detection
                                det = detections[best_det_idx]
                                track.update({
                                    'bbox': det['bbox'],
                                    'confidence': det['confidence'],
                                    'lost_count': 0
                                })
                                # Update trajectory
                                center = get_center_point(det['bbox'])
                                track['trajectory'].append(center)
                                if len(track['trajectory']) > args.max_trajectory:
                                    track['trajectory'].pop(0)
                                
                                matched_track_ids.add(track_id)
                                matched_detection_ids.add(best_det_idx)

                        # Create new tracks for unmatched detections
                        if detections:
                            for i, det in enumerate(detections):
                                if i not in matched_detection_ids:
                                    center = get_center_point(det['bbox'])
                                    tracks[next_id] = {
                                        'bbox': det['bbox'],
                                        'class_name': det['class_name'],
                                        'confidence': det['confidence'],
                                        'lost_count': 0,
                                        'trajectory': [center]
                                    }
                                    next_id += 1

                        # Increment lost count for unmatched tracks
                        for track_id, track in list(tracks.items()):
                            if track_id not in matched_track_ids:
                                track['lost_count'] += 1
                                if track['lost_count'] > max_lost:
                                    del tracks[track_id]

                    except Exception as e:
                        print(f"Error during detection: {e}")
                        continue

                # Draw tracks and trajectories
                for track_id, track in tracks.items():
                    if track['lost_count'] > 0:
                        continue

                    # Draw bounding box
                    x1, y1, x2, y2 = track['bbox']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{track['class_name']} {track['confidence']:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw trajectory
                    draw_trajectory(frame, track)

                # Export frame data if requested
                if args.export_data:
                    data_exporter.export_frame(frame_count, tracks, frame)

                # Calculate and display FPS
                if frame_count % fps_update_interval == 0:
                    end_time = time.time()
                    fps = fps_update_interval / (end_time - start_time)
                    fps_history.append(fps)
                    avg_fps = sum(fps_history) / len(fps_history)
                    print(f"Average FPS: {avg_fps:.1f}")
                    print(f"Active tracks: {len(tracks)}")
                    print(f"Frame processing time: {(time.time() - loop_start)*1000:.1f}ms")
                    start_time = time.time()

                # Show frame
                cv2.imshow('Object Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User requested exit")
                    break

                frame_count += 1

            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                break

    finally:
        # Clean up
        print("\nCleaning up...")
        if data_exporter:
            data_exporter.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected")
    except Exception as e:
        print(f"Unhandled error: {e}")
    finally:
        print("Exiting...")
        cv2.destroyAllWindows()
