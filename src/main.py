import cv2
import argparse
import time
from pathlib import Path
from detector.yolo import YOLODetector
import numpy as np

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
    return parser.parse_args()

def main(args):
    # Initialize video capture first to check if it works
    print("Opening video capture...")
    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {width}x{height} @ {fps}fps")
    
    # Initialize detector
    print("Initializing detector...")
    detector = YOLODetector()

    # Initialize tracking variables
    tracks = {}  # id -> {bbox, class_name, lost_count, confidence}
    next_id = 0
    max_lost = 30
    iou_threshold = 0.3
    conf_threshold = 0.4

    frame_count = 0
    start_time = time.time()
    detection_interval = args.detection_interval

    print("Starting main loop...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to read frame")
                break

            # Print frame info
            if args.debug:
                print(f"\nFrame {frame_count}:")
                print(f"Frame shape: {frame.shape}")
                print(f"Frame type: {frame.dtype}")

            # Run detection every N frames
            if frame_count % detection_interval == 0:
                try:
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
                            matched_track_ids.add(track_id)
                            matched_detection_ids.add(best_det_idx)

                    # Create new tracks for unmatched detections
                    if detections:
                        for i, det in enumerate(detections):
                            if i not in matched_detection_ids:
                                tracks[next_id] = {
                                    'bbox': det['bbox'],
                                    'class_name': det['class_name'],
                                    'confidence': det['confidence'],
                                    'lost_count': 0
                                }
                                next_id += 1

                    # Increment lost count for unmatched tracks
                    for track_id, track in list(tracks.items()):
                        if track_id not in matched_track_ids:
                            track['lost_count'] += 1
                            if track['lost_count'] > max_lost:
                                del tracks[track_id]

                except Exception as e:
                    print(f"Error during detection/tracking: {e}")
                    import traceback
                    traceback.print_exc()

            # Draw tracking results
            for track_id, track in tracks.items():
                if track['lost_count'] <= max_lost:
                    try:
                        x1, y1, x2, y2 = track['bbox']
                        # Color based on confidence
                        conf = track['confidence']
                        color = (0, int(255 * conf), 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{track_id}: {track['class_name']} ({conf:.2f})"
                        cv2.putText(frame, label, (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception as e:
                        print(f"Error drawing track {track_id}: {e}")

            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Object Detection and Tracking', frame)

            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    main(args)
