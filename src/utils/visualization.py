import cv2
import numpy as np
from tracker.trajectory import draw_trajectory

# Define colors for different classes (using distinct colors)
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Dark Blue
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Dark Red
]

def draw_detections(frame, detections, tracks=None):
    """Draw bounding boxes, labels and trajectories on the frame.
    
    Args:
        frame: Input frame
        detections: List of detections from YOLODetector
        tracks: Optional list of tracks for trajectory visualization
        
    Returns:
        frame: Frame with drawn detections and tracks
    """
    frame_with_boxes = frame.copy()
    
    if tracks:
        # Draw tracks
        for track in tracks:
            bbox = track['bbox']
            conf = track['confidence']
            class_name = track['class_name']
            track_id = track['track_id']
            
            # Get color for this class
            color = COLORS[track_id % len(COLORS)]
            
            # Draw bounding box
            cv2.rectangle(frame_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label with track ID
            label = f'{class_name} #{track_id} {conf:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_with_boxes, (bbox[0], bbox[1] - label_height - 10),
                         (bbox[0] + label_width, bbox[1]), color, -1)
            cv2.putText(frame_with_boxes, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw trajectory
            draw_trajectory(frame_with_boxes, track)
    else:
        # Draw detections only
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Get color for this detection
            color = COLORS[i % len(COLORS)]
            
            # Draw bounding box
            cv2.rectangle(frame_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f'{class_name} {conf:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_with_boxes, (bbox[0], bbox[1] - label_height - 10),
                         (bbox[0] + label_width, bbox[1]), color, -1)
            cv2.putText(frame_with_boxes, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame_with_boxes
