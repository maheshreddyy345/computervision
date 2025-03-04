import cv2
import numpy as np

def draw_trajectory(frame, track):
    """Draw trajectory for a track.
    
    Args:
        frame: Input frame
        track: Track information containing trajectory
    """
    trajectory = track['trajectory']
    if len(trajectory) < 2:
        return
    
    # Draw line through trajectory points
    points = np.array(trajectory, dtype=np.int32)
    cv2.polylines(frame, [points], False, (0, 255, 0), 2)
    
    # Draw direction arrow
    if len(trajectory) >= 2:
        last_point = trajectory[-1]
        prev_point = trajectory[-2]
        
        # Calculate angle for arrow
        angle = np.arctan2(last_point[1] - prev_point[1],
                          last_point[0] - prev_point[0])
        
        # Draw arrow head
        arrow_length = 20
        arrow_angle = np.pi/6  # 30 degrees
        
        p1 = (int(last_point[0] - arrow_length * np.cos(angle + arrow_angle)),
              int(last_point[1] - arrow_length * np.sin(angle + arrow_angle)))
        p2 = (int(last_point[0] - arrow_length * np.cos(angle - arrow_angle)),
              int(last_point[1] - arrow_length * np.sin(angle - arrow_angle)))
        
        cv2.line(frame, last_point, p1, (0, 255, 0), 2)
        cv2.line(frame, last_point, p2, (0, 255, 0), 2)
