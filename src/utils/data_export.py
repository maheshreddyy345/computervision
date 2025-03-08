import json
import csv
import cv2
import os
from datetime import datetime
from pathlib import Path

class DataExporter:
    def __init__(self, output_dir="data/output"):
        """Initialize data exporter.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / timestamp
        self.session_dir.mkdir(exist_ok=True)
        
        # Initialize CSV writer
        self.csv_file = open(self.session_dir / "tracking_data.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame", "track_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "center_x", "center_y"
        ])
        
        # Initialize video writer
        self.video_writer = None
        
    def setup_video_writer(self, frame_width, frame_height, fps):
        """Set up video writer with frame properties."""
        output_path = str(self.session_dir / "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height)
        )
        
    def export_frame(self, frame_number, tracks, frame=None):
        """Export tracking data for current frame.
        
        Args:
            frame_number: Current frame number
            tracks: Dictionary of active tracks
            frame: Optional frame to save to video
        """
        # Export tracking data to CSV
        for track_id, track in tracks.items():
            if track['lost_count'] > 0:
                continue
                
            bbox = track['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            self.csv_writer.writerow([
                frame_number,
                track_id,
                track['class_name'],
                track['confidence'],
                *bbox,
                center_x,
                center_y
            ])
        
        # Export frame to video if provided
        if frame is not None and self.video_writer is not None:
            self.video_writer.write(frame)
            
    def export_trajectory_data(self, tracks):
        """Export complete trajectory data as JSON.
        
        Args:
            tracks: Dictionary of all tracks
        """
        trajectory_data = {}
        
        for track_id, track in tracks.items():
            trajectory_data[track_id] = {
                'class_name': track['class_name'],
                'trajectory': track['trajectory'],
                'confidence': track.get('confidence', 0)
            }
            
        output_path = self.session_dir / "trajectories.json"
        with open(output_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
            
    def close(self):
        """Clean up and close file handles."""
        self.csv_file.close()
        if self.video_writer is not None:
            self.video_writer.release()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
