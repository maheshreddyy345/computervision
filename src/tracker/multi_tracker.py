import cv2
import numpy as np
from typing import Dict, List, Tuple

class Track:
    def __init__(self, tracker, bbox):
        self.tracker = tracker
        self.bbox = bbox
        self.lost = False
        self.lost_count = 0

class MultiTracker:
    def __init__(self, max_lost=30):
        """Initialize multi-object tracker.
        
        Args:
            max_lost (int): Maximum number of frames to keep lost tracks
        """
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.max_lost = max_lost
    
    def update_tracks(self, frame, detections):
        """Update tracks with new detections.
        
        Args:
            frame: Current frame
            detections: List of detections, each with 'bbox' key
        """
        if not detections:
            return
            
        # Convert frame to grayscale for better tracking
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Create new tracks for detections
        for det in detections:
            bbox = det['bbox']
            # Create tracker
            tracker = cv2.TrackerMIL_create()
            success = tracker.init(gray, bbox)
            if success:
                self.tracks[self.next_id] = Track(tracker, bbox)
                self.next_id += 1
    
    def update(self, frame):
        """Update all tracks with new frame.
        
        Args:
            frame: Current frame
            
        Returns:
            List of active track bounding boxes
        """
        if not self.tracks:
            return []
            
        # Convert frame to grayscale for better tracking
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Update each track
        tracks_to_delete = []
        for track_id, track in self.tracks.items():
            if track.lost:
                track.lost_count += 1
                if track.lost_count > self.max_lost:
                    tracks_to_delete.append(track_id)
                continue
                
            success, bbox = track.tracker.update(gray)
            if success:
                track.bbox = tuple(map(int, bbox))
                track.lost = False
                track.lost_count = 0
            else:
                track.lost = True
                track.lost_count += 1
                if track.lost_count > self.max_lost:
                    tracks_to_delete.append(track_id)
                    
        # Remove lost tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
            
        return [track.bbox for track in self.tracks.values() if not track.lost]
