from pathlib import Path
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import os

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        """Initialize YOLO detector with specified model.
        
        Args:
            model_name (str): Name of the YOLO model to use
        """
        try:
            print("Initializing YOLO detector...")
            
            # Set environment variables
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
            
            # Download and load the model
            model_path = str(Path('models') / model_name)
            if not os.path.exists('models'):
                os.makedirs('models')
            
            if not os.path.exists(model_path):
                print(f"Downloading model to {model_path}...")
                self.model = YOLO('yolov8n.pt')
                self.model.export()
                print("Model downloaded successfully")
            else:
                print(f"Loading model from {model_path}")
                self.model = YOLO(model_path)
            
            print("CUDA available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("Using GPU:", torch.cuda.get_device_name(0))
            else:
                print("Using CPU for inference")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def detect(self, frame, conf_thres=0.25):
        """Detect objects in a frame.
        
        Args:
            frame: Input image/frame
            conf_thres (float): Confidence threshold for detections
            
        Returns:
            list: List of detections, each containing:
                - bbox (tuple): (x1, y1, x2, y2)
                - confidence (float)
                - class_id (int)
                - class_name (str)
        """
        try:
            # Ensure frame is in correct format
            if frame is None:
                print("Error: Frame is None")
                return []
                
            if not isinstance(frame, np.ndarray):
                print(f"Error: Invalid frame type: {type(frame)}")
                return []
                
            if len(frame.shape) != 3:
                print(f"Error: Invalid frame shape: {frame.shape}")
                return []
            
            # Run inference
            results = self.model(frame, verbose=False)
            if not results:
                return []
            
            result = results[0]
            detections = []
            
            # Process each detection
            for box in result.boxes:
                try:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    conf = float(box.conf[0].cpu().numpy())
                    if conf < conf_thres:
                        continue
                        
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    # Ensure coordinates are integers and within frame bounds
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                    
                    # Skip if box is invalid
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    bbox = (x1, y1, x2, y2)
                    detections.append({
                        'bbox': bbox,
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': class_name
                    })
                    
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            if detections:
                print(f"Found {len(detections)} objects")
                for det in detections:
                    print(f"  {det['class_name']}: {det['confidence']:.2f}")
            
            return detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return []
