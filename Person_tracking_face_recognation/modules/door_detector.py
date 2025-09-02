"""
Door Detection Module
Detects doors in video frames using YOLO and shows visual indicators only when doors are found
"""

import cv2
import numpy as np
import time


class DoorDetector:
    """Detects doors in video frames and provides visualization"""
    
    def __init__(self, model=None, confidence_threshold=0.6):
        self.model = model  # Use existing YOLO model
        self.confidence_threshold = confidence_threshold
        
        # Door detection data
        self.detected_doors = []
        self.door_history = {}  # Track doors over time
        self.door_id_counter = 1
        self.detection_interval = 10  # Check every 10 frames for performance
        self.frame_counter = 0
        
        # Door tracking parameters
        self.door_stability_threshold = 5  # Frames needed to confirm door
        self.max_door_distance = 100  # Max distance to consider same door
        
    def detect_doors(self, frame):
        """Detect doors in the current frame"""
        self.frame_counter += 1
        
        # Skip frames for performance
        if self.frame_counter % self.detection_interval != 0:
            return self.detected_doors
        
        if self.model is None:
            return []
        
        try:
            # Run YOLO detection
            results = self.model.predict(
                frame, 
                verbose=False,
                conf=self.confidence_threshold,
                classes=[74]  # Door class in COCO dataset
            )
            
            current_doors = []
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a door with good confidence
                        if class_id == 74 and confidence >= self.confidence_threshold:
                            bbox = box.xyxy[0].tolist()
                            
                            # Validate bbox
                            if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                                x1, y1, x2, y2 = bbox
                                
                                # Basic size validation (doors should be reasonably sized)
                                width = x2 - x1
                                height = y2 - y1
                                
                                if (width > 30 and height > 50 and  # Minimum size
                                    height > width * 0.8):  # Doors are typically taller than wide
                                    
                                    door_info = {
                                        'bbox': bbox,
                                        'confidence': confidence,
                                        'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                                        'area': width * height,
                                        'timestamp': time.time()
                                    }
                                    current_doors.append(door_info)
            
            # Update door tracking with stability check
            self.update_door_tracking(current_doors)
            
            return self.detected_doors
            
        except Exception as e:
            print(f"⚠️ Door detection error: {e}")
            return []
    
    def update_door_tracking(self, current_doors):
        """Update door tracking with stability requirements"""
        try:
            current_time = time.time()
            
            # Match current doors to existing tracked doors
            matched_doors = []
            unmatched_current = current_doors.copy()
            
            for door_id, tracked_door in list(self.door_history.items()):
                best_match = None
                best_distance = float('inf')
                
                for i, current_door in enumerate(unmatched_current):
                    # Calculate distance between door centers
                    tracked_center = tracked_door['center']
                    current_center = current_door['center']
                    
                    distance = np.sqrt(
                        (tracked_center[0] - current_center[0]) ** 2 +
                        (tracked_center[1] - current_center[1]) ** 2
                    )
                    
                    if distance < best_distance and distance < self.max_door_distance:
                        best_distance = distance
                        best_match = i
                
                if best_match is not None:
                    # Update existing door
                    matched_door = unmatched_current.pop(best_match)
                    tracked_door.update({
                        'bbox': matched_door['bbox'],
                        'confidence': matched_door['confidence'],
                        'center': matched_door['center'],
                        'area': matched_door['area'],
                        'last_seen': current_time,
                        'stable_count': tracked_door.get('stable_count', 0) + 1
                    })
                    matched_doors.append(door_id)
                else:
                    # Door not seen, decrease stability
                    tracked_door['stable_count'] = max(0, tracked_door.get('stable_count', 0) - 2)
                    
                    # Remove if not seen for too long
                    if current_time - tracked_door.get('last_seen', current_time) > 5.0:  # 5 seconds
                        del self.door_history[door_id]
            
            # Add new unmatched doors
            for new_door in unmatched_current:
                door_id = self.door_id_counter
                self.door_id_counter += 1
                
                self.door_history[door_id] = {
                    **new_door,
                    'id': door_id,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'stable_count': 1
                }
            
            # Update detected doors list with only stable doors
            self.detected_doors = []
            for door_id, door_info in self.door_history.items():
                if door_info.get('stable_count', 0) >= self.door_stability_threshold:
                    self.detected_doors.append({
                        'id': door_id,
                        'bbox': door_info['bbox'],
                        'confidence': door_info['confidence'],
                        'center': door_info['center'],
                        'stable_count': door_info['stable_count']
                    })
            
        except Exception as e:
            print(f"⚠️ Door tracking error: {e}")
    
    def draw_doors(self, frame):
        """Draw door detection boxes only when doors are confirmed"""
        if not self.detected_doors:
            return  # Don't draw anything if no doors detected
        
        try:
            for door in self.detected_doors:
                bbox = door['bbox']
                confidence = door['confidence']
                door_id = door['id']
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1 = max(0, min(w-1, x1))
                y1 = max(0, min(h-1, y1))
                x2 = max(0, min(w-1, x2))
                y2 = max(0, min(h-1, y2))
                
                if x2 > x1 and y2 > y1:  # Valid rectangle
                    # Draw door rectangle in distinctive color
                    color = (0, 165, 255)  # Orange color for doors
                    thickness = 3
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add label
                    label = f"Door {door_id} ({confidence:.2f})"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    text_thickness = 2
                    
                    # Get text size for background
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
                    
                    # Draw text background
                    bg_y1 = y1 - text_height - 10
                    bg_y2 = y1
                    bg_x2 = x1 + text_width + 10
                    
                    if bg_y1 > 0:  # Text fits above rectangle
                        cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
                        cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), text_thickness)
                    else:  # Text goes below rectangle
                        bg_y1 = y2
                        bg_y2 = y2 + text_height + 10
                        cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
                        cv2.putText(frame, label, (x1 + 5, y2 + text_height), font, font_scale, (255, 255, 255), text_thickness)
                        
        except Exception as e:
            print(f"⚠️ Door drawing error: {e}")
    
    def get_door_count(self):
        """Get number of currently detected doors"""
        return len(self.detected_doors)
    
    def get_door_info(self):
        """Get detailed information about detected doors"""
        return self.detected_doors.copy()
    
    def reset_detection(self):
        """Reset door detection data"""
        self.detected_doors = []
        self.door_history = {}
        self.door_id_counter = 1
        print("🚪 Door detection reset")
    
    def set_confidence_threshold(self, threshold):
        """Update confidence threshold for door detection"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
        print(f"🚪 Door detection confidence set to {self.confidence_threshold:.2f}")
    
    def get_statistics(self):
        """Get door detection statistics"""
        return {
            'total_doors_detected': len(self.detected_doors),
            'detection_confidence_avg': np.mean([door['confidence'] for door in self.detected_doors]) if self.detected_doors else 0,
            'tracked_doors': len(self.door_history),
            'frame_counter': self.frame_counter
        }