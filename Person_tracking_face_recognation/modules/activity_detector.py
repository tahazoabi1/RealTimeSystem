"""
Activity Detection and Analysis Module
Handles motion analysis, posture detection, and activity classification
"""

import numpy as np
import math
import time
from datetime import datetime


class ActivityDetector:
    """Analyzes person activities based on movement patterns and posture"""
    
    def __init__(self, movement_history_size=15):
        self.movement_history_size = movement_history_size
        
        # Tracking data for all active tracks
        self.track_positions = {}
        self.track_velocities = {}
        self.track_directions = {}
        self.track_speeds = {}
        self.track_activities = {}
        self.track_posture_history = {}
        self.activity_confidence = {}
        self.track_height_history = {}  # For proper sitting detection
        
        # Activity smoothing and state management
        self.track_activity_votes = {}  # Vote counting for smoothing
        self.track_stable_activity = {}  # Current stable activity
        self.track_activity_duration = {}  # How long in current activity
        self.activity_change_threshold = 0  # Completely instant - no voting needed
        self.confidence_threshold = 0.1  # Extremely low threshold for immediate response
        
    def compute_iou(self, box1, box2):
        """Enhanced IoU computation"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update_track_motion(self, track_id, bbox, current_time, track_data):
        """Update motion analysis for a track"""
        try:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            
            # Update track data
            track_data['bbox'] = bbox
            track_data['lost'] = 0
            track_data['age'] += 1
            track_data['last_seen'] = current_time
            
            # Initialize tracking data if needed
            if track_id not in self.track_positions:
                self.track_positions[track_id] = [(cx, cy)]
                self.track_velocities[track_id] = []
                self.track_directions[track_id] = []
                self.track_speeds[track_id] = []
                return
            
            # Add position
            self.track_positions[track_id].append((cx, cy))
            if len(self.track_positions[track_id]) > self.movement_history_size:
                self.track_positions[track_id].pop(0)
            
            # Calculate motion if we have previous position
            if len(self.track_positions[track_id]) > 1:
                prev_pos = self.track_positions[track_id][-2]
                curr_pos = self.track_positions[track_id][-1]
                
                # Velocity calculation
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                velocity = (dx, dy)
                speed = math.sqrt(dx*dx + dy*dy)
                
                self.track_velocities[track_id].append(velocity)
                self.track_speeds[track_id].append(speed)
                
                # Direction calculation
                if speed > 0.5:  # Only calculate direction for meaningful movement
                    direction = math.degrees(math.atan2(dy, dx))
                    self.track_directions[track_id].append(direction)
                
                # Limit history size
                if len(self.track_velocities[track_id]) > self.movement_history_size:
                    self.track_velocities[track_id].pop(0)
                if len(self.track_speeds[track_id]) > self.movement_history_size:
                    self.track_speeds[track_id].pop(0)
                if len(self.track_directions[track_id]) > self.movement_history_size:
                    self.track_directions[track_id].pop(0)
            
            # Analyze activity
            self.analyze_activity(track_id, bbox, current_time)
            
        except Exception as e:
            print(f"⚠️ Motion update error for track {track_id}: {e}")
    
    def analyze_activity(self, track_id, bbox, current_time):
        """Enhanced activity analysis with intelligent smoothing and better recognition"""
        try:
            if len(self.track_positions[track_id]) < 2:  # Almost instant - just need 2 points for speed
                return
            
            # Get bbox dimensions
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1
            aspect_ratio = width / height if height > 0 else 1.0
            
            # Calculate motion metrics with minimal history for instant response
            recent_speeds = self.track_speeds[track_id][-3:] if track_id in self.track_speeds else []
            recent_directions = self.track_directions[track_id][-5:] if track_id in self.track_directions else []
            very_recent_speeds = self.track_speeds[track_id][-2:] if track_id in self.track_speeds else []
            
            if not recent_speeds:
                return
            
            avg_speed = np.mean(recent_speeds)
            recent_avg_speed = np.mean(very_recent_speeds) if very_recent_speeds else 0
            max_speed = max(recent_speeds) if recent_speeds else 0
            speed_variance = np.var(recent_speeds) if len(recent_speeds) > 1 else 0
            direction_variance = np.var(recent_directions) if len(recent_directions) > 2 else 0
            
            # Advanced activity classification
            candidate_activity = "Unknown"
            confidence = 0.0
            
            # 1. MOVING Detection (ultra sensitive - any movement shows "Moving")
            if avg_speed > 3 or max_speed > 5 or recent_avg_speed > 2:  # Any slight movement
                candidate_activity = "Moving"
                confidence = 0.9  # High confidence for any movement
            
            # 2. STOP Detection (everything else is stopped)
            else:
                candidate_activity = "Stop"
                confidence = 0.8
            
            # INSTANT RESPONSE SYSTEM - No smoothing, immediate changes
            if track_id not in self.track_stable_activity:
                self.track_stable_activity[track_id] = "Stop"  # Start with stopped
            
            # Instantly update activity based on current frame
            self.track_stable_activity[track_id] = candidate_activity
            self.track_activities[track_id] = candidate_activity
            self.activity_confidence[track_id] = confidence
            
            # Enhanced posture history
            if track_id not in self.track_posture_history:
                self.track_posture_history[track_id] = []
            
            self.track_posture_history[track_id].append({
                'timestamp': current_time,
                'activity': candidate_activity,
                'candidate_activity': candidate_activity,
                'confidence': confidence,
                'candidate_confidence': confidence,
                'speed': avg_speed,
                'aspect_ratio': aspect_ratio,
                'height': height,
                'votes': {}  # No votes in instant mode
            })
            
            # Keep reasonable history
            if len(self.track_posture_history[track_id]) > 60:  # 2 seconds at 30fps
                self.track_posture_history[track_id] = self.track_posture_history[track_id][-60:]
            
        except Exception as e:
            print(f"⚠️ Activity analysis error for track {track_id}: {e}")
    
    def get_track_activity(self, track_id):
        """Get current activity for a track"""
        if track_id in self.track_activities:
            return self.track_activities[track_id]
        return "Unknown"
    
    def get_track_confidence(self, track_id):
        """Get activity confidence for a track"""
        if track_id in self.activity_confidence:
            return self.activity_confidence[track_id]
        return 0.0
    
    def get_track_speed(self, track_id):
        """Get current speed for a track"""
        if track_id in self.track_speeds and self.track_speeds[track_id]:
            return self.track_speeds[track_id][-1]
        return 0.0
    
    def get_track_direction(self, track_id):
        """Get current direction for a track"""
        if track_id in self.track_directions and self.track_directions[track_id]:
            return self.track_directions[track_id][-1]
        return 0.0
    
    def get_track_positions(self, track_id):
        """Get position history for a track"""
        return self.track_positions.get(track_id, [])
    
    def cleanup_track(self, track_id):
        """Clean up tracking data for a removed track"""
        tracking_attrs = [
            self.track_positions, self.track_velocities, self.track_directions,
            self.track_speeds, self.track_activities, self.track_posture_history,
            self.activity_confidence, self.track_height_history,
            self.track_activity_votes, self.track_stable_activity, self.track_activity_duration
        ]
        
        for attr in tracking_attrs:
            if track_id in attr:
                del attr[track_id]
    
    def get_activity_summary(self):
        """Get summary of all current activities"""
        summary = {}
        for track_id in self.track_activities:
            activity = self.track_activities[track_id]
            confidence = self.activity_confidence.get(track_id, 0.0)
            speed = self.get_track_speed(track_id)
            
            summary[track_id] = {
                'activity': activity,
                'confidence': confidence,
                'speed': speed,
                'direction': self.get_track_direction(track_id)
            }
        
        return summary
    
    def get_statistics(self):
        """Get activity detection statistics"""
        activities = list(self.track_activities.values())
        activity_counts = {}
        
        for activity in activities:
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        total_tracks = len(activities)
        avg_speed = 0
        if self.track_speeds:
            all_speeds = []
            for speeds in self.track_speeds.values():
                if speeds:
                    all_speeds.extend(speeds[-5:])  # Recent speeds only
            avg_speed = np.mean(all_speeds) if all_speeds else 0
        
        return {
            'total_tracks': total_tracks,
            'activity_counts': activity_counts,
            'average_speed': avg_speed,
            'active_tracks': len(self.track_positions)
        }
    
    def get_detailed_track_info(self, track_id):
        """Get detailed information about a specific track for debugging"""
        if track_id not in self.track_activities:
            return None
        
        info = {
            'current_activity': self.track_activities.get(track_id, 'Unknown'),
            'confidence': self.activity_confidence.get(track_id, 0.0),
            'stable_activity': self.track_stable_activity.get(track_id, 'Unknown'),
            'activity_duration': self.track_activity_duration.get(track_id, 0),
            'votes': self.track_activity_votes.get(track_id, {}),
            'recent_speeds': self.track_speeds.get(track_id, [])[-5:],
            'recent_positions': self.track_positions.get(track_id, [])[-3:],
            'height_history': self.track_height_history.get(track_id, [])[-5:],
            'position_count': len(self.track_positions.get(track_id, [])),
        }
        
        return info