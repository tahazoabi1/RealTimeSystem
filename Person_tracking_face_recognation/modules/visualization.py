"""
Visualization and Drawing Module
Handles all visual elements: bounding boxes, trails, overlays, and UI
"""

import cv2
import numpy as np
import math
from datetime import datetime


class Visualizer:
    """Handles all visualization and drawing operations"""
    
    def __init__(self, trail_length=20):
        self.trail_length = trail_length
        self.colors = {}
        self.color_index = 0
        self.show_advanced_info = True
        
        # Pre-generate colors for performance
        self.generate_colors()
    
    def generate_colors(self):
        """Pre-generate colors for better performance"""
        self.base_colors = []
        for i in range(50):  # Generate 50 distinct colors
            hue = int((i * 137.508) % 180)  # Golden angle for good distribution
            color = np.uint8([[[hue, 255, 200]]])
            bgr_color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
            self.base_colors.append(tuple(map(int, bgr_color)))
    
    def get_track_color(self, track_id):
        """Get consistent color for a track"""
        if track_id not in self.colors:
            color_idx = len(self.colors) % len(self.base_colors)
            self.colors[track_id] = self.base_colors[color_idx]
        return self.colors[track_id]
    
    def draw_enhanced_bbox(self, frame, track_id, bbox, face_name, activity, confidence, speed):
        """Draw enhanced bounding box with all information"""
        try:
            # Validate inputs
            if frame is None or bbox is None or len(bbox) != 4:
                return
                
            # Validate frame properties
            if not hasattr(frame, 'shape') or len(frame.shape) != 3:
                return
                
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                return
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(0, min(w-1, x2))
            y2 = max(0, min(h-1, y2))
            
            # Skip if bbox is too small or invalid
            if x2 <= x1 or y2 <= y1:
                return
            
            color = self.get_track_color(track_id)
            
            # Draw bounding box
            thickness = 3 if confidence > 0.7 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare text information
            name_text = face_name if face_name and face_name != "Unknown" else f"Person {track_id}"
            activity_text = f"{activity}"
            speed_text = f"{speed:.1f} px/s" if speed > 1 else ""
            
            # Background for text
            text_lines = [name_text]
            if self.show_advanced_info:
                text_lines.append(activity_text)
                if speed_text:
                    text_lines.append(speed_text)
            
            # Calculate text background size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            line_height = 25
            max_width = 0
            
            for line in text_lines:
                (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, 2)
                max_width = max(max_width, text_width)
            
            # Draw text background
            bg_height = len(text_lines) * line_height + 10
            cv2.rectangle(frame, (x1, y1 - bg_height), (x1 + max_width + 10, y1), color, -1)
            cv2.rectangle(frame, (x1, y1 - bg_height), (x1 + max_width + 10, y1), (255, 255, 255), 1)
            
            # Draw text lines
            for i, line in enumerate(text_lines):
                text_y = y1 - bg_height + 20 + (i * line_height)
                cv2.putText(frame, line, (x1 + 5, text_y), font, font_scale, (255, 255, 255), 2)
            
            # Confidence indicator (colored bar on left side)
            if confidence > 0:
                bar_height = int((y2 - y1) * confidence)
                bar_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
                cv2.rectangle(frame, (x1 - 8, y2 - bar_height), (x1 - 3, y2), bar_color, -1)
            
        except Exception as e:
            print(f"⚠️ Draw bbox error for track {track_id}: {e}")
    
    def draw_motion_trail(self, frame, track_id, positions, color):
        """Draw motion trail with safe indexing"""
        try:
            if not positions or len(positions) < 2:
                return
            
            trail_length = min(len(positions), self.trail_length)
            
            # Draw only every other point for performance
            step = max(1, trail_length // 10)
            
            for i in range(step, trail_length, step):
                if i >= len(positions) or i + step >= len(positions):
                    break
                    
                # Ensure we don't access invalid indices
                prev_idx = len(positions) - i - step
                curr_idx = len(positions) - i
                
                if prev_idx < 0 or curr_idx < 0 or prev_idx >= len(positions) or curr_idx >= len(positions):
                    continue
                    
                thickness = max(1, int(2 * (i / trail_length)))
                alpha = max(0.3, i / trail_length)  # Ensure minimum visibility
                
                trail_color = tuple(int(c * alpha) for c in color)
                
                try:
                    cv2.line(frame, tuple(map(int, positions[prev_idx])), 
                            tuple(map(int, positions[curr_idx])), trail_color, thickness)
                except (IndexError, ValueError):
                    continue  # Skip invalid positions
                    
        except Exception as e:
            print(f"⚠️ Trail drawing error for track {track_id}: {e}")
    
    def draw_direction_arrow(self, frame, cx, cy, direction_deg, color, speed):
        """Draw direction arrow"""
        try:
            if speed < 2:  # Don't draw arrow for very slow movement
                return
            
            # Arrow length based on speed
            arrow_length = min(50, int(speed * 2))
            
            # Convert direction to radians
            direction_rad = math.radians(direction_deg)
            
            # Calculate arrow end point
            end_x = int(cx + arrow_length * math.cos(direction_rad))
            end_y = int(cy + arrow_length * math.sin(direction_rad))
            
            # Draw main arrow line
            cv2.arrowedLine(frame, (int(cx), int(cy)), (end_x, end_y), color, 3, tipLength=0.3)
            
        except Exception as e:
            print(f"⚠️ Arrow drawing error: {e}")
    
    def draw_zones(self, frame, zones):
        """Draw zone boundaries"""
        try:
            # Validate frame
            if frame is None or not hasattr(frame, 'shape') or len(frame.shape) != 3:
                return
                
            frame_h, frame_w = frame.shape[:2]
            if frame_h <= 0 or frame_w <= 0:
                return
            
            for zone in zones:
                if not isinstance(zone, dict) or 'bbox' not in zone:
                    continue
                    
                bbox = zone['bbox']
                color = tuple(zone.get('color', [255, 255, 255]))
                name = zone.get('name', 'Zone')
                
                if len(bbox) != 4:
                    continue
                
                x, y, w, h = map(int, bbox)
                
                # Ensure zone is within frame bounds
                x = max(0, min(frame_w-1, x))
                y = max(0, min(frame_h-1, y))
                w = max(1, min(frame_w - x, w))
                h = max(1, min(frame_h - y, h))
                
                # Draw zone rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw zone label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                (text_width, text_height), _ = cv2.getTextSize(name, font, font_scale, thickness)
                
                # Background for text
                cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width + 10, y), color, -1)
                cv2.putText(frame, name, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness)
                
        except Exception as e:
            print(f"⚠️ Zone drawing error: {e}")
    
    def add_advanced_overlay(self, frame, track_count, fps, session_stats=None):
        """Add comprehensive overlay information"""
        try:
            # Validate frame
            if frame is None or not hasattr(frame, 'shape') or len(frame.shape) != 3:
                return frame
                
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                return frame
            
            # Ensure frame is contiguous and writable
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            if not frame.flags['WRITEABLE']:
                frame = frame.copy()
                
            overlay = frame.copy()
            
            # System status panel
            panel_width = 300
            panel_height = 120
            panel_x = width - panel_width - 10
            panel_y = 10
            
            # Semi-transparent background
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # System information
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)
            thickness = 1
            
            info_lines = [
                f"FPS: {fps:.1f}",
                f"Active Tracks: {track_count}",
                f"Time: {datetime.now().strftime('%H:%M:%S')}",
            ]
            
            if session_stats:
                info_lines.append(f"Total Detections: {session_stats.get('total_detections', 0)}")
                if session_stats.get('average_speed'):
                    recent_avg = np.mean(session_stats['average_speed'][-10:]) if session_stats['average_speed'] else 0
                    info_lines.append(f"Avg Speed: {recent_avg:.1f} px/s")
            
            for i, line in enumerate(info_lines):
                y_pos = panel_y + 25 + (i * 20)
                cv2.putText(frame, line, (panel_x + 10, y_pos), font, font_scale, color, thickness)
            
            # Add activity legend if showing advanced info
            if self.show_advanced_info and track_count > 0:
                legend_y = panel_y + panel_height + 20
                legend_items = [
                    ("Walking", (0, 255, 0)),
                    ("Standing", (255, 255, 0)), 
                    ("Sitting", (255, 0, 0)),
                    ("Fidgeting", (255, 0, 255))
                ]
                
                for i, (activity, legend_color) in enumerate(legend_items):
                    item_x = panel_x + (i * 70)
                    cv2.circle(frame, (item_x + 10, legend_y + 10), 5, legend_color, -1)
                    cv2.putText(frame, activity, (item_x + 20, legend_y + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            print(f"⚠️ Overlay error: {e}")
            return frame
    
    def draw_fps_counter(self, frame, fps):
        """Draw FPS counter"""
        try:
            fps_text = f"FPS: {fps:.1f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
            
            # Background
            cv2.rectangle(frame, (10, 10), (text_width + 20, text_height + 20), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (text_width + 20, text_height + 20), (255, 255, 255), 2)
            
            # FPS color based on performance
            color = (0, 255, 0) if fps > 25 else (0, 255, 255) if fps > 15 else (0, 0, 255)
            
            cv2.putText(frame, fps_text, (15, text_height + 15), font, font_scale, color, thickness)
            
        except Exception as e:
            print(f"⚠️ FPS counter error: {e}")
    
    def save_screenshot(self, frame, prefix="enhanced_tracking"):
        """Save screenshot with timestamp"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{prefix}_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
            return filename
        except Exception as e:
            print(f"⚠️ Screenshot save error: {e}")
            return None
    
    def cleanup_track_visuals(self, track_id):
        """Clean up visual data for a removed track"""
        if track_id in self.colors:
            del self.colors[track_id]
    
    def toggle_advanced_info(self):
        """Toggle advanced information display"""
        self.show_advanced_info = not self.show_advanced_info
        return self.show_advanced_info
    
    def get_visual_stats(self):
        """Get visualization statistics"""
        return {
            'active_colors': len(self.colors),
            'trail_length': self.trail_length,
            'show_advanced_info': self.show_advanced_info,
            'total_base_colors': len(self.base_colors)
        }