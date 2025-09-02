#!/usr/bin/env python3
"""
Simple and Robust Exit Detection
Alternative approach using simpler, more reliable methods
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import time
import os
from collections import deque

class SimpleExitDetector:
    """Simple exit detector using distance-based gap analysis"""
    
    def __init__(self, detection_time=60.0, map_size_meters=10.0):
        self.detection_time = detection_time
        self.map_size_meters = map_size_meters
        
        # Data storage
        self.accumulated_points = []
        self.camera_positions = deque(maxlen=500)
        self.origin_offset = None
        
        # Simple exit tracking
        self.exit_candidates = []
        self.confirmed_exit = None
        self.detection_history = deque(maxlen=10)
        
        # Timing
        self.start_time = time.time()
        self.last_detection = time.time()
        
        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
    
    def setup_plot(self):
        """Setup simple visualization"""
        self.ax.set_xlim(-self.map_size_meters/2, self.map_size_meters/2)
        self.ax.set_ylim(-self.map_size_meters/2, self.map_size_meters/2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Z (meters)')
        self.ax.set_title('Simple Exit Detection')
    
    def read_map_points(self, filename='MapPoints.txt'):
        """Read map points with simple filtering"""
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            new_points = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        # Simple height filter
                        if -0.3 < y < 2.0:
                            new_points.append([x, z])
            
            if new_points:
                self.accumulated_points = new_points  # Replace, don't accumulate
                print(f"📊 Loaded {len(new_points)} points")
            
            return True
        except Exception as e:
            print(f"Error reading points: {e}")
            return False
    
    def read_camera_position(self, filename='CameraPosition.txt'):
        """Read camera position"""
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'r') as f:
                line = f.readline().strip()
            
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    
                    if self.origin_offset is None:
                        self.origin_offset = [x, z]
                        print(f"Origin set: ({x:.2f}, {z:.2f})")
                    
                    rel_pos = [x - self.origin_offset[0], z - self.origin_offset[1]]
                    self.camera_positions.append(rel_pos)
            
            return True
        except:
            return False
    
    def detect_exit_simple(self):
        """Simple exit detection using radial distance analysis"""
        if len(self.accumulated_points) < 50:
            return
        
        current_time = time.time() - self.start_time
        if (current_time - (self.last_detection - self.start_time)) < 2.0:  # Every 2 seconds
            return
        
        self.last_detection = time.time()
        
        # Center points on robot
        points = np.array(self.accumulated_points)
        if self.origin_offset:
            points[:, 0] -= self.origin_offset[0]
            points[:, 1] -= self.origin_offset[1]
        
        # Method 1: Radial Distance Analysis
        candidates = self._detect_by_radial_gaps(points)
        
        # Method 2: Point Density Analysis
        candidates.extend(self._detect_by_density_gaps(points))
        
        # Simple temporal consistency
        self.detection_history.append(candidates)
        
        # Find persistent candidates
        if len(self.detection_history) >= 3:
            persistent = self._find_persistent_candidates()
            if persistent and not self.confirmed_exit:
                self.confirmed_exit = persistent[0]
                print(f"\n🎯 SIMPLE EXIT DETECTED!")
                print(f"Position: ({self.confirmed_exit['center'][0]:.2f}, {self.confirmed_exit['center'][1]:.2f})")
                print(f"Direction: {np.degrees(self.confirmed_exit['direction']):.1f}°")
                print(f"Method: {self.confirmed_exit['method']}")
                print()
        
        self.exit_candidates = candidates
    
    def _detect_by_radial_gaps(self, points):
        """Detect gaps by analyzing radial distances"""
        candidates = []
        
        # Create radial sectors
        n_sectors = 36  # 10-degree sectors
        sector_angles = np.linspace(-np.pi, np.pi, n_sectors + 1)
        sector_centers = (sector_angles[:-1] + sector_angles[1:]) / 2
        
        # Find minimum distance in each sector
        min_distances = np.full(n_sectors, np.inf)
        
        for point in points:
            angle = np.arctan2(point[1], point[0])
            distance = np.linalg.norm(point)
            
            # Find which sector this point belongs to
            sector_idx = np.digitize(angle, sector_angles) - 1
            sector_idx = np.clip(sector_idx, 0, n_sectors - 1)
            
            if distance < min_distances[sector_idx]:
                min_distances[sector_idx] = distance
        
        # Find sectors with large distances (potential gaps)
        median_distance = np.median(min_distances[np.isfinite(min_distances)])
        gap_threshold = max(2.0, median_distance * 1.5)
        
        # Find contiguous gaps
        is_gap = min_distances > gap_threshold
        gap_start = None
        
        for i in range(n_sectors):
            if is_gap[i] and gap_start is None:
                gap_start = i
            elif not is_gap[i] and gap_start is not None:
                gap_end = i - 1
                gap_size = gap_end - gap_start + 1
                
                if gap_size >= 2:  # At least 20 degrees
                    center_idx = gap_start + gap_size // 2
                    center_angle = sector_centers[center_idx]
                    gap_distance = max(3.0, gap_threshold * 0.8)
                    
                    candidates.append({
                        'center': [gap_distance * np.cos(center_angle), 
                                 gap_distance * np.sin(center_angle)],
                        'direction': center_angle,
                        'gap_size': gap_size * 10.0,  # degrees to rough width
                        'confidence': min(1.0, gap_size / 6.0),
                        'method': 'radial_gap'
                    })
                
                gap_start = None
        
        return candidates
    
    def _detect_by_density_gaps(self, points):
        """Detect gaps by analyzing point density"""
        candidates = []
        
        if len(points) < 20:
            return candidates
        
        # Divide space into angular sectors and measure point density
        n_sectors = 24  # 15-degree sectors
        sector_points = [[] for _ in range(n_sectors)]
        
        for point in points:
            angle = np.arctan2(point[1], point[0])
            distance = np.linalg.norm(point)
            
            if distance < 6.0:  # Only consider nearby points
                sector_idx = int((angle + np.pi) / (2 * np.pi) * n_sectors)
                sector_idx = sector_idx % n_sectors
                sector_points[sector_idx].append(distance)
        
        # Calculate density scores (points per sector)
        densities = [len(sector) for sector in sector_points]
        median_density = np.median(densities) if densities else 0
        
        # Find low-density sectors (potential exits)
        low_density_threshold = max(1, median_density * 0.3)
        
        for i, density in enumerate(densities):
            if density <= low_density_threshold:
                angle = (i / n_sectors) * 2 * np.pi - np.pi
                distance = 3.0  # Default exit distance
                
                candidates.append({
                    'center': [distance * np.cos(angle), distance * np.sin(angle)],
                    'direction': angle,
                    'gap_size': 15.0,  # 15 degrees
                    'confidence': 1.0 - (density / max(1, median_density)),
                    'method': 'density_gap'
                })
        
        return candidates
    
    def _find_persistent_candidates(self):
        """Find candidates that appear consistently"""
        if len(self.detection_history) < 3:
            return []
        
        # Simple approach: count candidates in similar directions
        direction_votes = {}
        
        for frame_candidates in self.detection_history:
            for candidate in frame_candidates:
                direction = candidate['direction']
                
                # Quantize direction to 30-degree bins
                dir_bin = int((direction + np.pi) / (np.pi/6)) * (np.pi/6) - np.pi
                
                if dir_bin not in direction_votes:
                    direction_votes[dir_bin] = []
                direction_votes[dir_bin].append(candidate)
        
        # Find most voted direction
        persistent = []
        for direction, candidates in direction_votes.items():
            if len(candidates) >= 2:  # Appeared in at least 2 frames
                # Average the candidates
                avg_confidence = np.mean([c['confidence'] for c in candidates])
                avg_distance = np.mean([np.linalg.norm(c['center']) for c in candidates])
                
                persistent.append({
                    'center': [avg_distance * np.cos(direction), 
                             avg_distance * np.sin(direction)],
                    'direction': direction,
                    'gap_size': np.mean([c['gap_size'] for c in candidates]),
                    'confidence': avg_confidence,
                    'method': 'persistent',
                    'votes': len(candidates)
                })
        
        return sorted(persistent, key=lambda x: x['confidence'] * x['votes'], reverse=True)
    
    def update_plot(self, frame):
        """Update visualization"""
        self.ax.clear()
        self.setup_plot()
        
        # Read data
        self.read_map_points()
        self.read_camera_position()
        
        # Run detection
        self.detect_exit_simple()
        
        # Plot points
        if self.accumulated_points:
            points = np.array(self.accumulated_points)
            if self.origin_offset:
                points[:, 0] -= self.origin_offset[0]
                points[:, 1] -= self.origin_offset[1]
            self.ax.scatter(points[:, 0], points[:, 1], c='blue', s=2, alpha=0.5)
        
        # Plot robot path
        if len(self.camera_positions) > 1:
            path = np.array(self.camera_positions)
            self.ax.plot(path[:, 0], path[:, 1], 'g-', linewidth=2, alpha=0.7)
        
        # Plot robot
        self.ax.scatter(0, 0, c='red', s=200, marker='o', edgecolors='black', linewidth=2)
        self.ax.text(0.2, 0.2, 'YOU', fontsize=12, weight='bold')
        
        # Plot current candidates
        for i, candidate in enumerate(self.exit_candidates[:3]):
            color = ['orange', 'yellow', 'cyan'][i]
            alpha = 0.7 - i * 0.2
            
            self.ax.scatter(candidate['center'][0], candidate['center'][1], 
                          c=color, s=100, alpha=alpha, edgecolors='black')
            
            # Direction line
            dx = 1.5 * np.cos(candidate['direction'])
            dy = 1.5 * np.sin(candidate['direction'])
            self.ax.plot([0, dx], [0, dy], color=color, alpha=alpha, linewidth=2)
        
        # Plot confirmed exit
        if self.confirmed_exit:
            self.ax.add_patch(Circle(self.confirmed_exit['center'], 0.5, 
                                   color='red', fill=False, linewidth=3))
            
            self.ax.annotate('EXIT', self.confirmed_exit['center'], 
                           xytext=(0, 0), fontsize=16, weight='bold', color='red',
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Title
        elapsed = time.time() - self.start_time
        n_points = len(self.accumulated_points)
        n_candidates = len(self.exit_candidates)
        
        status = "DETECTED" if self.confirmed_exit else f"{n_candidates} candidates"
        self.ax.set_title(f'Simple Exit Detection | Points: {n_points} | Status: {status} | Time: {elapsed:.1f}s')
    
    def run(self):
        """Run the simple exit detector"""
        print("🚀 Simple Exit Detector Starting...")
        print("Using radial gap + density analysis")
        print("Press Ctrl+C to stop\n")
        
        try:
            ani = FuncAnimation(self.fig, self.update_plot, interval=500, blit=False)
            plt.show()
        except KeyboardInterrupt:
            print("\n🛑 Simple detector stopped.")

if __name__ == "__main__":
    detector = SimpleExitDetector(detection_time=60.0, map_size_meters=10.0)
    detector.run()