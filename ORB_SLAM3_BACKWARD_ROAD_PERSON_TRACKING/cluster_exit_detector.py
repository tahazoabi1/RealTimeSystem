#!/usr/bin/env python3
"""
Cluster-Based Exit Detection
Uses point clustering to identify walls and find gaps between them
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import time
import os
from collections import deque
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

class ClusterExitDetector:
    """Exit detector using clustering and wall detection"""
    
    def __init__(self, detection_time=60.0, map_size_meters=10.0):
        self.detection_time = detection_time
        self.map_size_meters = map_size_meters
        
        # Data storage
        self.accumulated_points = []
        self.camera_positions = deque(maxlen=500)
        self.origin_offset = None
        
        # Wall and exit detection
        self.detected_walls = []
        self.detected_exits = []
        self.confirmed_exit = None
        
        # Timing
        self.start_time = time.time()
        self.last_detection = time.time()
        
        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.setup_plot()
    
    def setup_plot(self):
        """Setup visualization"""
        self.ax.set_xlim(-self.map_size_meters/2, self.map_size_meters/2)
        self.ax.set_ylim(-self.map_size_meters/2, self.map_size_meters/2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Z (meters)')
        self.ax.set_title('Cluster-Based Exit Detection')
    
    def read_map_points(self, filename='MapPoints.txt'):
        """Read map points"""
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
                        if -0.3 < y < 2.0:
                            new_points.append([x, z])
            
            if new_points:
                # Take only recent points to avoid memory issues
                self.accumulated_points = new_points[-2000:]  # Keep last 2000 points
                
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
                    
                    rel_pos = [x - self.origin_offset[0], z - self.origin_offset[1]]
                    self.camera_positions.append(rel_pos)
            
            return True
        except:
            return False
    
    def detect_exits_by_clustering(self):
        """Detect exits using point clustering and wall detection"""
        if len(self.accumulated_points) < 100:
            return
        
        current_time = time.time() - self.start_time
        if (current_time - (self.last_detection - self.start_time)) < 3.0:
            return
        
        self.last_detection = time.time()
        
        # Center points
        points = np.array(self.accumulated_points)
        if self.origin_offset:
            points[:, 0] -= self.origin_offset[0]
            points[:, 1] -= self.origin_offset[1]
        
        # Filter points by distance (focus on nearby environment)
        distances = np.linalg.norm(points, axis=1)
        nearby_points = points[distances < 6.0]
        
        if len(nearby_points) < 50:
            return
        
        print(f"🔍 Analyzing {len(nearby_points)} nearby points...")
        
        # Method 1: Cluster points to identify wall segments
        walls = self._detect_walls_by_clustering(nearby_points)
        self.detected_walls = walls
        
        # Method 2: Find gaps between walls
        exits = self._find_gaps_between_walls(walls, nearby_points)
        
        # Method 3: Radial clustering approach
        radial_exits = self._detect_exits_by_radial_clustering(nearby_points)
        
        # Combine results
        all_exits = exits + radial_exits
        self.detected_exits = all_exits
        
        # Select best exit
        if all_exits and not self.confirmed_exit:
            # Score exits by distance and clearance
            scored_exits = []
            for exit in all_exits:
                distance = np.linalg.norm(exit['center'])
                clearance_score = self._calculate_clearance_score(exit['center'], nearby_points)
                score = clearance_score / max(0.5, distance / 4.0)  # Prefer closer exits with good clearance
                
                scored_exits.append({**exit, 'score': score})
            
            best_exit = max(scored_exits, key=lambda x: x['score'])
            if best_exit['score'] > 0.3:
                self.confirmed_exit = best_exit
                print(f"\n🎯 CLUSTER EXIT DETECTED!")
                print(f"Position: ({best_exit['center'][0]:.2f}, {best_exit['center'][1]:.2f})")
                print(f"Direction: {np.degrees(best_exit['direction']):.1f}°")
                print(f"Score: {best_exit['score']:.3f}")
                print(f"Method: {best_exit['method']}")
    
    def _detect_walls_by_clustering(self, points):
        """Detect wall segments using DBSCAN clustering"""
        if len(points) < 20:
            return []
        
        # Cluster points spatially
        clustering = DBSCAN(eps=0.3, min_samples=10).fit(points)
        
        walls = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_points = points[clustering.labels_ == cluster_id]
            if len(cluster_points) < 15:
                continue
            
            # Try to fit a line to this cluster using RANSAC
            try:
                ransac = RANSACRegressor(residual_threshold=0.1, max_trials=100)
                X = cluster_points[:, 0].reshape(-1, 1)
                y = cluster_points[:, 1]
                
                ransac.fit(X, y)
                
                # Get line parameters
                slope = ransac.estimator_.coef_[0]
                intercept = ransac.estimator_.intercept_
                
                # Calculate line endpoints within cluster bounds
                x_min, x_max = X.min(), X.max()
                y_min = slope * x_min + intercept
                y_max = slope * x_max + intercept
                
                wall = {
                    'start': [x_min, y_min],
                    'end': [x_max, y_max],
                    'slope': slope,
                    'intercept': intercept,
                    'points': cluster_points,
                    'length': np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
                }
                
                if wall['length'] > 0.5:  # Only keep walls longer than 50cm
                    walls.append(wall)
                    
            except Exception as e:
                continue
        
        return walls
    
    def _find_gaps_between_walls(self, walls, points):
        """Find gaps between detected wall segments"""
        exits = []
        
        if len(walls) < 2:
            return exits
        
        # Check gaps between each pair of walls
        for i, wall1 in enumerate(walls):
            for j, wall2 in enumerate(walls[i+1:], i+1):
                gap_info = self._calculate_wall_gap(wall1, wall2, points)
                if gap_info and gap_info['width'] > 0.7:  # At least 70cm gap
                    exits.append({
                        'center': gap_info['center'],
                        'direction': np.arctan2(gap_info['center'][1], gap_info['center'][0]),
                        'gap_size': gap_info['width'],
                        'confidence': min(1.0, gap_info['width'] / 2.0),
                        'method': 'wall_gap'
                    })
        
        return exits
    
    def _calculate_wall_gap(self, wall1, wall2, points):
        """Calculate gap between two walls"""
        # Find closest points between walls
        w1_points = wall1['points']
        w2_points = wall2['points']
        
        min_distance = float('inf')
        closest_pair = None
        
        for p1 in w1_points:
            for p2 in w2_points:
                dist = np.linalg.norm(p1 - p2)
                if dist < min_distance:
                    min_distance = dist
                    closest_pair = (p1, p2)
        
        if closest_pair is None or min_distance > 4.0:
            return None
        
        # Gap center and width
        gap_center = (closest_pair[0] + closest_pair[1]) / 2
        gap_width = min_distance
        
        # Check if gap is clear (no points in between)
        gap_clearance = self._check_gap_clearance(closest_pair[0], closest_pair[1], points)
        
        if gap_clearance > 0.7:  # At least 70% clear
            return {
                'center': gap_center,
                'width': gap_width,
                'clearance': gap_clearance
            }
        
        return None
    
    def _check_gap_clearance(self, point1, point2, points):
        """Check how clear the gap between two points is"""
        gap_vector = point2 - point1
        gap_length = np.linalg.norm(gap_vector)
        
        if gap_length < 0.1:
            return 0.0
        
        gap_direction = gap_vector / gap_length
        
        # Check points along the gap line
        blocking_points = 0
        total_checks = int(gap_length / 0.1)  # Check every 10cm
        
        for i in range(1, total_checks):
            check_point = point1 + (i / total_checks) * gap_vector
            
            # Count nearby points
            distances = np.linalg.norm(points - check_point, axis=1)
            nearby_count = np.sum(distances < 0.2)  # Within 20cm
            
            if nearby_count > 0:
                blocking_points += 1
        
        if total_checks == 0:
            return 1.0
        
        clearance = 1.0 - (blocking_points / total_checks)
        return clearance
    
    def _detect_exits_by_radial_clustering(self, points):
        """Detect exits using radial sector clustering"""
        exits = []
        
        # Divide space into radial sectors
        n_sectors = 16  # 22.5-degree sectors
        sector_points = [[] for _ in range(n_sectors)]
        
        # Distribute points into sectors
        for point in points:
            angle = np.arctan2(point[1], point[0])
            distance = np.linalg.norm(point)
            
            if 1.0 < distance < 5.0:  # Focus on mid-range points
                sector_idx = int((angle + np.pi) / (2 * np.pi) * n_sectors) % n_sectors
                sector_points[sector_idx].append(point)
        
        # Find sectors with very few points (potential exits)
        sector_counts = [len(sector) for sector in sector_points]
        median_count = np.median(sector_counts) if sector_counts else 0
        
        for i, count in enumerate(sector_counts):
            if count < max(3, median_count * 0.3):  # Much fewer points than median
                sector_angle = (i / n_sectors) * 2 * np.pi - np.pi
                exit_distance = 3.0
                
                exits.append({
                    'center': [exit_distance * np.cos(sector_angle), 
                             exit_distance * np.sin(sector_angle)],
                    'direction': sector_angle,
                    'gap_size': 22.5,  # degrees
                    'confidence': 1.0 - (count / max(1, median_count)),
                    'method': 'radial_cluster'
                })
        
        return exits
    
    def _calculate_clearance_score(self, position, points):
        """Calculate how clear the area around a position is"""
        distances = np.linalg.norm(points - position, axis=1)
        
        # Count points at different radii
        close_points = np.sum(distances < 0.5)    # Within 50cm
        medium_points = np.sum(distances < 1.0)   # Within 1m
        far_points = np.sum(distances < 2.0)      # Within 2m
        
        # Good exits have few close points but some far points
        if far_points == 0:
            return 0.1
        
        clearance = (1.0 / (1.0 + close_points)) * min(1.0, medium_points / 10.0)
        return clearance
    
    def update_plot(self, frame):
        """Update visualization"""
        self.ax.clear()
        self.setup_plot()
        
        # Read data
        self.read_map_points()
        self.read_camera_position()
        
        # Run detection
        self.detect_exits_by_clustering()
        
        # Plot points
        if self.accumulated_points:
            points = np.array(self.accumulated_points)
            if self.origin_offset:
                points[:, 0] -= self.origin_offset[0]
                points[:, 1] -= self.origin_offset[1]
            
            # Color-code by distance
            distances = np.linalg.norm(points, axis=1)
            colors = plt.cm.viridis(distances / np.max(distances))
            self.ax.scatter(points[:, 0], points[:, 1], c=colors, s=3, alpha=0.6)
        
        # Plot walls
        for wall in self.detected_walls:
            start, end = wall['start'], wall['end']
            self.ax.plot([start[0], end[0]], [start[1], end[1]], 
                        'r-', linewidth=3, alpha=0.8, label='Detected Wall' if len(self.detected_walls) < 2 else '')
        
        # Plot robot path
        if len(self.camera_positions) > 1:
            path = np.array(self.camera_positions)
            self.ax.plot(path[:, 0], path[:, 1], 'g-', linewidth=2, alpha=0.7)
        
        # Plot robot
        self.ax.scatter(0, 0, c='red', s=200, marker='o', edgecolors='black', linewidth=2)
        self.ax.text(0.3, 0.3, 'YOU', fontsize=12, weight='bold', color='red')
        
        # Plot exit candidates
        for i, exit in enumerate(self.detected_exits[:5]):
            color = ['orange', 'yellow', 'cyan', 'magenta', 'lime'][i]
            alpha = 0.8 - i * 0.15
            
            # Draw exit marker
            self.ax.scatter(exit['center'][0], exit['center'][1], 
                          c=color, s=80, alpha=alpha, edgecolors='black', 
                          marker='s')  # Square marker
            
            # Direction arrow
            dx = 1.5 * np.cos(exit['direction'])
            dy = 1.5 * np.sin(exit['direction'])
            self.ax.arrow(0, 0, dx, dy, head_width=0.15, head_length=0.2,
                         fc=color, ec=color, alpha=alpha)
        
        # Plot confirmed exit
        if self.confirmed_exit:
            self.ax.add_patch(Circle(self.confirmed_exit['center'], 0.6, 
                                   color='red', fill=False, linewidth=4))
            
            self.ax.annotate('CONFIRMED EXIT', self.confirmed_exit['center'], 
                           xytext=(0, 0), fontsize=14, weight='bold', color='red',
                           arrowprops=dict(arrowstyle='->', color='red', lw=3))
            
            # Score text
            score_text = f"Score: {self.confirmed_exit.get('score', 0):.2f}"
            self.ax.text(self.confirmed_exit['center'][0], 
                        self.confirmed_exit['center'][1] - 0.8,
                        score_text, ha='center', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Title and legend
        elapsed = time.time() - self.start_time
        n_points = len(self.accumulated_points)
        n_walls = len(self.detected_walls)
        n_exits = len(self.detected_exits)
        
        status = "CONFIRMED" if self.confirmed_exit else f"{n_exits} candidates"
        self.ax.set_title(f'Cluster Exit Detection | Points: {n_points} | Walls: {n_walls} | Exits: {status} | Time: {elapsed:.1f}s')
        
        if self.detected_walls and len(self.detected_walls) < 6:  # Don't clutter legend
            self.ax.legend()
    
    def run(self):
        """Run the cluster-based exit detector"""
        print("🚀 Cluster-Based Exit Detector Starting...")
        print("Using wall detection + gap analysis + radial clustering")
        print("Press Ctrl+C to stop\n")
        
        try:
            ani = FuncAnimation(self.fig, self.update_plot, interval=1000, blit=False)
            plt.show()
        except KeyboardInterrupt:
            print("\n🛑 Cluster detector stopped.")

if __name__ == "__main__":
    detector = ClusterExitDetector(detection_time=60.0, map_size_meters=8.0)
    detector.run()