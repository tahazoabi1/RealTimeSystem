#!/usr/bin/env python3
import numpy as np
import cv2
import time
from collections import deque
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

def _circular_smooth(x, win=5):
    if win <= 1: return x
    pad = win // 2
    xp = np.r_[x[-pad:], x, x[:pad]]
    kernel = np.ones(win) / win
    sm = np.convolve(xp, kernel, mode='same')[pad:-pad]
    return sm

def _contiguous_runs(mask):
    # returns (start_idx, end_idx) inclusive for each run on a circular array
    n = len(mask)
    runs = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while (j+1) % n != i and mask[(j+1) % n]:
                j = (j+1) % n
                if j == i: break
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    # merge wraparound run if needed
    if runs and runs[0][0] == 0 and runs[-1][1] == n-1:
        runs[0] = (runs[-1][0], runs[0][1])
        runs.pop()
    return runs

def angle_wrap(a):
    # wrap to [-pi, pi)
    return (a + np.pi) % (2*np.pi) - np.pi

def meters_at_range(width_angle_rad, r):
    return abs(width_angle_rad) * r

class ExitDetectorVisualizer:
    def __init__(self, detection_time=60.0, map_size_meters=10.0):
        self.detection_time = detection_time  # Time before exit detection (seconds)
        self.map_size_meters = map_size_meters
        self.fig_size = 10  # Figure size in inches
        
        # Point accumulation
        self.accumulated_points = []
        self.camera_positions = deque(maxlen=1000)  # Keep track of robot path
        self.current_camera_pos = None
        self.origin_offset = None
        
        # Timing
        self.start_time = time.time()
        self.exit_detected = False
        self.exit_location = None
        self.exit_direction = None
        
        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(self.fig_size, self.fig_size))
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize the plot with proper settings"""
        self.ax.set_xlim(-self.map_size_meters/2, self.map_size_meters/2)
        self.ax.set_ylim(-self.map_size_meters/2, self.map_size_meters/2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Z (meters)')
        self.ax.set_title('Exit Detection Map - Waiting for Detection...')
        
        # Create legend elements
        robot_patch = mpatches.Patch(color='red', label='Robot (You)')
        points_patch = mpatches.Patch(color='blue', label='Map Points')
        path_patch = mpatches.Patch(color='green', label='Robot Path')
        self.ax.legend(handles=[robot_patch, points_patch, path_patch], loc='upper right')
        
    def read_map_points(self, filename='MapPoints.txt'):
        """Read map points from file"""
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
                        # Filter by height (only keep points at reasonable height)
                        if -0.2 < y < 2.0:
                            new_points.append([x, z])  # Use x,z for 2D map
                            
            if new_points:
                self.accumulated_points.extend(new_points)
                # Remove duplicates using voxel grid
                self.accumulated_points = self.remove_duplicates(self.accumulated_points)
                
            return True
        except:
            return False
            
    def read_camera_position(self, filename='CameraPosition.txt'):
        """Read camera position from file"""
        if not os.path.exists(filename):
            return False
            
        try:
            with open(filename, 'r') as f:
                line = f.readline().strip()
                
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    self.current_camera_pos = [x, z]  # Use x,z for 2D
                    
                    # Set origin on first reading
                    if self.origin_offset is None:
                        self.origin_offset = [x, z]
                        
                    # Store camera path
                    self.camera_positions.append([x - self.origin_offset[0], 
                                                  z - self.origin_offset[1]])
                    
            return True
        except:
            return False
            
    def remove_duplicates(self, points, tolerance=0.05):
        """Remove duplicate points within tolerance"""
        if len(points) < 2:
            return points
            
        points_array = np.array(points)
        # Quantize to grid
        quantized = np.round(points_array / tolerance) * tolerance
        # Get unique points
        unique_points = np.unique(quantized, axis=0)
        return unique_points.tolist()
        
    def detect_exit(self):
        """Detect exit location after specified time using improved algorithms"""
        current_time = time.time() - self.start_time
        
        if current_time >= self.detection_time and not self.exit_detected and len(self.accumulated_points) > 50:
            print(f"\n{'='*50}")
            print(f"EXIT DETECTION TRIGGERED at {current_time:.1f} seconds!")
            print(f"Analyzing {len(self.accumulated_points)} map points...")
            
            # Convert points to numpy array centered at origin
            pts = np.array(self.accumulated_points)
            if self.origin_offset is not None:
                pts[:,0] -= self.origin_offset[0]
                pts[:,1] -= self.origin_offset[1]

            # candidates from both methods
            c1 = self.find_doors_by_edge_pairing(pts)
            c2 = []
            try:
                f = self.find_frontier_exit(pts)
                if f is not None:
                    c2 = [f]
            except Exception as e:
                print(f"Warning: Frontier detection failed: {e}")
                # Continue with edge-pairing results only

            all_cands = (c1 or []) + c2
            
            if all_cands:
                # simple fusion: prefer consistent/frontier-supported directions
                all_cands.sort(key=lambda c: c['score'], reverse=True)
                best = all_cands[0]
                self.exit_location  = best['center']
                self.exit_direction = best['direction']
                self.exit_detected  = True
                
                print(f"EXIT DETECTED at position: ({self.exit_location[0]:.2f}, {self.exit_location[1]:.2f})")
                print(f"Direction from center: {np.degrees(self.exit_direction):.1f}°")
                print(f"Gap size: {best['gap_size']:.2f} meters")
                print(f"Type: {best['kind'].upper()}")
            else:
                print("No clear exit found - map may be incomplete")
                
            print(f"{'='*50}\n")
            
    def find_map_gaps(self, points):
        """Find gaps in the point cloud that could be exits"""
        if len(points) < 10:
            return []
            
        # Use angular histogram to find gaps
        angles = np.arctan2(points[:, 1], points[:, 0])
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # Create angular bins
        n_bins = 72  # 5-degree bins
        angle_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        # Find the closest point in each angular bin
        closest_distances = np.full(n_bins, np.inf)
        for i in range(len(angles)):
            bin_idx = np.digitize(angles[i], angle_bins) - 1
            if 0 <= bin_idx < n_bins:
                closest_distances[bin_idx] = min(closest_distances[bin_idx], distances[i])
        
        # Find gaps (bins with no close points)
        gap_threshold = 2.0  # Points further than this are considered gaps
        gaps = []
        
        for i in range(n_bins):
            if closest_distances[i] > gap_threshold or np.isinf(closest_distances[i]):
                # Found a gap, find its extent
                gap_start = i
                gap_end = i
                
                # Extend gap
                while (gap_end + 1) % n_bins != gap_start and \
                      (closest_distances[(gap_end + 1) % n_bins] > gap_threshold or \
                       np.isinf(closest_distances[(gap_end + 1) % n_bins])):
                    gap_end = (gap_end + 1) % n_bins
                
                # Calculate gap properties
                if gap_end >= gap_start:
                    gap_angles = angle_bins[gap_start:gap_end+1]
                else:
                    gap_angles = np.concatenate([angle_bins[gap_start:], angle_bins[:gap_end+1]])
                
                gap_center_angle = np.mean(gap_angles)
                gap_size = len(gap_angles) * (2 * np.pi / n_bins)
                
                # Only consider significant gaps
                if gap_size > np.radians(30):  # At least 30 degrees
                    gap_distance = 3.0  # Place exit marker at 3 meters
                    gaps.append({
                        'center': [gap_distance * np.cos(gap_center_angle),
                                  gap_distance * np.sin(gap_center_angle)],
                        'direction': gap_center_angle,
                        'gap_size': gap_size * gap_distance  # Approximate gap width
                    })
                
                # Skip processed bins
                i = gap_end
                
        return gaps
    
    def find_doors_by_edge_pairing(self, points, n_bins=180, r_free=3.0,
                                   max_considered_range=8.0,
                                   min_door_w=0.8, max_door_w=3.0,
                                   min_exit_w=2.0):
        """
        Returns candidates sorted by score.
        Each candidate: {'center':[x,y], 'direction':theta, 'gap_size':width_m, 'kind': 'door'|'exit'}
        """
        if len(points) < 10:
            return []

        # polar
        angles = np.arctan2(points[:,1], points[:,0])  # [-pi, pi)
        dists  = np.sqrt(points[:,0]**2 + points[:,1]**2)

        # bin angles -> per-bin minimum distance
        edges = np.linspace(-np.pi, np.pi, n_bins+1)
        idx   = np.clip(np.digitize(angles, edges)-1, 0, n_bins-1)
        per_bin = np.full(n_bins, np.inf)
        for i, r in zip(idx, dists):
            if r < per_bin[i]:
                per_bin[i] = r

        # smooth + robust thresholds
        per_bin_sm = _circular_smooth(per_bin, win=7)
        finite = np.isfinite(per_bin_sm)
        if finite.sum() < 5:
            return []

        # classify bin as "free far" if beyond r_free (or unknown/inf)
        free_far = (per_bin_sm > r_free) | (~np.isfinite(per_bin_sm))

        # find contiguous free runs (potential openings)
        runs = _contiguous_runs(free_far)

        # pre-compute bin centers & bin width
        bin_centers = (edges[:-1] + edges[1:]) * 0.5
        bin_width   = (2*np.pi) / n_bins

        candidates = []
        for a0, a1 in runs:
            # angular span
            if a1 >= a0:
                span = a1 - a0 + 1
                span_angles = bin_centers[a0:a1+1]
                # choose center index
                c_idx = a0 + span//2
            else:
                # wrapped run
                span = (n_bins - a0) + (a1 + 1)
                span_angles = np.r_[bin_centers[a0:], bin_centers[:a1+1]]
                c_idx = (a0 + span//2) % n_bins

            width_angle = span * bin_width
            # estimate usable range for the opening: use the smaller of
            # - the median finite distance on the run
            # - max_considered_range
            run_vals = per_bin_sm[a0:a1+1] if a1>=a0 else np.r_[per_bin_sm[a0:], per_bin_sm[:a1+1]]
            finite_run = run_vals[np.isfinite(run_vals)]
            if len(finite_run) == 0:
                r_use = max_considered_range
            else:
                r_use = min(np.median(finite_run), max_considered_range)

            width_m = meters_at_range(width_angle, r_use)

            # Heuristic classification + score
            if width_m >= min_exit_w:
                kind = 'exit'
            elif min_door_w <= width_m <= max_door_w:
                kind = 'door'
            else:
                continue  # too narrow/wide to be useful

            center_theta = bin_centers[c_idx]
            center = [r_use * np.cos(center_theta), r_use * np.sin(center_theta)]

            # score: prefer wide + far + sharp edges
            score = width_m * (1.0 + 0.3 * (r_use / max_considered_range))

            candidates.append({
                'center': center,
                'direction': angle_wrap(center_theta),
                'gap_size': width_m,
                'kind': kind,
                'score': score
            })

        # sort best first
        candidates.sort(key=lambda c: c['score'], reverse=True)
        return candidates
    
    def _to_grid(self, xy, res, size_m):
        """xy: (N,2) in meters around (0,0) -> integer grid indices (col,row)"""
        half = size_m / 2.0
        max_idx = min(int(size_m/res)-1, 199)  # Cap to avoid memory issues
        gx = np.clip(((xy[:,0] + half) / res).astype(int), 0, max_idx)
        gy = np.clip(((half - xy[:,1]) / res).astype(int), 0, max_idx)  # y up->row down
        return gx, gy

    def find_frontier_exit(self, points, res=0.1, inflate_occ=2, path_dilate=3,
                           min_frontier_len=30, min_clearance=0.4):
        """
        Returns one candidate or None. Uses:
        - occupied from points (dilated)
        - free from dilated camera path
        - frontiers = unknown & adjacent_to_free
        """
        size_m = self.map_size_meters
        N = int(size_m / res)
        if N > 200: N = 200  # Cap grid size to avoid memory issues
        if N < 50: N = 50
        grid_unknown = 127
        grid_occ = 255
        grid_free = 0

        grid = np.full((N, N), grid_unknown, dtype=np.uint8)

        # 1) Occupied from points
        if len(points) > 0:
            gx, gy = self._to_grid(points, res, size_m)
            grid[gy, gx] = grid_occ
            if inflate_occ > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_occ*2+1, inflate_occ*2+1))
                grid = np.where(cv2.dilate((grid==grid_occ).astype(np.uint8), k), grid_occ, grid)

        # 2) Free from camera path
        free = np.zeros_like(grid, dtype=np.uint8)
        if len(self.camera_positions) > 0:
            path = np.array(self.camera_positions)  # already centered
            pgx, pgy = self._to_grid(path, res, size_m)
            for x, y in zip(pgx, pgy):
                cv2.circle(free, (x, y), path_dilate, 1, -1)
        free_mask = (free > 0).astype(np.uint8)

        # mark free cells where not occupied
        grid[(free_mask == 1) & (grid != grid_occ)] = grid_free

        # 3) Frontiers: unknown cells that neighbor free
        neigh = cv2.dilate((grid == grid_free).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        frontier = ((grid == grid_unknown) & (neigh == 1)).astype(np.uint8)

        # 4) Filter frontiers near obstacles (need clearance)
        # distance from occupied
        dist_to_occ = cv2.distanceTransform((grid != grid_occ).astype(np.uint8), cv2.DIST_L2, 3)
        pix_clearance = int(max(1, min_clearance / res))
        frontier[dist_to_occ < pix_clearance] = 0

        # 5) Connected components -> pick the biggest "arc" in robot-facing half-plane
        num, labels = cv2.connectedComponents(frontier)
        if num <= 1:
            return None

        best = None
        center_pix = np.array([N//2, N//2])  # col,row (x,y in grid terms)
        for cid in range(1, num):
            mask = (labels == cid)
            count = int(mask.sum())
            if count < min_frontier_len:
                continue

            ys, xs = np.where(mask)
            cx = xs.mean()
            cy = ys.mean()

            # direction from robot (center) to frontier centroid
            v = np.array([cx - center_pix[0], center_pix[1] - cy])  # convert to metric later
            theta = np.arctan2(v[1], v[0])
            r = np.linalg.norm(v) * res

            # score: longer frontier & farther is better
            score = count + 50.0 * r

            if (best is None) or (score > best['score']):
                best = {
                    'center_pix': (cx, cy),
                    'direction': theta,
                    'range': r,
                    'score': score
                }

        if best is None:
            return None

        # convert center to meters
        cx, cy = best['center_pix']
        x_m = (cx - (N/2.0)) * res
        y_m = ((N/2.0) - cy) * res

        return {
            'center': [x_m, y_m],
            'direction': best['direction'],
            'gap_size': 1.5,   # unknown; frontier doesn't give width; set nominal
            'kind': 'frontier',
            'score': best['score']
        }
        
    def update_plot(self, frame):
        """Update the plot animation"""
        self.ax.clear()
        self.setup_plot()
        
        # Read latest data
        self.read_map_points()
        self.read_camera_position()
        
        # Check for exit detection
        if not self.exit_detected:
            self.detect_exit()
        
        # Plot accumulated points
        if self.accumulated_points:
            points = np.array(self.accumulated_points)
            if self.origin_offset:
                points[:, 0] -= self.origin_offset[0]
                points[:, 1] -= self.origin_offset[1]
            self.ax.scatter(points[:, 0], points[:, 1], c='blue', s=1, alpha=0.5)
        
        # Plot robot path
        if len(self.camera_positions) > 1:
            path = np.array(self.camera_positions)
            self.ax.plot(path[:, 0], path[:, 1], 'g-', linewidth=2, alpha=0.7, label='Robot Path')
        
        # Plot current robot position (always at origin in centered view)
        self.ax.scatter(0, 0, c='red', s=200, marker='o', edgecolors='black', linewidth=2)
        self.ax.annotate('YOU', (0, 0), xytext=(0.3, 0.3), fontsize=12, weight='bold')
        
        # Update title with timer
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, self.detection_time - elapsed_time)
        if not self.exit_detected:
            self.ax.set_title(f'Exit Detection Map - Detection in {remaining_time:.1f}s | Points: {len(self.accumulated_points)}')
        else:
            self.ax.set_title(f'EXIT DETECTED! | Points: {len(self.accumulated_points)}')
        
        # Draw exit if detected
        if self.exit_detected and self.exit_location is not None:
            # Draw exit circle
            exit_circle = Circle(self.exit_location, 0.5, color='red', fill=False, linewidth=3)
            self.ax.add_patch(exit_circle)
            
            # Draw arrow pointing to exit
            self.ax.annotate('EXIT', self.exit_location, 
                           xytext=(0, 0), 
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           fontsize=16, weight='bold', color='red')
            
            # Draw exit direction line
            exit_range = 4.0
            end_x = exit_range * np.cos(self.exit_direction)
            end_y = exit_range * np.sin(self.exit_direction)
            self.ax.plot([0, end_x], [0, end_y], 'r--', linewidth=2, alpha=0.5)
            
            # Add exit info text
            exit_distance = np.sqrt(self.exit_location[0]**2 + self.exit_location[1]**2)
            info_text = f"Exit: {exit_distance:.1f}m @ {np.degrees(self.exit_direction):.0f}°"
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                        fontsize=12, weight='bold')
        
        return self.ax.artists
        
    def run(self):
        """Run the visualization"""
        print("Exit Detector Starting...")
        print(f"Exit detection will trigger after {self.detection_time} seconds")
        print("You are always at the center of the map (red circle)")
        print("Map points accumulate around you as you explore")
        print("Press Ctrl+C to stop\n")
        
        # Create animation
        try:
            # Try with cache_frame_data parameter for newer matplotlib versions
            ani = FuncAnimation(self.fig, self.update_plot, interval=500, blit=False, cache_frame_data=False)
        except TypeError:
            # Fall back to older version without the parameter
            ani = FuncAnimation(self.fig, self.update_plot, interval=500, blit=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nExit detector stopped.")
            

if __name__ == "__main__":
    # Create and run the exit detector
    detector = ExitDetectorVisualizer(detection_time=60.0, map_size_meters=10.0)
    detector.run()
