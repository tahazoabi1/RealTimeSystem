"""
ORB-SLAM3 Bridge for Enhanced Hybrid Tracker
Receives frames from ORB-SLAM3 via shared memory and returns YOLO detections
"""

import cv2
import numpy as np
import json
import mmap
import struct
import time
import threading
from multiprocessing import shared_memory
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_hybrid_tracker_modular import EnhancedHybridTracker


class ORBSLAMBridge:
    """Bridge between ORB-SLAM3 C++ and Python YOLO tracker"""
    
    def __init__(self, config_path='optimized_config.json'):
        # Shared memory configuration
        self.shm_name = "ORB_SLAM3_YOLO_BRIDGE"
        self.frame_shm = None
        self.result_shm = None
        
        # Frame buffer size (max 4K resolution)
        self.max_width = 3840
        self.max_height = 2160
        self.frame_buffer_size = self.max_width * self.max_height * 3  # RGB
        
        # Result buffer size (max 100 detections)
        self.max_detections = 100
        self.detection_size = 256  # bytes per detection
        self.result_buffer_size = self.max_detections * self.detection_size
        
        # Initialize tracker
        self.tracker = EnhancedHybridTracker(config_path)
        self.running = False
        
        # Performance monitoring
        self.fps_timer = time.time()
        self.frame_count = 0
        self.avg_processing_time = 0
        
        print("🌉 ORB-SLAM3 Bridge initialized")
    
    def setup_shared_memory(self):
        """Setup shared memory for frame exchange"""
        try:
            # Create shared memory for frame input
            try:
                self.frame_shm = shared_memory.SharedMemory(
                    name=f"{self.shm_name}_FRAME",
                    create=True,
                    size=self.frame_buffer_size + 16  # +16 for metadata
                )
            except FileExistsError:
                # Attach to existing
                self.frame_shm = shared_memory.SharedMemory(
                    name=f"{self.shm_name}_FRAME"
                )
            
            # Create shared memory for detection results
            try:
                self.result_shm = shared_memory.SharedMemory(
                    name=f"{self.shm_name}_RESULT",
                    create=True,
                    size=self.result_buffer_size + 16  # +16 for metadata
                )
            except FileExistsError:
                # Attach to existing
                self.result_shm = shared_memory.SharedMemory(
                    name=f"{self.shm_name}_RESULT"
                )
            
            print("✅ Shared memory setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Shared memory setup failed: {e}")
            return False
    
    def read_frame_from_shm(self):
        """Read frame from shared memory"""
        if not self.frame_shm:
            return None, None
        
        try:
            # Read metadata (width, height, channels, timestamp)
            metadata = struct.unpack('iiid', self.frame_shm.buf[:16])
            width, height, channels, timestamp = metadata
            
            if width <= 0 or height <= 0:
                return None, None
            
            # Read frame data
            frame_size = width * height * channels
            frame_data = bytes(self.frame_shm.buf[16:16+frame_size])
            
            # Convert to numpy array
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((height, width, channels))
            
            return frame, timestamp
            
        except Exception as e:
            print(f"⚠️ Error reading frame: {e}")
            return None, None
    
    def write_detections_to_shm(self, detections, zones):
        """Write detection results to shared memory"""
        if not self.result_shm:
            return
        
        try:
            # Prepare detection data
            num_detections = min(len(detections), self.max_detections)
            
            # Write metadata
            metadata = struct.pack('if', num_detections, time.time())
            self.result_shm.buf[:8] = metadata
            
            # Write each detection
            offset = 16
            for i, det in enumerate(detections[:num_detections]):
                # Pack detection data
                # Format: track_id, bbox(4 floats), confidence, label_len, label, name_len, name, activity_len, activity
                det_data = struct.pack(
                    'ifffff',
                    det.get('track_id', -1),
                    det.get('bbox', [0,0,0,0])[0],  # x
                    det.get('bbox', [0,0,0,0])[1],  # y
                    det.get('bbox', [0,0,0,0])[2],  # width
                    det.get('bbox', [0,0,0,0])[3],  # height
                    det.get('confidence', 0.0)
                )
                
                # Add string data
                label = det.get('label', 'person').encode('utf-8')[:32]
                name = det.get('name', 'Unknown').encode('utf-8')[:32]
                activity = det.get('activity', 'idle').encode('utf-8')[:32]
                
                # Write to buffer
                self.result_shm.buf[offset:offset+24] = det_data
                offset += 24
                
                # Write strings with length prefixes
                self.result_shm.buf[offset:offset+1] = struct.pack('B', len(label))
                offset += 1
                self.result_shm.buf[offset:offset+len(label)] = label
                offset += 32  # Fixed size allocation
                
                self.result_shm.buf[offset:offset+1] = struct.pack('B', len(name))
                offset += 1
                self.result_shm.buf[offset:offset+len(name)] = name
                offset += 32
                
                self.result_shm.buf[offset:offset+1] = struct.pack('B', len(activity))
                offset += 1
                self.result_shm.buf[offset:offset+len(activity)] = activity
                offset += 32
            
        except Exception as e:
            print(f"⚠️ Error writing detections: {e}")
    
    def process_frame(self, frame):
        """Process frame through YOLO tracker"""
        start_time = time.time()
        
        try:
            # Run YOLO detection and tracking
            detections, zones = self.tracker.process_frame_headless(frame)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.avg_processing_time = 0.9 * self.avg_processing_time + 0.1 * processing_time
            
            return detections, zones
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return [], []
    
    def run(self):
        """Main processing loop"""
        if not self.setup_shared_memory():
            return
        
        self.running = True
        print("🚀 ORB-SLAM3 Bridge running...")
        
        while self.running:
            try:
                # Read frame from shared memory
                frame, timestamp = self.read_frame_from_shm()
                
                if frame is not None:
                    # Process through YOLO
                    detections, zones = self.process_frame(frame)
                    
                    # Write results back
                    self.write_detections_to_shm(detections, zones)
                    
                    # Update FPS counter
                    self.frame_count += 1
                    if time.time() - self.fps_timer > 1.0:
                        fps = self.frame_count / (time.time() - self.fps_timer)
                        print(f"📊 Bridge FPS: {fps:.1f}, Avg processing: {self.avg_processing_time*1000:.1f}ms")
                        self.fps_timer = time.time()
                        self.frame_count = 0
                else:
                    # No frame available, small sleep
                    time.sleep(0.001)
                    
            except KeyboardInterrupt:
                print("\n⏹ Stopping bridge...")
                break
            except Exception as e:
                print(f"❌ Bridge error: {e}")
                time.sleep(0.1)
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup shared memory"""
        self.running = False
        
        if self.frame_shm:
            self.frame_shm.close()
            try:
                self.frame_shm.unlink()
            except:
                pass
        
        if self.result_shm:
            self.result_shm.close()
            try:
                self.result_shm.unlink()
            except:
                pass
        
        print("🧹 Bridge cleanup complete")


def main():
    """Main entry point for standalone testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ORB-SLAM3 YOLO Bridge')
    parser.add_argument('--config', default='optimized_config.json',
                        help='Path to configuration file')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with video file')
    args = parser.parse_args()
    
    bridge = ORBSLAMBridge(args.config)
    
    if args.test:
        # Test mode - process a video file
        print("🧪 Running in test mode...")
        # Add test code here
    else:
        # Production mode - wait for ORB-SLAM3
        bridge.run()


if __name__ == "__main__":
    main()