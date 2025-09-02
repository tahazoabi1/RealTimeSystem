"""
 ORB-SLAM3 Bridge for Enhanced Hybrid Tracker (Windows Version)
Receives frames from ORB-SLAM3 via Windows shared memory and returns YOLO detections
"""

import cv2
import numpy as np
import json
import struct
import time
import threading
import sys
import os

# Windows-specific shared memory
import mmap
import ctypes
from ctypes import wintypes

# Windows API functions
kernel32 = ctypes.windll.kernel32
OpenFileMapping = kernel32.OpenFileMappingW
MapViewOfFile = kernel32.MapViewOfFile
UnmapViewOfFile = kernel32.UnmapViewOfFile
CloseHandle = kernel32.CloseHandle

# Constants
FILE_MAP_ALL_ACCESS = 0xF001F

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_hybrid_tracker_modular import EnhancedHybridTracker


class ORBSLAMBridgeWindows:
    """Windows Bridge between ORB-SLAM3 C++ and Python YOLO tracker"""
    
    def __init__(self, config_path='optimized_config.json'):
        print(f"🤖 Initializing ORB-SLAM3 Bridge with config: {config_path}")
        
        # Shared memory configuration
        self.shm_name = "ORB_SLAM3_YOLO_SharedMemory"
        self.shared_memory_handle = None
        self.shared_memory_ptr = None
        self.shared_memory_size = 1920 * 1080 * 3 + 1024 + 50 * 256  # Frame + header + detections
        
        # Initialize tracker
        if os.path.exists(config_path):
            self.tracker = EnhancedHybridTracker(config_path)
            print("✅ Enhanced Hybrid Tracker initialized")
        else:
            print(f"⚠️ Config file not found: {config_path}, using default config")
            self.tracker = EnhancedHybridTracker()
            
        self.running = False
        
        # Performance monitoring
        self.fps_timer = time.time()
        self.frame_count = 0
        self.avg_processing_time = 0
        
        # Data structures matching C++ SharedFrameData and SharedDetectionData
        self.frame_header_format = 'iiidbbi'  # width, height, channels, timestamp, frame_ready, processing_complete, num_detections
        self.frame_header_size = struct.calcsize(self.frame_header_format)
        
        self.detection_format = 'fffff32si64s32sfffi'  # x, y, width, height, confidence, label, track_id, person_name, activity, velocity_x, velocity_y, trail_length
        self.detection_size = struct.calcsize(self.detection_format)
    
    def connect_shared_memory(self):
        """Connect to the shared memory created by ORB-SLAM3"""
        try:
            # Wait for shared memory to be created by C++ side
            max_retries = 30  # 30 seconds timeout
            retry_count = 0
            
            while retry_count < max_retries:
                # Try to open existing shared memory
                self.shared_memory_handle = OpenFileMapping(
                    FILE_MAP_ALL_ACCESS,
                    False,
                    self.shm_name
                )
                
                if self.shared_memory_handle:
                    break
                    
                print(f"⏳ Waiting for shared memory... ({retry_count + 1}/{max_retries})")
                time.sleep(1)
                retry_count += 1
            
            if not self.shared_memory_handle:
                print(f"❌ Failed to open shared memory after {max_retries} seconds: {self.shm_name}")
                return False
            
            # Map the shared memory
            self.shared_memory_ptr = MapViewOfFile(
                self.shared_memory_handle,
                FILE_MAP_ALL_ACCESS,
                0,
                0,
                self.shared_memory_size
            )
            
            if not self.shared_memory_ptr:
                error_code = ctypes.windll.kernel32.GetLastError()
                print(f"❌ Failed to map shared memory view. Error code: {error_code}")
                return False
                
            print("✅ Connected to ORB-SLAM3 shared memory")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to shared memory: {e}")
            return False
    
    def disconnect_shared_memory(self):
        """Disconnect from shared memory"""
        if self.shared_memory_ptr:
            UnmapViewOfFile(self.shared_memory_ptr)
            self.shared_memory_ptr = None
            
        if self.shared_memory_handle:
            CloseHandle(self.shared_memory_handle)
            self.shared_memory_handle = None
    
    def read_frame_header(self):
        """Read frame header from shared memory"""
        if not self.shared_memory_ptr:
            return None
            
        try:
            # Read header data
            header_data = ctypes.string_at(self.shared_memory_ptr, self.frame_header_size)
            width, height, channels, timestamp, frame_ready, processing_complete, num_detections = struct.unpack(
                self.frame_header_format, header_data
            )
            
            return {
                'width': width,
                'height': height, 
                'channels': channels,
                'timestamp': timestamp,
                'frame_ready': bool(frame_ready),
                'processing_complete': bool(processing_complete),
                'num_detections': num_detections
            }
        except Exception as e:
            print(f"❌ Error reading frame header: {e}")
            return None
    
    def read_frame_data(self, header):
        """Read frame pixel data from shared memory"""
        if not self.shared_memory_ptr or not header:
            return None
            
        try:
            # Calculate frame data offset and size
            frame_offset = self.frame_header_size
            frame_size = header['width'] * header['height'] * header['channels']
            
            # Read frame pixels
            frame_data = ctypes.string_at(
                self.shared_memory_ptr + frame_offset, 
                frame_size
            )
            
            # Convert to numpy array and reshape
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame_array.reshape(header['height'], header['width'], header['channels'])
            
            return frame
            
        except Exception as e:
            print(f"❌ Error reading frame data: {e}")
            return None
    
    def write_detection_results(self, detections):
        """Write detection results back to shared memory"""
        if not self.shared_memory_ptr:
            return False
            
        try:
            # Update header with number of detections
            num_detections = min(len(detections), 50)  # Limit to 50 detections
            
            # Write number of detections to header (6th field, 0-indexed)
            num_det_offset = struct.calcsize('iiidbb')  # Offset to num_detections field
            struct.pack_into('i', ctypes.string_at(self.shared_memory_ptr, self.frame_header_size), 
                           num_det_offset, num_detections)
            
            # Calculate detection data offset (after header and frame data)
            header = self.read_frame_header()
            if not header:
                return False
                
            frame_size = header['width'] * header['height'] * header['channels']
            detection_offset = self.frame_header_size + frame_size
            
            # Write each detection
            for i, det in enumerate(detections[:num_detections]):
                det_offset = detection_offset + i * self.detection_size
                
                # Pack detection data
                detection_data = struct.pack(
                    self.detection_format,
                    float(det.get('bbox', {}).get('x', 0)),
                    float(det.get('bbox', {}).get('y', 0)),
                    float(det.get('bbox', {}).get('width', 0)),
                    float(det.get('bbox', {}).get('height', 0)),
                    float(det.get('confidence', 0.0)),
                    det.get('class', 'person').encode('utf-8')[:31].ljust(32, b'\0'),
                    int(det.get('track_id', 0)),
                    det.get('name', 'Unknown').encode('utf-8')[:63].ljust(64, b'\0'),
                    det.get('activity', '').encode('utf-8')[:31].ljust(32, b'\0'),
                    float(det.get('velocity', {}).get('x', 0)),
                    float(det.get('velocity', {}).get('y', 0)),
                    0  # trail_length (not implemented yet)
                )
                
                # Write to shared memory
                ctypes.memmove(self.shared_memory_ptr + det_offset, detection_data, len(detection_data))
            
            # Mark processing as complete
            complete_offset = struct.calcsize('iiidbb')  # Offset to processing_complete field
            struct.pack_into('b', ctypes.string_at(self.shared_memory_ptr, self.frame_header_size),
                           complete_offset - 1, 1)  # Set processing_complete = true
            
            return True
            
        except Exception as e:
            print(f"❌ Error writing detection results: {e}")
            return False
    
    def process_frame(self, frame):
        """Process frame through enhanced hybrid tracker"""
        try:
            start_time = time.time()
            
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Process through enhanced tracker
            results = self.tracker.process_frame_headless(frame_rgb)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.avg_processing_time = 0.9 * self.avg_processing_time + 0.1 * processing_time
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return []
    
    def run(self):
        """Main processing loop"""
        if not self.connect_shared_memory():
            print("❌ Failed to connect to shared memory, exiting")
            return
        
        self.running = True
        print("🚀 ORB-SLAM3 Bridge started, waiting for frames...")
        
        try:
            while self.running:
                # Read frame header
                header = self.read_frame_header()
                if not header:
                    time.sleep(0.001)  # 1ms sleep
                    continue
                
                # Check if new frame is ready
                if header['frame_ready'] and not header['processing_complete']:
                    # Read frame data
                    frame = self.read_frame_data(header)
                    if frame is not None:
                        # Process frame through YOLO
                        detections = self.process_frame(frame)
                        
                        # Write results back
                        self.write_detection_results(detections)
                        
                        self.frame_count += 1
                        
                        # Print FPS every 30 frames
                        if self.frame_count % 30 == 0:
                            current_time = time.time()
                            fps = 30 / (current_time - self.fps_timer)
                            self.fps_timer = current_time
                            print(f"🎯 Bridge FPS: {fps:.1f} | Avg Processing: {self.avg_processing_time*1000:.1f}ms")
                else:
                    time.sleep(0.001)  # 1ms sleep when no frame ready
                    
        except KeyboardInterrupt:
            print("🛑 Bridge interrupted by user")
        except Exception as e:
            print(f"❌ Bridge error: {e}")
        finally:
            self.running = False
            self.disconnect_shared_memory()
            print("✅ Bridge shutdown complete")


def main():
    """Main entry point"""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "optimized_config.json"
    
    bridge = ORBSLAMBridgeWindows(config_path)
    bridge.run()


if __name__ == "__main__":
    main()