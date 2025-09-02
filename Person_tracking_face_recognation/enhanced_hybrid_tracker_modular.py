"""
Enhanced Hybrid Tracker - Modular Version
Multi-object tracking system with face recognition, activity analysis, and zone analytics
Refactored into modular components for better maintainability
"""

import cv2
import numpy as np
import time
from datetime import datetime, timedelta
import json
import os
import math

# Import modular components
from modules import DeviceManager, ActivityDetector, Visualizer, ZoneAnalytics

# Try to import face recognition
try:
    from face_recognizer import FaceRecognizer
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ Face recognition available")
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    print(f"❌ Face recognition not available: {e}")


class EnhancedHybridTracker:
    """Main tracking system with modular components"""
    
    def __init__(self, config_path='optimized_config.json', video_source=None):
        # Fast initialization - prioritize video connection over components
        print("🚀 Fast initialization for ORB-SLAM3 synchronization...")
        
        # Load configuration
        self.load_config(config_path)
        
        # Set video source (can be camera index, video file path, or stream URL)
        self.video_source = video_source or self.config.get('video_source', self.config.get('camera_index', 0))
        
        # Optimize initialization order: video first, then components
        print("⚡ Initializing video connection first...")
        
        # Initialize lightweight modular components first
        self.activity_detector = ActivityDetector(self.movement_history_size)
        self.visualizer = Visualizer(self.trail_length)
        self.zone_analytics = ZoneAnalytics()
        
        # Setup zones from config
        if 'zones' in self.config:
            self.zone_analytics.setup_zones_from_config(self.config['zones'])
        
        # Initialize face recognition (non-blocking)
        self.face_recognizer = None
        self.face_recognition_enabled = False
        self.init_face_recognition_safe()
        
        # Initialize device manager last (heaviest component)
        print("⚡ Initializing YOLO model...")
        self.device_manager = DeviceManager(self.config)
        
        # Core tracking parameters
        self.confidence_threshold = self.config['confidence_threshold']
        self.camera_index = self.config['camera_index']  # Keep for backward compatibility
        self.show_fps = self.config['show_fps']
        
        # Core tracking data
        self.tracks = {}
        self.next_track_id = 1
        self.track_face_attempts = {}
        self.track_face_ids = {}
        self.max_face_attempts = 12
        
        # Performance optimizations
        self.frame_skip_counter = 0
        self.face_recognition_interval = self.config.get('face_recognition_interval', 15)
        self.memory_cleanup_counter = 0
        self.max_tracks = 8
        
        # Session statistics
        self.session_stats = {
            'total_detections': 0,
            'unique_people': set(),
            'average_speed': [],
            'zone_visits': {},
            'activity_log': []
        }
        
        # Person stopping detection for ORB-SLAM3 coordination
        self.person_positions = {}  # track_id -> list of recent positions
        self.person_stop_times = {}  # track_id -> time when stopping was detected
        self.person_last_saved = {}  # track_id -> time when last saved to prevent duplicates
        self.stop_detection_threshold = 2.0  # seconds to consider person stopped
        self.movement_threshold = 20.0  # pixels - movement less than this = stopped
        self.position_history_size = 30  # frames to keep for movement analysis
        self.save_cooldown_period = 5.0  # seconds to wait before saving same person again
        
        print("✅ Enhanced Hybrid Tracker initialized with modular components")
    
    def initialize_video_source(self):
        """Initialize video source (camera, file, or stream) with sync optimization"""
        cap = None
        
        try:
            # Determine source type
            if isinstance(self.video_source, str):
                # Check if it's a file path
                if os.path.exists(self.video_source):
                    print(f"📹 Opening video file: {self.video_source}")
                    cap = cv2.VideoCapture(self.video_source)
                # Check if it's a stream URL
                elif self.video_source.startswith(('http://', 'https://', 'rtmp://', 'rtsp://')):
                    print(f"🌐 Opening video stream: {self.video_source}")
                    cap = cv2.VideoCapture(self.video_source)
                    
                    # SYNC OPTIMIZATION: Set stream properties for minimal latency
                    if cap:
                        print("⚡ Optimizing stream for synchronization...")
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer - only 1 frame
                        cap.set(cv2.CAP_PROP_FPS, 30)        # Match ORB-SLAM3 expectation
                        
                else:
                    print(f"❌ Video source not found: {self.video_source}")
                    print("🔄 Falling back to camera...")
                    cap = self.device_manager.initialize_camera_safe()
            else:
                # Integer camera index
                print(f"📷 Opening camera index: {self.video_source}")
                cap = cv2.VideoCapture(int(self.video_source))
                
                if not cap or not cap.isOpened():
                    print("🔄 Trying device manager...")
                    cap = self.device_manager.initialize_camera_safe()
            
            # Verify video source is working
            if cap and cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print("✅ Video source initialized successfully")
                    
                    # Get video properties for info
                    if hasattr(cap, 'get'):
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        print(f"📊 Video info: {width}x{height} @ {fps:.1f}fps")
                        if total_frames > 0:
                            print(f"📊 Total frames: {total_frames} (~{total_frames/fps:.1f}s)")
                    
                    return cap
                else:
                    print("❌ Could not read from video source")
                    if cap:
                        cap.release()
            else:
                print("❌ Could not open video source")
                
        except Exception as e:
            print(f"❌ Error initializing video source: {e}")
            if cap:
                cap.release()
        
        return None
    
    def load_config(self, config_path):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"⚠️ Config load error: {e}, using defaults")
            self.config = {
                'model': 'yolov9c.pt',
                'confidence_threshold': 0.4,
                'camera_index': 0,
                'video_source': 0,  # Can be camera index, file path, or stream URL
                'show_fps': True,
                'resolution': {'width': 1280, 'height': 720},
                'max_detections': 10,
                'memory_cleanup_interval': 50,
                'face_recognition_interval': 15,
                'zones': [
                    {'name': 'Entrance', 'bbox': [50, 50, 200, 150], 'color': [0, 255, 0]},
                    {'name': 'Exit', 'bbox': [400, 50, 200, 150], 'color': [0, 0, 255]}
                ]
            }
        
        # Set derived parameters
        self.movement_history_size = self.config.get('movement_history_size', 15)
        self.trail_length = self.config.get('trail_length', 20)
    
    def init_face_recognition_safe(self):
        """Initialize face recognition with comprehensive error handling"""
        if not FACE_RECOGNITION_AVAILABLE:
            print("⚠️ Face recognition unavailable")
            return
        
        try:
            print("👤 Initializing face recognition...")
            self.face_recognizer = FaceRecognizer()
            self.setup_predefined_names()
            self.face_recognition_enabled = True
            print("✅ Face recognition ready")
        except Exception as e:
            print(f"❌ Face recognition failed: {e}")
            print("🔄 Continuing without face recognition...")
            self.face_recognizer = None
            self.face_recognition_enabled = False
    
    def setup_predefined_names(self):
        """Setup predefined names and clean database to keep only Salah and Taha"""
        if self.face_recognizer:
            # First, clean the database to keep only Salah and Taha
            self.clean_face_database_for_salah_taha()
            
            # Set predefined names for any existing faces
            predefined_names = {1: "Taha", 4: "Salah"}
            for face_id, name in predefined_names.items():
                if face_id in self.face_recognizer.known_face_ids:
                    self.face_recognizer.set_face_name(face_id, name)
                    print(f"👤 Face ID {face_id}: {name}")
    
    def clean_face_database_for_salah_taha(self):
        """Clean face database to keep only Salah and Taha"""
        if not self.face_recognizer:
            return
            
        print("🧹 Cleaning face database to keep only Salah and Taha...")
        
        # Find Salah and Taha in current database
        salah_data = None
        taha_data = None
        
        for face_id, name in self.face_recognizer.face_names.items():
            if name.lower() == "salah":
                if face_id in self.face_recognizer.known_face_ids:
                    idx = self.face_recognizer.known_face_ids.index(face_id)
                    salah_data = {
                        'id': face_id,
                        'encoding': self.face_recognizer.known_face_encodings[idx],
                        'name': name
                    }
            elif name.lower() == "taha":
                if face_id in self.face_recognizer.known_face_ids:
                    idx = self.face_recognizer.known_face_ids.index(face_id)
                    taha_data = {
                        'id': face_id,
                        'encoding': self.face_recognizer.known_face_encodings[idx],
                        'name': name
                    }
        
        # If we have Salah and/or Taha, clean and re-add only them
        if salah_data or taha_data:
            original_count = len(self.face_recognizer.known_face_encodings)
            
            # Clear database
            self.face_recognizer.known_face_encodings = []
            self.face_recognizer.known_face_ids = []
            self.face_recognizer.face_names = {}
            self.face_recognizer.face_id_counter = 1
            
            # Re-add only Salah and Taha
            if salah_data:
                self.face_recognizer.known_face_encodings.append(salah_data['encoding'])
                self.face_recognizer.known_face_ids.append(salah_data['id'])
                self.face_recognizer.face_names[salah_data['id']] = salah_data['name']
                self.face_recognizer.face_id_counter = max(self.face_recognizer.face_id_counter, salah_data['id'] + 1)
                print(f"✅ Kept Salah with ID {salah_data['id']}")
            
            if taha_data:
                self.face_recognizer.known_face_encodings.append(taha_data['encoding'])
                self.face_recognizer.known_face_ids.append(taha_data['id'])
                self.face_recognizer.face_names[taha_data['id']] = taha_data['name']
                self.face_recognizer.face_id_counter = max(self.face_recognizer.face_id_counter, taha_data['id'] + 1)
                print(f"✅ Kept Taha with ID {taha_data['id']}")
            
            # Save cleaned database
            self.face_recognizer.save_face_database()
            
            final_count = len(self.face_recognizer.known_face_encodings)
            removed_count = original_count - final_count
            
            if removed_count > 0:
                print(f"🗑️  Removed {removed_count} other faces from database")
            print(f"📊 Database now contains {final_count} faces (Salah and Taha only)")
        else:
            print("ℹ️  No Salah or Taha found in database - no cleanup needed")
    
    def update_advanced_tracking(self, detections):
        """Advanced tracking with motion analysis using modular components"""
        current_time = time.time()
        
        # Age existing tracks and limit total tracks
        track_ids_to_check = list(self.tracks.keys())
        
        # If too many tracks, remove oldest lost tracks first
        if len(track_ids_to_check) > self.max_tracks:
            lost_tracks = [(tid, self.tracks[tid]['lost']) for tid in track_ids_to_check 
                          if self.tracks[tid]['lost'] > 0]
            lost_tracks.sort(key=lambda x: x[1], reverse=True)
            
            for tid, _ in lost_tracks[:len(track_ids_to_check) - self.max_tracks]:
                self.cleanup_track(tid)
                track_ids_to_check.remove(tid)
        
        for track_id in track_ids_to_check:
            if track_id in self.tracks:
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > 60:
                    self.cleanup_track(track_id)
        
        if len(detections) == 0:
            return []
        
        # Match detections to tracks
        matched_tracks = set()
        matched_dets = set()
        
        if self.tracks:
            track_ids = list(self.tracks.keys())
            
            for i, det in enumerate(detections):
                best_match = None
                best_iou = 0
                
                for track_id in track_ids:
                    if track_id in matched_tracks:
                        continue
                    
                    iou = self.activity_detector.compute_iou(self.tracks[track_id]['bbox'], det)
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_match = track_id
                
                if best_match is not None:
                    # Update existing track using activity detector
                    self.activity_detector.update_track_motion(best_match, det, current_time, self.tracks[best_match])
                    matched_tracks.add(best_match)
                    matched_dets.add(i)
        
        # Create new tracks
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.create_new_track(det, current_time)
        
        # Update zone analytics
        self.zone_analytics.update_zone_analytics(self.tracks)
        
        return self.get_active_tracks()
    
    def create_new_track(self, bbox, current_time):
        """Create new track with advanced initialization"""
        track_id = self.next_track_id
        
        self.tracks[track_id] = {
            'bbox': bbox,
            'lost': 0,
            'age': 1,
            'created_at': current_time,
            'last_seen': current_time
        }
        
        # Initialize face recognition attempts
        self.track_face_attempts[track_id] = 0
        
        self.next_track_id += 1
        self.session_stats['total_detections'] += 1
        
        print(f"🆕 New track {track_id} created")
    
    def get_active_tracks(self):
        """Get currently active tracks"""
        return [(tid, data) for tid, data in self.tracks.items() if data['lost'] < 30]
    
    def process_face_recognition_smart(self, frame, track_id, bbox):
        """Enhanced face recognition with comprehensive error handling"""
        try:
            if not self.face_recognition_enabled or not self.face_recognizer:
                return None
            
            if frame is None or track_id is None or bbox is None:
                return None
            
            if track_id not in self.track_face_attempts:
                self.track_face_attempts[track_id] = 0
            
            if track_id in self.track_face_ids:
                return self.track_face_ids[track_id]
            
            if self.track_face_attempts[track_id] >= self.max_face_attempts:
                try:
                    face_id = self.face_recognizer.process_detection(frame, bbox)
                    if face_id:
                        self.track_face_ids[track_id] = face_id
                        self.session_stats['unique_people'].add(face_id)
                        print(f"👤 Track {track_id} -> New Face ID {face_id}")
                        return face_id
                except Exception as e:
                    print(f"⚠️ Final face recognition attempt failed: {e}")
                return None
            
            try:
                self.frame_skip_counter += 1
                if self.frame_skip_counter % self.face_recognition_interval == 0:
                    self.track_face_attempts[track_id] += 1
                    
                    if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = bbox
                        if (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                            face_id = self.face_recognizer.process_detection(frame, bbox)
                            
                            if face_id:
                                self.track_face_ids[track_id] = face_id
                                self.session_stats['unique_people'].add(face_id)
                                print(f"✅ Track {track_id} -> Face ID {face_id}")
                                return face_id
                        else:
                            print(f"⚠️ Invalid bbox for track {track_id}: {bbox}")
                    else:
                        print(f"⚠️ Malformed bbox for track {track_id}: {bbox}")
            except Exception as e:
                print(f"⚠️ Face recognition processing failed for track {track_id}: {e}")
            
            return None
            
        except Exception as e:
            print(f"❌ Face recognition error for track {track_id}: {e}")
            return None
    
    def cleanup_track(self, track_id):
        """Clean up track data using modular components"""
        # Remove from core tracking
        for attr in [self.tracks, self.track_face_attempts, self.track_face_ids]:
            if track_id in attr:
                del attr[track_id]
        
        # Clean up modular components
        self.activity_detector.cleanup_track(track_id)
        self.visualizer.cleanup_track_visuals(track_id)
        self.zone_analytics.cleanup_track_zones(track_id)
    
    def cleanup_old_data(self):
        """Periodic cleanup of old data"""
        try:
            self.memory_cleanup_counter += 1
            if self.memory_cleanup_counter % self.config.get('memory_cleanup_interval', 100) == 0:
                # Clean up old tracks
                current_time = time.time()
                old_tracks = [tid for tid, data in self.tracks.items() 
                             if current_time - data.get('last_seen', current_time) > 300]  # 5 minutes
                
                for track_id in old_tracks:
                    self.cleanup_track(track_id)
                
                if old_tracks:
                    print(f"🧹 Cleaned up {len(old_tracks)} old tracks")
                
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
    
    def process_frame(self, frame):
        """Process single frame with modular components"""
        try:
            # Validate frame first
            if frame is None:
                print("⚠️ Received None frame")
                return frame, 0
                
            if not hasattr(frame, 'shape') or len(frame.shape) != 3:
                print("⚠️ Invalid frame format")
                return frame, 0
                
            if frame.size == 0:
                print("⚠️ Empty frame received")
                return frame, 0
            
            # Ensure frame is contiguous in memory
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Detect persons using device manager
            detections, confidences = self.device_manager.detect_persons_optimized(frame)
            
            # Update tracking
            active_tracks = self.update_advanced_tracking(detections)
            
            # Check for stopped persons and save coordinates
            self.check_stopped_persons(active_tracks, frame.shape)
            
            # Process each active track
            for track_id, track_data in active_tracks:
                bbox = track_data['bbox']
                
                # Face recognition
                face_id = self.process_face_recognition_smart(frame, track_id, bbox)
                face_name = self.get_face_name(face_id) if face_id else "Unknown"
                
                # Get activity information
                activity = self.activity_detector.get_track_activity(track_id)
                confidence = self.activity_detector.get_track_confidence(track_id)
                speed = self.activity_detector.get_track_speed(track_id)
                direction = self.activity_detector.get_track_direction(track_id)
                
                # Draw enhanced bounding box
                self.visualizer.draw_enhanced_bbox(frame, track_id, bbox, face_name, activity, confidence, speed)
                
                # Draw motion trail
                positions = self.activity_detector.get_track_positions(track_id)
                color = self.visualizer.get_track_color(track_id)
                self.visualizer.draw_motion_trail(frame, track_id, positions, color)
                
                # Draw direction arrow
                if len(positions) > 0:
                    cx, cy = positions[-1]
                    self.visualizer.draw_direction_arrow(frame, cx, cy, direction, color, speed)
                
                # Log activity
                if len(self.session_stats['activity_log']) < 1000:  # Limit log size
                    self.session_stats['activity_log'].append({
                        'timestamp': time.time(),
                        'track_id': track_id,
                        'activity': activity,
                        'confidence': confidence,
                        'speed': speed
                    })
            
            # Periodic cleanup
            self.cleanup_old_data()
            
            return frame, len(active_tracks)
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            import traceback
            traceback.print_exc()
            return frame, 0
    
    def get_face_name(self, face_id):
        """Get face name from ID"""
        if self.face_recognizer and face_id:
            faces = self.face_recognizer.list_faces_with_names()
            for face in faces:
                if face['id'] == face_id:
                    return face['name']
        return "Unknown"
    
    def run(self):
        """Enhanced tracking loop with comprehensive error handling"""
        print("📹 Initializing video system...")
        
        cap = None
        try:
            cap = self.initialize_video_source()
            if cap is None:
                print("❌ Failed to initialize video source")
                return
            
            self.run_main_loop(cap)
                
        except Exception as e:
            print(f"❌ Critical system error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            self.cleanup_on_exit()
            print("✅ System shutdown complete")
    
    def run_main_loop(self, cap):
        """Main tracking loop with window-triggered frame processing"""
        
        print("\n" + "="*60)
        print("ENHANCED HYBRID TRACKING SYSTEM - MODULAR")
        print("="*60)
        print("Modular Architecture")
        print("Device Management Module")
        print("Activity Detection Module")
        print("Visualization Module")
        print("Zone Analytics Module")
        print("\nControls:")
        print("  'q' - Quit    's' - Screenshot  'n' - Name face")
        print("  'l' - List    'r' - Reset       'i' - Toggle info")
        print("  'z' - Zones   'a' - Analytics   'h' - Help")
        print("="*60 + "\n")
        
        fps = 0
        frame_count = 0
        start_time = time.time()
        show_zones = True
        consecutive_failures = 0
        max_failures = 10
        window_initialized = False
        
        # Frame delay detection for automatic buffer flushing
        total_frames_processed = 0
        expected_frames = 0
        last_delay_check = time.time()
        frames_behind_threshold = 60  # Trigger flush when 60 frames behind
        
        print("⏸️  WAITING FOR WINDOW TO START...")
        print("📺 Window will initialize on first frame display")
        
        # First, create a placeholder to initialize the window
        placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_frame, "Initializing...", (200, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Enhanced Hybrid Tracker - Modular', placeholder_frame)
        cv2.waitKey(100)  # Small delay to ensure window creation
        
        print("🖥️  WINDOW STARTED! Now beginning frame capture from latest stream position...")
        window_initialized = True
        
        # Now flush the video buffer to start from the most recent frames
        if self.video_source.startswith(('rtmp://', 'rtsp://', 'http://')):
            print("⏩ Flushing video buffer to get latest frames...")
            time.sleep(0.5)  # Brief pause
            
            # Flush buffer by reading and discarding frames rapidly
            frames_flushed = 0
            flush_start = time.time()
            while time.time() - flush_start < 1.0:  # Flush for 1 second
                ret, flush_frame = cap.read()
                if ret and flush_frame is not None:
                    frames_flushed += 1
                else:
                    break
            print(f"⚡ Flushed {frames_flushed} frames - now starting from latest stream position")
        
        try:
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consecutive_failures += 1
                        print(f"⚠️ Frame read failed ({consecutive_failures}/{max_failures})")
                        
                        if consecutive_failures >= max_failures:
                            print("❌ Too many consecutive frame failures, stopping...")
                            break
                        
                        time.sleep(0.1)
                        continue
                    else:
                        consecutive_failures = 0
                    
                    # Enhanced frame validation
                    if frame is None or frame.size == 0:
                        print("⚠️ Invalid frame received")
                        continue
                    
                    # Check frame properties
                    if not hasattr(frame, 'shape') or len(frame.shape) != 3:
                        print("⚠️ Frame has invalid shape")
                        continue
                    
                    # Ensure frame is contiguous and writable
                    try:
                        if not frame.flags['C_CONTIGUOUS']:
                            frame = np.ascontiguousarray(frame)
                        if not frame.flags['WRITEABLE']:
                            frame = frame.copy()
                    except Exception as frame_fix_error:
                        print(f"⚠️ Frame memory fix error: {frame_fix_error}")
                        continue
                    
                    try:
                        # Draw zones
                        if show_zones:
                            self.visualizer.draw_zones(frame, self.zone_analytics.zones)
                        
                        # Process frame
                        processed_frame, track_count = self.process_frame(frame)
                        
                        # Calculate FPS and detect frame delays
                        try:
                            frame_count += 1
                            total_frames_processed += 1
                            elapsed_time = time.time() - start_time
                            if elapsed_time > 1:
                                fps = frame_count / elapsed_time
                                frame_count = 0
                                start_time = time.time()
                                
                                # Update session stats safely
                                if track_count > 0:
                                    try:
                                        activity_summary = self.activity_detector.get_statistics()
                                        if activity_summary['average_speed'] > 0:
                                            self.session_stats['average_speed'].append(activity_summary['average_speed'])
                                            if len(self.session_stats['average_speed']) > 100:
                                                self.session_stats['average_speed'] = self.session_stats['average_speed'][-50:]
                                    except Exception:
                                        pass
                            
                            # FRAME DELAY DETECTION: Check every 5 seconds if we're falling behind
                            current_time = time.time()
                            if current_time - last_delay_check > 5.0:  # Check every 5 seconds
                                if fps > 0:  # Only check if we have valid FPS
                                    time_elapsed = current_time - last_delay_check
                                    expected_frames += fps * time_elapsed  # Expected frames in this period
                                    frames_behind = expected_frames - total_frames_processed
                                    
                                    if frames_behind >= frames_behind_threshold and self.video_source.startswith(('rtmp://', 'rtsp://', 'http://')):
                                        print(f"⚠️  FRAME DELAY DETECTED: {frames_behind:.0f} frames behind expected!")
                                        print("⏩ Auto-flushing video buffer to catch up to latest frames...")
                                        
                                        # Flush buffer similar to initial startup
                                        frames_flushed = 0
                                        flush_start = time.time()
                                        while time.time() - flush_start < 0.8:  # Shorter flush period (0.8 seconds)
                                            ret_flush, flush_frame = cap.read()
                                            if ret_flush and flush_frame is not None:
                                                frames_flushed += 1
                                            else:
                                                break
                                        
                                        print(f"⚡ Auto-flushed {frames_flushed} frames - back to real-time processing")
                                        
                                        # Reset counters after flush
                                        total_frames_processed = 0
                                        expected_frames = 0
                                        start_time = time.time()  # Reset FPS timing
                                        frame_count = 0
                                
                                last_delay_check = current_time
                                
                        except Exception as fps_error:
                            print(f"⚠️ FPS calculation error: {fps_error}")
                        
                        # Add overlay
                        try:
                            display_frame = self.visualizer.add_advanced_overlay(processed_frame, track_count, fps, self.session_stats)
                        except Exception as overlay_error:
                            print(f"⚠️ Overlay error: {overlay_error}")
                            display_frame = processed_frame
                        
                        # Display
                        try:
                            cv2.imshow('Enhanced Hybrid Tracker - Modular', display_frame)
                        except Exception as display_error:
                            print(f"⚠️ Display error: {display_error}")
                        
                    except Exception as frame_error:
                        print(f"⚠️ Frame processing error: {frame_error}")
                        continue
                    
                    # Handle controls
                    try:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("🛑 Quit requested by user")
                            break
                        elif key == ord('s'):
                            try:
                                self.visualizer.save_screenshot(display_frame, "modular_tracker")
                            except Exception as e:
                                print(f"⚠️ Screenshot failed: {e}")
                        elif key == ord('z'):
                            show_zones = not show_zones
                            print(f"🎯 Zones: {'ON' if show_zones else 'OFF'}")
                        elif key == ord('i'):
                            self.visualizer.toggle_advanced_info()
                            print(f"📊 Advanced Info: {'ON' if self.visualizer.show_advanced_info else 'OFF'}")
                        elif key == ord('a'):
                            try:
                                self.print_analytics()
                            except Exception as e:
                                print(f"⚠️ Analytics error: {e}")
                        elif key == ord('h'):
                            try:
                                self.print_help()
                            except Exception as e:
                                print(f"⚠️ Help display error: {e}")
                        elif key == ord('n') and self.face_recognition_enabled:
                            try:
                                self.interactive_name_face()
                            except Exception as e:
                                print(f"⚠️ Face naming error: {e}")
                        elif key == ord('l') and self.face_recognition_enabled:
                            try:
                                self.list_all_faces()
                            except Exception as e:
                                print(f"⚠️ Face listing error: {e}")
                        elif key == ord('r'):
                            try:
                                self.reset_system()
                            except Exception as e:
                                print(f"⚠️ Reset error: {e}")
                    except Exception as key_error:
                        print(f"⚠️ Key handling error: {key_error}")
                        
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    self.cleanup_on_exit()
                    break
                except Exception as loop_error:
                    print(f"⚠️ Main loop error: {loop_error}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("❌ Too many loop errors, stopping...")
                        break
                    
        except Exception as main_error:
            print(f"❌ Critical main loop error: {main_error}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                print("💾 Saving session data...")
                self.save_session_data()
            except Exception as save_error:
                print(f"⚠️ Session save error: {save_error}")
    
    def print_analytics(self):
        """Print comprehensive analytics using modular components"""
        print("\n" + "="*50)
        print("MODULAR TRACKER ANALYTICS")
        print("="*50)
        print(f"Total Detections: {self.session_stats['total_detections']}")
        print(f"Unique People: {len(self.session_stats['unique_people'])}")
        if self.session_stats['average_speed']:
            recent_avg = np.mean(self.session_stats['average_speed'][-10:])
            print(f"Average Speed: {recent_avg:.1f} px/s")
        
        # Zone analytics
        zone_status = self.zone_analytics.get_zone_status()
        print("\nZone Analytics:")
        for zone_name, status in zone_status.items():
            print(f"  {zone_name}: {status['total_visits']} visits, "
                  f"avg dwell: {status['average_dwell_time']:.1f}s, "
                  f"current: {status['current_occupancy']}")
        
        # Activity summary
        activity_stats = self.activity_detector.get_statistics()
        print("\nActivity Summary:")
        for activity, count in activity_stats['activity_counts'].items():
            print(f"  {activity}: {count} people")
        
        # Device info
        device_info = self.device_manager.get_device_info()
        print("\nSystem Status:")
        print(f"  Device: {device_info['device'].upper()}")
        print(f"  YOLO Available: {device_info['yolo_available']}")
        print(f"  Face Recognition: {self.face_recognition_enabled}")
    
    def interactive_name_face(self):
        """Interactive face naming with error handling"""
        if not self.face_recognizer:
            return
            
        face_list = self.face_recognizer.list_faces_with_names()
        if face_list:
            print("\nCurrent faces:")
            for face in face_list:
                print(f"  ID {face['id']}: {face['name']}")
            try:
                face_id_input = input("Enter Face ID to name (or 'cancel' to exit): ").strip()
                if face_id_input.lower() in ['cancel', 'exit', 'quit', '']:
                    print("Operation cancelled")
                    return
                    
                face_id = int(face_id_input)
                
                valid_ids = [face['id'] for face in face_list]
                if face_id not in valid_ids:
                    print(f"❌ Face ID {face_id} not found. Valid IDs: {valid_ids}")
                    return
                
                name_input = input(f"Enter name for Face ID {face_id} (or 'cancel' to exit): ").strip()
                if name_input.lower() in ['cancel', 'exit', 'quit', '']:
                    print("Operation cancelled")
                    return
                    
                if name_input and len(name_input) > 0:
                    self.face_recognizer.set_face_name(face_id, name_input)
                    print(f"Face ID {face_id} named as '{name_input}'")
                else:
                    print("Name cannot be empty")
                    
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
            except Exception as e:
                print(f"Error during face naming: {e}")
    
    def list_all_faces(self):
        """List all faces"""
        if not self.face_recognizer:
            return
        face_list = self.face_recognizer.list_faces_with_names()
        if face_list:
            print("\nAll faces:")
            for face in face_list:
                print(f"  ID {face['id']}: {face['name']}")
        else:
            print("No faces found in database")
    
    def reset_system(self):
        """Reset tracking system"""
        self.tracks.clear()
        self.track_face_attempts.clear()
        self.track_face_ids.clear()
        self.next_track_id = 1
        
        # Reset person stopping detection
        self.person_positions.clear()
        self.person_stop_times.clear()
        self.person_last_saved.clear()
        
        # Reset modular components
        self.activity_detector = ActivityDetector(self.movement_history_size)
        self.zone_analytics.reset_analytics()
        
        # Reset session stats
        self.session_stats = {
            'total_detections': 0,
            'unique_people': set(),
            'average_speed': [],
            'zone_visits': {},
            'activity_log': []
        }
        
        print("🔄 System reset complete")
    
    def print_help(self):
        """Print help information"""
        print("\n" + "="*50)
        print("MODULAR TRACKER CONTROLS")
        print("="*50)
        print("Usage:")
        print("  python enhanced_hybrid_tracker_modular.py [video_source]")
        print("  video_source can be:")
        print("    - Camera index (0, 1, 2, etc.)")
        print("    - Video file path (path/to/video.mp4)")
        print("    - Stream URL (http://..., rtsp://..., etc.)")
        print("\nBasic Controls:")
        print("  'q' - Quit application")
        print("  's' - Save screenshot")
        print("  'i' - Toggle advanced info display")
        print("  'z' - Toggle zone visualization")
        print("  'h' - Show this help")
        print("\nAnalytics:")
        print("  'a' - Show detailed analytics")
        print("  'r' - Reset tracking system")
        print("\nFace Recognition:")
        print("  'n' - Name a face")
        print("  'l' - List all faces")
        print("="*50)
    
    def check_stopped_persons(self, active_tracks, frame_shape):
        """Check for persons who have stopped moving for 2+ seconds and save coordinates"""
        current_time = time.time()
        frame_height, frame_width = frame_shape[:2]
        
        # Track current active track IDs
        current_track_ids = set()
        
        for track_id, track_data in active_tracks:
            current_track_ids.add(track_id)
            bbox = track_data['bbox']
            
            # Calculate center point of bounding box
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            current_position = (center_x, center_y)
            
            # Initialize position history for new tracks
            if track_id not in self.person_positions:
                self.person_positions[track_id] = []
                self.person_stop_times[track_id] = None
                self.person_last_saved[track_id] = None
            
            # Add current position to history
            self.person_positions[track_id].append((current_position, current_time))
            
            # Keep only recent positions (last N frames)
            if len(self.person_positions[track_id]) > self.position_history_size:
                self.person_positions[track_id] = self.person_positions[track_id][-self.position_history_size:]
            
            # Check if person has stopped moving
            if len(self.person_positions[track_id]) >= 10:  # Need at least 10 positions for analysis
                positions = self.person_positions[track_id]
                recent_positions = [pos[0] for pos in positions[-10:]]  # Last 10 positions
                
                # Calculate maximum distance moved in recent positions
                max_distance = 0
                for i in range(len(recent_positions)):
                    for j in range(i + 1, len(recent_positions)):
                        pos1, pos2 = recent_positions[i], recent_positions[j]
                        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        max_distance = max(max_distance, distance)
                
                # Person is considered stopped if max movement < threshold
                is_stopped = max_distance < self.movement_threshold
                
                if is_stopped:
                    if self.person_stop_times[track_id] is None:
                        # Just started stopping
                        self.person_stop_times[track_id] = current_time
                        print(f"🟡 Person {track_id} stopped moving at position ({center_x}, {center_y})")
                    else:
                        # Check if stopped for required duration
                        stop_duration = current_time - self.person_stop_times[track_id]
                        if stop_duration >= self.stop_detection_threshold:
                            # Check cooldown period - only save if enough time has passed since last save
                            last_save_time = self.person_last_saved[track_id]
                            can_save = (last_save_time is None or 
                                      (current_time - last_save_time) >= self.save_cooldown_period)
                            
                            if can_save:
                                # Get person's name if available from face recognition
                                face_id = self.track_face_ids.get(track_id, None)
                                person_name = self.get_face_name(face_id) if face_id else None
                                
                                # Person has been stopped for 2+ seconds - save coordinates!
                                self.save_stopped_person_coordinates(track_id, bbox, frame_width, frame_height, person_name)
                                # Record the save time to enforce cooldown
                                self.person_last_saved[track_id] = current_time
                                # Reset stop time to avoid multiple saves for same stop
                                self.person_stop_times[track_id] = None
                            else:
                                # Still in cooldown period
                                time_remaining = self.save_cooldown_period - (current_time - last_save_time)
                                print(f"⏳ Person {track_id} in cooldown ({time_remaining:.1f}s remaining)")
                                # Reset stop time to continue monitoring
                                self.person_stop_times[track_id] = None
                else:
                    # Person is moving again
                    if self.person_stop_times[track_id] is not None:
                        print(f"🟢 Person {track_id} started moving again")
                        self.person_stop_times[track_id] = None
        
        # Clean up data for tracks that are no longer active
        tracks_to_remove = []
        for track_id in self.person_positions.keys():
            if track_id not in current_track_ids:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.person_positions[track_id]
            if track_id in self.person_stop_times:
                del self.person_stop_times[track_id]
            if track_id in self.person_last_saved:
                del self.person_last_saved[track_id]
    
    def save_stopped_person_coordinates(self, track_id, bbox, frame_width, frame_height, person_name=None):
        """Save rectangle coordinates when person stops for ORB-SLAM3 integration"""
        try:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = bbox
            
            # Calculate center and dimensions
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            # Create data structure for ORB-SLAM3
            stop_data = {
                'timestamp': time.time(),
                'track_id': track_id,
                'person_name': person_name if person_name and person_name != "Unknown" else None,
                'frame_dimensions': {
                    'width': frame_width,
                    'height': frame_height
                },
                'bounding_box': {
                    'x1': int(x1),
                    'y1': int(y1), 
                    'x2': int(x2),
                    'y2': int(y2),
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height
                },
                'normalized_coordinates': {
                    'x1': float(x1 / frame_width),
                    'y1': float(y1 / frame_height),
                    'x2': float(x2 / frame_width),
                    'y2': float(y2 / frame_height),
                    'center_x': float(center_x / frame_width),
                    'center_y': float(center_y / frame_height)
                }
            }
            
            # Save to file for ORB-SLAM3 C++ integration
            filename = "stopped_here.json"
            with open(filename, 'w') as f:
                json.dump(stop_data, f, indent=2)
            
            print(f"🔴 PERSON STOPPED! Saved coordinates to '{filename}'")
            print(f"   Track ID: {track_id}")
            if person_name and person_name != "Unknown":
                print(f"   Person Name: {person_name}")
            print(f"   Rectangle: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
            print(f"   Center: ({center_x}, {center_y})")
            print(f"   Frame: {frame_width}x{frame_height}")
            print(f"   For ORB-SLAM3 point clustering integration")
            
        except Exception as e:
            print(f"❌ Error saving stopped person coordinates: {e}")
    
    def cleanup_on_exit(self):
        """Clean up files when application exits"""
        try:
            # Delete the stopped_here.json file if it exists
            filename = "stopped_here.json"
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️  Cleaned up {filename}")
        except Exception as e:
            print(f"⚠️ Error cleaning up files: {e}")
    
    def save_session_data(self):
        """Save session data with proper serialization"""
        try:
            session_data = {
                'session_info': {
                    'end_time': datetime.now().isoformat(),
                    'total_detections': self.session_stats['total_detections'],
                    'unique_people': len(self.session_stats['unique_people']),
                    'device_info': self.device_manager.get_device_info()
                },
                'activity_stats': self.activity_detector.get_statistics(),
                'zone_analytics': self.zone_analytics.export_analytics_data(),
                'visual_stats': self.visualizer.get_visual_stats()
            }
            
            filename = f'modular_session_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            try:
                with open(filename, 'w') as f:
                    # Properly convert sets to lists for JSON serialization
                    serializable_data = session_data.copy()
                    
                    # Convert sets to lists recursively
                    def convert_sets(obj):
                        if isinstance(obj, set):
                            return list(obj)
                        elif isinstance(obj, dict):
                            return {k: convert_sets(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_sets(item) for item in obj]
                        return obj
                    
                    serializable_data = convert_sets(serializable_data)
                    json.dump(serializable_data, f, indent=2)
            except Exception as e:
                print(f"⚠️ Failed to save session data: {e}")
                return
            
            print(f"💾 Modular session data saved: {filename}")
            
        except Exception as e:
            print(f"⚠️ Session data preparation error: {e}")
    
    def process_frame_headless(self, frame):
        """
        Process a single frame without display - for ORB-SLAM3 bridge
        Returns: (detections, zones)
        """
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                return [], []
            
            # Ensure frame is in correct format
            if len(frame.shape) == 3:
                if frame.shape[2] == 3:  # BGR
                    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    input_frame = frame
            else:  # Grayscale
                input_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Run YOLO detection
            results = self.device_manager.model(input_frame, verbose=False)
            
            # Process detections
            detections_list = []
            current_frame_detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        if conf >= self.confidence_threshold and int(cls) == 0:  # Person class
                            x1, y1, x2, y2 = box
                            detection = {
                                'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                                'confidence': float(conf),
                                'class': int(cls),
                                'label': 'person'
                            }
                            current_frame_detections.append(detection)
            
            # Update tracking
            if current_frame_detections:
                for det in current_frame_detections:
                    x, y, w, h = det['bbox']
                    center = [x + w/2, y + h/2]
                    
                    # Find matching track or create new one
                    track_id = self.find_or_create_track(center, det)
                    det['track_id'] = track_id
                    
                    # Try face recognition if enabled
                    if self.face_recognition_enabled and track_id not in self.track_face_ids:
                        face_name = self.try_face_recognition(input_frame, det['bbox'])
                        if face_name and face_name != "Unknown":
                            self.track_face_ids[track_id] = face_name
                    
                    det['name'] = self.track_face_ids.get(track_id, "Unknown")
                    
                    # Activity detection
                    if track_id in self.tracks:
                        activity = self.activity_detector.analyze_movement(track_id, center, self.tracks[track_id])
                        det['activity'] = activity
                    else:
                        det['activity'] = 'idle'
                    
                    # Update track
                    self.tracks[track_id] = {
                        'last_seen': time.time(),
                        'center': center,
                        'bbox': det['bbox'],
                        'confidence': det['confidence']
                    }
                    
                    detections_list.append(det)
            
            # Update zone analytics
            zones_list = []
            try:
                self.zone_analytics.update_zones(current_frame_detections, input_frame.shape)
                for zone in self.zone_analytics.zones:
                    zone_data = {
                        'name': zone['name'],
                        'area': zone['bbox'],  # [x, y, width, height]
                        'color': zone['color'],
                        'person_count': len([d for d in current_frame_detections 
                                           if self.zone_analytics.point_in_zone([d['bbox'][0] + d['bbox'][2]/2, 
                                                                               d['bbox'][1] + d['bbox'][3]/2], zone)])
                    }
                    zones_list.append(zone_data)
            except Exception as zone_error:
                print(f"⚠️ Zone analytics error: {zone_error}")
            
            # Clean up old tracks
            self.cleanup_old_tracks()
            
            return detections_list, zones_list
            
        except Exception as e:
            print(f"❌ Headless processing error: {e}")
            return [], []
    
    def find_or_create_track(self, center, detection):
        """Find existing track or create new one"""
        min_distance = float('inf')
        best_track_id = None
        
        # Find closest existing track
        for track_id, track_data in self.tracks.items():
            if time.time() - track_data['last_seen'] < 2.0:  # Only consider recent tracks
                distance = np.sqrt((center[0] - track_data['center'][0])**2 + 
                                 (center[1] - track_data['center'][1])**2)
                if distance < min_distance and distance < 100:  # Threshold for matching
                    min_distance = distance
                    best_track_id = track_id
        
        # Create new track if no match found
        if best_track_id is None:
            best_track_id = self.next_track_id
            self.next_track_id += 1
        
        return best_track_id
    
    def try_face_recognition(self, frame, bbox):
        """Try to recognize face in bounding box"""
        if not self.face_recognition_enabled or not self.face_recognizer:
            return "Unknown"
        
        try:
            x, y, w, h = [int(coord) for coord in bbox]
            
            # Expand bbox slightly for better face detection
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2 * margin)
            h = min(frame.shape[0] - y, h + 2 * margin)
            
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size > 0:
                return self.face_recognizer.recognize_face(face_region)
                
        except Exception as e:
            print(f"⚠️ Face recognition error: {e}")
        
        return "Unknown"
    
    def cleanup_old_tracks(self):
        """Remove old tracks that haven't been seen recently"""
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id, track_data in self.tracks.items():
            if current_time - track_data['last_seen'] > 5.0:  # 5 seconds timeout
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            if track_id in self.track_face_ids:
                del self.track_face_ids[track_id]
            if track_id in self.track_face_attempts:
                del self.track_face_attempts[track_id]


def main():
    """Main function"""
    import sys
    
    try:
        # Parse command line arguments
        video_source = None
        if len(sys.argv) > 1:
            video_source = sys.argv[1]
            print(f"📹 Using video source from command line: {video_source}")
        
        # Create tracker with configuration
        config_file = 'optimized_config.json'
        if not os.path.exists(config_file):
            print(f"⚠️ Config file {config_file} not found, using defaults")
        
        tracker = EnhancedHybridTracker(config_file, video_source=video_source)
        tracker.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        # Cleanup on keyboard interrupt
        try:
            filename = "stopped_here.json"
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️  Cleaned up {filename}")
        except:
            pass
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()