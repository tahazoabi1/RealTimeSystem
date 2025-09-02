# People Landmark System - Real-Time Person Intelligence

## Overview

The People Landmark System is a sophisticated modular tracking system that creates persistent spatial memories of where people have been. Built with Python's enhanced hybrid tracker, it detects when people stop moving, recognizes their faces, and exports their exact coordinates for integration with ORB-SLAM3 navigation.

**Key Innovation**: The system automatically detects when someone stops for 2+ seconds, captures their exact screen coordinates, and exports this data to JSON for the C++ ORB-SLAM3 system to convert into 3D world landmarks.

---

## System Architecture - Modular Design

### Core Components

The system is built with a modular architecture for maintainability and extensibility:

```
Person_tracking_face_recognation/
├── enhanced_hybrid_tracker_modular.py  # Main tracking orchestrator
├── face_recognizer.py                  # Face recognition and database
├── modules/                            # Modular components
│   ├── device_manager.py               # GPU/CPU and YOLO management
│   ├── activity_detector.py            # Motion and activity analysis
│   ├── visualization.py                # Drawing and visual components
│   └── zone_analytics.py               # Zone-based analytics
├── optimized_config.json               # System configuration
├── yolov10m.pt                        # YOLO person detection model
└── face_database.pkl                   # Persistent face recognition database
```

### The Main Orchestrator (enhanced_hybrid_tracker_modular.py)

**System Initialization**:

```python
class EnhancedHybridTracker:
    def __init__(self, config_path='optimized_config.json', video_source=None):
        # Fast initialization for ORB-SLAM3 synchronization
        print("🚀 Fast initialization for ORB-SLAM3 synchronization...")
      
        # Initialize modular components
        self.activity_detector = ActivityDetector(self.movement_history_size)
        self.visualizer = Visualizer(self.trail_length) 
        self.zone_analytics = ZoneAnalytics()
        self.device_manager = DeviceManager(self.config)
      
        # Person stopping detection for ORB-SLAM3 coordination
        self.person_positions = {}      # track_id -> list of recent positions
        self.person_stop_times = {}     # track_id -> time when stopping detected
        self.person_last_saved = {}     # track_id -> time when last saved
        self.stop_detection_threshold = 2.0    # seconds to consider stopped
        self.movement_threshold = 20.0         # pixels - less = stopped
        self.position_history_size = 30        # frames for movement analysis
        self.save_cooldown_period = 5.0        # seconds between saves
```

---

## Person Detection and Tracking

### YOLO-Based Person Detection

**Device Manager handles YOLO initialization**:

```python
# From modules/device_manager.py
class DeviceManager:
    def __init__(self, config):
        # Smart device detection with GPU/CPU fallbacks
        self.device = self.detect_device()
      
        # Load YOLO model (YOLOv10m for accuracy)
        self.model = YOLO(config['model'])  # yolov10m.pt
        self.model.to(self.device)
      
        # Configuration
        self.confidence_threshold = config['confidence_threshold']  # 0.4
        self.max_detections = config['optimizations']['max_detections']  # 10
```

**Real-Time Person Detection**:

```python
def detect_persons(self, frame):
    # YOLO detection for person class (class 0)
    results = self.model(frame, 
                        conf=self.confidence_threshold,
                        classes=[0],  # Person class only
                        max_det=self.max_detections,
                        verbose=False)
  
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
              
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                })
  
    return detections
```

### Multi-Object Tracking

**Track Assignment and Management**:

```python
def update_tracks(self, detections, current_time):
    # IoU-based track assignment
    matched_tracks = []
    unmatched_detections = list(range(len(detections)))
  
    for track_id, track_data in self.tracks.items():
        if track_data['lost'] > 10:  # Remove lost tracks
            continue
          
        # Find best detection match using IoU
        best_match = self.find_best_match(track_data['bbox'], detections)
      
        if best_match is not None:
            det_idx, iou_score = best_match
            if iou_score > 0.3:  # IoU threshold for matching
                # Update track with matched detection
                matched_tracks.append((track_id, detections[det_idx]))
                unmatched_detections.remove(det_idx)
                track_data['lost'] = 0
            else:
                track_data['lost'] += 1
  
    # Create new tracks for unmatched detections
    for det_idx in unmatched_detections:
        new_track_id = self.next_track_id
        self.tracks[new_track_id] = {
            'bbox': detections[det_idx]['bbox'],
            'confidence': detections[det_idx]['confidence'], 
            'age': 1,
            'lost': 0,
            'last_seen': current_time
        }
        matched_tracks.append((new_track_id, detections[det_idx]))
        self.next_track_id += 1
  
    return matched_tracks
```

---

## Person Stopping Detection Algorithm

### The Core Innovation

The system monitors each person's movement over time and detects when they stop moving for 2+ seconds:

**Movement Analysis**:

```python
def check_stopped_persons(self, active_tracks, frame_shape):
    """Check for persons who have stopped moving for 2+ seconds"""
    current_time = time.time()
  
    for track_id, track_data in active_tracks:
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
      
        # Add current position to history
        self.person_positions[track_id].append((current_position, current_time))
      
        # Keep only recent positions (last 30 frames)
        if len(self.person_positions[track_id]) > self.position_history_size:
            self.person_positions[track_id] = self.person_positions[track_id][-30:]
```

**Stopping Detection Logic**:

```python
        # Check if person has stopped moving
        if len(self.person_positions[track_id]) >= 10:  # Need 10 positions minimum
            positions = self.person_positions[track_id]
            recent_positions = [pos[0] for pos in positions[-10:]]  # Last 10 positions
          
            # Calculate maximum distance moved in recent positions
            max_distance = 0
            for i in range(len(recent_positions)):
                for j in range(i + 1, len(recent_positions)):
                    pos1, pos2 = recent_positions[i], recent_positions[j]
                    distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    max_distance = max(max_distance, distance)
          
            # Person is stopped if max movement < 20 pixels
            is_stopped = max_distance < self.movement_threshold  # 20.0 pixels
          
            if is_stopped:
                if self.person_stop_times[track_id] is None:
                    # Just started stopping
                    self.person_stop_times[track_id] = current_time
                    print(f"🟡 Person {track_id} stopped at ({center_x}, {center_y})")
                else:
                    # Check if stopped for required duration (2+ seconds)
                    stop_duration = current_time - self.person_stop_times[track_id]
                    if stop_duration >= self.stop_detection_threshold:  # 2.0 seconds
                        # Person qualifies for landmark creation!
                        self.save_stopped_person_coordinates(track_id, bbox, 
                                                            frame_width, frame_height)
```

### Cooldown System

**Preventing Duplicate Saves**:

```python
                        # Check cooldown period - only save if enough time passed
                        last_save_time = self.person_last_saved[track_id]
                        can_save = (last_save_time is None or 
                                  (current_time - last_save_time) >= self.save_cooldown_period)
                      
                        if can_save:
                            # Save the landmark
                            self.save_stopped_person_coordinates(track_id, bbox, 
                                                                frame_width, frame_height, 
                                                                person_name)
                            # Record save time to enforce 5-second cooldown
                            self.person_last_saved[track_id] = current_time
                            self.person_stop_times[track_id] = None  # Reset
                        else:
                            # Still in cooldown period
                            time_remaining = self.save_cooldown_period - (current_time - last_save_time)
                            print(f"⏳ Person {track_id} in cooldown ({time_remaining:.1f}s remaining)")
```

---

## Face Recognition System

### Database-Driven Face Recognition

**Face Recognition Class (face_recognizer.py)**:

```python
class FaceRecognizer:
    def __init__(self, face_db_path='face_database.pkl', min_face_size=60):
        self.face_db_path = face_db_path
        self.min_face_size = min_face_size
        self.known_face_encodings = []
        self.known_face_ids = []
        self.face_names = {}  # Dictionary mapping face_id to name
        self.face_id_counter = 1
      
        self.load_face_database()
```

**Face Database Management**:

```python
    def load_face_database(self):
        if os.path.exists(self.face_db_path):
            with open(self.face_db_path, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data.get('encodings', [])
                self.known_face_ids = data.get('ids', [])
                self.face_id_counter = data.get('counter', 1)
                self.face_names = data.get('names', {})
            print(f"Loaded {len(self.known_face_encodings)} known faces")
        else:
            print("No existing face database found, starting fresh")
  
    def save_face_database(self):
        data = {
            'encodings': self.known_face_encodings,
            'ids': self.known_face_ids,
            'counter': self.face_id_counter,
            'names': self.face_names,
            'saved_at': datetime.now().isoformat()
        }
        with open(self.face_db_path, 'wb') as f:
            pickle.dump(data, f)
```

**Real-Time Face Recognition**:

```python
    def recognize_face(self, face_encoding, tolerance=0.5):
        if face_encoding is None or len(self.known_face_encodings) == 0:
            return None
      
        # Compare with known faces using face_recognition library
        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
      
        if len(distances) > 0:
            best_match_idx = np.argmin(distances)
            if distances[best_match_idx] <= tolerance:
                face_id = self.known_face_ids[best_match_idx]
                return face_id
      
        return None  # Unknown face
  
    def add_new_face(self, face_encoding):
        if face_encoding is not None:
            self.known_face_encodings.append(face_encoding)
            new_id = self.face_id_counter
            self.known_face_ids.append(new_id)
            self.face_id_counter += 1
          
            print(f"Added new face with ID: {new_id}")
            return new_id
        return None
```

### Interactive Face Naming

**Manual Face Naming System**:

```python
    def set_face_name(self, face_id, name):
        """Set a name for a specific face ID"""
        self.face_names[face_id] = name
        print(f"Face ID {face_id} named as '{name}'")
        self.save_face_database()
  
    def get_face_name(self, face_id):
        """Get the name for a face ID"""
        return self.face_names.get(face_id, f"ID {face_id}")
  
    def list_faces_with_names(self):
        """List all faces with their names"""
        face_list = []
        for face_id in self.known_face_ids:
            name = self.face_names.get(face_id, f"ID {face_id}")
            face_list.append({'id': face_id, 'name': name})
        return face_list
```

**Interactive Controls**:

- **'n'** - Name a face interactively
- **'l'** - List all known faces with names
- The system automatically assigns IDs to new faces and allows naming them later

---

## JSON Export for ORB-SLAM3 Integration

### The Critical Bridge Between Systems

When a person stops for 2+ seconds, the system exports their exact coordinates to JSON for C++ integration:

**Landmark Data Export**:

```python
def save_stopped_person_coordinates(self, track_id, bbox, frame_width, frame_height, person_name=None):
    """Save coordinates when person stops for ORB-SLAM3 integration"""
    try:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = bbox
      
        # Calculate center and dimensions
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        width = int(x2 - x1)
        height = int(y2 - y1)
      
        # Create comprehensive data structure for ORB-SLAM3
        stop_data = {
            'timestamp': time.time(),
            'track_id': track_id,
            'person_name': person_name if person_name and person_name != "Unknown" else None,
            'frame_dimensions': {
                'width': frame_width,
                'height': frame_height
            },
            'bounding_box': {
                'x1': int(x1), 'y1': int(y1),
                'x2': int(x2), 'y2': int(y2),
                'center_x': center_x, 'center_y': center_y,
                'width': width, 'height': height
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
      
        # Export to JSON file for C++ ORB-SLAM3 system
        filename = "stopped_here.json"
        with open(filename, 'w') as f:
            json.dump(stop_data, f, indent=2)
      
        print(f"🔴 PERSON STOPPED! Saved coordinates to '{filename}'")
```

**Example JSON Output**:

```json
{
  "timestamp": 1756664924.123,
  "track_id": 2,
  "person_name": "Salah",
  "frame_dimensions": {
    "width": 1280,
    "height": 720
  },
  "bounding_box": {
    "x1": 456,
    "y1": 123,
    "x2": 612,
    "y2": 456,
    "center_x": 534,
    "center_y": 289,
    "width": 156,
    "height": 333
  },
  "normalized_coordinates": {
    "x1": 0.35625,
    "y1": 0.17083,
    "x2": 0.47812,
    "y2": 0.63333,
    "center_x": 0.41719,
    "center_y": 0.40139
  }
}
```

---

## Activity Detection and Analysis

### Motion Pattern Analysis

**Activity Detector Module (modules/activity_detector.py)**:

```python
class ActivityDetector:
    def __init__(self, movement_history_size=15):
        self.movement_history_size = movement_history_size
      
        # Tracking data for motion analysis
        self.track_positions = {}
        self.track_velocities = {}
        self.track_speeds = {}
        self.track_activities = {}
  
    def update_track_motion(self, track_id, bbox, current_time, track_data):
        """Update motion analysis for a track"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
      
        # Initialize tracking data if new
        if track_id not in self.track_positions:
            self.track_positions[track_id] = [(cx, cy)]
            self.track_velocities[track_id] = []
            return
      
        # Add current position
        self.track_positions[track_id].append((cx, cy))
        if len(self.track_positions[track_id]) > self.movement_history_size:
            self.track_positions[track_id].pop(0)
      
        # Calculate velocity if we have previous position
        if len(self.track_positions[track_id]) > 1:
            prev_pos = self.track_positions[track_id][-2]
            curr_pos = self.track_positions[track_id][-1]
          
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            speed = math.sqrt(dx*dx + dy*dy)
          
            self.track_velocities[track_id].append((dx, dy))
            self.track_speeds[track_id].append(speed)
```

### Activity Classification

**Activity Analysis**:

```python
    def classify_activity(self, track_id):
        """Classify current activity based on motion patterns"""
        if track_id not in self.track_speeds or len(self.track_speeds[track_id]) < 5:
            return "Unknown"
      
        recent_speeds = self.track_speeds[track_id][-5:]
        avg_speed = sum(recent_speeds) / len(recent_speeds)
        speed_variance = np.var(recent_speeds)
      
        # Activity classification thresholds
        if avg_speed < 2.0:
            return "Standing"
        elif avg_speed < 8.0:
            return "Walking"
        elif avg_speed < 15.0:
            return "Fast Walking"
        else:
            return "Running"
```

---

## Configuration and Usage

### System Configuration (optimized_config.json)

**Core Settings**:

```json
{
  "model": "yolov10m.pt",
  "confidence_threshold": 0.4,
  "camera_index": 0,
  "show_fps": true,
  "resolution": {
    "width": 1280,
    "height": 720
  },
  "device": "cuda",
  "face_recognition_interval": 15,
  "optimizations": {
    "use_half_precision": true,
    "max_detections": 10,
    "memory_cleanup_interval": 100
  }
}
```

### Running the System

**Basic Usage**:

```bash
# Run with default camera
python enhanced_hybrid_tracker_modular.py

# Run with specific camera
python enhanced_hybrid_tracker_modular.py 0

# Run with video file
python enhanced_hybrid_tracker_modular.py /path/to/video.mp4

# Run with RTMP stream (for ORB-SLAM3 integration)
python enhanced_hybrid_tracker_modular.py rtmp://localhost:1935/live/stream
```

**Interactive Controls**:

- **'q'** - Quit application
- **'s'** - Save screenshot
- **'n'** - Name a face interactively
- **'l'** - List all known faces
- **'r'** - Reset tracking system
- **'i'** - Toggle advanced info display
- **'a'** - Show detailed analytics
- **'h'** - Show help

---

## Integration with ORB-SLAM3

### The Complete Pipeline

**Step 1: Python Detection System**

1. Real-time person detection using YOLOv10m
2. Multi-object tracking with IoU matching
3. Movement analysis and stopping detection
4. Face recognition and naming
5. Export stopped person coordinates to JSON

**Step 2: JSON Data Bridge**

- `stopped_here.json` contains precise 2D coordinates
- Both absolute pixel coordinates and normalized coordinates
- Person identity information if face was recognized
- Frame dimensions for coordinate transformation

**Step 3: C++ ORB-SLAM3 Integration**

- C++ system reads JSON file
- Converts 2D screen coordinates to 3D world coordinates using SLAM poses
- Creates persistent person landmarks in 3D space
- Provides proximity alerts during navigation

**Step 4: Navigation Experience**

- "Salah was here!" when approaching known person landmarks
- "Someone was here!" for unknown person landmarks
- Persistent spatial memory of human activity patterns

---

## C++ Integration Code - mono_integration_vv.cc

### Core Data Structures

**Person Data Structure**:

```cpp
struct StoppedPersonData {
    double timestamp;
    int track_id;
    std::string person_name;
    struct {
        int x1, y1, x2, y2;
        int center_x, center_y;
    } bounding_box;
    Sophus::SE3f camera_pose;  // Camera pose when person stopped
    bool has_camera_pose;
};
```

This structure holds all data about a stopped person imported from Python's JSON export, including their screen coordinates, identity, and the exact camera pose when detected.

### Python Process Integration

**Cross-Platform Python Subprocess Launch**:

```cpp
bool StartPythonTracker(RTMPData* pRTMPData) {
    // Build command for Python enhanced_hybrid_tracker_modular.py
    string python_cmd = "python enhanced_hybrid_tracker_modular.py " + pRTMPData->rtmp_url;
  
    // Cross-platform process creation
#ifdef _WIN32
    // Windows implementation using CreateProcess
    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);
  
    BOOL result = CreateProcessA(NULL, (char*)python_cmd.c_str(), NULL, NULL, 
                                FALSE, CREATE_NEW_CONSOLE, NULL, 
                                project_directory.c_str(), &si, &pi);
    return result != 0;
#else
    // Unix/Linux/macOS implementation using fork/exec
    pid_t pid = fork();
    if (pid == 0) {
        // Child process - execute Python script
        execl("/usr/bin/python3", "python3", "enhanced_hybrid_tracker_modular.py", 
              pRTMPData->rtmp_url.c_str(), (char*)NULL);
        exit(1); // If execl fails
    }
    return pid > 0; // Return true if fork succeeded
#endif
}
```

This starts the Python person detection system as a subprocess, feeding it the same RTMP stream that ORB-SLAM3 uses. Both systems process the live video in parallel with cross-platform compatibility.

### JSON File Monitoring

**Cross-Platform File System Monitoring**:

```cpp
bool CheckForNewStoppedPerson(RTMPData* pRTMPData) {
    // Check if stopped_here.json exists
    if (!std::filesystem::exists(pRTMPData->json_file_path)) {
        return false;
    }
  
    // Get file modification time
    auto ftime = std::filesystem::last_write_time(pRTMPData->json_file_path);
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ftime - std::filesystem::file_time_type::clock::now() + 
        std::chrono::system_clock::now());
    double current_modified_time = std::chrono::duration<double>(
        sctp.time_since_epoch()).count();
  
    if (current_modified_time > pRTMPData->last_json_modified_time) {
        // Parse JSON and capture current camera pose
        if (ParseStoppedPersonJSON(pRTMPData->json_file_path, 
                                 pRTMPData->last_stopped_person)) {
            pRTMPData->last_stopped_person.camera_pose = pRTMPData->current_camera_pose;
            pRTMPData->last_json_modified_time = current_modified_time;
          
            // Delete file to prevent reprocessing
            std::filesystem::remove(pRTMPData->json_file_path);
            return true;
        }
    }
    return false;
}
```

This monitors the JSON file that Python creates when someone stops. When detected, it captures the exact camera pose at that moment and deletes the file to prevent duplicate processing.

### MapPoint Accumulation System

**Temporal Point Collection**:

```cpp
void AccumulateFramePointsSimplified(ORB_SLAM3::System* pSLAM, 
                                    const Sophus::SE3f& current_camera_pose) {
    // Export current MapPoints from ORB-SLAM3
    std::string temp_filename = "temp_mappoints.txt";
    map_drawer->ExportMapPoints(temp_filename, 0.0f);
  
    // Camera transformation matrices
    Eigen::Matrix3f Rcw = current_camera_pose.rotationMatrix();
    Eigen::Vector3f tcw = current_camera_pose.translation();
  
    // Parse MapPoints and filter by person's bounding box
    std::ifstream file(temp_filename);
    std::string line;
    double timestamp = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
  
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float x, y, z;
        if (iss >> x >> y >> z) {
            // Transform world point to camera coordinates
            Eigen::Vector3f world_pos(x, y, z);
            Eigen::Vector3f camera_pos = Rcw * world_pos + tcw;
          
            // Project to image coordinates using camera intrinsics
            if (camera_pos.z() > 0) { // Point is in front of camera
                float u = fx * camera_pos.x() / camera_pos.z() + cx;
                float v = fy * camera_pos.y() / camera_pos.z() + cy;
              
                // Check if point is inside person's bounding box
                if (u >= bbox.x1 && u <= bbox.x2 && v >= bbox.y1 && v <= bbox.y2) {
                    temporal_points.push_back({cv::Point3f(x, y, z), timestamp});
                    accumulated_depths.push_back(camera_pos.z());
                }
            }
        }
    }
    file.close();
    std::remove(temp_filename.c_str()); // Clean up temporary file
}
```

This collects SLAM MapPoints over 2 seconds while a person is stopped. It projects 3D points to 2D screen coordinates and only keeps points that fall within the person's bounding box, building a precise 3D model of where they were.

### 3D World Position Calculation

**Ray Casting Algorithm**:

```cpp
Eigen::Vector3f CalculateWorldPositionFromPixelAndDepth(
    float pixel_u, float pixel_v, float depth,
    float fx, float fy, float cx, float cy,
    const Eigen::Matrix3f& Rcw, const Eigen::Vector3f& tcw) {
  
    // Convert pixel to normalized camera coordinates
    float x_norm = (pixel_u - cx) / fx;
    float y_norm = (pixel_v - cy) / fy;
  
    // Create ray in camera coordinates at specified depth
    Eigen::Vector3f camera_point(x_norm * depth, y_norm * depth, depth);
  
    // Transform to world coordinates using inverse camera transformation
    Eigen::Matrix3f Rwc = Rcw.transpose(); // World to camera rotation inverse
    Eigen::Vector3f twc = -Rwc * tcw;      // World to camera translation inverse
    Eigen::Vector3f world_point = Rwc * camera_point + twc;
  
    return world_point;
}
```

This is the core math that converts 2D screen coordinates to 3D world coordinates. It uses ray casting from the camera through the person's center point, combined with depth from MapPoints, to calculate their exact 3D location.

### Proximity Detection and Audio Alerts

**Cross-Platform Audio Announcement System**:

```cpp
void CheckProximityToStoppedPersons(RTMPData* pRTMPData, 
                                   const Sophus::SE3f& current_camera_pose) {
    // Get current camera position in world coordinates
    Eigen::Matrix3f Rwc = current_camera_pose.rotationMatrix().transpose();
    Eigen::Vector3f twc = -Rwc * current_camera_pose.translation();
    Eigen::Vector3f camera_world_pos = twc;
  
    // Check distance to each person landmark
    for (const auto& point : pRTMPData->clustered_points) {
        float dx = camera_world_pos.x() - point.x;
        float dy = camera_world_pos.y() - point.y;
        float dz = camera_world_pos.z() - point.z;
        float distance = sqrt(dx*dx + dy*dy + dz*dz);
      
        if (distance < 0.15f) { // 15cm threshold
            std::string person_name = pRTMPData->last_stopped_person.person_name;
            if (person_name.empty()) person_name = "Someone";
          
            std::string speech_message = person_name + " was here";
          
            // Cross-platform text-to-speech
#ifdef _WIN32
            // Windows PowerShell text-to-speech
            std::string tts_command = "powershell -Command \"Add-Type -AssemblyName System.Speech; "
                                    "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                                    "$synth.Speak('" + speech_message + "')\"";
#elif __APPLE__
            // macOS say command
            std::string tts_command = "say \"" + speech_message + "\"";
#else
            // Linux espeak or festival
            std::string tts_command = "espeak \"" + speech_message + "\" 2>/dev/null || "
                                    "echo \"" + speech_message + "\" | festival --tts 2>/dev/null";
#endif
          
            std::system(tts_command.c_str());
            pRTMPData->last_announcement_time = current_time;
            return;
        }
    }
}
```

This continuously monitors your current position and calculates distance to all stored person landmarks. When you get within 15cm of where someone stopped, it announces their name using the appropriate text-to-speech system for each platform with a 10-second cooldown.

### Main Integration Loop

**Real-Time Processing Pipeline**:

```cpp
void RTMPProcessingThread(RTMPData* pRTMPData) {
    while (!pRTMPData->finished) {
        // Capture frame from RTMP stream
        cv::Mat frame;
        if (!pRTMPData->pCapture->read(frame)) {
            continue;
        }
      
        // Convert to grayscale for ORB-SLAM3
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        double timestamp = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
      
        // Process RTMP frame with ORB-SLAM3
        Sophus::SE3f pose_result = pRTMPData->pSLAM->TrackMonocular(gray, timestamp);
        pRTMPData->current_camera_pose = pose_result;
        pRTMPData->frameCount++;
      
        // Check proximity to existing person landmarks
        CheckProximityToStoppedPersons(pRTMPData, pose_result);
      
        // Check for new stopped person every 30 frames (~1 second)
        if (pRTMPData->frameCount % 30 == 0) {
            if (CheckForNewStoppedPerson(pRTMPData)) {
                // Start collecting MapPoints for 2 seconds
                g_point_accumulator.StartCollection(
                    pRTMPData->last_stopped_person.track_id,
                    pRTMPData->last_stopped_person);
            }
        }
      
        // Continue accumulating MapPoints if in collection phase
        if (g_point_accumulator.ShouldContinueCollecting()) {
            g_point_accumulator.AccumulateFramePointsSimplified(pRTMPData->pSLAM, pose_result);
        }
        // Process accumulated points when collection completes
        else if (!g_point_accumulator.temporal_points.empty()) {
            std::vector<cv::Point3f> accumulated_points = g_point_accumulator.GetAccumulatedPoints();
          
            // Add landmark to map and proximity system
            std::string person_name = pRTMPData->last_stopped_person.person_name;
            int track_id = pRTMPData->last_stopped_person.track_id;
            map_drawer->AddStoppedPersonLandmark(accumulated_points, person_name, track_id);
            pRTMPData->clustered_points = accumulated_points;
          
            g_point_accumulator.temporal_points.clear();
        }
    }
}
```

This is the main processing loop that ties everything together. It processes SLAM frames, monitors for new JSON files from Python, accumulates MapPoints for stopped persons, and continuously checks proximity for audio announcements. The loop runs at 30fps, checking for new people every second while providing real-time proximity alerts.

---

## Technical Achievements

### Modular Architecture Benefits

**1. Maintainability**

- Each component has single responsibility
- Easy to debug and modify specific features
- Clear separation between detection, tracking, recognition, and export

**2. Performance Optimization**

- Device-aware YOLO loading (GPU/CPU automatic fallback)
- Efficient IoU-based tracking
- Memory cleanup routines
- Configurable processing intervals

**3. Cross-Platform Compatibility**

- Works on Windows, macOS, and Linux
- Platform-specific optimizations for subprocess management
- Universal text-to-speech integration across operating systems

**4. Real-Time Capability**

- Optimized for live video processing
- Fast initialization for ORB-SLAM3 synchronization
- Minimal latency person detection and export

### Key Innovation: Automatic Landmark Creation

The system solves the fundamental challenge of creating spatial memories of human activity:

- **Automatic Detection**: No manual intervention needed - system detects stopping automatically
- **Precise Timing**: 2-second threshold ensures intentional stops vs. brief pauses
- **Coordinate Accuracy**: Both pixel and normalized coordinates for robust 3D mapping
- **Identity Preservation**: Face recognition maintains person identity across sessions
- **Cross-Platform Integration**: JSON export works universally across operating systems

## Technical Innovation Summary

The key breakthrough is the **temporal accumulation of MapPoints over 2 seconds**, which creates highly accurate 3D person landmarks by:

1. **Collecting dozens of SLAM points** that fall within the person's bounding box
2. **Using median depth calculation** for precise world coordinate positioning
3. **Maintaining spatial persistence** - landmarks survive across navigation sessions
4. **Providing centimeter-level accuracy** for proximity detection (15cm threshold)
5. **Enabling personalized navigation** - "Salah was here!" vs "Someone was here!"

This system transforms ORB-SLAM3 from a purely spatial navigation tool into a **person-aware spatial memory system** that remembers where people have been and guides you back to those locations with spoken announcements, working seamlessly across all major operating systems.

---

## Demonstration Videos

### Video 1

https://youtu.be/pEepJBsWwrw

### Video 2

https://youtu.be/QGgwQavPWUY
