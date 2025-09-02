/**
* Real-time RTMP streaming with Python person detection integration
* Combines working ORB-SLAM3 mono_rtmp_stream with Python enhanced_hybrid_tracker_modular
* Runs Python script in parallel for person detection visualization
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include<thread>
#include<mutex>
#include<condition_variable>
#ifdef __APPLE__
#include<pthread.h>
#endif
#ifdef _WIN32
#include <windows.h>
#define sleep(x) Sleep((x)*1000)
#else
#include <unistd.h>
#endif

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>

#include"System.h"
#include "Converter.h"
#include "MapDrawer.h"
#include "AudioGuide.h"
#include "Tracking.h"
#include "Frame.h"
#include "MapPoint.h"

// For JSON file monitoring and point clustering
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>

// Simple JSON parsing for stopped_here.json
struct StoppedPersonData {
    double timestamp;
    int track_id;
    std::string person_name;
    struct {
        int width, height;
    } frame_dimensions;
    struct {
        int x1, y1, x2, y2;
        int center_x, center_y;
        int width, height;
    } bounding_box;
    struct {
        float x1, y1, x2, y2;
        float center_x, center_y;
    } normalized_coordinates;
    Sophus::SE3f camera_pose;  // Camera pose when person was detected
    bool has_camera_pose;       // Whether we have a valid pose
    struct {
        float r, g, b;  // RGB color values (0.0-1.0)
        int color_index; // Color index used for this person
    } assigned_color;
    bool has_assigned_color;     // Whether color has been assigned
};


using namespace std;

struct RTMPData {
    std::mutex mutex;
    std::condition_variable cv;
    bool finished = false;
    bool dataReady = false;
    cv::Mat currentFrame;
    double currentTimestamp = 0.0;
    ORB_SLAM3::System* pSLAM = nullptr;
    cv::VideoCapture* pCapture = nullptr;
    string rtmp_url = "";
    int frameCount = 0;
    float imageScale = 1.0f;
    bool recording_started = false;
    
    // Point clustering for stopped persons
    std::string json_file_path = "D:\\Learning\\realtimesystem\\stopped_here.json";
    double last_json_check_time = 0.0;
    double last_json_modified_time = 0.0;
    std::vector<cv::Point3f> clustered_points;  // Store clustered 3D points
    StoppedPersonData last_stopped_person;
    bool has_new_stopped_person = false;
    
    // Proximity notifications using existing AudioGuide from PathGuide
    std::set<int> announced_persons;  // Track which person track_ids have been announced
    double last_proximity_check = 0.0;
    double last_announcement_time = 0.0;  // Track last announcement for cooldown
    float proximity_threshold = 0.15f;  // 15cm detection radius
    float announcement_cooldown = 10.0f;  // 10 second cooldown between announcements
    Sophus::SE3f current_camera_pose;  // Store current camera pose
    bool has_valid_pose = false;
    
    // Python process handle
#ifdef _WIN32
    PROCESS_INFORMATION python_process = {0};
    STARTUPINFOA startup_info = {0};
#else
    pid_t python_pid = 0;
#endif
};

// Structure to track MapPoints over time for a stopped person
struct TemporalMapPointAccumulator {
    int track_id;
    struct {
        int x1, y1, x2, y2;
    } bbox;
    std::chrono::steady_clock::time_point start_time;
    std::vector<std::pair<cv::Point3f, double>> temporal_points; // (3D point, timestamp)
    std::vector<float> accumulated_depths;
    bool is_collecting = false;
    float collection_duration = 2.0f; // seconds
    
    void StartCollection(int id, int x1, int y1, int x2, int y2) {
        track_id = id;
        bbox.x1 = x1;
        bbox.y1 = y1;
        bbox.x2 = x2;
        bbox.y2 = y2;
        start_time = std::chrono::steady_clock::now();
        temporal_points.clear();
        accumulated_depths.clear();
        is_collecting = true;
        cout << "🎯 Started collecting MapPoints for person " << id << endl;
    }
    
    void StartCollection(int id, const StoppedPersonData& person_data) {
        StartCollection(id, person_data.bounding_box.x1, person_data.bounding_box.y1, 
                       person_data.bounding_box.x2, person_data.bounding_box.y2);
    }
    
    bool ShouldContinueCollecting() {
        if (!is_collecting) return false;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<float>(now - start_time).count();
        
        if (elapsed > collection_duration) {
            is_collecting = false;
            cout << "⏱️ Finished collecting after " << elapsed << " seconds" << endl;
            return false;
        }
        return true;
    }
    
    void AccumulateFramePointsSimplified(ORB_SLAM3::System* pSLAM, const Sophus::SE3f& current_camera_pose) {
        if (!is_collecting) return;
        
        // Use MapDrawer to export current MapPoints and parse them
        ORB_SLAM3::MapDrawer* map_drawer = pSLAM->GetMapDrawer();
        if (!map_drawer) return;
        
        // Export current MapPoints to temporary file
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::string temp_filename = "temp_mappoints_" + std::to_string(timestamp) + ".txt";
        
        map_drawer->ExportMapPoints(temp_filename, 0.0f);
        
        // Parse the exported MapPoints and filter by bounding box
        std::ifstream file(temp_filename);
        if (!file.is_open()) return;
        
        std::string line;
        int points_this_frame = 0;
        double current_timestamp_sec = timestamp / 1000.0;
        
        // Use the passed camera pose
        Eigen::Matrix3f Rcw = current_camera_pose.rotationMatrix();
        Eigen::Vector3f tcw = current_camera_pose.translation();
        
        // Camera intrinsics for projection
        float fx = 2622.0f;
        float fy = 2622.0f;
        float cx = 320.0f;  // Approximate
        float cy = 240.0f;  // Approximate
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            float x, y, z;
            
            if (iss >> x >> y >> z) {
                // Transform world point to camera coordinates
                Eigen::Vector3f world_pos(x, y, z);
                Eigen::Vector3f camera_pos = Rcw * world_pos + tcw;
                
                // Skip points behind camera
                if (camera_pos.z() <= 0.1f) continue;
                
                // Project to image coordinates
                float u = fx * camera_pos.x() / camera_pos.z() + cx;
                float v = fy * camera_pos.y() / camera_pos.z() + cy;
                
                // Check if point projects inside bounding box
                if (u >= bbox.x1 && u <= bbox.x2 && v >= bbox.y1 && v <= bbox.y2) {
                    float depth = camera_pos.z();
                    if (depth < 10.0f) { // Reasonable depth range
                        cv::Point3f point3d(x, y, z);
                        temporal_points.push_back({point3d, current_timestamp_sec});
                        accumulated_depths.push_back(depth);
                        points_this_frame++;
                    }
                }
            }
        }
        file.close();
        
        // Clean up temporary file
        std::filesystem::remove(temp_filename);
        
        if (points_this_frame > 0) {
            cout << "📍 Accumulated " << points_this_frame << " points this frame (total: " 
                 << temporal_points.size() << ")" << endl;
        }
    }
    
    float GetMedianDepth() {
        if (accumulated_depths.empty()) return -1.0f;
        
        std::vector<float> depths_copy = accumulated_depths;
        std::sort(depths_copy.begin(), depths_copy.end());
        return depths_copy[depths_copy.size() / 2];
    }
    
    std::vector<cv::Point3f> GetAccumulatedPoints() {
        std::vector<cv::Point3f> points;
        for (const auto& tp : temporal_points) {
            points.push_back(tp.first);
        }
        return points;
    }
};

// Global accumulator for current person being tracked
TemporalMapPointAccumulator g_point_accumulator;

// Function to get color from MapDrawer's frame colors
void AssignColorToPerson(StoppedPersonData& person_data, ORB_SLAM3::MapDrawer* map_drawer) {
    if (!map_drawer || person_data.has_assigned_color) return;
    
    // Get color based on track ID (same logic as MapDrawer)
    int color_idx = person_data.track_id % 6;
    
    // Access MapDrawer's frame colors (we need to get these values somehow)
    // For now, define standard colors that match typical frame colors
    float frame_colors[6][3] = {
        {1.0f, 0.0f, 0.0f},  // Red
        {0.0f, 1.0f, 0.0f},  // Green  
        {0.0f, 0.0f, 1.0f},  // Blue
        {0.0f, 0.0f, 0.0f},  // Yellow
        {1.0f, 0.0f, 1.0f},  // Magenta
        {0.0f, 1.0f, 1.0f}   // Cyan
    };
    
    person_data.assigned_color.r = frame_colors[color_idx][0];
    person_data.assigned_color.g = frame_colors[color_idx][1];
    person_data.assigned_color.b = frame_colors[color_idx][2];
    person_data.assigned_color.color_index = color_idx;
    person_data.has_assigned_color = true;
    
    cout << "🎨 Assigned color (" << person_data.assigned_color.r << ", " 
         << person_data.assigned_color.g << ", " << person_data.assigned_color.b 
         << ") to person ID " << person_data.track_id << endl;
}

// Function to save person data with color to JSON file
void SavePersonDataWithColor(const StoppedPersonData& person_data) {
    std::string filename = "person_" + std::to_string(person_data.track_id) + "_data.json";
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        cout << "⚠️ Failed to create person data file: " << filename << endl;
        return;
    }
    
    file << "{\n";
    file << "  \"timestamp\": " << person_data.timestamp << ",\n";
    file << "  \"track_id\": " << person_data.track_id << ",\n";
    file << "  \"person_name\": \"" << person_data.person_name << "\",\n";
    file << "  \"assigned_color\": {\n";
    file << "    \"r\": " << person_data.assigned_color.r << ",\n";
    file << "    \"g\": " << person_data.assigned_color.g << ",\n";
    file << "    \"b\": " << person_data.assigned_color.b << ",\n";
    file << "    \"color_index\": " << person_data.assigned_color.color_index << "\n";
    file << "  },\n";
    file << "  \"bounding_box\": {\n";
    file << "    \"x1\": " << person_data.bounding_box.x1 << ",\n";
    file << "    \"y1\": " << person_data.bounding_box.y1 << ",\n";
    file << "    \"x2\": " << person_data.bounding_box.x2 << ",\n";
    file << "    \"y2\": " << person_data.bounding_box.y2 << "\n";
    file << "  },\n";
    file << "  \"frame_dimensions\": {\n";
    file << "    \"width\": " << person_data.frame_dimensions.width << ",\n";
    file << "    \"height\": " << person_data.frame_dimensions.height << "\n";
    file << "  }\n";
    file << "}\n";
    
    file.close();
    
    cout << "💾 Saved person data with color to: " << filename << endl;
}

void* RTMPProcessingThreadWrapper(void* arg);
void RTMPProcessingThread(RTMPData* pRTMPData);

// JSON file monitoring and point clustering functions
bool CheckForNewStoppedPerson(RTMPData* pRTMPData);
bool ParseStoppedPersonJSON(const std::string& json_file, StoppedPersonData& data);

// Helper functions for proper 3D positioning
float EstimateDepthFromMapPoints(const std::string& mappoints_file, float pixel_u, float pixel_v,
                                float fx, float fy, float cx, float cy, 
                                const Eigen::Matrix3f& Rcw, const Eigen::Vector3f& tcw);
Eigen::Vector3f CalculateWorldPositionFromPixelAndDepth(float pixel_u, float pixel_v, float depth,
                                                       float fx, float fy, float cx, float cy,
                                                       const Eigen::Matrix3f& Rcw, const Eigen::Vector3f& tcw);

Eigen::Vector3f GetTrajectoryPointForPerson(RTMPData* pRTMPData, const StoppedPersonData& person_data);
void ClusterPointsInRectangle(RTMPData* pRTMPData, const StoppedPersonData& person_data);
void SaveClusteredPoints(const std::vector<cv::Point3f>& points, const StoppedPersonData& person_data);

// Proximity detection and audio notification
void CheckProximityToStoppedPersons(RTMPData* pRTMPData, const Sophus::SE3f& current_camera_pose);
bool StartPythonTracker(RTMPData* pRTMPData);
void StopPythonTracker(RTMPData* pRTMPData);

bool StartPythonTracker(RTMPData* pRTMPData)
{
    cout << "🐍 Starting Python Enhanced Hybrid Tracker..." << endl;
    
#ifdef _WIN32
    // Windows: Create Python process
    ZeroMemory(&pRTMPData->startup_info, sizeof(pRTMPData->startup_info));
    pRTMPData->startup_info.cb = sizeof(pRTMPData->startup_info);
    ZeroMemory(&pRTMPData->python_process, sizeof(pRTMPData->python_process));
    
    // Build command: python enhanced_hybrid_tracker_modular.py rtmp://127.0.0.1:1935/live/stream
    string python_cmd = "python D:\\Learning\\realtimesystem\\enhanced_hybrid_tracker_modular.py " + pRTMPData->rtmp_url;
    
    cout << "🔧 Running command: " << python_cmd << endl;
    
    // Convert string to LPSTR
    char* cmd_cstr = new char[python_cmd.length() + 1];
    strcpy_s(cmd_cstr, python_cmd.length() + 1, python_cmd.c_str());
    
    BOOL result = CreateProcessA(
        NULL,                   // No module name (use command line)
        cmd_cstr,              // Command line
        NULL,                  // Process handle not inheritable
        NULL,                  // Thread handle not inheritable
        FALSE,                 // Set handle inheritance to FALSE
        CREATE_NEW_CONSOLE,    // Creation flags - create new console window
        NULL,                  // Use parent's environment block
        "D:\\Learning\\realtimesystem", // Use realtimesystem as working directory
        &pRTMPData->startup_info,      // Pointer to STARTUPINFO structure
        &pRTMPData->python_process     // Pointer to PROCESS_INFORMATION structure
    );
    
    delete[] cmd_cstr;
    
    if (result) {
        cout << "✅ Python Enhanced Hybrid Tracker started successfully" << endl;
        cout << "📱 Python tracker will process the same RTMP stream: " << pRTMPData->rtmp_url << endl;
        return true;
    } else {
        DWORD error = GetLastError();
        cout << "❌ Failed to start Python tracker. Error: " << error << endl;
        cout << "💡 Make sure Python is in PATH and realtimesystem directory exists" << endl;
        return false;
    }
    
#else
    // Linux/macOS: Use fork and exec
    pRTMPData->python_pid = fork();
    
    if (pRTMPData->python_pid == 0) {
        // Child process - exec Python
        string python_script = "D/Learning/realtimesystem/enhanced_hybrid_tracker_modular.py";
        execl("/usr/bin/python3", "python3", python_script.c_str(), pRTMPData->rtmp_url.c_str(), (char*)NULL);
        
        // If we reach here, exec failed
        cout << "❌ Failed to exec Python script" << endl;
        exit(1);
    } else if (pRTMPData->python_pid > 0) {
        // Parent process
        cout << "✅ Python Enhanced Hybrid Tracker started (PID: " << pRTMPData->python_pid << ")" << endl;
        return true;
    } else {
        cout << "❌ Failed to fork Python process" << endl;
        return false;
    }
#endif
}

void StopPythonTracker(RTMPData* pRTMPData)
{
    cout << "🛑 Stopping Python Enhanced Hybrid Tracker..." << endl;
    
#ifdef _WIN32
    if (pRTMPData->python_process.hProcess != NULL) {
        // Try graceful termination first
        if (!TerminateProcess(pRTMPData->python_process.hProcess, 0)) {
            cout << "⚠️  Failed to terminate Python process gracefully" << endl;
        } else {
            cout << "✅ Python tracker terminated" << endl;
        }
        
        // Clean up handles
        CloseHandle(pRTMPData->python_process.hProcess);
        CloseHandle(pRTMPData->python_process.hThread);
        
        // Reset
        ZeroMemory(&pRTMPData->python_process, sizeof(pRTMPData->python_process));
    }
#else
    if (pRTMPData->python_pid > 0) {
        // Send SIGTERM to Python process
        kill(pRTMPData->python_pid, SIGTERM);
        
        // Wait a bit for graceful shutdown
        sleep(2);
        
        // Force kill if still running
        kill(pRTMPData->python_pid, SIGKILL);
        
        cout << "✅ Python tracker stopped" << endl;
        pRTMPData->python_pid = 0;
    }
#endif
}

void RTMPProcessingThread(RTMPData* pRTMPData)
{
    cv::Mat frame;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    
    cout << "🎥 Starting frame processing..." << endl;
    cout << "📱 Reading directly from RTMP stream" << endl;
    cout << "🌐 Stream URL: " << pRTMPData->rtmp_url << endl;
    cout << "🐍 Python tracker will run in parallel for person detection" << endl;
    
    cout << "⏸️  WAITING FOR ORB-SLAM3 VIEWER TO START..." << endl;
    cout << "📺 Viewer will initialize on main thread" << endl;
    
    // Give viewer time to initialize
    usleep(2000000); // 2 second delay for viewer startup
    
    cout << "🖥️  ORB-SLAM3 VIEWER STARTED! Now flushing stream buffer to latest frames..." << endl;
    
    // Flush video buffer to start from the most recent frames (similar to Python implementation)
    if (pRTMPData->rtmp_url.find("rtmp://") != std::string::npos || 
        pRTMPData->rtmp_url.find("rtsp://") != std::string::npos) {
        
        cout << "⏩ Flushing video buffer to get latest frames..." << endl;
        
        // Flush buffer by reading and discarding frames rapidly for 1.5 seconds
        auto flush_start = std::chrono::steady_clock::now();
        auto flush_duration = std::chrono::milliseconds(1500); // 1.5 seconds
        int frames_flushed = 0;
        cv::Mat flush_frame;
        
        while (std::chrono::steady_clock::now() - flush_start < flush_duration) {
            if (pRTMPData->pCapture->read(flush_frame)) {
                frames_flushed++;
                // Don't process these frames, just discard them
            } else {
                // If we can't read frames, break out
                break;
            }
            
            // Check if we should stop
            if (pRTMPData->finished) {
                return;
            }
        }
        
        cout << "⚡ Flushed " << frames_flushed << " frames - now starting from latest stream position" << endl;
    }
    
    cout << "🚀 Starting real-time SLAM processing from current stream position..." << endl;
    
    while (!pRTMPData->finished) {
        // Capture frame from RTMP stream
        if (!pRTMPData->pCapture->read(frame)) {
            cerr << "⚠️  Lost connection to RTMP stream, attempting to reconnect..." << endl;
            usleep(1000000); // Wait 1 second before retry
            continue;
        }
        
        if (frame.empty()) {
            cerr << "⚠️  Empty frame received from RTMP stream" << endl;
            continue;
        }
        
        // Convert to grayscale for SLAM
        cv::Mat gray;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame.clone();
        }
        
        // Apply image scaling if needed
        if (pRTMPData->imageScale != 1.0f) {
            int width = gray.cols * pRTMPData->imageScale;
            int height = gray.rows * pRTMPData->imageScale;
            cv::resize(gray, gray, cv::Size(width, height));
        }
        
        // Apply CLAHE for better feature detection
        clahe->apply(gray, gray);
        
        // Validate frame quality before SLAM processing
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        if (stddev[0] < 5.0) {
            // Skip frames with too little variation (likely bad quality)
            continue;
        }
        
        // Get current timestamp
        double timestamp = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
        
        {
            std::lock_guard<std::mutex> lock(pRTMPData->mutex);
            if (pRTMPData->finished) break;
            pRTMPData->frameCount++;
        }
        
        // Process with ORB-SLAM3 (same as working version)
        if (pRTMPData->pSLAM) {
            // Track consecutive tracking failures
            static int consecutive_failures = 0;
            
            try {
                Sophus::SE3f pose_result = pRTMPData->pSLAM->TrackMonocular(gray, timestamp, 
                    vector<ORB_SLAM3::IMU::Point>(), 
                    "rtmp_frame_" + to_string(pRTMPData->frameCount));
                
                // Store the current camera pose for point clustering
                pRTMPData->current_camera_pose = pose_result;
                pRTMPData->has_valid_pose = true;
                
                // Check proximity to stopped person landmarks and announce them
                CheckProximityToStoppedPersons(pRTMPData, pose_result);
                    
                consecutive_failures = 0; // Reset on success
            } catch (const std::exception& e) {
                cerr << "⚠️ Exception in TrackMonocular: " << e.what() << endl;
                consecutive_failures++;
                
                // If too many failures, try reinitializing
                if (consecutive_failures > 50) {
                    cerr << "🔄 Too many tracking failures, attempting recovery..." << endl;
                    consecutive_failures = 0;
                    // Let system attempt to reinitialize on next good frame
                }
            }
        }
        
        // Auto-start recording after 100 frames (when SLAM is stable)
        if (pRTMPData->frameCount == 100 && !pRTMPData->recording_started) {
            string filename = "live_path_" + to_string(time(nullptr)) + ".txt";
            pRTMPData->pSLAM->StartPathRecording(filename);
            pRTMPData->recording_started = true;
            cout << "🔴 AUTO-STARTED path recording: " << filename << endl;
        }
        
        // Check for stopped person JSON file (every 30 frames = ~1 second)
        if (pRTMPData->frameCount % 30 == 0) {
            if (CheckForNewStoppedPerson(pRTMPData)) {
                cout << "🔍 NEW STOPPED PERSON DETECTED! Starting temporal accumulation..." << endl;
                
                // Start collecting MapPoints for this person
                g_point_accumulator.StartCollection(
                    pRTMPData->last_stopped_person.track_id,
                    pRTMPData->last_stopped_person
                );
                
                pRTMPData->has_new_stopped_person = false;
            }
        }
        
        // Continue accumulating points if we're in collection phase
        if (g_point_accumulator.ShouldContinueCollecting()) {
            g_point_accumulator.AccumulateFramePointsSimplified(pRTMPData->pSLAM, pRTMPData->current_camera_pose);
        }
        else if (!g_point_accumulator.is_collecting && 
                 !g_point_accumulator.temporal_points.empty()) {
            // Collection finished, process the accumulated points
            cout << "✅ Processing " << g_point_accumulator.temporal_points.size() 
                 << " accumulated MapPoints" << endl;
            
            float median_depth = g_point_accumulator.GetMedianDepth();
            std::vector<cv::Point3f> accumulated_points = g_point_accumulator.GetAccumulatedPoints();
            
            if (median_depth > 0) {
                cout << "📏 Final median depth from temporal accumulation: " 
                     << median_depth << "m" << endl;
                
                // Calculate world position using median depth
                float person_center_u = (pRTMPData->last_stopped_person.bounding_box.x1 + 
                                       pRTMPData->last_stopped_person.bounding_box.x2) / 2.0f;
                float person_center_v = (pRTMPData->last_stopped_person.bounding_box.y1 + 
                                       pRTMPData->last_stopped_person.bounding_box.y2) / 2.0f;
                
                // Use proper camera intrinsics
                float fx = 2622.0f;
                float fy = 2622.0f;
                float cx = pRTMPData->last_stopped_person.frame_dimensions.width / 2.0f;
                float cy = pRTMPData->last_stopped_person.frame_dimensions.height / 2.0f;
                
                // Get camera pose for transformation
                Eigen::Matrix3f Rcw = pRTMPData->last_stopped_person.camera_pose.rotationMatrix();
                Eigen::Vector3f tcw = pRTMPData->last_stopped_person.camera_pose.translation();
                
                // Assign color to person and save data
                ORB_SLAM3::MapDrawer* map_drawer = pRTMPData->pSLAM->GetMapDrawer();
                if (map_drawer) {
                    AssignColorToPerson(pRTMPData->last_stopped_person, map_drawer);
                    SavePersonDataWithColor(pRTMPData->last_stopped_person);
                }
                
                // If we have accumulated points, use them directly
                if (!accumulated_points.empty()) {
                    // Add the landmark to the map with accumulated points
                    if (map_drawer) {
                        map_drawer->AddStoppedPersonLandmark(
                            accumulated_points,
                            pRTMPData->last_stopped_person.person_name,
                            pRTMPData->last_stopped_person.track_id
                        );
                        cout << "🎯 Added landmark with " << accumulated_points.size() 
                             << " accumulated points to map!" << endl;
                        
                        // 🔧 FIX: Also add points to clustered_points for proximity detection
                        pRTMPData->clustered_points = accumulated_points;
                        cout << "🔧 Fixed: Added " << accumulated_points.size() << " points to proximity detection!" << endl;
                    }
                } else {
                    // Fallback: calculate single point using median depth
                    Eigen::Vector3f world_position = CalculateWorldPositionFromPixelAndDepth(
                        person_center_u, person_center_v, median_depth,
                        fx, fy, cx, cy, Rcw, tcw
                    );
                    
                    std::vector<cv::Point3f> single_point;
                    single_point.push_back(cv::Point3f(world_position.x(), 
                                                      world_position.y(), 
                                                      world_position.z()));
                    
                    if (map_drawer) {
                        map_drawer->AddStoppedPersonLandmark(
                            single_point,
                            pRTMPData->last_stopped_person.person_name,
                            pRTMPData->last_stopped_person.track_id
                        );
                        
                        // 🔧 FIX: Also add point to clustered_points for proximity detection
                        pRTMPData->clustered_points = single_point;
                        cout << "🔧 Fixed: Added fallback point to proximity detection!" << endl;
                    }
                }
            } else {
                cout << "⚠️ No valid depth found from accumulated points" << endl;
            }
            
            // Clear accumulator for next detection
            g_point_accumulator.temporal_points.clear();
            g_point_accumulator.accumulated_depths.clear();
        }
        
        // Progress indicator
        if (pRTMPData->frameCount % 100 == 0) {
            cout << "📹 Processed " << pRTMPData->frameCount << " frames from iPhone stream" << endl;
            cout << "🐍 Python Enhanced Hybrid Tracker running in parallel for person detection" << endl;
        }
        
        // No artificial delay for maximum real-time performance
        // usleep(1000); // Removed delay
    }
    
    cout << "🛑 RTMP processing thread finished" << endl;
}

void* RTMPProcessingThreadWrapper(void* arg)
{
    RTMPData* pRTMPData = static_cast<RTMPData*>(arg);
    RTMPProcessingThread(pRTMPData);
    return nullptr;
}

// JSON file monitoring and point clustering implementation
bool CheckForNewStoppedPerson(RTMPData* pRTMPData)
{
    try {
        // Check if JSON file exists and get modification time
        if (!std::filesystem::exists(pRTMPData->json_file_path)) {
            return false;
        }
        
        auto ftime = std::filesystem::last_write_time(pRTMPData->json_file_path);
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
        double current_modified_time = std::chrono::duration<double>(sctp.time_since_epoch()).count();
        
        // Check if file was modified since last check
        if (current_modified_time > pRTMPData->last_json_modified_time) {
            pRTMPData->last_json_modified_time = current_modified_time;
            
            // Parse the JSON file
            if (ParseStoppedPersonJSON(pRTMPData->json_file_path, pRTMPData->last_stopped_person)) {
                // Capture current camera pose when person stops
                if (pRTMPData->has_valid_pose) {
                    pRTMPData->last_stopped_person.camera_pose = pRTMPData->current_camera_pose;
                    pRTMPData->last_stopped_person.has_camera_pose = true;
                } else {
                    pRTMPData->last_stopped_person.has_camera_pose = false;
                }
                
                // Delete the JSON file after processing to avoid reprocessing
                try {
                    std::filesystem::remove(pRTMPData->json_file_path);
                    cout << "🗑️ Deleted processed stopped_here.json file" << endl;
                } catch (const std::exception& e) {
                    cerr << "⚠️ Warning: Could not delete JSON file: " << e.what() << endl;
                }
                
                pRTMPData->has_new_stopped_person = true;
                return true;
            }
        }
        
    } catch (const std::exception& e) {
        cerr << "⚠️ Error checking stopped person JSON: " << e.what() << endl;
    }
    
    return false;
}

bool ParseStoppedPersonJSON(const std::string& json_file, StoppedPersonData& data)
{
    try {
        // Initialize color fields
        data.has_assigned_color = false;
        data.assigned_color.r = 0.0f;
        data.assigned_color.g = 0.0f;
        data.assigned_color.b = 0.0f;
        data.assigned_color.color_index = -1;
        
        std::ifstream file(json_file);
        if (!file.is_open()) {
            return false;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();
        
        // Simple JSON parsing (basic implementation)
        // Look for key values in the JSON content
        size_t pos;
        
        // Parse timestamp
        pos = content.find("\"timestamp\":");
        if (pos != std::string::npos) {
            pos = content.find(":", pos) + 1;
            size_t end = content.find(",", pos);
            if (end == std::string::npos) end = content.find("}", pos);
            std::string value = content.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            data.timestamp = std::stod(value);
        }
        
        // Parse track_id
        pos = content.find("\"track_id\":");
        if (pos != std::string::npos) {
            pos = content.find(":", pos) + 1;
            size_t end = content.find(",", pos);
            if (end == std::string::npos) end = content.find("}", pos);
            std::string value = content.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            data.track_id = std::stoi(value);
        }
        
        // Parse person_name
        pos = content.find("\"person_name\":");
        if (pos != std::string::npos) {
            pos = content.find(":", pos) + 1;
            size_t end = content.find(",", pos);
            if (end == std::string::npos) end = content.find("}", pos);
            std::string value = content.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            if (value != "null" && value.length() > 2) {
                data.person_name = value.substr(1, value.length() - 2); // Remove quotes
            }
        }
        
        // Parse frame dimensions
        pos = content.find("\"width\":");
        if (pos != std::string::npos) {
            pos = content.find(":", pos) + 1;
            size_t end = content.find(",", pos);
            std::string value = content.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            data.frame_dimensions.width = std::stoi(value);
        }
        
        pos = content.find("\"height\":", content.find("frame_dimensions"));
        if (pos != std::string::npos) {
            pos = content.find(":", pos) + 1;
            size_t end = content.find("}", pos);
            std::string value = content.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            data.frame_dimensions.height = std::stoi(value);
        }
        
        // Parse bounding box coordinates
        pos = content.find("\"x1\":", content.find("bounding_box"));
        if (pos != std::string::npos) {
            pos = content.find(":", pos) + 1;
            size_t end = content.find(",", pos);
            std::string value = content.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            data.bounding_box.x1 = std::stoi(value);
        }
        
        pos = content.find("\"y1\":", content.find("bounding_box"));
        if (pos != std::string::npos) {
            pos = content.find(":", pos) + 1;
            size_t end = content.find(",", pos);
            std::string value = content.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            data.bounding_box.y1 = std::stoi(value);
        }
        
        pos = content.find("\"x2\":", content.find("bounding_box"));
        if (pos != std::string::npos) {
            pos = content.find(":", pos) + 1;
            size_t end = content.find(",", pos);
            std::string value = content.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            data.bounding_box.x2 = std::stoi(value);
        }
        
        pos = content.find("\"y2\":", content.find("bounding_box"));
        if (pos != std::string::npos) {
            pos = content.find(":", pos) + 1;
            size_t end = content.find(",", pos);
            std::string value = content.substr(pos, end - pos);
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            data.bounding_box.y2 = std::stoi(value);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        cerr << "⚠️ Error parsing JSON file: " << e.what() << endl;
        return false;
    }
}

Eigen::Vector3f GetTrajectoryPointForPerson(RTMPData* pRTMPData, const StoppedPersonData& person_data)
{
    if (!person_data.has_camera_pose || !pRTMPData->pSLAM) {
        cout << "⚠️ No camera pose or SLAM available, using origin" << endl;
        return Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    }
    
    cout << "🎯 Calculating 3D position for person using proper depth projection:" << endl;
    cout << "   Rectangle: (" << person_data.bounding_box.x1 << ", " << person_data.bounding_box.y1 
         << ") to (" << person_data.bounding_box.x2 << ", " << person_data.bounding_box.y2 << ")" << endl;
    
    // Camera intrinsics
    float fx = 2622.0f;
    float fy = 2622.0f;
    float cx = person_data.frame_dimensions.width / 2.0f;
    float cy = person_data.frame_dimensions.height / 2.0f;
    
    // Camera pose: Tcw (world-to-camera transformation)
    Eigen::Matrix3f Rcw = person_data.camera_pose.rotationMatrix();
    Eigen::Vector3f tcw = person_data.camera_pose.translation();
    
    // Calculate camera position in world coordinates (inverse of Tcw)
    Eigen::Vector3f camera_world_pos = -Rcw.transpose() * tcw;
    
    // Calculate person's center in image coordinates
    float person_center_u = (person_data.bounding_box.x1 + person_data.bounding_box.x2) / 2.0f;
    float person_center_v = (person_data.bounding_box.y1 + person_data.bounding_box.y2) / 2.0f;
    
    // Get MapDrawer to find nearby MapPoints for depth estimation
    ORB_SLAM3::MapDrawer* map_drawer = pRTMPData->pSLAM->GetMapDrawer();
    if (!map_drawer) {
        cout << "⚠️ Cannot access MapDrawer, using default depth" << endl;
        // Use default depth if no MapPoints available
        float default_depth = 2.0f; // 2 meters default
        return CalculateWorldPositionFromPixelAndDepth(person_center_u, person_center_v, default_depth, 
                                                     fx, fy, cx, cy, Rcw, tcw);
    }
    
    // Export MapPoints to find depth information
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::string filename = "mappoints_depth_" + std::to_string(person_data.track_id) + 
                          "_" + std::to_string(timestamp) + ".txt";
    
    map_drawer->ExportMapPoints(filename, 0.0f);
    
    // Find the median depth of MapPoints near the person
    float estimated_depth = EstimateDepthFromMapPoints(filename, person_center_u, person_center_v, 
                                                      fx, fy, cx, cy, Rcw, tcw);
    
    cout << "📏 Estimated depth for person: " << estimated_depth << " meters" << endl;
    
    // Calculate 3D world position using ray casting from camera through person center
    Eigen::Vector3f world_position = CalculateWorldPositionFromPixelAndDepth(
        person_center_u, person_center_v, estimated_depth, fx, fy, cx, cy, Rcw, tcw);
    
    cout << "🌍 Calculated world position: (" << world_position.x() << ", " 
         << world_position.y() << ", " << world_position.z() << ")" << endl;
    
    return world_position;
}

// Helper function to estimate depth from nearby MapPoints
float EstimateDepthFromMapPoints(const std::string& mappoints_file, float pixel_u, float pixel_v,
                                float fx, float fy, float cx, float cy, 
                                const Eigen::Matrix3f& Rcw, const Eigen::Vector3f& tcw)
{
    std::vector<float> nearby_depths;
    std::ifstream file(mappoints_file);
    
    if (!file.is_open()) {
        cout << "⚠️ Cannot read MapPoints file for depth estimation, using default" << endl;
        return 2.0f; // Default 2 meters
    }
    
    std::string line;
    float search_radius = 100.0f; // 100 pixel search radius around person center
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        float x, y, z, timestamp;
        
        if (iss >> x >> y >> z) {
            // Transform world point to camera coordinates using Tcw
            Eigen::Vector3f world_pos(x, y, z);
            Eigen::Vector3f camera_pos = Rcw * world_pos + tcw;
            
            // Skip points behind camera
            if (camera_pos.z() > 0.1f) {
                // Project to image coordinates
                float u = fx * camera_pos.x() / camera_pos.z() + cx;
                float v = fy * camera_pos.y() / camera_pos.z() + cy;
                
                // Check if point is near the person's center
                float distance = sqrt((u - pixel_u) * (u - pixel_u) + (v - pixel_v) * (v - pixel_v));
                if (distance < search_radius) {
                    nearby_depths.push_back(camera_pos.z()); // Use camera Z as depth
                }
            }
        }
    }
    file.close();
    
    if (nearby_depths.empty()) {
        cout << "📏 No nearby MapPoints found, using default depth: 2.0m" << endl;
        return 2.0f;
    }
    
    // Calculate median depth for robustness
    std::sort(nearby_depths.begin(), nearby_depths.end());
    float median_depth = nearby_depths[nearby_depths.size() / 2];
    
    cout << "📏 Found " << nearby_depths.size() << " nearby MapPoints, median depth: " << median_depth << "m" << endl;
    return median_depth;
}

// Helper function to calculate world position from pixel coordinates and depth using proper ray casting
Eigen::Vector3f CalculateWorldPositionFromPixelAndDepth(float pixel_u, float pixel_v, float depth,
                                                       float fx, float fy, float cx, float cy,
                                                       const Eigen::Matrix3f& Rcw, const Eigen::Vector3f& tcw)
{
    // Convert pixel to normalized camera coordinates
    float x_norm = (pixel_u - cx) / fx;
    float y_norm = (pixel_v - cy) / fy;
    
    // Create ray in camera coordinates at the specified depth
    Eigen::Vector3f camera_point(x_norm * depth, y_norm * depth, depth);
    
    // Transform from camera coordinates to world coordinates using inverse of Tcw
    // Tcw transforms world to camera: Camera = Rcw * World + tcw
    // Therefore inverse (Twc) transforms camera to world: World = Rcw^T * Camera - Rcw^T * tcw
    Eigen::Vector3f world_point = Rcw.transpose() * camera_point - Rcw.transpose() * tcw;
    
    cout << "🎯 Ray cast from pixel (" << pixel_u << ", " << pixel_v << ") at depth " 
         << depth << "m to world (" << world_point.x() << ", " << world_point.y() << ", " << world_point.z() << ")" << endl;
    
    return world_point;
}

// Check proximity to stopped person landmarks and announce them
void CheckProximityToStoppedPersons(RTMPData* pRTMPData, const Sophus::SE3f& current_camera_pose)
{
    if (!pRTMPData->pSLAM) {
        return;
    }
    
    // Check for cooldown after recent announcement
    double current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    
    if (current_time - pRTMPData->last_announcement_time < pRTMPData->announcement_cooldown) {
        // In cooldown period, don't check distances
        return;
    }
    
    // Throttle proximity checks to every 0.5 seconds for responsive detection
    if (current_time - pRTMPData->last_proximity_check < 0.5) {
        return;
    }
    pRTMPData->last_proximity_check = current_time;
    
    // Get current camera position in world coordinates
    Eigen::Vector3f camera_world_pos = -current_camera_pose.rotationMatrix().transpose() * current_camera_pose.translation();
    
    // Get MapDrawer to access stopped person landmarks
    ORB_SLAM3::MapDrawer* map_drawer = pRTMPData->pSLAM->GetMapDrawer();
    if (!map_drawer) {
        return;
    }
    
    // Check distance to each stopped person landmark
    // Note: We need to access the landmarks from MapDrawer
    // For now, we'll implement a simple approach using the clustered points
    
    // Check if we're close to any existing clustered points (silent unless detection)
    for (const auto& point : pRTMPData->clustered_points) {
        float dx = camera_world_pos.x() - point.x;
        float dy = camera_world_pos.y() - point.y;
        float dz = camera_world_pos.z() - point.z;
        float distance = sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance < pRTMPData->proximity_threshold) {
            // Found a close landmark - announce it!
            std::string person_name = pRTMPData->last_stopped_person.person_name;
            if (person_name.empty()) {
                person_name = "Someone";  // Fallback if no name
            }
            
            cout << "\n🎆 DETECTION! Distance: " << std::fixed << std::setprecision(3) << distance << "m" << endl;
            cout << "🔊 ANNOUNCEMENT: " << person_name << " was here!" << endl;
            cout << "🎵 SPEAKING: \"" << person_name << " was here\"" << endl;
            cout << "🕒 Cooldown: 10 seconds...\n" << endl;
            
            // Use Windows built-in text-to-speech (same as system startup)
            std::string speech_message = person_name + " was here";
            std::string tts_command = "powershell -Command \"Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak('" + speech_message + "')\"";
            
            // Run TTS command in background
            std::system(tts_command.c_str());
            
            // Set cooldown timer
            pRTMPData->last_announcement_time = current_time;
            
            // Exit after first detection to avoid multiple announcements
            return;
        }
    }
}

void ClusterPointsInRectangle(RTMPData* pRTMPData, const StoppedPersonData& person_data)
{
    try {
        if (!pRTMPData->pSLAM) {
            cerr << "⚠️ SLAM system not available for point clustering" << endl;
            return;
        }
        
        cout << "🎯 Getting trajectory point for stopped person:" << endl;
        cout << "   Track ID: " << person_data.track_id << endl;
        if (!person_data.person_name.empty()) {
            cout << "   Person: " << person_data.person_name << endl;
        }
        cout << "   Rectangle: (" << person_data.bounding_box.x1 << ", " << person_data.bounding_box.y1 
             << ") to (" << person_data.bounding_box.x2 << ", " << person_data.bounding_box.y2 << ")" << endl;
        
        // Simple approach: just get the trajectory point and use it directly
        std::vector<cv::Point3f> clustered_points;
        
        // Get trajectory point with real X,Y,Z coordinates for sign placement
        Eigen::Vector3f trajectory_point = GetTrajectoryPointForPerson(pRTMPData, person_data);
        
        // Add the trajectory point directly - using real coordinates from camera pose
        clustered_points.push_back(cv::Point3f(
            trajectory_point[0],
            trajectory_point[1],  
            trajectory_point[2]
        ));
        
        cout << "✅ Using trajectory point: (" << trajectory_point[0] << ", " 
             << trajectory_point[1] << ", " << trajectory_point[2] << ")" << endl;
        
        if (!clustered_points.empty()) {
            pRTMPData->clustered_points = clustered_points;
            SaveClusteredPoints(clustered_points, person_data);
            
            // Add landmark to map viewer for visual display
            if (pRTMPData->pSLAM) {
                ORB_SLAM3::MapDrawer* map_drawer = pRTMPData->pSLAM->GetMapDrawer();
                if (map_drawer) {
                    map_drawer->AddStoppedPersonLandmark(clustered_points, person_data.person_name, person_data.track_id);
                    cout << "🎯 Added landmark to Pangolin map viewer!" << endl;
                    if (!person_data.person_name.empty()) {
                        cout << "👤 Person: " << person_data.person_name << endl;
                    }
                } else {
                    cout << "⚠️ Could not access map drawer for visualization" << endl;
                }
            }
        }
        
    } catch (const std::exception& e) {
        cerr << "⚠️ Error clustering points: " << e.what() << endl;
    }
}

void SaveClusteredPoints(const std::vector<cv::Point3f>& points, const StoppedPersonData& person_data)
{
    try {
        std::string filename = "clustered_points_track_" + std::to_string(person_data.track_id) + "_" + 
                              std::to_string(static_cast<int>(person_data.timestamp)) + ".txt";
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            cerr << "⚠️ Could not create clustered points file: " << filename << endl;
            return;
        }
        
        file << "# Clustered 3D points for stopped person" << std::endl;
        file << "# Track ID: " << person_data.track_id << std::endl;
        if (!person_data.person_name.empty()) {
            file << "# Person: " << person_data.person_name << std::endl;
        }
        file << "# Timestamp: " << person_data.timestamp << std::endl;
        file << "# Rectangle: (" << person_data.bounding_box.x1 << ", " << person_data.bounding_box.y1 
             << ") to (" << person_data.bounding_box.x2 << ", " << person_data.bounding_box.y2 << ")" << std::endl;
        file << "# Format: X Y Z" << std::endl;
        
        for (const auto& point : points) {
            file << std::fixed << std::setprecision(6) 
                 << point.x << " " << point.y << " " << point.z << std::endl;
        }
        
        file.close();
        
        cout << "💾 Saved " << points.size() << " clustered points to: " << filename << endl;
        cout << "🎯 These points mark where person stopped - can be used for navigation guidance" << endl;
        
    } catch (const std::exception& e) {
        cerr << "⚠️ Error saving clustered points: " << e.what() << endl;
    }
}

int main(int argc, char **argv)
{
    if(argc < 4) {
        cerr << endl << "Usage: ./mono_integration_vv path_to_vocabulary path_to_settings rtmp_url" << endl;
        cerr << "Example: ./mono_integration_vv Vocabulary/ORBvoc.txt Examples/Monocular/iPhone16Plus.yaml rtmp://localhost:1935/live/stream" << endl;
        return 1;
    }
    
    RTMPData rtmpData;
    
    // Set RTMP URL from command line argument
    rtmpData.rtmp_url = string(argv[3]);
    
    cout << "🎬 ORB-SLAM3 + Python Enhanced Hybrid Tracker Integration" << endl;
    cout << "📱 iPhone 16 Plus Camera Configuration" << endl;
    cout << "🌐 RTMP Stream: " << rtmpData.rtmp_url << endl;
    cout << "🐍 Python Person Detection: Will run in parallel" << endl;
    cout << "===================================================" << endl;
    
    // Initialize OpenCV VideoCapture for RTMP
    rtmpData.pCapture = new cv::VideoCapture();
    
    cout << "🔗 Connecting to RTMP stream..." << endl;
    
    // Try multiple connection methods with different backends
    vector<pair<string, int>> urls_and_backends = {
        {"http://localhost:8000/live/stream.flv", cv::CAP_FFMPEG},
        {"http://127.0.0.1:8000/live/stream.flv", cv::CAP_FFMPEG},
        {rtmpData.rtmp_url, cv::CAP_FFMPEG},
        {"rtmp://localhost:1935/live/stream", cv::CAP_FFMPEG},
        {"rtmp://127.0.0.1:1935/live/stream", cv::CAP_FFMPEG},
        {"http://localhost:8000/live/stream.flv", cv::CAP_ANY},
        {"http://127.0.0.1:8000/live/stream.flv", cv::CAP_ANY},
        {rtmpData.rtmp_url, cv::CAP_ANY},
        {"rtmp://localhost:1935/live/stream", cv::CAP_ANY},
        {"rtmp://127.0.0.1:1935/live/stream", cv::CAP_ANY}
    };
    
    bool connected = false;
    for (const auto& url_backend : urls_and_backends) {
        const string& url = url_backend.first;
        int backend = url_backend.second;
        string backend_name = (backend == cv::CAP_FFMPEG) ? "FFMPEG" : "ANY";
        
        cout << "🔄 Trying: " << url << " with " << backend_name << " backend" << endl;
        
        // Release previous capture
        rtmpData.pCapture->release();
        
        // Set FFMPEG-specific options for minimal latency
        if (backend == cv::CAP_FFMPEG) {
            rtmpData.pCapture->set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000); // 5 second timeout
            rtmpData.pCapture->set(cv::CAP_PROP_READ_TIMEOUT_MSEC, 1000);  // 1 second read timeout
            rtmpData.pCapture->set(cv::CAP_PROP_BUFFERSIZE, 0); // No buffering for minimal latency
            
            // Low-latency streaming options
            rtmpData.pCapture->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('H','2','6','4'));
            rtmpData.pCapture->set(cv::CAP_PROP_FORMAT, -1); // Auto format
            rtmpData.pCapture->set(cv::CAP_PROP_CONVERT_RGB, 0); // No RGB conversion
        }
        
        if (rtmpData.pCapture->open(url, backend)) {
            cout << "📡 Connection established, testing video data..." << endl;
            
            // Test if we can read frames with retries
            cv::Mat testFrame;
            bool frameReceived = false;
            for (int attempt = 0; attempt < 5; attempt++) {
                if (rtmpData.pCapture->read(testFrame) && !testFrame.empty()) {
                    frameReceived = true;
                    break;
                }
                cout << "⏳ Waiting for video data, attempt " << (attempt + 1) << "/5..." << endl;
                usleep(500000); // Wait 0.5 seconds
            }
            
            if (frameReceived) {
                rtmpData.rtmp_url = url;
                connected = true;
                cout << "✅ Successfully connected to: " << url << " with " << backend_name << endl;
                cout << "📹 Frame size: " << testFrame.cols << "x" << testFrame.rows << endl;
                break;
            } else {
                cout << "⚠️  Connected but no video data from: " << url << endl;
                rtmpData.pCapture->release();
            }
        } else {
            cout << "❌ Failed to connect to: " << url << " with " << backend_name << endl;
        }
        usleep(2000000); // Wait 2 seconds between attempts
    }
    
    if (!connected) {
        cerr << "❌ Failed to connect to any RTMP stream" << endl;
        cerr << "💡 Make sure your Node.js server is running and iPhone is streaming" << endl;
        cerr << "📱 iPhone should stream to: rtmp://[YOUR_MAC_IP]:1935/live/stream" << endl;
        return -1;
    }
    
    // Set capture properties for absolute minimum latency
    rtmpData.pCapture->set(cv::CAP_PROP_BUFFERSIZE, 0); // No buffering - real-time only
    rtmpData.pCapture->set(cv::CAP_PROP_FPS, 30); // Match iPhone stream FPS
    
    cout << "✅ Connected to RTMP stream successfully!" << endl;
    
    // Start Python Enhanced Hybrid Tracker in parallel
    bool python_started = StartPythonTracker(&rtmpData);
    if (!python_started) {
        cout << "⚠️  Python tracker failed to start, continuing with SLAM only" << endl;
    }
    
    // Create ORB-SLAM3 system with viewer enabled (same parameters as working version)
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true, 0, "");
    rtmpData.pSLAM = &SLAM;
    rtmpData.imageScale = SLAM.GetImageScale();
    
    cout << "=== SLAM + PYTHON PERSON DETECTION ====" << endl;
    cout << "REAL-TIME CONTROLS:" << endl;
    cout << "  'R' - Start recording current path" << endl;
    cout << "  'S' - Stop recording" << endl;
    cout << "  'L' - Load recorded path for guidance" << endl;
    cout << "  'G' - Start REAL-TIME spoken guidance" << endl;
    cout << "  'H' - Stop guidance" << endl;
    cout << "  'B' - Toggle backwards navigation mode" << endl;
    cout << "📱 Point your iPhone camera and start moving!" << endl;
    if (python_started) {
        cout << "🐍 Python Enhanced Hybrid Tracker will show person detection in separate window" << endl;
    }
    cout << "🔊 Audio navigation will guide you back along recorded paths" << endl;
    cout << "=========================================" << endl;
    
    // Start RTMP processing thread
#ifdef __APPLE__
    pthread_t rtmpThreadHandle;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    
    size_t stackSize = 16 * 1024 * 1024; // 16MB stack
    int stackResult = pthread_attr_setstacksize(&attr, stackSize);
    if (stackResult != 0) {
        cerr << "Failed to set stack size: " << stackResult << endl;
    }
    
    cout << "🎥 Starting real-time processing thread..." << endl;
    int result = pthread_create(&rtmpThreadHandle, &attr, RTMPProcessingThreadWrapper, &rtmpData);
    if (result != 0) {
        cerr << "❌ Failed to create RTMP thread: " << result << endl;
        StopPythonTracker(&rtmpData);
        return -1;
    }
    pthread_attr_destroy(&attr);
    
    // Main thread: Run viewer on main thread for macOS compatibility
    SLAM.RunViewerOnMainThread();
    
    // Wait for processing thread to complete
    pthread_join(rtmpThreadHandle, nullptr);
#else
    // Windows/Linux: Use std::thread
    cout << "🎥 Starting real-time processing thread..." << endl;
    std::thread rtmpThread(RTMPProcessingThread, &rtmpData);
    
    // Wait for processing thread
    rtmpThread.join();
#endif
    
    // Cleanup
    rtmpData.finished = true;
    
    // Stop Python tracker
    StopPythonTracker(&rtmpData);
    
    SLAM.Shutdown();
    
    if (rtmpData.pCapture) {
        rtmpData.pCapture->release();
        delete rtmpData.pCapture;
    }
    
    cout << "🏁 ORB-SLAM3 + Python tracker integration shutdown complete" << endl;
    return 0;
}