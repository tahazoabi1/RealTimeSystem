/**
 * YOLOInterface.h
 * Bridge between ORB-SLAM3 and YOLO Python tracker
 * Provides shared memory communication for real-time frame processing
 */

#ifndef YOLO_INTERFACE_H
#define YOLO_INTERFACE_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <thread>

namespace ORB_SLAM3
{

// Structure to hold YOLO detection results
struct YOLODetection {
    int track_id;
    cv::Rect2f bbox;
    float confidence;
    std::string label;
    std::string person_name;  // From face recognition
    std::string activity;     // From activity detection
    cv::Point2f velocity;      // Movement velocity
    std::vector<cv::Point2f> trail;  // Movement trail
};

// Structure for zone information
struct Zone {
    std::string name;
    cv::Rect2f area;
    cv::Scalar color;
    int person_count;
};

class YOLOInterface {
public:
    YOLOInterface();
    ~YOLOInterface();
    
    // Initialize YOLO tracker connection
    bool Initialize(const std::string& config_path = "");
    
    // Process frame through YOLO tracker
    bool ProcessFrame(const cv::Mat& frame, double timestamp);
    
    // Get latest detection results
    std::vector<YOLODetection> GetDetections();
    
    // Get zone analytics
    std::vector<Zone> GetZones();
    
    // Check if YOLO processing is ready
    bool IsReady() const { return initialized_ && !processing_; }
    
    // Enable/disable YOLO processing
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_; }
    
    // Get performance metrics
    float GetProcessingTime() const { return last_processing_time_; }
    int GetTrackedCount() const { return static_cast<int>(current_detections_.size()); }
    
    // Shutdown YOLO tracker
    void Shutdown();
    
private:
    // Shared memory for frame exchange
    struct SharedMemory;
    std::unique_ptr<SharedMemory> shared_mem_;
    
    // Python process management
    void LaunchPythonTracker();
    void StopPythonTracker();
    std::thread python_thread_;
    void* python_process_;
    
    // Detection data
    std::vector<YOLODetection> current_detections_;
    std::vector<Zone> current_zones_;
    std::mutex detection_mutex_;
    
    // State management
    bool initialized_;
    bool enabled_;
    bool processing_;
    float last_processing_time_;
    
    // Configuration
    std::string config_path_;
    int frame_width_;
    int frame_height_;
    
    // Inter-process communication
    bool SendFrameToYOLO(const cv::Mat& frame, double timestamp);
    bool ReceiveDetectionsFromYOLO();
    
    // Shared memory key
    static constexpr const char* SHM_KEY = "ORB_SLAM3_YOLO_BRIDGE";
};

} // namespace ORB_SLAM3

#endif // YOLO_INTERFACE_H