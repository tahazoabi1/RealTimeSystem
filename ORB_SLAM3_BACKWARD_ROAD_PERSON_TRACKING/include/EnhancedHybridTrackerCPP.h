/**
 * EnhancedHybridTrackerCPP.h
 * Exact port of enhanced_hybrid_tracker_modular.py
 * Integrates all modules: DeviceManager, ActivityDetector, Visualizer, ZoneAnalytics, FaceRecognizer
 */

#ifndef ENHANCED_HYBRID_TRACKER_CPP_H
#define ENHANCED_HYBRID_TRACKER_CPP_H

#include "orbslam3_export.h"
#include "DeviceManagerCPP.h"
#include "ActivityDetectorCPP.h"
#include "VisualizerCPP.h"
#include "ZoneAnalyticsCPP.h"
#include "FaceRecognizerCPP.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <chrono>

namespace ORB_SLAM3 {

struct TrackerConfigCPP {
    // Device configuration (exact Python config)
    float confidence_threshold = 0.4f;
    float nms_threshold = 0.5f;
    bool use_yolo = true;
    
    // Activity detection configuration
    float movement_threshold = 2.0f;
    float stop_threshold = 3.0f;
    float moving_threshold = 5.0f;
    int history_size = 30;
    
    // Visualization configuration
    bool show_trails = true;
    bool show_zones = true;
    bool show_ids = true;
    bool show_activities = true;
    bool show_face_names = true;
    int trail_length = 30;
    
    // Zone analytics configuration
    double dwell_time_threshold = 5.0;
    double transition_timeout = 2.0;
    
    // Face recognition configuration
    float face_distance_threshold = 0.6f;
    std::string face_model_path = "face_recognition.onnx";
    std::string face_database_path = "face_database.dat";
    
    // Performance configuration
    int frame_skip_interval = 3;
    bool enable_face_recognition = true;
    bool enable_zone_analytics = true;
};

struct TrackerDetectionCPP {
    cv::Rect2f bbox;
    float confidence;
    int track_id;
    std::string activity;
    std::string face_name;
    cv::Point2f velocity;
    std::string current_zone;
    std::vector<cv::Point2f> trail_points;
    std::chrono::steady_clock::time_point last_seen;
};

class ORB_SLAM3_API EnhancedHybridTrackerCPP {
public:
    EnhancedHybridTrackerCPP();
    ~EnhancedHybridTrackerCPP();
    
    // Main interface (exact Python interface)
    bool Initialize(const TrackerConfigCPP& config = TrackerConfigCPP{});
    std::vector<TrackerDetectionCPP> ProcessFrame(const cv::Mat& frame, double timestamp);
    cv::Mat GetAnnotatedFrame() const;
    
    // Module access methods (exact Python module access)
    DeviceManagerCPP* GetDeviceManager() { return device_manager_.get(); }
    ActivityDetectorCPP* GetActivityDetector() { return activity_detector_.get(); }
    VisualizerCPP* GetVisualizer() { return visualizer_.get(); }
    ZoneAnalyticsCPP* GetZoneAnalytics() { return zone_analytics_.get(); }
    FaceRecognizerCPP* GetFaceRecognizer() { return face_recognizer_.get(); }
    
    // Configuration methods
    void SetConfig(const TrackerConfigCPP& config) { config_ = config; }
    TrackerConfigCPP GetConfig() const { return config_; }
    void UpdateConfig(const std::string& key, float value);
    
    // Statistics and performance (exact Python statistics)
    float GetFPS() const;
    float GetProcessingTime() const;
    int GetActiveTrackCount() const;
    std::map<std::string, int> GetActivityStatistics() const;
    std::map<std::string, int> GetZoneOccupancyCounts() const;
    
    // Session management (exact Python session)
    void ResetSession();
    void ExportAnalytics(const std::string& filename) const;
    bool SaveFaceDatabase(const std::string& filename = "") const;
    bool LoadFaceDatabase(const std::string& filename = "");
    
    // Real-time controls (exact Python controls)
    void ToggleVisualization(const std::string& element);
    void AddZone(const std::string& name, const cv::Rect2f& area, const cv::Scalar& color);
    void RemoveZone(const std::string& name);
    
    // Error handling and diagnostics
    std::vector<std::string> GetActiveAlerts() const;
    std::string GetSystemStatus() const;
    bool IsInitialized() const { return initialized_; }
    
private:
    // Core tracking methods (exact Python tracking logic)
    std::vector<TrackerDetectionCPP> DetectAndTrack(const cv::Mat& frame, double timestamp);
    void UpdateTracking(const std::vector<YOLODetectionCPP>& detections, double timestamp);
    void AssignTrackIDs(std::vector<YOLODetectionCPP>& detections);
    void UpdateActivityAnalysis(const std::vector<TrackerDetectionCPP>& detections, double timestamp);
    void UpdateZoneAnalytics(const std::vector<TrackerDetectionCPP>& detections, double timestamp);
    void UpdateFaceRecognition(const cv::Mat& frame, std::vector<TrackerDetectionCPP>& detections);
    
    // Tracking state management (exact Python state management)
    struct TrackStateCPP {
        cv::Rect2f bbox;
        int lost_frames;
        int age;
        double created_at;
        double last_seen;
        std::string activity;
        std::string face_name;
        cv::Point2f velocity;
        std::string current_zone;
    };
    
    void CreateNewTrack(const YOLODetectionCPP& detection, double timestamp);
    void UpdateExistingTrack(int track_id, const YOLODetectionCPP& detection, double timestamp);
    void RemoveOldTracks(double timestamp);
    float CalculateIOU(const cv::Rect2f& box1, const cv::Rect2f& box2);
    
    // Performance optimization methods (exact Python optimization)
    bool ShouldProcessFrame();
    void OptimizePerformance();
    void CleanupOldData(double timestamp);
    
private:
    // Module instances (exact Python module architecture)
    std::unique_ptr<DeviceManagerCPP> device_manager_;
    std::unique_ptr<ActivityDetectorCPP> activity_detector_;
    std::unique_ptr<VisualizerCPP> visualizer_;
    std::unique_ptr<ZoneAnalyticsCPP> zone_analytics_;
    std::unique_ptr<FaceRecognizerCPP> face_recognizer_;
    
    // Configuration
    TrackerConfigCPP config_;
    
    // Tracking state (exact Python tracking state)
    std::map<int, TrackStateCPP> active_tracks_;
    int next_track_id_;
    
    // Performance tracking (exact Python performance tracking)
    std::chrono::steady_clock::time_point last_frame_time_;
    std::chrono::steady_clock::time_point processing_start_time_;
    float processing_time_ms_;
    float fps_;
    int frame_count_;
    int frames_processed_;
    
    // Frame processing control (exact Python frame control)
    int frame_skip_counter_;
    cv::Mat last_annotated_frame_;
    
    // State flags
    bool initialized_;
    bool face_recognition_available_;
    bool zone_analytics_available_;
    
    // Constants (exact Python constants)
    static const float DEFAULT_IOU_THRESHOLD;
    static const int MAX_LOST_FRAMES;
    static const int TRACK_CLEANUP_INTERVAL;
    static const int FACE_RECOGNITION_INTERVAL;
    static const double TRACK_TIMEOUT_SECONDS;
};

} // namespace ORB_SLAM3

#endif // ENHANCED_HYBRID_TRACKER_CPP_H