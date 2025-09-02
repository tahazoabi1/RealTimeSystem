/**
 * Exact port of enhanced_hybrid_tracker_modular.py algorithm
 * NO modifications - exactly matching the Python data structures and methods
 */

#ifndef YOLO_DETECTOR_CPP_H
#define YOLO_DETECTOR_CPP_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <chrono>
#include <deque>

namespace ORB_SLAM3
{

// Exact port of Python YOLODetectionCPP structure
struct YOLODetectionCPP {
    cv::Rect2f bbox;
    float confidence;
    std::string label;
    int track_id;
    std::string person_name;
    std::string activity;
    cv::Point2f velocity;
    std::vector<cv::Point2f> trail;
    std::chrono::steady_clock::time_point last_seen;
};

// Exact port of Python ZoneCPP structure
struct ZoneCPP {
    std::string name;
    cv::Rect2f area;
    cv::Scalar color;
    int person_count;
    std::vector<int> person_ids;
};

class YOLODetectorCPP {
public:
    YOLODetectorCPP();
    ~YOLODetectorCPP();
    
    // Main interface matching Python
    bool Initialize(const std::string& config_path = "");
    std::vector<YOLODetectionCPP> DetectAndTrack(const cv::Mat& frame, double timestamp);
    std::vector<ZoneCPP> GetZones() const;
    
    // Performance metrics (simplified)
    float GetProcessingTime() const;
    float GetFPS() const;
    int GetTrackedCount() const;

private:
    // Exact port of Python TrackData structure
    struct TrackData {
        cv::Rect2f bbox;
        int lost;
        int age;
        double created_at;
        double last_seen;
    };
    
    // Exact port of Python PostureHistoryEntry
    struct PostureHistoryEntry {
        double timestamp;
        std::string activity;
        std::string candidate_activity;
        float confidence;
        float candidate_confidence;
        float speed;
        float aspect_ratio;
        float height;
    };
    
    // Exact port of Python ZoneAnalytics structure
    struct ZoneAnalytics {
        std::set<int> current_occupancy;
        int total_visits;
        double total_time_spent;
        double average_dwell_time;
        int peak_occupancy;
        double last_entry_time;
    };
    
    // Exact port of Python SessionStats structure
    struct SessionStats {
        int total_detections;
        std::vector<float> average_speed;
        std::map<std::string, int> zone_visits;
        std::vector<std::map<std::string, std::string>> activity_log;
    };

private:
    // Core tracking parameters (exact Python values)
    float confidence_threshold_;
    int camera_index_;
    bool show_fps_;
    
    // Core tracking data (exact Python structures)
    std::map<int, TrackData> tracks_;
    int next_track_id_;
    std::map<int, int> track_face_attempts_;
    std::map<int, std::string> track_face_ids_;
    int max_face_attempts_;
    
    // Performance optimizations (exact Python values)
    int frame_skip_counter_;
    int face_recognition_interval_;
    int memory_cleanup_counter_;
    int max_tracks_;
    
    // Activity detector data (exact Python ActivityDetector)
    int movement_history_size_;
    std::map<int, std::vector<cv::Point2f>> track_positions_;
    std::map<int, std::vector<cv::Point2f>> track_velocities_;
    std::map<int, std::vector<float>> track_directions_;
    std::map<int, std::vector<float>> track_speeds_;
    std::map<int, std::string> track_activities_;
    std::map<int, std::vector<PostureHistoryEntry>> track_posture_history_;
    std::map<int, float> activity_confidence_;
    std::map<int, std::string> track_stable_activity_;
    
    // Visualizer data (exact Python Visualizer)
    int trail_length_;
    std::map<int, cv::Scalar> track_colors_;
    std::vector<cv::Scalar> base_colors_;
    bool show_zones_;
    bool show_advanced_info_;
    
    // Zone analytics (exact Python ZoneAnalytics)
    std::vector<ZoneCPP> zones_;
    std::vector<ZoneCPP> default_zones_;
    std::map<std::string, ZoneAnalytics> zone_analytics_;
    std::map<std::string, double> zone_entry_times_;
    
    // Session statistics (exact Python)
    SessionStats session_stats_;
    
    // Face recognition (exact Python flags)
    bool face_recognition_enabled_;
    
    // HOG detector (exact Python fallback)
    cv::HOGDescriptor hog_;
    
    // Initialization flag
    bool initialized_;

private:
    // Core methods - exact Python ports
    void detect_persons_hog(const cv::Mat& frame, std::vector<cv::Rect2f>& detections, std::vector<float>& confidences);
    
    // Tracking methods - exact Python ports
    std::vector<std::pair<int, TrackData>> update_advanced_tracking(const std::vector<cv::Rect2f>& detections, double current_time);
    float compute_iou(const cv::Rect2f& box1, const cv::Rect2f& box2);
    void update_track_motion(int track_id, const cv::Rect2f& bbox, double current_time);
    void analyze_activity(int track_id, const cv::Rect2f& bbox, double current_time);
    void create_new_track(const cv::Rect2f& bbox, double current_time);
    std::vector<std::pair<int, TrackData>> get_active_tracks();
    void cleanup_track(int track_id);
    void cleanup_old_data();
    
    // Activity methods - exact Python ports
    std::string get_track_activity(int track_id) const;
    cv::Point2f get_track_velocity(int track_id) const;
    
    // Face recognition methods - exact Python ports
    std::string get_face_name(int track_id) const;
    
    // Zone methods - exact Python ports
    void update_zone_analytics();
    bool point_in_zone(const cv::Point2f& point, const cv::Rect2f& zone_bbox) const;
    
    // Visualization methods - exact Python ports
    void generate_colors();
    cv::Scalar get_track_color(int track_id);
};

} // namespace ORB_SLAM3

#endif // YOLO_DETECTOR_CPP_H