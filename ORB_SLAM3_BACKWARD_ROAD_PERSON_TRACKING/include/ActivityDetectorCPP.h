/**
 * ActivityDetectorCPP.h
 * Exact port of modules/activity_detector.py
 * Activity detection and tracking with movement analysis
 */

#ifndef ACTIVITY_DETECTOR_CPP_H
#define ACTIVITY_DETECTOR_CPP_H

#include "orbslam3_export.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <deque>
#include <chrono>

namespace ORB_SLAM3 {

struct ActivityHistoryEntry {
    double timestamp;
    std::string activity;
    std::string candidate_activity;
    float confidence;
    float candidate_confidence;
    float speed;
    float aspect_ratio;
    float height;
};

struct MovementDataCPP {
    std::vector<cv::Point2f> positions;
    std::vector<cv::Point2f> velocities;
    std::vector<float> speeds;
    std::vector<float> directions;
    std::deque<ActivityHistoryEntry> history;
    std::string current_activity;
    float activity_confidence;
    std::string stable_activity;
    std::chrono::steady_clock::time_point last_update;
};

class ORB_SLAM3_API ActivityDetectorCPP {
public:
    ActivityDetectorCPP();
    ~ActivityDetectorCPP();
    
    // Exact port of Python ActivityDetector methods
    bool initialize(const std::map<std::string, float>& config = {});
    void update_track_activity(int track_id, const cv::Rect2f& bbox, double timestamp);
    std::string get_track_activity(int track_id) const;
    float get_activity_confidence(int track_id) const;
    std::string get_stable_activity(int track_id) const;
    cv::Point2f get_track_velocity(int track_id) const;
    float get_track_speed(int track_id) const;
    
    // Statistics and analytics (exact Python interface)
    std::map<std::string, int> get_activity_statistics() const;
    std::vector<ActivityHistoryEntry> get_track_history(int track_id) const;
    void cleanup_track(int track_id);
    void cleanup_old_tracks(double current_time);
    
    // Configuration methods
    void set_movement_threshold(float threshold) { movement_threshold_ = threshold; }
    void set_history_size(int size) { history_size_ = size; }
    bool is_available() const { return initialized_; }
    
private:
    // Core activity analysis methods
    std::string analyze_activity(int track_id, const cv::Rect2f& bbox, double timestamp);
    void update_movement_history(int track_id, const cv::Point2f& center, double timestamp);
    std::string classify_activity(const std::vector<float>& speeds, const cv::Rect2f& bbox);
    float calculate_confidence(const std::string& activity, const std::vector<float>& speeds);
    
    // Movement calculation methods
    cv::Point2f calculate_velocity(const std::vector<cv::Point2f>& positions, double time_delta);
    float calculate_speed(const cv::Point2f& velocity);
    float calculate_direction(const cv::Point2f& velocity);
    float calculate_average_speed(const std::vector<float>& speeds, int window_size = 5);
    
    // Data management methods
    void trim_history(int track_id);
    bool validate_bbox(const cv::Rect2f& bbox);
    
private:
    // Movement tracking data (exact Python data structures)
    std::map<int, MovementDataCPP> track_data_;
    
    // Configuration (exact Python defaults)
    float movement_threshold_;
    float stop_threshold_;
    float moving_threshold_;
    int history_size_;
    int velocity_window_size_;
    int activity_stability_frames_;
    float confidence_threshold_;
    
    // Activity classification thresholds (exact Python values)
    float slow_speed_threshold_;
    float medium_speed_threshold_;
    float fast_speed_threshold_;
    float aspect_ratio_threshold_;
    
    // State
    bool initialized_;
    std::chrono::steady_clock::time_point last_cleanup_;
    
    // Constants (exact Python values)
    static const float DEFAULT_MOVEMENT_THRESHOLD;
    static const float DEFAULT_STOP_THRESHOLD;
    static const float DEFAULT_MOVING_THRESHOLD;
    static const int DEFAULT_HISTORY_SIZE;
    static const int DEFAULT_VELOCITY_WINDOW;
    static const int DEFAULT_STABILITY_FRAMES;
    static const float DEFAULT_CONFIDENCE_THRESHOLD;
    static const int CLEANUP_INTERVAL_SECONDS;
};

} // namespace ORB_SLAM3

#endif // ACTIVITY_DETECTOR_CPP_H