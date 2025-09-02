/**
 * ActivityDetectorCPP.cc
 * Exact port of modules/activity_detector.py implementation
 */

#include "ActivityDetectorCPP.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace ORB_SLAM3 {

// Constants (exact Python values)
const float ActivityDetectorCPP::DEFAULT_MOVEMENT_THRESHOLD = 2.0f;
const float ActivityDetectorCPP::DEFAULT_STOP_THRESHOLD = 3.0f;
const float ActivityDetectorCPP::DEFAULT_MOVING_THRESHOLD = 5.0f;
const int ActivityDetectorCPP::DEFAULT_HISTORY_SIZE = 30;
const int ActivityDetectorCPP::DEFAULT_VELOCITY_WINDOW = 5;
const int ActivityDetectorCPP::DEFAULT_STABILITY_FRAMES = 5;
const float ActivityDetectorCPP::DEFAULT_CONFIDENCE_THRESHOLD = 0.7f;
const int ActivityDetectorCPP::CLEANUP_INTERVAL_SECONDS = 30;

ActivityDetectorCPP::ActivityDetectorCPP()
    : movement_threshold_(DEFAULT_MOVEMENT_THRESHOLD)
    , stop_threshold_(DEFAULT_STOP_THRESHOLD)
    , moving_threshold_(DEFAULT_MOVING_THRESHOLD)
    , history_size_(DEFAULT_HISTORY_SIZE)
    , velocity_window_size_(DEFAULT_VELOCITY_WINDOW)
    , activity_stability_frames_(DEFAULT_STABILITY_FRAMES)
    , confidence_threshold_(DEFAULT_CONFIDENCE_THRESHOLD)
    , slow_speed_threshold_(1.0f)
    , medium_speed_threshold_(3.0f)
    , fast_speed_threshold_(6.0f)
    , aspect_ratio_threshold_(0.4f)
    , initialized_(false)
{
    last_cleanup_ = std::chrono::steady_clock::now();
}

ActivityDetectorCPP::~ActivityDetectorCPP() = default;

bool ActivityDetectorCPP::initialize(const std::map<std::string, float>& config)
{
    std::cout << "🔧 Initializing ActivityDetector C++..." << std::endl;
    
    // Load configuration (exact Python config loading)
    if (config.find("movement_threshold") != config.end()) {
        movement_threshold_ = config.at("movement_threshold");
    }
    if (config.find("stop_threshold") != config.end()) {
        stop_threshold_ = config.at("stop_threshold");
    }
    if (config.find("moving_threshold") != config.end()) {
        moving_threshold_ = config.at("moving_threshold");
    }
    if (config.find("history_size") != config.end()) {
        history_size_ = static_cast<int>(config.at("history_size"));
    }
    
    initialized_ = true;
    std::cout << "✅ ActivityDetector C++ initialized" << std::endl;
    return true;
}

void ActivityDetectorCPP::update_track_activity(int track_id, const cv::Rect2f& bbox, double timestamp)
{
    if (!initialized_ || !validate_bbox(bbox)) {
        return;
    }
    
    // Initialize track data if new (exact Python initialization)
    if (track_data_.find(track_id) == track_data_.end()) {
        MovementDataCPP& data = track_data_[track_id];
        data.current_activity = "Unknown";
        data.activity_confidence = 0.0f;
        data.stable_activity = "Unknown";
        data.last_update = std::chrono::steady_clock::now();
    }
    
    // Update movement history (exact Python logic)
    cv::Point2f center(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height * 0.5f);
    update_movement_history(track_id, center, timestamp);
    
    // Analyze activity (exact Python analysis)
    std::string new_activity = analyze_activity(track_id, bbox, timestamp);
    
    MovementDataCPP& data = track_data_[track_id];
    data.current_activity = new_activity;
    data.last_update = std::chrono::steady_clock::now();
    
    // Update history entry (exact Python history format)
    ActivityHistoryEntry entry;
    entry.timestamp = timestamp;
    entry.activity = data.current_activity;
    entry.candidate_activity = new_activity;
    entry.confidence = data.activity_confidence;
    entry.candidate_confidence = calculate_confidence(new_activity, data.speeds);
    entry.speed = data.speeds.empty() ? 0.0f : data.speeds.back();
    entry.aspect_ratio = bbox.width / bbox.height;
    entry.height = bbox.height;
    
    data.history.push_back(entry);
    
    // Trim history to size (exact Python trimming)
    while (data.history.size() > static_cast<size_t>(history_size_)) {
        data.history.pop_front();
    }
    
    // Update stable activity (exact Python stability logic)
    if (data.history.size() >= static_cast<size_t>(activity_stability_frames_)) {
        int same_activity_count = 0;
        for (int i = data.history.size() - activity_stability_frames_; i < data.history.size(); ++i) {
            if (data.history[i].activity == new_activity) {
                same_activity_count++;
            }
        }
        
        if (same_activity_count >= activity_stability_frames_) {
            data.stable_activity = new_activity;
        }
    }
}

std::string ActivityDetectorCPP::analyze_activity(int track_id, const cv::Rect2f& bbox, double timestamp)
{
    MovementDataCPP& data = track_data_[track_id];
    
    if (data.speeds.empty()) {
        return "Unknown";
    }
    
    // Calculate activity based on movement (exact Python algorithm)
    std::string activity = classify_activity(data.speeds, bbox);
    float confidence = calculate_confidence(activity, data.speeds);
    
    data.activity_confidence = confidence;
    return activity;
}

void ActivityDetectorCPP::update_movement_history(int track_id, const cv::Point2f& center, double timestamp)
{
    MovementDataCPP& data = track_data_[track_id];
    
    // Add position (exact Python position tracking)
    data.positions.push_back(center);
    
    // Calculate velocity if we have previous positions (exact Python calculation)
    if (data.positions.size() >= 2) {
        cv::Point2f velocity = calculate_velocity(data.positions, 1.0/30.0); // Assume 30fps
        data.velocities.push_back(velocity);
        
        float speed = calculate_speed(velocity);
        data.speeds.push_back(speed);
        
        float direction = calculate_direction(velocity);
        data.directions.push_back(direction);
    }
    
    // Trim vectors to history size (exact Python trimming)
    while (data.positions.size() > static_cast<size_t>(history_size_)) {
        data.positions.erase(data.positions.begin());
    }
    while (data.velocities.size() > static_cast<size_t>(history_size_)) {
        data.velocities.erase(data.velocities.begin());
    }
    while (data.speeds.size() > static_cast<size_t>(history_size_)) {
        data.speeds.erase(data.speeds.begin());
    }
    while (data.directions.size() > static_cast<size_t>(history_size_)) {
        data.directions.erase(data.directions.begin());
    }
}

std::string ActivityDetectorCPP::classify_activity(const std::vector<float>& speeds, const cv::Rect2f& bbox)
{
    if (speeds.empty()) {
        return "Unknown";
    }
    
    // Calculate movement statistics (exact Python calculations)
    float recent_avg_speed = calculate_average_speed(speeds, std::min(5, static_cast<int>(speeds.size())));
    float overall_avg_speed = calculate_average_speed(speeds, speeds.size());
    float max_speed = *std::max_element(speeds.begin(), speeds.end());
    
    // Activity classification with exact Python thresholds and logic
    if (recent_avg_speed > stop_threshold_ || max_speed > moving_threshold_ || overall_avg_speed > movement_threshold_) {
        return "Moving";
    } else {
        return "Stop";
    }
}

float ActivityDetectorCPP::calculate_confidence(const std::string& activity, const std::vector<float>& speeds)
{
    if (speeds.empty()) {
        return 0.0f;
    }
    
    float avg_speed = calculate_average_speed(speeds, speeds.size());
    
    // Confidence calculation (exact Python logic)
    if (activity == "Moving") {
        return std::min(0.9f, avg_speed / moving_threshold_);
    } else if (activity == "Stop") {
        return std::max(0.1f, 1.0f - (avg_speed / stop_threshold_));
    }
    
    return 0.5f; // Default confidence
}

cv::Point2f ActivityDetectorCPP::calculate_velocity(const std::vector<cv::Point2f>& positions, double time_delta)
{
    if (positions.size() < 2) {
        return cv::Point2f(0, 0);
    }
    
    // Simple velocity calculation (exact Python calculation)
    cv::Point2f current = positions.back();
    cv::Point2f previous = positions[positions.size() - 2];
    
    return (current - previous) / time_delta;
}

float ActivityDetectorCPP::calculate_speed(const cv::Point2f& velocity)
{
    // Euclidean distance (exact Python calculation)
    return std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
}

float ActivityDetectorCPP::calculate_direction(const cv::Point2f& velocity)
{
    // Angle in radians (exact Python calculation)
    return std::atan2(velocity.y, velocity.x);
}

float ActivityDetectorCPP::calculate_average_speed(const std::vector<float>& speeds, int window_size)
{
    if (speeds.empty()) {
        return 0.0f;
    }
    
    int start_idx = std::max(0, static_cast<int>(speeds.size()) - window_size);
    float sum = 0.0f;
    int count = 0;
    
    for (int i = start_idx; i < speeds.size(); ++i) {
        sum += speeds[i];
        count++;
    }
    
    return count > 0 ? sum / count : 0.0f;
}

std::string ActivityDetectorCPP::get_track_activity(int track_id) const
{
    auto it = track_data_.find(track_id);
    return (it != track_data_.end()) ? it->second.current_activity : "Unknown";
}

float ActivityDetectorCPP::get_activity_confidence(int track_id) const
{
    auto it = track_data_.find(track_id);
    return (it != track_data_.end()) ? it->second.activity_confidence : 0.0f;
}

std::string ActivityDetectorCPP::get_stable_activity(int track_id) const
{
    auto it = track_data_.find(track_id);
    return (it != track_data_.end()) ? it->second.stable_activity : "Unknown";
}

cv::Point2f ActivityDetectorCPP::get_track_velocity(int track_id) const
{
    auto it = track_data_.find(track_id);
    if (it != track_data_.end() && !it->second.velocities.empty()) {
        return it->second.velocities.back();
    }
    return cv::Point2f(0, 0);
}

float ActivityDetectorCPP::get_track_speed(int track_id) const
{
    auto it = track_data_.find(track_id);
    if (it != track_data_.end() && !it->second.speeds.empty()) {
        return it->second.speeds.back();
    }
    return 0.0f;
}

std::map<std::string, int> ActivityDetectorCPP::get_activity_statistics() const
{
    std::map<std::string, int> stats;
    
    // Count activities (exact Python statistics)
    for (const auto& pair : track_data_) {
        const std::string& activity = pair.second.current_activity;
        stats[activity]++;
    }
    
    return stats;
}

std::vector<ActivityHistoryEntry> ActivityDetectorCPP::get_track_history(int track_id) const
{
    auto it = track_data_.find(track_id);
    if (it != track_data_.end()) {
        return std::vector<ActivityHistoryEntry>(it->second.history.begin(), it->second.history.end());
    }
    return {};
}

void ActivityDetectorCPP::cleanup_track(int track_id)
{
    track_data_.erase(track_id);
}

void ActivityDetectorCPP::cleanup_old_tracks(double current_time)
{
    auto now = std::chrono::steady_clock::now();
    
    // Only cleanup if interval has passed (exact Python cleanup logic)
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_cleanup_).count() < CLEANUP_INTERVAL_SECONDS) {
        return;
    }
    
    // Remove tracks not updated recently
    auto it = track_data_.begin();
    while (it != track_data_.end()) {
        auto track_age = std::chrono::duration_cast<std::chrono::seconds>(now - it->second.last_update).count();
        if (track_age > CLEANUP_INTERVAL_SECONDS) {
            it = track_data_.erase(it);
        } else {
            ++it;
        }
    }
    
    last_cleanup_ = now;
}

bool ActivityDetectorCPP::validate_bbox(const cv::Rect2f& bbox)
{
    return bbox.width > 0 && bbox.height > 0 && bbox.area() > 100; // Minimum area threshold
}

} // namespace ORB_SLAM3