/**
* This file is part of ORB-SLAM3 Path Guidance System
* 
* PathRecorder: Records trajectory data during SLAM operation for later playback/guidance
*/

#ifndef PATHRECORDER_H
#define PATHRECORDER_H

#include <vector>
#include <string>
#include <mutex>
#include <fstream>
#include <queue>
#include <thread>
#include <atomic>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

namespace ORB_SLAM3
{

struct PathPoint {
    double timestamp;           // Frame timestamp
    Sophus::SE3f pose;         // Camera pose (translation + rotation)
    cv::Mat keyframe;          // Optional: store keyframe for visual verification
    int tracking_state;        // ORB-SLAM3 tracking state when recorded
    float tracking_confidence; // Confidence metric for this pose
    size_t index;              // Index in recorded path (for PathMatcher)
    
    PathPoint() : timestamp(0.0), tracking_state(0), tracking_confidence(0.0f), index(0) {}
    PathPoint(double ts, const Sophus::SE3f& p, int state, float conf = 1.0f, size_t idx = 0) 
        : timestamp(ts), pose(p), tracking_state(state), tracking_confidence(conf), index(idx) {}
};

class PathRecorder
{
public:
    PathRecorder();
    ~PathRecorder();
    
    // Recording control
    void StartRecording(const std::string& output_filename);
    void StopRecording();
    bool IsRecording() const { return recording_active_.load(); }
    
    // Add pose to recording queue
    void AddPose(double timestamp, const Sophus::SE3f& pose, int tracking_state, float confidence = 1.0f);
    void AddPose(const PathPoint& point);
    
    // Configuration
    void SetMinDistanceThreshold(float min_dist) { min_distance_threshold_ = min_dist; }
    void SetMinRotationThreshold(float min_rot) { min_rotation_threshold_ = min_rot; }
    void SetSaveKeyframes(bool save_kf) { save_keyframes_ = save_kf; }
    void SetMaxQueueSize(size_t max_size) { max_queue_size_ = max_size; }
    
    // Statistics
    size_t GetRecordedPointsCount() const { return recorded_points_count_.load(); }
    double GetRecordingDuration() const;
    std::string GetOutputFilename() const { return output_filename_; }
    
    // File operations
    static std::vector<PathPoint> LoadPath(const std::string& filename);
    static bool SavePath(const std::vector<PathPoint>& path, const std::string& filename);
    
private:
    // Recording state
    std::atomic<bool> recording_active_;
    std::atomic<bool> worker_should_stop_;
    std::atomic<size_t> recorded_points_count_;
    std::string output_filename_;
    
    // Pose filtering parameters
    float min_distance_threshold_;     // Minimum distance between recorded poses (meters)
    float min_rotation_threshold_;     // Minimum rotation between recorded poses (radians)
    bool save_keyframes_;              // Whether to save keyframe images
    
    // Threading
    std::thread worker_thread_;
    std::queue<PathPoint> pose_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    size_t max_queue_size_;
    
    // Filtering state
    PathPoint last_recorded_pose_;
    bool has_last_pose_;
    
    // Timing
    std::chrono::steady_clock::time_point recording_start_time_;
    
    // Worker thread function
    void WorkerThread();
    
    // Helper functions
    bool ShouldRecordPose(const PathPoint& new_pose) const;
    float ComputeDistance(const Sophus::SE3f& pose1, const Sophus::SE3f& pose2) const;
    float ComputeRotationAngle(const Sophus::SE3f& pose1, const Sophus::SE3f& pose2) const;
    void SavePoseToFile(const PathPoint& pose, std::ofstream& file) const;
    
    // File format helpers
    void WriteHeader(std::ofstream& file) const;
    static PathPoint ParsePathPoint(const std::string& line);
};

} // namespace ORB_SLAM3

#endif // PATHRECORDER_H