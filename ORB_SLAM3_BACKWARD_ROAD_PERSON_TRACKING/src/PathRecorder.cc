/**
* This file is part of ORB-SLAM3 Path Guidance System
*/

#include "PathRecorder.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace ORB_SLAM3
{

PathRecorder::PathRecorder() 
    : recording_active_(false)
    , worker_should_stop_(false)
    , recorded_points_count_(0)
    , min_distance_threshold_(0.05f)    // 5cm minimum distance
    , min_rotation_threshold_(0.087f)   // ~5 degrees minimum rotation
    , save_keyframes_(false)
    , max_queue_size_(1000)
    , has_last_pose_(false)
{
}

PathRecorder::~PathRecorder()
{
    StopRecording();
}

void PathRecorder::StartRecording(const std::string& output_filename)
{
    if (recording_active_.load()) {
        std::cout << "PathRecorder: Already recording, stopping previous session..." << std::endl;
        StopRecording();
    }
    
    output_filename_ = output_filename;
    recording_active_.store(true);
    worker_should_stop_.store(false);
    recorded_points_count_.store(0);
    has_last_pose_ = false;
    
    // Clear any remaining poses in queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!pose_queue_.empty()) {
            pose_queue_.pop();
        }
    }
    
    recording_start_time_ = std::chrono::steady_clock::now();
    worker_thread_ = std::thread(&PathRecorder::WorkerThread, this);
    
    std::cout << "PathRecorder: Started recording to " << output_filename_ << std::endl;
}

void PathRecorder::StopRecording()
{
    if (!recording_active_.load()) {
        return;
    }
    
    recording_active_.store(false);
    worker_should_stop_.store(true);
    
    // Wake up worker thread
    queue_cv_.notify_all();
    
    // Wait for worker thread to finish
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    
    std::cout << "PathRecorder: Stopped recording. Saved " << recorded_points_count_.load() 
              << " poses to " << output_filename_ << std::endl;
    std::cout << "PathRecorder: Recording duration: " << GetRecordingDuration() << " seconds" << std::endl;
}

void PathRecorder::AddPose(double timestamp, const Sophus::SE3f& pose, int tracking_state, float confidence)
{
    AddPose(PathPoint(timestamp, pose, tracking_state, confidence));
}

void PathRecorder::AddPose(const PathPoint& point)
{
    if (!recording_active_.load()) {
        return;
    }
    
    // Check if queue is full
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (pose_queue_.size() >= max_queue_size_) {
            std::cerr << "PathRecorder: Warning - pose queue full, dropping pose" << std::endl;
            return;
        }
        
        pose_queue_.push(point);
    }
    
    queue_cv_.notify_one();
}

double PathRecorder::GetRecordingDuration() const
{
    if (!recording_active_.load() && recorded_points_count_.load() == 0) {
        return 0.0;
    }
    
    auto current_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - recording_start_time_);
    return duration.count() / 1000.0;
}

void PathRecorder::WorkerThread()
{
    std::ofstream output_file(output_filename_);
    if (!output_file.is_open()) {
        std::cerr << "PathRecorder: Error - Cannot open output file " << output_filename_ << std::endl;
        recording_active_.store(false);
        return;
    }
    
    WriteHeader(output_file);
    
    while (!worker_should_stop_.load() || !pose_queue_.empty()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for poses in queue or stop signal
        queue_cv_.wait(lock, [this]() { 
            return !pose_queue_.empty() || worker_should_stop_.load(); 
        });
        
        // Process all poses in queue
        while (!pose_queue_.empty()) {
            PathPoint pose = pose_queue_.front();
            pose_queue_.pop();
            lock.unlock();
            
            // Apply filtering - only record if pose has moved sufficiently
            if (ShouldRecordPose(pose)) {
                SavePoseToFile(pose, output_file);
                last_recorded_pose_ = pose;
                has_last_pose_ = true;
                recorded_points_count_++;
            }
            
            lock.lock();
        }
    }
    
    output_file.close();
    std::cout << "PathRecorder: Worker thread finished" << std::endl;
}

bool PathRecorder::ShouldRecordPose(const PathPoint& new_pose) const
{
    // Always record first pose
    if (!has_last_pose_) {
        return true;
    }
    
    // Skip poses with poor tracking
    if (new_pose.tracking_state != 2) { // 2 = OK tracking in ORB-SLAM3
        return false;
    }
    
    // Check distance and rotation thresholds
    float distance = ComputeDistance(last_recorded_pose_.pose, new_pose.pose);
    float rotation = ComputeRotationAngle(last_recorded_pose_.pose, new_pose.pose);
    
    return (distance >= min_distance_threshold_) || (rotation >= min_rotation_threshold_);
}

float PathRecorder::ComputeDistance(const Sophus::SE3f& pose1, const Sophus::SE3f& pose2) const
{
    Eigen::Vector3f translation_diff = pose2.translation() - pose1.translation();
    return translation_diff.norm();
}

float PathRecorder::ComputeRotationAngle(const Sophus::SE3f& pose1, const Sophus::SE3f& pose2) const
{
    // Compute relative rotation and extract angle
    Sophus::SO3f relative_rotation = pose2.so3() * pose1.so3().inverse();
    return relative_rotation.log().norm();
}

void PathRecorder::WriteHeader(std::ofstream& file) const
{
    file << "# ORB-SLAM3 Path Recording" << std::endl;
    file << "# Format: timestamp tx ty tz qx qy qz qw tracking_state confidence" << std::endl;
    file << "# Translation in meters, quaternion (x,y,z,w), timestamp in seconds" << std::endl;
    file << "# Coordinate system: Camera frame" << std::endl;
}

void PathRecorder::SavePoseToFile(const PathPoint& pose, std::ofstream& file) const
{
    // Extract translation and quaternion from SE3
    Eigen::Vector3f translation = pose.pose.translation();
    Eigen::Quaternionf quaternion = pose.pose.unit_quaternion();
    
    // Save in TUM format with additional metadata
    file << std::fixed << std::setprecision(6)
         << pose.timestamp << " "
         << translation.x() << " " << translation.y() << " " << translation.z() << " "
         << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << " " << quaternion.w() << " "
         << pose.tracking_state << " " << pose.tracking_confidence << std::endl;
}

std::vector<PathPoint> PathRecorder::LoadPath(const std::string& filename)
{
    std::vector<PathPoint> path;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "PathRecorder: Error - Cannot open file " << filename << std::endl;
        return path;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        try {
            PathPoint point = ParsePathPoint(line);
            path.push_back(point);
        } catch (const std::exception& e) {
            std::cerr << "PathRecorder: Warning - Failed to parse line: " << line << std::endl;
        }
    }
    
    file.close();
    std::cout << "PathRecorder: Loaded " << path.size() << " poses from " << filename << std::endl;
    return path;
}

PathPoint PathRecorder::ParsePathPoint(const std::string& line)
{
    std::istringstream iss(line);
    PathPoint point;
    
    float tx, ty, tz, qx, qy, qz, qw;
    
    if (!(iss >> point.timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
        throw std::runtime_error("Invalid pose format");
    }
    
    // Optional fields (tracking state and confidence)
    if (!(iss >> point.tracking_state)) {
        point.tracking_state = 2; // Default to OK tracking
    }
    if (!(iss >> point.tracking_confidence)) {
        point.tracking_confidence = 1.0f; // Default confidence
    }
    
    // Construct SE3 pose from translation and quaternion
    Eigen::Vector3f translation(tx, ty, tz);
    Eigen::Quaternionf quaternion(qw, qx, qy, qz); // Note: Eigen quaternion constructor is (w,x,y,z)
    quaternion.normalize();
    
    point.pose = Sophus::SE3f(quaternion, translation);
    
    return point;
}

bool PathRecorder::SavePath(const std::vector<PathPoint>& path, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "PathRecorder: Error - Cannot open output file " << filename << std::endl;
        return false;
    }
    
    // Create temporary PathRecorder just for header writing
    PathRecorder temp_recorder;
    temp_recorder.WriteHeader(file);
    
    for (const auto& point : path) {
        temp_recorder.SavePoseToFile(point, file);
    }
    
    file.close();
    std::cout << "PathRecorder: Saved " << path.size() << " poses to " << filename << std::endl;
    return true;
}

} // namespace ORB_SLAM3