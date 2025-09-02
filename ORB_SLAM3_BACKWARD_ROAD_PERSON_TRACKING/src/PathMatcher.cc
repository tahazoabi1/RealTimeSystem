/**
 * PathMatcher.cc
 * 
 * Implementation of path matching and localization system
 * Provides nearest neighbor search, progress tracking, and deviation detection
 */

#include "PathMatcher.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

#ifdef _WIN32
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#endif

namespace ORB_SLAM3
{

PathMatcher::PathMatcher()
    : current_path_index_(0)
    , path_progress_(0.0f)
    , path_completed_(false)
    , position_tolerance_(0.5f)      // 50cm default tolerance
    , orientation_tolerance_(0.26f)   // ~15 degrees default tolerance
    , progress_smoothing_(0.7f)       // Smooth progress updates
    , total_matches_(0)
    , average_match_distance_(0.0f)
{
    std::cout << "PathMatcher: Initialized with position tolerance: " 
              << position_tolerance_ << "m, orientation tolerance: " 
              << orientation_tolerance_ * 180.0f / M_PI << "°" << std::endl;
}

PathMatcher::~PathMatcher()
{
    ClearPath();
}

bool PathMatcher::LoadPath(const std::string& path_filename)
{
    std::lock_guard<std::mutex> lock(path_mutex_);
    
    std::ifstream file(path_filename);
    if (!file.is_open()) {
        std::cerr << "PathMatcher: Failed to open path file: " << path_filename << std::endl;
        return false;
    }
    
    path_points_.clear();
    std::string line;
    size_t line_number = 0;
    size_t valid_points = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        PathPoint point;
        if (ParseTUMFormat(line, point, valid_points)) {
            path_points_.push_back(point);
            valid_points++;
        } else {
            std::cerr << "PathMatcher: Warning - Invalid line " << line_number 
                      << " in path file" << std::endl;
        }
    }
    
    file.close();
    
    if (path_points_.empty()) {
        std::cerr << "PathMatcher: No valid path points loaded" << std::endl;
        return false;
    }
    
    // Build KD-tree for fast nearest neighbor search
    BuildKDTree();
    
    // Reset progress tracking
    current_path_index_ = 0;
    path_progress_ = 0.0f;
    path_completed_ = false;
    total_matches_ = 0;
    average_match_distance_ = 0.0f;
    
    std::cout << "PathMatcher: Successfully loaded " << path_points_.size() 
              << " path points from " << path_filename << std::endl;
    
    return true;
}

bool PathMatcher::IsPathLoaded() const
{
    std::lock_guard<std::mutex> lock(path_mutex_);
    return !path_points_.empty();
}

size_t PathMatcher::GetPathLength() const
{
    std::lock_guard<std::mutex> lock(path_mutex_);
    return path_points_.size();
}

void PathMatcher::ClearPath()
{
    std::lock_guard<std::mutex> lock(path_mutex_);
    path_points_.clear();
    kdtree_root_.reset();
    current_path_index_ = 0;
    path_progress_ = 0.0f;
    path_completed_ = false;
}

const std::vector<PathPoint>& PathMatcher::GetLoadedPath() const
{
    std::lock_guard<std::mutex> lock(path_mutex_);
    return path_points_;
}

MatchResult PathMatcher::FindNearestPoint(const Sophus::SE3f& current_pose)
{
    std::lock_guard<std::mutex> lock(path_mutex_);
    
    if (path_points_.empty()) {
        MatchResult empty_result = {};
        empty_result.distance = std::numeric_limits<float>::max();
        empty_result.is_on_path = false;
        return empty_result;
    }
    
    // Use KD-tree for efficient nearest neighbor search
    MatchResult result = SearchKDTree(current_pose);
    
    // Update statistics
    total_matches_++;
    average_match_distance_ = (average_match_distance_ * (total_matches_ - 1) + result.distance) / total_matches_;
    
    // Determine if moving forward along path
    if (current_path_index_ < path_points_.size() && result.path_index > 0) {
        result.is_moving_forward = (result.path_index >= current_path_index_);
    } else {
        result.is_moving_forward = true;
    }
    
    // Check if within tolerance
    result.is_on_path = (result.distance <= position_tolerance_) && 
                       (result.orientation_error <= orientation_tolerance_);
    
    return result;
}

DeviationInfo PathMatcher::CalculateDeviation(const Sophus::SE3f& current_pose)
{
    MatchResult match = FindNearestPoint(current_pose);
    return AnalyzeDeviation(match);
}

void PathMatcher::UpdateProgress(const MatchResult& match)
{
    if (match.path_index >= path_points_.size()) {
        return;
    }
    
    // Smooth progress updates to avoid jitter
    size_t new_index = match.path_index;
    if (match.is_moving_forward || new_index > current_path_index_) {
        current_path_index_ = static_cast<size_t>(
            progress_smoothing_ * current_path_index_ + 
            (1.0f - progress_smoothing_) * new_index
        );
    }
    
    UpdatePathProgress(current_path_index_);
}

float PathMatcher::GetPathProgress() const
{
    return path_progress_;
}

size_t PathMatcher::GetCurrentPathIndex() const
{
    return current_path_index_;
}

bool PathMatcher::IsPathCompleted() const
{
    return path_completed_;
}

void PathMatcher::SetPositionTolerance(float tolerance_meters)
{
    position_tolerance_ = std::max(0.1f, tolerance_meters);
    std::cout << "PathMatcher: Position tolerance set to " << position_tolerance_ << "m" << std::endl;
}

void PathMatcher::SetOrientationTolerance(float tolerance_radians)
{
    orientation_tolerance_ = std::max(0.05f, tolerance_radians);
    std::cout << "PathMatcher: Orientation tolerance set to " 
              << orientation_tolerance_ * 180.0f / M_PI << "°" << std::endl;
}

void PathMatcher::SetProgressSmoothing(float smoothing_factor)
{
    progress_smoothing_ = std::max(0.0f, std::min(1.0f, smoothing_factor));
}

PathPoint PathMatcher::GetTargetWaypoint(size_t lookahead_points) const
{
    std::lock_guard<std::mutex> lock(path_mutex_);
    
    if (path_points_.empty()) {
        return PathPoint{};
    }
    
    size_t target_index = std::min(current_path_index_ + lookahead_points, path_points_.size() - 1);
    return path_points_[target_index];
}

Sophus::SE3f PathMatcher::GetExpectedPose() const
{
    std::lock_guard<std::mutex> lock(path_mutex_);
    
    if (path_points_.empty() || current_path_index_ >= path_points_.size()) {
        return Sophus::SE3f();
    }
    
    return path_points_[current_path_index_].pose;
}

std::vector<PathPoint> PathMatcher::GetNearbyPoints(const Sophus::SE3f& pose, float radius) const
{
    std::lock_guard<std::mutex> lock(path_mutex_);
    std::vector<PathPoint> nearby_points;
    
    for (const auto& point : path_points_) {
        if (CalculateDistance(pose, point.pose) <= radius) {
            nearby_points.push_back(point);
        }
    }
    
    return nearby_points;
}

// Private implementation methods

void PathMatcher::BuildKDTree()
{
    if (path_points_.empty()) {
        return;
    }
    
    std::vector<PathPoint> points_copy = path_points_;
    kdtree_root_ = BuildKDTreeRecursive(points_copy, 0);
    
    std::cout << "PathMatcher: Built KD-tree for " << path_points_.size() << " points" << std::endl;
}

std::unique_ptr<PathMatcher::KDNode> PathMatcher::BuildKDTreeRecursive(
    std::vector<PathPoint>& points, int depth)
{
    if (points.empty()) {
        return nullptr;
    }
    
    if (points.size() == 1) {
        auto node = std::make_unique<KDNode>();
        node->point = points[0];
        node->axis = depth % 3;
        return node;
    }
    
    // Choose splitting axis (cycle through x, y, z)
    int axis = depth % 3;
    
    // Sort points by the chosen axis
    std::sort(points.begin(), points.end(), [axis](const PathPoint& a, const PathPoint& b) {
        Eigen::Vector3f pos_a = a.pose.translation();
        Eigen::Vector3f pos_b = b.pose.translation();
        return pos_a[axis] < pos_b[axis];
    });
    
    // Find median
    size_t median = points.size() / 2;
    
    auto node = std::make_unique<KDNode>();
    node->point = points[median];
    node->axis = axis;
    
    // Recursively build left and right subtrees
    std::vector<PathPoint> left_points(points.begin(), points.begin() + median);
    std::vector<PathPoint> right_points(points.begin() + median + 1, points.end());
    
    node->left = BuildKDTreeRecursive(left_points, depth + 1);
    node->right = BuildKDTreeRecursive(right_points, depth + 1);
    
    return node;
}

MatchResult PathMatcher::SearchKDTree(const Sophus::SE3f& query_pose) const
{
    if (!kdtree_root_) {
        MatchResult empty_result = {};
        empty_result.distance = std::numeric_limits<float>::max();
        return empty_result;
    }
    
    MatchResult best_match;
    best_match.distance = std::numeric_limits<float>::max();
    
    SearchKDTreeRecursive(kdtree_root_.get(), query_pose, 0, best_match);
    
    return best_match;
}

MatchResult PathMatcher::SearchKDTreeRecursive(const KDNode* node, const Sophus::SE3f& query_pose, 
                                              int depth, MatchResult& best_match) const
{
    if (!node) {
        return best_match;
    }
    
    // Check current node
    float distance = CalculateDistance(query_pose, node->point.pose);
    if (distance < best_match.distance) {
        best_match.nearest_point = node->point;
        best_match.distance = distance;
        best_match.orientation_error = CalculateOrientationError(query_pose, node->point.pose);
        best_match.path_index = node->point.index;
    }
    
    // Determine which side to search first
    Eigen::Vector3f query_pos = query_pose.translation();
    Eigen::Vector3f node_pos = node->point.pose.translation();
    int axis = node->axis;
    
    const KDNode* near_side = (query_pos[axis] < node_pos[axis]) ? node->left.get() : node->right.get();
    const KDNode* far_side = (query_pos[axis] < node_pos[axis]) ? node->right.get() : node->left.get();
    
    // Search near side first
    SearchKDTreeRecursive(near_side, query_pose, depth + 1, best_match);
    
    // Check if we need to search far side
    float axis_distance = std::abs(query_pos[axis] - node_pos[axis]);
    if (axis_distance < best_match.distance) {
        SearchKDTreeRecursive(far_side, query_pose, depth + 1, best_match);
    }
    
    return best_match;
}

float PathMatcher::CalculateDistance(const Sophus::SE3f& pose1, const Sophus::SE3f& pose2) const
{
    Eigen::Vector3f pos1 = pose1.translation();
    Eigen::Vector3f pos2 = pose2.translation();
    return (pos1 - pos2).norm();
}

float PathMatcher::CalculateOrientationError(const Sophus::SE3f& pose1, const Sophus::SE3f& pose2) const
{
    Sophus::SE3f relative_pose = pose1.inverse() * pose2;
    return relative_pose.so3().log().norm();
}

bool PathMatcher::IsMovingForward(const PathPoint& current, const PathPoint& previous) const
{
    return current.index > previous.index;
}

void PathMatcher::UpdatePathProgress(size_t new_index)
{
    if (path_points_.empty()) {
        return;
    }
    
    path_progress_ = static_cast<float>(new_index) / static_cast<float>(path_points_.size() - 1);
    path_progress_ = std::max(0.0f, std::min(1.0f, path_progress_));
    
    // Check if path is completed (within 90% of the end)
    path_completed_ = (path_progress_ > 0.9f);
}

DeviationInfo PathMatcher::AnalyzeDeviation(const MatchResult& match) const
{
    DeviationInfo deviation;
    deviation.distance_from_path = match.distance;
    deviation.orientation_deviation = match.orientation_error;
    deviation.needs_correction = !match.is_on_path;
    deviation.correction_hint = GenerateCorrectionHint(deviation);
    
    return deviation;
}

std::string PathMatcher::GenerateCorrectionHint(const DeviationInfo& deviation) const
{
    if (!deviation.needs_correction) {
        return "On path";
    }
    
    if (deviation.distance_from_path > position_tolerance_) {
        return "Move closer to path";
    }
    
    if (deviation.orientation_deviation > orientation_tolerance_) {
        // Simplified direction hint - could be enhanced with more sophisticated logic
        return "Adjust orientation";
    }
    
    return "Continue forward";
}

bool PathMatcher::ParseTUMFormat(const std::string& line, PathPoint& point, size_t index)
{
    std::istringstream iss(line);
    double timestamp, tx, ty, tz, qx, qy, qz, qw;
    int tracking_state = 2;  // Default to good tracking
    float confidence = 1.0f; // Default confidence
    
    // Parse standard TUM format: timestamp tx ty tz qx qy qz qw
    if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
        return false;
    }
    
    // Try to parse additional fields if present
    iss >> tracking_state >> confidence;
    
    // Build pose from translation and quaternion
    Eigen::Vector3f translation(tx, ty, tz);
    Eigen::Quaternionf quaternion(qw, qx, qy, qz);
    quaternion.normalize();
    
    point.timestamp = timestamp;
    point.pose = Sophus::SE3f(quaternion, translation);
    point.tracking_state = tracking_state;
    point.tracking_confidence = confidence;
    point.index = index;
    
    return true;
}

std::string PathMatcher::GetStatusString() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "PathMatcher Status: ";
    
    if (!IsPathLoaded()) {
        oss << "No path loaded";
    } else {
        oss << path_points_.size() << " points, " 
            << (path_progress_ * 100.0f) << "% complete, "
            << "avg distance: " << average_match_distance_ << "m";
    }
    
    return oss.str();
}

void PathMatcher::PrintMatchStatistics() const
{
    std::cout << "=== PathMatcher Statistics ===" << std::endl;
    std::cout << "Path points: " << path_points_.size() << std::endl;
    std::cout << "Total matches: " << total_matches_ << std::endl;
    std::cout << "Average match distance: " << average_match_distance_ << "m" << std::endl;
    std::cout << "Current progress: " << (path_progress_ * 100.0f) << "%" << std::endl;
    std::cout << "Path completed: " << (path_completed_ ? "Yes" : "No") << std::endl;
    std::cout << "Position tolerance: " << position_tolerance_ << "m" << std::endl;
    std::cout << "Orientation tolerance: " << (orientation_tolerance_ * 180.0f / M_PI) << "°" << std::endl;
    std::cout << "=============================" << std::endl;
}

} // namespace ORB_SLAM3