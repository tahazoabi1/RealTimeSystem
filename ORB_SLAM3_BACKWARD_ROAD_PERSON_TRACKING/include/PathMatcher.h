/**
 * PathMatcher.h
 * 
 * Path matching and localization system for ORB-SLAM3 guidance
 * Implements nearest neighbor search, progress tracking, and deviation detection
 * Part of Phase 4: Path Matching & Localization
 */

#ifndef PATHMATCHER_H
#define PATHMATCHER_H

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include "sophus/se3.hpp"
#include <opencv2/opencv.hpp>
#include "PathRecorder.h"  // For PathPoint definition

namespace ORB_SLAM3
{

// PathPoint defined in PathRecorder.h

struct MatchResult {
    PathPoint nearest_point;
    float distance;
    float orientation_error;  // Angular difference in radians
    size_t path_index;
    bool is_on_path;  // Within tolerance
    bool is_moving_forward;  // Direction along path
};

struct DeviationInfo {
    float distance_from_path;
    float orientation_deviation;
    bool needs_correction;
    std::string correction_hint;  // "Turn left", "Turn right", "Go straight"
};

class PathMatcher
{
public:
    PathMatcher();
    ~PathMatcher();
    
    // Path loading and management
    bool LoadPath(const std::string& path_filename);
    bool IsPathLoaded() const;
    size_t GetPathLength() const;
    void ClearPath();
    const std::vector<PathPoint>& GetLoadedPath() const;
    
    // Core matching functionality
    MatchResult FindNearestPoint(const Sophus::SE3f& current_pose);
    DeviationInfo CalculateDeviation(const Sophus::SE3f& current_pose);
    
    // Progress tracking
    void UpdateProgress(const MatchResult& match);
    float GetPathProgress() const;  // 0.0 to 1.0
    size_t GetCurrentPathIndex() const;
    bool IsPathCompleted() const;
    
    // Configuration
    void SetPositionTolerance(float tolerance_meters);
    void SetOrientationTolerance(float tolerance_radians);
    void SetProgressSmoothing(float smoothing_factor);
    
    // Path analysis
    PathPoint GetTargetWaypoint(size_t lookahead_points = 5) const;
    Sophus::SE3f GetExpectedPose() const;
    std::vector<PathPoint> GetNearbyPoints(const Sophus::SE3f& pose, float radius) const;
    
    // Debugging and status
    std::string GetStatusString() const;
    void PrintMatchStatistics() const;

private:
    // Path data storage
    std::vector<PathPoint> path_points_;
    mutable std::mutex path_mutex_;
    
    // KD-tree implementation (simplified for pose positions)
    struct KDNode {
        PathPoint point;
        std::unique_ptr<KDNode> left;
        std::unique_ptr<KDNode> right;
        int axis;  // 0=x, 1=y, 2=z splitting axis
    };
    std::unique_ptr<KDNode> kdtree_root_;
    
    // Progress tracking state
    size_t current_path_index_;
    float path_progress_;  // 0.0 to 1.0
    bool path_completed_;
    
    // Configuration parameters
    float position_tolerance_;     // meters
    float orientation_tolerance_;  // radians
    float progress_smoothing_;     // 0.0 to 1.0
    
    // Statistics
    size_t total_matches_;
    float average_match_distance_;
    
    // Internal helper methods
    void BuildKDTree();
    std::unique_ptr<KDNode> BuildKDTreeRecursive(std::vector<PathPoint>& points, int depth);
    MatchResult SearchKDTree(const Sophus::SE3f& query_pose) const;
    MatchResult SearchKDTreeRecursive(const KDNode* node, const Sophus::SE3f& query_pose, 
                                     int depth, MatchResult& best_match) const;
    
    // Pose utility functions
    float CalculateDistance(const Sophus::SE3f& pose1, const Sophus::SE3f& pose2) const;
    float CalculateOrientationError(const Sophus::SE3f& pose1, const Sophus::SE3f& pose2) const;
    bool IsMovingForward(const PathPoint& current, const PathPoint& previous) const;
    
    // Path progress helpers
    void UpdatePathProgress(size_t new_index);
    DeviationInfo AnalyzeDeviation(const MatchResult& match) const;
    std::string GenerateCorrectionHint(const DeviationInfo& deviation) const;
    
    // File I/O helpers
    bool ParseTUMFormat(const std::string& line, PathPoint& point, size_t index);
};

} // namespace ORB_SLAM3

#endif // PATHMATCHER_H