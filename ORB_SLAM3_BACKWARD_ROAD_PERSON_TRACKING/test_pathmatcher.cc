/**
 * Simple test program to verify PathMatcher functionality
 * Loads recorded path and tests nearest neighbor search
 */

#include "PathMatcher.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

using namespace ORB_SLAM3;

int main(int argc, char** argv)
{
    std::cout << "=== PathMatcher Test Program ===" << std::endl;
    
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path_file.txt>" << std::endl;
        std::cout << "Example: " << argv[0] << " recorded_path_1756539848.txt" << std::endl;
        return 1;
    }
    
    std::string path_file = argv[1];
    
    // Create PathMatcher instance
    PathMatcher matcher;
    
    // Test path loading
    std::cout << "\n1. Testing path loading..." << std::endl;
    if (!matcher.LoadPath(path_file)) {
        std::cerr << "Failed to load path: " << path_file << std::endl;
        return 1;
    }
    
    std::cout << "✅ Path loaded successfully!" << std::endl;
    std::cout << "Path length: " << matcher.GetPathLength() << " points" << std::endl;
    
    // Test nearest neighbor search with some poses from the path
    std::cout << "\n2. Testing nearest neighbor search..." << std::endl;
    
    // Create a test pose near the beginning of the path
    Sophus::SE3f test_pose;
    test_pose.translation() = Eigen::Vector3f(-0.3f, -0.1f, -0.2f);  // Near recorded start
    test_pose.setRotationMatrix(Eigen::Matrix3f::Identity());
    
    MatchResult result = matcher.FindNearestPoint(test_pose);
    
    std::cout << "Test pose: [" << test_pose.translation().transpose() << "]" << std::endl;
    std::cout << "Nearest point distance: " << result.distance << "m" << std::endl;
    std::cout << "Nearest point index: " << result.path_index << std::endl;
    std::cout << "Is on path: " << (result.is_on_path ? "Yes" : "No") << std::endl;
    
    // Test progress tracking
    std::cout << "\n3. Testing progress tracking..." << std::endl;
    matcher.UpdateProgress(result);
    std::cout << "Path progress: " << (matcher.GetPathProgress() * 100.0f) << "%" << std::endl;
    std::cout << "Current path index: " << matcher.GetCurrentPathIndex() << std::endl;
    
    // Test deviation calculation
    std::cout << "\n4. Testing deviation detection..." << std::endl;
    DeviationInfo deviation = matcher.CalculateDeviation(test_pose);
    std::cout << "Distance from path: " << deviation.distance_from_path << "m" << std::endl;
    std::cout << "Orientation deviation: " << (deviation.orientation_deviation * 180.0f / M_PI) << "°" << std::endl;
    std::cout << "Needs correction: " << (deviation.needs_correction ? "Yes" : "No") << std::endl;
    std::cout << "Correction hint: " << deviation.correction_hint << std::endl;
    
    // Test target waypoint
    std::cout << "\n5. Testing target waypoint..." << std::endl;
    PathPoint target = matcher.GetTargetWaypoint(5);
    std::cout << "Target waypoint (5 ahead): [" << target.pose.translation().transpose() << "]" << std::endl;
    std::cout << "Target index: " << target.index << std::endl;
    
    // Print overall statistics
    std::cout << "\n6. PathMatcher Statistics:" << std::endl;
    matcher.PrintMatchStatistics();
    
    // Test multiple random poses for performance
    std::cout << "\n7. Performance test (100 random queries)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        // Generate random test pose
        Sophus::SE3f random_pose;
        random_pose.translation() = Eigen::Vector3f(
            static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f,  // -10 to 10
            static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f,    // -1 to 1  
            static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f     // -2 to 2
        );
        random_pose.setRotationMatrix(Eigen::Matrix3f::Identity());
        
        MatchResult perf_result = matcher.FindNearestPoint(random_pose);
        matcher.UpdateProgress(perf_result);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "✅ 100 queries completed in " << duration.count() << " microseconds" << std::endl;
    std::cout << "Average query time: " << (duration.count() / 100.0f) << " microseconds" << std::endl;
    
    std::cout << "\n=== PathMatcher Test Complete ===" << std::endl;
    std::cout << "Status: " << matcher.GetStatusString() << std::endl;
    
    return 0;
}