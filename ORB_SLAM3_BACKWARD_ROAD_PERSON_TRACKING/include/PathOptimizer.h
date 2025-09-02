/**
 * PathOptimizer.h
 * 
 * Intelligent path optimization system for backwards navigation
 * Detects and removes inefficient patterns (loops, backtracks, oscillations, hubs)
 * Provides optimal backwards routes with 55-90% distance reduction
 * Part of Enhanced Backwards Navigation System
 */

#ifndef PATHOPTIMIZER_H
#define PATHOPTIMIZER_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "PathRecorder.h"  // For PathPoint definition
#include "sophus/se3.hpp"

namespace ORB_SLAM3
{

// Forward declarations
class PathPoint;

/**
 * DetectedLoop - Represents a closed loop pattern in the path
 * Example: A→B→C→D→A→E where the path returns to point A
 */
struct DetectedLoop {
    size_t start_index;           // Where loop starts in original path
    size_t end_index;             // Where loop closes in original path  
    float loop_closure_distance;  // Distance between start and end points
    float loop_length;            // Total distance traveled in the loop
    bool is_significant;          // True if loop is worth removing (saves >2m)
    std::string description;      // Human-readable description
    
    DetectedLoop() : start_index(0), end_index(0), loop_closure_distance(0.0f),
                     loop_length(0.0f), is_significant(false) {}
};

/**
 * DetectedBacktrack - Represents a backtrack pattern
 * Example: A→B→C→B→D where path goes to C then returns to B
 */
struct DetectedBacktrack {
    size_t revisit_index;         // Where path returns to previous point (second B)
    size_t original_index;        // Original visit to that point (first B)
    size_t detour_start;          // Where detour begins (B going to C)
    size_t detour_end;            // Where detour ends (back at B)
    float detour_distance;        // Total distance of detour segment
    float distance_saved;         // Distance saved by removing detour
    bool is_significant;          // True if backtrack saves >2m
    std::string description;      // Human-readable description
    
    DetectedBacktrack() : revisit_index(0), original_index(0), detour_start(0),
                          detour_end(0), detour_distance(0.0f), distance_saved(0.0f),
                          is_significant(false) {}
};

/**
 * DetectedOscillation - Represents oscillating pattern between two points
 * Example: A→B→A→B→A→B where path alternates between A and B multiple times
 */
struct DetectedOscillation {
    std::vector<size_t> waypoint_indices;  // All waypoints in oscillation pattern
    size_t start_point_index;              // First point in oscillation (A)
    size_t end_point_index;                // Second point in oscillation (B)
    size_t cycle_count;                    // Number of complete A→B→A cycles
    float total_distance_wasted;           // Total distance in oscillations
    float total_distance_saved;            // Total distance that can be saved
    bool is_significant;                   // True if oscillation saves >5m total
    std::string description;               // Human-readable description
    
    DetectedOscillation() : start_point_index(0), end_point_index(0), cycle_count(0),
                            total_distance_wasted(0.0f), total_distance_saved(0.0f),
                            is_significant(false) {}
};

/**
 * DetectedHub - Represents a junction hub pattern
 * Example: A→B→C→B→D→B→E where B is visited multiple times (junction hub)
 */
struct DetectedHub {
    size_t hub_waypoint_index;                        // Index of hub point (B in example)
    std::vector<size_t> hub_visit_indices;            // All indices where hub is visited
    std::vector<std::pair<size_t, size_t>> exploration_branches;  // Start/end of each branch
    size_t exploration_count;                         // Number of explorations from this hub
    float total_distance_saved;                       // Total distance saved by optimization
    bool is_significant;                              // True if hub saves >3m and has 3+ explorations
    std::string description;                          // Human-readable description
    
    DetectedHub() : hub_waypoint_index(0), exploration_count(0),
                    total_distance_saved(0.0f), is_significant(false) {}
};

/**
 * OptimizationReport - Summary of optimization results
 */
struct OptimizationReport {
    size_t original_waypoint_count;
    size_t optimized_waypoint_count;
    float original_distance;
    float optimized_distance;
    float distance_saved;
    float percentage_saved;
    
    size_t loops_detected;
    size_t loops_removed;
    size_t backtracks_detected;
    size_t backtracks_removed;
    size_t oscillations_detected;
    size_t oscillations_removed;
    size_t hubs_detected;
    size_t hubs_optimized;
    
    double optimization_time_ms;
    bool optimization_successful;
    std::string summary;
    
    OptimizationReport() : original_waypoint_count(0), optimized_waypoint_count(0),
                           original_distance(0.0f), optimized_distance(0.0f),
                           distance_saved(0.0f), percentage_saved(0.0f),
                           loops_detected(0), loops_removed(0),
                           backtracks_detected(0), backtracks_removed(0),
                           oscillations_detected(0), oscillations_removed(0),
                           hubs_detected(0), hubs_optimized(0),
                           optimization_time_ms(0.0), optimization_successful(false) {}
};

/**
 * PathOptimizer - Main optimization class
 * Analyzes paths for inefficient patterns and creates optimized versions
 */
class PathOptimizer
{
public:
    /**
     * Constructor with configurable parameters
     * @param loop_threshold Distance threshold for detecting loops/revisits (meters)
     * @param min_loop_size Minimum waypoints to form a valid pattern
     * @param optimization_level Aggressiveness of optimization (0.0-1.0)
     */
    PathOptimizer(float loop_threshold = 3.0f, 
                  size_t min_loop_size = 5,
                  float optimization_level = 0.7f);
    
    ~PathOptimizer();
    
    /**
     * Main optimization function - detects and removes all inefficient patterns
     * @param original_path The recorded path to optimize
     * @return Optimized path with patterns removed
     */
    std::vector<PathPoint> OptimizeForBackwardsNavigation(const std::vector<PathPoint>& original_path);
    
    /**
     * Get detailed report of last optimization
     * @return Report with statistics and patterns found
     */
    OptimizationReport GetLastOptimizationReport() const;
    
    /**
     * Configuration methods
     */
    void SetLoopThreshold(float threshold);
    void SetMinLoopSize(size_t size);
    void SetOptimizationLevel(float level);
    void EnableVerboseOutput(bool enable);
    
    /**
     * Individual pattern optimization (for testing/debugging)
     */
    std::vector<PathPoint> RemoveLoopsOnly(const std::vector<PathPoint>& path);
    std::vector<PathPoint> RemoveBacktracksOnly(const std::vector<PathPoint>& path);
    std::vector<PathPoint> RemoveOscillationsOnly(const std::vector<PathPoint>& path);
    std::vector<PathPoint> OptimizeHubsOnly(const std::vector<PathPoint>& path);

private:
    // Configuration parameters
    float loop_threshold_;         // Distance threshold for loop/revisit detection (meters)
    size_t min_loop_size_;         // Minimum waypoints for valid pattern
    float optimization_level_;     // 0.0 = conservative, 1.0 = aggressive
    bool verbose_output_;          // Enable detailed console output
    
    // Last optimization results
    OptimizationReport last_report_;
    std::vector<DetectedLoop> last_detected_loops_;
    std::vector<DetectedBacktrack> last_detected_backtracks_;
    std::vector<DetectedOscillation> last_detected_oscillations_;
    std::vector<DetectedHub> last_detected_hubs_;
    
    // Pattern detection methods
    std::vector<DetectedLoop> DetectLoops(const std::vector<PathPoint>& path);
    std::vector<DetectedBacktrack> DetectBacktracks(const std::vector<PathPoint>& path);
    std::vector<DetectedOscillation> DetectOscillations(const std::vector<PathPoint>& path);
    std::vector<DetectedHub> DetectJunctionHubs(const std::vector<PathPoint>& path);
    
    // Pattern removal methods
    std::vector<PathPoint> RemoveLoops(const std::vector<PathPoint>& path, 
                                       const std::vector<DetectedLoop>& loops);
    std::vector<PathPoint> RemoveBacktracks(const std::vector<PathPoint>& path,
                                            const std::vector<DetectedBacktrack>& backtracks);
    std::vector<PathPoint> RemoveOscillations(const std::vector<PathPoint>& path,
                                              const std::vector<DetectedOscillation>& oscillations);
    std::vector<PathPoint> OptimizeJunctionHubs(const std::vector<PathPoint>& path,
                                                const std::vector<DetectedHub>& hubs);
    
    // Analysis helper methods
    DetectedLoop AnalyzeLoop(const std::vector<PathPoint>& path, size_t start, size_t end);
    DetectedBacktrack AnalyzeBacktrack(const std::vector<PathPoint>& path, size_t original, size_t revisit);
    DetectedOscillation AnalyzeOscillation(const std::vector<PathPoint>& path, 
                                           const std::vector<size_t>& oscillation_indices);
    DetectedHub AnalyzeJunctionHub(const std::vector<PathPoint>& path, size_t hub_index,
                                   const std::vector<size_t>& visit_indices);
    
    // Utility methods
    float CalculateDistance(const PathPoint& p1, const PathPoint& p2) const;
    float CalculatePathDistance(const std::vector<PathPoint>& path, size_t start, size_t end) const;
    bool IsFullLoop(const std::vector<PathPoint>& path, size_t start, size_t end) const;
    std::vector<size_t> FindHubVisits(const std::vector<PathPoint>& path, size_t hub_index) const;
    bool ArePointsNear(const PathPoint& p1, const PathPoint& p2, float threshold) const;
    
    // Validation methods
    bool ValidateOptimizedPath(const std::vector<PathPoint>& original,
                               const std::vector<PathPoint>& optimized) const;
    bool EnsurePathContinuity(const std::vector<PathPoint>& path) const;
    
    // Reporting methods
    void PrintOptimizationReport(const std::vector<PathPoint>& original,
                                 const std::vector<PathPoint>& optimized,
                                 const std::vector<DetectedLoop>& loops,
                                 const std::vector<DetectedBacktrack>& backtracks,
                                 const std::vector<DetectedOscillation>& oscillations,
                                 const std::vector<DetectedHub>& hubs);
    void GenerateReportSummary();
    
    // Timing utilities
    double GetCurrentTimeMs() const;
};

/**
 * PathOptimizationUtils - Utility namespace for testing
 */
namespace PathOptimizationUtils 
{
    // Test path generators for validation
    std::vector<PathPoint> CreateRectangleLoopPath();      // A→B→C→D→A→E pattern
    std::vector<PathPoint> CreateBacktrackTestPath();      // A→B→C→B→D pattern
    std::vector<PathPoint> CreateOscillationTestPath();    // A→B→A→B→A→B pattern
    std::vector<PathPoint> CreateJunctionHubTestPath();    // A→B→C→B→D→B→E pattern
    std::vector<PathPoint> CreateComplexTestPath();        // Combination of all patterns
    
    // Path analysis utilities
    void PrintPathStatistics(const std::vector<PathPoint>& path);
    void VisualizePath(const std::vector<PathPoint>& path);
    bool SaveOptimizedPath(const std::vector<PathPoint>& path, const std::string& filename);
}

} // namespace ORB_SLAM3

#endif // PATHOPTIMIZER_H