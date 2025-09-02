/**
 * PathGuide.h
 * 
 * Navigation guidance system for ORB-SLAM3 path following
 * Generates audio instructions for blind navigation assistance
 * Part of Phase 5: Guidance Interface Development
 */

#ifndef PATHGUIDE_H
#define PATHGUIDE_H

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include "sophus/se3.hpp"
#include "PathRecorder.h"  // For PathPoint definition
#include "PathMatcher.h"   // For MatchResult and DeviationInfo definitions
#include "PathOptimizer.h" // For intelligent path optimization

namespace ORB_SLAM3
{

// Forward declarations
class AudioGuide;
class PathOptimizer;

/**
 * TurnInfo - Represents a detected turn in the path for backwards navigation
 */
struct TurnInfo {
    size_t waypoint_index;     // Waypoint where turn occurs
    float turn_angle;          // Angle of turn in degrees
    bool is_left_turn;         // true = left, false = right
    bool is_sharp_turn;        // true if >70°, false if <70°
    
    TurnInfo() : waypoint_index(0), turn_angle(0.0f), is_left_turn(false), is_sharp_turn(false) {}
    TurnInfo(size_t idx, float angle, bool left, bool sharp) 
        : waypoint_index(idx), turn_angle(angle), is_left_turn(left), is_sharp_turn(sharp) {}
};

enum class GuidanceState {
    IDLE,           // No guidance active
    LOADING_PATH,   // Loading recorded path
    READY,          // Path loaded, ready to guide
    GUIDING,        // Actively providing guidance
    PAUSED,         // Guidance paused
    COMPLETED,      // Path completed successfully
    ERROR_STATE     // Error state (renamed to avoid Windows ERROR macro)
};

enum class InstructionType {
    START,          // "Starting navigation"
    GO_STRAIGHT,    // "Go straight X meters"
    TURN_LEFT,      // "Turn left X degrees"
    TURN_RIGHT,     // "Turn right X degrees"
    CORRECT_LEFT,   // "Correct course - turn left"
    CORRECT_RIGHT,  // "Correct course - turn right"
    SLOW_DOWN,      // "Slow down"
    STOP,           // "Stop"
    WAYPOINT,       // "Waypoint reached"
    DESTINATION,    // "Destination reached"
    OFF_PATH,       // "You are off the path"
    OBSTACLE,       // "Obstacle detected"
    ERROR_MESSAGE,  // Error messages (renamed to avoid Windows ERROR macro)
    TURN_AROUND,    // "Turn around - you're facing the wrong way"
    CONTINUE_FORWARD, // "Continue straight ahead"
    STOPPED_MOVING,  // "You've stopped - keep moving forward"
    AT_JUNCTION      // "Turn left/right at this junction"
};

struct GuidanceInstruction {
    InstructionType type;
    std::string message;
    float distance_meters;      // For distance-based instructions
    float angle_degrees;        // For turn instructions
    int priority;              // 1=urgent, 2=normal, 3=info
    double timestamp;
    bool spatial_audio;        // Use 3D audio positioning
    
    GuidanceInstruction() : type(InstructionType::START), distance_meters(0.0f), 
                           angle_degrees(0.0f), priority(2), timestamp(0.0), spatial_audio(false) {}
};

class PathGuide
{
public:
    PathGuide();
    ~PathGuide();
    
    // Core guidance functionality
    bool LoadPath(const std::string& path_filename);
    bool StartGuidance();
    void PauseGuidance();
    void ResumeGuidance();
    void StopGuidance();
    
    // Real-time guidance updates
    void UpdateCurrentPose(const Sophus::SE3f& current_pose);
    GuidanceInstruction GetNextInstruction();
    bool HasPendingInstructions() const;
    
    // Configuration
    void SetVoiceSettings(float speed = 1.0f, float volume = 0.8f);
    void SetGuidanceParameters(float lookahead_distance = 2.0f, 
                              float instruction_frequency = 3.0f,
                              float off_path_threshold = 1.0f);
    void SetNavigationMode(bool verbose = true, bool distance_callouts = true);
    
    // Audio control
    void EnableAudio(bool enable = true);
    void TestAudio();
    bool IsAudioEnabled() const;
    AudioGuide* GetAudioGuide() const;
    
    // Backwards navigation control
    void SetBackwardsMode(bool backwards = true);
    bool IsBackwardsMode() const;
    void EnableBackwardsMode();  // Convenience method that also triggers optimization
    
    // Status and control
    GuidanceState GetState() const;
    std::string GetStatusString() const;
    float GetPathProgress() const;
    float GetDistanceToDestination() const;
    size_t GetRemainingWaypoints() const;
    
    // Emergency and safety
    void EmergencyStop();
    bool IsEmergencyStopped() const;
    void ClearEmergencyStop();
    
    // State checking
    bool IsGuidanceActive() const;
    
    // Debugging and analysis
    void PrintGuidanceStatistics() const;
    std::vector<GuidanceInstruction> GetInstructionHistory() const;
    
    // Path optimization methods
    void SetOptimizationLevel(float level);
    OptimizationReport GetLastOptimizationReport() const;
    
    // Turn detection and navigation
    void AnalyzePathGeometry();
    const TurnInfo* FindNearbyTurn(const Eigen::Vector3f& current_position, float search_radius = 5.0f) const;
    size_t GetDetectedTurnsCount() const;

private:
    // Core components
    std::unique_ptr<PathMatcher> path_matcher_;
    std::unique_ptr<AudioGuide> audio_guide_;
    std::unique_ptr<PathOptimizer> path_optimizer_;
    std::atomic<GuidanceState> current_state_;
    
    // Threading for real-time guidance
    std::thread guidance_thread_;
    std::atomic<bool> should_stop_thread_;
    std::mutex guidance_mutex_;
    
    // Instruction queue and management
    std::queue<GuidanceInstruction> pending_instructions_;
    std::vector<GuidanceInstruction> instruction_history_;
    mutable std::mutex instruction_mutex_;
    
    // Current navigation state
    Sophus::SE3f last_pose_;
    MatchResult last_match_;
    double last_update_time_;
    double last_instruction_time_;
    bool has_valid_pose_;
    
    // Configuration parameters
    float voice_speed_;
    float voice_volume_;
    float lookahead_distance_;        // How far ahead to look for waypoints
    float instruction_frequency_;     // Seconds between regular updates
    float off_path_threshold_;        // Distance threshold for off-path detection
    bool verbose_mode_;              // Detailed vs minimal instructions
    bool distance_callouts_;         // Include distance information
    bool backwards_mode_;            // Backwards navigation mode
    
    // Movement tracking for stationary detection
    std::vector<std::pair<Sophus::SE3f, double>> pose_history_;  // Recent poses with timestamps
    static const size_t MAX_POSE_HISTORY = 10;                  // Keep last 10 poses
    
    // Instruction completion tracking
    bool starting_point_announced_;       // Track if "starting point" was announced
    InstructionType active_instruction_;  // Current instruction waiting for completion
    double active_instruction_time_;      // When current instruction was issued
    
    // Emergency state
    std::atomic<bool> emergency_stopped_;
    
    // Turn detection system
    std::vector<TurnInfo> detected_turns_;
    mutable std::mutex turns_mutex_;
    
    // Statistics
    size_t total_instructions_;
    double guidance_start_time_;
    float total_distance_traveled_;
    
    // Core guidance logic - runs in separate thread
    void GuidanceThreadMain();
    
    // Instruction generation
    GuidanceInstruction GenerateInstruction(const MatchResult& match);
    GuidanceInstruction AnalyzeMovement(const MatchResult& current_match, const MatchResult& previous_match);
    GuidanceInstruction CheckPathDeviation(const DeviationInfo& deviation);
    GuidanceInstruction GenerateProgressUpdate();
    
    // Direction and movement analysis
    float CalculateRequiredTurn(const Sophus::SE3f& current_pose, const PathPoint& target_waypoint);
    float CalculateDistance(const Sophus::SE3f& pose1, const Sophus::SE3f& pose2) const;
    std::string FormatDistanceMessage(float distance_meters);
    std::string FormatAngleMessage(float angle_degrees, bool is_left);
    
    // Camera direction detection for backwards navigation
    Eigen::Vector3f ExtractCameraDirection(const Sophus::SE3f& pose) const;
    Eigen::Vector3f CalculateRequiredDirection(const PathPoint& from_point, const PathPoint& to_point) const;
    float CalculateOrientationError(const Sophus::SE3f& current_pose, const Eigen::Vector3f& required_direction) const;
    bool IsUserMoving(const Sophus::SE3f& current_pose, double current_time) const;
    GuidanceInstruction AnalyzeBackwardsNavigation(const MatchResult& match);
    
    // Instruction management
    void AddInstruction(const GuidanceInstruction& instruction);
    void ClearInstructions();
    bool ShouldGenerateInstruction(InstructionType type);
    
    // State management helpers
    void SetState(GuidanceState new_state);
    void UpdateStatistics(const GuidanceInstruction& instruction);
    
    // Safety and validation
    bool ValidatePose(const Sophus::SE3f& pose);
    bool IsMovementReasonable(const Sophus::SE3f& new_pose, const Sophus::SE3f& old_pose);
    
    // Utility functions
    double GetCurrentTime() const;
    std::string InstructionTypeToString(InstructionType type);
};

} // namespace ORB_SLAM3

#endif // PATHGUIDE_H