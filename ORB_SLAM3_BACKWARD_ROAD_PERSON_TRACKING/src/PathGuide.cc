/**
 * PathGuide.cc
 * 
 * Implementation of navigation guidance system for blind users
 * Provides real-time audio instructions for path following
 */

#include "PathGuide.h"
#include "PathMatcher.h"
#include "AudioGuide.h"
#include "PathOptimizer.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <chrono>
#include <algorithm>

#ifdef _WIN32
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#endif

namespace ORB_SLAM3
{

PathGuide::PathGuide()
    : path_matcher_(std::make_unique<PathMatcher>())
    , audio_guide_(std::make_unique<AudioGuide>())
    , path_optimizer_(std::make_unique<PathOptimizer>())
    , current_state_(GuidanceState::IDLE)
    , should_stop_thread_(false)
    , last_update_time_(0.0)
    , last_instruction_time_(0.0)
    , has_valid_pose_(false)
    , voice_speed_(1.0f)
    , voice_volume_(0.8f)
    , lookahead_distance_(2.0f)
    , instruction_frequency_(3.0f)
    , off_path_threshold_(1.0f)
    , verbose_mode_(true)
    , distance_callouts_(true)
    , backwards_mode_(false)
    , starting_point_announced_(false)
    , active_instruction_(InstructionType::START)
    , active_instruction_time_(0.0)
    , emergency_stopped_(false)
    , total_instructions_(0)
    , guidance_start_time_(0.0)
    , total_distance_traveled_(0.0f)
{
    std::cout << "PathGuide: Initialized with lookahead=" << lookahead_distance_ 
              << "m, frequency=" << instruction_frequency_ << "s" << std::endl;
    
    // Initialize audio system
    if (audio_guide_ && audio_guide_->Initialize()) {
        std::cout << "PathGuide: Audio guidance system initialized successfully" << std::endl;
    } else {
        std::cerr << "PathGuide: Warning - Audio guidance system failed to initialize" << std::endl;
    }
}

PathGuide::~PathGuide()
{
    StopGuidance();
    if (guidance_thread_.joinable()) {
        should_stop_thread_ = true;
        guidance_thread_.join();
    }
    
    // Shutdown audio system
    if (audio_guide_) {
        audio_guide_->Shutdown();
    }
}

bool PathGuide::LoadPath(const std::string& path_filename)
{
    std::lock_guard<std::mutex> lock(guidance_mutex_);
    
    if (current_state_ == GuidanceState::GUIDING) {
        std::cerr << "PathGuide: Cannot load path while guidance is active" << std::endl;
        return false;
    }
    
    SetState(GuidanceState::LOADING_PATH);
    
    if (!path_matcher_->LoadPath(path_filename)) {
        std::cerr << "PathGuide: Failed to load path: " << path_filename << std::endl;
        SetState(GuidanceState::ERROR_STATE);
        return false;
    }
    
    // Reset navigation state
    has_valid_pose_ = false;
    last_update_time_ = 0.0;
    last_instruction_time_ = 0.0;
    total_distance_traveled_ = 0.0f;
    
    // Clear instruction queues
    ClearInstructions();
    instruction_history_.clear();
    
    SetState(GuidanceState::READY);
    
    std::cout << "PathGuide: Loaded path with " << path_matcher_->GetPathLength() 
              << " waypoints" << std::endl;
    
    return true;
}

bool PathGuide::StartGuidance()
{
    std::lock_guard<std::mutex> lock(guidance_mutex_);
    
    if (current_state_ != GuidanceState::READY && current_state_ != GuidanceState::PAUSED) {
        std::cerr << "PathGuide: Cannot start guidance from current state" << std::endl;
        return false;
    }
    
    if (!path_matcher_->IsPathLoaded()) {
        std::cerr << "PathGuide: No path loaded for guidance" << std::endl;
        return false;
    }
    
    SetState(GuidanceState::GUIDING);
    emergency_stopped_ = false;
    guidance_start_time_ = GetCurrentTime();
    
    // Reset instruction completion tracking
    starting_point_announced_ = false;
    active_instruction_ = InstructionType::START;
    active_instruction_time_ = 0.0;
    
    // Start guidance thread if not already running
    if (!guidance_thread_.joinable()) {
        should_stop_thread_ = false;
        guidance_thread_ = std::thread(&PathGuide::GuidanceThreadMain, this);
    }
    
    // Add welcome instruction
    GuidanceInstruction start_instruction;
    start_instruction.type = InstructionType::START;
    start_instruction.message = "Navigation started. Begin moving forward.";
    start_instruction.priority = 1;
    start_instruction.timestamp = GetCurrentTime();
    AddInstruction(start_instruction);
    
    std::cout << "PathGuide: Guidance started" << std::endl;
    return true;
}

void PathGuide::PauseGuidance()
{
    std::lock_guard<std::mutex> lock(guidance_mutex_);
    if (current_state_ == GuidanceState::GUIDING) {
        SetState(GuidanceState::PAUSED);
        std::cout << "PathGuide: Guidance paused" << std::endl;
    }
}

void PathGuide::ResumeGuidance()
{
    std::lock_guard<std::mutex> lock(guidance_mutex_);
    if (current_state_ == GuidanceState::PAUSED) {
        SetState(GuidanceState::GUIDING);
        std::cout << "PathGuide: Guidance resumed" << std::endl;
    }
}

void PathGuide::StopGuidance()
{
    {
        std::lock_guard<std::mutex> lock(guidance_mutex_);
        if (current_state_ == GuidanceState::GUIDING || current_state_ == GuidanceState::PAUSED) {
            SetState(GuidanceState::READY);
            std::cout << "PathGuide: Guidance stopped" << std::endl;
        }
    }
    
    // Stop guidance thread
    if (guidance_thread_.joinable()) {
        should_stop_thread_ = true;
        guidance_thread_.join();
        should_stop_thread_ = false;
    }
}

void PathGuide::UpdateCurrentPose(const Sophus::SE3f& current_pose)
{
    if (current_state_ != GuidanceState::GUIDING || emergency_stopped_) {
        return;
    }
    
    if (!ValidatePose(current_pose)) {
        std::cerr << "PathGuide: Invalid pose received" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(guidance_mutex_);
    
    // Calculate distance traveled since last update
    if (has_valid_pose_) {
        float distance_delta = CalculateDistance(current_pose, last_pose_);
        total_distance_traveled_ += distance_delta;
        
        // Check for reasonable movement (avoid GPS jumps, etc.)
        if (!IsMovementReasonable(current_pose, last_pose_)) {
            std::cerr << "PathGuide: Unreasonable movement detected, ignoring update" << std::endl;
            return;
        }
    }
    
    last_pose_ = current_pose;
    last_update_time_ = GetCurrentTime();
    has_valid_pose_ = true;
}

GuidanceInstruction PathGuide::GetNextInstruction()
{
    std::lock_guard<std::mutex> lock(instruction_mutex_);
    
    if (pending_instructions_.empty()) {
        GuidanceInstruction empty_instruction;
        empty_instruction.type = InstructionType::ERROR_MESSAGE;
        empty_instruction.message = "";
        return empty_instruction;
    }
    
    GuidanceInstruction instruction = pending_instructions_.front();
    pending_instructions_.pop();
    
    // Add to history
    instruction_history_.push_back(instruction);
    
    return instruction;
}

bool PathGuide::HasPendingInstructions() const
{
    std::lock_guard<std::mutex> lock(instruction_mutex_);
    return !pending_instructions_.empty();
}

void PathGuide::SetVoiceSettings(float speed, float volume)
{
    voice_speed_ = std::max(0.5f, std::min(2.0f, speed));
    voice_volume_ = std::max(0.1f, std::min(1.0f, volume));
    std::cout << "PathGuide: Voice settings - speed=" << voice_speed_ 
              << ", volume=" << voice_volume_ << std::endl;
}

void PathGuide::SetGuidanceParameters(float lookahead_distance, float instruction_frequency, float off_path_threshold)
{
    lookahead_distance_ = std::max(1.0f, std::min(10.0f, lookahead_distance));
    instruction_frequency_ = std::max(1.0f, std::min(10.0f, instruction_frequency));
    off_path_threshold_ = std::max(0.5f, std::min(5.0f, off_path_threshold));
    
    std::cout << "PathGuide: Parameters updated - lookahead=" << lookahead_distance_ 
              << "m, frequency=" << instruction_frequency_ << "s, threshold=" 
              << off_path_threshold_ << "m" << std::endl;
}

void PathGuide::SetNavigationMode(bool verbose, bool distance_callouts)
{
    verbose_mode_ = verbose;
    distance_callouts_ = distance_callouts;
    std::cout << "PathGuide: Navigation mode - verbose=" << (verbose ? "on" : "off") 
              << ", distance_callouts=" << (distance_callouts ? "on" : "off") << std::endl;
}

GuidanceState PathGuide::GetState() const
{
    return current_state_.load();
}

std::string PathGuide::GetStatusString() const
{
    std::ostringstream oss;
    
    switch (current_state_.load()) {
        case GuidanceState::IDLE: oss << "Idle"; break;
        case GuidanceState::LOADING_PATH: oss << "Loading path"; break;
        case GuidanceState::READY: oss << "Ready"; break;
        case GuidanceState::GUIDING: oss << "Guiding"; break;
        case GuidanceState::PAUSED: oss << "Paused"; break;
        case GuidanceState::COMPLETED: oss << "Completed"; break;
        case GuidanceState::ERROR_STATE: oss << "Error"; break;
    }
    
    if (path_matcher_ && path_matcher_->IsPathLoaded()) {
        oss << " (" << path_matcher_->GetPathLength() << " waypoints, " 
            << std::fixed << std::setprecision(1) << (GetPathProgress() * 100.0f) << "%)";
    }
    
    return oss.str();
}

float PathGuide::GetPathProgress() const
{
    if (path_matcher_) {
        return path_matcher_->GetPathProgress();
    }
    return 0.0f;
}

float PathGuide::GetDistanceToDestination() const
{
    if (!path_matcher_ || !has_valid_pose_) {
        return -1.0f;
    }
    
    // Calculate distance to final waypoint
    if (path_matcher_->GetPathLength() > 0) {
        PathPoint target = path_matcher_->GetTargetWaypoint(path_matcher_->GetPathLength());
        return CalculateDistance(last_pose_, target.pose);
    }
    
    return -1.0f;
}

size_t PathGuide::GetRemainingWaypoints() const
{
    if (path_matcher_) {
        return path_matcher_->GetPathLength() - path_matcher_->GetCurrentPathIndex();
    }
    return 0;
}

void PathGuide::EmergencyStop()
{
    emergency_stopped_ = true;
    
    GuidanceInstruction emergency_instruction;
    emergency_instruction.type = InstructionType::STOP;
    emergency_instruction.message = "Emergency stop activated. Navigation halted.";
    emergency_instruction.priority = 1;
    emergency_instruction.timestamp = GetCurrentTime();
    AddInstruction(emergency_instruction);
    
    std::cout << "PathGuide: EMERGENCY STOP activated" << std::endl;
}

bool PathGuide::IsEmergencyStopped() const
{
    return emergency_stopped_.load();
}

void PathGuide::ClearEmergencyStop()
{
    emergency_stopped_ = false;
    std::cout << "PathGuide: Emergency stop cleared" << std::endl;
}

// Private implementation methods

void PathGuide::GuidanceThreadMain()
{
    std::cout << "PathGuide: Guidance thread started" << std::endl;
    
    while (!should_stop_thread_ && current_state_ != GuidanceState::ERROR_STATE) {
        try {
            if (current_state_ == GuidanceState::GUIDING && !emergency_stopped_ && has_valid_pose_) {
                // Get current match from PathMatcher
                MatchResult current_match = path_matcher_->FindNearestPoint(last_pose_);
                
                // Debug: Show matching information every 10 iterations (~5 seconds)
                static int debug_counter = 0;
                if (++debug_counter % 10 == 0) {
                    std::cout << "📍 PathGuide: Matched to waypoint " << current_match.path_index 
                              << "/" << path_matcher_->GetPathLength() << ", distance: " 
                              << std::fixed << std::setprecision(3) << current_match.distance << "m, "
                              << (current_match.is_on_path ? "ON PATH" : "OFF PATH") << std::endl;
                }
                
                // Update path progress
                path_matcher_->UpdateProgress(current_match);
                
                // Generate guidance instruction based on current situation
                GuidanceInstruction instruction = GenerateInstruction(current_match);
                
                if (instruction.type != InstructionType::ERROR_MESSAGE) {
                    // Debug: Print navigation instruction to console
                    std::cout << "🧭 PathGuide: " << instruction.message << " [Progress: " 
                              << std::fixed << std::setprecision(1) << (path_matcher_->GetPathProgress() * 100.0f) << "%]" << std::endl;
                    
                    // Speak instruction via audio system
                    if (audio_guide_ && audio_guide_->IsEnabled()) {
                        AudioPriority audio_priority = (instruction.priority == 1) ? AudioPriority::URGENT : AudioPriority::NORMAL;
                        audio_guide_->Speak(instruction.message, audio_priority);
                    }
                    
                    AddInstruction(instruction);
                    UpdateStatistics(instruction);
                }
                
                // Check if path completed
                if (path_matcher_->IsPathCompleted()) {
                    SetState(GuidanceState::COMPLETED);
                    
                    GuidanceInstruction completion_instruction;
                    completion_instruction.type = InstructionType::DESTINATION;
                    completion_instruction.message = "Destination reached. Navigation complete.";
                    completion_instruction.priority = 1;
                    completion_instruction.timestamp = GetCurrentTime();
                    
                    // Speak completion message
                    if (audio_guide_ && audio_guide_->IsEnabled()) {
                        audio_guide_->Speak(completion_instruction.message, AudioPriority::HIGH);
                    }
                    
                    AddInstruction(completion_instruction);
                }
                
                last_match_ = current_match;
            }
            
            // Sleep for guidance update interval
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 2Hz update rate
            
        } catch (const std::exception& e) {
            std::cerr << "PathGuide: Exception in guidance thread: " << e.what() << std::endl;
            SetState(GuidanceState::ERROR_STATE);
        }
    }
    
    std::cout << "PathGuide: Guidance thread stopped" << std::endl;
}

GuidanceInstruction PathGuide::GenerateInstruction(const MatchResult& match)
{
    GuidanceInstruction instruction;
    double current_time = GetCurrentTime();
    
    // Check if we should generate a new instruction based on timing
    if ((current_time - last_instruction_time_) < instruction_frequency_) {
        instruction.type = InstructionType::ERROR_MESSAGE; // No instruction needed yet
        return instruction;
    }
    
    // Priority 0: Use backwards navigation if enabled
    if (backwards_mode_) {
        instruction = AnalyzeBackwardsNavigation(match);
        if (instruction.type != InstructionType::ERROR_MESSAGE) {
            last_instruction_time_ = current_time;
            return instruction;
        }
    }
    
    // Priority 1: Check for path deviation
    DeviationInfo deviation = path_matcher_->CalculateDeviation(last_pose_);
    if (deviation.needs_correction) {
        instruction = CheckPathDeviation(deviation);
        if (instruction.type != InstructionType::ERROR_MESSAGE) {
            last_instruction_time_ = current_time;
            return instruction;
        }
    }
    
    // Priority 2: Generate forward movement instruction
    PathPoint target_waypoint = path_matcher_->GetTargetWaypoint(
        static_cast<size_t>(lookahead_distance_ * 2)); // Look ahead based on distance
    
    float required_turn = CalculateRequiredTurn(last_pose_, target_waypoint);
    float distance_to_target = CalculateDistance(last_pose_, target_waypoint.pose);
    
    // Determine instruction type based on required turn
    if (std::abs(required_turn) > 15.0f) { // Significant turn needed
        instruction.type = (required_turn > 0) ? InstructionType::TURN_LEFT : InstructionType::TURN_RIGHT;
        instruction.angle_degrees = std::abs(required_turn);
        instruction.message = FormatAngleMessage(std::abs(required_turn), required_turn > 0);
    } else { // Go straight
        instruction.type = InstructionType::GO_STRAIGHT;
        instruction.distance_meters = distance_to_target;
        
        if (distance_callouts_ && distance_to_target > 1.0f) {
            instruction.message = "Continue straight " + FormatDistanceMessage(distance_to_target);
        } else {
            instruction.message = "Continue straight";
        }
    }
    
    instruction.priority = 2;
    instruction.timestamp = current_time;
    last_instruction_time_ = current_time;
    
    return instruction;
}

GuidanceInstruction PathGuide::CheckPathDeviation(const DeviationInfo& deviation)
{
    GuidanceInstruction instruction;
    
    if (deviation.distance_from_path > off_path_threshold_) {
        instruction.type = InstructionType::OFF_PATH;
        instruction.message = "You are off the path. " + deviation.correction_hint;
        instruction.priority = 1;
    } else if (deviation.needs_correction) {
        // Minor correction needed
        if (deviation.correction_hint.find("left") != std::string::npos) {
            instruction.type = InstructionType::CORRECT_LEFT;
            instruction.message = "Slight correction - turn left";
        } else if (deviation.correction_hint.find("right") != std::string::npos) {
            instruction.type = InstructionType::CORRECT_RIGHT;
            instruction.message = "Slight correction - turn right";
        } else {
            instruction.type = InstructionType::GO_STRAIGHT;
            instruction.message = "Continue forward";
        }
        instruction.priority = 2;
    } else {
        instruction.type = InstructionType::ERROR_MESSAGE; // No correction needed
    }
    
    return instruction;
}

float PathGuide::CalculateRequiredTurn(const Sophus::SE3f& current_pose, const PathPoint& target_waypoint)
{
    // Calculate vector from current position to target
    Eigen::Vector3f current_pos = current_pose.translation();
    Eigen::Vector3f target_pos = target_waypoint.pose.translation();
    Eigen::Vector3f to_target = target_pos - current_pos;
    to_target.normalize();
    
    // Get current forward direction (assuming -Z is forward in camera frame)
    Eigen::Vector3f current_forward = current_pose.rotationMatrix() * Eigen::Vector3f(0, 0, -1);
    current_forward.normalize();
    
    // Calculate angle between current forward direction and direction to target
    float dot_product = current_forward.dot(to_target);
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product)); // Clamp for acos
    
    float angle_rad = std::acos(dot_product);
    float angle_deg = angle_rad * 180.0f / M_PI;
    
    // Determine left/right using cross product
    Eigen::Vector3f cross = current_forward.cross(to_target);
    if (cross.y() < 0) { // Assuming Y is up
        angle_deg = -angle_deg; // Turn right (negative)
    }
    
    return angle_deg;
}

float PathGuide::CalculateDistance(const Sophus::SE3f& pose1, const Sophus::SE3f& pose2) const
{
    Eigen::Vector3f pos1 = pose1.translation();
    Eigen::Vector3f pos2 = pose2.translation();
    return (pos1 - pos2).norm();
}

std::string PathGuide::FormatDistanceMessage(float distance_meters)
{
    std::ostringstream oss;
    
    if (distance_meters >= 10.0f) {
        oss << std::fixed << std::setprecision(0) << distance_meters << " meters";
    } else if (distance_meters >= 1.0f) {
        oss << std::fixed << std::setprecision(1) << distance_meters << " meters";
    } else {
        int cm = static_cast<int>(distance_meters * 100);
        oss << cm << " centimeters";
    }
    
    return oss.str();
}

std::string PathGuide::FormatAngleMessage(float angle_degrees, bool is_left)
{
    std::ostringstream oss;
    oss << "Turn " << (is_left ? "left" : "right") << " " 
        << std::fixed << std::setprecision(0) << angle_degrees << " degrees";
    return oss.str();
}

void PathGuide::AddInstruction(const GuidanceInstruction& instruction)
{
    std::lock_guard<std::mutex> lock(instruction_mutex_);
    pending_instructions_.push(instruction);
    total_instructions_++;
}

void PathGuide::ClearInstructions()
{
    std::lock_guard<std::mutex> lock(instruction_mutex_);
    while (!pending_instructions_.empty()) {
        pending_instructions_.pop();
    }
}

void PathGuide::SetState(GuidanceState new_state)
{
    current_state_ = new_state;
}

bool PathGuide::IsGuidanceActive() const
{
    GuidanceState state = current_state_.load();
    return (state == GuidanceState::GUIDING || state == GuidanceState::PAUSED);
}

void PathGuide::UpdateStatistics(const GuidanceInstruction& instruction)
{
    // Statistics are updated by the calling functions
    // This can be expanded for detailed analytics
}

bool PathGuide::ValidatePose(const Sophus::SE3f& pose)
{
    Eigen::Vector3f translation = pose.translation();
    
    // Check for NaN or infinite values
    if (!translation.allFinite()) {
        return false;
    }
    
    // Check for reasonable bounds (assuming indoor navigation)
    if (translation.norm() > 1000.0f) { // 1km limit
        return false;
    }
    
    return true;
}

bool PathGuide::IsMovementReasonable(const Sophus::SE3f& new_pose, const Sophus::SE3f& old_pose)
{
    float distance = CalculateDistance(new_pose, old_pose);
    
    // Check for reasonable movement (max 5 m/s assuming updates every 0.5s)
    if (distance > 2.5f) {
        return false;
    }
    
    return true;
}

double PathGuide::GetCurrentTime() const
{
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1e6;
}

std::string PathGuide::InstructionTypeToString(InstructionType type)
{
    switch (type) {
        case InstructionType::START: return "START";
        case InstructionType::GO_STRAIGHT: return "GO_STRAIGHT";
        case InstructionType::TURN_LEFT: return "TURN_LEFT";
        case InstructionType::TURN_RIGHT: return "TURN_RIGHT";
        case InstructionType::CORRECT_LEFT: return "CORRECT_LEFT";
        case InstructionType::CORRECT_RIGHT: return "CORRECT_RIGHT";
        case InstructionType::SLOW_DOWN: return "SLOW_DOWN";
        case InstructionType::STOP: return "STOP";
        case InstructionType::WAYPOINT: return "WAYPOINT";
        case InstructionType::DESTINATION: return "DESTINATION";
        case InstructionType::OFF_PATH: return "OFF_PATH";
        case InstructionType::OBSTACLE: return "OBSTACLE";
        case InstructionType::ERROR_MESSAGE: return "ERROR";
        case InstructionType::TURN_AROUND: return "TURN_AROUND";
        case InstructionType::CONTINUE_FORWARD: return "CONTINUE_FORWARD";
        case InstructionType::STOPPED_MOVING: return "STOPPED_MOVING";
        case InstructionType::AT_JUNCTION: return "AT_JUNCTION";
        default: return "UNKNOWN";
    }
}

void PathGuide::PrintGuidanceStatistics() const
{
    std::cout << "=== PathGuide Statistics ===" << std::endl;
    std::cout << "Current state: " << GetStatusString() << std::endl;
    std::cout << "Total instructions: " << total_instructions_ << std::endl;
    std::cout << "Distance traveled: " << total_distance_traveled_ << "m" << std::endl;
    std::cout << "Path progress: " << (GetPathProgress() * 100.0f) << "%" << std::endl;
    std::cout << "Remaining waypoints: " << GetRemainingWaypoints() << std::endl;
    std::cout << "Emergency stopped: " << (emergency_stopped_ ? "Yes" : "No") << std::endl;
    std::cout << "============================" << std::endl;
}

std::vector<GuidanceInstruction> PathGuide::GetInstructionHistory() const
{
    std::lock_guard<std::mutex> lock(instruction_mutex_);
    return instruction_history_;
}

void PathGuide::EnableAudio(bool enable)
{
    if (audio_guide_) {
        audio_guide_->SetEnabled(enable);
        std::cout << "PathGuide: Audio output " << (enable ? "enabled" : "disabled") << std::endl;
    }
}

void PathGuide::TestAudio()
{
    if (audio_guide_) {
        audio_guide_->TestAudio();
    }
}

bool PathGuide::IsAudioEnabled() const
{
    return audio_guide_ && audio_guide_->IsEnabled();
}

AudioGuide* PathGuide::GetAudioGuide() const
{
    return audio_guide_.get();
}

void PathGuide::SetBackwardsMode(bool backwards)
{
    std::lock_guard<std::mutex> lock(guidance_mutex_);
    backwards_mode_ = backwards;
    pose_history_.clear(); // Clear pose history when switching modes
    
    std::cout << "PathGuide: " << (backwards ? "Enabled" : "Disabled") 
              << " backwards navigation mode" << std::endl;
              
    if (backwards) {
        // Trigger automatic path optimization when backwards mode is enabled
        if (path_matcher_ && path_matcher_->IsPathLoaded()) {
            std::cout << "PathGuide: Backwards mode detected - optimizing path for direct navigation..." << std::endl;
            
            const std::vector<PathPoint>& original_path = path_matcher_->GetLoadedPath();
            if (!original_path.empty()) {
                std::vector<PathPoint> optimized_path = path_optimizer_->OptimizeForBackwardsNavigation(original_path);
                
                OptimizationReport report = path_optimizer_->GetLastOptimizationReport();
                if (report.optimization_successful && optimized_path.size() != original_path.size()) {
                    // TODO: Update PathMatcher with optimized path
                    // For now, just report the optimization results
                    std::cout << "PathGuide: Loop optimization completed - " << report.summary << std::endl;
                    
                    // Re-analyze path geometry for optimized path
                    AnalyzePathGeometry();
                } else {
                    std::cout << "PathGuide: No significant optimizations found" << std::endl;
                    
                    // Still analyze geometry for the original path
                    AnalyzePathGeometry();
                }
            }
        }
        
        if (audio_guide_) {
            audio_guide_->Speak("Backwards navigation mode activated", AudioPriority::HIGH);
        }
    }
}

bool PathGuide::IsBackwardsMode() const
{
    return backwards_mode_;
}

Eigen::Vector3f PathGuide::ExtractCameraDirection(const Sophus::SE3f& pose) const
{
    // Camera coordinate system: Z-forward, X-right, Y-down
    // Extract forward direction (Z-axis) from rotation matrix
    Eigen::Matrix3f rotation = pose.rotationMatrix();
    return rotation.col(2); // Z-axis column = forward direction
}

Eigen::Vector3f PathGuide::CalculateRequiredDirection(const PathPoint& from_point, const PathPoint& to_point) const
{
    Eigen::Vector3f direction = to_point.pose.translation() - from_point.pose.translation();
    return direction.normalized();
}

float PathGuide::CalculateOrientationError(const Sophus::SE3f& current_pose, const Eigen::Vector3f& required_direction) const
{
    Eigen::Vector3f camera_direction = ExtractCameraDirection(current_pose);
    
    // Calculate angle between camera direction and required direction
    float dot_product = camera_direction.dot(required_direction);
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product)); // Clamp for numerical stability
    
    return std::acos(std::abs(dot_product)) * 180.0f / M_PI; // Return angle in degrees
}

bool PathGuide::IsUserMoving(const Sophus::SE3f& current_pose, double current_time) const
{
    if (pose_history_.size() < 2) {
        return true; // Assume moving if insufficient history
    }
    
    // Check if user has moved significantly in last few seconds (sensitive to ORB-SLAM3)
    const float MIN_MOVEMENT_DISTANCE = 0.15f; // 15cm (sensitive to small steps)
    const double MOVEMENT_TIME_WINDOW = 3.0;   // 3 seconds
    
    for (auto it = pose_history_.rbegin(); it != pose_history_.rend(); ++it) {
        double time_diff = current_time - it->second;
        if (time_diff > MOVEMENT_TIME_WINDOW) break;
        
        float distance = CalculateDistance(current_pose, it->first);
        if (distance > MIN_MOVEMENT_DISTANCE) {
            return true;
        }
    }
    
    return false; // User appears to be stationary
}

GuidanceInstruction PathGuide::AnalyzeBackwardsNavigation(const MatchResult& match)
{
    GuidanceInstruction instruction;
    double current_time = GetCurrentTime();
    
    if (match.path_index >= path_matcher_->GetPathLength()) {
        instruction.type = InstructionType::ERROR_MESSAGE;
        instruction.message = "Lost path tracking";
        instruction.priority = 1;
        return instruction;
    }
    
    // Get path points for direction analysis
    const auto& path_points = path_matcher_->GetLoadedPath();
    const PathPoint& current_waypoint = path_points[match.path_index];
    
    // Update pose history for movement detection
    pose_history_.push_back({last_pose_, current_time});
    if (pose_history_.size() > MAX_POSE_HISTORY) {
        pose_history_.erase(pose_history_.begin());
    }
    
    // Check if user is moving using ORB-SLAM3 tracking quality
    bool is_moving = IsUserMoving(last_pose_, current_time);
    std::cout << "🧭 MOVEMENT DEBUG: is_moving = " << (is_moving ? "YES" : "NO") 
              << ", pose_history_size = " << pose_history_.size() << std::endl;
    
    // For backwards navigation, calculate required direction to previous waypoint
    // BUT CHECK FOR COMPLETION FIRST - only at the TRUE STARTING POINT (index 0)
    Eigen::Vector3f required_direction;
    
    // Check if we've reached the actual starting point (first waypoint, index 0)
    if (match.path_index == 0 && match.distance < 2.0f) {
        // At the true starting point - announce completion only once
        if (!starting_point_announced_) {
            instruction.type = InstructionType::DESTINATION;
            instruction.message = "You've reached the starting point";
            instruction.priority = 1;
            starting_point_announced_ = true;
            
            std::cout << "🎯 PathGuide: You've reached the starting point" << std::endl;
            return instruction;
        } else {
            // Already announced, set state to stop but don't call StopGuidance from guidance thread
            SetState(GuidanceState::READY);
            should_stop_thread_ = true;
            instruction.type = InstructionType::START;
            instruction.message = "";
            instruction.priority = 3;
            return instruction;
        }
    } else if (match.path_index > 0) {
        // Still navigating backwards - calculate direction to previous waypoint
        const PathPoint& previous_waypoint = path_points[match.path_index - 1];
        required_direction = CalculateRequiredDirection(previous_waypoint, current_waypoint);
        
        std::cout << "🧭 PathGuide: Walking backwards to waypoint " << (match.path_index - 1) 
                  << " [Progress: " << std::fixed << std::setprecision(1) 
                  << ((float)(path_points.size() - match.path_index) / path_points.size() * 100.0f) << "%]" << std::endl;
    } else {
        // Shouldn't reach here, but handle gracefully
        instruction.type = InstructionType::ERROR_MESSAGE;
        instruction.message = "Navigation error";
        return instruction;
    }
    
    // Calculate orientation error
    float orientation_error = CalculateOrientationError(last_pose_, required_direction);
    
    // Generate contextual instruction based on orientation error  
    std::cout << "🧭 BACKWARDS DEBUG: Orientation error = " << orientation_error << "°" << std::endl;
    
    // Check for nearby turns and provide turn-aware guidance
    Eigen::Vector3f current_position = last_pose_.translation();
    const TurnInfo* nearby_turn = FindNearbyTurn(current_position, 5.0f);
    
    if (nearby_turn && backwards_mode_) {
        if (nearby_turn->is_sharp_turn) {
            if (nearby_turn->is_left_turn) {
                if (audio_guide_) {
                    audio_guide_->SpeakWithPriority("Sharp left turn - walk backwards while turning left", AudioPriority::URGENT);
                }
                instruction.type = InstructionType::TURN_LEFT;
                instruction.message = "Sharp left turn - walk backwards while turning left";
                instruction.priority = 1;
                instruction.angle_degrees = nearby_turn->turn_angle;
                return instruction;
            } else {
                if (audio_guide_) {
                    audio_guide_->SpeakWithPriority("Sharp right turn - walk backwards while turning right", AudioPriority::URGENT);
                }
                instruction.type = InstructionType::TURN_RIGHT;
                instruction.message = "Sharp right turn - walk backwards while turning right";
                instruction.priority = 1;
                instruction.angle_degrees = nearby_turn->turn_angle;
                return instruction;
            }
        } else {
            // Handle gentle turns
            if (nearby_turn->is_left_turn) {
                instruction.type = InstructionType::CORRECT_LEFT;
                instruction.message = "Gentle left turn ahead - adjust slightly while walking backwards";
                instruction.priority = 2;
                instruction.angle_degrees = nearby_turn->turn_angle;
                return instruction;
            } else {
                instruction.type = InstructionType::CORRECT_RIGHT;
                instruction.message = "Gentle right turn ahead - adjust slightly while walking backwards";
                instruction.priority = 2;
                instruction.angle_degrees = nearby_turn->turn_angle;
                return instruction;
            }
        }
    }
    
    // Check if current instruction is completed
    bool instruction_completed = false;
    if (active_instruction_ == InstructionType::TURN_AROUND && orientation_error < 30.0f) {
        instruction_completed = true;
        active_instruction_ = InstructionType::START;
        std::cout << "✅ TURN AROUND instruction completed!" << std::endl;
    } else if (active_instruction_ == InstructionType::CORRECT_LEFT || active_instruction_ == InstructionType::CORRECT_RIGHT) {
        if (orientation_error < 15.0f) {
            instruction_completed = true;
            active_instruction_ = InstructionType::START;
            std::cout << "✅ TURN instruction completed!" << std::endl;
        }
    }
    
    // Don't give new instructions if current one is not completed
    if (active_instruction_ != InstructionType::START && !instruction_completed) {
        instruction.type = InstructionType::START;
        instruction.message = "";
        instruction.priority = 3;
        return instruction;
    }
    
    // Reduce orientation instruction spamming
    static double last_orientation_warning = 0.0;
    
    // For backwards navigation, flip the orientation logic
    // If user is facing ~180° from required direction, that's correct for walking backwards
    float backwards_adjusted_error = orientation_error;
    if (orientation_error > 150.0f) {
        backwards_adjusted_error = 360.0f - orientation_error; // Convert to forward-equivalent error
    }
    
    if (backwards_adjusted_error < 30.0f || orientation_error > 150.0f) {
        // User is facing correct direction for backwards walking
        instruction.type = InstructionType::CONTINUE_FORWARD;
        instruction.message = "Walk backwards - keep looking forward";
        instruction.priority = 1;
        instruction.angle_degrees = orientation_error;
        last_orientation_warning = current_time;
    } else if (backwards_adjusted_error > 30.0f && backwards_adjusted_error < 60.0f) {
        // User needs to adjust orientation, but keep it minimal to avoid tracking loss
        Eigen::Vector3f camera_direction = ExtractCameraDirection(last_pose_);
        Eigen::Vector3f cross_product = camera_direction.cross(required_direction);
        bool turn_left = cross_product.y() > 0; // Assuming Y-up coordinate system
        
        instruction.type = turn_left ? InstructionType::CORRECT_LEFT : InstructionType::CORRECT_RIGHT;
        instruction.message = turn_left ? "Turn slightly left while walking backwards" : "Turn slightly right while walking backwards";
        instruction.priority = 2;
        instruction.angle_degrees = orientation_error;
        last_orientation_warning = current_time;
    } else {
        // Large orientation error - user is facing completely wrong way
        instruction.type = InstructionType::CONTINUE_FORWARD;
        instruction.message = "Walk backwards - keep camera steady";
        instruction.priority = 1;
        last_orientation_warning = current_time;
    }
    
    // Check for stopped movement ONLY if we haven't reached starting point
    static double last_stop_warning = 0.0;
    if (!is_moving && pose_history_.size() >= 5 && (current_time - last_stop_warning) > 10.0) {
        instruction.type = InstructionType::STOPPED_MOVING;
        if (backwards_mode_) {
            instruction.message = "You've stopped - continue walking backwards";
        } else {
            instruction.message = "You've stopped - continue forward";
        }
        instruction.priority = 2;
        last_stop_warning = current_time;
        return instruction;
    }
    
    // No instruction needed - avoid spam
    if (instruction.type == InstructionType::START) {
        instruction.type = InstructionType::ERROR_MESSAGE;
        return instruction;
    }
    
    return instruction;
}

void PathGuide::EnableBackwardsMode()
{
    SetBackwardsMode(true);
}

void PathGuide::SetOptimizationLevel(float level)
{
    if (path_optimizer_) {
        path_optimizer_->SetOptimizationLevel(level);
        std::cout << "PathGuide: Optimization level set to " << level << std::endl;
    }
}

OptimizationReport PathGuide::GetLastOptimizationReport() const
{
    if (path_optimizer_) {
        return path_optimizer_->GetLastOptimizationReport();
    }
    return OptimizationReport();
}

void PathGuide::AnalyzePathGeometry()
{
    std::lock_guard<std::mutex> lock(turns_mutex_);
    detected_turns_.clear();
    
    if (!path_matcher_ || !path_matcher_->IsPathLoaded()) {
        return;
    }
    
    const std::vector<PathPoint>& path_points = path_matcher_->GetLoadedPath();
    
    if (path_points.size() < 3) {
        std::cout << "📐 PathGeometry: Path too short for turn analysis" << std::endl;
        return;
    }
    
    std::cout << "📐 PathGeometry: Analyzing " << path_points.size() << " waypoints for turns..." << std::endl;
    
    // Analyze path for turns (need at least 3 points)
    for (size_t i = 1; i < path_points.size() - 1; ++i) {
        // Calculate direction vectors
        Eigen::Vector3f dir1 = (path_points[i].pose.translation() - 
                                path_points[i-1].pose.translation()).normalized();
        Eigen::Vector3f dir2 = (path_points[i+1].pose.translation() - 
                                path_points[i].pose.translation()).normalized();
        
        // Calculate angle between directions
        float dot_product = dir1.dot(dir2);
        dot_product = std::clamp(dot_product, -1.0f, 1.0f);
        float angle_rad = std::acos(dot_product);
        float angle_deg = angle_rad * 180.0f / M_PI;
        
        // Only consider significant turns (>45 degrees)
        if (angle_deg > 45.0f) {
            // Determine turn direction using cross product
            Eigen::Vector3f cross_product = dir1.cross(dir2);
            bool is_left_turn = cross_product.y() > 0;  // Assuming Y-up coordinate system
            bool is_sharp_turn = angle_deg > 70.0f;
            
            TurnInfo turn_info(i, angle_deg, is_left_turn, is_sharp_turn);
            detected_turns_.push_back(turn_info);
            
            std::cout << "  🔄 Turn detected at waypoint " << i << "/" << path_points.size() 
                      << ": " << std::fixed << std::setprecision(1) << angle_deg << "° "
                      << (is_left_turn ? "LEFT" : "RIGHT") 
                      << (is_sharp_turn ? " (SHARP)" : " (GENTLE)") << std::endl;
        }
    }
    
    std::cout << "✅ PathGeometry: Analysis complete - " << detected_turns_.size() 
              << " turns detected and stored" << std::endl;
    
    if (!detected_turns_.empty()) {
        std::cout << "💾 Turn data: " << detected_turns_.size() 
                  << " turns stored for backwards navigation" << std::endl;
    }
}

const TurnInfo* PathGuide::FindNearbyTurn(const Eigen::Vector3f& current_position, float search_radius) const
{
    std::lock_guard<std::mutex> lock(turns_mutex_);
    
    if (!path_matcher_ || !path_matcher_->IsPathLoaded()) {
        return nullptr;
    }
    
    const std::vector<PathPoint>& path_points = path_matcher_->GetLoadedPath();
    
    // Find the closest turn within the search radius
    const TurnInfo* closest_turn = nullptr;
    float closest_distance = search_radius + 1.0f;
    
    for (const auto& turn : detected_turns_) {
        if (turn.waypoint_index < path_points.size()) {
            Eigen::Vector3f turn_position = path_points[turn.waypoint_index].pose.translation();
            float distance = (current_position - turn_position).norm();
            
            if (distance <= search_radius && distance < closest_distance) {
                closest_turn = &turn;
                closest_distance = distance;
            }
        }
    }
    
    return closest_turn;
}

size_t PathGuide::GetDetectedTurnsCount() const
{
    std::lock_guard<std::mutex> lock(turns_mutex_);
    return detected_turns_.size();
}

} // namespace ORB_SLAM3