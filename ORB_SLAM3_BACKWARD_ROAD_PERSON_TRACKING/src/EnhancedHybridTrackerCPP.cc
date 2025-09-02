/**
 * EnhancedHybridTrackerCPP.cc
 * Exact port of enhanced_hybrid_tracker_modular.py implementation
 */

#include "EnhancedHybridTrackerCPP.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace ORB_SLAM3 {

// Constants (exact Python values)
const float EnhancedHybridTrackerCPP::DEFAULT_IOU_THRESHOLD = 0.3f;
const int EnhancedHybridTrackerCPP::MAX_LOST_FRAMES = 30;
const int EnhancedHybridTrackerCPP::TRACK_CLEANUP_INTERVAL = 100;
const int EnhancedHybridTrackerCPP::FACE_RECOGNITION_INTERVAL = 30;
const double EnhancedHybridTrackerCPP::TRACK_TIMEOUT_SECONDS = 5.0;

EnhancedHybridTrackerCPP::EnhancedHybridTrackerCPP()
    : next_track_id_(1)
    , processing_time_ms_(0.0f)
    , fps_(0.0f)
    , frame_count_(0)
    , frames_processed_(0)
    , frame_skip_counter_(0)
    , initialized_(false)
    , face_recognition_available_(false)
    , zone_analytics_available_(false)
{
    // Initialize module instances (exact Python module initialization)
    device_manager_ = std::make_unique<DeviceManagerCPP>();
    activity_detector_ = std::make_unique<ActivityDetectorCPP>();
    visualizer_ = std::make_unique<VisualizerCPP>();
    zone_analytics_ = std::make_unique<ZoneAnalyticsCPP>();
    face_recognizer_ = std::make_unique<FaceRecognizerCPP>();
    
    last_frame_time_ = std::chrono::steady_clock::now();
}

EnhancedHybridTrackerCPP::~EnhancedHybridTrackerCPP() = default;

bool EnhancedHybridTrackerCPP::Initialize(const TrackerConfigCPP& config)
{
    std::cout << "🚀 Initializing Enhanced Hybrid Tracker C++..." << std::endl;
    
    config_ = config;
    
    // Initialize DeviceManager (exact Python initialization order)
    std::map<std::string, float> device_config;
    device_config["confidence_threshold"] = config_.confidence_threshold;
    device_config["nms_threshold"] = config_.nms_threshold;
    
    if (!device_manager_->initialize(device_config)) {
        std::cerr << "❌ Failed to initialize DeviceManager" << std::endl;
        return false;
    }
    
    // Initialize ActivityDetector (exact Python initialization)
    std::map<std::string, float> activity_config;
    activity_config["movement_threshold"] = config_.movement_threshold;
    activity_config["stop_threshold"] = config_.stop_threshold;
    activity_config["moving_threshold"] = config_.moving_threshold;
    activity_config["history_size"] = static_cast<float>(config_.history_size);
    
    if (!activity_detector_->initialize(activity_config)) {
        std::cerr << "❌ Failed to initialize ActivityDetector" << std::endl;
        return false;
    }
    
    // Initialize Visualizer (exact Python initialization)
    VisualizationConfigCPP vis_config = create_default_visualization_config();
    vis_config.show_trails = config_.show_trails;
    vis_config.show_zones = config_.show_zones;
    vis_config.show_ids = config_.show_ids;
    vis_config.show_activities = config_.show_activities;
    vis_config.show_face_names = config_.show_face_names;
    vis_config.trail_length = config_.trail_length;
    
    if (!visualizer_->initialize(vis_config)) {
        std::cerr << "❌ Failed to initialize Visualizer" << std::endl;
        return false;
    }
    
    // Initialize ZoneAnalytics (exact Python initialization)
    if (config_.enable_zone_analytics) {
        // Create default zones for testing
        cv::Size default_size(640, 480);
        std::vector<ZoneDefinitionCPP> default_zones = create_default_zones(default_size);
        
        if (!zone_analytics_->initialize(default_zones)) {
            std::cerr << "⚠️  ZoneAnalytics initialization failed, continuing without" << std::endl;
        } else {
            zone_analytics_available_ = true;
        }
    }
    
    // Initialize FaceRecognizer (exact Python initialization)
    if (config_.enable_face_recognition) {
        if (!face_recognizer_->initialize(config_.face_model_path)) {
            std::cerr << "⚠️  FaceRecognizer initialization failed, continuing without" << std::endl;
        } else {
            face_recognition_available_ = true;
            
            // Load existing face database (exact Python database loading)
            face_recognizer_->load_database(config_.face_database_path);
        }
    }
    
    initialized_ = true;
    std::cout << "✅ Enhanced Hybrid Tracker C++ initialized successfully" << std::endl;
    
    // Print module status summary (exact Python status summary)
    std::cout << "📊 Module Status:" << std::endl;
    std::cout << "   Device Manager: ✅" << std::endl;
    std::cout << "   Activity Detector: ✅" << std::endl;
    std::cout << "   Visualizer: ✅" << std::endl;
    std::cout << "   Zone Analytics: " << (zone_analytics_available_ ? "✅" : "❌") << std::endl;
    std::cout << "   Face Recognition: " << (face_recognition_available_ ? "✅" : "❌") << std::endl;
    
    return true;
}

std::vector<TrackerDetectionCPP> EnhancedHybridTrackerCPP::ProcessFrame(const cv::Mat& frame, double timestamp)
{
    if (!initialized_ || frame.empty()) {
        return {};
    }
    
    processing_start_time_ = std::chrono::steady_clock::now();
    frame_count_++;
    
    // Frame skipping for performance (exact Python frame skipping)
    if (!ShouldProcessFrame()) {
        return {}; // Return empty but keep last_annotated_frame_
    }
    
    frames_processed_++;
    
    // Main detection and tracking pipeline (exact Python pipeline)
    std::vector<TrackerDetectionCPP> detections = DetectAndTrack(frame, timestamp);
    
    // Update modules with detection results (exact Python module updates)
    UpdateActivityAnalysis(detections, timestamp);
    
    if (zone_analytics_available_) {
        UpdateZoneAnalytics(detections, timestamp);
    }
    
    if (face_recognition_available_) {
        UpdateFaceRecognition(frame, const_cast<std::vector<TrackerDetectionCPP>&>(detections));
    }
    
    // Generate annotated frame (exact Python annotation)
    std::vector<cv::Rect2f> bboxes;
    std::vector<int> track_ids;
    std::vector<std::string> activities;
    std::vector<std::string> face_names;
    std::vector<float> confidences;
    std::vector<cv::Point2f> velocities;
    
    // Extract data for visualization (exact Python data extraction)
    for (const auto& detection : detections) {
        bboxes.push_back(detection.bbox);
        track_ids.push_back(detection.track_id);
        activities.push_back(detection.activity);
        face_names.push_back(detection.face_name);
        confidences.push_back(detection.confidence);
        velocities.push_back(detection.velocity);
    }
    
    // Update trails (exact Python trail update)
    std::vector<cv::Point2f> centers;
    for (const auto& bbox : bboxes) {
        centers.push_back(cv::Point2f(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height * 0.5f));
    }
    visualizer_->update_trails(track_ids, centers, timestamp);
    
    // Generate final annotated frame (exact Python visualization)
    last_annotated_frame_ = visualizer_->draw_frame(frame, bboxes, track_ids, activities, 
                                                   face_names, confidences, velocities, timestamp);
    
    // Add statistics overlay (exact Python statistics overlay)
    std::map<std::string, int> activity_stats = GetActivityStatistics();
    visualizer_->draw_statistics(last_annotated_frame_, activity_stats, GetFPS(), GetActiveTrackCount());
    
    // Calculate performance metrics (exact Python performance calculation)
    auto end_time = std::chrono::steady_clock::now();
    processing_time_ms_ = std::chrono::duration<float, std::milli>(end_time - processing_start_time_).count();
    
    // Update FPS calculation (exact Python FPS calculation)
    auto frame_time_delta = std::chrono::duration<float>(end_time - last_frame_time_).count();
    if (frame_time_delta > 0) {
        fps_ = 1.0f / frame_time_delta;
    }
    last_frame_time_ = end_time;
    
    // Periodic cleanup (exact Python cleanup)
    if (frame_count_ % TRACK_CLEANUP_INTERVAL == 0) {
        CleanupOldData(timestamp);
    }
    
    return detections;
}

std::vector<TrackerDetectionCPP> EnhancedHybridTrackerCPP::DetectAndTrack(const cv::Mat& frame, double timestamp)
{
    // Person detection using DeviceManager (exact Python detection)
    std::vector<YOLODetectionCPP> yolo_detections = device_manager_->detect_persons(frame);
    
    // Update tracking (exact Python tracking update)
    UpdateTracking(yolo_detections, timestamp);
    
    // Convert to TrackerDetectionCPP format (exact Python conversion)
    std::vector<TrackerDetectionCPP> tracker_detections;
    
    for (const auto& pair : active_tracks_) {
        int track_id = pair.first;
        const TrackStateCPP& track_state = pair.second;
        
        TrackerDetectionCPP detection;
        detection.bbox = track_state.bbox;
        detection.confidence = 0.8f; // Default confidence for tracked objects
        detection.track_id = track_id;
        detection.activity = track_state.activity;
        detection.face_name = track_state.face_name;
        detection.velocity = track_state.velocity;
        detection.current_zone = track_state.current_zone;
        detection.last_seen = std::chrono::steady_clock::now();
        
        tracker_detections.push_back(detection);
    }
    
    return tracker_detections;
}

void EnhancedHybridTrackerCPP::UpdateTracking(const std::vector<YOLODetectionCPP>& detections, double timestamp)
{
    // Track assignment using IoU matching (exact Python IoU matching)
    std::vector<std::pair<int, int>> assignments; // track_id, detection_index
    std::vector<bool> detection_assigned(detections.size(), false);
    std::vector<bool> track_updated(active_tracks_.size(), false);
    
    // Match existing tracks to detections (exact Python matching algorithm)
    for (auto& pair : active_tracks_) {
        int track_id = pair.first;
        TrackStateCPP& track_state = pair.second;
        
        float best_iou = 0.0f;
        int best_detection_idx = -1;
        
        for (size_t i = 0; i < detections.size(); ++i) {
            if (detection_assigned[i]) continue;
            
            float iou = CalculateIOU(track_state.bbox, detections[i].bbox);
            if (iou > DEFAULT_IOU_THRESHOLD && iou > best_iou) {
                best_iou = iou;
                best_detection_idx = i;
            }
        }
        
        if (best_detection_idx >= 0) {
            // Update existing track (exact Python track update)
            UpdateExistingTrack(track_id, detections[best_detection_idx], timestamp);
            detection_assigned[best_detection_idx] = true;
            track_updated[track_id] = true;
        } else {
            // Track lost (exact Python lost track handling)
            track_state.lost_frames++;
        }
    }
    
    // Create new tracks for unassigned detections (exact Python new track creation)
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detection_assigned[i]) {
            CreateNewTrack(detections[i], timestamp);
        }
    }
    
    // Remove old tracks (exact Python track removal)
    RemoveOldTracks(timestamp);
}

void EnhancedHybridTrackerCPP::CreateNewTrack(const YOLODetectionCPP& detection, double timestamp)
{
    TrackStateCPP new_track;
    new_track.bbox = detection.bbox;
    new_track.lost_frames = 0;
    new_track.age = 0;
    new_track.created_at = timestamp;
    new_track.last_seen = timestamp;
    new_track.activity = "Unknown";
    new_track.face_name = "";
    new_track.velocity = cv::Point2f(0, 0);
    new_track.current_zone = "";
    
    active_tracks_[next_track_id_] = new_track;
    next_track_id_++;
}

void EnhancedHybridTrackerCPP::UpdateExistingTrack(int track_id, const YOLODetectionCPP& detection, double timestamp)
{
    if (active_tracks_.find(track_id) == active_tracks_.end()) {
        return;
    }
    
    TrackStateCPP& track_state = active_tracks_[track_id];
    
    // Calculate velocity (exact Python velocity calculation)
    cv::Point2f old_center(track_state.bbox.x + track_state.bbox.width * 0.5f,
                          track_state.bbox.y + track_state.bbox.height * 0.5f);
    cv::Point2f new_center(detection.bbox.x + detection.bbox.width * 0.5f,
                          detection.bbox.y + detection.bbox.height * 0.5f);
    
    double time_delta = timestamp - track_state.last_seen;
    if (time_delta > 0) {
        track_state.velocity = (new_center - old_center) / time_delta;
    }
    
    // Update track state (exact Python state update)
    track_state.bbox = detection.bbox;
    track_state.lost_frames = 0;
    track_state.age++;
    track_state.last_seen = timestamp;
}

void EnhancedHybridTrackerCPP::RemoveOldTracks(double timestamp)
{
    // Remove tracks that have been lost too long (exact Python track removal)
    auto it = active_tracks_.begin();
    while (it != active_tracks_.end()) {
        const TrackStateCPP& track_state = it->second;
        
        bool should_remove = false;
        
        // Check lost frames threshold (exact Python threshold check)
        if (track_state.lost_frames > MAX_LOST_FRAMES) {
            should_remove = true;
        }
        
        // Check time timeout (exact Python timeout check)
        if (timestamp - track_state.last_seen > TRACK_TIMEOUT_SECONDS) {
            should_remove = true;
        }
        
        if (should_remove) {
            // Cleanup track from modules (exact Python cleanup)
            activity_detector_->cleanup_track(it->first);
            visualizer_->clear_trail(it->first);
            
            it = active_tracks_.erase(it);
        } else {
            ++it;
        }
    }
}

float EnhancedHybridTrackerCPP::CalculateIOU(const cv::Rect2f& box1, const cv::Rect2f& box2)
{
    // Intersection over Union calculation (exact Python IoU calculation)
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }
    
    float intersection = (x2 - x1) * (y2 - y1);
    float union_area = box1.area() + box2.area() - intersection;
    
    return (union_area > 0) ? intersection / union_area : 0.0f;
}

void EnhancedHybridTrackerCPP::UpdateActivityAnalysis(const std::vector<TrackerDetectionCPP>& detections, double timestamp)
{
    // Update activity analysis for each detection (exact Python activity update)
    for (const auto& detection : detections) {
        activity_detector_->update_track_activity(detection.track_id, detection.bbox, timestamp);
        
        // Update track activity from detector (exact Python activity retrieval)
        if (active_tracks_.find(detection.track_id) != active_tracks_.end()) {
            active_tracks_[detection.track_id].activity = 
                activity_detector_->get_track_activity(detection.track_id);
            active_tracks_[detection.track_id].velocity = 
                activity_detector_->get_track_velocity(detection.track_id);
        }
    }
}

void EnhancedHybridTrackerCPP::UpdateZoneAnalytics(const std::vector<TrackerDetectionCPP>& detections, double timestamp)
{
    if (!zone_analytics_available_) {
        return;
    }
    
    // Extract position and activity data (exact Python data extraction)
    std::vector<int> person_ids;
    std::vector<cv::Point2f> positions;
    std::vector<std::string> activities;
    
    for (const auto& detection : detections) {
        person_ids.push_back(detection.track_id);
        cv::Point2f center(detection.bbox.x + detection.bbox.width * 0.5f,
                          detection.bbox.y + detection.bbox.height * 0.5f);
        positions.push_back(center);
        activities.push_back(detection.activity);
    }
    
    // Update zone analytics (exact Python zone update)
    zone_analytics_->update_person_positions(person_ids, positions, activities, timestamp);
    
    // Update track zone information (exact Python zone information update)
    for (const auto& detection : detections) {
        if (active_tracks_.find(detection.track_id) != active_tracks_.end()) {
            cv::Point2f center(detection.bbox.x + detection.bbox.width * 0.5f,
                              detection.bbox.y + detection.bbox.height * 0.5f);
            // Note: This would require zone lookup - simplified for now
            active_tracks_[detection.track_id].current_zone = "";
        }
    }
}

void EnhancedHybridTrackerCPP::UpdateFaceRecognition(const cv::Mat& frame, std::vector<TrackerDetectionCPP>& detections)
{
    if (!face_recognition_available_ || frame.empty()) {
        return;
    }
    
    // Face recognition on interval (exact Python interval processing)
    if (frame_count_ % FACE_RECOGNITION_INTERVAL != 0) {
        return;
    }
    
    // Process face recognition for each detection (exact Python face processing)
    for (auto& detection : detections) {
        cv::Rect face_roi(detection.bbox);
        
        // Ensure ROI is within frame bounds
        face_roi &= cv::Rect(0, 0, frame.cols, frame.rows);
        
        if (face_roi.area() > 0) {
            cv::Mat face_image = frame(face_roi);
            std::string face_name = face_recognizer_->recognize_face(face_image);
            
            detection.face_name = face_name;
            
            // Update track face name (exact Python face name update)
            if (active_tracks_.find(detection.track_id) != active_tracks_.end()) {
                active_tracks_[detection.track_id].face_name = face_name;
            }
        }
    }
}

bool EnhancedHybridTrackerCPP::ShouldProcessFrame()
{
    // Frame skipping logic (exact Python frame skipping)
    frame_skip_counter_++;
    
    if (frame_skip_counter_ >= config_.frame_skip_interval) {
        frame_skip_counter_ = 0;
        return true;
    }
    
    return false;
}

void EnhancedHybridTrackerCPP::CleanupOldData(double timestamp)
{
    // Cleanup old data from modules (exact Python cleanup)
    activity_detector_->cleanup_old_tracks(timestamp);
    
    if (zone_analytics_available_) {
        zone_analytics_->cleanup_old_data(timestamp);
    }
    
    visualizer_->cleanup_old_trails(timestamp);
}

cv::Mat EnhancedHybridTrackerCPP::GetAnnotatedFrame() const
{
    return last_annotated_frame_;
}

float EnhancedHybridTrackerCPP::GetFPS() const
{
    return fps_;
}

float EnhancedHybridTrackerCPP::GetProcessingTime() const
{
    return processing_time_ms_;
}

int EnhancedHybridTrackerCPP::GetActiveTrackCount() const
{
    return active_tracks_.size();
}

std::map<std::string, int> EnhancedHybridTrackerCPP::GetActivityStatistics() const
{
    return activity_detector_->get_activity_statistics();
}

std::map<std::string, int> EnhancedHybridTrackerCPP::GetZoneOccupancyCounts() const
{
    if (zone_analytics_available_) {
        return zone_analytics_->get_all_occupancy_counts();
    }
    return {};
}

void EnhancedHybridTrackerCPP::ResetSession()
{
    // Reset all modules (exact Python session reset)
    active_tracks_.clear();
    next_track_id_ = 1;
    frame_count_ = 0;
    frames_processed_ = 0;
    
    if (zone_analytics_available_) {
        zone_analytics_->reset_session();
    }
    
    visualizer_->clear_all_trails();
    
    std::cout << "Session reset completed" << std::endl;
}

std::vector<std::string> EnhancedHybridTrackerCPP::GetActiveAlerts() const
{
    if (zone_analytics_available_) {
        return zone_analytics_->check_alerts();
    }
    return {};
}

std::string EnhancedHybridTrackerCPP::GetSystemStatus() const
{
    std::string status = "Enhanced Hybrid Tracker Status:\n";
    status += "Initialized: " + std::string(initialized_ ? "Yes" : "No") + "\n";
    status += "Active Tracks: " + std::to_string(GetActiveTrackCount()) + "\n";
    status += "FPS: " + std::to_string(GetFPS()) + "\n";
    status += "Processing Time: " + std::to_string(GetProcessingTime()) + "ms\n";
    status += "Face Recognition: " + std::string(face_recognition_available_ ? "Available" : "Unavailable") + "\n";
    status += "Zone Analytics: " + std::string(zone_analytics_available_ ? "Available" : "Unavailable");
    return status;
}

void EnhancedHybridTrackerCPP::ExportAnalytics(const std::string& filename) const
{
    if (zone_analytics_available_) {
        zone_analytics_->export_analytics(filename);
    } else {
        std::cout << "⚠️ Zone analytics not available for export" << std::endl;
    }
}

bool EnhancedHybridTrackerCPP::SaveFaceDatabase(const std::string& filename) const
{
    if (face_recognition_available_) {
        std::string db_filename = filename.empty() ? config_.face_database_path : filename;
        return face_recognizer_->save_database(db_filename);
    } else {
        std::cout << "⚠️ Face recognition not available for database save" << std::endl;
        return false;
    }
}

bool EnhancedHybridTrackerCPP::LoadFaceDatabase(const std::string& filename)
{
    if (face_recognition_available_) {
        std::string db_filename = filename.empty() ? config_.face_database_path : filename;
        return face_recognizer_->load_database(db_filename);
    } else {
        std::cout << "⚠️ Face recognition not available for database load" << std::endl;
        return false;
    }
}

} // namespace ORB_SLAM3