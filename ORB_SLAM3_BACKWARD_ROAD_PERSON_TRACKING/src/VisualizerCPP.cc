/**
 * VisualizerCPP.cc
 * Exact port of modules/visualization.py implementation
 */

#include "VisualizerCPP.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace ORB_SLAM3 {

// Constants (exact Python values)
const int VisualizerCPP::DEFAULT_TRAIL_LENGTH = 30;
const float VisualizerCPP::DEFAULT_ALPHA = 0.7f;
const int VisualizerCPP::CLEANUP_INTERVAL_SECONDS = 5;
const int VisualizerCPP::MAX_TRAIL_AGE_SECONDS = 10;
const cv::Scalar VisualizerCPP::DEFAULT_TEXT_COLOR = cv::Scalar(255, 255, 255);
const cv::Scalar VisualizerCPP::DEFAULT_BG_COLOR = cv::Scalar(0, 0, 0);
const int VisualizerCPP::FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const double VisualizerCPP::FONT_SCALE = 0.6;
const int VisualizerCPP::FONT_THICKNESS = 2;

VisualizerCPP::VisualizerCPP()
    : initialized_(false)
{
    // Initialize default configuration (exact Python defaults)
    config_.show_trails = true;
    config_.show_zones = true;
    config_.show_ids = true;
    config_.show_activities = true;
    config_.show_speeds = true;
    config_.show_confidence = true;
    config_.show_face_names = true;
    config_.show_fps = true;
    config_.show_statistics = true;
    config_.trail_length = DEFAULT_TRAIL_LENGTH;
    config_.font_size = 12;
    config_.text_color = DEFAULT_TEXT_COLOR;
    config_.background_color = DEFAULT_BG_COLOR;
    config_.alpha_transparency = DEFAULT_ALPHA;
    
    last_cleanup_ = std::chrono::steady_clock::now();
}

VisualizerCPP::~VisualizerCPP() = default;

bool VisualizerCPP::initialize(const VisualizationConfigCPP& config)
{
    std::cout << "🔧 Initializing Visualizer C++..." << std::endl;
    
    config_ = config;
    
    // Generate base color palette (exact Python color generation)
    generate_color_palette();
    
    // Initialize activity colors (exact Python activity colors)
    activity_colors_["Moving"] = cv::Scalar(0, 255, 0);    // Green
    activity_colors_["Stop"] = cv::Scalar(0, 0, 255);      // Red
    activity_colors_["Unknown"] = cv::Scalar(128, 128, 128); // Gray
    activity_colors_["Standing"] = cv::Scalar(255, 255, 0);  // Yellow
    activity_colors_["Walking"] = cv::Scalar(0, 255, 255);   // Cyan
    activity_colors_["Running"] = cv::Scalar(255, 0, 255);   // Magenta
    
    initialized_ = true;
    std::cout << "✅ Visualizer C++ initialized" << std::endl;
    return true;
}

cv::Mat VisualizerCPP::draw_frame(const cv::Mat& frame,
                                 const std::vector<cv::Rect2f>& detections,
                                 const std::vector<int>& track_ids,
                                 const std::vector<std::string>& activities,
                                 const std::vector<std::string>& face_names,
                                 const std::vector<float>& confidences,
                                 const std::vector<cv::Point2f>& velocities,
                                 double current_time)
{
    if (!initialized_ || frame.empty()) {
        return frame.clone();
    }
    
    cv::Mat result = frame.clone();
    
    // Draw zones first (background layer - exact Python layer order)
    if (config_.show_zones) {
        draw_zones(result);
    }
    
    // Draw trails (exact Python trail rendering)
    if (config_.show_trails) {
        draw_trails(result);
    }
    
    // Draw detection boxes and information (exact Python detection rendering)
    for (size_t i = 0; i < detections.size(); ++i) {
        const cv::Rect2f& bbox = detections[i];
        int track_id = (i < track_ids.size()) ? track_ids[i] : -1;
        std::string activity = (i < activities.size()) ? activities[i] : "Unknown";
        std::string face_name = (i < face_names.size()) ? face_names[i] : "";
        float confidence = (i < confidences.size()) ? confidences[i] : 0.0f;
        cv::Point2f velocity = (i < velocities.size()) ? velocities[i] : cv::Point2f(0, 0);
        
        // Get colors (exact Python color system)
        cv::Scalar box_color = get_track_color(track_id);
        cv::Scalar activity_color = get_activity_color(activity);
        
        // Draw detection box (exact Python box style)
        draw_detection_box(result, bbox, box_color, 2, false);
        
        // Prepare info text (exact Python text format)
        std::vector<std::string> info_lines;
        
        if (config_.show_ids && track_id >= 0) {
            info_lines.push_back("ID: " + std::to_string(track_id));
        }
        
        if (config_.show_activities) {
            info_lines.push_back("Act: " + activity);
        }
        
        if (config_.show_face_names && !face_name.empty()) {
            info_lines.push_back("Name: " + face_name);
        }
        
        if (config_.show_confidence) {
            char conf_str[16];
            sprintf(conf_str, "Conf: %.2f", confidence);
            info_lines.push_back(std::string(conf_str));
        }
        
        if (config_.show_speeds) {
            float speed = std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
            char speed_str[16];
            sprintf(speed_str, "Speed: %.1f", speed);
            info_lines.push_back(std::string(speed_str));
        }
        
        // Draw info text (exact Python text positioning)
        cv::Point text_pos(static_cast<int>(bbox.x), static_cast<int>(bbox.y) - 5);
        for (size_t j = 0; j < info_lines.size(); ++j) {
            draw_text_with_background(result, info_lines[j], 
                                    cv::Point(text_pos.x, text_pos.y - j * 20),
                                    config_.text_color, config_.background_color, FONT_SCALE);
        }
        
        // Draw velocity arrow (exact Python arrow rendering)
        if (config_.show_speeds && (velocity.x != 0 || velocity.y != 0)) {
            cv::Point2f center(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height * 0.5f);
            cv::Point2f arrow_end = center + velocity * 10.0f; // Scale factor
            draw_arrow(result, center, arrow_end, activity_color, 2);
        }
    }
    
    return result;
}

void VisualizerCPP::update_trails(const std::vector<int>& track_ids,
                                 const std::vector<cv::Point2f>& centers,
                                 double current_time)
{
    if (!config_.show_trails) {
        return;
    }
    
    // Add trail points for each track (exact Python trail update)
    for (size_t i = 0; i < track_ids.size() && i < centers.size(); ++i) {
        add_trail_point(track_ids[i], centers[i], current_time);
    }
    
    // Cleanup old trails periodically (exact Python cleanup)
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_cleanup_).count() >= CLEANUP_INTERVAL_SECONDS) {
        cleanup_old_trails(current_time);
        last_cleanup_ = now;
    }
}

void VisualizerCPP::draw_trails(cv::Mat& frame)
{
    if (!config_.show_trails) {
        return;
    }
    
    // Draw trails for each track (exact Python trail rendering)
    for (const auto& pair : track_trails_) {
        int track_id = pair.first;
        const std::deque<TrailPointCPP>& trail = pair.second;
        
        if (trail.size() < 2) {
            continue;
        }
        
        cv::Scalar trail_color = get_track_color(track_id);
        
        // Draw trail lines with fading effect (exact Python fading)
        for (size_t i = 1; i < trail.size(); ++i) {
            float alpha = static_cast<float>(i) / trail.size();
            cv::Scalar faded_color = trail_color * alpha;
            
            cv::line(frame, trail[i-1].point, trail[i].point, faded_color, 2);
        }
        
        // Draw trail points (exact Python point rendering)
        for (size_t i = 0; i < trail.size(); ++i) {
            float alpha = static_cast<float>(i) / trail.size() * config_.alpha_transparency;
            cv::Scalar point_color = trail_color * alpha;
            cv::circle(frame, trail[i].point, 3, point_color, -1);
        }
    }
}

void VisualizerCPP::set_zones(const std::vector<ZoneVisualizationCPP>& zones)
{
    zones_ = zones;
}

void VisualizerCPP::draw_zones(cv::Mat& frame)
{
    if (!config_.show_zones) {
        return;
    }
    
    // Draw zones (exact Python zone rendering)
    for (const auto& zone : zones_) {
        // Draw zone rectangle with transparency (exact Python style)
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, zone.area, zone.color, -1);
        cv::addWeighted(frame, 1.0 - config_.alpha_transparency, overlay, config_.alpha_transparency, 0, frame);
        
        // Draw zone border (exact Python border style)
        cv::rectangle(frame, zone.area, zone.color, 2);
        
        // Draw zone info (exact Python zone info)
        std::string zone_info = zone.name + " (" + std::to_string(zone.person_count) + ")";
        cv::Point text_pos(static_cast<int>(zone.area.x + 5), static_cast<int>(zone.area.y + 25));
        draw_text_with_background(frame, zone_info, text_pos, config_.text_color, 
                                config_.background_color, FONT_SCALE);
    }
}

void VisualizerCPP::update_zone_activity(const std::string& zone_name, int person_count, 
                                        const std::vector<int>& person_ids)
{
    // Update zone information (exact Python zone update)
    for (auto& zone : zones_) {
        if (zone.name == zone_name) {
            zone.person_count = person_count;
            zone.person_ids = person_ids;
            zone.is_active = person_count > 0;
            zone.last_activity = std::chrono::steady_clock::now();
            break;
        }
    }
}

void VisualizerCPP::draw_statistics(cv::Mat& frame, const std::map<std::string, int>& activity_stats,
                                   float fps, int total_tracks)
{
    if (!config_.show_statistics) {
        return;
    }
    
    // Prepare statistics text (exact Python statistics format)
    std::vector<std::string> stats_lines;
    
    if (config_.show_fps) {
        char fps_str[32];
        sprintf(fps_str, "FPS: %.1f", fps);
        stats_lines.push_back(std::string(fps_str));
    }
    
    stats_lines.push_back("Tracks: " + std::to_string(total_tracks));
    
    // Add activity statistics (exact Python activity stats)
    for (const auto& pair : activity_stats) {
        stats_lines.push_back(pair.first + ": " + std::to_string(pair.second));
    }
    
    // Draw statistics panel (exact Python panel style)
    cv::Point panel_pos(10, 30);
    cv::Rect panel_rect(5, 5, 200, 25 * stats_lines.size() + 10);
    
    // Draw background panel
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, panel_rect, cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(frame, 0.7, overlay, 0.3, 0, frame);
    
    // Draw statistics text
    for (size_t i = 0; i < stats_lines.size(); ++i) {
        cv::Point text_pos(panel_pos.x, panel_pos.y + i * 25);
        cv::putText(frame, stats_lines[i], text_pos, FONT_FACE, FONT_SCALE, 
                   config_.text_color, FONT_THICKNESS);
    }
}

cv::Scalar VisualizerCPP::get_track_color(int track_id)
{
    if (track_id < 0) {
        return cv::Scalar(128, 128, 128); // Gray for unknown tracks
    }
    
    // Get or generate color for track (exact Python color system)
    if (track_colors_.find(track_id) == track_colors_.end()) {
        track_colors_[track_id] = generate_unique_color(track_id);
    }
    
    return track_colors_[track_id];
}

cv::Scalar VisualizerCPP::get_activity_color(const std::string& activity)
{
    auto it = activity_colors_.find(activity);
    return (it != activity_colors_.end()) ? it->second : cv::Scalar(128, 128, 128);
}

cv::Scalar VisualizerCPP::get_confidence_color(float confidence)
{
    // Color based on confidence level (exact Python confidence colors)
    if (confidence > 0.8f) {
        return cv::Scalar(0, 255, 0);      // Green - high confidence
    } else if (confidence > 0.5f) {
        return cv::Scalar(0, 255, 255);    // Yellow - medium confidence  
    } else {
        return cv::Scalar(0, 0, 255);      // Red - low confidence
    }
}

void VisualizerCPP::generate_color_palette()
{
    // Generate diverse color palette (exact Python palette generation)
    base_colors_.clear();
    
    const int num_colors = 20;
    for (int i = 0; i < num_colors; ++i) {
        float hue = (i * 360.0f / num_colors);
        cv::Scalar color = hsv_to_bgr(hue, 1.0f, 1.0f);
        base_colors_.push_back(color);
    }
}

cv::Scalar VisualizerCPP::generate_unique_color(int id)
{
    if (!base_colors_.empty()) {
        return base_colors_[id % base_colors_.size()];
    }
    
    // Fallback color generation (exact Python fallback)
    float hue = (id * 137.5f); // Golden angle
    return hsv_to_bgr(fmod(hue, 360.0f), 0.8f, 0.9f);
}

cv::Scalar VisualizerCPP::hsv_to_bgr(float h, float s, float v)
{
    // HSV to BGR conversion (exact Python conversion)
    cv::Mat hsv(1, 1, CV_32FC3, cv::Scalar(h, s * 255, v * 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    
    cv::Vec3f pixel = bgr.at<cv::Vec3f>(0, 0);
    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

void VisualizerCPP::add_trail_point(int track_id, const cv::Point2f& point, double timestamp)
{
    TrailPointCPP trail_point;
    trail_point.point = point;
    trail_point.timestamp = std::chrono::steady_clock::now();
    trail_point.color = get_track_color(track_id);
    
    track_trails_[track_id].push_back(trail_point);
    
    // Trim trail to max length (exact Python trimming)
    while (track_trails_[track_id].size() > static_cast<size_t>(config_.trail_length)) {
        track_trails_[track_id].pop_front();
    }
}

void VisualizerCPP::cleanup_old_trails(double current_time)
{
    auto now = std::chrono::steady_clock::now();
    
    // Remove old trail points (exact Python cleanup)
    for (auto& pair : track_trails_) {
        auto& trail = pair.second;
        auto it = trail.begin();
        
        while (it != trail.end()) {
            auto age = std::chrono::duration_cast<std::chrono::seconds>(now - it->timestamp).count();
            if (age > MAX_TRAIL_AGE_SECONDS) {
                it = trail.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    // Remove empty trails
    auto it = track_trails_.begin();
    while (it != track_trails_.end()) {
        if (it->second.empty()) {
            it = track_trails_.erase(it);
        } else {
            ++it;
        }
    }
}

void VisualizerCPP::draw_text_with_background(cv::Mat& frame, const std::string& text,
                                             const cv::Point& position, const cv::Scalar& text_color,
                                             const cv::Scalar& bg_color, float scale)
{
    // Get text size (exact Python text sizing)
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, FONT_FACE, scale, FONT_THICKNESS, &baseline);
    
    // Draw background rectangle (exact Python background)
    cv::Rect bg_rect(position.x - 2, position.y - text_size.height - 2,
                    text_size.width + 4, text_size.height + baseline + 4);
    cv::rectangle(frame, bg_rect, bg_color, -1);
    
    // Draw text (exact Python text rendering)
    cv::putText(frame, text, position, FONT_FACE, scale, text_color, FONT_THICKNESS);
}

void VisualizerCPP::draw_detection_box(cv::Mat& frame, const cv::Rect2f& bbox, const cv::Scalar& color,
                                      int thickness, bool filled)
{
    if (filled) {
        cv::rectangle(frame, bbox, color, -1);
    } else {
        cv::rectangle(frame, bbox, color, thickness);
    }
}

void VisualizerCPP::draw_arrow(cv::Mat& frame, const cv::Point2f& start, const cv::Point2f& end,
                              const cv::Scalar& color, int thickness)
{
    // Draw arrow line (exact Python arrow rendering)
    cv::line(frame, start, end, color, thickness);
    
    // Calculate arrow head (exact Python arrow head calculation)
    cv::Point2f direction = end - start;
    float length = std::sqrt(direction.x * direction.x + direction.y * direction.y);
    
    if (length > 0) {
        direction /= length;
        cv::Point2f perpendicular(-direction.y, direction.x);
        
        cv::Point2f arrow_point1 = end - direction * 10 + perpendicular * 5;
        cv::Point2f arrow_point2 = end - direction * 10 - perpendicular * 5;
        
        cv::line(frame, end, arrow_point1, color, thickness);
        cv::line(frame, end, arrow_point2, color, thickness);
    }
}

void VisualizerCPP::clear_trail(int track_id)
{
    track_trails_.erase(track_id);
}

void VisualizerCPP::clear_all_trails()
{
    track_trails_.clear();
}

VisualizationConfigCPP create_default_visualization_config()
{
    VisualizationConfigCPP config;
    config.show_trails = true;
    config.show_zones = true;
    config.show_ids = true;
    config.show_activities = true;
    config.show_speeds = true;
    config.show_confidence = true;
    config.show_face_names = true;
    config.show_fps = true;
    config.show_statistics = true;
    config.trail_length = 30;
    config.font_size = 12;
    config.text_color = cv::Scalar(255, 255, 255);
    config.background_color = cv::Scalar(0, 0, 0);
    config.alpha_transparency = 0.7f;
    return config;
}

} // namespace ORB_SLAM3