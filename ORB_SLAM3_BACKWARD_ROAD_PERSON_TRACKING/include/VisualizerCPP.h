/**
 * VisualizerCPP.h
 * Exact port of modules/visualization.py
 * Handles all visualization features including trails, zones, and overlays
 */

#ifndef VISUALIZER_CPP_H
#define VISUALIZER_CPP_H

#include "orbslam3_export.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <deque>
#include <chrono>

namespace ORB_SLAM3 {

struct TrailPointCPP {
    cv::Point2f point;
    std::chrono::steady_clock::time_point timestamp;
    cv::Scalar color;
};

struct ZoneVisualizationCPP {
    std::string name;
    cv::Rect2f area;
    cv::Scalar color;
    int person_count;
    std::vector<int> person_ids;
    bool is_active;
    std::chrono::steady_clock::time_point last_activity;
};

struct VisualizationConfigCPP {
    bool show_trails;
    bool show_zones; 
    bool show_ids;
    bool show_activities;
    bool show_speeds;
    bool show_confidence;
    bool show_face_names;
    bool show_fps;
    bool show_statistics;
    int trail_length;
    int font_size;
    cv::Scalar text_color;
    cv::Scalar background_color;
    float alpha_transparency;
};

class ORB_SLAM3_API VisualizerCPP {
public:
    VisualizerCPP();
    ~VisualizerCPP();
    
    // Exact port of Python Visualizer methods
    bool initialize(const VisualizationConfigCPP& config = VisualizationConfigCPP{});
    cv::Mat draw_frame(const cv::Mat& frame, 
                      const std::vector<cv::Rect2f>& detections,
                      const std::vector<int>& track_ids,
                      const std::vector<std::string>& activities,
                      const std::vector<std::string>& face_names,
                      const std::vector<float>& confidences,
                      const std::vector<cv::Point2f>& velocities,
                      double current_time);
    
    // Trail visualization methods (exact Python interface)
    void update_trails(const std::vector<int>& track_ids, 
                      const std::vector<cv::Point2f>& centers,
                      double current_time);
    void draw_trails(cv::Mat& frame);
    void clear_trail(int track_id);
    void clear_all_trails();
    
    // Zone visualization methods (exact Python interface)
    void set_zones(const std::vector<ZoneVisualizationCPP>& zones);
    void draw_zones(cv::Mat& frame);
    void update_zone_activity(const std::string& zone_name, int person_count, const std::vector<int>& person_ids);
    
    // Statistics and overlays (exact Python interface)
    void draw_statistics(cv::Mat& frame, const std::map<std::string, int>& activity_stats, 
                        float fps, int total_tracks);
    void draw_info_panel(cv::Mat& frame, const std::string& info_text);
    void draw_detection_box(cv::Mat& frame, const cv::Rect2f& bbox, const cv::Scalar& color, 
                           int thickness = 2, bool filled = false);
    
    // Configuration methods
    void set_config(const VisualizationConfigCPP& config) { config_ = config; }
    VisualizationConfigCPP get_config() const { return config_; }
    void toggle_trails() { config_.show_trails = !config_.show_trails; }
    void toggle_zones() { config_.show_zones = !config_.show_zones; }
    void set_trail_length(int length) { config_.trail_length = length; }
    
    // Color management (exact Python color system)
    cv::Scalar get_track_color(int track_id);
    cv::Scalar get_activity_color(const std::string& activity);
    cv::Scalar get_confidence_color(float confidence);
    void generate_color_palette();
    
    // Utility methods
    bool is_available() const { return initialized_; }
    void cleanup_old_trails(double current_time);
    
private:
    // Drawing helper methods
    void draw_text_with_background(cv::Mat& frame, const std::string& text, 
                                  const cv::Point& position, const cv::Scalar& text_color, 
                                  const cv::Scalar& bg_color, float scale = 0.6f);
    void draw_rounded_rectangle(cv::Mat& frame, const cv::Rect& rect, 
                               const cv::Scalar& color, int radius = 5);
    void draw_arrow(cv::Mat& frame, const cv::Point2f& start, const cv::Point2f& end, 
                   const cv::Scalar& color, int thickness = 2);
    
    // Trail management methods
    void add_trail_point(int track_id, const cv::Point2f& point, double timestamp);
    void cleanup_trail(int track_id, double current_time);
    std::vector<TrailPointCPP> get_trail_points(int track_id) const;
    
    // Color generation methods
    cv::Scalar hsv_to_bgr(float h, float s, float v);
    cv::Scalar generate_unique_color(int id);
    
private:
    // Configuration
    VisualizationConfigCPP config_;
    
    // Trail data (exact Python trail system)
    std::map<int, std::deque<TrailPointCPP>> track_trails_;
    
    // Zone data (exact Python zone system)
    std::vector<ZoneVisualizationCPP> zones_;
    
    // Color management (exact Python color system)
    std::map<int, cv::Scalar> track_colors_;
    std::vector<cv::Scalar> base_colors_;
    std::map<std::string, cv::Scalar> activity_colors_;
    
    // Performance tracking
    std::chrono::steady_clock::time_point last_cleanup_;
    
    // State
    bool initialized_;
    
    // Constants (exact Python values)
    static const int DEFAULT_TRAIL_LENGTH;
    static const float DEFAULT_ALPHA;
    static const int CLEANUP_INTERVAL_SECONDS;
    static const int MAX_TRAIL_AGE_SECONDS;
    static const cv::Scalar DEFAULT_TEXT_COLOR;
    static const cv::Scalar DEFAULT_BG_COLOR;
    
    // Font settings (exact Python font settings)
    static const int FONT_FACE;
    static const double FONT_SCALE;
    static const int FONT_THICKNESS;
};

// Default configuration factory function
ORB_SLAM3_API VisualizationConfigCPP create_default_visualization_config();

} // namespace ORB_SLAM3

#endif // VISUALIZER_CPP_H