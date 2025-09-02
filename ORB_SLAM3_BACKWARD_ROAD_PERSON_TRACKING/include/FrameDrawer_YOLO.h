/**
* FrameDrawer with YOLO Integration
* Enhanced version that overlays YOLO detections on ORB-SLAM3 feature visualization
*/

#ifndef FRAMEDRAWER_YOLO_H
#define FRAMEDRAWER_YOLO_H

#include "FrameDrawer.h"
#include "YOLOInterface.h"
#include <deque>

namespace ORB_SLAM3
{

class ORB_SLAM3_API FrameDrawerYOLO : public FrameDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    FrameDrawerYOLO(Atlas* pAtlas);
    ~FrameDrawerYOLO();
    
    // Initialize YOLO interface
    bool InitializeYOLO(const std::string& config_path = "");
    
    // Enhanced DrawFrame methods with YOLO detections
    cv::Mat DrawFrame(float imageScale=1.f);
    cv::Mat DrawRightFrame(float imageScale=1.f);
    
    // Enable/disable YOLO overlay
    void SetYOLOEnabled(bool enabled) { yolo_enabled_ = enabled; }
    bool IsYOLOEnabled() const { return yolo_enabled_; }
    
    // Get YOLO statistics
    int GetTrackedPersonCount() const;
    float GetYOLOProcessingTime() const;
    
protected:
    // Draw YOLO detections on frame
    void DrawYOLODetections(cv::Mat& im, float imageScale);
    
    // Draw person bounding box with tracking info
    void DrawPersonBox(cv::Mat& im, const YOLODetection& det, float imageScale);
    
    // Draw movement trail for tracked person
    void DrawMovementTrail(cv::Mat& im, const YOLODetection& det, float imageScale);
    
    // Draw zone overlays
    void DrawZones(cv::Mat& im, float imageScale);
    
    // Draw YOLO statistics
    void DrawYOLOStats(cv::Mat& im);
    
    // Update frame for YOLO processing
    void UpdateYOLOFrame(const cv::Mat& frame, double timestamp);
    
private:
    // YOLO interface
    std::unique_ptr<YOLOInterface> yolo_interface_;
    bool yolo_enabled_;
    bool yolo_initialized_;
    
    // Current YOLO detections
    std::vector<YOLODetection> current_detections_;
    std::vector<Zone> current_zones_;
    
    // Tracking visualization
    std::map<int, std::deque<cv::Point2f>> person_trails_;
    std::map<int, cv::Scalar> person_colors_;
    int max_trail_length_;
    
    // Performance metrics
    float yolo_fps_;
    float yolo_processing_time_;
    std::chrono::steady_clock::time_point last_yolo_update_;
    
    // Visualization settings
    bool show_trails_;
    bool show_zones_;
    bool show_names_;
    bool show_activities_;
    float detection_alpha_;  // Transparency for overlays
    
    // Color palette for different track IDs
    std::vector<cv::Scalar> color_palette_;
    void InitializeColorPalette();
    cv::Scalar GetTrackColor(int track_id);
    
    // Thread safety
    std::mutex yolo_mutex_;
};

} // namespace ORB_SLAM3

#endif // FRAMEDRAWER_YOLO_H