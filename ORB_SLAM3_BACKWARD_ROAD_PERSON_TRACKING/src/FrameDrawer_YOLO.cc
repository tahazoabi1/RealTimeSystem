/**
* FrameDrawer with YOLO Integration
* Implementation of enhanced FrameDrawer with YOLO person detection overlay
*/

#include "FrameDrawer_YOLO.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
#include <sstream>
#include <chrono>

namespace ORB_SLAM3
{

FrameDrawerYOLO::FrameDrawerYOLO(Atlas* pAtlas) 
    : FrameDrawer(pAtlas)
    , yolo_enabled_(true)
    , yolo_initialized_(false)
    , max_trail_length_(30)
    , yolo_fps_(0.0f)
    , yolo_processing_time_(0.0f)
    , show_trails_(true)
    , show_zones_(true)
    , show_names_(true)
    , show_activities_(true)
    , detection_alpha_(0.3f)
{
    yolo_interface_ = std::make_unique<YOLOInterface>();
    InitializeColorPalette();
    last_yolo_update_ = std::chrono::steady_clock::now();
}

FrameDrawerYOLO::~FrameDrawerYOLO()
{
    if (yolo_interface_) {
        yolo_interface_->Shutdown();
    }
}

bool FrameDrawerYOLO::InitializeYOLO(const std::string& config_path)
{
    if (!yolo_interface_) {
        return false;
    }
    
    yolo_initialized_ = yolo_interface_->Initialize(config_path);
    
    if (yolo_initialized_) {
        std::cout << "✅ YOLO integration initialized in FrameDrawer" << std::endl;
    } else {
        std::cout << "❌ Failed to initialize YOLO integration" << std::endl;
    }
    
    return yolo_initialized_;
}

cv::Mat FrameDrawerYOLO::DrawFrame(float imageScale)
{
    // First draw the standard ORB-SLAM3 features
    cv::Mat imWithFeatures = FrameDrawer::DrawFrame(imageScale);
    
    // If YOLO is enabled and initialized, add YOLO overlays
    if (yolo_enabled_ && yolo_initialized_) {
        // Process current frame through YOLO
        cv::Mat grayFrame;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mIm.copyTo(grayFrame);
        }
        
        if (!grayFrame.empty()) {
            // Get current timestamp
            auto now = std::chrono::steady_clock::now();
            double timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
            
            // Update YOLO with current frame
            UpdateYOLOFrame(grayFrame, timestamp);
            
            // Draw YOLO detections on top of features
            DrawYOLODetections(imWithFeatures, imageScale);
            
            // Draw zones if enabled
            if (show_zones_) {
                DrawZones(imWithFeatures, imageScale);
            }
            
            // Draw YOLO statistics
            DrawYOLOStats(imWithFeatures);
        }
    }
    
    return imWithFeatures;
}

cv::Mat FrameDrawerYOLO::DrawRightFrame(float imageScale)
{
    // For stereo, just use base implementation for now
    // YOLO processing only on left frame
    return FrameDrawer::DrawRightFrame(imageScale);
}

void FrameDrawerYOLO::UpdateYOLOFrame(const cv::Mat& frame, double timestamp)
{
    if (!yolo_interface_ || !yolo_interface_->IsReady()) {
        return;
    }
    
    // Send frame to YOLO for processing
    if (yolo_interface_->ProcessFrame(frame, timestamp)) {
        // Get detection results
        std::lock_guard<std::mutex> lock(yolo_mutex_);
        current_detections_ = yolo_interface_->GetDetections();
        current_zones_ = yolo_interface_->GetZones();
        
        // Update performance metrics
        yolo_processing_time_ = yolo_interface_->GetProcessingTime();
        
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_yolo_update_).count();
        if (dt > 0) {
            yolo_fps_ = 0.9f * yolo_fps_ + 0.1f * (1.0f / dt);
        }
        last_yolo_update_ = now;
        
        // Update trails for each tracked person
        for (const auto& det : current_detections_) {
            cv::Point2f center(det.bbox.x + det.bbox.width/2, 
                             det.bbox.y + det.bbox.height/2);
            
            if (person_trails_[det.track_id].size() >= max_trail_length_) {
                person_trails_[det.track_id].pop_front();
            }
            person_trails_[det.track_id].push_back(center);
        }
    }
}

void FrameDrawerYOLO::DrawYOLODetections(cv::Mat& im, float imageScale)
{
    std::lock_guard<std::mutex> lock(yolo_mutex_);
    
    for (const auto& det : current_detections_) {
        // Draw person bounding box
        DrawPersonBox(im, det, imageScale);
        
        // Draw movement trail
        if (show_trails_ && person_trails_.count(det.track_id) > 0) {
            DrawMovementTrail(im, det, imageScale);
        }
    }
}

void FrameDrawerYOLO::DrawPersonBox(cv::Mat& im, const YOLODetection& det, float imageScale)
{
    // Scale bbox coordinates
    cv::Rect2f bbox = det.bbox;
    if (imageScale != 1.0f) {
        bbox.x /= imageScale;
        bbox.y /= imageScale;
        bbox.width /= imageScale;
        bbox.height /= imageScale;
    }
    
    // Get color for this track ID
    cv::Scalar color = GetTrackColor(det.track_id);
    
    // Draw semi-transparent filled rectangle
    cv::Mat overlay;
    im.copyTo(overlay);
    cv::rectangle(overlay, bbox, color, -1);
    cv::addWeighted(overlay, detection_alpha_, im, 1 - detection_alpha_, 0, im);
    
    // Draw solid border
    cv::rectangle(im, bbox, color, 2);
    
    // Prepare label text
    std::stringstream label;
    label << "ID:" << det.track_id;
    
    if (show_names_ && !det.person_name.empty() && det.person_name != "Unknown") {
        label << " | " << det.person_name;
    }
    
    if (show_activities_ && !det.activity.empty()) {
        label << " | " << det.activity;
    }
    
    label << " (" << std::fixed << std::setprecision(0) << (det.confidence * 100) << "%)";
    
    // Draw label background
    int baseline;
    cv::Size textSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 
                                       0.5, 1, &baseline);
    
    cv::Point2f label_pos(bbox.x, bbox.y - 5);
    if (label_pos.y - textSize.height < 0) {
        label_pos.y = bbox.y + bbox.height + textSize.height + 5;
    }
    
    cv::rectangle(im, 
                 cv::Point2f(label_pos.x, label_pos.y - textSize.height - 3),
                 cv::Point2f(label_pos.x + textSize.width, label_pos.y + 3),
                 color, -1);
    
    // Draw label text
    cv::putText(im, label.str(), label_pos,
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    // Draw velocity indicator if moving
    if (det.velocity.x != 0 || det.velocity.y != 0) {
        cv::Point2f center(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
        cv::Point2f vel_end = center + det.velocity * 20;  // Scale velocity for visibility
        cv::arrowedLine(im, center, vel_end, color, 2);
    }
}

void FrameDrawerYOLO::DrawMovementTrail(cv::Mat& im, const YOLODetection& det, float imageScale)
{
    const auto& trail = person_trails_[det.track_id];
    if (trail.size() < 2) return;
    
    cv::Scalar color = GetTrackColor(det.track_id);
    
    for (size_t i = 1; i < trail.size(); ++i) {
        cv::Point2f pt1 = trail[i-1];
        cv::Point2f pt2 = trail[i];
        
        if (imageScale != 1.0f) {
            pt1 /= imageScale;
            pt2 /= imageScale;
        }
        
        // Fade trail based on age
        float alpha = static_cast<float>(i) / trail.size();
        int thickness = std::max(1, static_cast<int>(3 * alpha));
        cv::Scalar trail_color = color * alpha;
        
        cv::line(im, pt1, pt2, trail_color, thickness);
    }
}

void FrameDrawerYOLO::DrawZones(cv::Mat& im, float imageScale)
{
    std::lock_guard<std::mutex> lock(yolo_mutex_);
    
    for (const auto& zone : current_zones_) {
        cv::Rect2f area = zone.area;
        if (imageScale != 1.0f) {
            area.x /= imageScale;
            area.y /= imageScale;
            area.width /= imageScale;
            area.height /= imageScale;
        }
        
        // Draw semi-transparent zone
        cv::Mat overlay;
        im.copyTo(overlay);
        cv::rectangle(overlay, area, zone.color, -1);
        cv::addWeighted(overlay, 0.2f, im, 0.8f, 0, im);
        
        // Draw zone border
        cv::rectangle(im, area, zone.color, 2);
        
        // Draw zone label
        std::stringstream label;
        label << zone.name << " (" << zone.person_count << ")";
        
        cv::putText(im, label.str(), 
                   cv::Point2f(area.x + 5, area.y + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, zone.color, 2);
    }
}

void FrameDrawerYOLO::DrawYOLOStats(cv::Mat& im)
{
    // Draw YOLO performance stats in top-right corner
    std::stringstream stats;
    stats << "YOLO: " << std::fixed << std::setprecision(1) << yolo_fps_ << " FPS";
    stats << " | " << std::setprecision(1) << (yolo_processing_time_ * 1000) << "ms";
    stats << " | Tracking: " << current_detections_.size();
    
    int baseline;
    cv::Size textSize = cv::getTextSize(stats.str(), cv::FONT_HERSHEY_SIMPLEX,
                                       0.5, 1, &baseline);
    
    cv::Point2f stats_pos(im.cols - textSize.width - 10, 30);
    
    // Draw background
    cv::rectangle(im,
                 cv::Point2f(stats_pos.x - 5, stats_pos.y - textSize.height - 5),
                 cv::Point2f(stats_pos.x + textSize.width + 5, stats_pos.y + 5),
                 cv::Scalar(0, 0, 0), -1);
    
    // Draw text
    cv::putText(im, stats.str(), stats_pos,
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
}

void FrameDrawerYOLO::InitializeColorPalette()
{
    // Create a diverse color palette for different track IDs
    color_palette_ = {
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 255, 0),    // Green-Yellow
        cv::Scalar(255, 128, 0),    // Orange
        cv::Scalar(255, 0, 128),    // Pink
        cv::Scalar(128, 0, 255),    // Purple
        cv::Scalar(0, 128, 255),    // Sky Blue
        cv::Scalar(255, 128, 128),  // Light Red
        cv::Scalar(128, 255, 128),  // Light Green
        cv::Scalar(128, 128, 255),  // Light Blue
        cv::Scalar(255, 255, 128),  // Light Cyan
    };
}

cv::Scalar FrameDrawerYOLO::GetTrackColor(int track_id)
{
    // Use cached color if available
    if (person_colors_.count(track_id) > 0) {
        return person_colors_[track_id];
    }
    
    // Assign new color from palette
    cv::Scalar color = color_palette_[track_id % color_palette_.size()];
    person_colors_[track_id] = color;
    return color;
}

int FrameDrawerYOLO::GetTrackedPersonCount() const
{
    if (yolo_interface_) {
        return yolo_interface_->GetTrackedCount();
    }
    return 0;
}

float FrameDrawerYOLO::GetYOLOProcessingTime() const
{
    return yolo_processing_time_;
}

} // namespace ORB_SLAM3