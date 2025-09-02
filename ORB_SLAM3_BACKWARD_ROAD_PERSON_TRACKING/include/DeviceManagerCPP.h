/**
 * DeviceManagerCPP.h
 * Exact port of modules/device_manager.py
 * Handles YOLO detection, camera management, and device switching
 */

#ifndef DEVICE_MANAGER_CPP_H
#define DEVICE_MANAGER_CPP_H

#include "orbslam3_export.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <memory>

// ONNX Runtime includes
#include "onnxruntime_cxx_api.h"

namespace ORB_SLAM3 {

struct YOLODetectionCPP {
    cv::Rect2f bbox;
    float confidence;
    std::string label;
    int class_id;
};

class ORB_SLAM3_API DeviceManagerCPP {
public:
    DeviceManagerCPP();
    ~DeviceManagerCPP();
    
    // Exact port of Python DeviceManager methods
    bool initialize(const std::map<std::string, float>& config = {});
    std::vector<YOLODetectionCPP> detect_persons(const cv::Mat& frame);
    bool is_yolo_available() const { return yolo_available_; }
    bool is_camera_available() const { return camera_available_; }
    
    // Performance methods
    float get_detection_time() const { return detection_time_; }
    float get_fps() const;
    
    // Configuration methods
    void set_confidence_threshold(float threshold);
    void set_nms_threshold(float threshold);
    
    // GPU detection methods (exact Python GPU detection)
    bool is_gpu_available() const;
    std::string get_device_info() const;
    
private:
    // YOLO detection methods
    bool load_yolo_model(const std::string& model_path = "yolo10m.onnx");
    std::vector<YOLODetectionCPP> detect_yolo(const cv::Mat& frame);
    std::vector<YOLODetectionCPP> detect_hog_fallback(const cv::Mat& frame);
    
    // Post-processing methods
    std::vector<YOLODetectionCPP> post_process_detections(
        const std::vector<float>& output_data, 
        const cv::Mat& frame,
        float conf_threshold,
        float nms_threshold);
    
private:
    // ONNX Runtime session
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::Env> ort_env_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    bool yolo_available_;
    std::vector<std::string> class_names_;
    
    // HOG fallback
    cv::HOGDescriptor hog_;
    
    // Configuration (exact Python defaults)
    float confidence_threshold_;
    float nms_threshold_;
    cv::Size input_size_;
    
    // Performance tracking
    float detection_time_;
    std::chrono::steady_clock::time_point last_detection_;
    
    // Device status
    bool camera_available_;
    bool initialized_;
    bool gpu_available_;
    
    // Constants (exact Python values)
    static const int YOLO_INPUT_WIDTH = 640;
    static const int YOLO_INPUT_HEIGHT = 640;
    static const float DEFAULT_CONF_THRESHOLD;
    static const float DEFAULT_NMS_THRESHOLD;
};

} // namespace ORB_SLAM3

#endif // DEVICE_MANAGER_CPP_H