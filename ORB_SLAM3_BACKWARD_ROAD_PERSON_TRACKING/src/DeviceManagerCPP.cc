/**
 * DeviceManagerCPP.cc
 * Exact port of modules/device_manager.py implementation
 */

#include "DeviceManagerCPP.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace ORB_SLAM3 {

// Constants (exact Python values)
const float DeviceManagerCPP::DEFAULT_CONF_THRESHOLD = 0.4f;
const float DeviceManagerCPP::DEFAULT_NMS_THRESHOLD = 0.5f;

DeviceManagerCPP::DeviceManagerCPP()
    : yolo_available_(false)
    , confidence_threshold_(DEFAULT_CONF_THRESHOLD)
    , nms_threshold_(DEFAULT_NMS_THRESHOLD)
    , input_size_(YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT)
    , detection_time_(0.0f)
    , camera_available_(false)
    , initialized_(false)
    , gpu_available_(false)
    , ort_session_(nullptr)
    , ort_env_(nullptr)
{
    // Initialize HOG descriptor for fallback (exact Python setup)
    hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    
    // Initialize YOLO class names (COCO classes - exact Python list)
    class_names_ = {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
}

DeviceManagerCPP::~DeviceManagerCPP() = default;

bool DeviceManagerCPP::initialize(const std::map<std::string, float>& config)
{
    std::cout << "🔧 Initializing DeviceManager C++..." << std::endl;
    
    // Load configuration (exact Python config loading)
    if (config.find("confidence_threshold") != config.end()) {
        confidence_threshold_ = config.at("confidence_threshold");
    }
    if (config.find("nms_threshold") != config.end()) {
        nms_threshold_ = config.at("nms_threshold");
    }
    
    // Try to load YOLO model (same as Python version: YOLOv10m)
    std::string model_path = "D:/Learning/ORB_SLAM3_macosx/yolov10m.onnx";
    std::cout << "🔍 Loading YOLO model from: " << model_path << std::endl;
    yolo_available_ = load_yolo_model(model_path);
    
    std::cout << "🎯 YOLO Config: conf=" << confidence_threshold_ 
              << ", nms=" << nms_threshold_ 
              << ", available=" << yolo_available_ << std::endl;
    
    if (!yolo_available_) {
        std::cout << "⚠️  YOLO model not available, using HOG fallback (exact Python fallback)" << std::endl;
    } else {
        std::cout << "✅ YOLO model loaded successfully" << std::endl;
    }
    
    camera_available_ = true; // Assume camera is available
    initialized_ = true;
    
    std::cout << "✅ DeviceManager C++ initialized" << std::endl;
    return true;
}

bool DeviceManagerCPP::load_yolo_model(const std::string& model_path)
{
    try {
        // Check if YOLO model file exists first
        std::ifstream model_file(model_path);
        if (!model_file.good()) {
            std::cout << "⚠️  YOLO model file not found: " << model_path << " (using HOG fallback)" << std::endl;
            return false;
        }
        model_file.close();
        
        // Initialize ONNX Runtime environment
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLODetection");
        
        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Smart GPU detection and setup (exact Python GPU logic)
        bool gpu_available = false;
        
        // Check available providers first
        std::cout << "🔍 Available execution providers:" << std::endl;
        auto available_providers = Ort::GetAvailableProviders();
        for (const auto& provider : available_providers) {
            std::cout << "  - " << provider << std::endl;
        }
        
        // Try CUDA provider if available
        bool cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") != available_providers.end();
        
        if (cuda_available) {
            try {
                std::cout << "🔍 Attempting to enable CUDA execution provider..." << std::endl;
                
                // Try to create CUDA execution provider with minimal options
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                cuda_options.arena_extend_strategy = 0;
                cuda_options.gpu_mem_limit = SIZE_MAX;
                
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                gpu_available = true;
                gpu_available_ = true;
                std::cout << "🚀 CUDA execution provider enabled successfully" << std::endl;
                
            } catch (const std::exception& gpu_error) {
                std::cout << "⚠️  CUDA provider failed: " << gpu_error.what() << std::endl;
                std::cout << "🔄 Falling back to CPU execution..." << std::endl;
                gpu_available = false;
            }
        } else {
            std::cout << "⚠️  CUDA execution provider not available in this build" << std::endl;
            std::cout << "🔄 Using CPU execution..." << std::endl;
            gpu_available = false;
        }
        
        // Create ONNX Runtime session
        std::wstring model_path_w(model_path.begin(), model_path.end());
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, model_path_w.c_str(), session_options);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input names and shapes
        size_t num_input_nodes = ort_session_->GetInputCount();
        input_names_.reserve(num_input_nodes);
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = ort_session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(input_name.get());
            
            Ort::TypeInfo input_type_info = ort_session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = input_tensor_info.GetShape();
        }
        
        // Get output names
        size_t num_output_nodes = ort_session_->GetOutputCount();
        output_names_.reserve(num_output_nodes);
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = ort_session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(output_name.get());
        }
        
        if (gpu_available) {
            std::cout << "🚀 YOLO model loaded on GPU (ONNX Runtime CUDA): " << model_path << std::endl;
        } else {
            std::cout << "✅ YOLO model loaded on CPU (ONNX Runtime): " << model_path << std::endl;
        }
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cout << "❌ ONNX Runtime error loading YOLO: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cout << "❌ Error loading YOLO: " << e.what() << std::endl;
        return false;
    }
}

std::vector<YOLODetectionCPP> DeviceManagerCPP::detect_persons(const cv::Mat& frame)
{
    if (!initialized_ || frame.empty()) {
        return {};
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    std::vector<YOLODetectionCPP> detections;
    
    // Use YOLO if available, otherwise HOG fallback (exact Python logic)
    if (yolo_available_) {
        detections = detect_yolo(frame);
    } else {
        detections = detect_hog_fallback(frame);
    }
    
    auto end_time = std::chrono::steady_clock::now();
    detection_time_ = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    last_detection_ = end_time;
    
    // Filter for person class only (exact Python filtering)
    std::vector<YOLODetectionCPP> person_detections;
    for (const auto& det : detections) {
        if (det.label == "person") {
            person_detections.push_back(det);
        }
    }
    
    return person_detections;
}

std::vector<YOLODetectionCPP> DeviceManagerCPP::detect_yolo(const cv::Mat& frame)
{
    try {
        // Prepare input data (exact Python preprocessing)
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, input_size_);
        
        // Convert BGR to RGB and normalize to [0, 1]
        cv::Mat rgb_frame;
        cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);
        rgb_frame.convertTo(rgb_frame, CV_32F, 1.0/255.0);
        
        // Create input tensor
        std::vector<int64_t> input_shape = {1, 3, input_size_.height, input_size_.width};
        size_t input_tensor_size = 1 * 3 * input_size_.height * input_size_.width;
        std::vector<float> input_tensor_values(input_tensor_size);
        
        // Fill input tensor (CHW format)
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < input_size_.height; ++h) {
                for (int w = 0; w < input_size_.width; ++w) {
                    input_tensor_values[c * input_size_.height * input_size_.width + h * input_size_.width + w] = 
                        rgb_frame.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        
        // Create memory info for CPU or GPU
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Create input tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_size,
            input_shape.data(), input_shape.size()
        );
        
        // Run inference
        auto output_tensors = ort_session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names_.data(), 
            &input_tensor, 
            1, 
            output_names_.data(), 
            output_names_.size()
        );
        
        // Get output data
        float* float_array = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        std::cout << "🎯 YOLO Output Shape: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Convert to vector for post-processing
        size_t output_size = 1;
        for (auto& dim : output_shape) {
            output_size *= dim;
        }
        std::cout << "🎯 Total output size: " << output_size << std::endl;
        
        // Safely allocate memory
        std::vector<float> output_data;
        try {
            output_data.assign(float_array, float_array + output_size);
        } catch (const std::bad_alloc& e) {
            std::cout << "❌ Memory allocation failed for size: " << output_size << std::endl;
            return {};
        }
        
        // Post-process detections (exact Python post-processing)
        auto detections = post_process_detections(output_data, frame, confidence_threshold_, nms_threshold_);
        std::cout << "🎯 YOLO Output: " << detections.size() << " detections" << std::endl;
        return detections;
        
    } catch (const Ort::Exception& e) {
        std::cout << "❌ ONNX Runtime detection error: " << e.what() << std::endl;
        return {};
    } catch (const std::exception& e) {
        std::cout << "❌ YOLO detection error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<YOLODetectionCPP> DeviceManagerCPP::detect_hog_fallback(const cv::Mat& frame)
{
    std::vector<YOLODetectionCPP> detections;
    
    try {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        std::vector<cv::Rect> boxes;
        std::vector<double> weights;
        
        // HOG detection with exact Python parameters
        hog_.detectMultiScale(gray, boxes, weights, 0.5, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
        
        // Convert to YOLODetectionCPP format (exact Python conversion)
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (weights[i] > 0.5) { // Confidence threshold
                YOLODetectionCPP detection;
                detection.bbox = cv::Rect2f(boxes[i]);
                detection.confidence = static_cast<float>(weights[i]);
                detection.label = "person";
                detection.class_id = 0; // Person class ID
                detections.push_back(detection);
            }
        }
        
    } catch (const cv::Exception& e) {
        std::cout << "❌ HOG detection error: " << e.what() << std::endl;
    }
    
    return detections;
}

std::vector<YOLODetectionCPP> DeviceManagerCPP::post_process_detections(
    const std::vector<float>& output_data, 
    const cv::Mat& frame,
    float conf_threshold,
    float nms_threshold)
{
    std::vector<YOLODetectionCPP> detections;
    
    try {
        if (output_data.empty()) {
            std::cout << "⚠️  Output data is empty" << std::endl;
            return detections;
        }
        
        size_t total_size = output_data.size();
        std::cout << "🎯 Post-processing " << total_size << " output values" << std::endl;
        
        // YOLOv10n typically outputs [batch, 300, 6] or similar
        // Let's detect the format automatically
        if (total_size < 6) {
            std::cout << "⚠️  Output data too small: " << total_size << std::endl;
            return detections;
        }
        
        // Try different possible formats
        std::vector<std::pair<int, int>> possible_formats = {
            {6, static_cast<int>(total_size / 6)},     // [N, 6] format
            {7, static_cast<int>(total_size / 7)},     // [N, 7] format  
            {85, static_cast<int>(total_size / 85)},   // [N, 85] format (traditional YOLO)
        };
        
        int num_values = 6;
        int num_detections = 0;
        
        for (const auto& format : possible_formats) {
            if (total_size % format.first == 0 && format.second > 0) {
                num_values = format.first;
                num_detections = format.second;
                std::cout << "🎯 Detected format: " << num_detections << "x" << num_values << std::endl;
                break;
            }
        }
        
        if (num_detections == 0) {
            std::cout << "⚠️  Cannot determine output format for size: " << total_size << std::endl;
            return detections;
        }
        
        // Validate dimensions
        if (num_detections <= 0 || num_values != 6) {
            std::cout << "⚠️  Invalid calculated dimensions: " << num_detections << "x" << num_values << std::endl;
            return detections;
        }
        
        // Limit processing to prevent memory issues
        if (num_detections > 8400) { // YOLOv10 typical max detections
            std::cout << "⚠️  Too many detections (" << num_detections << "), limiting to 8400" << std::endl;
            num_detections = 8400;
        }
        
        const float* data = output_data.data();
        if (data == nullptr) {
            std::cout << "⚠️  Output data pointer is null" << std::endl;
            return detections;
        }
        
        // Process detections (assuming batch_size = 1)
        for (int i = 0; i < num_detections; ++i) {
            const float* detection = data + i * num_values;
            
            // YOLOv10 format: [x1, y1, x2, y2, confidence, class_id]
            if (num_values >= 6) {
                float x1 = detection[0];
                float y1 = detection[1];
                float x2 = detection[2];
                float y2 = detection[3];
                float confidence = detection[4];
                float class_id_f = detection[5];
                
                int class_id = static_cast<int>(class_id_f);
                
                // Filter for person class (0) and confidence threshold
                if (class_id == 0 && confidence >= conf_threshold) {
                    // Convert to frame coordinates
                    int x = std::max(0, static_cast<int>(x1));
                    int y = std::max(0, static_cast<int>(y1));
                    int width = std::max(1, static_cast<int>(x2 - x1));
                    int height = std::max(1, static_cast<int>(y2 - y1));
                    
                    // Clamp to frame bounds
                    x = std::min(x, frame.cols - 1);
                    y = std::min(y, frame.rows - 1);
                    width = std::min(width, frame.cols - x);
                    height = std::min(height, frame.rows - y);
                    
                    if (width > 0 && height > 0) {
                        YOLODetectionCPP detection;
                        detection.bbox = cv::Rect2f(static_cast<float>(x), static_cast<float>(y), 
                                                   static_cast<float>(width), static_cast<float>(height));
                        detection.confidence = confidence;
                        detection.class_id = class_id;
                        detection.label = "person";
                        detections.push_back(detection);
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Post-processing error: " << e.what() << std::endl;
    }
    
    return detections;
}

float DeviceManagerCPP::get_fps() const
{
    if (detection_time_ > 0) {
        return 1000.0f / detection_time_;
    }
    return 0.0f;
}

void DeviceManagerCPP::set_confidence_threshold(float threshold)
{
    confidence_threshold_ = threshold;
}

void DeviceManagerCPP::set_nms_threshold(float threshold)
{
    nms_threshold_ = threshold;
}

bool DeviceManagerCPP::is_gpu_available() const
{
    return gpu_available_;
}

std::string DeviceManagerCPP::get_device_info() const
{
    std::stringstream info;
    
    try {
        int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
        info << "CUDA devices available: " << cuda_devices;
        
        if (cuda_devices > 0) {
            info << " | YOLO running on: " << (gpu_available_ ? "GPU" : "CPU");
            
            // Get device properties
            cv::cuda::DeviceInfo device_info(0);  // First device
            info << " | GPU: " << device_info.name();
        } else {
            info << " | YOLO running on: CPU (no CUDA)";
        }
    } catch (const cv::Exception& e) {
        info << "Device info unavailable: " << e.what();
    }
    
    return info.str();
}

} // namespace ORB_SLAM3