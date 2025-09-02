/**
 * FaceRecognizerCPP.cc
 * Exact port of face_recognizer.py implementation
 */

#include "FaceRecognizerCPP.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace ORB_SLAM3 {

// Constants (exact Python values)
const float FaceRecognizerCPP::DEFAULT_DISTANCE_THRESHOLD = 0.6f;
const float FaceRecognizerCPP::DEFAULT_CONFIDENCE_THRESHOLD = 0.5f;
const int FaceRecognizerCPP::DEFAULT_MAX_DATABASE_SIZE = 1000;
const int FaceRecognizerCPP::DEFAULT_ENCODING_SIZE = 128;
const int FaceRecognizerCPP::CLEANUP_INTERVAL_MINUTES = 30;

FaceRecognizerCPP::FaceRecognizerCPP()
    : distance_threshold_(DEFAULT_DISTANCE_THRESHOLD)
    , confidence_threshold_(DEFAULT_CONFIDENCE_THRESHOLD)
    , max_database_size_(DEFAULT_MAX_DATABASE_SIZE)
    , encoding_size_(DEFAULT_ENCODING_SIZE)
    , initialized_(false)
    , use_dnn_detector_(false)
    , last_cleanup_(std::chrono::steady_clock::now())
{
    std::cout << "🔧 Initializing Face Recognizer C++..." << std::endl;
}

FaceRecognizerCPP::~FaceRecognizerCPP() = default;

bool FaceRecognizerCPP::initialize(const std::string& model_path)
{
    std::cout << "🔧 Loading face recognition models..." << std::endl;
    
    try {
        // Check if ONNX model file exists first
        std::ifstream model_file(model_path);
        if (model_file.good()) {
            model_file.close();
            
            try {
                // Try to load DNN face recognition model (exact Python model loading)
                face_encoder_ = cv::dnn::readNetFromONNX(model_path);
                if (!face_encoder_.empty()) {
                    // Smart GPU detection for face recognition
                    bool gpu_available = false;
                    try {
                        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                            face_encoder_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                            face_encoder_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                            
                            // Test GPU inference
                            cv::Mat dummy_face(128, 128, CV_8UC3, cv::Scalar(0));
                            cv::Mat blob;
                            cv::dnn::blobFromImage(dummy_face, blob, 1.0/255.0, cv::Size(128, 128));
                            face_encoder_.setInput(blob);
                            cv::Mat test_output = face_encoder_.forward();
                            
                            gpu_available = true;
                            std::cout << "🚀 Face recognition model loaded on GPU (CUDA)" << std::endl;
                        }
                    } catch (const cv::Exception& gpu_error) {
                        std::cout << "⚠️  Face recognition GPU failed, using CPU: " << gpu_error.what() << std::endl;
                        gpu_available = false;
                    }
                    
                    // Fallback to CPU
                    if (!gpu_available) {
                        face_encoder_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                        face_encoder_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                        std::cout << "✅ Face recognition model loaded on CPU" << std::endl;
                    }
                    
                    use_dnn_detector_ = true;
                } else {
                    std::cout << "⚠️  Failed to load DNN model: " << model_path << std::endl;
                }
            } catch (const cv::Exception& onnx_error) {
                std::cout << "⚠️  ONNX loading failed: " << onnx_error.what() << " (face recognition disabled)" << std::endl;
            }
        } else {
            std::cout << "⚠️  ONNX model file not found: " << model_path << " (face recognition disabled)" << std::endl;
        }
        
        // Load Haar cascade as fallback (exact Python fallback)
        if (!haar_detector_.load("haarcascade_frontalface_alt.xml")) {
            std::cout << "⚠️  Haar cascade not available, face detection disabled" << std::endl;
        } else {
            std::cout << "✅ Haar cascade face detector loaded" << std::endl;
        }
        
        // Load existing database (exact Python database loading)
        load_database();
        
        // Initialize even without models (graceful degradation)
        initialized_ = true;
        std::cout << "✅ Face Recognizer C++ initialized with " << face_database_.size() << " known faces" << std::endl;
        std::cout << "   Face recognition " << (use_dnn_detector_ ? "enabled" : "disabled") << " (models available: " << (use_dnn_detector_ ? "DNN" : "none") << ")" << std::endl;
        return true;
        
    } catch (const cv::Exception& e) {
        std::cout << "❌ OpenCV error initializing face recognition: " << e.what() << std::endl;
        // Still initialize with limited functionality
        initialized_ = true;
        load_database();
        return true;
    } catch (const std::exception& e) {
        std::cout << "❌ Error initializing face recognition: " << e.what() << std::endl;
        // Still initialize with limited functionality  
        initialized_ = true;
        load_database();
        return true;
    }
}

std::vector<cv::Rect> FaceRecognizerCPP::detect_faces(const cv::Mat& frame)
{
    if (!initialized_ || frame.empty()) {
        return {};
    }
    
    // Use DNN detector if available, otherwise Haar (exact Python detection logic)
    if (use_dnn_detector_ && !face_detector_.empty()) {
        return detect_faces_dnn(frame);
    } else if (!haar_detector_.empty()) {
        return detect_faces_haar(frame);
    }
    
    return {};
}

std::vector<cv::Rect> FaceRecognizerCPP::detect_faces_dnn(const cv::Mat& frame)
{
    std::vector<cv::Rect> faces;
    
    try {
        // DNN face detection (exact Python DNN implementation)
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0, cv::Size(300, 300), cv::Scalar(104, 117, 123));
        face_detector_.setInput(blob);
        
        cv::Mat detection = face_detector_.forward();
        cv::Mat detection_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        
        for (int i = 0; i < detection_mat.rows; i++) {
            float confidence = detection_mat.at<float>(i, 2);
            if (confidence > confidence_threshold_) {
                int x1 = static_cast<int>(detection_mat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detection_mat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detection_mat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detection_mat.at<float>(i, 6) * frame.rows);
                
                faces.emplace_back(x1, y1, x2 - x1, y2 - y1);
            }
        }
    } catch (const cv::Exception& e) {
        std::cout << "❌ DNN face detection error: " << e.what() << std::endl;
    }
    
    return faces;
}

std::vector<cv::Rect> FaceRecognizerCPP::detect_faces_haar(const cv::Mat& frame)
{
    std::vector<cv::Rect> faces;
    
    try {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        
        // Haar cascade detection (exact Python Haar implementation)
        haar_detector_.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
        
    } catch (const cv::Exception& e) {
        std::cout << "❌ Haar face detection error: " << e.what() << std::endl;
    }
    
    return faces;
}

std::vector<std::vector<float>> FaceRecognizerCPP::get_face_encodings(const cv::Mat& frame, const std::vector<cv::Rect>& face_locations)
{
    std::vector<std::vector<float>> encodings;
    
    if (!initialized_ || frame.empty() || face_locations.empty()) {
        return encodings;
    }
    
    // Extract face encodings for each detected face (exact Python encoding extraction)
    for (const auto& face_rect : face_locations) {
        // Ensure face rect is within image bounds
        cv::Rect safe_rect = face_rect & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safe_rect.width > 0 && safe_rect.height > 0) {
            cv::Mat face_roi = frame(safe_rect);
            std::vector<float> encoding = compute_face_encoding(face_roi);
            if (!encoding.empty()) {
                encodings.push_back(encoding);
            }
        }
    }
    
    return encodings;
}

std::vector<float> FaceRecognizerCPP::compute_face_encoding(const cv::Mat& face_image)
{
    std::vector<float> encoding;
    
    try {
        if (use_dnn_detector_ && !face_encoder_.empty()) {
            // DNN-based face encoding (exact Python encoding computation)
            cv::Mat face_resized;
            cv::resize(face_image, face_resized, cv::Size(160, 160));
            
            cv::Mat blob;
            cv::dnn::blobFromImage(face_resized, blob, 1.0/255.0, cv::Size(160, 160), cv::Scalar(0,0,0), true, false);
            face_encoder_.setInput(blob);
            
            cv::Mat encoding_mat = face_encoder_.forward();
            encoding.assign((float*)encoding_mat.data, (float*)encoding_mat.data + encoding_mat.total());
            
        } else {
            // Fallback: Simple histogram-based encoding (simplified version)
            cv::Mat gray, hist;
            cv::cvtColor(face_image, gray, cv::COLOR_BGR2GRAY);
            cv::resize(gray, gray, cv::Size(64, 64));
            
            // Compute histogram as simple encoding
            int histSize = 256;
            float range[] = {0, 256};
            const float* histRange = {range};
            cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
            
            // Normalize and convert to encoding
            cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
            encoding.assign((float*)hist.data, (float*)hist.data + hist.total());
        }
        
    } catch (const cv::Exception& e) {
        std::cout << "❌ Face encoding error: " << e.what() << std::endl;
    }
    
    return encoding;
}

std::string FaceRecognizerCPP::recognize_face(const cv::Mat& face_image, float distance_threshold)
{
    if (!initialized_ || face_image.empty()) {
        return "Unknown";
    }
    
    // Compute encoding for the face
    std::vector<float> encoding = compute_face_encoding(face_image);
    if (encoding.empty()) {
        return "Unknown";
    }
    
    // Match against database (exact Python matching logic)
    FaceMatchCPP match = match_face_encoding(encoding);
    
    if (match.is_match && match.distance < distance_threshold) {
        update_face_encounter(match.name);
        return match.name;
    }
    
    return "Unknown";
}

FaceMatchCPP FaceRecognizerCPP::match_face_encoding(const std::vector<float>& unknown_encoding)
{
    FaceMatchCPP best_match;
    best_match.name = "Unknown";
    best_match.distance = std::numeric_limits<float>::max();
    best_match.confidence = 0.0f;
    best_match.is_match = false;
    
    if (face_database_.empty() || !validate_encoding(unknown_encoding)) {
        return best_match;
    }
    
    // Find best match in database (exact Python matching algorithm)
    for (const auto& entry : face_database_) {
        const std::string& name = entry.first;
        const FaceDataCPP& face_data = entry.second;
        
        float distance = compute_face_distance(unknown_encoding, face_data.encoding);
        
        if (distance < best_match.distance) {
            best_match.name = name;
            best_match.distance = distance;
            best_match.confidence = 1.0f - (distance / distance_threshold_);
            best_match.is_match = (distance < distance_threshold_);
        }
    }
    
    return best_match;
}

float FaceRecognizerCPP::compute_face_distance(const std::vector<float>& encoding1, const std::vector<float>& encoding2)
{
    if (encoding1.size() != encoding2.size() || encoding1.empty()) {
        return std::numeric_limits<float>::max();
    }
    
    // Euclidean distance (exact Python distance computation)
    float sum = 0.0f;
    for (size_t i = 0; i < encoding1.size(); ++i) {
        float diff = encoding1[i] - encoding2[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}

bool FaceRecognizerCPP::add_face_to_database(const std::vector<float>& encoding, const std::string& name)
{
    if (!validate_encoding(encoding) || name.empty() || name == "Unknown") {
        return false;
    }
    
    // Add or update face in database (exact Python database management)
    FaceDataCPP face_data;
    face_data.encoding = encoding;
    face_data.name = name;
    face_data.created_at = std::chrono::system_clock::now();
    face_data.last_seen = face_data.created_at;
    face_data.encounter_count = 1;
    face_data.confidence_score = 1.0f;
    
    face_database_[name] = face_data;
    
    std::cout << "✅ Added face to database: " << name << std::endl;
    
    // Auto-save database (exact Python auto-save)
    save_database();
    
    return true;
}

std::string FaceRecognizerCPP::get_name_for_unknown_face(int track_id)
{
    // Interactive naming (simplified for console - exact Python concept)
    std::string name;
    
    std::cout << "\n🔍 Unknown face detected (Track ID: " << track_id << ")" << std::endl;
    std::cout << "Enter name for this person (or 'skip' to ignore): ";
    std::getline(std::cin, name);
    
    if (name != "skip" && !name.empty()) {
        pending_names_[track_id] = name;
        session_names_.insert(name);
        std::cout << "✅ Name recorded: " << name << std::endl;
        return name;
    }
    
    return "Unknown";
}

void FaceRecognizerCPP::update_face_encounter(const std::string& name)
{
    auto it = face_database_.find(name);
    if (it != face_database_.end()) {
        it->second.last_seen = std::chrono::system_clock::now();
        it->second.encounter_count++;
    }
}

bool FaceRecognizerCPP::save_database(const std::string& filename)
{
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        // Save database size
        size_t db_size = face_database_.size();
        file.write(reinterpret_cast<const char*>(&db_size), sizeof(db_size));
        
        // Save each face entry (exact Python serialization format)
        for (const auto& entry : face_database_) {
            const std::string& name = entry.first;
            const FaceDataCPP& face_data = entry.second;
            
            // Save name length and name
            size_t name_len = name.length();
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            file.write(name.c_str(), name_len);
            
            // Save encoding size and encoding
            size_t encoding_size = face_data.encoding.size();
            file.write(reinterpret_cast<const char*>(&encoding_size), sizeof(encoding_size));
            file.write(reinterpret_cast<const char*>(face_data.encoding.data()), 
                      encoding_size * sizeof(float));
            
            // Save metadata
            auto created_time = face_data.created_at.time_since_epoch().count();
            auto last_seen_time = face_data.last_seen.time_since_epoch().count();
            file.write(reinterpret_cast<const char*>(&created_time), sizeof(created_time));
            file.write(reinterpret_cast<const char*>(&last_seen_time), sizeof(last_seen_time));
            file.write(reinterpret_cast<const char*>(&face_data.encounter_count), sizeof(face_data.encounter_count));
            file.write(reinterpret_cast<const char*>(&face_data.confidence_score), sizeof(face_data.confidence_score));
        }
        
        file.close();
        std::cout << "✅ Face database saved: " << filename << " (" << db_size << " faces)" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error saving face database: " << e.what() << std::endl;
        return false;
    }
}

bool FaceRecognizerCPP::load_database(const std::string& filename)
{
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "ℹ️  No existing face database found, starting fresh" << std::endl;
            return true; // Not an error, just no existing database
        }
        
        // Load database size
        size_t db_size;
        file.read(reinterpret_cast<char*>(&db_size), sizeof(db_size));
        
        // Load each face entry (exact Python deserialization)
        for (size_t i = 0; i < db_size; ++i) {
            // Load name
            size_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            file.read(&name[0], name_len);
            
            // Load encoding
            size_t encoding_size;
            file.read(reinterpret_cast<char*>(&encoding_size), sizeof(encoding_size));
            std::vector<float> encoding(encoding_size);
            file.read(reinterpret_cast<char*>(encoding.data()), encoding_size * sizeof(float));
            
            // Load metadata
            FaceDataCPP face_data;
            face_data.encoding = encoding;
            face_data.name = name;
            
            decltype(face_data.created_at.time_since_epoch().count()) created_time, last_seen_time;
            file.read(reinterpret_cast<char*>(&created_time), sizeof(created_time));
            file.read(reinterpret_cast<char*>(&last_seen_time), sizeof(last_seen_time));
            file.read(reinterpret_cast<char*>(&face_data.encounter_count), sizeof(face_data.encounter_count));
            file.read(reinterpret_cast<char*>(&face_data.confidence_score), sizeof(face_data.confidence_score));
            
            face_data.created_at = std::chrono::system_clock::time_point(std::chrono::system_clock::duration(created_time));
            face_data.last_seen = std::chrono::system_clock::time_point(std::chrono::system_clock::duration(last_seen_time));
            
            face_database_[name] = face_data;
        }
        
        file.close();
        std::cout << "✅ Face database loaded: " << filename << " (" << face_database_.size() << " faces)" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error loading face database: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, int> FaceRecognizerCPP::get_face_statistics() const
{
    std::map<std::string, int> stats;
    for (const auto& entry : face_database_) {
        stats[entry.first] = entry.second.encounter_count;
    }
    return stats;
}

std::vector<std::string> FaceRecognizerCPP::get_known_names() const
{
    std::vector<std::string> names;
    for (const auto& entry : face_database_) {
        names.push_back(entry.first);
    }
    return names;
}

bool FaceRecognizerCPP::validate_encoding(const std::vector<float>& encoding)
{
    return !encoding.empty() && encoding.size() >= 64; // Minimum reasonable encoding size
}

void FaceRecognizerCPP::cleanup_old_entries()
{
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::minutes>(now - last_cleanup_).count() < CLEANUP_INTERVAL_MINUTES) {
        return;
    }
    
    // Cleanup logic (exact Python cleanup)
    auto cutoff_time = std::chrono::system_clock::now() - std::chrono::hours(24 * 30); // 30 days
    
    for (auto it = face_database_.begin(); it != face_database_.end();) {
        if (it->second.encounter_count < 3 && it->second.last_seen < cutoff_time) {
            it = face_database_.erase(it);
        } else {
            ++it;
        }
    }
    
    last_cleanup_ = now;
}

} // namespace ORB_SLAM3