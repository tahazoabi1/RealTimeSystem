/**
 * FaceRecognizerCPP.h
 * Exact port of face_recognizer.py
 * Face recognition with database storage and interactive naming
 */

#ifndef FACE_RECOGNIZER_CPP_H
#define FACE_RECOGNIZER_CPP_H

#include "orbslam3_export.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <chrono>

namespace ORB_SLAM3 {

struct FaceDataCPP {
    std::vector<float> encoding;
    std::string name;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_seen;
    int encounter_count;
    float confidence_score;
};

struct FaceMatchCPP {
    std::string name;
    float distance;
    float confidence;
    bool is_match;
};

class ORB_SLAM3_API FaceRecognizerCPP {
public:
    FaceRecognizerCPP();
    ~FaceRecognizerCPP();
    
    // Exact port of Python FaceRecognizer methods
    bool initialize(const std::string& model_path = "face_recognition.onnx");
    std::vector<cv::Rect> detect_faces(const cv::Mat& frame);
    std::vector<std::vector<float>> get_face_encodings(const cv::Mat& frame, const std::vector<cv::Rect>& face_locations);
    
    // Recognition methods (exact Python interface)
    std::string recognize_face(const cv::Mat& face_image, float distance_threshold = 0.6f);
    FaceMatchCPP match_face_encoding(const std::vector<float>& unknown_encoding);
    
    // Database methods (exact Python database interface)
    bool add_face_to_database(const std::vector<float>& encoding, const std::string& name);
    bool save_database(const std::string& filename = "face_database.dat");
    bool load_database(const std::string& filename = "face_database.dat");
    
    // Interactive methods (exact Python interactive features)
    std::string get_name_for_unknown_face(int track_id);
    void update_face_encounter(const std::string& name);
    
    // Statistics methods (exact Python analytics)
    std::map<std::string, int> get_face_statistics() const;
    int get_total_faces() const { return face_database_.size(); }
    std::vector<std::string> get_known_names() const;
    
    // Configuration methods
    void set_distance_threshold(float threshold) { distance_threshold_ = threshold; }
    float get_distance_threshold() const { return distance_threshold_; }
    bool is_available() const { return initialized_; }
    
private:
    // Face detection methods
    std::vector<cv::Rect> detect_faces_dnn(const cv::Mat& frame);
    std::vector<cv::Rect> detect_faces_haar(const cv::Mat& frame);
    
    // Encoding methods
    std::vector<float> compute_face_encoding(const cv::Mat& face_image);
    float compute_face_distance(const std::vector<float>& encoding1, const std::vector<float>& encoding2);
    
    // Database methods
    void cleanup_old_entries();
    bool validate_encoding(const std::vector<float>& encoding);
    
private:
    // Models
    cv::dnn::Net face_detector_;
    cv::dnn::Net face_encoder_;
    cv::CascadeClassifier haar_detector_;
    
    // Database (exact Python data structure)
    std::map<std::string, FaceDataCPP> face_database_;
    
    // Configuration (exact Python defaults)
    float distance_threshold_;
    float confidence_threshold_;
    int max_database_size_;
    int encoding_size_;
    
    // State
    bool initialized_;
    bool use_dnn_detector_;
    std::map<int, std::string> pending_names_; // For interactive naming
    std::set<std::string> session_names_;
    
    // Performance tracking
    std::chrono::steady_clock::time_point last_cleanup_;
    
    // Constants (exact Python values)
    static const float DEFAULT_DISTANCE_THRESHOLD;
    static const float DEFAULT_CONFIDENCE_THRESHOLD;
    static const int DEFAULT_MAX_DATABASE_SIZE;
    static const int DEFAULT_ENCODING_SIZE;
    static const int CLEANUP_INTERVAL_MINUTES;
};

} // namespace ORB_SLAM3

#endif // FACE_RECOGNIZER_CPP_H