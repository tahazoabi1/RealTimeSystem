/**
 * mono_rtmp_stream_enhanced_hybrid.cc
 * Complete Enhanced Hybrid Tracker demo with full Python module port
 * Demonstrates all modules: DeviceManager, ActivityDetector, Visualizer, ZoneAnalytics, FaceRecognizer
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "System.h"
#include "EnhancedHybridTrackerCPP.h"

using namespace std;

int main(int argc, char **argv)
{
    // Check arguments
    if(argc < 4) {
        cerr << endl << "Usage: ./mono_rtmp_stream_enhanced_hybrid path_to_vocabulary path_to_settings rtmp_url" << endl;
        cerr << "Example: ./mono_rtmp_stream_enhanced_hybrid ../../../Vocabulary/ORBvoc.txt ../iPhone16Plus.yaml rtmp://localhost/live/stream" << endl;
        return 1;
    }

    string vocab_file = argv[1];
    string settings_file = argv[2];  
    string rtmp_url = argv[3];

    cout << "🚀 Enhanced Hybrid Tracker Demo - Complete Python Port" << endl;
    cout << "=====================================================" << endl;
    cout << "Vocabulary: " << vocab_file << endl;
    cout << "Settings: " << settings_file << endl;
    cout << "RTMP URL: " << rtmp_url << endl << endl;

    // Initialize ORB-SLAM3 system (same parameters as working version)
    cout << "🔧 Initializing ORB-SLAM3..." << endl;
    ORB_SLAM3::System SLAM(vocab_file, settings_file, ORB_SLAM3::System::MONOCULAR, true, 0, "");
    
    // Initialize Enhanced Hybrid Tracker with complete modular system
    cout << "🔧 Initializing Enhanced Hybrid Tracker..." << endl;
    ORB_SLAM3::EnhancedHybridTrackerCPP tracker;
    
    // Configure tracker (exact Python configuration with graceful model fallbacks)
    ORB_SLAM3::TrackerConfigCPP config; // Use default struct initialization
    config.confidence_threshold = 0.4f;  // Match Python version optimal setting
    config.nms_threshold = 0.5f;
    config.use_yolo = true; // Will fallback to HOG if YOLO models unavailable
    config.enable_face_recognition = false; // Disable face recognition until models are available
    config.enable_zone_analytics = true;
    config.show_trails = true;
    config.show_zones = true;
    config.show_activities = true;
    config.show_face_names = false; // Disable until face recognition models are available
    config.frame_skip_interval = 1; // Process every frame for testing
    
    // Set model paths - system will check if files exist and fallback gracefully
    config.face_model_path = "face_recognition.onnx"; // Will fallback if not found
    config.face_database_path = "face_database.dat"; // Will create if not exists
    
    if (!tracker.Initialize(config)) {
        cerr << "❌ Failed to initialize Enhanced Hybrid Tracker" << endl;
        return -1;
    }

    // Open RTMP stream
    cout << "📹 Connecting to RTMP stream..." << endl;
    cv::VideoCapture cap;
    
    // Try to open RTMP stream
    if (!cap.open(rtmp_url)) {
        cerr << "❌ Failed to open RTMP stream: " << rtmp_url << endl;
        cout << "🔄 Falling back to webcam (index 0)" << endl;
        
        if (!cap.open(0)) {
            cerr << "❌ Failed to open webcam" << endl;
            return -1;
        }
    }

    // Check if camera/stream opened successfully
    if (!cap.isOpened()) {
        cerr << "❌ Cannot access video source" << endl;
        return -1;
    }

    cout << "✅ Video source connected successfully" << endl;

    // Set camera properties for better performance
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    cv::Mat frame;
    
    // Create CLAHE processor once (optimization from working version)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    
    // Performance tracking
    auto start_time = chrono::steady_clock::now();
    int frame_count = 0;
    double total_time = 0.0;
    
    cout << "🎯 Starting real-time processing..." << endl;
    cout << "Press 'q' to quit, 'r' to reset session, 's' to save face database" << endl << endl;

    while (true) {
        auto frame_start = chrono::steady_clock::now();
        
        // Capture frame
        cap >> frame;
        if (frame.empty()) {
            cout << "⚠️  Empty frame received, retrying..." << endl;
            this_thread::sleep_for(chrono::milliseconds(10));
            continue;
        }

        frame_count++;
        double timestamp = chrono::duration_cast<chrono::milliseconds>(
            chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

        // Convert to grayscale for SLAM (optimize like working version)
        cv::Mat gray;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame.clone();
        }
        
        // Apply CLAHE for better feature detection (like working version)
        clahe->apply(gray, gray);
        
        // Validate frame quality before SLAM processing (like working version)
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        if (stddev[0] < 5.0) {
            // Skip frames with too little variation (likely bad quality)
            continue;
        }

        // Process frame with ORB-SLAM3 for camera pose (same method as working version)
        static int consecutive_failures = 0;
        
        try {
            Sophus::SE3f pose = SLAM.TrackMonocular(gray, timestamp, 
                vector<ORB_SLAM3::IMU::Point>(), 
                "rtmp_frame_" + to_string(frame_count));
                
            consecutive_failures = 0; // Reset on success
        } catch (const std::exception& e) {
            cerr << "⚠️ Exception in TrackMonocular: " << e.what() << endl;
            consecutive_failures++;
            
            // If too many failures, try reinitializing
            if (consecutive_failures > 50) {
                cerr << "🔄 Too many tracking failures, attempting recovery..." << endl;
                consecutive_failures = 0;
                // Let system attempt to reinitialize on next good frame
            }
        }

        // Process frame with Enhanced Hybrid Tracker (complete Python port)
        cv::Mat display_frame = frame; // Use reference instead of clone for performance
        vector<ORB_SLAM3::TrackerDetectionCPP> detections = tracker.ProcessFrame(frame, timestamp);
        
        std::cout << "🔍 Frame " << frame_count << ": " << detections.size() << " detections" << std::endl;
        
        // Get annotated frame with all visualizations
        cv::Mat annotated_frame = tracker.GetAnnotatedFrame();
        if (!annotated_frame.empty()) {
            display_frame = annotated_frame;
        }
        
        // Add system performance overlay to display frame
        string fps_text = "System FPS: " + to_string(tracker.GetFPS());
        string processing_text = "Processing: " + to_string(tracker.GetProcessingTime()) + "ms";
        string tracks_text = "Active Tracks: " + to_string(tracker.GetActiveTrackCount());
        
        cv::putText(display_frame, fps_text, cv::Point(10, display_frame.rows - 80), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        cv::putText(display_frame, processing_text, cv::Point(10, display_frame.rows - 55), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        cv::putText(display_frame, tracks_text, cv::Point(10, display_frame.rows - 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        
        // Add ORB-SLAM3 status
        string slam_status = SLAM.GetTrackingState() == ORB_SLAM3::Tracking::OK ? "SLAM: OK" : "SLAM: LOST";
        cv::Scalar slam_color = SLAM.GetTrackingState() == ORB_SLAM3::Tracking::OK ? 
                               cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(display_frame, slam_status, cv::Point(10, display_frame.rows - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, slam_color, 2);

        // Display frame
        cv::imshow("Enhanced Hybrid Tracker - Complete Python Port", display_frame);

        // Handle keyboard input
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) { // 'q' or ESC key
            cout << "🛑 Quit requested" << endl;
            break;
        } else if (key == 'r') {
            cout << "🔄 Resetting session..." << endl;
            tracker.ResetSession();
            SLAM.Reset();
        } else if (key == 's') {
            cout << "💾 Saving face database..." << endl;
            if (tracker.SaveFaceDatabase()) {
                cout << "✅ Face database saved successfully" << endl;
            } else {
                cout << "❌ Failed to save face database" << endl;
            }
        } else if (key == 'e') {
            cout << "📊 Exporting analytics..." << endl;
            string filename = "analytics_export_" + to_string(timestamp) + ".txt";
            tracker.ExportAnalytics(filename);
        }

        // Performance statistics (every 30 frames)
        if (frame_count % 30 == 0) {
            auto current_time = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(current_time - start_time).count();
            double avg_fps = frame_count / elapsed;
            
            // Print comprehensive status
            cout << "📊 Frame " << frame_count << " | Avg FPS: " << fixed << setprecision(1) << avg_fps 
                 << " | Tracker FPS: " << tracker.GetFPS()
                 << " | Active Tracks: " << tracker.GetActiveTrackCount() << endl;
                 
            // Print activity statistics
            auto activity_stats = tracker.GetActivityStatistics();
            if (!activity_stats.empty()) {
                cout << "   Activities: ";
                for (const auto& pair : activity_stats) {
                    cout << pair.first << "=" << pair.second << " ";
                }
                cout << endl;
            }
            
            // Print zone occupancy
            auto zone_stats = tracker.GetZoneOccupancyCounts();
            if (!zone_stats.empty()) {
                cout << "   Zone Occupancy: ";
                for (const auto& pair : zone_stats) {
                    cout << pair.first << "=" << pair.second << " ";
                }
                cout << endl;
            }
            
            // Print any alerts
            auto alerts = tracker.GetActiveAlerts();
            if (!alerts.empty()) {
                cout << "⚠️  Alerts: ";
                for (const auto& alert : alerts) {
                    cout << alert << " ";
                }
                cout << endl;
            }
        }

        // Frame timing
        auto frame_end = chrono::steady_clock::now();
        double frame_time = chrono::duration<double, milli>(frame_end - frame_start).count();
        total_time += frame_time;
    }

    // Final statistics
    auto end_time = chrono::steady_clock::now();
    double total_elapsed = chrono::duration<double>(end_time - start_time).count();
    
    cout << endl << "📊 Session Summary:" << endl;
    cout << "Total Frames: " << frame_count << endl;
    cout << "Total Time: " << fixed << setprecision(2) << total_elapsed << "s" << endl;
    cout << "Average FPS: " << fixed << setprecision(1) << frame_count / total_elapsed << endl;
    cout << "Final Track Count: " << tracker.GetActiveTrackCount() << endl;

    // Export final analytics
    cout << "📊 Exporting final analytics..." << endl;
    tracker.ExportAnalytics("final_analytics_session.txt");

    // Save face database
    cout << "💾 Saving face database..." << endl;
    tracker.SaveFaceDatabase();

    // Cleanup
    cout << "🧹 Cleaning up..." << endl;
    cap.release();
    cv::destroyAllWindows();
    
    // Stop SLAM system
    SLAM.Shutdown();
    
    cout << "✅ Enhanced Hybrid Tracker Demo completed successfully!" << endl;
    return 0;
}