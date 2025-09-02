/**
 * Enhanced ORB-SLAM3 Monocular RTMP Stream with Native C++ YOLO Integration
 * Directly integrates C++ YOLO detector without Python bridge
 * Eliminates shared memory complexity for high-performance person detection
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <System.h>
#include <YOLODetectorCPP.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#endif

using namespace std;
using namespace cv;

class RTMPStreamYOLO {
public:
    RTMPStreamYOLO(const string& vocab_path, const string& settings_path, const string& rtmp_url)
        : vocab_path_(vocab_path), settings_path_(settings_path), rtmp_url_(rtmp_url), running_(false)
    {
        // Initialize ORB-SLAM3
        cout << "🚀 Initializing ORB-SLAM3..." << endl;
        slam_system_ = new ORB_SLAM3::System(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, true);
        
        // Initialize native C++ YOLO detector
        cout << "🤖 Initializing Native C++ YOLO Detector..." << endl;
        yolo_detector_ = new ORB_SLAM3::YOLODetectorCPP();
        
        if (!yolo_detector_->Initialize()) {
            cout << "⚠️ YOLO detector initialization failed, continuing with dummy detection" << endl;
        }
        
        // Add demo zones
        yolo_detector_->AddZone("Entrance", Rect2f(50, 50, 200, 150), Scalar(0, 255, 0));
        yolo_detector_->AddZone("Main Area", Rect2f(300, 200, 300, 200), Scalar(255, 0, 0));
        yolo_detector_->AddZone("Exit", Rect2f(650, 400, 150, 100), Scalar(0, 0, 255));
    }
    
    ~RTMPStreamYOLO() {
        stop();
        if (slam_system_) {
            slam_system_->Shutdown();
            delete slam_system_;
        }
        if (yolo_detector_) {
            delete yolo_detector_;
        }
    }
    
    bool initialize() {
        cout << "📱 Connecting to RTMP stream: " << rtmp_url_ << endl;
        
        // Try to open RTMP stream
        if (!cap_.open(rtmp_url_)) {
            cout << "❌ Failed to connect to RTMP stream" << endl;
            cout << "📹 Falling back to default camera..." << endl;
            
            if (!cap_.open(0)) {
                cout << "❌ Failed to open camera" << endl;
                return false;
            }
        }
        
        // Set camera properties
        cap_.set(CAP_PROP_FRAME_WIDTH, 640);
        cap_.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap_.set(CAP_PROP_FPS, 30);
        
        cout << "✅ Camera initialized successfully" << endl;
        return true;
    }
    
    void run() {
        if (!initialize()) {
            return;
        }
        
        running_ = true;
        Mat frame, frame_rgb;
        double timestamp;
        int frame_count = 0;
        auto fps_timer = chrono::steady_clock::now();
        
        cout << "🎬 Starting RTMP stream processing with native C++ YOLO..." << endl;
        cout << "Press 'q' to quit, 'p' to pause" << endl;
        
        while (running_) {
            // Capture frame
            if (!cap_.read(frame)) {
                cout << "⚠️ Failed to read frame" << endl;
                break;
            }
            
            if (frame.empty()) {
                continue;
            }
            
            timestamp = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;
            
            // Convert BGR to RGB for ORB-SLAM3
            cvtColor(frame, frame_rgb, COLOR_BGR2RGB);
            
            // Process frame through ORB-SLAM3
            auto Tcw = slam_system_->TrackMonocular(frame_rgb, timestamp);
            
            // Process frame through native C++ YOLO detector
            auto yolo_start = chrono::steady_clock::now();
            vector<ORB_SLAM3::YOLODetectionCPP> detections = yolo_detector_->DetectAndTrack(frame, timestamp);
            auto yolo_end = chrono::steady_clock::now();
            auto yolo_time = chrono::duration_cast<chrono::milliseconds>(yolo_end - yolo_start);
            
            // Draw SLAM features and YOLO detections
            Mat display_frame = DrawFrame(frame, detections);
            
            // Add performance overlay
            DrawPerformanceOverlay(display_frame, yolo_time.count());
            
            // Display the frame
            imshow("ORB-SLAM3 + Native C++ YOLO", display_frame);
            
            // Handle keyboard input
            char key = waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) { // 'q' or ESC
                break;
            } else if (key == 'p') { // Pause
                cout << "⏸ Paused. Press any key to continue..." << endl;
                waitKey(0);
            }
            
            // Calculate and display FPS every 30 frames
            frame_count++;
            if (frame_count % 30 == 0) {
                auto current_time = chrono::steady_clock::now();
                auto duration = chrono::duration_cast<chrono::milliseconds>(current_time - fps_timer);
                float fps = 30000.0f / duration.count();
                fps_timer = current_time;
                
                cout << "📊 System FPS: " << fps << " | YOLO FPS: " << yolo_detector_->GetFPS() 
                     << " | Tracked: " << yolo_detector_->GetTrackedCount() << endl;
            }
        }
        
        stop();
    }
    
    void stop() {
        running_ = false;
        if (cap_.isOpened()) {
            cap_.release();
        }
        destroyAllWindows();
    }

private:
    Mat DrawFrame(const Mat& frame, const vector<ORB_SLAM3::YOLODetectionCPP>& detections) {
        Mat display_frame = frame.clone();
        
        // Draw zones
        auto zones = yolo_detector_->GetZones();
        for (const auto& zone : zones) {
            rectangle(display_frame, zone.area, zone.color, 2);
            
            // Add zone label with person count
            string zone_text = zone.name + ": " + to_string(zone.person_count);
            putText(display_frame, zone_text, 
                   Point(zone.area.x, zone.area.y - 10),
                   FONT_HERSHEY_SIMPLEX, 0.6, zone.color, 2);
        }
        
        // Draw YOLO detections
        for (const auto& detection : detections) {
            // Draw bounding box
            Scalar box_color(0, 255, 0); // Green for confirmed tracks
            rectangle(display_frame, detection.bbox, box_color, 2);
            
            // Draw track ID
            string track_text = "ID:" + to_string(detection.track_id);
            putText(display_frame, track_text,
                   Point(detection.bbox.x, detection.bbox.y - 30),
                   FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);
            
            // Draw person name
            if (!detection.person_name.empty() && detection.person_name != "Unknown") {
                putText(display_frame, detection.person_name,
                       Point(detection.bbox.x, detection.bbox.y - 10),
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
            }
            
            // Draw activity
            if (!detection.activity.empty()) {
                putText(display_frame, detection.activity,
                       Point(detection.bbox.x, detection.bbox.y + detection.bbox.height + 20),
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
            
            // Draw confidence
            string conf_text = cv::format("%.2f", detection.confidence);
            putText(display_frame, conf_text,
                   Point(detection.bbox.x + detection.bbox.width - 50, detection.bbox.y - 10),
                   FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
            
            // Draw trail
            if (detection.trail.size() > 1) {
                for (size_t i = 1; i < detection.trail.size(); ++i) {
                    line(display_frame, detection.trail[i-1], detection.trail[i], 
                         Scalar(255, 0, 255), 2);
                }
            }
            
            // Draw velocity arrow
            if (cv::norm(detection.velocity) > 1.0f) {
                Point2f center(detection.bbox.x + detection.bbox.width/2, 
                              detection.bbox.y + detection.bbox.height/2);
                Point2f arrow_end = center + detection.velocity * 5; // Scale for visibility
                arrowedLine(display_frame, center, arrow_end, Scalar(0, 255, 255), 2);
            }
        }
        
        return display_frame;
    }
    
    void DrawPerformanceOverlay(Mat& frame, int yolo_time_ms) {
        // Semi-transparent overlay
        Mat overlay = frame.clone();
        rectangle(overlay, Rect(10, 10, 300, 120), Scalar(0, 0, 0), -1);
        addWeighted(frame, 0.7, overlay, 0.3, 0, frame);
        
        // Performance text
        vector<string> info = {
            "Native C++ YOLO Integration",
            "YOLO Processing: " + to_string(yolo_time_ms) + "ms",
            "YOLO FPS: " + cv::format("%.1f", yolo_detector_->GetFPS()),
            "Tracked Objects: " + to_string(yolo_detector_->GetTrackedCount()),
            "Press 'q' to quit, 'p' to pause"
        };
        
        for (size_t i = 0; i < info.size(); ++i) {
            putText(frame, info[i], Point(15, 30 + i * 20),
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        }
    }
    
private:
    string vocab_path_;
    string settings_path_;
    string rtmp_url_;
    bool running_;
    
    VideoCapture cap_;
    ORB_SLAM3::System* slam_system_;
    ORB_SLAM3::YOLODetectorCPP* yolo_detector_;
};

int main(int argc, char **argv) {
    cout << "🚀 ORB-SLAM3 Monocular RTMP Stream with Native C++ YOLO" << endl;
    cout << "==========================================================" << endl;
    
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " path_to_vocabulary path_to_settings [rtmp_url]" << endl;
        cerr << "Example: " << argv[0] << " ../../Vocabulary/ORBvoc.txt iPhone16Plus.yaml rtmp://192.168.1.100:1935/live/test" << endl;
        return 1;
    }
    
    string vocab_path = argv[1];
    string settings_path = argv[2];
    string rtmp_url = (argc > 3) ? argv[3] : "rtmp://192.168.1.134:1935/live/test";
    
    try {
        RTMPStreamYOLO app(vocab_path, settings_path, rtmp_url);
        app.run();
        
        cout << "✅ Application finished successfully" << endl;
        return 0;
        
    } catch (const exception& e) {
        cerr << "❌ Application error: " << e.what() << endl;
        return 1;
    }
}