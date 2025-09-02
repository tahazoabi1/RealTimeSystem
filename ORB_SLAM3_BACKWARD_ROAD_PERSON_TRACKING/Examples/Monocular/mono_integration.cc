/**
* Real-time RTMP streaming with integrated person detection and tracking
* Combines working ORB-SLAM3 from mono_rtmp_stream with YOLO person detection
* Clean integration without complex hybrid system
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include<thread>
#include<mutex>
#include<condition_variable>
#include<vector>
#include<map>
#ifdef __APPLE__
#include<pthread.h>
#endif
#ifdef _WIN32
#include <windows.h>
#define sleep(x) Sleep((x)*1000)
#else
#include <unistd.h>
#endif

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#include"System.h"
#include "Converter.h"
#include "DeviceManagerCPP.h"

using namespace std;

// Simple tracking structure
struct PersonTrack {
    int id;
    cv::Rect2f bbox;
    float confidence;
    std::chrono::steady_clock::time_point last_seen;
    std::vector<cv::Point2f> trail;
};

struct RTMPData {
    std::mutex mutex;
    std::condition_variable cv;
    bool finished = false;
    bool dataReady = false;
    cv::Mat currentFrame;
    double currentTimestamp = 0.0;
    ORB_SLAM3::System* pSLAM = nullptr;
    cv::VideoCapture* pCapture = nullptr;
    string rtmp_url = "";
    int frameCount = 0;
    float imageScale = 1.0f;
    bool recording_started = false;
    
    // Person detection and tracking
    std::unique_ptr<ORB_SLAM3::DeviceManagerCPP> device_manager;
    std::map<int, PersonTrack> active_tracks;
    int next_track_id = 1;
    cv::Mat display_frame;
    std::mutex tracks_mutex;
};

void* RTMPProcessingThreadWrapper(void* arg);
void RTMPProcessingThread(RTMPData* pRTMPData);
void UpdatePersonTracking(RTMPData* pRTMPData, const std::vector<ORB_SLAM3::YOLODetectionCPP>& detections);
void DrawPersonTracking(RTMPData* pRTMPData, cv::Mat& frame);
float CalculateIOU(const cv::Rect2f& rect1, const cv::Rect2f& rect2);

float CalculateIOU(const cv::Rect2f& rect1, const cv::Rect2f& rect2)
{
    float x1 = std::max(rect1.x, rect2.x);
    float y1 = std::max(rect1.y, rect2.y);
    float x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    float y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = rect1.width * rect1.height;
    float area2 = rect2.width * rect2.height;
    float unionArea = area1 + area2 - intersection;
    
    return unionArea > 0 ? intersection / unionArea : 0.0f;
}

void UpdatePersonTracking(RTMPData* pRTMPData, const std::vector<ORB_SLAM3::YOLODetectionCPP>& detections)
{
    std::lock_guard<std::mutex> lock(pRTMPData->tracks_mutex);
    
    auto current_time = std::chrono::steady_clock::now();
    const float IOU_THRESHOLD = 0.3f;
    const float TIMEOUT_SECONDS = 2.0f;
    
    // Mark all tracks as not updated
    std::vector<bool> track_updated(pRTMPData->active_tracks.size(), false);
    std::vector<bool> detection_used(detections.size(), false);
    
    // Update existing tracks
    int track_idx = 0;
    for (auto& pair : pRTMPData->active_tracks) {
        PersonTrack& track = pair.second;
        
        float best_iou = 0.0f;
        int best_detection = -1;
        
        // Find best matching detection
        for (size_t i = 0; i < detections.size(); ++i) {
            if (detection_used[i]) continue;
            
            float iou = CalculateIOU(track.bbox, detections[i].bbox);
            if (iou > best_iou && iou > IOU_THRESHOLD) {
                best_iou = iou;
                best_detection = static_cast<int>(i);
            }
        }
        
        if (best_detection >= 0) {
            // Update track
            track.bbox = detections[best_detection].bbox;
            track.confidence = detections[best_detection].confidence;
            track.last_seen = current_time;
            
            // Add to trail
            cv::Point2f center(track.bbox.x + track.bbox.width/2, 
                             track.bbox.y + track.bbox.height/2);
            track.trail.push_back(center);
            if (track.trail.size() > 20) {
                track.trail.erase(track.trail.begin());
            }
            
            detection_used[best_detection] = true;
            track_updated[track_idx] = true;
        }
        
        track_idx++;
    }
    
    // Add new tracks for unmatched detections
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detection_used[i]) {
            PersonTrack new_track;
            new_track.id = pRTMPData->next_track_id++;
            new_track.bbox = detections[i].bbox;
            new_track.confidence = detections[i].confidence;
            new_track.last_seen = current_time;
            
            cv::Point2f center(new_track.bbox.x + new_track.bbox.width/2, 
                             new_track.bbox.y + new_track.bbox.height/2);
            new_track.trail.push_back(center);
            
            pRTMPData->active_tracks[new_track.id] = new_track;
        }
    }
    
    // Remove old tracks
    auto it = pRTMPData->active_tracks.begin();
    while (it != pRTMPData->active_tracks.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(
            current_time - it->second.last_seen);
        
        if (duration.count() > TIMEOUT_SECONDS) {
            it = pRTMPData->active_tracks.erase(it);
        } else {
            ++it;
        }
    }
}

void DrawPersonTracking(RTMPData* pRTMPData, cv::Mat& frame)
{
    std::lock_guard<std::mutex> lock(pRTMPData->tracks_mutex);
    
    const cv::Scalar colors[] = {
        cv::Scalar(0, 255, 0),   // Green
        cv::Scalar(255, 0, 0),   // Blue  
        cv::Scalar(0, 0, 255),   // Red
        cv::Scalar(255, 255, 0), // Cyan
        cv::Scalar(255, 0, 255), // Magenta
        cv::Scalar(0, 255, 255)  // Yellow
    };
    
    for (const auto& pair : pRTMPData->active_tracks) {
        const PersonTrack& track = pair.second;
        cv::Scalar color = colors[track.id % 6];
        
        // Draw bounding box
        cv::rectangle(frame, track.bbox, color, 2);
        
        // Draw track ID and confidence
        string label = "ID:" + to_string(track.id) + " " + 
                      to_string(static_cast<int>(track.confidence * 100)) + "%";
        
        cv::putText(frame, label, 
                   cv::Point(static_cast<int>(track.bbox.x), static_cast<int>(track.bbox.y - 10)),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        
        // Draw trail
        if (track.trail.size() > 1) {
            for (size_t i = 1; i < track.trail.size(); ++i) {
                cv::line(frame, track.trail[i-1], track.trail[i], color, 2);
            }
        }
    }
    
    // Add tracking info
    string track_info = "Active Tracks: " + to_string(pRTMPData->active_tracks.size());
    cv::putText(frame, track_info, cv::Point(10, frame.rows - 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
}

void RTMPProcessingThread(RTMPData* pRTMPData)
{
    cv::Mat frame;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    
    cout << "🎥 Starting RTMP stream processing with person detection..." << endl;
    cout << "📱 Waiting for iPhone stream at: " << pRTMPData->rtmp_url << endl;
    
    // Initialize person detection
    std::map<std::string, float> device_config;
    device_config["confidence_threshold"] = 0.4f;
    device_config["nms_threshold"] = 0.5f;
    
    if (!pRTMPData->device_manager->initialize(device_config)) {
        cout << "❌ Failed to initialize person detection, continuing with SLAM only" << endl;
    } else {
        cout << "✅ Person detection initialized successfully" << endl;
    }
    
    // Give viewer time to initialize
    usleep(1000000); // 1 second delay
    
    while (!pRTMPData->finished) {
        
        // Capture frame from RTMP stream
        if (!pRTMPData->pCapture->read(frame)) {
            cerr << "⚠️  Lost connection to RTMP stream, attempting to reconnect..." << endl;
            usleep(1000000); // Wait 1 second before retry
            continue;
        }
        
        if (frame.empty()) {
            cerr << "⚠️  Empty frame received from RTMP stream" << endl;
            continue;
        }
        
        // Create display frame copy for person detection visualization
        pRTMPData->display_frame = frame.clone();
        
        // Convert to grayscale for SLAM
        cv::Mat gray;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame.clone();
        }
        
        // Apply image scaling if needed
        if (pRTMPData->imageScale != 1.0f) {
            int width = gray.cols * pRTMPData->imageScale;
            int height = gray.rows * pRTMPData->imageScale;
            cv::resize(gray, gray, cv::Size(width, height));
        }
        
        // Apply CLAHE for better feature detection
        clahe->apply(gray, gray);
        
        // Validate frame quality before SLAM processing
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        if (stddev[0] < 5.0) {
            // Skip frames with too little variation (likely bad quality)
            continue;
        }
        
        // Get current timestamp
        double timestamp = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
        
        {
            std::lock_guard<std::mutex> lock(pRTMPData->mutex);
            if (pRTMPData->finished) break;
            pRTMPData->frameCount++;
        }
        
        // Process with ORB-SLAM3 (same as working version)
        if (pRTMPData->pSLAM) {
            static int consecutive_failures = 0;
            
            try {
                pRTMPData->pSLAM->TrackMonocular(gray, timestamp, 
                    vector<ORB_SLAM3::IMU::Point>(), 
                    "rtmp_frame_" + to_string(pRTMPData->frameCount));
                    
                consecutive_failures = 0; // Reset on success
            } catch (const std::exception& e) {
                cerr << "⚠️ Exception in TrackMonocular: " << e.what() << endl;
                consecutive_failures++;
                
                if (consecutive_failures > 50) {
                    cerr << "🔄 Too many tracking failures, attempting recovery..." << endl;
                    consecutive_failures = 0;
                }
            }
        }
        
        // Person detection and tracking (every 3rd frame for performance)
        if (pRTMPData->frameCount % 3 == 0) {
            try {
                std::vector<ORB_SLAM3::YOLODetectionCPP> detections = 
                    pRTMPData->device_manager->detect_persons(frame);
                
                if (!detections.empty()) {
                    UpdatePersonTracking(pRTMPData, detections);
                }
                
                // Draw person tracking on display frame
                DrawPersonTracking(pRTMPData, pRTMPData->display_frame);
                
            } catch (const std::exception& e) {
                // Continue without person detection if it fails
                cout << "⚠️ Person detection error: " << e.what() << endl;
            }
        } else {
            // Still draw existing tracks even when not detecting
            DrawPersonTracking(pRTMPData, pRTMPData->display_frame);
        }
        
        // Auto-start recording after 100 frames (when SLAM is stable)
        if (pRTMPData->frameCount == 100 && !pRTMPData->recording_started) {
            string filename = "live_path_" + to_string(time(nullptr)) + ".txt";
            pRTMPData->pSLAM->StartPathRecording(filename);
            pRTMPData->recording_started = true;
            cout << "🔴 AUTO-STARTED path recording: " << filename << endl;
        }
        
        // Progress indicator
        if (pRTMPData->frameCount % 100 == 0) {
            cout << "📹 Processed " << pRTMPData->frameCount << " frames from iPhone stream" 
                 << " | Active tracks: " << pRTMPData->active_tracks.size() << endl;
        }
    }
    
    cout << "🛑 RTMP processing thread finished" << endl;
}

void* RTMPProcessingThreadWrapper(void* arg)
{
    RTMPData* pRTMPData = static_cast<RTMPData*>(arg);
    RTMPProcessingThread(pRTMPData);
    return nullptr;
}

int main(int argc, char **argv)
{
    if(argc < 4) {
        cerr << endl << "Usage: ./mono_integration path_to_vocabulary path_to_settings rtmp_url" << endl;
        cerr << "Example: ./mono_integration Vocabulary/ORBvoc.txt Examples/Monocular/iPhone16Plus.yaml rtmp://localhost:1935/live/stream" << endl;
        return 1;
    }
    
    RTMPData rtmpData;
    
    // Set RTMP URL from command line argument
    rtmpData.rtmp_url = string(argv[3]);
    
    cout << "🎬 ORB-SLAM3 Real-Time iPhone Navigation with Person Detection" << endl;
    cout << "📱 iPhone 16 Plus Camera Configuration" << endl;
    cout << "🌐 RTMP Stream: " << rtmpData.rtmp_url << endl;
    cout << "👥 Person Detection: Enabled" << endl;
    cout << "============================================" << endl;
    
    // Initialize person detection
    rtmpData.device_manager = std::make_unique<ORB_SLAM3::DeviceManagerCPP>();
    
    // Initialize OpenCV VideoCapture for RTMP
    rtmpData.pCapture = new cv::VideoCapture();
    
    cout << "🔗 Connecting to RTMP stream..." << endl;
    
    // Try multiple connection methods with different backends (same as working version)
    vector<pair<string, int>> urls_and_backends = {
        {"http://localhost:8000/live/stream.flv", cv::CAP_FFMPEG},
        {"http://127.0.0.1:8000/live/stream.flv", cv::CAP_FFMPEG},
        {rtmpData.rtmp_url, cv::CAP_FFMPEG},
        {"rtmp://localhost:1935/live/stream", cv::CAP_FFMPEG},
        {"rtmp://127.0.0.1:1935/live/stream", cv::CAP_FFMPEG},
        {"http://localhost:8000/live/stream.flv", cv::CAP_ANY},
        {"http://127.0.0.1:8000/live/stream.flv", cv::CAP_ANY},
        {rtmpData.rtmp_url, cv::CAP_ANY},
        {"rtmp://localhost:1935/live/stream", cv::CAP_ANY},
        {"rtmp://127.0.0.1:1935/live/stream", cv::CAP_ANY}
    };
    
    bool connected = false;
    for (const auto& url_backend : urls_and_backends) {
        const string& url = url_backend.first;
        int backend = url_backend.second;
        string backend_name = (backend == cv::CAP_FFMPEG) ? "FFMPEG" : "ANY";
        
        cout << "🔄 Trying: " << url << " with " << backend_name << " backend" << endl;
        
        // Release previous capture
        rtmpData.pCapture->release();
        
        // Set FFMPEG-specific options for minimal latency
        if (backend == cv::CAP_FFMPEG) {
            rtmpData.pCapture->set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000); // 5 second timeout
            rtmpData.pCapture->set(cv::CAP_PROP_READ_TIMEOUT_MSEC, 1000);  // 1 second read timeout
            rtmpData.pCapture->set(cv::CAP_PROP_BUFFERSIZE, 0); // No buffering for minimal latency
            
            // Low-latency streaming options
            rtmpData.pCapture->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('H','2','6','4'));
            rtmpData.pCapture->set(cv::CAP_PROP_FORMAT, -1); // Auto format
            rtmpData.pCapture->set(cv::CAP_PROP_CONVERT_RGB, 0); // No RGB conversion
        }
        
        if (rtmpData.pCapture->open(url, backend)) {
            cout << "📡 Connection established, testing video data..." << endl;
            
            // Test if we can read frames with retries
            cv::Mat testFrame;
            bool frameReceived = false;
            for (int attempt = 0; attempt < 5; attempt++) {
                if (rtmpData.pCapture->read(testFrame) && !testFrame.empty()) {
                    frameReceived = true;
                    break;
                }
                cout << "⏳ Waiting for video data, attempt " << (attempt + 1) << "/5..." << endl;
                usleep(500000); // Wait 0.5 seconds
            }
            
            if (frameReceived) {
                rtmpData.rtmp_url = url;
                connected = true;
                cout << "✅ Successfully connected to: " << url << " with " << backend_name << endl;
                cout << "📹 Frame size: " << testFrame.cols << "x" << testFrame.rows << endl;
                break;
            } else {
                cout << "⚠️  Connected but no video data from: " << url << endl;
                rtmpData.pCapture->release();
            }
        } else {
            cout << "❌ Failed to connect to: " << url << " with " << backend_name << endl;
        }
        usleep(2000000); // Wait 2 seconds between attempts
    }
    
    if (!connected) {
        cerr << "❌ Failed to connect to any RTMP stream" << endl;
        cerr << "💡 Make sure your Node.js server is running and iPhone is streaming" << endl;
        cerr << "📱 iPhone should stream to: rtmp://[YOUR_MAC_IP]:1935/live/stream" << endl;
        return -1;
    }
    
    // Set capture properties for absolute minimum latency
    rtmpData.pCapture->set(cv::CAP_PROP_BUFFERSIZE, 0); // No buffering - real-time only
    rtmpData.pCapture->set(cv::CAP_PROP_FPS, 30); // Match iPhone stream FPS
    
    cout << "✅ Connected to RTMP stream successfully!" << endl;
    
    // Create ORB-SLAM3 system with viewer enabled (same parameters as working version)
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true, 0, "");
    rtmpData.pSLAM = &SLAM;
    rtmpData.imageScale = SLAM.GetImageScale();
    
    cout << "=== LIVE NAVIGATION + PERSON TRACKING =====" << endl;
    cout << "REAL-TIME CONTROLS:" << endl;
    cout << "  'R' - Start recording current path" << endl;
    cout << "  'S' - Stop recording" << endl;
    cout << "  'L' - Load recorded path for guidance" << endl;
    cout << "  'G' - Start REAL-TIME spoken guidance" << endl;
    cout << "  'H' - Stop guidance" << endl;
    cout << "  'B' - Toggle backwards navigation mode" << endl;
    cout << "📱 Point your iPhone camera and start moving!" << endl;
    cout << "👥 Person detection and tracking will be displayed" << endl;
    cout << "🔊 Audio navigation will guide you back along recorded paths" << endl;
    cout << "=============================================" << endl;
    
    // Start RTMP processing thread
#ifdef __APPLE__
    pthread_t rtmpThreadHandle;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    
    size_t stackSize = 16 * 1024 * 1024; // 16MB stack
    int stackResult = pthread_attr_setstacksize(&attr, stackSize);
    if (stackResult != 0) {
        cerr << "Failed to set stack size: " << stackResult << endl;
    }
    
    cout << "🎥 Starting real-time processing thread..." << endl;
    int result = pthread_create(&rtmpThreadHandle, &attr, RTMPProcessingThreadWrapper, &rtmpData);
    if (result != 0) {
        cerr << "❌ Failed to create RTMP thread: " << result << endl;
        return -1;
    }
    pthread_attr_destroy(&attr);
    
    // Main thread: Run viewer on main thread for macOS compatibility
    SLAM.RunViewerOnMainThread();
    
    // Wait for processing thread to complete
    pthread_join(rtmpThreadHandle, nullptr);
#else
    // Windows/Linux: Use std::thread
    cout << "🎥 Starting real-time processing thread..." << endl;
    std::thread rtmpThread(RTMPProcessingThread, &rtmpData);
    
    // Display thread for Windows
    cv::namedWindow("Person Detection + SLAM", cv::WINDOW_AUTOSIZE);
    
    while (!rtmpData.finished) {
        if (!rtmpData.display_frame.empty()) {
            cv::imshow("Person Detection + SLAM", rtmpData.display_frame);
            
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) { // 'q' or ESC
                rtmpData.finished = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    cv::destroyAllWindows();
    
    // Wait for processing thread
    rtmpThread.join();
#endif
    
    // Cleanup
    rtmpData.finished = true;
    SLAM.Shutdown();
    
    if (rtmpData.pCapture) {
        rtmpData.pCapture->release();
        delete rtmpData.pCapture;
    }
    
    cout << "🏁 Real-time navigation + person detection system shutdown complete" << endl;
    return 0;
}