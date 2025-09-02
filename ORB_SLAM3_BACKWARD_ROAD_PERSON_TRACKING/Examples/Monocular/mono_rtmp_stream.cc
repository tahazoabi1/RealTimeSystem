/**
* Real-time RTMP streaming input for ORB-SLAM3
* Live iPhone camera feed with navigation guidance
* Designed for blind navigation testing
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include<thread>
#include<mutex>
#include<condition_variable>
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

#include"System.h"
#include "Converter.h"

using namespace std;

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
};

void* RTMPProcessingThreadWrapper(void* arg);
void RTMPProcessingThread(RTMPData* pRTMPData);

void RTMPProcessingThread(RTMPData* pRTMPData)
{
    cv::Mat frame;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    
    cout << "🎥 Starting RTMP stream processing..." << endl;
    cout << "📱 Waiting for iPhone stream at: " << pRTMPData->rtmp_url << endl;
    
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
        
        // Process with ORB-SLAM3
        if (pRTMPData->pSLAM) {
            // Track consecutive tracking failures
            static int consecutive_failures = 0;
            
            try {
                pRTMPData->pSLAM->TrackMonocular(gray, timestamp, 
                    vector<ORB_SLAM3::IMU::Point>(), 
                    "rtmp_frame_" + to_string(pRTMPData->frameCount));
                    
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
            cout << "📹 Processed " << pRTMPData->frameCount << " frames from iPhone stream" << endl;
        }
        
        // No artificial delay for maximum real-time performance
        // usleep(1000); // Removed delay
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
        cerr << endl << "Usage: ./mono_rtmp_stream path_to_vocabulary path_to_settings rtmp_url" << endl;
        cerr << "Example: ./mono_rtmp_stream Vocabulary/ORBvoc.txt Examples/Monocular/iPhone16Plus.yaml rtmp://localhost:1935/live/stream" << endl;
        return 1;
    }
    
    RTMPData rtmpData;
    
    // Set RTMP URL from command line argument
    rtmpData.rtmp_url = string(argv[3]);
    
    cout << "🎬 ORB-SLAM3 Real-Time iPhone Navigation System" << endl;
    cout << "📱 iPhone 16 Plus Camera Configuration" << endl;
    cout << "🌐 RTMP Stream: " << rtmpData.rtmp_url << endl;
    cout << "===========================================" << endl;
    
    // Initialize OpenCV VideoCapture for RTMP
    rtmpData.pCapture = new cv::VideoCapture();
    
    cout << "🔗 Connecting to RTMP stream..." << endl;
    
    // Try multiple connection methods with different backends
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
    
    // Create ORB-SLAM3 system with viewer enabled
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true, 0, "");
    rtmpData.pSLAM = &SLAM;
    rtmpData.imageScale = SLAM.GetImageScale();
    
    cout << "=== LIVE NAVIGATION SYSTEM READY ====" << endl;
    cout << "REAL-TIME CONTROLS:" << endl;
    cout << "  'R' - Start recording current path" << endl;
    cout << "  'S' - Stop recording" << endl;
    cout << "  'L' - Load recorded path for guidance" << endl;
    cout << "  'G' - Start REAL-TIME spoken guidance" << endl;
    cout << "  'H' - Stop guidance" << endl;
    cout << "  'B' - Toggle backwards navigation mode" << endl;
    cout << "📱 Point your iPhone camera and start moving!" << endl;
    cout << "🔊 Audio navigation will guide you back along recorded paths" << endl;
    cout << "======================================" << endl;
    
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
    
    cout << "🏁 Real-time navigation system shutdown complete" << endl;
    return 0;
}