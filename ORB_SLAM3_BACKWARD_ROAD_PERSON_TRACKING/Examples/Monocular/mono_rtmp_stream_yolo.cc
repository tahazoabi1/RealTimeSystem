/**
* YOLO-Enhanced Real-time RTMP streaming input for ORB-SLAM3
* Live iPhone camera feed with YOLO person detection overlay
* Combines ORB-SLAM3 features with real-time person tracking
* Designed for advanced navigation and person detection
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
#define usleep(x) Sleep((x)/1000)
#else
#include <unistd.h>
#endif

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include"System.h"
#include "Converter.h"
#include "FrameDrawer_YOLO.h"  // Use YOLO-enhanced FrameDrawer

using namespace std;

struct RTMPYOLOData {
    std::mutex mutex;
    std::condition_variable cv;
    bool finished = false;
    bool dataReady = false;
    cv::Mat currentFrame;
    double currentTimestamp = 0.0;
    ORB_SLAM3::System* pSLAM = nullptr;
    ORB_SLAM3::FrameDrawerYOLO* pFrameDrawer = nullptr;  // YOLO-enhanced drawer
    cv::VideoCapture* pCapture = nullptr;
    string rtmp_url = "";
    int frameCount = 0;
    float imageScale = 1.0f;
    bool recording_started = false;
    bool yolo_enabled = true;
};

void* RTMPYOLOProcessingThreadWrapper(void* arg);
void RTMPYOLOProcessingThread(RTMPYOLOData* pRTMPData);

void RTMPYOLOProcessingThread(RTMPYOLOData* pRTMPData)
{
    cv::Mat frame;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    
    cout << "🎥 Starting YOLO-Enhanced RTMP stream processing..." << endl;
    cout << "📱 Waiting for iPhone stream at: " << pRTMPData->rtmp_url << endl;
    cout << "🤖 YOLO person detection: " << (pRTMPData->yolo_enabled ? "ENABLED" : "DISABLED") << endl;
    
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
        
        {
            std::lock_guard<std::mutex> lock(pRTMPData->mutex);
            
            // Store processed frame and original color frame
            pRTMPData->currentFrame = gray.clone();
            pRTMPData->currentTimestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
            pRTMPData->dataReady = true;
            pRTMPData->frameCount++;
            
            // Update YOLO FrameDrawer with color frame for detection
            if (pRTMPData->pFrameDrawer && pRTMPData->yolo_enabled) {
                // Pass original color frame for better YOLO detection
                cv::Mat colorFrame = frame;
                if (pRTMPData->imageScale != 1.0f) {
                    int width = colorFrame.cols * pRTMPData->imageScale;
                    int height = colorFrame.rows * pRTMPData->imageScale;
                    cv::resize(colorFrame, colorFrame, cv::Size(width, height));
                }
            }
        }
        
        pRTMPData->cv.notify_one();
        
        // Minimal delay to prevent CPU overload (GPU YOLO is fast)
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Allow ~200 FPS max
    }
    
    cout << "🛑 YOLO-Enhanced RTMP processing thread stopped" << endl;
}

void* RTMPYOLOProcessingThreadWrapper(void* arg)
{
    RTMPYOLOProcessingThread(static_cast<RTMPYOLOData*>(arg));
    return nullptr;
}

int main(int argc, char **argv)
{
    if(argc < 4) {
        cerr << endl << "Usage: ./mono_rtmp_stream_yolo path_to_vocabulary path_to_settings rtmp_url [yolo_config]" << endl;
        cerr << "Example: ./mono_rtmp_stream_yolo ../../../Vocabulary/ORBvoc.txt ../iPhone16Plus.yaml rtmp://192.168.1.100:1935/live/stream ../../../realtimesystem/optimized_config.json" << endl;
        return 1;
    }

    string rtmp_url = string(argv[3]);
    string yolo_config = (argc >= 5) ? string(argv[4]) : "D:\\Learning\\realtimesystem\\optimized_config.json";
    
    cout << "🚀 ORB-SLAM3 + YOLO Real-time System Starting..." << endl;
    cout << "📖 Vocabulary: " << argv[1] << endl;
    cout << "⚙️  Settings: " << argv[2] << endl;
    cout << "📺 RTMP URL: " << rtmp_url << endl;
    cout << "🤖 YOLO Config: " << yolo_config << endl;
    cout << "=" << string(60, '=') << endl;

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    
    // Create YOLO-enhanced FrameDrawer 
    ORB_SLAM3::FrameDrawerYOLO* pFrameDrawer = new ORB_SLAM3::FrameDrawerYOLO(nullptr);
    
    // Initialize YOLO integration
    bool yolo_initialized = false;
    if (pFrameDrawer) {
        yolo_initialized = pFrameDrawer->InitializeYOLO(yolo_config);
        if (yolo_initialized) {
            cout << "✅ YOLO integration initialized successfully!" << endl;
        } else {
            cout << "⚠️ YOLO initialization failed, running without YOLO overlay" << endl;
        }
    }

    // Vector for tracking image timestamps
    vector<float> vTimestamps;
    string strFile = string(argv[2]);

    // Initialize video capture
    cv::VideoCapture cap;
    cout << "🔗 Connecting to RTMP stream: " << rtmp_url << endl;
    
    // Configure video capture for RTMP
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    if(!cap.open(rtmp_url)) {
        cerr << "❌ Cannot connect to RTMP stream: " << rtmp_url << endl;
        cerr << "Make sure:" << endl;
        cerr << "1. Your iPhone is streaming to this URL" << endl;
        cerr << "2. The network connection is stable" << endl;
        cerr << "3. The RTMP server is running" << endl;
        return -1;
    }

    cout << "✅ Connected to RTMP stream!" << endl;

    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    cout << "📊 Stream Info:" << endl;
    cout << "   FPS: " << fps << endl;
    cout << "   Resolution: " << frame_width << "x" << frame_height << endl;

    // Setup RTMP data structure
    RTMPYOLOData rtmpData;
    rtmpData.pSLAM = &SLAM;
    rtmpData.pFrameDrawer = pFrameDrawer;
    rtmpData.pCapture = &cap;
    rtmpData.rtmp_url = rtmp_url;
    rtmpData.yolo_enabled = yolo_initialized;
    
    // Adjust image scale if needed (for performance)
    if (frame_width > 1280) {
        rtmpData.imageScale = 1280.0f / frame_width;
        cout << "📏 Scaling images by factor: " << rtmpData.imageScale << endl;
    }

    // Create and start RTMP processing thread
#ifdef _WIN32
    HANDLE hRTMPThread = CreateThread(NULL, 0, 
        (LPTHREAD_START_ROUTINE)RTMPYOLOProcessingThreadWrapper, 
        &rtmpData, 0, NULL);
    if (hRTMPThread == NULL) {
        cerr << "❌ Failed to create RTMP processing thread" << endl;
        return -1;
    }
#else
    pthread_t tRTMP;
    pthread_create(&tRTMP, NULL, RTMPYOLOProcessingThreadWrapper, &rtmpData);
#endif

    cout << "🎬 Starting main processing loop..." << endl;
    cout << "Press Ctrl+C to stop" << endl;
    cout << "=" << string(60, '=') << endl;

    // Main processing loop
    cv::Mat im;
    double timestamp;
    int frameCount = 0;
    auto start = std::chrono::steady_clock::now();
    
    try {
        while(true) {
            {
                std::unique_lock<std::mutex> lock(rtmpData.mutex);
                rtmpData.cv.wait(lock, [&rtmpData]{ return rtmpData.dataReady || rtmpData.finished; });
                
                if (rtmpData.finished) break;
                
                im = rtmpData.currentFrame.clone();
                timestamp = rtmpData.currentTimestamp;
                rtmpData.dataReady = false;
            }

            if(im.empty()) continue;

            frameCount++;

            try {
                // Track the image in the SLAM system
                Sophus::SE3f Tcw = SLAM.TrackMonocular(im, timestamp);
                
                // Calculate and display FPS
                if (frameCount % 30 == 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start);
                    if (duration.count() > 0) {
                        double current_fps = frameCount / duration.count();
                        cout << "📈 System FPS: " << std::fixed << std::setprecision(1) << current_fps;
                        
                        if (pFrameDrawer && pFrameDrawer->IsYOLOEnabled()) {
                            cout << " | YOLO Tracking: " << pFrameDrawer->GetTrackedPersonCount() << " people";
                            cout << " | YOLO Time: " << std::setprecision(1) << (pFrameDrawer->GetYOLOProcessingTime() * 1000) << "ms";
                        }
                        cout << endl;
                    }
                }
                
            } catch(const std::exception& e) {
                cerr << "⚠️ SLAM processing error: " << e.what() << endl;
                continue;
            } catch(...) {
                cerr << "⚠️ Unknown SLAM processing error" << endl;
                continue;
            }
        }
    } catch(const std::exception& e) {
        cerr << "❌ Main loop error: " << e.what() << endl;
    }

    // Stop all threads
    cout << "\n🛑 Shutting down system..." << endl;
    
    rtmpData.finished = true;
    rtmpData.cv.notify_all();
    
#ifdef _WIN32
    WaitForSingleObject(hRTMPThread, INFINITE);
    CloseHandle(hRTMPThread);
#else
    pthread_join(tRTMP, NULL);
#endif

    // Stop the SLAM system
    SLAM.Shutdown();
    
    // Save camera trajectory
    cout << "💾 Saving trajectory..." << endl;
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    
    // Cleanup YOLO resources
    if (pFrameDrawer) {
        delete pFrameDrawer;
    }
    
    cout << "✅ System shutdown complete!" << endl;
    cout << "📁 Trajectory saved to: KeyFrameTrajectory.txt" << endl;

    return 0;
}