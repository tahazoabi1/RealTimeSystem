/**
* Simple YOLO-Enhanced Real-time RTMP streaming input for ORB-SLAM3
* Live iPhone camera feed with basic YOLO overlay
* Standalone implementation without complex inheritance
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
#include<opencv2/highgui/highgui.hpp>

#include"System.h"
#include "Converter.h"

using namespace std;

struct SimpleYOLOData {
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
    bool yolo_enabled = true;
    cv::Mat yolo_overlay;
    std::mutex overlay_mutex;
};

// YOLO detection struct matching enhanced_hybrid_tracker_modular.py output
struct RealDetection {
    cv::Rect2f bbox;
    float confidence;
    std::string label;
    int track_id;
    std::string person_name;
    std::string activity;
    cv::Point2f velocity;
    std::vector<cv::Point2f> trail;
};

// Real Python YOLO interface using enhanced_hybrid_tracker_modular.py
class RealPythonYOLO {
private:
    bool initialized_ = false;
    std::string config_path_;
    std::vector<RealDetection> current_detections_;
    std::mutex detections_mutex_;
    
#ifdef _WIN32
    HANDLE hSharedMemory = NULL;
    void* pSharedMemory = nullptr;
    HANDLE hPythonProcess = NULL;
#endif
    
    struct SharedFrameData {
        int width;
        int height;
        int channels;
        double timestamp;
        bool frame_ready;
        bool processing_complete;
        int num_detections;
        // Frame data follows this header
        // Detection data follows frame data
    };
    
    struct SharedDetectionData {
        float x, y, width, height;
        float confidence;
        char label[32];
        int track_id;
        char person_name[64];
        char activity[32];
        float velocity_x, velocity_y;
        int trail_length;
        // Trail points follow if trail_length > 0
    };
    
public:
    RealPythonYOLO() = default;
    
    ~RealPythonYOLO() {
#ifdef _WIN32
        if (hPythonProcess) {
            TerminateProcess(hPythonProcess, 0);
            CloseHandle(hPythonProcess);
        }
        if (pSharedMemory) UnmapViewOfFile(pSharedMemory);
        if (hSharedMemory) CloseHandle(hSharedMemory);
#endif
    }
    
    bool Initialize(const std::string& config_path) {
        config_path_ = config_path;
        
        std::cout << "🤖 Initializing Real YOLO with enhanced_hybrid_tracker_modular.py..." << std::endl;
        
#ifdef _WIN32
        // Create shared memory for frame exchange
        const size_t SHARED_MEMORY_SIZE = 1920 * 1080 * 3 + sizeof(SharedFrameData) + 50 * sizeof(SharedDetectionData);
        
        hSharedMemory = CreateFileMappingA(
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            0,
            SHARED_MEMORY_SIZE,
            "ORB_SLAM3_YOLO_SharedMemory"
        );
        
        if (!hSharedMemory) {
            std::cerr << "❌ Failed to create shared memory" << std::endl;
            return false;
        }
        
        pSharedMemory = MapViewOfFile(hSharedMemory, FILE_MAP_ALL_ACCESS, 0, 0, SHARED_MEMORY_SIZE);
        if (!pSharedMemory) {
            std::cerr << "❌ Failed to map shared memory" << std::endl;
            return false;
        }
        
        // Initialize shared memory
        memset(pSharedMemory, 0, SHARED_MEMORY_SIZE);
        
        // Launch Python process with proper config path
        std::string python_cmd = "python D:\\Learning\\realtimesystem\\orbslam_bridge_windows.py";
        if (!config_path_.empty() && config_path_ != "default_config") {
            python_cmd += " " + config_path_;
        } else {
            python_cmd += " D:\\Learning\\realtimesystem\\optimized_config.json";
        }
        
        STARTUPINFO si = {0};
        PROCESS_INFORMATION pi = {0};
        si.cb = sizeof(si);
        
        if (!CreateProcessA(NULL, const_cast<char*>(python_cmd.c_str()), NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
            std::cerr << "❌ Failed to start Python process" << std::endl;
            return false;
        }
        
        hPythonProcess = pi.hProcess;
        CloseHandle(pi.hThread);
        
        // Wait for Python initialization and shared memory connection
        std::cout << "⏳ Waiting for Python bridge to connect..." << std::endl;
        Sleep(5000);  // 5 seconds for Python to initialize and connect
        
        initialized_ = true;
        std::cout << "✅ Real YOLO interface initialized with enhanced_hybrid_tracker_modular.py!" << std::endl;
        return true;
#else
        // Linux/macOS implementation would go here
        initialized_ = true;
        return true;
#endif
    }
    
    bool ProcessFrame(const cv::Mat& frame) {
        if (!initialized_ || !pSharedMemory) return false;
        
#ifdef _WIN32
        SharedFrameData* frame_header = static_cast<SharedFrameData*>(pSharedMemory);
        
        // Wait for previous processing to complete
        while (frame_header->frame_ready && !frame_header->processing_complete) {
            Sleep(1);
        }
        
        // Copy frame data to shared memory
        frame_header->width = frame.cols;
        frame_header->height = frame.rows;
        frame_header->channels = frame.channels();
        frame_header->timestamp = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        frame_header->processing_complete = false;
        
        // Copy frame pixels
        uint8_t* frame_data = static_cast<uint8_t*>(pSharedMemory) + sizeof(SharedFrameData);
        memcpy(frame_data, frame.data, frame.total() * frame.elemSize());
        
        // Signal frame ready
        frame_header->frame_ready = true;
        
        // Wait for processing (with timeout)
        int timeout_ms = 100; // 100ms timeout
        while (timeout_ms > 0 && !frame_header->processing_complete) {
            Sleep(1);
            timeout_ms--;
        }
        
        if (frame_header->processing_complete) {
            // Read detection results
            std::lock_guard<std::mutex> lock(detections_mutex_);
            current_detections_.clear();
            
            int num_detections = frame_header->num_detections;
            if (num_detections > 0) {
                SharedDetectionData* det_data = reinterpret_cast<SharedDetectionData*>(
                    frame_data + frame.total() * frame.elemSize()
                );
                
                for (int i = 0; i < num_detections && i < 50; ++i) {
                    RealDetection det;
                    det.bbox = cv::Rect2f(det_data[i].x, det_data[i].y, det_data[i].width, det_data[i].height);
                    det.confidence = det_data[i].confidence;
                    det.label = std::string(det_data[i].label);
                    det.track_id = det_data[i].track_id;
                    det.person_name = std::string(det_data[i].person_name);
                    det.activity = std::string(det_data[i].activity);
                    det.velocity = cv::Point2f(det_data[i].velocity_x, det_data[i].velocity_y);
                    
                    current_detections_.push_back(det);
                }
            }
            
            return true;
        }
#endif
        return false;
    }
    
    std::vector<RealDetection> GetDetections() {
        std::lock_guard<std::mutex> lock(detections_mutex_);
        return current_detections_;
    }
    
    bool IsReady() const { return initialized_; }
};

RealPythonYOLO g_yolo_interface;

void* RTMPProcessingThreadWrapper(void* arg);
void RTMPProcessingThread(SimpleYOLOData* pRTMPData);

void RTMPProcessingThread(SimpleYOLOData* pRTMPData)
{
    cv::Mat frame;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    
    cout << "🎥 Starting Simple YOLO-Enhanced RTMP stream processing..." << endl;
    cout << "📱 Waiting for iPhone stream at: " << pRTMPData->rtmp_url << endl;
    
    usleep(1000000); // 1 second delay
    
    while (!pRTMPData->finished) {
        
        if (!pRTMPData->pCapture->read(frame)) {
            cerr << "⚠️  Lost connection to RTMP stream, attempting to reconnect..." << endl;
            usleep(1000000); // Wait 1 second before retry
            continue;
        }
        
        if (frame.empty()) {
            cerr << "⚠️  Empty frame received from RTMP stream" << endl;
            continue;
        }
        
        // Process YOLO on color frame
        if (pRTMPData->yolo_enabled && g_yolo_interface.IsReady()) {
            g_yolo_interface.ProcessFrame(frame);
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
            continue;
        }
        
        {
            std::lock_guard<std::mutex> lock(pRTMPData->mutex);
            
            pRTMPData->currentFrame = gray.clone();
            pRTMPData->currentTimestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
            pRTMPData->dataReady = true;
            pRTMPData->frameCount++;
            
            // Create YOLO overlay
            if (pRTMPData->yolo_enabled) {
                cv::Mat overlay_frame;
                cv::cvtColor(gray, overlay_frame, cv::COLOR_GRAY2BGR);
                
                auto detections = g_yolo_interface.GetDetections();
                for (const auto& det : detections) {
                    cv::Rect2f scaled_bbox = det.bbox;
                    if (pRTMPData->imageScale != 1.0f) {
                        scaled_bbox.x *= pRTMPData->imageScale;
                        scaled_bbox.y *= pRTMPData->imageScale;
                        scaled_bbox.width *= pRTMPData->imageScale;
                        scaled_bbox.height *= pRTMPData->imageScale;
                    }
                    
                    // Choose color based on track ID
                    cv::Scalar color;
                    switch (det.track_id % 6) {
                        case 0: color = cv::Scalar(0, 255, 0); break;    // Green
                        case 1: color = cv::Scalar(255, 0, 0); break;    // Blue
                        case 2: color = cv::Scalar(0, 0, 255); break;    // Red
                        case 3: color = cv::Scalar(255, 255, 0); break;  // Cyan
                        case 4: color = cv::Scalar(255, 0, 255); break;  // Magenta
                        case 5: color = cv::Scalar(0, 255, 255); break;  // Yellow
                        default: color = cv::Scalar(255, 255, 255); break; // White
                    }
                    
                    // Draw bounding box
                    cv::rectangle(overlay_frame, scaled_bbox, color, 2);
                    
                    // Draw enhanced label with name, activity, and confidence
                    std::string label_text = "ID:" + std::to_string(det.track_id);
                    if (!det.person_name.empty() && det.person_name != "Unknown") {
                        label_text += " | " + det.person_name;
                    }
                    if (!det.activity.empty()) {
                        label_text += " | " + det.activity;
                    }
                    label_text += " (" + std::to_string((int)(det.confidence * 100)) + "%)";
                    
                    // Calculate label background size
                    int baseline;
                    cv::Size textSize = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    
                    cv::Point2f label_pos(scaled_bbox.x, scaled_bbox.y - 5);
                    if (label_pos.y - textSize.height < 0) {
                        label_pos.y = scaled_bbox.y + scaled_bbox.height + textSize.height + 5;
                    }
                    
                    // Draw label background
                    cv::rectangle(overlay_frame,
                                 cv::Point2f(label_pos.x, label_pos.y - textSize.height - 3),
                                 cv::Point2f(label_pos.x + textSize.width, label_pos.y + 3),
                                 color, -1);
                    
                    // Draw label text
                    cv::putText(overlay_frame, label_text, label_pos,
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    
                    // Draw velocity arrow if moving
                    if (det.velocity.x != 0 || det.velocity.y != 0) {
                        cv::Point2f center(scaled_bbox.x + scaled_bbox.width/2, 
                                         scaled_bbox.y + scaled_bbox.height/2);
                        cv::Point2f vel_end = center + det.velocity * 20;  // Scale velocity for visibility
                        cv::arrowedLine(overlay_frame, center, vel_end, color, 2);
                    }
                }
                
                {
                    std::lock_guard<std::mutex> overlay_lock(pRTMPData->overlay_mutex);
                    pRTMPData->yolo_overlay = overlay_frame.clone();
                }
            }
        }
        
        pRTMPData->cv.notify_one();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }
    
    cout << "🛑 Simple YOLO-Enhanced RTMP processing thread stopped" << endl;
}

void* RTMPProcessingThreadWrapper(void* arg)
{
    RTMPProcessingThread(static_cast<SimpleYOLOData*>(arg));
    return nullptr;
}

int main(int argc, char **argv)
{
    if(argc < 4) {
        cerr << endl << "Usage: ./mono_rtmp_stream_yolo_simple path_to_vocabulary path_to_settings rtmp_url [yolo_config]" << endl;
        cerr << "Example: ./mono_rtmp_stream_yolo_simple ../../../Vocabulary/ORBvoc.txt ../iPhone16Plus.yaml rtmp://192.168.1.100:1935/live/stream" << endl;
        return 1;
    }

    string rtmp_url = string(argv[3]);
    string yolo_config = (argc >= 5) ? string(argv[4]) : "";
    
    cout << "🚀 Simple ORB-SLAM3 + YOLO Real-time System Starting..." << endl;
    cout << "📖 Vocabulary: " << argv[1] << endl;
    cout << "⚙️  Settings: " << argv[2] << endl;
    cout << "📺 RTMP URL: " << rtmp_url << endl;
    cout << "==" << string(58, '=') << endl;

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    
    // Initialize simple YOLO interface
    bool yolo_initialized = false;
    if (!yolo_config.empty()) {
        yolo_initialized = g_yolo_interface.Initialize(yolo_config);
    } else {
        yolo_initialized = g_yolo_interface.Initialize("default_config");
    }

    if (yolo_initialized) {
        cout << "✅ Simple YOLO integration ready!" << endl;
    } else {
        cout << "⚠️ YOLO initialization failed, running SLAM only" << endl;
    }

    // Initialize video capture
    cv::VideoCapture cap;
    cout << "🔗 Connecting to RTMP stream: " << rtmp_url << endl;
    
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    if(!cap.open(rtmp_url)) {
        cerr << "❌ Cannot connect to RTMP stream: " << rtmp_url << endl;
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
    SimpleYOLOData rtmpData;
    rtmpData.pSLAM = &SLAM;
    rtmpData.pCapture = &cap;
    rtmpData.rtmp_url = rtmp_url;
    rtmpData.yolo_enabled = yolo_initialized;
    
    // Adjust image scale if needed
    if (frame_width > 1280) {
        rtmpData.imageScale = 1280.0f / frame_width;
        cout << "📏 Scaling images by factor: " << rtmpData.imageScale << endl;
    }

    // Create and start RTMP processing thread
#ifdef _WIN32
    HANDLE hRTMPThread = CreateThread(NULL, 0, 
        (LPTHREAD_START_ROUTINE)RTMPProcessingThreadWrapper, 
        &rtmpData, 0, NULL);
    if (hRTMPThread == NULL) {
        cerr << "❌ Failed to create RTMP processing thread" << endl;
        return -1;
    }
#else
    pthread_t tRTMP;
    pthread_create(&tRTMP, NULL, RTMPProcessingThreadWrapper, &rtmpData);
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
                
                // Display YOLO overlay if available
                if (rtmpData.yolo_enabled) {
                    cv::Mat overlay_display;
                    {
                        std::lock_guard<std::mutex> overlay_lock(rtmpData.overlay_mutex);
                        if (!rtmpData.yolo_overlay.empty()) {
                            overlay_display = rtmpData.yolo_overlay.clone();
                        }
                    }
                    
                    if (!overlay_display.empty()) {
                        cv::imshow("YOLO + SLAM", overlay_display);
                        cv::waitKey(1);
                    }
                }
                
                // Calculate and display FPS
                if (frameCount % 30 == 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start);
                    if (duration.count() > 0) {
                        double current_fps = frameCount / duration.count();
                        cout << "📈 System FPS: " << std::fixed << std::setprecision(1) << current_fps << endl;
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
    cout << "\\n🛑 Shutting down system..." << endl;
    
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
    
    cout << "✅ System shutdown complete!" << endl;
    cout << "📁 Trajectory saved to: KeyFrameTrajectory.txt" << endl;

    return 0;
}