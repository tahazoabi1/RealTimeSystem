/**
 * YOLOInterface.cc
 * Implementation of YOLO-ORB-SLAM3 bridge
 */

#include "YOLOInterface.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cstring>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#define popen _popen
#define pclose _pclose
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ORB_SLAM3
{

// Shared memory structure for frame exchange
struct YOLOInterface::SharedMemory {
    // Frame buffer
    void* frame_buffer;
    size_t frame_buffer_size;
    int frame_shm_fd;
    
    // Result buffer  
    void* result_buffer;
    size_t result_buffer_size;
    int result_shm_fd;
    
    // Metadata
    struct FrameMetadata {
        int width;
        int height;
        int channels;
        double timestamp;
    } frame_meta;
    
    struct ResultMetadata {
        int num_detections;
        float processing_time;
    } result_meta;
};

YOLOInterface::YOLOInterface() 
    : initialized_(false)
    , enabled_(true)
    , processing_(false)
    , last_processing_time_(0.0f)
    , python_process_(nullptr)
    , frame_width_(1280)
    , frame_height_(720)
{
    shared_mem_ = std::make_unique<SharedMemory>();
}

YOLOInterface::~YOLOInterface() {
    Shutdown();
}

bool YOLOInterface::Initialize(const std::string& config_path) {
    if (initialized_) {
        return true;
    }
    
    config_path_ = config_path.empty() ? "D:\\Learning\\realtimesystem\\optimized_config.json" : config_path;
    
    std::cout << "🚀 Initializing YOLO Interface..." << std::endl;
    
    // Setup shared memory
    const size_t max_frame_size = 3840 * 2160 * 3;  // Max 4K RGB
    const size_t max_result_size = 100 * 256;  // Max 100 detections
    
#ifdef _WIN32
    // Windows shared memory using memory-mapped files
    HANDLE hMapFile;
    
    // Create frame shared memory
    hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        max_frame_size + sizeof(SharedMemory::FrameMetadata),
        TEXT("ORB_SLAM3_YOLO_BRIDGE_FRAME"));
    
    if (hMapFile == NULL) {
        std::cerr << "❌ Could not create frame shared memory" << std::endl;
        return false;
    }
    
    shared_mem_->frame_buffer = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0, 0,
        max_frame_size + sizeof(SharedMemory::FrameMetadata));
    
    if (shared_mem_->frame_buffer == NULL) {
        CloseHandle(hMapFile);
        std::cerr << "❌ Could not map frame shared memory" << std::endl;
        return false;
    }
    
    // Create result shared memory
    hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        max_result_size + sizeof(SharedMemory::ResultMetadata),
        TEXT("ORB_SLAM3_YOLO_BRIDGE_RESULT"));
    
    if (hMapFile == NULL) {
        std::cerr << "❌ Could not create result shared memory" << std::endl;
        return false;
    }
    
    shared_mem_->result_buffer = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0, 0,
        max_result_size + sizeof(SharedMemory::ResultMetadata));
    
    if (shared_mem_->result_buffer == NULL) {
        CloseHandle(hMapFile);
        std::cerr << "❌ Could not map result shared memory" << std::endl;
        return false;
    }
#else
    // Linux/macOS shared memory using POSIX
    // Frame shared memory
    shared_mem_->frame_shm_fd = shm_open("/ORB_SLAM3_YOLO_BRIDGE_FRAME", 
                                         O_CREAT | O_RDWR, 0666);
    if (shared_mem_->frame_shm_fd == -1) {
        std::cerr << "❌ Could not create frame shared memory" << std::endl;
        return false;
    }
    
    ftruncate(shared_mem_->frame_shm_fd, max_frame_size + sizeof(SharedMemory::FrameMetadata));
    
    shared_mem_->frame_buffer = mmap(0, max_frame_size + sizeof(SharedMemory::FrameMetadata),
                                     PROT_READ | PROT_WRITE, MAP_SHARED,
                                     shared_mem_->frame_shm_fd, 0);
    
    // Result shared memory
    shared_mem_->result_shm_fd = shm_open("/ORB_SLAM3_YOLO_BRIDGE_RESULT",
                                          O_CREAT | O_RDWR, 0666);
    if (shared_mem_->result_shm_fd == -1) {
        std::cerr << "❌ Could not create result shared memory" << std::endl;
        return false;
    }
    
    ftruncate(shared_mem_->result_shm_fd, max_result_size + sizeof(SharedMemory::ResultMetadata));
    
    shared_mem_->result_buffer = mmap(0, max_result_size + sizeof(SharedMemory::ResultMetadata),
                                      PROT_READ | PROT_WRITE, MAP_SHARED,
                                      shared_mem_->result_shm_fd, 0);
#endif
    
    shared_mem_->frame_buffer_size = max_frame_size;
    shared_mem_->result_buffer_size = max_result_size;
    
    // Launch Python tracker process
    LaunchPythonTracker();
    
    initialized_ = true;
    std::cout << "✅ YOLO Interface initialized successfully" << std::endl;
    
    return true;
}

void YOLOInterface::LaunchPythonTracker() {
    std::cout << "🐍 Launching Python YOLO tracker..." << std::endl;
    
    python_thread_ = std::thread([this]() {
        std::string command = "python \"D:\\Learning\\realtimesystem\\orbslam_bridge.py\" --config \"" 
                            + config_path_ + "\"";
        
#ifdef _WIN32
        // Windows: Launch Python process
        STARTUPINFO si;
        PROCESS_INFORMATION pi;
        
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        ZeroMemory(&pi, sizeof(pi));
        
        // Create new console window for Python output
        if (CreateProcess(NULL,
            const_cast<char*>(command.c_str()),
            NULL, NULL, FALSE,
            CREATE_NEW_CONSOLE,
            NULL, NULL,
            &si, &pi))
        {
            python_process_ = pi.hProcess;
            std::cout << "✅ Python tracker launched (PID: " << pi.dwProcessId << ")" << std::endl;
        }
        else
        {
            std::cerr << "❌ Failed to launch Python tracker" << std::endl;
        }
#else
        // Linux/macOS: Use system call
        python_process_ = popen(command.c_str(), "r");
        if (python_process_) {
            std::cout << "✅ Python tracker launched" << std::endl;
        } else {
            std::cerr << "❌ Failed to launch Python tracker" << std::endl;
        }
#endif
    });
}

bool YOLOInterface::ProcessFrame(const cv::Mat& frame, double timestamp) {
    if (!initialized_ || !enabled_ || processing_) {
        return false;
    }
    
    processing_ = true;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Send frame to Python
    if (!SendFrameToYOLO(frame, timestamp)) {
        processing_ = false;
        return false;
    }
    
    // Wait for results (with timeout)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Receive detections
    if (!ReceiveDetectionsFromYOLO()) {
        processing_ = false;
        return false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_processing_time_ = std::chrono::duration<float>(end_time - start_time).count();
    
    processing_ = false;
    return true;
}

bool YOLOInterface::SendFrameToYOLO(const cv::Mat& frame, double timestamp) {
    if (!shared_mem_->frame_buffer) {
        return false;
    }
    
    try {
        // Prepare frame (ensure RGB format)
        cv::Mat rgb_frame;
        if (frame.channels() == 1) {
            cv::cvtColor(frame, rgb_frame, cv::COLOR_GRAY2RGB);
        } else if (frame.channels() == 4) {
            cv::cvtColor(frame, rgb_frame, cv::COLOR_BGRA2RGB);
        } else if (frame.channels() == 3) {
            cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        } else {
            rgb_frame = frame;
        }
        
        // Write metadata
        shared_mem_->frame_meta.width = rgb_frame.cols;
        shared_mem_->frame_meta.height = rgb_frame.rows;
        shared_mem_->frame_meta.channels = rgb_frame.channels();
        shared_mem_->frame_meta.timestamp = timestamp;
        
        memcpy(shared_mem_->frame_buffer, &shared_mem_->frame_meta, sizeof(SharedMemory::FrameMetadata));
        
        // Write frame data
        size_t frame_size = rgb_frame.total() * rgb_frame.elemSize();
        if (frame_size <= shared_mem_->frame_buffer_size) {
            memcpy((char*)shared_mem_->frame_buffer + sizeof(SharedMemory::FrameMetadata),
                   rgb_frame.data, frame_size);
            return true;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error sending frame to YOLO: " << e.what() << std::endl;
    }
    
    return false;
}

bool YOLOInterface::ReceiveDetectionsFromYOLO() {
    if (!shared_mem_->result_buffer) {
        return false;
    }
    
    try {
        // Read metadata
        memcpy(&shared_mem_->result_meta, shared_mem_->result_buffer, sizeof(SharedMemory::ResultMetadata));
        
        int num_detections = shared_mem_->result_meta.num_detections;
        
        // Clear current detections
        std::lock_guard<std::mutex> lock(detection_mutex_);
        current_detections_.clear();
        
        // Read each detection
        char* buffer_ptr = (char*)shared_mem_->result_buffer + sizeof(SharedMemory::ResultMetadata);
        
        for (int i = 0; i < num_detections && i < 100; ++i) {
            YOLODetection det;
            
            // Read basic detection data
            int track_id;
            float x, y, w, h, conf;
            memcpy(&track_id, buffer_ptr, sizeof(int)); buffer_ptr += sizeof(int);
            memcpy(&x, buffer_ptr, sizeof(float)); buffer_ptr += sizeof(float);
            memcpy(&y, buffer_ptr, sizeof(float)); buffer_ptr += sizeof(float);
            memcpy(&w, buffer_ptr, sizeof(float)); buffer_ptr += sizeof(float);
            memcpy(&h, buffer_ptr, sizeof(float)); buffer_ptr += sizeof(float);
            memcpy(&conf, buffer_ptr, sizeof(float)); buffer_ptr += sizeof(float);
            
            det.track_id = track_id;
            det.bbox = cv::Rect2f(x, y, w, h);
            det.confidence = conf;
            
            // Read string data (label, name, activity)
            uint8_t str_len;
            char str_buffer[33];
            
            // Label
            memcpy(&str_len, buffer_ptr, 1); buffer_ptr += 1;
            memcpy(str_buffer, buffer_ptr, str_len); buffer_ptr += 32;
            str_buffer[str_len] = '\0';
            det.label = std::string(str_buffer);
            
            // Person name
            memcpy(&str_len, buffer_ptr, 1); buffer_ptr += 1;
            memcpy(str_buffer, buffer_ptr, str_len); buffer_ptr += 32;
            str_buffer[str_len] = '\0';
            det.person_name = std::string(str_buffer);
            
            // Activity
            memcpy(&str_len, buffer_ptr, 1); buffer_ptr += 1;
            memcpy(str_buffer, buffer_ptr, str_len); buffer_ptr += 32;
            str_buffer[str_len] = '\0';
            det.activity = std::string(str_buffer);
            
            current_detections_.push_back(det);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error receiving detections from YOLO: " << e.what() << std::endl;
    }
    
    return false;
}

std::vector<YOLODetection> YOLOInterface::GetDetections() {
    std::lock_guard<std::mutex> lock(detection_mutex_);
    return current_detections_;
}

std::vector<Zone> YOLOInterface::GetZones() {
    std::lock_guard<std::mutex> lock(detection_mutex_);
    return current_zones_;
}

void YOLOInterface::StopPythonTracker() {
    if (python_process_) {
#ifdef _WIN32
        TerminateProcess((HANDLE)python_process_, 0);
        CloseHandle((HANDLE)python_process_);
#else
        pclose((FILE*)python_process_);
#endif
        python_process_ = nullptr;
    }
    
    if (python_thread_.joinable()) {
        python_thread_.join();
    }
}

void YOLOInterface::Shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "🛑 Shutting down YOLO Interface..." << std::endl;
    
    enabled_ = false;
    
    // Stop Python process
    StopPythonTracker();
    
    // Cleanup shared memory
#ifdef _WIN32
    if (shared_mem_->frame_buffer) {
        UnmapViewOfFile(shared_mem_->frame_buffer);
    }
    if (shared_mem_->result_buffer) {
        UnmapViewOfFile(shared_mem_->result_buffer);
    }
#else
    if (shared_mem_->frame_buffer) {
        munmap(shared_mem_->frame_buffer, shared_mem_->frame_buffer_size + sizeof(SharedMemory::FrameMetadata));
        close(shared_mem_->frame_shm_fd);
        shm_unlink("/ORB_SLAM3_YOLO_BRIDGE_FRAME");
    }
    if (shared_mem_->result_buffer) {
        munmap(shared_mem_->result_buffer, shared_mem_->result_buffer_size + sizeof(SharedMemory::ResultMetadata));
        close(shared_mem_->result_shm_fd);
        shm_unlink("/ORB_SLAM3_YOLO_BRIDGE_RESULT");
    }
#endif
    
    initialized_ = false;
    std::cout << "✅ YOLO Interface shutdown complete" << std::endl;
}

} // namespace ORB_SLAM3