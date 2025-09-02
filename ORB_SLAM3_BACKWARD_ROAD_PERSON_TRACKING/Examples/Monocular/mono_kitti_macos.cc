/**
* This file is part of ORB-SLAM3 - macOS threading compatible version
* Main thread runs Pangolin viewer, worker thread processes SLAM data
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include<thread>
#include<mutex>
#include<condition_variable>
#include<pthread.h>

#include<opencv2/core/core.hpp>

#include"System.h"

using namespace std;

struct SharedData {
    std::mutex mutex;
    std::condition_variable cv;
    bool finished = false;
    bool dataReady = false;
    int currentFrame = 0;
    int totalFrames = 0;
    cv::Mat currentImage;
    double currentTimestamp = 0.0;
    string currentImagePath = "";
    ORB_SLAM3::System* pSLAM = nullptr;
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    vector<float> vTimesTrack;
    float imageScale = 1.0f;
};

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

// Wrapper function for pthread
void* SLAMProcessingThreadWrapper(void* arg);

// Worker thread that processes SLAM but doesn't create viewer
void SLAMProcessingThread(SharedData* pSharedData)
{
    cv::Mat im;
    
    // Give the viewer time to initialize
    usleep(500000); // 500ms delay
    
    for(int ni = 0; ni < pSharedData->totalFrames; ni++)
    {
        // Check if main thread wants us to stop
        {
            std::lock_guard<std::mutex> lock(pSharedData->mutex);
            if(pSharedData->finished) return;
        }
        
        // Read and prepare image
        im = cv::imread(pSharedData->vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = pSharedData->vTimestamps[ni];
        
        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << pSharedData->vstrImageFilenames[ni] << endl;
            std::lock_guard<std::mutex> lock(pSharedData->mutex);
            pSharedData->finished = true;
            pSharedData->cv.notify_all();
            return;
        }
        
        if(pSharedData->imageScale != 1.f)
        {
            int width = im.cols * pSharedData->imageScale;
            int height = im.rows * pSharedData->imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Process SLAM (this updates the viewer data internally)
        if(pSharedData->pSLAM)
        {
            try {
                pSharedData->pSLAM->TrackMonocular(im, tframe, vector<ORB_SLAM3::IMU::Point>(), pSharedData->vstrImageFilenames[ni]);
                cout << "Successfully processed frame " << ni << endl;
            } catch (const std::exception& e) {
                cerr << "Exception in TrackMonocular: " << e.what() << endl;
                std::lock_guard<std::mutex> lock(pSharedData->mutex);
                pSharedData->finished = true;
                return;
            }
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        
        {
            std::lock_guard<std::mutex> lock(pSharedData->mutex);
            pSharedData->vTimesTrack[ni] = ttrack;
            pSharedData->currentFrame = ni;
        }
        
        // Progress indicator every 50 frames
        if(ni % 50 == 0)
            cout << "Processing frame " << ni << "/" << pSharedData->totalFrames << endl;
        
        // Wait to load the next frame
        double T = 0;
        if(ni < pSharedData->totalFrames - 1)
            T = pSharedData->vTimestamps[ni+1] - tframe;
        else if(ni > 0)
            T = tframe - pSharedData->vTimestamps[ni-1];

        if(ttrack < T)
            usleep((T-ttrack)*1e6);
    }
    
    {
        std::lock_guard<std::mutex> lock(pSharedData->mutex);
        pSharedData->finished = true;
    }
}

// Wrapper function for pthread
void* SLAMProcessingThreadWrapper(void* arg)
{
    SharedData* pSharedData = static_cast<SharedData*>(arg);
    SLAMProcessingThread(pSharedData);
    return nullptr;
}

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_kitti_macos path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    SharedData sharedData;
    
    // Retrieve paths to images
    LoadImages(string(argv[3]), sharedData.vstrImageFilenames, sharedData.vTimestamps);
    
    sharedData.totalFrames = sharedData.vstrImageFilenames.size();
    sharedData.vTimesTrack.resize(sharedData.totalFrames);
    
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << sharedData.totalFrames << endl << endl;

    // Create SLAM system with viewer enabled - viewer will run on main thread
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    sharedData.pSLAM = &SLAM;
    sharedData.imageScale = SLAM.GetImageScale();
    
    // Start SLAM processing thread with larger stack size using pthread
    pthread_t slamThreadHandle;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    
    // Set stack size to 16MB (making it even larger)
    size_t stackSize = 16 * 1024 * 1024; // 16MB
    int stackResult = pthread_attr_setstacksize(&attr, stackSize);
    if (stackResult != 0) {
        cerr << "Failed to set stack size: " << stackResult << endl;
    }
    
    // Verify the stack size was set
    size_t verifyStackSize;
    pthread_attr_getstacksize(&attr, &verifyStackSize);
    cout << "Creating SLAM processing thread with " << verifyStackSize / (1024*1024) << "MB stack..." << endl;
    
    int result = pthread_create(&slamThreadHandle, &attr, SLAMProcessingThreadWrapper, &sharedData);
    if (result != 0) {
        cerr << "Failed to create SLAM thread: " << result << endl;
        return -1;
    }
    pthread_attr_destroy(&attr);
    
#ifdef __APPLE__
    // Main thread: Run viewer on main thread for macOS compatibility
    SLAM.RunViewerOnMainThread();
    
    // Wait for processing thread to complete
    pthread_join(slamThreadHandle, nullptr);
#else
    // Non-macOS: Wait for processing thread
    pthread_join(slamThreadHandle, nullptr);
#endif
    
    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(sharedData.vTimesTrack.begin(), sharedData.vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < sharedData.totalFrames; ni++)
    {
        totaltime += sharedData.vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << sharedData.vTimesTrack[sharedData.totalFrames/2] << endl;
    cout << "mean tracking time: " << totaltime/sharedData.totalFrames << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}