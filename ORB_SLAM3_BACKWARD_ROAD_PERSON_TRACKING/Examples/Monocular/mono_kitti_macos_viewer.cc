/**
* This file is part of ORB-SLAM3 - macOS threading compatible version with viewer
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include<thread>
#include<mutex>
#include<condition_variable>

#include<opencv2/core/core.hpp>

#include"System.h"

using namespace std;

struct SharedData {
    std::mutex mutex;
    std::condition_variable cv;
    bool finished = false;
    bool processing = false;
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

// Worker thread that processes images but doesn't handle GUI
void ProcessingWorkerThread(SharedData* pSharedData)
{
    cv::Mat im;
    
    for(int ni = 0; ni < pSharedData->totalFrames; ni++)
    {
        {
            std::unique_lock<std::mutex> lock(pSharedData->mutex);
            pSharedData->cv.wait(lock, [pSharedData] { return pSharedData->processing || pSharedData->finished; });
            
            if(pSharedData->finished) break;
            
            pSharedData->currentFrame = ni;
        }
        
        // Read and process image
        im = cv::imread(pSharedData->vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = pSharedData->vTimestamps[ni];
        
        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << pSharedData->vstrImageFilenames[ni] << endl;
            {
                std::lock_guard<std::mutex> lock(pSharedData->mutex);
                pSharedData->finished = true;
            }
            break;
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

        // Pass the image to the SLAM system (this will update the viewer)
        pSharedData->pSLAM->TrackMonocular(im, tframe, vector<ORB_SLAM3::IMU::Point>(), pSharedData->vstrImageFilenames[ni]);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        
        {
            std::lock_guard<std::mutex> lock(pSharedData->mutex);
            pSharedData->vTimesTrack[ni] = ttrack;
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

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_kitti_macos_viewer path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Shared data structure
    SharedData sharedData;
    
    // Retrieve paths to images
    LoadImages(string(argv[3]), sharedData.vstrImageFilenames, sharedData.vTimestamps);
    
    sharedData.totalFrames = sharedData.vstrImageFilenames.size();
    sharedData.vTimesTrack.resize(sharedData.totalFrames);
    
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << sharedData.totalFrames << endl << endl;

    // Create SLAM system with viewer enabled - this must be done on main thread
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    sharedData.pSLAM = &SLAM;
    sharedData.imageScale = SLAM.GetImageScale();
    
    // Start processing thread
    sharedData.processing = true;
    std::thread processingThread(ProcessingWorkerThread, &sharedData);
    
    // Main thread: Handle viewer events and GUI
    // The viewer runs its own event loop which must be on the main thread for macOS
    while(!sharedData.finished)
    {
        // Small sleep to prevent busy waiting
        usleep(1000); // 1ms
        
        // Check if we need to handle any viewer updates
        // The SLAM system internally manages viewer updates
    }
    
    // Wait for processing to complete
    processingThread.join();
    
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