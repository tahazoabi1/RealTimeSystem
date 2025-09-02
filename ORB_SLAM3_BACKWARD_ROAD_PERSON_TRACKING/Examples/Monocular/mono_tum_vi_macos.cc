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
#include <unistd.h>

#include<opencv2/core/core.hpp>

#include"System.h"
#include "Converter.h"
#include "PathMatcher.h"

using namespace std;

struct SharedData {
    std::mutex mutex;
    std::condition_variable cv;
    bool finished = false;
    bool dataReady = false;
    int currentSequence = 0;
    int currentFrame = 0;
    int totalFrames = 0;
    cv::Mat currentImage;
    double currentTimestamp = 0.0;
    string currentImagePath = "";
    ORB_SLAM3::System* pSLAM = nullptr;
    vector< vector<string> > vstrImageFilenames;
    vector< vector<double> > vTimestampsCam;
    vector<int> nImages;
    vector<float> vTimesTrack;
    float imageScale = 1.0f;
    string trajectoryFileName = "";
};

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

void TestPathMatcher();

// Wrapper function for pthread
void* SLAMProcessingThreadWrapper(void* arg);

// Worker thread that processes SLAM but doesn't create viewer
void SLAMProcessingThread(SharedData* pSharedData)
{
    cv::Mat im;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    
    // Give the viewer time to initialize
    usleep(500000); // 500ms delay
    
    int proccIm = 0;
    
    for (int seq = 0; seq < pSharedData->nImages.size(); seq++)
    {
        for(int ni = 0; ni < pSharedData->nImages[seq]; ni++, proccIm++)
        {
            // Check if main thread wants us to stop
            {
                std::lock_guard<std::mutex> lock(pSharedData->mutex);
                if(pSharedData->finished) return;
                pSharedData->currentSequence = seq;
                pSharedData->currentFrame = ni;
            }
            
            // Read and prepare image
            im = cv::imread(pSharedData->vstrImageFilenames[seq][ni], cv::IMREAD_GRAYSCALE);
            double tframe = pSharedData->vTimestampsCam[seq][ni];
            
            if(im.empty())
            {
                cerr << endl << "Failed to load image at: " << pSharedData->vstrImageFilenames[seq][ni] << endl;
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
            
            // Apply CLAHE
            clahe->apply(im, im);

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

            
            // Process SLAM (this updates the viewer data internally)
            if(pSharedData->pSLAM)
            {
                try {
                    pSharedData->pSLAM->TrackMonocular(im, tframe, vector<ORB_SLAM3::IMU::Point>(), pSharedData->vstrImageFilenames[seq][ni]);
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
                pSharedData->vTimesTrack[proccIm] = ttrack;
            }
            
            // Progress indicator every 1000 frames
            if(ni % 1000 == 0) {
                cout << "Processing frame " << ni << "/" << pSharedData->nImages[seq] << " from sequence " << seq << endl;
            }
            
            // Wait to load the next frame
            double T = 0;
            if(ni < pSharedData->nImages[seq] - 1)
                T = pSharedData->vTimestampsCam[seq][ni+1] - tframe;
            else if(ni > 0)
                T = tframe - pSharedData->vTimestampsCam[seq][ni-1];

            if(ttrack < T)
                usleep((T-ttrack)*1e6);
        }
        
        if(seq < pSharedData->nImages.size() - 1)
        {
            cout << "Changing the dataset" << endl;
            if(pSharedData->pSLAM) {
                pSharedData->pSLAM->ChangeDataset();
            }
        }
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

double ttrack_tot = 0;
int main(int argc, char **argv)
{
    const int num_seq = (argc-3)/2;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName = (((argc-3) % 2) == 1);

    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc-1]);
        cout << "file name: " << file_name << endl;
    }

    if(argc < 4)
    {
        cerr << endl << "Usage: ./mono_tum_vi_macos path_to_vocabulary path_to_settings path_to_image_folder_1 path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... path_to_image_folder_N path_to_times_file_N) (trajectory_file_name)" << endl;
        return 1;
    }

    SharedData sharedData;
    sharedData.trajectoryFileName = file_name;
    
    // Load all sequences
    sharedData.vstrImageFilenames.resize(num_seq);
    sharedData.vTimestampsCam.resize(num_seq);
    sharedData.nImages.resize(num_seq);

    int tot_images = 0;
    for (int seq = 0; seq < num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        LoadImages(string(argv[(2*seq)+3]), string(argv[(2*seq)+4]), sharedData.vstrImageFilenames[seq], sharedData.vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        sharedData.nImages[seq] = sharedData.vstrImageFilenames[seq].size();
        tot_images += sharedData.nImages[seq];

        if((sharedData.nImages[seq]<=0))
        {
            cerr << "ERROR: Failed to load images for sequence" << seq << endl;
            return 1;
        }
    }
    
    sharedData.totalFrames = tot_images;
    sharedData.vTimesTrack.resize(tot_images);
    
    cout << endl << "-------" << endl;
    cout.precision(17);
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << tot_images << endl << endl;

    // Create SLAM system with viewer enabled - viewer will run on main thread
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true, 0, file_name);
    sharedData.pSLAM = &SLAM;
    sharedData.imageScale = SLAM.GetImageScale();
    
    cout << "=== PATH RECORDING & GUIDANCE READY ===" << endl;
    cout << "RECORDING CONTROLS:" << endl;
    cout << "  Press 'R' to start recording a path" << endl;
    cout << "  Press 'S' to stop recording" << endl;
    cout << "GUIDANCE CONTROLS:" << endl;
    cout << "  Press 'L' to load recorded path" << endl;
    cout << "  Press 'G' to start guidance" << endl;
    cout << "  Press 'H' to stop guidance" << endl;
    cout << "Status shown in Pangolin viewer panel" << endl;
    cout << "========================================" << endl;
    
    // Test PathMatcher with existing recorded path
    TestPathMatcher();
    
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

    // Save camera trajectory
    if (bFileName)
    {
        const string kf_file = "kf_" + string(argv[argc-1]) + ".txt";
        const string f_file = "f_" + string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    // Tracking time statistics
    sort(sharedData.vTimesTrack.begin(), sharedData.vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < tot_images; ni++)
    {
        totaltime += sharedData.vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << sharedData.vTimesTrack[tot_images/2] << endl;
    cout << "mean tracking time: " << totaltime/tot_images << endl;

    return 0;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);

        if(!s.empty())
        {
            if (s[0] == '#')
                continue;

            int pos = s.find(' ');
            string item = s.substr(0, pos);

            vstrImages.push_back(strImagePath + "/" + item + ".png");
            double t = stod(item);
            vTimeStamps.push_back(t/1e9);
        }
    }
}

void TestPathMatcher()
{
    cout << "\n=== Testing PathMatcher ===" << endl;
    
    ORB_SLAM3::PathMatcher matcher;
    
    // Test if we can load the previously recorded path
    string path_file = "recorded_path_1756539848.txt";
    if (matcher.LoadPath(path_file)) {
        cout << "✅ PathMatcher loaded " << matcher.GetPathLength() << " path points" << endl;
        
        // Test nearest neighbor search
        Sophus::SE3f test_pose;
        test_pose.translation() = Eigen::Vector3f(-0.3f, -0.1f, -0.2f);
        test_pose.setRotationMatrix(Eigen::Matrix3f::Identity());
        
        ORB_SLAM3::MatchResult result = matcher.FindNearestPoint(test_pose);
        cout << "✅ Nearest neighbor search: distance = " << result.distance << "m" << endl;
        
        matcher.UpdateProgress(result);
        cout << "✅ Progress tracking: " << (matcher.GetPathProgress() * 100.0f) << "%" << endl;
        
        cout << "PathMatcher test completed successfully!" << endl;
    } else {
        cout << "⚠️  PathMatcher test skipped - no recorded path file found" << endl;
        cout << "   (Run with 'R' key to record a path first)" << endl;
    }
    
    cout << "=========================" << endl;
}