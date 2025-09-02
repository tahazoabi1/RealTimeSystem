/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "Viewer.h"
#ifdef HAVE_PANGOLIN
#include <pangolin/pangolin.h>
#endif

#include <mutex>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>
#ifdef _WIN32
#include <io.h>
#include <direct.h>
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

namespace ORB_SLAM3
{

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath, Settings* settings):
    both(false), mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
{
    if(settings){
        newParameterLoader(settings);
    }
    else{

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    mbStopTrack = false;
}

void Viewer::newParameterLoader(Settings *settings) {
    mImageViewerScale = 1.f;

    float fps = settings->fps();
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::Size imSize = settings->newImSize();
    mImageHeight = imSize.height;
    mImageWidth = imSize.width;

    mImageViewerScale = settings->imageViewerScale();
    mViewpointX = settings->viewPointX();
    mViewpointY = settings->viewPointY();
    mViewpointZ = settings->viewPointZ();
    mViewpointF = settings->viewPointF();
}

bool Viewer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    mImageViewerScale = 1.f;

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::FileNode node = fSettings["Camera.width"];
    if(!node.empty())
    {
        mImageWidth = node.real();
    }
    else
    {
        std::cerr << "*Camera.width parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Camera.height"];
    if(!node.empty())
    {
        mImageHeight = node.real();
    }
    else
    {
        std::cerr << "*Camera.height parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.imageViewScale"];
    if(!node.empty())
    {
        mImageViewerScale = node.real();
    }

    node = fSettings["Viewer.ViewpointX"];
    if(!node.empty())
    {
        mViewpointX = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointX parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointY"];
    if(!node.empty())
    {
        mViewpointY = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointY parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointZ"];
    if(!node.empty())
    {
        mViewpointZ = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointZ parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointF"];
    if(!node.empty())
    {
        mViewpointF = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointF parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

#ifdef HAVE_PANGOLIN
    pangolin::CreateWindowAndBind("ORB-SLAM3: Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",false,true);
    pangolin::Var<bool> menuCamView("menu.Camera View",false,false);
    pangolin::Var<bool> menuTopView("menu.Top View",false,false);
    // pangolin::Var<bool> menuSideView("menu.Side View",false,false);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",false,true);
    pangolin::Var<bool> menuShowInertialGraph("menu.Show Inertial Graph",true,true);
    pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);
    pangolin::Var<bool> menuStop("menu.Stop",false,false);
    pangolin::Var<bool> menuStepByStep("menu.Step By Step",false,true);  // false, true
    pangolin::Var<bool> menuStep("menu.Step",false,false);

    pangolin::Var<bool> menuShowOptLba("menu.Show LBA opt", false, true);
    
    // Path recording controls
    pangolin::Var<bool> menuStartRecording("menu.Start Recording (R)",false,false);
    pangolin::Var<bool> menuStopRecording("menu.Stop Recording (S)",false,false);
    pangolin::Var<std::string> menuRecordingStatus("menu.Recording Status","STOPPED",false);
    
    // Path guidance controls
    pangolin::Var<bool> menuLoadPath("menu.Load Path (L)",false,false);
    pangolin::Var<bool> menuStartGuidance("menu.Start Guidance (G)",false,false);
    pangolin::Var<bool> menuStopGuidance("menu.Stop Guidance (H)",false,false);
    pangolin::Var<bool> menuToggleBackwards("menu.Backwards Mode (B)",false,false);
    pangolin::Var<std::string> menuGuidanceStatus("menu.Guidance Status","IDLE",false);
    
    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc, Twr;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();
    cv::namedWindow("ORB-SLAM3: Current Frame");

    bool bFollow = true;
    bool bLocalizationMode = false;
    bool bStepByStep = false;
    bool bCameraView = true;

    if(mpTracker->mSensor == mpSystem->MONOCULAR || mpTracker->mSensor == mpSystem->STEREO || mpTracker->mSensor == mpSystem->RGBD)
    {
        menuShowGraph = true;
    }

    float trackedImageScale = mpTracker->GetImageScale();

    cout << "Starting the Viewer" << endl;
    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc,Ow);

        if(mbStopTrack)
        {
            menuStepByStep = true;
            mbStopTrack = false;
        }

        if(menuFollowCamera && bFollow)
        {
            if(bCameraView)
                s_cam.Follow(Twc);
            else
                s_cam.Follow(Ow);
        }
        else if(menuFollowCamera && !bFollow)
        {
            if(bCameraView)
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
            }
            else
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,10, 0,0,0,0.0,0.0, 1.0));
                s_cam.Follow(Ow);
            }
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        if(menuCamView)
        {
            menuCamView = false;
            bCameraView = true;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
        }

        if(menuTopView && mpMapDrawer->mpAtlas->isImuInitialized())
        {
            menuTopView = false;
            bCameraView = false;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,50, 0,0,0,0.0,0.0, 1.0));
            s_cam.Follow(Ow);
        }

        if(menuLocalizationMode && !bLocalizationMode)
        {
            mpSystem->ActivateLocalizationMode();
            bLocalizationMode = true;
        }
        else if(!menuLocalizationMode && bLocalizationMode)
        {
            mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
        }

        if(menuStepByStep && !bStepByStep)
        {
            //cout << "Viewer: step by step" << endl;
            mpTracker->SetStepByStep(true);
            bStepByStep = true;
        }
        else if(!menuStepByStep && bStepByStep)
        {
            mpTracker->SetStepByStep(false);
            bStepByStep = false;
        }

        if(menuStep)
        {
            mpTracker->mbStep = true;
            menuStep = false;
        }
        
        // Handle path recording controls
        if(menuStartRecording)
        {
            if(!mpSystem->IsPathRecording())
            {
                std::string filename = "recorded_path_" + std::to_string(time(nullptr)) + ".txt";
                mpSystem->StartPathRecording(filename);
                menuRecordingStatus = "RECORDING to " + filename;
                std::cout << "=== PATH RECORDING STARTED: " << filename << " ===" << std::endl;
            }
            menuStartRecording = false;
        }
        
        if(menuStopRecording)
        {
            if(mpSystem->IsPathRecording())
            {
                mpSystem->StopPathRecording();
                menuRecordingStatus = "STOPPED";
                std::cout << "=== PATH RECORDING STOPPED ===" << std::endl;
            }
            menuStopRecording = false;
        }
        
        // Update recording status display
        if(mpSystem->IsPathRecording())
        {
            size_t count = mpSystem->GetRecordedPointsCount();
            double duration = mpSystem->GetPathRecordingDuration();
            std::ostringstream oss;
            oss << "RECORDING (" << count << " poses, " << std::fixed << std::setprecision(1) << duration << "s)";
            menuRecordingStatus = oss.str();
        }
        
        // Handle path guidance controls
        if(menuLoadPath)
        {
            // Find and load the most recent recorded path
            std::string path_file = "";
            time_t latest_time = 0;
            
            // Search for recorded_path_*.txt files in current directory
#ifdef _WIN32
            // Windows implementation using FindFirstFile/FindNextFile
            WIN32_FIND_DATA findData;
            HANDLE hFind = FindFirstFile(".\\*.txt", &findData);
            if (hFind != INVALID_HANDLE_VALUE) {
                do {
                    std::string filename = findData.cFileName;
                    if ((filename.length() > 18 && filename.substr(0, 14) == "recorded_path_" && filename.substr(filename.length() - 4) == ".txt") ||
                        (filename.length() > 15 && filename.substr(0, 10) == "live_path_" && filename.substr(filename.length() - 4) == ".txt")) {
                        // Extract timestamp from filename
                        std::string timestamp_str;
                        if (filename.substr(0, 14) == "recorded_path_") {
                            timestamp_str = filename.substr(14, filename.length() - 18);
                        } else {
                            timestamp_str = filename.substr(10, filename.length() - 14);
                        }
                        try {
                            time_t file_timestamp = std::stoll(timestamp_str);
                            if (file_timestamp > latest_time) {
                                latest_time = file_timestamp;
                                path_file = filename;
                            }
                        } catch (...) {
                            // Skip files with invalid timestamp format
                        }
                    }
                } while (FindNextFile(hFind, &findData));
                FindClose(hFind);
            }
#else
            // POSIX implementation for Linux/macOS
            DIR* dir = opendir(".");
            if (dir != nullptr) {
                struct dirent* entry;
                while ((entry = readdir(dir)) != nullptr) {
                    std::string filename = entry->d_name;
                    if ((filename.length() > 18 && filename.substr(0, 14) == "recorded_path_" && filename.substr(filename.length() - 4) == ".txt") ||
                        (filename.length() > 15 && filename.substr(0, 10) == "live_path_" && filename.substr(filename.length() - 4) == ".txt")) {
                        // Extract timestamp from filename
                        std::string timestamp_str;
                        if (filename.substr(0, 14) == "recorded_path_") {
                            timestamp_str = filename.substr(14, filename.length() - 18);
                        } else {
                            timestamp_str = filename.substr(10, filename.length() - 14);
                        }
                        try {
                            time_t file_timestamp = std::stoll(timestamp_str);
                            if (file_timestamp > latest_time) {
                                latest_time = file_timestamp;
                                path_file = filename;
                            }
                        } catch (...) {
                            // Skip files with invalid timestamp format
                        }
                    }
                }
                closedir(dir);
            }
#endif
            
            if (!path_file.empty() && mpSystem->LoadPathForGuidance(path_file))
            {
                menuGuidanceStatus = "PATH LOADED - Ready for guidance";
                std::cout << "=== PATH LOADED FOR GUIDANCE: " << path_file << " ===" << std::endl;
            }
            else
            {
                if (path_file.empty()) {
                    menuGuidanceStatus = "NO PATHS FOUND - Record path first";
                    std::cout << "=== NO RECORDED PATHS FOUND ====" << std::endl;
                } else {
                    menuGuidanceStatus = "LOAD FAILED - Check path file";
                    std::cout << "=== FAILED TO LOAD PATH: " << path_file << " ===" << std::endl;
                }
            }
            menuLoadPath = false;
        }
        
        if(menuStartGuidance)
        {
            if(mpSystem->StartPathGuidance())
            {
                menuGuidanceStatus = "GUIDING - Follow audio instructions";
                std::cout << "=== PATH GUIDANCE STARTED ===" << std::endl;
            }
            else
            {
                menuGuidanceStatus = "START FAILED - Load path first";
                std::cout << "=== GUIDANCE START FAILED - Load path first ===" << std::endl;
            }
            menuStartGuidance = false;
        }
        
        if(menuStopGuidance)
        {
            mpSystem->StopPathGuidance();
            menuGuidanceStatus = "STOPPED";
            std::cout << "=== PATH GUIDANCE STOPPED ===" << std::endl;
            menuStopGuidance = false;
        }
        
        if(menuToggleBackwards)
        {
            bool current_backwards = mpSystem->IsBackwardsNavigationMode();
            mpSystem->SetBackwardsNavigationMode(!current_backwards);
            std::cout << "=== BACKWARDS NAVIGATION " << (current_backwards ? "DISABLED" : "ENABLED") << " ===" << std::endl;
            menuToggleBackwards = false;
        }
        
        // Update guidance status display
        if(mpSystem->IsGuidanceActive())
        {
            float progress = mpSystem->GetGuidanceProgress();
            std::ostringstream oss;
            oss << "GUIDING (" << std::fixed << std::setprecision(1) << (progress * 100.0f) << "%)";
            menuGuidanceStatus = oss.str();
        }


        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        mpMapDrawer->DrawCurrentCamera(Twc);
        if(menuShowKeyFrames || menuShowGraph || menuShowInertialGraph || menuShowOptLba)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames,menuShowGraph, menuShowInertialGraph, menuShowOptLba);
        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints();
            
        // Draw stopped person landmarks with special markers
        mpMapDrawer->DrawStoppedPersonLandmarks();

        pangolin::FinishFrame();

        cv::Mat toShow;
        cv::Mat im = mpFrameDrawer->DrawFrame(trackedImageScale);

        if(both){
            cv::Mat imRight = mpFrameDrawer->DrawRightFrame(trackedImageScale);
            cv::hconcat(im,imRight,toShow);
        }
        else{
            toShow = im;
        }

        if(mImageViewerScale != 1.f)
        {
            int width = toShow.cols * mImageViewerScale;
            int height = toShow.rows * mImageViewerScale;
            cv::resize(toShow, toShow, cv::Size(width, height));
        }

        cv::imshow("ORB-SLAM3: Current Frame",toShow);
        int key = cv::waitKey(mT);
        
        // Handle keyboard input for path recording
        if(key == 'r' || key == 'R')
        {
            menuStartRecording = true;
        }
        else if(key == 's' || key == 'S')
        {
            menuStopRecording = true;
        }
        // Handle keyboard input for path guidance
        else if(key == 'l' || key == 'L')
        {
            menuLoadPath = true;
        }
        else if(key == 'g' || key == 'G')
        {
            menuStartGuidance = true;
        }
        else if(key == 'h' || key == 'H')
        {
            menuStopGuidance = true;
        }
        else if(key == 'b' || key == 'B')
        {
            menuToggleBackwards = true;
        }

        if(menuReset)
        {
            menuShowGraph = true;
            menuShowInertialGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            mpSystem->ResetActiveMap();
            menuReset = false;
        }

        if(menuStop)
        {
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();

            // Stop all threads
            mpSystem->Shutdown();

            // Save camera trajectory
            mpSystem->SaveTrajectoryEuRoC("CameraTrajectory.txt");
            mpSystem->SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
            menuStop = false;
        }

        if(Stop())
        {
            while(isStopped())
            {
                usleep(3000);
            }
        }

        if(CheckFinish())
            break;
    }

    SetFinish();
#else
    // Pangolin not available - run a simple non-GUI mode
    std::cout << "Viewer: Running in non-GUI mode (Pangolin not available)" << std::endl;
    
    while(!CheckFinish())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    SetFinish();
#endif
}

void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

/*void Viewer::SetTrackingPause()
{
    mbStopTrack = true;
}*/

}
