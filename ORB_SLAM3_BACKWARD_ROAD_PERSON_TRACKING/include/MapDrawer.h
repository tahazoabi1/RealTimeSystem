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


#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Atlas.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include "Settings.h"
#include "orbslam3_export.h"

#ifdef HAVE_PANGOLIN
#ifdef _WIN32
// On Windows, GLEW must be included before any GL headers
#include <GL/glew.h>
#endif
#include<pangolin/pangolin.h>
#endif

#include<mutex>

namespace ORB_SLAM3
{

class Settings;

class ORB_SLAM3_API MapDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings);

    void newParameterLoader(Settings* settings);

    Atlas* mpAtlas;

    void DrawMapPoints();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba);
    
    // Draw stopped person landmarks with special markers
    void DrawStoppedPersonLandmarks();
    void AddStoppedPersonLandmark(const std::vector<cv::Point3f>& points, const std::string& person_name, int track_id);
#ifdef HAVE_PANGOLIN
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
#else
    void DrawCurrentCamera(void* Twc);
#endif
    void SetCurrentCameraPose(const Sophus::SE3f &Tcw);
    void SetReferenceKeyFrame(KeyFrame *pKF);
#ifdef HAVE_PANGOLIN
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);
#else
    void GetCurrentOpenGLCameraMatrix(void* M, void* MOw);
#endif
    
    void ExportMapPoints(const std::string& filename, float export_dt = 0.3f );
    void ExportCameraPosition(const std::string& filename, float period_sec = 0.1f);
   

private:

    bool ParseViewerParamFile(cv::FileStorage &fSettings);

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    Sophus::SE3f mCameraPose;

    std::mutex mMutexCamera;
    
    // Stopped person landmark data
    struct StoppedPersonLandmark {
        std::vector<cv::Point3f> points;
        std::string person_name;
        int track_id;
        double timestamp;
    };
    std::vector<StoppedPersonLandmark> mStoppedPersonLandmarks;
    std::mutex mMutexStoppedPersons;

    float mfFrameColors[6][3] = {{0.0f, 0.0f, 1.0f},
                                {0.8f, 0.4f, 1.0f},
                                {1.0f, 0.2f, 0.4f},
                                {0.6f, 0.0f, 1.0f},
                                {1.0f, 1.0f, 0.0f},
                                {0.0f, 1.0f, 1.0f}};

};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
