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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#ifdef HAVE_PANGOLIN
#include <pangolin/pangolin.h>
#ifdef _WIN32
#include <GL/gl.h>
#include <GL/glu.h>
#else
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#endif
#else
// Define empty macros when Pangolin is not available
#define glPointSize(x) do {} while(0)
#define glBegin(x) do {} while(0)
#define glEnd() do {} while(0)
#define glColor3f(r,g,b) do {} while(0)
#define glColor4f(r,g,b,a) do {} while(0)
#define glVertex3f(x,y,z) do {} while(0)
#define glLineWidth(x) do {} while(0)
#define glPushMatrix() do {} while(0)
#define glPopMatrix() do {} while(0)
#define glMultMatrixf(x) do {} while(0)
#define glMultMatrixd(x) do {} while(0)
#define GL_POINTS 0
#define GL_LINES 0
typedef float GLfloat;
#endif
#include <mutex>
#include <fstream>
#include <chrono>

namespace ORB_SLAM3
{
static auto last_export_time = std::chrono::steady_clock::now();
static auto last_camera_export_time = std::chrono::steady_clock::now();


MapDrawer::MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings):mpAtlas(pAtlas)
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
}

void MapDrawer::newParameterLoader(Settings *settings) {
    mKeyFrameSize = settings->keyFrameSize();
    mKeyFrameLineWidth = settings->keyFrameLineWidth();
    mGraphLineWidth = settings->graphLineWidth();
    mPointSize = settings->pointSize();
    mCameraSize = settings->cameraSize();
    mCameraLineWidth  = settings->cameraLineWidth();
}

bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
    if(!node.empty())
    {
        mKeyFrameSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.KeyFrameLineWidth"];
    if(!node.empty())
    {
        mKeyFrameLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.GraphLineWidth"];
    if(!node.empty())
    {
        mGraphLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.PointSize"];
    if(!node.empty())
    {
        mPointSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraSize"];
    if(!node.empty())
    {
        mCameraSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraLineWidth"];
    if(!node.empty())
    {
        mCameraLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void MapDrawer::DrawMapPoints()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap)
        return;

    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        Eigen::Matrix<float,3,1> pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        Eigen::Matrix<float,3,1> pos = (*sit)->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));

    }
    // --- Real-time MapPoints export every 2 seconds ---
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_export_time).count() >= 2)  {
        ExportMapPoints("MapPoints.txt");
        last_export_time = now;
    }
    glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    Map* pActiveMap = mpAtlas->GetCurrentMap();
    // DEBUG LBA
    std::set<long unsigned int> sOptKFs = pActiveMap->msOptKFs;
    std::set<long unsigned int> sFixedKFs = pActiveMap->msFixedKFs;

    if(!pActiveMap)
        return;

    const vector<KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
            unsigned int index_color = pKF->mnOriginMapId;

            glPushMatrix();

            glMultMatrixf((GLfloat*)Twc.data());

            if(!pKF->GetParent()) // It is the first KF in the map
            {
                glLineWidth(mKeyFrameLineWidth*5);
                glColor3f(1.0f,0.0f,0.0f);
                glBegin(GL_LINES);
            }
            else
            {
                //cout << "Child KF: " << vpKFs[i]->mnId << endl;
                glLineWidth(mKeyFrameLineWidth);
                if (bDrawOptLba) {
                    if(sOptKFs.find(pKF->mnId) != sOptKFs.end())
                    {
                        glColor3f(0.0f,1.0f,0.0f); // Green -> Opt KFs
                    }
                    else if(sFixedKFs.find(pKF->mnId) != sFixedKFs.end())
                    {
                        glColor3f(1.0f,0.0f,0.0f); // Red -> Fixed KFs
                    }
                    else
                    {
                        glColor3f(0.0f,0.0f,1.0f); // Basic color
                    }
                }
                else
                {
                    glColor3f(0.0f,0.0f,1.0f); // Basic color
                }
                glBegin(GL_LINES);
            }

            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();

            glEnd();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        // cout << "-----------------Draw graph-----------------" << endl;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow(0),Ow(1),Ow(2));
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                Eigen::Vector3f Owp = pParent->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
    }

    if(bDrawInertialGraph && pActiveMap->isImuInitialized())
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(1.0f,0.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        //Draw inertial links
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();
            KeyFrame* pNext = pKFi->mNextKF;
            if(pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }
        }

        glEnd();
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();

    if(bDrawKF)
    {
        for(Map* pMap : vpMaps)
        {
            if(pMap == pActiveMap)
                continue;

            vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                unsigned int index_color = pKF->mnOriginMapId;

                glPushMatrix();

                glMultMatrixf((GLfloat*)Twc.data());

                if(!vpKFs[i]->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth*5);
                    glColor3f(1.0f,0.0f,0.0f);
                    glBegin(GL_LINES);
                }
                else
                {
                    glLineWidth(mKeyFrameLineWidth);
                    glColor3f(mfFrameColors[index_color][0],mfFrameColors[index_color][1],mfFrameColors[index_color][2]);
                    glBegin(GL_LINES);
                }

                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }
    }
}

#ifdef HAVE_PANGOLIN
void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
#else
void MapDrawer::DrawCurrentCamera(void* Twc)
#endif
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_PANGOLIN
#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif
#else
    // No-op when pangolin is not available
    (void)Twc;
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
    auto now_cam = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now_cam - last_camera_export_time).count() >= 2)  {
        ExportCameraPosition("CameraPosition.txt");
        last_camera_export_time = now_cam;
    }
}


void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.inverse();
}

#ifdef HAVE_PANGOLIN
void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
#else
void MapDrawer::GetCurrentOpenGLCameraMatrix(void* M, void* MOw)
#endif
{
#ifdef HAVE_PANGOLIN
    Eigen::Matrix4f Twc;
    {
        unique_lock<mutex> lock(mMutexCamera);
        Twc = mCameraPose.matrix();
    }

    for (int i = 0; i<4; i++) {
        M.m[4*i] = Twc(0,i);
        M.m[4*i+1] = Twc(1,i);
        M.m[4*i+2] = Twc(2,i);
        M.m[4*i+3] = Twc(3,i);
    }

    MOw.SetIdentity();
    MOw.m[12] = Twc(0,3);
    MOw.m[13] = Twc(1,3);
    MOw.m[14] = Twc(2,3);
#else
    // No-op when pangolin is not available
    (void)M;
    (void)MOw;
#endif
}
void MapDrawer::ExportMapPoints(const std::string& filename,
                                float export_dt /* = 0.3f */)   // 0.3 s → ≈ 3 Hz
{
    using clock = std::chrono::steady_clock;
    static auto  last = clock::now();

    // ─── 1. Throttle ──────────────────────────────────────────────────────────
    if (std::chrono::duration<float>(clock::now() - last).count() < export_dt)
        return;                        // update still “fresh” → skip this call
    last = clock::now();

    // ─── 2. Grab active map ───────────────────────────────────────────────────
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if (!pActiveMap) return;

    const std::vector<MapPoint*>& vpMPs = pActiveMap->GetAllMapPoints();

    // ─── 3. Overwrite file (truncate) ─────────────────────────────────────────
    std::ofstream out(filename, std::ios_base::trunc);
    if (!out) return;                  // fail-safe

    // ─── 4. Dump 3-D coordinates with timestamps ────────────────────────────
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    for (auto* mp : vpMPs)
    {
        if (!mp || mp->isBad()) continue;

        const Eigen::Vector3f& pos = mp->GetWorldPos();
        // Format: X Y Z TIMESTAMP
        out << pos.x() << ' ' << pos.y() << ' ' << pos.z() << ' ' << now << '\n';
    }
}
void MapDrawer::ExportCameraPosition(const std::string& filename,
                                     float period_sec /* =0.1f */)
{
    using clock = std::chrono::steady_clock;
    static auto last = clock::now();
    auto now         = clock::now();

    if (std::chrono::duration<float>(now - last).count() < period_sec)
        return;                                 // too soon – skip

    last = now;

    Eigen::Matrix4f Twc;
    {   std::unique_lock<std::mutex> lock(mMutexCamera);
        Twc = mCameraPose.matrix();
    }

    std::ofstream out(filename, std::ios::trunc);   // overwrite
    if (!out) return;

    out << Twc(0,3) << ' ' << Twc(1,3) << ' ' << Twc(2,3) << '\n';
}

void MapDrawer::AddStoppedPersonLandmark(const std::vector<cv::Point3f>& points, const std::string& person_name, int track_id)
{
    std::unique_lock<std::mutex> lock(mMutexStoppedPersons);
    
    StoppedPersonLandmark landmark;
    landmark.points = points;
    landmark.person_name = person_name;
    landmark.track_id = track_id;
    landmark.timestamp = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    
    mStoppedPersonLandmarks.push_back(landmark);
    
    // Limit to last 50 landmarks to avoid memory issues
    if (mStoppedPersonLandmarks.size() > 50) {
        mStoppedPersonLandmarks.erase(mStoppedPersonLandmarks.begin());
    }
}

void MapDrawer::DrawStoppedPersonLandmarks()
{
#ifdef HAVE_PANGOLIN
    std::unique_lock<std::mutex> lock(mMutexStoppedPersons);
    
    if (mStoppedPersonLandmarks.empty()) {
        return;
    }
    
    // Draw person landmarks with special markers
    for (const auto& landmark : mStoppedPersonLandmarks) {
        if (landmark.points.empty()) continue;
        
        // Set color based on track ID (cycle through colors)
        int color_idx = landmark.track_id % 6;
        float r = mfFrameColors[color_idx][0];
        float g = mfFrameColors[color_idx][1]; 
        float b = mfFrameColors[color_idx][2];
        
        // Draw slightly larger markers for stopped person locations
        glPointSize(mPointSize * 1.5f); // 1.5x larger than normal map points
        glColor3f(r, g, b);
        
        glBegin(GL_POINTS);
        // Only draw the center point (first point) to avoid clutter
        if (!landmark.points.empty()) {
            glVertex3f(landmark.points[0].x, landmark.points[0].y, landmark.points[0].z);
        }
        glEnd();
        
        // Draw a small cube around the center point to make it more visible
        if (!landmark.points.empty()) {
            cv::Point3f center = landmark.points[0]; // Use first point as center
            float cube_size = 0.02f; // 2cm cube
            
            glColor4f(r, g, b, 0.5f); // Semi-transparent
            glLineWidth(mGraphLineWidth * 2.0f);
            
            // Draw cube wireframe
            glBegin(GL_LINES);
            
            // Bottom face
            glVertex3f(center.x - cube_size, center.y - cube_size, center.z - cube_size);
            glVertex3f(center.x + cube_size, center.y - cube_size, center.z - cube_size);
            
            glVertex3f(center.x + cube_size, center.y - cube_size, center.z - cube_size);
            glVertex3f(center.x + cube_size, center.y + cube_size, center.z - cube_size);
            
            glVertex3f(center.x + cube_size, center.y + cube_size, center.z - cube_size);
            glVertex3f(center.x - cube_size, center.y + cube_size, center.z - cube_size);
            
            glVertex3f(center.x - cube_size, center.y + cube_size, center.z - cube_size);
            glVertex3f(center.x - cube_size, center.y - cube_size, center.z - cube_size);
            
            // Top face
            glVertex3f(center.x - cube_size, center.y - cube_size, center.z + cube_size);
            glVertex3f(center.x + cube_size, center.y - cube_size, center.z + cube_size);
            
            glVertex3f(center.x + cube_size, center.y - cube_size, center.z + cube_size);
            glVertex3f(center.x + cube_size, center.y + cube_size, center.z + cube_size);
            
            glVertex3f(center.x + cube_size, center.y + cube_size, center.z + cube_size);
            glVertex3f(center.x - cube_size, center.y + cube_size, center.z + cube_size);
            
            glVertex3f(center.x - cube_size, center.y + cube_size, center.z + cube_size);
            glVertex3f(center.x - cube_size, center.y - cube_size, center.z + cube_size);
            
            // Vertical edges
            glVertex3f(center.x - cube_size, center.y - cube_size, center.z - cube_size);
            glVertex3f(center.x - cube_size, center.y - cube_size, center.z + cube_size);
            
            glVertex3f(center.x + cube_size, center.y - cube_size, center.z - cube_size);
            glVertex3f(center.x + cube_size, center.y - cube_size, center.z + cube_size);
            
            glVertex3f(center.x + cube_size, center.y + cube_size, center.z - cube_size);
            glVertex3f(center.x + cube_size, center.y + cube_size, center.z + cube_size);
            
            glVertex3f(center.x - cube_size, center.y + cube_size, center.z - cube_size);
            glVertex3f(center.x - cube_size, center.y + cube_size, center.z + cube_size);
            
            glEnd();
        }
        
        // Draw connecting lines between landmark points
        if (landmark.points.size() > 1) {
            glColor3f(r, g, b);
            glLineWidth(mGraphLineWidth);
            
            glBegin(GL_LINES);
            for (size_t i = 0; i < landmark.points.size() - 1; i++) {
                glVertex3f(landmark.points[i].x, landmark.points[i].y, landmark.points[i].z);
                glVertex3f(landmark.points[i+1].x, landmark.points[i+1].y, landmark.points[i+1].z);
            }
            glEnd();
        }
    }
#endif
}


} //namespace ORB_SLAM
