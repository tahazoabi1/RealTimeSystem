#ifndef ORBSLAM3_EXPORT_H
#define ORBSLAM3_EXPORT_H

#ifdef _WIN32
    #ifdef ORB_SLAM3_EXPORTS
        #define ORB_SLAM3_API __declspec(dllexport)
    #else
        #define ORB_SLAM3_API __declspec(dllimport)
    #endif
#else
    #define ORB_SLAM3_API
#endif

#endif // ORBSLAM3_EXPORT_H