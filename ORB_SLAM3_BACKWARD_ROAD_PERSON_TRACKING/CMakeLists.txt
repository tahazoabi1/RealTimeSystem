cmake_minimum_required(VERSION 3.5)
project(ORB_SLAM3)

# Disable main thread checker on macOS to allow Pangolin threading
if(APPLE)
    set(CMAKE_XCODE_SCHEME_DISABLE_MAIN_THREAD_CHECKER ON)
    # Add flags to disable main thread checker at runtime
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-main-thread-checker")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-main-thread-checker")
endif()

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Windows-specific fixes
if(WIN32)
    # Prevent Windows.h from defining min/max macros that conflict with std::min/std::max
    add_definitions(-DNOMINMAX)
    # Fix byte ambiguity between std::byte and Windows byte
    add_definitions(-D_HAS_STD_BYTE=0)
    # Use Windows-specific compiler flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

# Use C++17 for Pangolin compatibility on Windows
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DCOMPILEDWITHC11)
message(STATUS "Using C++17 standard.")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Set OpenCV_ROOT for vcpkg as suggested by vcpkg output
set(OpenCV_ROOT "${CMAKE_PREFIX_PATH}/share/opencv4")

# Find OpenCV
find_package(OpenCV REQUIRED)

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin QUIET)
find_package(realsense2)

# ONNX Runtime configuration
set(ONNXRUNTIME_ROOT_DIR "${PROJECT_SOURCE_DIR}/onnxruntime-win-x64-gpu-1.18.1")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_DIR}/include")
set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.lib")

# Check if ONNX Runtime is available
if(EXISTS ${ONNXRUNTIME_INCLUDE_DIRS} AND EXISTS ${ONNXRUNTIME_LIBRARIES})
    message(STATUS "Found ONNX Runtime at ${ONNXRUNTIME_ROOT_DIR}")
    set(HAVE_ONNXRUNTIME 1)
    add_definitions(-DHAVE_ONNXRUNTIME)
else()
    message(STATUS "ONNX Runtime not found, disabling GPU acceleration")
    set(HAVE_ONNXRUNTIME 0)
endif()

if(Pangolin_FOUND)
    message(STATUS "Found Pangolin")
    set(HAVE_PANGOLIN 1)
    add_definitions(-DHAVE_PANGOLIN)
else()
    message(STATUS "Pangolin not found, building without visualization")
    set(HAVE_PANGOLIN 0)
endif()

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
)

if(HAVE_ONNXRUNTIME)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
endif()

if(Pangolin_FOUND)
include_directories(${Pangolin_INCLUDE_DIRS})
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(ORB_SLAM3_SOURCES
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
src/PathRecorder.cc
src/PathMatcher.cc
src/PathGuide.cc
src/PathOptimizer.cc
src/YOLOInterface.cc
src/FrameDrawer_YOLO.cc
src/YOLODetectorCPP.cc
src/DeviceManagerCPP.cc
src/ActivityDetectorCPP.cc
src/VisualizerCPP.cc
src/ZoneAnalyticsCPP.cc
src/FaceRecognizerCPP.cc
src/EnhancedHybridTrackerCPP.cc
)

if(WIN32)
    list(APPEND ORB_SLAM3_SOURCES 
        src/AudioGuide.cc
        src/boost_exception_stub.cpp
    )
    message(STATUS "Using Windows AudioGuide implementation: AudioGuide.cc")
    # Force Windows macros to be defined
    add_compile_definitions(_WIN32 WIN32 __WIN32__ __NT__)
elseif(APPLE)
    list(APPEND ORB_SLAM3_SOURCES src/AudioGuide.mm)
    message(STATUS "Using macOS AudioGuide implementation: AudioGuide.mm")
else()
    list(APPEND ORB_SLAM3_SOURCES src/AudioGuide.cc)
    message(STATUS "Using default AudioGuide implementation: AudioGuide.cc")
    # Also define Windows macros for non-Apple systems (likely Windows)
    add_compile_definitions(_WIN32 WIN32 __WIN32__ __NT__)
endif()

add_library(${PROJECT_NAME} SHARED ${ORB_SLAM3_SOURCES}
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h
include/PathRecorder.h
include/PathMatcher.h
include/PathGuide.h
include/PathOptimizer.h
include/YOLOInterface.h
include/FrameDrawer_YOLO.h
include/YOLODetectorCPP.h
include/DeviceManagerCPP.h
include/ActivityDetectorCPP.h
include/VisualizerCPP.h
include/ZoneAnalyticsCPP.h
include/FaceRecognizerCPP.h
include/EnhancedHybridTrackerCPP.h
)

if(NOT WIN32)
list(APPEND ORB_SLAM3_SOURCES include/AudioGuide.h)
endif()

add_subdirectory(Thirdparty/g2o)

# Define export macro for DLL on Windows
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
        ORB_SLAM3_EXPORTS
        BOOST_EXCEPTION_SOURCE_LOCATION_NOEXCEPT
    )
    # Enable exception handling and suppress DLL interface warnings
    target_compile_options(${PROJECT_NAME} PRIVATE "/EHsc" "/wd4251" "/wd4275")
endif()

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
g2o
)

if(Pangolin_FOUND)
    target_link_libraries(${PROJECT_NAME} ${Pangolin_LIBRARIES})
endif()

if(WIN32)
    find_package(Boost REQUIRED COMPONENTS serialization system)
    target_link_libraries(${PROJECT_NAME} 
        ${Boost_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/Release/DBoW2.lib
        # Windows system libraries for YOLO integration
        kernel32
        user32
        advapi32
    )
    
    # Add ONNX Runtime libraries if available
    if(HAVE_ONNXRUNTIME)
        target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIBRARIES})
    endif()
else()
    target_link_libraries(${PROJECT_NAME}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.dylib
        -lboost_serialization
        -lcrypto
        "-framework AVFoundation"
        "-framework Foundation"
        # POSIX shared memory for YOLO integration
        -lrt
        -lpthread
    )
    
    # Add ONNX Runtime libraries if available (macOS/Linux)
    if(HAVE_ONNXRUNTIME)
        target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIBRARIES})
    endif()
endif()

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
    ${realsense_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    )
endif()


# Build examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i
            Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265
            Examples/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i
            Examples/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

add_executable(mono_kitti_headless
        Examples/Monocular/mono_kitti_headless.cc)
target_link_libraries(mono_kitti_headless ${PROJECT_NAME})

if(APPLE)
    add_executable(mono_kitti_macos
            Examples/Monocular/mono_kitti_macos.cc)
    target_link_libraries(mono_kitti_macos ${PROJECT_NAME})

    add_executable(mono_tum_vi_macos
            Examples/Monocular/mono_tum_vi_macos.cc)
    target_link_libraries(mono_tum_vi_macos ${PROJECT_NAME})
endif()

add_executable(mono_rtmp_stream
        Examples/Monocular/mono_rtmp_stream.cc)
target_link_libraries(mono_rtmp_stream ${PROJECT_NAME})

# Native C++ YOLO-integrated RTMP stream executable with exact Python algorithm port
add_executable(mono_rtmp_stream_yolo_cpp
        Examples/Monocular/mono_rtmp_stream_yolo_cpp.cc
        src/YOLODetectorCPP.cc)
target_link_libraries(mono_rtmp_stream_yolo_cpp ${PROJECT_NAME})

# Complete Enhanced Hybrid Tracker with all modules - exact Python port
add_executable(mono_rtmp_stream_enhanced_hybrid
        Examples/Monocular/mono_rtmp_stream_enhanced_hybrid.cc)
target_link_libraries(mono_rtmp_stream_enhanced_hybrid ${PROJECT_NAME})

# Clean integration: mono_rtmp_stream + person detection
add_executable(mono_integration
        Examples/Monocular/mono_integration.cc)
target_link_libraries(mono_integration ${PROJECT_NAME})

# Python integration: mono_rtmp_stream + Python Enhanced Hybrid Tracker
add_executable(mono_integration_vv
        Examples/Monocular/mono_integration_vv.cc)
target_link_libraries(mono_integration_vv ${PROJECT_NAME})

if(APPLE)
    add_executable(mono_kitti_macos_viewer
            Examples/Monocular/mono_kitti_macos_viewer.cc)
    target_link_libraries(mono_kitti_macos_viewer ${PROJECT_NAME})
endif()

if(realsense2_FOUND)
    add_executable(mono_realsense_t265
            Examples/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_realsense_D435i
            Examples/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
        Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265
            Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i
            Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc
        Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi
        Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265
            Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i
            Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Calibration)
if(realsense2_FOUND)
    add_executable(recorder_realsense_D435i
            Examples/Calibration/recorder_realsense_D435i.cc)
    target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})

    add_executable(recorder_realsense_T265
            Examples/Calibration/recorder_realsense_T265.cc)
    target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
endif()

#Old examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D)

add_executable(rgbd_tum_old
        Examples_old/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i_old
            Examples_old/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i_old ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i_old
            Examples_old/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo)

add_executable(stereo_kitti_old
        Examples_old/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti_old ${PROJECT_NAME})

add_executable(stereo_euroc_old
        Examples_old/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc_old ${PROJECT_NAME})

add_executable(stereo_tum_vi_old
        Examples_old/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265_old
            Examples_old/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265_old ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i_old
            Examples_old/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i_old ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular)

add_executable(mono_tum_old
        Examples_old/Monocular/mono_tum.cc)
target_link_libraries(mono_tum_old ${PROJECT_NAME})

add_executable(mono_kitti_old
        Examples_old/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti_old ${PROJECT_NAME})

add_executable(mono_euroc_old
        Examples_old/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc_old ${PROJECT_NAME})

add_executable(mono_tum_vi_old
        Examples_old/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265_old
            Examples_old/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265_old ${PROJECT_NAME})

    add_executable(mono_realsense_D435i_old
            Examples_old/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i_old ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular-Inertial)

add_executable(mono_inertial_euroc_old
        Examples_old/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc_old ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi_old
        Examples_old/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265_old ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo-Inertial)

add_executable(stereo_inertial_euroc_old
        Examples_old/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc_old ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi_old
        Examples_old/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265_old ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()
