@echo off
echo "Windows build script for ORB-SLAM3"
echo "Note: This is a macOS-specific build adapted for Windows"

REM Check for required tools
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo "CMake not found. Please install CMake and add to PATH"
    exit /b 1
)

REM Clean old build directories
echo "Cleaning old build directories..."
if exist "Thirdparty\DBoW2\build" rd /s /q "Thirdparty\DBoW2\build"
if exist "Thirdparty\g2o\build" rd /s /q "Thirdparty\g2o\build" 
if exist "Thirdparty\Sophus\build" rd /s /q "Thirdparty\Sophus\build"
if exist "build" rd /s /q "build"

REM Build DBoW2
echo "Building DBoW2..."
cd Thirdparty\DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo "DBoW2 build failed"
    exit /b 1
)
cd ..\..\..

REM Build g2o
echo "Building g2o..."
cd Thirdparty\g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo "g2o build failed"
    exit /b 1
)
cd ..\..\..

REM Build Sophus
echo "Building Sophus..."
cd Thirdparty\Sophus
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo "Sophus build failed"
    exit /b 1
)
cd ..\..\..

REM Build main library
echo "Building ORB_SLAM3..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo "ORB_SLAM3 build failed"
    exit /b 1
)

echo "Build completed successfully!"