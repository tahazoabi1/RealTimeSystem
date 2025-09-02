@echo off
echo "Building ORB-SLAM3 with vcpkg dependencies"

REM Set vcpkg toolchain path
set VCPKG_ROOT=C:\vcpkg
set CMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake

REM Clean old build directories
echo "Cleaning old build directories..."
if exist "Thirdparty\DBoW2\build" rd /s /q "Thirdparty\DBoW2\build"
if exist "Thirdparty\g2o\build" rd /s /q "Thirdparty\g2o\build" 
if exist "Thirdparty\Sophus\build" rd /s /q "Thirdparty\Sophus\build"
if exist "build" rd /s /q "build"

echo "Building third-party libraries with vcpkg toolchain..."

REM Build DBoW2
echo "Building DBoW2..."
cd Thirdparty\DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=%CMAKE_TOOLCHAIN_FILE%
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo "DBoW2 build failed"
    cd ..\..\..
    pause
    exit /b 1
)
cd ..\..\..

REM Build g2o  
echo "Building g2o..."
cd Thirdparty\g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=%CMAKE_TOOLCHAIN_FILE%
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo "g2o build failed"
    cd ..\..\..
    pause
    exit /b 1
)
cd ..\..\..

REM Build Sophus
echo "Building Sophus..."
cd Thirdparty\Sophus
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=%CMAKE_TOOLCHAIN_FILE%
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo "Sophus build failed"
    cd ..\..\..
    pause
    exit /b 1
)
cd ..\..\..

REM Build main ORB_SLAM3
echo "Building main ORB_SLAM3 library..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=%CMAKE_TOOLCHAIN_FILE%
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo "ORB_SLAM3 build failed"
    cd ..
    pause
    exit /b 1
)

echo "Build completed successfully!"
echo "Libraries built in lib/ directory"
echo "Executables built in Examples/ subdirectories"
pause