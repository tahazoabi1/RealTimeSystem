#!/bin/bash
# Launch script for ORB-SLAM3 with viewer on macOS
# This script attempts to work around NSApplication threading issues

# Set environment variables to attempt runtime fixes
export OBJC_DISABLE_MAIN_THREAD_CHECKER=1
export SDL_HINT_MAC_OPENGL_ASYNC_DISPATCH=1
export NSAppSleepDisabled=1

# Check if we have the right number of arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <executable> <vocab_path> <settings_path> <dataset_path>"
    echo "Example: $0 ./Examples/Monocular/mono_kitti_macos Vocabulary/ORBvoc.txt Examples/Monocular/Indoor_Drone.yaml /Volumes/Hi/datasets/indoor_drone/"
    exit 1
fi

# Get the executable and arguments
EXECUTABLE="$1"
shift
ARGS="$@"

echo "Attempting to run ORB-SLAM3 with viewer on macOS..."
echo "Executable: $EXECUTABLE"
echo "Arguments: $ARGS"
echo ""

# Try to run with modified environment
exec "$EXECUTABLE" $ARGS