@echo off
echo Starting mono_rtmp_stream...
mono_rtmp_stream.exe ../../../Vocabulary/ORBvoc.txt ../../../Examples/Monocular/iPhone16Plus.yaml rtmp://localhost:1935/live/stream
echo Exit code: %ERRORLEVEL%
pause