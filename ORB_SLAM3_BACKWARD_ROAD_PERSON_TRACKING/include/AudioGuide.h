/**
 * AudioGuide.h
 * 
 * Text-to-speech audio guidance system for blind navigation
 * macOS AVSpeechSynthesizer integration for PathGuide
 * Part of Phase 5.1.2: Text-to-speech integration
 */

#ifndef AUDIOGUIDE_H
#define AUDIOGUIDE_H

#include <string>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <memory>
#include "orbslam3_export.h"

// Platform-specific implementations use void* pointers to avoid header conflicts

namespace ORB_SLAM3
{

enum class ORB_SLAM3_API AudioPriority {
    LOW = 3,        // Info messages, progress updates
    NORMAL = 2,     // Standard navigation instructions  
    HIGH = 1,       // Important corrections, warnings
    URGENT = 0      // Emergency stops, critical alerts
};

struct AudioInstruction {
    std::string text;
    AudioPriority priority;
    float volume;           // 0.0 to 1.0
    float rate;            // 0.1 to 2.0 (speech speed)
    bool interrupt_current; // Should interrupt currently speaking
    double timestamp;
    
    AudioInstruction() : priority(AudioPriority::NORMAL), volume(0.8f), 
                        rate(1.0f), interrupt_current(false), timestamp(0.0) {}
};

class ORB_SLAM3_API AudioGuide
{
public:
    AudioGuide();
    ~AudioGuide();
    
    // Core audio functionality
    bool Initialize();
    void Shutdown();
    bool IsInitialized() const;
    
    // Speech queue management
    void Speak(const std::string& text, AudioPriority priority = AudioPriority::NORMAL);
    void SpeakImmediate(const std::string& text, AudioPriority priority = AudioPriority::URGENT);
    void SpeakWithPriority(const std::string& text, AudioPriority priority);  // New wrapper method
    void StopSpeaking();
    void ClearQueue();
    
    // Voice configuration
    void SetVoiceSettings(float volume = 0.8f, float rate = 1.0f);
    void SetVoice(const std::string& voice_identifier);
    std::vector<std::string> GetAvailableVoices();
    
    // Queue status
    bool IsSpeaking() const;
    bool HasPendingInstructions() const;
    size_t GetQueueSize() const;
    
    // Audio guidance modes
    void EnableSpatialAudio(bool enable);
    void SetAudioDirection(float azimuth_degrees); // For 3D audio positioning
    
    // Utility methods
    void TestAudio(); // Speak test message
    void SetEnabled(bool enabled);
    bool IsEnabled() const;

private:
    // Core components
    std::atomic<bool> initialized_;
    std::atomic<bool> enabled_;
public:
    std::atomic<bool> is_speaking_; // Public for Objective-C delegate access
private:
    
    // Speech queue management
    std::queue<AudioInstruction> instruction_queue_;
    mutable std::mutex queue_mutex_;
    
    // Audio processing thread
    std::thread audio_thread_;
    std::atomic<bool> should_stop_thread_;
    
    // Voice settings
    float current_volume_;
    float current_rate_;
    std::string current_voice_;
    bool spatial_audio_enabled_;
    float audio_direction_;
    
#ifdef __APPLE__
    // macOS-specific synthesizer (using void* to avoid header conflicts)
    void* speech_synthesizer_;
    void* synthesizer_delegate_;
    void* current_voice_id_;
#endif
    
    // Thread management
    void AudioThreadMain();
    void ProcessNextInstruction();
    
    // Platform-specific implementations
    bool InitializeMacOS();
    void SpeakTextMacOS(const AudioInstruction& instruction);
    void StopSpeakingMacOS();
    std::vector<std::string> GetAvailableVoicesMacOS();
    
    // Utility functions
    double GetCurrentTimeSeconds();
    bool ShouldInterruptCurrent(AudioPriority new_priority);
    void AddToQueue(const AudioInstruction& instruction);
    
    // Voice validation
    bool IsValidVoice(const std::string& voice_id);
    std::string GetDefaultVoice();
};

// Platform-specific delegate will be implemented later
#ifdef __APPLE__
// Forward declaration for future Objective-C++ implementation
class SpeechSynthesizerDelegate;
#endif

} // namespace ORB_SLAM3

#endif // AUDIOGUIDE_H