/**
 * AudioGuide.cc
 * 
 * Implementation of text-to-speech audio guidance system
 * Cross-platform implementation with Windows SAPI support
 */

#include "AudioGuide.h"
#include <iostream>
#include <chrono>
#include <algorithm>

#if defined(FORCE_WINDOWS_TTS) || defined(_WIN32) || defined(WIN32) || defined(__WIN32__) || defined(__NT__)
#include <windows.h>
#include <sapi.h>
#include <atlbase.h>
#include <codecvt>
#include <locale>
// Using direct COM SAPI calls for reliable TTS
#endif

namespace ORB_SLAM3
{

#if defined(FORCE_WINDOWS_TTS) || defined(_WIN32) || defined(WIN32) || defined(__WIN32__) || defined(__NT__)
// Helper functions for Windows direct SAPI TTS
std::wstring to_wide(const std::string& utf8) {
    if (utf8.empty()) return std::wstring();
    
    int len = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, NULL, 0);
    if (len == 0) return std::wstring();
    
    std::wstring wstr(len - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wstr[0], len);
    return wstr;
}

bool SpeakWithSAPI(const std::wstring& wtext, int vol /*0-100*/, int rate /*-10..10*/) {
    // Use CoInitialize as recommended by Microsoft docs
    HRESULT hr = CoInitialize(NULL);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        std::cout << "AudioGuide: COM initialization failed: " << std::hex << hr << std::endl;
        return false;
    }

    ISpVoice* pVoice = NULL;
    hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);
    
    bool success = false;
    if (SUCCEEDED(hr)) {
        std::cout << "AudioGuide: SAPI voice created successfully" << std::endl;
        
        // Set volume and rate with error checking
        HRESULT hrVol = pVoice->SetVolume((USHORT)std::max(0, std::min(100, vol)));
        HRESULT hrRate = pVoice->SetRate((LONG)std::max(-10, std::min(10, rate)));
        
        std::cout << "AudioGuide: Volume set result: " << std::hex << hrVol << std::endl;
        std::cout << "AudioGuide: Rate set result: " << std::hex << hrRate << std::endl;
        
        // Speak with error checking
        hr = pVoice->Speak(wtext.c_str(), 0, NULL);
        std::cout << "AudioGuide: Speak result: " << std::hex << hr << std::endl;
        
        success = SUCCEEDED(hr);
        pVoice->Release();
    } else {
        std::cout << "AudioGuide: Failed to create SAPI voice: " << std::hex << hr << std::endl;
    }
    
    CoUninitialize();
    return success;
}
#endif

AudioGuide::AudioGuide()
    : initialized_(false)
    , enabled_(true)
    , is_speaking_(false)
    , should_stop_thread_(false)
    , current_volume_(0.8f)
    , current_rate_(1.0f)
    , current_voice_("default")
    , spatial_audio_enabled_(false)
    , audio_direction_(0.0f)
#ifdef __APPLE__
    , speech_synthesizer_(nullptr)
    , synthesizer_delegate_(nullptr)
    , current_voice_id_(nullptr)
#endif
{
    std::cout << "AudioGuide: Initializing text-to-speech system..." << std::endl;
}

AudioGuide::~AudioGuide()
{
    Shutdown();
}

bool AudioGuide::Initialize()
{
    if (initialized_) {
        return true;
    }
    
    std::cout << "AudioGuide: Starting initialization..." << std::endl;
    
// Force Windows mode only - no conditional checks
std::cout << "AudioGuide: Windows platform detected - using PowerShell TTS" << std::endl;
    
    // Start audio processing thread
    should_stop_thread_ = false;
    audio_thread_ = std::thread(&AudioGuide::AudioThreadMain, this);
    
    initialized_ = true;
    std::cout << "AudioGuide: Initialization completed successfully (Windows TTS mode)" << std::endl;
    
    // Test audio system
    TestAudio();
    
    return true;
}

void AudioGuide::Shutdown()
{
    if (!initialized_) {
        return;
    }
    
    std::cout << "AudioGuide: Shutting down..." << std::endl;
    
    // Stop audio thread
    should_stop_thread_ = true;
    if (audio_thread_.joinable()) {
        audio_thread_.join();
    }
    
    // Stop any current speech
    StopSpeaking();
    ClearQueue();
    
    // Windows cleanup (no special cleanup needed for PowerShell TTS)
    
    initialized_ = false;
    std::cout << "AudioGuide: Shutdown completed" << std::endl;
}

bool AudioGuide::IsInitialized() const
{
    return initialized_;
}

void AudioGuide::Speak(const std::string& text, AudioPriority priority)
{
    if (!initialized_ || !enabled_ || text.empty()) {
        return;
    }
    
    AudioInstruction instruction;
    instruction.text = text;
    instruction.priority = priority;
    instruction.volume = current_volume_;
    instruction.rate = current_rate_;
    instruction.interrupt_current = (priority == AudioPriority::URGENT);
    instruction.timestamp = GetCurrentTimeSeconds();
    
    AddToQueue(instruction);
    
    std::cout << "🔊 AudioGuide: \"" << text << "\" (priority " 
              << static_cast<int>(priority) << ")" << std::endl;
}

void AudioGuide::SpeakImmediate(const std::string& text, AudioPriority priority)
{
    if (!initialized_ || !enabled_ || text.empty()) {
        return;
    }
    
    // Stop current speech for immediate instructions
    StopSpeaking();
    
    AudioInstruction instruction;
    instruction.text = text;
    instruction.priority = priority;
    instruction.volume = current_volume_;
    instruction.rate = current_rate_;
    instruction.interrupt_current = true;
    instruction.timestamp = GetCurrentTimeSeconds();
    
    // Add to front of queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::queue<AudioInstruction> temp_queue;
        temp_queue.push(instruction);
        
        while (!instruction_queue_.empty()) {
            temp_queue.push(instruction_queue_.front());
            instruction_queue_.pop();
        }
        
        instruction_queue_ = std::move(temp_queue);
    }
    
    std::cout << "🔊 IMMEDIATE AudioGuide: \"" << text << "\"" << std::endl;
}

void AudioGuide::SpeakWithPriority(const std::string& text, AudioPriority priority)
{
    if (!initialized_ || !enabled_ || text.empty()) {
        return;
    }
    
    // Route to appropriate method based on priority level
    if (priority == AudioPriority::URGENT || priority == AudioPriority::HIGH) {
        SpeakImmediate(text, priority);
    } else {
        Speak(text, priority);
    }
}

void AudioGuide::StopSpeaking()
{
    if (!initialized_) {
        return;
    }
    
    // Stop current speech (PowerShell processes will be killed by Windows)
    
    is_speaking_ = false;
}

void AudioGuide::ClearQueue()
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!instruction_queue_.empty()) {
        instruction_queue_.pop();
    }
}

void AudioGuide::SetVoiceSettings(float volume, float rate)
{
    current_volume_ = std::max(0.1f, std::min(1.0f, volume));
    current_rate_ = std::max(0.1f, std::min(2.0f, rate));
    
    std::cout << "AudioGuide: Voice settings updated - volume=" << current_volume_ 
              << ", rate=" << current_rate_ << std::endl;
}

void AudioGuide::SetVoice(const std::string& voice_identifier)
{
    current_voice_ = voice_identifier;
    std::cout << "AudioGuide: Voice changed to: " << voice_identifier << std::endl;
}

std::vector<std::string> AudioGuide::GetAvailableVoices()
{
    return {"default", "console_output"};
}

bool AudioGuide::IsSpeaking() const
{
    return is_speaking_;
}

bool AudioGuide::HasPendingInstructions() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return !instruction_queue_.empty();
}

size_t AudioGuide::GetQueueSize() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return instruction_queue_.size();
}

void AudioGuide::EnableSpatialAudio(bool enable)
{
    spatial_audio_enabled_ = enable;
    std::cout << "AudioGuide: Spatial audio " << (enable ? "enabled" : "disabled") << std::endl;
}

void AudioGuide::SetAudioDirection(float azimuth_degrees)
{
    audio_direction_ = azimuth_degrees;
    if (spatial_audio_enabled_) {
        std::cout << "AudioGuide: Audio direction set to " << azimuth_degrees << "°" << std::endl;
    }
}

void AudioGuide::TestAudio()
{
    Speak("Audio guidance system ready. Navigation instructions will be displayed in console.", AudioPriority::NORMAL);
}

void AudioGuide::SetEnabled(bool enabled)
{
    enabled_ = enabled;
    if (!enabled) {
        StopSpeaking();
        ClearQueue();
    }
    std::cout << "AudioGuide: Audio output " << (enabled ? "enabled" : "disabled") << std::endl;
}

bool AudioGuide::IsEnabled() const
{
    return enabled_;
}

// Private implementation methods

void AudioGuide::AudioThreadMain()
{
    std::cout << "AudioGuide: Audio processing thread started" << std::endl;
    
    while (!should_stop_thread_) {
        if (HasPendingInstructions() && !is_speaking_) {
            ProcessNextInstruction();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "AudioGuide: Audio processing thread stopped" << std::endl;
}

void AudioGuide::ProcessNextInstruction()
{
    AudioInstruction instruction;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (instruction_queue_.empty()) {
            return;
        }
        
        instruction = instruction_queue_.front();
        instruction_queue_.pop();
    }
    
    if (instruction.interrupt_current && is_speaking_) {
        StopSpeaking();
    }
    
    std::cout << "🎵 SPEAKING: \"" << instruction.text << "\" (vol=" 
              << instruction.volume << ", rate=" << instruction.rate << ")" << std::endl;
    
    is_speaking_ = true;
    
    // Use direct COM SAPI calls - no PowerShell needed
    std::wstring wtext = to_wide(instruction.text);
    int vol = (int)std::round(instruction.volume * 100.0f);
    int rate = (int)std::round((instruction.rate - 1.0f) * 10.0f);
    
    std::cout << "AudioGuide: Speaking (Direct SAPI): '" << instruction.text << "'" << std::endl;
    std::cout << "AudioGuide: Volume=" << vol << ", Rate=" << rate << std::endl;
    
    bool ok = SpeakWithSAPI(wtext, vol, rate);
    if (!ok) {
        std::cout << "AudioGuide: Direct SAPI TTS failed" << std::endl;
        // Fallback to timing simulation
        float duration = instruction.text.length() * 0.08f / instruction.rate;
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration * 1000)));
    } else {
        std::cout << "AudioGuide: Direct SAPI TTS completed successfully" << std::endl;
    }
    
    is_speaking_ = false;
}

double AudioGuide::GetCurrentTimeSeconds()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1e6;
}

bool AudioGuide::ShouldInterruptCurrent(AudioPriority new_priority)
{
    return static_cast<int>(new_priority) <= static_cast<int>(AudioPriority::HIGH);
}

void AudioGuide::AddToQueue(const AudioInstruction& instruction)
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    instruction_queue_.push(instruction);
}

bool AudioGuide::IsValidVoice(const std::string& voice_id)
{
    auto available_voices = GetAvailableVoices();
    return std::find(available_voices.begin(), available_voices.end(), voice_id) != available_voices.end();
}

std::string AudioGuide::GetDefaultVoice()
{
    return "default";
}

} // namespace ORB_SLAM3