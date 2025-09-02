/**
 * ZoneAnalyticsCPP.cc  
 * Exact port of modules/zone_analytics.py implementation
 */

#include "ZoneAnalyticsCPP.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ORB_SLAM3 {

// Constants (exact Python values)
const double ZoneAnalyticsCPP::DEFAULT_DWELL_THRESHOLD_SECONDS = 5.0;
const double ZoneAnalyticsCPP::DEFAULT_TRANSITION_TIMEOUT_SECONDS = 2.0;
const double ZoneAnalyticsCPP::DEFAULT_CLEANUP_INTERVAL_SECONDS = 60.0;
const int ZoneAnalyticsCPP::DEFAULT_MAX_HISTORY_ENTRIES = 1000;
const int ZoneAnalyticsCPP::ANALYTICS_UPDATE_INTERVAL_SECONDS = 1;

ZoneAnalyticsCPP::ZoneAnalyticsCPP()
    : dwell_time_threshold_(DEFAULT_DWELL_THRESHOLD_SECONDS)
    , transition_timeout_(DEFAULT_TRANSITION_TIMEOUT_SECONDS)
    , cleanup_interval_(DEFAULT_CLEANUP_INTERVAL_SECONDS)
    , max_history_entries_(DEFAULT_MAX_HISTORY_ENTRIES)
    , initialized_(false)
{
    // Initialize session data (exact Python session initialization)
    session_data_.session_start = std::chrono::steady_clock::now();
    session_data_.total_unique_visitors = 0;
    
    last_cleanup_ = std::chrono::steady_clock::now();
    last_update_ = std::chrono::steady_clock::now();
}

ZoneAnalyticsCPP::~ZoneAnalyticsCPP() = default;

bool ZoneAnalyticsCPP::initialize(const std::vector<ZoneDefinitionCPP>& zones)
{
    std::cout << "🔧 Initializing ZoneAnalytics C++..." << std::endl;
    
    // Set zones (exact Python zone initialization)
    set_zones(zones);
    
    // Initialize default alert conditions if none set (exact Python defaults)
    if (alert_conditions_.empty()) {
        AlertCondition occupancy_alert;
        occupancy_alert.condition_type = "occupancy_exceed";
        occupancy_alert.threshold_value = 5.0f;
        occupancy_alert.is_active = true;
        
        AlertCondition dwell_alert;
        dwell_alert.condition_type = "dwell_time_exceed";
        dwell_alert.threshold_value = 30.0f;
        dwell_alert.is_active = true;
        
        alert_conditions_ = {occupancy_alert, dwell_alert};
    }
    
    initialized_ = true;
    std::cout << "✅ ZoneAnalytics C++ initialized with " << zones_.size() << " zones" << std::endl;
    return true;
}

void ZoneAnalyticsCPP::update_person_positions(const std::vector<int>& person_ids,
                                              const std::vector<cv::Point2f>& positions,
                                              const std::vector<std::string>& activities,
                                              double timestamp)
{
    if (!initialized_ || person_ids.empty()) {
        return;
    }
    
    auto now = std::chrono::steady_clock::now();
    
    // Process each person (exact Python person processing)
    for (size_t i = 0; i < person_ids.size() && i < positions.size(); ++i) {
        int person_id = person_ids[i];
        cv::Point2f position = positions[i];
        std::string activity = (i < activities.size()) ? activities[i] : "Unknown";
        
        // Determine current zone (exact Python zone detection)
        std::string current_zone = get_person_zone(position);
        std::string previous_zone = (person_current_zones_.find(person_id) != person_current_zones_.end()) 
                                   ? person_current_zones_[person_id] : "";
        
        // Update person tracking data (exact Python tracking update)
        person_last_positions_[person_id] = position;
        person_current_activities_[person_id] = activity;
        
        // Handle zone changes (exact Python zone change logic)
        if (current_zone != previous_zone) {
            if (!previous_zone.empty() && is_valid_zone(previous_zone)) {
                handle_zone_exit(person_id, previous_zone, position, timestamp);
            }
            
            if (!current_zone.empty() && is_valid_zone(current_zone)) {
                handle_zone_entry(person_id, current_zone, position, timestamp);
            }
            
            if (!previous_zone.empty() && !current_zone.empty() && 
                is_valid_zone(previous_zone) && is_valid_zone(current_zone)) {
                handle_zone_transition(person_id, previous_zone, current_zone, position, timestamp);
            }
        }
        
        // Update current zone tracking (exact Python zone tracking)
        if (!current_zone.empty()) {
            person_current_zones_[person_id] = current_zone;
        } else if (previous_zone.empty()) {
            person_current_zones_.erase(person_id);
        }
        
        // Add to unique visitors (exact Python visitor tracking)
        if (session_data_.unique_visitor_ids.find(person_id) == session_data_.unique_visitor_ids.end()) {
            session_data_.unique_visitor_ids.insert(person_id);
            session_data_.total_unique_visitors++;
        }
    }
    
    // Periodic analytics update (exact Python periodic update)
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_update_).count() >= 
        ANALYTICS_UPDATE_INTERVAL_SECONDS) {
        
        // Update zone statistics for all zones
        for (const auto& zone : zones_) {
            update_zone_statistics(zone.name);
            check_peak_occupancy(zone.name);
        }
        
        last_update_ = now;
    }
    
    // Periodic cleanup (exact Python cleanup)
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_cleanup_).count() >= cleanup_interval_) {
        cleanup_old_data(timestamp);
        last_cleanup_ = now;
    }
}

std::string ZoneAnalyticsCPP::get_person_zone(const cv::Point2f& position) const
{
    // Find which zone contains the position (exact Python zone detection)
    for (const auto& zone : zones_) {
        if (zone.is_active && point_in_zone(position, zone.area)) {
            return zone.name;
        }
    }
    return ""; // Outside all zones
}

bool ZoneAnalyticsCPP::point_in_zone(const cv::Point2f& point, const cv::Rect2f& zone_area) const
{
    // Point in rectangle check (exact Python collision detection)
    return (point.x >= zone_area.x && point.x <= zone_area.x + zone_area.width &&
            point.y >= zone_area.y && point.y <= zone_area.y + zone_area.height);
}

void ZoneAnalyticsCPP::handle_zone_entry(int person_id, const std::string& zone_name,
                                         const cv::Point2f& position, double timestamp)
{
    auto now = std::chrono::steady_clock::now();
    
    // Record entry time (exact Python entry recording)
    person_zone_entry_times_[person_id] = now;
    
    // Update zone occupancy (exact Python occupancy update)
    if (session_data_.zone_stats.find(zone_name) == session_data_.zone_stats.end()) {
        session_data_.zone_stats[zone_name] = ZoneOccupancyCPP{};
    }
    
    ZoneOccupancyCPP& zone_stats = session_data_.zone_stats[zone_name];
    zone_stats.current_occupants.insert(person_id);
    zone_stats.total_entries++;
    zone_stats.last_activity = now;
    
    std::cout << "Zone Entry: Person " << person_id << " entered " << zone_name << std::endl;
}

void ZoneAnalyticsCPP::handle_zone_exit(int person_id, const std::string& zone_name,
                                       const cv::Point2f& position, double timestamp)
{
    auto now = std::chrono::steady_clock::now();
    
    // Calculate dwell time (exact Python dwell calculation)
    double dwell_time = 0.0;
    if (person_zone_entry_times_.find(person_id) != person_zone_entry_times_.end()) {
        auto entry_time = person_zone_entry_times_[person_id];
        dwell_time = std::chrono::duration<double>(now - entry_time).count();
        person_zone_entry_times_.erase(person_id);
    }
    
    // Update zone occupancy (exact Python occupancy update)
    if (session_data_.zone_stats.find(zone_name) != session_data_.zone_stats.end()) {
        ZoneOccupancyCPP& zone_stats = session_data_.zone_stats[zone_name];
        zone_stats.current_occupants.erase(person_id);
        zone_stats.total_exits++;
        zone_stats.total_dwell_time += dwell_time;
        zone_stats.last_activity = now;
        
        // Update average dwell time (exact Python average calculation)
        if (zone_stats.total_exits > 0) {
            zone_stats.average_dwell_time = zone_stats.total_dwell_time / zone_stats.total_exits;
        }
    }
    
    // Record visit history (exact Python history recording)
    PersonZoneHistoryCPP visit;
    visit.person_id = person_id;
    visit.zone_name = zone_name;
    visit.exit_time = now;
    visit.dwell_time_seconds = dwell_time;
    visit.activity_during_visit = (person_current_activities_.find(person_id) != person_current_activities_.end()) 
                                 ? person_current_activities_[person_id] : "Unknown";
    visit.exit_point = position;
    
    // Find corresponding entry point
    if (person_last_positions_.find(person_id) != person_last_positions_.end()) {
        visit.entry_point = person_last_positions_[person_id];
    }
    
    session_data_.visit_history.push_back(visit);
    
    // Trim history if too large (exact Python history trimming)
    while (session_data_.visit_history.size() > static_cast<size_t>(max_history_entries_)) {
        session_data_.visit_history.erase(session_data_.visit_history.begin());
    }
    
    std::cout << "Zone Exit: Person " << person_id << " exited " << zone_name 
              << " (dwell time: " << std::fixed << std::setprecision(1) << dwell_time << "s)" << std::endl;
}

void ZoneAnalyticsCPP::handle_zone_transition(int person_id, const std::string& from_zone,
                                             const std::string& to_zone, const cv::Point2f& position,
                                             double timestamp)
{
    // Record transition (exact Python transition recording)
    ZoneTransitionCPP transition;
    transition.person_id = person_id;
    transition.from_zone = from_zone;
    transition.to_zone = to_zone;
    transition.transition_time = std::chrono::steady_clock::now();
    transition.transition_point = position;
    
    session_data_.transition_history.push_back(transition);
    
    // Update transition matrix (exact Python matrix update)
    update_transition_matrix(from_zone, to_zone);
    
    std::cout << "Zone Transition: Person " << person_id << " moved from " 
              << from_zone << " to " << to_zone << std::endl;
}

void ZoneAnalyticsCPP::update_zone_statistics(const std::string& zone_name)
{
    if (session_data_.zone_stats.find(zone_name) == session_data_.zone_stats.end()) {
        return;
    }
    
    ZoneOccupancyCPP& stats = session_data_.zone_stats[zone_name];
    
    // Update current occupancy count (exact Python occupancy calculation)
    int current_count = stats.current_occupants.size();
    
    // Update peak occupancy (exact Python peak tracking)
    if (current_count > stats.peak_occupancy) {
        stats.peak_occupancy = current_count;
        stats.peak_time = std::chrono::steady_clock::now();
    }
}

void ZoneAnalyticsCPP::check_peak_occupancy(const std::string& zone_name)
{
    if (session_data_.zone_stats.find(zone_name) != session_data_.zone_stats.end()) {
        update_zone_statistics(zone_name);
    }
}

void ZoneAnalyticsCPP::update_transition_matrix(const std::string& from_zone, const std::string& to_zone)
{
    // Update transition counts (exact Python matrix update)
    session_data_.transition_matrix[from_zone][to_zone]++;
}

std::set<int> ZoneAnalyticsCPP::get_zone_occupants(const std::string& zone_name) const
{
    auto it = session_data_.zone_stats.find(zone_name);
    return (it != session_data_.zone_stats.end()) ? it->second.current_occupants : std::set<int>{};
}

int ZoneAnalyticsCPP::get_zone_occupancy_count(const std::string& zone_name) const
{
    return get_zone_occupants(zone_name).size();
}

ZoneOccupancyCPP ZoneAnalyticsCPP::get_zone_statistics(const std::string& zone_name) const
{
    auto it = session_data_.zone_stats.find(zone_name);
    return (it != session_data_.zone_stats.end()) ? it->second : ZoneOccupancyCPP{};
}

std::map<std::string, int> ZoneAnalyticsCPP::get_all_occupancy_counts() const
{
    std::map<std::string, int> counts;
    for (const auto& pair : session_data_.zone_stats) {
        counts[pair.first] = pair.second.current_occupants.size();
    }
    return counts;
}

std::vector<PersonZoneHistoryCPP> ZoneAnalyticsCPP::get_person_history(int person_id) const
{
    std::vector<PersonZoneHistoryCPP> person_history;
    
    // Filter history for specific person (exact Python filtering)
    for (const auto& visit : session_data_.visit_history) {
        if (visit.person_id == person_id) {
            person_history.push_back(visit);
        }
    }
    
    return person_history;
}

std::vector<ZoneTransitionCPP> ZoneAnalyticsCPP::get_zone_transitions() const
{
    return session_data_.transition_history;
}

std::map<std::string, std::map<std::string, int>> ZoneAnalyticsCPP::get_transition_matrix() const
{
    return session_data_.transition_matrix;
}

double ZoneAnalyticsCPP::get_average_dwell_time(const std::string& zone_name) const
{
    auto it = session_data_.zone_stats.find(zone_name);
    return (it != session_data_.zone_stats.end()) ? it->second.average_dwell_time : 0.0;
}

int ZoneAnalyticsCPP::get_peak_occupancy(const std::string& zone_name) const
{
    auto it = session_data_.zone_stats.find(zone_name);
    return (it != session_data_.zone_stats.end()) ? it->second.peak_occupancy : 0;
}

ZoneAnalyticsSessionCPP ZoneAnalyticsCPP::get_session_summary() const
{
    return session_data_;
}

void ZoneAnalyticsCPP::reset_session()
{
    // Clear all session data (exact Python reset)
    session_data_.zone_stats.clear();
    session_data_.visit_history.clear();
    session_data_.transition_history.clear();
    session_data_.transition_matrix.clear();
    session_data_.unique_visitor_ids.clear();
    session_data_.total_unique_visitors = 0;
    session_data_.session_start = std::chrono::steady_clock::now();
    
    // Clear tracking state
    person_current_zones_.clear();
    person_zone_entry_times_.clear();
    person_last_positions_.clear();
    person_current_activities_.clear();
    
    std::cout << "Zone analytics session reset" << std::endl;
}

void ZoneAnalyticsCPP::add_zone(const ZoneDefinitionCPP& zone)
{
    // Add new zone (exact Python zone addition)
    zones_.push_back(zone);
    zone_name_to_index_[zone.name] = zones_.size() - 1;
    
    // Initialize zone statistics
    session_data_.zone_stats[zone.name] = ZoneOccupancyCPP{};
}

void ZoneAnalyticsCPP::remove_zone(const std::string& zone_name)
{
    // Remove zone (exact Python zone removal)
    auto it = std::find_if(zones_.begin(), zones_.end(),
                          [&zone_name](const ZoneDefinitionCPP& zone) {
                              return zone.name == zone_name;
                          });
    
    if (it != zones_.end()) {
        zones_.erase(it);
        zone_name_to_index_.erase(zone_name);
        session_data_.zone_stats.erase(zone_name);
    }
}

void ZoneAnalyticsCPP::set_zones(const std::vector<ZoneDefinitionCPP>& zones)
{
    zones_ = zones;
    zone_name_to_index_.clear();
    
    // Build name to index mapping (exact Python mapping)
    for (size_t i = 0; i < zones_.size(); ++i) {
        zone_name_to_index_[zones_[i].name] = i;
        
        // Initialize zone statistics if not exists
        if (session_data_.zone_stats.find(zones_[i].name) == session_data_.zone_stats.end()) {
            session_data_.zone_stats[zones_[i].name] = ZoneOccupancyCPP{};
        }
    }
}

std::vector<ZoneDefinitionCPP> ZoneAnalyticsCPP::get_zones() const
{
    return zones_;
}

void ZoneAnalyticsCPP::set_alert_conditions(const std::vector<AlertCondition>& conditions)
{
    alert_conditions_ = conditions;
}

std::vector<std::string> ZoneAnalyticsCPP::check_alerts() const
{
    std::vector<std::string> active_alerts;
    
    // Check each alert condition (exact Python alert checking)
    for (const auto& condition : alert_conditions_) {
        if (!condition.is_active) {
            continue;
        }
        
        if (condition.condition_type == "occupancy_exceed") {
            for (const auto& zone : zones_) {
                int occupancy = get_zone_occupancy_count(zone.name);
                if (occupancy > condition.threshold_value) {
                    active_alerts.push_back("Occupancy alert in " + zone.name + ": " + 
                                          std::to_string(occupancy) + " people");
                }
            }
        }
        else if (condition.condition_type == "dwell_time_exceed") {
            for (const auto& zone : zones_) {
                double avg_dwell = get_average_dwell_time(zone.name);
                if (avg_dwell > condition.threshold_value) {
                    active_alerts.push_back("Dwell time alert in " + zone.name + ": " + 
                                          std::to_string(avg_dwell) + "s average");
                }
            }
        }
    }
    
    return active_alerts;
}

void ZoneAnalyticsCPP::cleanup_old_data(double current_time)
{
    auto now = std::chrono::steady_clock::now();
    
    // Remove expired entries from person tracking (exact Python cleanup)
    auto person_it = person_zone_entry_times_.begin();
    while (person_it != person_zone_entry_times_.end()) {
        auto age = std::chrono::duration<double>(now - person_it->second).count();
        if (age > transition_timeout_ * 10) { // Extended timeout for cleanup
            person_it = person_zone_entry_times_.erase(person_it);
        } else {
            ++person_it;
        }
    }
    
    // Limit visit history size (exact Python history limiting)
    while (session_data_.visit_history.size() > static_cast<size_t>(max_history_entries_)) {
        session_data_.visit_history.erase(session_data_.visit_history.begin());
    }
    
    // Limit transition history size (exact Python transition limiting)
    while (session_data_.transition_history.size() > static_cast<size_t>(max_history_entries_)) {
        session_data_.transition_history.erase(session_data_.transition_history.begin());
    }
}

bool ZoneAnalyticsCPP::is_valid_zone(const std::string& zone_name) const
{
    return zone_name_to_index_.find(zone_name) != zone_name_to_index_.end();
}

void ZoneAnalyticsCPP::export_analytics(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for analytics export: " << filename << std::endl;
        return;
    }
    
    // Export session summary (exact Python export format)
    file << "Zone Analytics Export\n";
    file << "====================\n\n";
    
    file << "Session Summary:\n";
    file << "Total Unique Visitors: " << session_data_.total_unique_visitors << "\n";
    file << "Total Zones: " << zones_.size() << "\n\n";
    
    // Export zone statistics
    file << "Zone Statistics:\n";
    for (const auto& pair : session_data_.zone_stats) {
        const std::string& zone_name = pair.first;
        const ZoneOccupancyCPP& stats = pair.second;
        
        file << "Zone: " << zone_name << "\n";
        file << "  Current Occupancy: " << stats.current_occupants.size() << "\n";
        file << "  Total Entries: " << stats.total_entries << "\n";
        file << "  Total Exits: " << stats.total_exits << "\n";
        file << "  Average Dwell Time: " << std::fixed << std::setprecision(2) 
             << stats.average_dwell_time << "s\n";
        file << "  Peak Occupancy: " << stats.peak_occupancy << "\n\n";
    }
    
    file.close();
    std::cout << "Analytics exported to: " << filename << std::endl;
}

// Factory functions
std::vector<ZoneDefinitionCPP> create_default_zones(const cv::Size& frame_size)
{
    std::vector<ZoneDefinitionCPP> zones;
    
    // Create default zones covering different areas (exact Python defaults)
    float width = frame_size.width;
    float height = frame_size.height;
    
    // Entry zone
    zones.push_back(create_zone("Entry", 
                               cv::Rect2f(0, 0, width * 0.3f, height),
                               cv::Scalar(0, 255, 0), // Green
                               "Main entrance area"));
    
    // Center zone  
    zones.push_back(create_zone("Center",
                               cv::Rect2f(width * 0.3f, height * 0.2f, width * 0.4f, height * 0.6f),
                               cv::Scalar(255, 0, 0), // Blue
                               "Central activity area"));
    
    // Exit zone
    zones.push_back(create_zone("Exit",
                               cv::Rect2f(width * 0.7f, 0, width * 0.3f, height),
                               cv::Scalar(0, 0, 255), // Red
                               "Exit area"));
    
    return zones;
}

ZoneDefinitionCPP create_zone(const std::string& name, const cv::Rect2f& area,
                             const cv::Scalar& color, const std::string& description)
{
    ZoneDefinitionCPP zone;
    zone.name = name;
    zone.area = area;
    zone.color = color;
    zone.description = description;
    zone.is_active = true;
    return zone;
}

} // namespace ORB_SLAM3