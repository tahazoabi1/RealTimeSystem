/**
 * ZoneAnalyticsCPP.h
 * Exact port of modules/zone_analytics.py
 * Zone-based analytics including occupancy, dwell time, and traffic flow
 */

#ifndef ZONE_ANALYTICS_CPP_H
#define ZONE_ANALYTICS_CPP_H

#include "orbslam3_export.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <chrono>

namespace ORB_SLAM3 {

struct ZoneDefinitionCPP {
    std::string name;
    cv::Rect2f area;
    cv::Scalar color;
    std::string description;
    bool is_active;
};

struct ZoneOccupancyCPP {
    std::set<int> current_occupants;
    int total_entries;
    int total_exits;
    double total_dwell_time;
    double average_dwell_time;
    int peak_occupancy;
    std::chrono::steady_clock::time_point peak_time;
    std::chrono::steady_clock::time_point last_activity;
};

struct PersonZoneHistoryCPP {
    int person_id;
    std::string zone_name;
    std::chrono::steady_clock::time_point entry_time;
    std::chrono::steady_clock::time_point exit_time;
    double dwell_time_seconds;
    std::string activity_during_visit;
    cv::Point2f entry_point;
    cv::Point2f exit_point;
};

struct ZoneTransitionCPP {
    int person_id;
    std::string from_zone;
    std::string to_zone;
    std::chrono::steady_clock::time_point transition_time;
    cv::Point2f transition_point;
};

struct ZoneAnalyticsSessionCPP {
    std::map<std::string, ZoneOccupancyCPP> zone_stats;
    std::vector<PersonZoneHistoryCPP> visit_history;
    std::vector<ZoneTransitionCPP> transition_history;
    std::map<std::string, std::map<std::string, int>> transition_matrix;
    std::chrono::steady_clock::time_point session_start;
    int total_unique_visitors;
    std::set<int> unique_visitor_ids;
};

class ORB_SLAM3_API ZoneAnalyticsCPP {
public:
    ZoneAnalyticsCPP();
    ~ZoneAnalyticsCPP();
    
    // Exact port of Python ZoneAnalytics methods
    bool initialize(const std::vector<ZoneDefinitionCPP>& zones = {});
    void update_person_positions(const std::vector<int>& person_ids, 
                                const std::vector<cv::Point2f>& positions,
                                const std::vector<std::string>& activities,
                                double timestamp);
    
    // Zone management methods (exact Python interface)
    void add_zone(const ZoneDefinitionCPP& zone);
    void remove_zone(const std::string& zone_name);
    void set_zones(const std::vector<ZoneDefinitionCPP>& zones);
    std::vector<ZoneDefinitionCPP> get_zones() const;
    
    // Occupancy analysis methods (exact Python interface)
    std::set<int> get_zone_occupants(const std::string& zone_name) const;
    int get_zone_occupancy_count(const std::string& zone_name) const;
    ZoneOccupancyCPP get_zone_statistics(const std::string& zone_name) const;
    std::map<std::string, int> get_all_occupancy_counts() const;
    
    // Analytics methods (exact Python analytics)
    std::vector<PersonZoneHistoryCPP> get_person_history(int person_id) const;
    std::vector<ZoneTransitionCPP> get_zone_transitions() const;
    std::map<std::string, std::map<std::string, int>> get_transition_matrix() const;
    double get_average_dwell_time(const std::string& zone_name) const;
    int get_peak_occupancy(const std::string& zone_name) const;
    
    // Session management (exact Python session interface)
    ZoneAnalyticsSessionCPP get_session_summary() const;
    void reset_session();
    void export_analytics(const std::string& filename) const;
    void cleanup_old_data(double current_time);
    
    // Real-time alerts (exact Python alert system)
    struct AlertCondition {
        std::string zone_name;
        std::string condition_type; // "occupancy_exceed", "dwell_time_exceed", "rapid_entry_exit"
        float threshold_value;
        bool is_active;
    };
    
    void set_alert_conditions(const std::vector<AlertCondition>& conditions);
    std::vector<std::string> check_alerts() const;
    
    // Configuration methods
    void set_dwell_time_threshold(double threshold_seconds) { dwell_time_threshold_ = threshold_seconds; }
    void set_transition_timeout(double timeout_seconds) { transition_timeout_ = timeout_seconds; }
    bool is_available() const { return initialized_; }
    
private:
    // Core zone analysis methods
    std::string get_person_zone(const cv::Point2f& position) const;
    bool point_in_zone(const cv::Point2f& point, const cv::Rect2f& zone_area) const;
    void handle_zone_entry(int person_id, const std::string& zone_name, 
                          const cv::Point2f& position, double timestamp);
    void handle_zone_exit(int person_id, const std::string& zone_name, 
                         const cv::Point2f& position, double timestamp);
    void handle_zone_transition(int person_id, const std::string& from_zone, 
                               const std::string& to_zone, const cv::Point2f& position, 
                               double timestamp);
    
    // Analytics calculation methods
    void update_zone_statistics(const std::string& zone_name);
    void calculate_dwell_times();
    void update_transition_matrix(const std::string& from_zone, const std::string& to_zone);
    void check_peak_occupancy(const std::string& zone_name);
    
    // Data management methods
    void cleanup_expired_visits(double current_time);
    void validate_zone_data();
    bool is_valid_zone(const std::string& zone_name) const;
    
private:
    // Zone definitions (exact Python zone system)
    std::vector<ZoneDefinitionCPP> zones_;
    std::map<std::string, int> zone_name_to_index_;
    
    // Current tracking state (exact Python tracking state)
    std::map<int, std::string> person_current_zones_;
    std::map<int, std::chrono::steady_clock::time_point> person_zone_entry_times_;
    std::map<int, cv::Point2f> person_last_positions_;
    std::map<int, std::string> person_current_activities_;
    
    // Analytics data (exact Python analytics structures)
    ZoneAnalyticsSessionCPP session_data_;
    std::vector<AlertCondition> alert_conditions_;
    
    // Configuration (exact Python defaults)
    double dwell_time_threshold_;
    double transition_timeout_;
    double cleanup_interval_;
    int max_history_entries_;
    
    // State
    bool initialized_;
    std::chrono::steady_clock::time_point last_cleanup_;
    std::chrono::steady_clock::time_point last_update_;
    
    // Constants (exact Python values)
    static const double DEFAULT_DWELL_THRESHOLD_SECONDS;
    static const double DEFAULT_TRANSITION_TIMEOUT_SECONDS;
    static const double DEFAULT_CLEANUP_INTERVAL_SECONDS;
    static const int DEFAULT_MAX_HISTORY_ENTRIES;
    static const int ANALYTICS_UPDATE_INTERVAL_SECONDS;
};

// Factory functions for creating default zones
ORB_SLAM3_API std::vector<ZoneDefinitionCPP> create_default_zones(const cv::Size& frame_size);
ORB_SLAM3_API ZoneDefinitionCPP create_zone(const std::string& name, const cv::Rect2f& area, 
                             const cv::Scalar& color, const std::string& description = "");

} // namespace ORB_SLAM3

#endif // ZONE_ANALYTICS_CPP_H