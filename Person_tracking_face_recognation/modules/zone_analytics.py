"""
Zone Analytics and Tracking Module
Handles zone-based analytics, occupancy tracking, and alerts
"""

import time
from datetime import datetime
import numpy as np


class ZoneAnalytics:
    """Manages zone-based analytics and occupancy tracking"""
    
    def __init__(self, zones=None):
        self.zones = zones or self.create_default_zones()
        
        # Zone analytics data
        self.zone_analytics = {}
        self.zone_entry_times = {}
        self.zone_alerts = []
        
        # Initialize analytics for each zone
        for i, zone in enumerate(self.zones):
            zone_name = zone['name']
            self.zone_analytics[zone_name] = {
                'current_occupancy': set(),
                'total_visits': 0,
                'total_time_spent': 0,
                'average_dwell_time': 0,
                'peak_occupancy': 0,
                'last_entry_time': None
            }
    
    def create_default_zones(self):
        """Create default zone configuration - empty by default"""
        return []  # No default zones
    
    def setup_zones_from_config(self, config_zones):
        """Setup zones from configuration"""
        try:
            self.zones = []
            for zone_config in config_zones:
                zone = {
                    'name': zone_config.get('name', 'Zone'),
                    'bbox': zone_config.get('bbox', [0, 0, 100, 100]),
                    'color': zone_config.get('color', [255, 255, 255])
                }
                self.zones.append(zone)
            
            # Reinitialize analytics
            self.zone_analytics = {}
            for zone in self.zones:
                zone_name = zone['name']
                self.zone_analytics[zone_name] = {
                    'current_occupancy': set(),
                    'total_visits': 0,
                    'total_time_spent': 0,
                    'average_dwell_time': 0,
                    'peak_occupancy': 0,
                    'last_entry_time': None
                }
                
        except Exception as e:
            print(f"⚠️ Zone setup error: {e}")
            self.zones = self.create_default_zones()
    
    def point_in_zone(self, point, zone_bbox):
        """Check if point is inside zone"""
        try:
            x, y = point
            zx, zy, zw, zh = zone_bbox
            return zx <= x <= zx + zw and zy <= y <= zy + zh
        except Exception:
            return False
    
    def update_zone_analytics(self, active_tracks):
        """Update zone analytics for all active tracks"""
        try:
            current_time = time.time()
            
            for i, zone in enumerate(self.zones):
                zone_name = zone['name']
                zone_bbox = zone['bbox']
                
                # Find tracks currently in this zone
                current_occupants = set()
                
                for track_id, track_data in active_tracks.items():
                    if 'bbox' not in track_data:
                        continue
                        
                    # Calculate center point of bounding box
                    x1, y1, x2, y2 = track_data['bbox']
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    if self.point_in_zone((center_x, center_y), zone_bbox):
                        current_occupants.add(track_id)
                
                # Update analytics
                analytics = self.zone_analytics[zone_name]
                previous_occupants = analytics['current_occupancy']
                
                # Track entries and exits
                new_entries = current_occupants - previous_occupants
                exits = previous_occupants - current_occupants
                
                # Handle new entries
                for track_id in new_entries:
                    analytics['total_visits'] += 1
                    analytics['last_entry_time'] = current_time
                    entry_key = f"{zone_name}_{track_id}"
                    self.zone_entry_times[entry_key] = current_time
                    
                    # Create alert for entry
                    self.zone_alerts.append({
                        'timestamp': datetime.now(),
                        'zone': zone_name,
                        'track_id': track_id,
                        'event': 'entry',
                        'occupancy': len(current_occupants)
                    })
                
                # Handle exits (calculate dwell time)
                for track_id in exits:
                    entry_key = f"{zone_name}_{track_id}"
                    if entry_key in self.zone_entry_times:
                        dwell_time = current_time - self.zone_entry_times[entry_key]
                        analytics['total_time_spent'] += dwell_time
                        
                        # Update average dwell time
                        if analytics['total_visits'] > 0:
                            analytics['average_dwell_time'] = analytics['total_time_spent'] / analytics['total_visits']
                        
                        # Clean up entry time record
                        del self.zone_entry_times[entry_key]
                        
                        # Create alert for exit
                        self.zone_alerts.append({
                            'timestamp': datetime.now(),
                            'zone': zone_name,
                            'track_id': track_id,
                            'event': 'exit',
                            'dwell_time': dwell_time,
                            'occupancy': len(current_occupants)
                        })
                
                # Update current occupancy and peak
                analytics['current_occupancy'] = current_occupants
                if len(current_occupants) > analytics['peak_occupancy']:
                    analytics['peak_occupancy'] = len(current_occupants)
            
            # Clean up old alerts (keep last 50)
            while len(self.zone_alerts) > 50:
                self.zone_alerts.pop(0)
                
        except Exception as e:
            print(f"⚠️ Zone analytics update error: {e}")
    
    def get_zone_status(self):
        """Get current status of all zones"""
        status = {}
        for zone_name, analytics in self.zone_analytics.items():
            status[zone_name] = {
                'current_occupancy': len(analytics['current_occupancy']),
                'occupant_ids': list(analytics['current_occupancy']),
                'total_visits': analytics['total_visits'],
                'average_dwell_time': analytics['average_dwell_time'],
                'peak_occupancy': analytics['peak_occupancy']
            }
        return status
    
    def get_recent_alerts(self, limit=10):
        """Get recent zone alerts"""
        return self.zone_alerts[-limit:] if self.zone_alerts else []
    
    def get_zone_summary(self):
        """Get comprehensive zone analytics summary"""
        summary = {
            'total_zones': len(self.zones),
            'total_alerts': len(self.zone_alerts),
            'zone_details': {}
        }
        
        for zone_name, analytics in self.zone_analytics.items():
            summary['zone_details'][zone_name] = {
                'current_occupancy': len(analytics['current_occupancy']),
                'total_visits': analytics['total_visits'],
                'total_time_spent': analytics['total_time_spent'],
                'average_dwell_time': analytics['average_dwell_time'],
                'peak_occupancy': analytics['peak_occupancy'],
                'activity_level': 'High' if analytics['total_visits'] > 10 else 'Medium' if analytics['total_visits'] > 3 else 'Low'
            }
        
        return summary
    
    def cleanup_track_zones(self, track_id):
        """Clean up zone data for removed track"""
        try:
            # Remove from current occupancy
            for analytics in self.zone_analytics.values():
                analytics['current_occupancy'].discard(track_id)
            
            # Clean up entry times
            keys_to_remove = [k for k in self.zone_entry_times.keys() if k.endswith(f"_{track_id}")]
            for key in keys_to_remove:
                del self.zone_entry_times[key]
                
        except Exception as e:
            print(f"⚠️ Zone cleanup error for track {track_id}: {e}")
    
    def export_analytics_data(self):
        """Export zone analytics data for external analysis"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'zones': self.zones,
                'analytics': {},
                'recent_alerts': self.get_recent_alerts(20)
            }
            
            # Convert sets to lists for JSON serialization
            for zone_name, analytics in self.zone_analytics.items():
                export_data['analytics'][zone_name] = {
                    'current_occupancy': list(analytics['current_occupancy']),
                    'total_visits': analytics['total_visits'],
                    'total_time_spent': analytics['total_time_spent'],
                    'average_dwell_time': analytics['average_dwell_time'],
                    'peak_occupancy': analytics['peak_occupancy'],
                    'last_entry_time': analytics['last_entry_time']
                }
            
            return export_data
            
        except Exception as e:
            print(f"⚠️ Analytics export error: {e}")
            return {}
    
    def reset_analytics(self):
        """Reset all zone analytics data"""
        try:
            for analytics in self.zone_analytics.values():
                analytics.update({
                    'current_occupancy': set(),
                    'total_visits': 0,
                    'total_time_spent': 0,
                    'average_dwell_time': 0,
                    'peak_occupancy': 0,
                    'last_entry_time': None
                })
            
            self.zone_entry_times.clear()
            self.zone_alerts.clear()
            
            print("✅ Zone analytics reset complete")
            
        except Exception as e:
            print(f"⚠️ Analytics reset error: {e}")
    
    def detect_anomalies(self):
        """Detect anomalous patterns in zone activity"""
        anomalies = []
        
        try:
            current_time = time.time()
            
            for zone_name, analytics in self.zone_analytics.items():
                # Check for overcrowding
                if len(analytics['current_occupancy']) > 5:  # Configurable threshold
                    anomalies.append({
                        'type': 'overcrowding',
                        'zone': zone_name,
                        'occupancy': len(analytics['current_occupancy']),
                        'timestamp': current_time
                    })
                
                # Check for unusually long dwell times (over 5 minutes)
                for entry_key, entry_time in self.zone_entry_times.items():
                    if entry_key.startswith(zone_name):
                        dwell_time = current_time - entry_time
                        if dwell_time > 300:  # 5 minutes
                            track_id = entry_key.split('_')[-1]
                            anomalies.append({
                                'type': 'long_dwell',
                                'zone': zone_name,
                                'track_id': track_id,
                                'dwell_time': dwell_time,
                                'timestamp': current_time
                            })
            
        except Exception as e:
            print(f"⚠️ Anomaly detection error: {e}")
        
        return anomalies
    
    def get_heatmap_data(self):
        """Get data for zone activity heatmap"""
        heatmap_data = {}
        
        for zone_name, analytics in self.zone_analytics.items():
            # Normalize activity level (0-1 scale)
            max_visits = max([a['total_visits'] for a in self.zone_analytics.values()]) or 1
            activity_intensity = analytics['total_visits'] / max_visits
            
            heatmap_data[zone_name] = {
                'intensity': activity_intensity,
                'visits': analytics['total_visits'],
                'current_occupancy': len(analytics['current_occupancy']),
                'bbox': None  # Will be filled from zone configuration
            }
            
            # Find corresponding zone bbox
            for zone in self.zones:
                if zone['name'] == zone_name:
                    heatmap_data[zone_name]['bbox'] = zone['bbox']
                    break
        
        return heatmap_data