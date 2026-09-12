# map_data.py

from datetime import datetime


events = [
    {
        "event_id": 1,
        "event_type": "traffic_density",
        "traffic_level": "HIGH",
        "vehicle_count": 14,
        "latitude": 13.3161,
        "longitude": 75.7720,
        "location": "Chikkamagaluru",
        "timestamp": "2026-09-11 10:15:00",
        "source": "recorded-road-video",
        "estimated_congestion_km": 0.80,
        "reporting_buses": 3,
    },
    {
        "event_id": 2,
        "event_type": "traffic_density",
        "traffic_level": "MEDIUM",
        "vehicle_count": 9,
        "latitude": 13.3215,
        "longitude": 75.7845,
        "location": "Chikkamagaluru",
        "timestamp": "2026-09-11 10:18:00",
        "source": "recorded-road-video",
        "estimated_congestion_km": 0.45,
        "reporting_buses": 2,
    },
    {
        "event_id": 3,
        "event_type": "traffic_density",
        "traffic_level": "LOW",
        "vehicle_count": 4,
        "latitude": 13.3098,
        "longitude": 75.7642,
        "location": "Chikkamagaluru",
        "timestamp": "2026-09-11 10:20:00",
        "source": "recorded-road-video",
        "estimated_congestion_km": 0.20,
        "reporting_buses": 1,
    },

    # Separate road-hazard demonstration event
    {
        "event_id": 100,
        "event_type": "road_hazard",
        "hazard_type": "pothole",
        "pothole_count": 2,
        "highest_confidence": 0.764,
        "severity": "HIGH",
        "priority": "HIGH",
        "latitude": 13.3180,
        "longitude": 75.7780,
        "location": "Chikkamagaluru",
        "timestamp": "2026-09-11 10:22:00",
        "source": "separate-pothole-validation",
        "estimated_congestion_km": 0.80,
        "reporting_buses": 3,
    },
]