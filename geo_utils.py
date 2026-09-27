# geo_utils.py

import math


EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate approximate great-circle distance
    between two latitude/longitude points.
    """

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return EARTH_RADIUS_KM * c


def find_nearest_event(
    target_event: dict,
    events: list[dict],
    event_type: str | None = None,
):
    """
    Find the nearest event to target_event.

    The target event itself is excluded.
    """

    candidates = []

    for event in events:

        if event["event_id"] == target_event["event_id"]:
            continue

        if event_type is not None:
            if event.get("event_type") != event_type:
                continue

        distance = haversine_distance_km(
            target_event["latitude"],
            target_event["longitude"],
            event["latitude"],
            event["longitude"],
        )

        candidates.append(
            (distance, event)
        )

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0]