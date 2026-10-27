"""Shared geodesic distance helpers."""

import math

EARTH_M = 6_371_009.0   # mean Earth radius in metres


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodesic distance in metres between two WGS-84 points."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_M * math.asin(math.sqrt(a))


def haversine_km(lat1, lon1, lat2, lon2):
    """Geodesic distance in kilometres; returns None if any argument is None or NaN."""
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in (lat1, lon1, lat2, lon2)):
        return None
    return haversine_m(lat1, lon1, lat2, lon2) / 1000.0
