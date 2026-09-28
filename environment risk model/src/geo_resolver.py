"""
AgriNode AI - Environmental Risk Intelligence
Phase 2: Geographic Coordinate Resolver

Resolves arbitrary (latitude, longitude) coordinates to the nearest Indian district
using a 3D spherical KDTree in sub-millisecond runtime (<1 ms).
Guarantees zero GDAL/GeoPandas dependency issues on Windows.
"""

import math
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree

CENTROIDS_PATH = Path("data/processed/district_centroids.csv")
EARTH_RADIUS_KM = 6371.0088


def latlon_to_cartesian(lat_deg: float, lon_deg: float):
    """Convert (latitude, longitude) in degrees to 3D Cartesian coordinates on unit sphere."""
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi)
    return x, y, z


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in kilometers."""
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)
    dphi = phi2 - phi1
    dlam = lam2 - lam1

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


class GeoResolver:
    """Instant KDTree-based geographic resolver for Indian districts."""

    def __init__(self, centroids_csv: Path = CENTROIDS_PATH):
        if not centroids_csv.exists():
            raise FileNotFoundError(f"Centroids file not found at: {centroids_csv}")

        self.df = pd.read_csv(centroids_csv)
        self.districts = self.df.to_dict(orient="records")

        # Build 3D coordinates on unit sphere for exact spherical nearest-neighbor search
        lats = self.df["latitude"].values
        lons = self.df["longitude"].values
        xs, ys, zs = latlon_to_cartesian(lats, lons)
        points_3d = np.column_stack([xs, ys, zs])

        self.kdtree = KDTree(points_3d)

    def resolve(self, lat: float, lon: float) -> dict:
        """
        Resolve (lat, lon) to the nearest Indian district.
        Returns district metadata, distance in km, and latency in ms.
        """
        t0 = time.perf_counter()

        # Convert query point to 3D cartesian
        qx, qy, qz = latlon_to_cartesian(lat, lon)
        _, nearest_idx = self.kdtree.query([qx, qy, qz], k=1)

        match = self.districts[nearest_idx]
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        dist_km = haversine_distance(lat, lon, match["latitude"], match["longitude"])

        return {
            "district_id": int(match["district_id"]),
            "district": str(match["district"]),
            "state": str(match["state"]),
            "centroid_lat": float(match["latitude"]),
            "centroid_lon": float(match["longitude"]),
            "distance_km": round(dist_km, 2),
            "latency_ms": round(latency_ms, 3)
        }


# Global singleton instance for high-performance reuse
_RESOLVER_INSTANCE = None


def get_geo_resolver(centroids_csv: Path = CENTROIDS_PATH) -> GeoResolver:
    """Retrieve or initialize singleton GeoResolver."""
    global _RESOLVER_INSTANCE
    if _RESOLVER_INSTANCE is None:
        _RESOLVER_INSTANCE = GeoResolver(centroids_csv)
    return _RESOLVER_INSTANCE


def resolve_district(lat: float, lon: float) -> dict:
    """Convenience function to resolve (lat, lon) in < 1 ms."""
    resolver = get_geo_resolver()
    return resolver.resolve(lat, lon)


if __name__ == "__main__":
    print("Testing GeoResolver initialization & resolution...")
    res = get_geo_resolver()
    
    # Test query: Durgapur (23.52, 87.31)
    q_lat, q_lon = 23.52, 87.31
    info = res.resolve(q_lat, q_lon)
    print(f"Query ({q_lat}, {q_lon}) -> {info['district']}, {info['state']} "
          f"({info['distance_km']} km away, latency: {info['latency_ms']} ms)")

    # Test query: Kochi / Ernakulam (9.98, 76.30)
    info_kerala = res.resolve(9.98, 76.30)
    print(f"Query (9.98, 76.30) -> {info_kerala['district']}, {info_kerala['state']} "
          f"({info_kerala['distance_km']} km away, latency: {info_kerala['latency_ms']} ms)")
