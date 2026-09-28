"""
AgriNode AI - Environmental Risk Intelligence
Phase 3: Multi-Scale Feature Engineering Module

Extracts multi-scale rolling feature vectors for any (lat, lon, date) or (district_id, date):
- Rainfall Indicators:
  - rain_1d, rain_7d, rain_30d, rain_90d
  - anomaly_30d, anomaly_90d (% departure relative to baseline normals)
  - current_dry_spell (consecutive days with rain < 2.5 mm)
  - max_rainfall_1d_30d (peak 1-day rain in past 30 days)
  - heavy_rain_days_7d (days exceeding local 90th percentile)
- Thermal Indicators:
  - tmax, tmin, tmean
  - tmax_anomaly (°C departure from monthly historical mean)
  - consecutive_hot_days (consecutive days exceeding local 90th percentile threshold)
  - max_tmax_7d (highest Tmax in past 7 days)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geo_resolver import get_geo_resolver, GeoResolver
from src.baseline import get_baseline_calibrator, BaselineCalibrator
from src.preprocessing import (
    clean_and_harmonize_district_data,
    get_district_metadata_map,
    RAW_DISTRICTS_DIR
)


class FeatureExtractor:
    """Extracts rolling multi-hazard feature vectors given spatial and temporal coordinates."""

    def __init__(
        self,
        geo_resolver: Optional[GeoResolver] = None,
        baseline_calibrator: Optional[BaselineCalibrator] = None
    ):
        self.resolver = geo_resolver or get_geo_resolver()
        self.calibrator = baseline_calibrator or get_baseline_calibrator()
        self.meta_map = get_district_metadata_map()
        # In-memory district cache for ultra-fast repeated queries: district_id -> DataFrame
        self._district_cache: Dict[int, pd.DataFrame] = {}

    def _get_district_series(self, district_id: int) -> pd.DataFrame:
        """Fetch cleaned daily series for a district with caching."""
        if district_id not in self._district_cache:
            meta = self.meta_map.get(district_id, {"district": f"District_{district_id}", "state": "Unknown"})
            df = clean_and_harmonize_district_data(district_id, meta)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            self._district_cache[district_id] = df
        return self._district_cache[district_id]

    def extract_features(
        self,
        query_date: Union[str, datetime],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        district_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract complete feature dictionary for either (lat, lon, query_date) or (district_id, query_date).
        """
        start_t = datetime.now()

        # 1. Resolve district if coordinates provided
        if district_id is None:
            if lat is None or lon is None:
                raise ValueError("Must provide either (lat, lon) or district_id")
            resolved = self.resolver.resolve(lat, lon)
            district_id = resolved["district_id"]
            district_name = resolved["district"]
            state_name = resolved["state"]
            dist_km = resolved["distance_km"]
        else:
            meta = self.meta_map.get(district_id, {})
            district_name = meta.get("district", f"District_{district_id}")
            state_name = meta.get("state", "Unknown")
            dist_km = 0.0

        # 2. Parse query date
        dt = pd.to_datetime(query_date)
        month = dt.month

        # 3. Fetch monthly baseline statistics
        baseline = self.calibrator.get_baseline(district_id, month)

        # 4. Slice time series up to query_date (90 days lookback needed)
        df = self._get_district_series(district_id)
        if dt not in df.index:
            raise KeyError(f"Date {dt.strftime('%Y-%m-%d')} not in historical record range (1981-2024).")

        # Slice 90 days of history leading up to dt
        lookback_90d = dt - timedelta(days=89)
        window_df = df.loc[lookback_90d:dt]

        # 5. Compute Precipitation Indicators
        rain_vals_90 = window_df["rainfall"].values
        rain_1d = float(rain_vals_90[-1])
        rain_7d = float(np.sum(rain_vals_90[-7:])) if len(rain_vals_90) >= 7 else float(np.sum(rain_vals_90))
        rain_30d = float(np.sum(rain_vals_90[-30:])) if len(rain_vals_90) >= 30 else float(np.sum(rain_vals_90))
        rain_90d = float(np.sum(rain_vals_90))

        # Deficits relative to baseline normals (%)
        base_30d = max(baseline["baseline_30d_normal"], 5.0)  # Avoid division by zero
        base_90d = max(baseline["baseline_90d_normal"], 15.0)
        anomaly_30d = round(((rain_30d - base_30d) / base_30d) * 100.0, 2)
        anomaly_90d = round(((rain_90d - base_90d) / base_90d) * 100.0, 2)

        # Dry spell length (consecutive days with rain < 2.5 mm up to query_date)
        dry_spell = 0
        for r in reversed(rain_vals_90):
            if r < 2.5:
                dry_spell += 1
            else:
                break

        # Max 1-day rainfall in past 30 days
        rain_30d_slice = rain_vals_90[-30:] if len(rain_vals_90) >= 30 else rain_vals_90
        max_rainfall_1d_30d = float(np.max(rain_30d_slice))

        # Heavy rain days in past 7 days (exceeding local 90th percentile)
        rain_7d_slice = rain_vals_90[-7:] if len(rain_vals_90) >= 7 else rain_vals_90
        heavy_rain_days_7d = int(np.sum(rain_7d_slice >= baseline["rain_p90"]))

        # 6. Compute Thermal Indicators
        tmax_vals_90 = window_df["tmax"].values
        tmin_vals_90 = window_df["tmin"].values
        tmean_vals_90 = window_df["tmean"].values

        tmax_curr = float(tmax_vals_90[-1])
        tmin_curr = float(tmin_vals_90[-1])
        tmean_curr = float(tmean_vals_90[-1])

        tmax_anomaly = round(tmax_curr - baseline["tmax_mean"], 2)

        # Consecutive hot days (exceeding local 90th percentile threshold)
        consecutive_hot_days = 0
        tmax_p90 = baseline["tmax_p90"]
        for t in reversed(tmax_vals_90):
            if t >= tmax_p90:
                consecutive_hot_days += 1
            else:
                break

        # Max Tmax in past 7 days
        tmax_7d_slice = tmax_vals_90[-7:] if len(tmax_vals_90) >= 7 else tmax_vals_90
        max_tmax_7d = float(np.max(tmax_7d_slice))

        calc_latency_ms = (datetime.now() - start_t).total_seconds() * 1000.0

        return {
            # Spatial & Temporal Context
            "district_id": district_id,
            "district": district_name,
            "state": state_name,
            "date": dt.strftime("%Y-%m-%d"),
            "distance_km": round(dist_km, 2),
            "month": month,
            "calc_latency_ms": round(calc_latency_ms, 3),

            # Baselines for context
            "baseline_rain_monthly_mean": baseline["rain_mean"],
            "baseline_rain_p90": baseline["rain_p90"],
            "baseline_rain_p95": baseline["rain_p95"],
            "baseline_rain_p99": baseline["rain_p99"],
            "baseline_tmax_monthly_mean": baseline["tmax_mean"],
            "baseline_tmax_p90": baseline["tmax_p90"],

            # Rainfall Features
            "rain_1d": round(rain_1d, 2),
            "rain_7d": round(rain_7d, 2),
            "rain_30d": round(rain_30d, 2),
            "rain_90d": round(rain_90d, 2),
            "anomaly_30d": anomaly_30d,
            "anomaly_90d": anomaly_90d,
            "current_dry_spell": dry_spell,
            "max_rainfall_1d_30d": round(max_rainfall_1d_30d, 2),
            "heavy_rain_days_7d": heavy_rain_days_7d,

            # Thermal Features
            "tmax": round(tmax_curr, 2),
            "tmin": round(tmin_curr, 2),
            "tmean": round(tmean_curr, 2),
            "tmax_anomaly": tmax_anomaly,
            "consecutive_hot_days": consecutive_hot_days,
            "max_tmax_7d": round(max_tmax_7d, 2)
        }


_FEATURE_EXTRACTOR_INSTANCE: Optional[FeatureExtractor] = None

def get_feature_extractor() -> FeatureExtractor:
    """Singleton accessor for FeatureExtractor."""
    global _FEATURE_EXTRACTOR_INSTANCE
    if _FEATURE_EXTRACTOR_INSTANCE is None:
        _FEATURE_EXTRACTOR_INSTANCE = FeatureExtractor()
    return _FEATURE_EXTRACTOR_INSTANCE


if __name__ == "__main__":
    extractor = get_feature_extractor()
    # Test sample query: Ernakulam 2018 flood peak
    sample = extractor.extract_features(lat=9.9816, lon=76.2999, query_date="2018-08-16")
    print("Sample Extracted Features (Ernakulam Flood 2018):")
    for k, v in sample.items():
        print(f"  {k:30s}: {v}")
