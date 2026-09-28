"""
AgriNode AI - Environmental Risk Intelligence
Phase 4: Flood Risk Proxy Engine (src/flood.py)

NOTE: This engine measures acute excess-precipitation stress (soil saturation,
surface-runoff proxy) using district-level daily rain records. It does NOT
incorporate topographic, drainage, or reservoir-level data and is therefore
explicitly labelled a **Flood Risk PROXY**.

Scoring Formula (continuous 0–100):
  flood_score = 50% * acute_intensity_score   (peak 1-day rain vs baselines)
              + 30% * accumulation_score       (7-day rain accumulation)
              + 20% * frequency_score          (heavy-rain days in past 7 days)

Standardized Risk Tiers:
  0–30   → LOW
  31–60  → MODERATE
  61–80  → HIGH
  81–100 → SEVERE
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


TIER_BOUNDARIES = [(81, "SEVERE"), (61, "HIGH"), (31, "MODERATE"), (0, "LOW")]


def score_to_tier(score: float) -> str:
    for threshold, label in TIER_BOUNDARIES:
        if score >= threshold:
            return label
    return "LOW"


def _acute_intensity_score(
    rain_1d: float,
    rain_p90: float,
    rain_p95: float,
    rain_p99: float,
) -> float:
    """
    Score the peak single-day rainfall relative to local historical percentile thresholds.
    """
    if rain_p99 <= 0:
        return float(np.clip(rain_1d / 10.0 * 60.0, 0, 100))

    anchors = [
        (0.0,            0.0),
        (rain_p90 * 0.7, 20.0),
        (rain_p90,       45.0),
        (rain_p95,       70.0),   # >= 95th percentile is HIGH acute stress
        (rain_p99,       90.0),   # >= 99th percentile is SEVERE acute stress
        (rain_p99 * 1.5, 100.0),
    ]
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    score = float(np.interp(rain_1d, xs, ys))
    return float(np.clip(score, 0, 100))


def _accumulation_score(rain_7d: float, rain_p90: float) -> float:
    """
    Score 7-day cumulative rainfall relative to baseline 7-day threshold.
    """
    baseline_7d = max(rain_p90 * 7.0, 15.0)
    ratio = rain_7d / baseline_7d
    anchors = [(0.0, 0.0), (0.4, 20.0), (0.8, 50.0), (1.2, 75.0), (1.8, 90.0), (2.5, 100.0)]
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    score = float(np.interp(ratio, xs, ys))
    return float(np.clip(score, 0, 100))


def _frequency_score(heavy_rain_days_7d: int) -> float:
    """
    Map count of heavy-rain days (> local p90) in past 7 days to 0–100.
    """
    anchors = [(0, 0.0), (1, 30.0), (2, 50.0), (3, 70.0), (5, 90.0), (7, 100.0)]
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    score = float(np.interp(heavy_rain_days_7d, xs, ys))
    return float(np.clip(score, 0, 100))


def compute_flood_risk(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute flood risk proxy score and tier from feature vector.
    """
    rain_1d            = features.get("rain_1d", 0.0)
    rain_7d            = features.get("rain_7d", 0.0)
    rain_p90           = features.get("baseline_rain_p90", 0.0)
    rain_p95           = features.get("baseline_rain_p95", 0.0)
    rain_p99           = features.get("baseline_rain_p99", 0.0)
    heavy_rain_days_7d = features.get("heavy_rain_days_7d", 0)

    s_intensity    = _acute_intensity_score(rain_1d, rain_p90, rain_p95, rain_p99)
    s_accumulation = _accumulation_score(rain_7d, rain_p90)
    s_frequency    = _frequency_score(heavy_rain_days_7d)

    flood_score = round(0.50 * s_intensity + 0.30 * s_accumulation + 0.20 * s_frequency, 2)
    flood_tier  = score_to_tier(flood_score)

    return {
        "flood_score": flood_score,
        "flood_tier": flood_tier,
        "flood_factors": {
            "rain_1d_mm": rain_1d,
            "rain_7d_mm": rain_7d,
            "heavy_rain_days_7d": heavy_rain_days_7d,
            "baseline_rain_p90": rain_p90,
            "baseline_rain_p95": rain_p95,
            "baseline_rain_p99": rain_p99,
            "sub_score_acute_intensity": round(s_intensity, 2),
            "sub_score_accumulation": round(s_accumulation, 2),
            "sub_score_frequency": round(s_frequency, 2),
            "weights": "50% acute intensity + 30% 7d accumulation + 20% heavy-rain frequency",
            "note": "Flood Risk PROXY — excludes topographic, drainage, and reservoir data.",
        }
    }
