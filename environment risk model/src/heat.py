"""
AgriNode AI - Environmental Risk Intelligence
Phase 4: Heat Stress Engine (src/heat.py)

Scoring Formula (continuous 0–100):
  heat_score = 40% * departure_score       (Tmax anomaly vs local monthly mean)
             + 30% * absolute_score        (raw Tmax vs absolute danger thresholds)
             + 30% * persistence_score     (consecutive hot days above local 90th percentile)

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


def _departure_score(tmax_anomaly: float) -> float:
    """
    Score daily Tmax departure from local monthly historical mean (°C).
      <= 0.0°C -> 0
      +1.5°C   -> 35
      +2.5°C   -> 60 (significant heat alert)
      +4.0°C   -> 80 (severe departure)
      >= +6.0°C -> 100
    """
    dep = max(tmax_anomaly, 0.0)
    anchors = [(0.0, 0.0), (1.5, 35.0), (2.5, 62.0), (3.5, 80.0), (6.0, 100.0)]
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    return float(np.interp(dep, xs, ys))


def _absolute_score(tmax: float) -> float:
    """
    Score raw Tmax against pan-India absolute heat thresholds (°C).
      < 34°C -> 0
      38°C   -> 40
      42°C   -> 65
      45°C   -> 85
      >= 48°C -> 100
    """
    anchors = [(34.0, 0.0), (38.0, 40.0), (42.0, 65.0), (45.0, 85.0), (48.0, 100.0)]
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    score = float(np.interp(tmax, xs, ys))
    return float(np.clip(score, 0, 100))


def _persistence_score(consecutive_hot_days: int) -> float:
    """
    Score heatwave persistence (consecutive days above local 90th percentile).
      0 days -> 0
      2 days -> 40
      4 days -> 70
      6 days -> 90
      >= 8 days -> 100
    """
    anchors = [(0, 0.0), (1, 20.0), (2, 40.0), (4, 70.0), (6, 90.0), (8, 100.0)]
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    return float(np.interp(consecutive_hot_days, xs, ys))


def compute_heat_risk(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute heat stress score and tier from feature vector.
    """
    tmax              = features.get("tmax", 30.0)
    tmax_anomaly      = features.get("tmax_anomaly", 0.0)
    consecutive_hot   = features.get("consecutive_hot_days", 0)

    s_departure   = _departure_score(tmax_anomaly)
    s_absolute    = _absolute_score(tmax)
    s_persistence = _persistence_score(consecutive_hot)

    heat_score = round(0.40 * s_departure + 0.30 * s_absolute + 0.30 * s_persistence, 2)
    heat_tier  = score_to_tier(heat_score)

    return {
        "heat_score": heat_score,
        "heat_tier": heat_tier,
        "heat_factors": {
            "tmax_celsius": tmax,
            "tmax_anomaly_celsius": tmax_anomaly,
            "consecutive_hot_days": consecutive_hot,
            "baseline_tmax_monthly_mean": features.get("baseline_tmax_monthly_mean"),
            "baseline_tmax_p90": features.get("baseline_tmax_p90"),
            "sub_score_departure": round(s_departure, 2),
            "sub_score_absolute": round(s_absolute, 2),
            "sub_score_persistence": round(s_persistence, 2),
            "weights": "40% departure + 30% absolute + 30% persistence",
        }
    }
