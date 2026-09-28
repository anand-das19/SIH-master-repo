"""
AgriNode AI - Environmental Risk Intelligence
Phase 4: Drought Risk Engine (src/drought.py)

Scoring Formula (continuous 0–100):
  drought_score = 40% * long_term_deficit_score  (90-day rainfall anomaly)
                + 30% * medium_term_deficit_score (30-day rainfall anomaly)
                + 30% * dry_spell_score           (consecutive dry days)

Standardized Risk Tiers:
  0–30   → LOW
  31–60  → MODERATE
  61–80  → HIGH
  81–100 → SEVERE
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


# Standardized 4-tier boundaries (per Phase_Wise_Plan.md §4)
TIER_BOUNDARIES = [(81, "SEVERE"), (61, "HIGH"), (31, "MODERATE"), (0, "LOW")]


def score_to_tier(score: float) -> str:
    """Map a continuous 0–100 score to a standardized risk tier label."""
    for threshold, label in TIER_BOUNDARIES:
        if score >= threshold:
            return label
    return "LOW"


def _long_term_deficit_score(anomaly_90d: float) -> float:
    """
    Score 90-day rainfall anomaly (%).
    Negative anomalies indicate deficit:
      0% deficit   -> 0
      -15% deficit -> 48
      -30% deficit -> 72 (HIGH range)
      -50% deficit -> 88 (SEVERE range)
      <=-75% deficit -> 100
    """
    deficit = -min(anomaly_90d, 0.0)
    anchors = [(0.0, 0.0), (15.0, 48.0), (30.0, 72.0), (50.0, 88.0), (75.0, 100.0)]
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    return float(np.interp(deficit, xs, ys))


def _medium_term_deficit_score(anomaly_30d: float) -> float:
    """
    Score 30-day rainfall anomaly (%).
    Negative anomalies indicate acute monthly deficit:
      0% deficit   -> 0
      -15% deficit -> 48
      -30% deficit -> 75 (HIGH range)
      -50% deficit -> 90 (SEVERE range)
      <=-75% deficit -> 100
    """
    deficit = -min(anomaly_30d, 0.0)
    anchors = [(0.0, 0.0), (15.0, 48.0), (30.0, 75.0), (50.0, 90.0), (75.0, 100.0)]
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    return float(np.interp(deficit, xs, ys))


def _dry_spell_score(current_dry_spell: int) -> float:
    """
    Score consecutive dry-day count (< 2.5 mm).
      0-4 days   -> 10
      7 days     -> 30
      14 days    -> 60 (significant agricultural break)
      21 days    -> 80 (severe moisture stress)
      >= 30 days -> 100
    """
    anchors = [(0, 0.0), (4, 15.0), (7, 35.0), (14, 65.0), (21, 85.0), (30, 100.0)]
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    return float(np.interp(current_dry_spell, xs, ys))


def compute_drought_risk(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute drought risk score and tier from extracted feature vector.
    """
    anomaly_90d = features.get("anomaly_90d", 0.0)
    anomaly_30d = features.get("anomaly_30d", 0.0)
    dry_spell   = features.get("current_dry_spell", 0)

    # Seasonal modulation:
    # In dry/pre-monsoon months (baseline normal < 20 mm across 30 days),
    # zero precipitation is natural dry season, not a drought crisis.
    baseline_30d = features.get("baseline_rain_monthly_mean", 5.0) * 30.0
    if baseline_30d < 20.0:
        seasonal_weight = 0.15
    elif baseline_30d < 80.0:
        seasonal_weight = 0.15 + 0.85 * ((baseline_30d - 20.0) / 60.0)
    else:
        seasonal_weight = 1.0

    s_long   = _long_term_deficit_score(anomaly_90d) * seasonal_weight
    s_medium = _medium_term_deficit_score(anomaly_30d) * seasonal_weight
    s_dry    = _dry_spell_score(dry_spell) * max(seasonal_weight, 0.25)

    # 40% long-term + 30% medium-term + 30% dry-spell
    drought_score = round(0.40 * s_long + 0.30 * s_medium + 0.30 * s_dry, 2)
    drought_tier  = score_to_tier(drought_score)

    return {
        "drought_score": drought_score,
        "drought_tier": drought_tier,
        "drought_factors": {
            "anomaly_90d_pct": anomaly_90d,
            "anomaly_30d_pct": anomaly_30d,
            "current_dry_spell_days": dry_spell,
            "seasonal_weight": round(seasonal_weight, 3),
            "sub_score_long_term_deficit": round(s_long, 2),
            "sub_score_medium_term_deficit": round(s_medium, 2),
            "sub_score_dry_spell": round(s_dry, 2),
            "weights": "40% long-term + 30% medium-term + 30% dry-spell",
        }
    }
