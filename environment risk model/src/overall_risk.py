"""
AgriNode AI - Environmental Risk Intelligence
Phase 4: Multi-Hazard Synthesis Engine (src/overall_risk.py)

Aggregates Drought, Flood (Proxy) and Heat Stress into a single overall
environmental risk level using SEVERITY DOMINANCE logic (as specified in
Phase_Wise_Plan.md §4):

    if any hazard == SEVERE  →  overall = SEVERE
    elif ≥2 hazards == HIGH, OR any == HIGH  →  overall = HIGH
    elif any hazard == MODERATE  →  overall = MODERATE
    else  →  overall = LOW

Returns a comprehensive risk report dict ready for Phase 5 (Decision Engine)
and Phase 6 (Benchmarking).
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional

from src.drought import compute_drought_risk
from src.flood   import compute_flood_risk
from src.heat    import compute_heat_risk


# ── Tier ordering (for human-readable comparison) ─────────────────────────────
TIER_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "SEVERE": 3}


def _dominant_tier(tiers: list[str]) -> str:
    """
    Severity dominance aggregation:
    - Any SEVERE  → SEVERE
    - 2+ HIGH OR any HIGH → HIGH
    - Any MODERATE → MODERATE
    - Otherwise  → LOW
    """
    if any(t == "SEVERE" for t in tiers):
        return "SEVERE"
    high_count = sum(1 for t in tiers if t == "HIGH")
    if high_count >= 2 or any(t == "HIGH" for t in tiers):
        return "HIGH"
    if any(t == "MODERATE" for t in tiers):
        return "MODERATE"
    return "LOW"


# ── Main synthesis function ────────────────────────────────────────────────────

def compute_overall_risk(
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all three hazard engines on a feature vector and synthesize them
    into a multi-hazard risk report.

    Parameters
    ----------
    features : dict
        Output of `src.features.FeatureExtractor.extract_features()`.

    Returns
    -------
    dict with keys:
      - overall_tier       (str)
      - overall_score      (float, average of three hazard scores)
      - primary_hazard     (str, the single highest-scoring hazard label)
      - drought            (full drought engine result dict)
      - flood              (full flood engine result dict)
      - heat               (full heat engine result dict)
      - spatial_context    (district, state, date, distance_km)
      - inference_latency_ms (float, total engine inference time)
    """
    t_start = time.perf_counter()

    drought_result = compute_drought_risk(features)
    flood_result   = compute_flood_risk(features)
    heat_result    = compute_heat_risk(features)

    tiers = [
        drought_result["drought_tier"],
        flood_result["flood_tier"],
        heat_result["heat_tier"],
    ]
    scores = [
        drought_result["drought_score"],
        flood_result["flood_score"],
        heat_result["heat_score"],
    ]

    overall_tier  = _dominant_tier(tiers)
    overall_score = round(sum(scores) / 3.0, 2)

    # Identify the single dominant hazard (highest score)
    hazard_names = ["DROUGHT", "FLOOD", "HEAT"]
    primary_hazard = hazard_names[int(max(range(3), key=lambda i: scores[i]))]

    inference_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

    return {
        "overall_tier": overall_tier,
        "overall_score": overall_score,
        "primary_hazard": primary_hazard,
        "drought": drought_result,
        "flood": flood_result,
        "heat": heat_result,
        "spatial_context": {
            "district": features.get("district"),
            "state": features.get("state"),
            "date": features.get("date"),
            "distance_km": features.get("distance_km"),
        },
        "inference_latency_ms": inference_latency_ms,
    }


def format_risk_report(report: Dict[str, Any], verbose: bool = True) -> str:
    """
    Render a human-readable ASCII risk summary for CLI display.
    """
    tier_colors = {
        "LOW":      "[LOW]",
        "MODERATE": "[MODERATE]",
        "HIGH":     "[HIGH]",
        "SEVERE":   "[SEVERE]",
    }

    ctx = report["spatial_context"]
    lines = [
        "=" * 68,
        f"  AgriNode AI — Environmental Risk Assessment",
        "=" * 68,
        f"  District : {ctx['district']}, {ctx['state']}",
        f"  Date     : {ctx['date']}",
        f"  Latency  : {report['inference_latency_ms']:.2f} ms (engine inference)",
        "-" * 68,
        f"  OVERALL RISK : {tier_colors.get(report['overall_tier'], report['overall_tier'])}  "
        f"(score: {report['overall_score']:.1f}/100)",
        f"  Primary Hazard: {report['primary_hazard']}",
        "-" * 68,
    ]

    if verbose:
        for key, label in [("drought", "Drought"), ("flood", "Flood Proxy"), ("heat", "Heat Stress")]:
            r   = report[key]
            s_key = f"{key}_score"
            t_key = f"{key}_tier"
            lines.append(
                f"  {label:<14}: {tier_colors.get(r[t_key], r[t_key]):<24}  score={r[s_key]:.1f}/100"
            )

    lines.append("=" * 68)
    return "\n".join(lines)
