"""
AgriNode AI - Environmental Risk Intelligence
Phase 7: Final Interactive Judge-Ready Demo (src/demo_predict.py)

Single Responsibility:
    Provide a polished, judge-ready CLI demonstration tool built directly on
    the benchmarked and validated pipeline.

Modes:
    1. --demo: Automated preset case studies (ideal for pitch presentation)
    2. --lat --lon --date: Specific coordinate & date testing
    3. Interactive CLI prompting (default if no args)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import get_feature_extractor
from src.overall_risk import compute_overall_risk
from src.decision_engine import generate_decision
from src.geo_resolver import get_geo_resolver

# Preset cases for --demo
DEMO_CASES = [
    {
        "name": "Marathwada 2015 Severe Drought",
        "lat": 18.4088,
        "lon": 76.5604,
        "date": "2015-08-30",
    },
    {
        "name": "Kerala 2018 Historic Deluge",
        "lat": 9.9816,
        "lon": 76.2999,
        "date": "2018-08-16",
    },
    {
        "name": "Phalodi 2016 National Heat Record",
        "lat": 27.1300,
        "lon": 72.3600,
        "date": "2016-05-19",
    },
    {
        "name": "Calm Baseline Control (Pune)",
        "lat": 18.5204,
        "lon": 73.8567,
        "date": "2022-10-31",
    }
]

def generate_explainability(risk_report: Dict[str, Any]) -> List[str]:
    """Generate human-readable explanations for the risk scores."""
    reasons = []
    
    # Drought
    df = risk_report["drought"]["drought_factors"]
    if risk_report["drought"]["drought_tier"] in ("HIGH", "SEVERE"):
        if df.get("anomaly_30d_pct", 0) < -30:
            reasons.append(f"30-day cumulative rainfall is {abs(df['anomaly_30d_pct'])}% below local historical monthly normal.")
        if df.get("anomaly_90d_pct", 0) < -30:
            reasons.append(f"90-day cumulative rainfall is {abs(df['anomaly_90d_pct'])}% below local historical normal.")
        if df.get("current_dry_spell_days", 0) >= 10:
            reasons.append(f"Extended dry spell detected ({df['current_dry_spell_days']} consecutive days with rainfall < 2.5mm).")
            
    # Flood
    ff = risk_report["flood"]["flood_factors"]
    if risk_report["flood"]["flood_tier"] in ("HIGH", "SEVERE"):
        if ff.get("rain_1d_mm", 0) > ff.get("baseline_rain_p95", 999):
            reasons.append(f"Acute 1-day rainfall ({ff['rain_1d_mm']:.1f} mm) exceeds the 95th percentile baseline.")
        if ff.get("heavy_rain_days_7d", 0) >= 3:
            reasons.append(f"High frequency of heavy rain: {ff['heavy_rain_days_7d']} extreme rain days in the past week.")
            
    # Heat
    hf = risk_report["heat"]["heat_factors"]
    if risk_report["heat"]["heat_tier"] in ("HIGH", "SEVERE"):
        if hf.get("tmax_anomaly_celsius", 0) >= 2.0:
            reasons.append(f"Current Tmax ({hf['tmax_celsius']} C) is {hf['tmax_anomaly_celsius']:.1f} C above historical monthly baseline.")
        if hf.get("consecutive_hot_days", 0) >= 3:
            reasons.append(f"Prolonged heatwave: {hf['consecutive_hot_days']} consecutive days above the 90th percentile threshold.")

    if not reasons:
        reasons.append("Environmental indicators are within normal seasonal bounds.")
        
    return reasons

def print_presentation_report(lat: float, lon: float, date_str: str, title: str = ""):
    """Print the final formatted judge-ready CLI output."""
    extractor = get_feature_extractor()
    resolver = get_geo_resolver()
    
    t0 = time.perf_counter()
    # Resolve first for UI display
    resolved = resolver.resolve(lat, lon)
    
    # Extract & Compute
    features = extractor.extract_features(lat=lat, lon=lon, query_date=date_str)
    risk_report = compute_overall_risk(features)
    decision = generate_decision(risk_report)
    
    latency = (time.perf_counter() - t0) * 1000.0
    
    # Extract fields
    district = resolved["district"]
    state = resolved["state"]
    overall_tier = risk_report["overall_tier"]
    overall_score = risk_report["overall_score"]
    
    # Build Output lines
    lines = []
    lines.append("=" * 80)
    lines.append("                    AGRINODE AI -- ENVIRONMENTAL RISK MODEL")
    if title:
        lines.append(f"                    Case Study: {title}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(" [LOCATION DETAILS]")
    lines.append(f"    Nearest District : {district}, {state}")
    lines.append(f"    Coordinates      : Lat {lat:.4f}, Lon {lon:.4f}")
    lines.append(f"    Evaluation Date  : {date_str}")
    lines.append(f"    Inference Time   : {latency:.1f} ms")
    lines.append("")
    lines.append("-" * 80)
    lines.append(" [MULTI-HAZARD RISK ASSESSMENT]")
    lines.append(f"    * Drought Risk       : {risk_report['drought']['drought_tier']:<10} [Score: {risk_report['drought']['drought_score']:>4.1f} / 100]")
    lines.append(f"    * Flood Risk Proxy   : {risk_report['flood']['flood_tier']:<10} [Score: {risk_report['flood']['flood_score']:>4.1f} / 100]")
    lines.append(f"    * Heat Stress        : {risk_report['heat']['heat_tier']:<10} [Score: {risk_report['heat']['heat_score']:>4.1f} / 100]")
    lines.append("")
    lines.append(f"    >> OVERALL ENVIRONMENTAL RISK: {overall_tier} [Score: {overall_score:>4.1f} / 100]")
    lines.append("")
    lines.append("-" * 80)
    lines.append(" [TRANSPARENT EXPLAINABILITY (\"WHY?\")]")
    
    reasons = generate_explainability(risk_report)
    for i, reason in enumerate(reasons, 1):
        lines.append(f"    {i}. {reason}")
        
    lines.append("")
    lines.append("-" * 80)
    lines.append(" [OPERATIONAL FARM DECISIONS (FIELD ORDERS)]")
    
    if overall_tier == "LOW":
        lines.append("    [ADVISORY]         : Continue routine monitoring and scheduled maintenance.")
    else:
        unified_actions = decision.get("unified_action_plan", [])
        for action in unified_actions:
            priority = action["priority"]
            cat = action["category"]
            haz = action["hazard"]
            txt = action["action"]
            
            # Format nicely
            label = f"[{cat}]"
            if priority == "CRITICAL":
                label = f"[{cat}-CRITICAL]"
                
            lines.append(f"    {label:<18} : {txt}")
            
    lines.append("=" * 80)
    
    # Print the report
    print("\n".join(lines))

def main():
    parser = argparse.ArgumentParser(description="AgriNode AI Final Judge-Ready Demo")
    parser.add_argument("--demo", action="store_true", help="Run automated preset case studies")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--date", type=str, help="Evaluation Date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    if args.demo:
        print("\nStarting Automated Pitch Demo...")
        for case in DEMO_CASES:
            time.sleep(1) # Add a small pause for dramatic effect between cases
            print_presentation_report(case["lat"], case["lon"], case["date"], title=case["name"])
            print("\n")
    elif args.lat is not None and args.lon is not None and args.date is not None:
        print_presentation_report(args.lat, args.lon, args.date)
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("  AgriNode AI - Interactive Mode")
        print("=" * 60)
        try:
            lat_in = input("Enter Latitude (e.g. 18.4088): ")
            lon_in = input("Enter Longitude (e.g. 76.5604): ")
            date_in = input("Enter Date (YYYY-MM-DD): ")
            
            lat = float(lat_in.strip())
            lon = float(lon_in.strip())
            date_str = date_in.strip()
            
            print("\nEvaluating risk models...\n")
            print_presentation_report(lat, lon, date_str)
        except KeyboardInterrupt:
            print("\nExiting.")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
