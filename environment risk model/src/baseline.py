"""
AgriNode AI - Environmental Risk Intelligence
Phase 3: Baseline Calibration Engine

Computes monthly historical climatology baselines (1981-2015) for every district:
- Precipitation: mean, std, 90th, 95th, 99th percentiles, and 30d/90d baseline normals
- Temperature: Tmax mean, std, 90th percentile; Tmin mean, std; Tmean mean

Saves pre-computed parameters to data/processed/district_baselines.csv.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    get_district_metadata_map,
    clean_and_harmonize_district_data,
    CALIBRATION_START,
    CALIBRATION_END,
    RAW_DISTRICTS_DIR
)

BASELINES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "district_baselines.csv"


class BaselineCalibrator:
    """Manages access to pre-computed district climatological baselines."""

    def __init__(self, baselines_csv: Optional[Path] = None):
        self.path = baselines_csv or BASELINES_CSV_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"Baselines file not found at {self.path}. Run compute_and_save_all_baselines() first.")
        self.df = pd.read_csv(self.path)
        # Fast lookup indexed by (district_id, month)
        self.lookup = self.df.set_index(["district_id", "month"]).to_dict("index")

    def get_baseline(self, district_id: int, month: int) -> dict:
        """Fetch baseline parameters for a specific district and calendar month."""
        key = (int(district_id), int(month))
        if key not in self.lookup:
            raise KeyError(f"No baseline found for district_id={district_id}, month={month}")
        return self.lookup[key]


def compute_district_baseline(district_id: int, meta: dict) -> pd.DataFrame:
    """
    Compute calibration monthly baseline statistics for a single district (1981-2015).
    """
    df = clean_and_harmonize_district_data(district_id, meta)
    df["date"] = pd.to_datetime(df["date"])
    
    # Strictly filter calibration period
    cal_mask = (df["date"] >= CALIBRATION_START) & (df["date"] <= CALIBRATION_END)
    cal_df = df[cal_mask].copy()
    cal_df["r30"] = cal_df["rainfall"].rolling(30, min_periods=15).sum()
    cal_df["r90"] = cal_df["rainfall"].rolling(90, min_periods=45).sum()

    records = []
    for month in range(1, 13):
        m_df = cal_df[cal_df["month"] == month]
        rain_vals = m_df["rainfall"].values
        tmax_vals = m_df["tmax"].values
        tmin_vals = m_df["tmin"].values
        tmean_vals = m_df["tmean"].values

        # Daily rainfall percentiles (thresholds for extreme rain / flash flood proxy)
        rain_mean = float(np.mean(rain_vals))
        rain_std = float(np.std(rain_vals))
        rain_p90 = float(np.percentile(rain_vals, 90))
        rain_p95 = float(np.percentile(rain_vals, 95))
        rain_p99 = float(np.percentile(rain_vals, 99))

        # Expected 30-day and 90-day seasonal rainfall sums ending in this month
        # Calculated from true empirical 30d and 90d rolling sums across calibration history
        sub_r30 = cal_df.loc[cal_df["month"] == month, "r30"].dropna()
        sub_r90 = cal_df.loc[cal_df["month"] == month, "r90"].dropna()
        baseline_30d_normal = round(float(sub_r30.mean()), 2) if len(sub_r30) > 0 else round(rain_mean * 30.0, 2)
        baseline_90d_normal = round(float(sub_r90.mean()), 2) if len(sub_r90) > 0 else round(rain_mean * 90.0, 2)

        # Thermal normals
        tmax_mean = float(np.mean(tmax_vals))
        tmax_std = float(np.std(tmax_vals))
        tmax_p90 = float(np.percentile(tmax_vals, 90))

        tmin_mean = float(np.mean(tmin_vals))
        tmin_std = float(np.std(tmin_vals))

        tmean_mean = float(np.mean(tmean_vals))

        records.append({
            "district_id": district_id,
            "district": meta.get("district", f"District_{district_id}"),
            "state": meta.get("state", "Unknown"),
            "month": month,
            "rain_mean": round(rain_mean, 2),
            "rain_std": round(rain_std, 2),
            "rain_p90": round(rain_p90, 2),
            "rain_p95": round(rain_p95, 2),
            "rain_p99": round(rain_p99, 2),
            "baseline_30d_normal": baseline_30d_normal,
            "baseline_90d_normal": baseline_90d_normal,
            "tmax_mean": round(tmax_mean, 2),
            "tmax_std": round(tmax_std, 2),
            "tmax_p90": round(tmax_p90, 2),
            "tmin_mean": round(tmin_mean, 2),
            "tmin_std": round(tmin_std, 2),
            "tmean_mean": round(tmean_mean, 2)
        })

    return pd.DataFrame(records)


def compute_and_save_all_baselines(max_districts: Optional[int] = None) -> Path:
    """
    Compute and save baseline catalog for all districts present in raw data.
    """
    print("=" * 70)
    print("AGRINODE AI - PHASE 3: HISTORICAL BASELINE CALIBRATION (1981-2015)")
    print("=" * 70)

    meta_map = get_district_metadata_map()
    
    # Discover available district IDs in raw directory
    raw_files = list(RAW_DISTRICTS_DIR.glob("data_District_*.csv"))
    available_ids = sorted([
        int(f.stem.replace("data_District_", ""))
        for f in raw_files
        if f.stem.replace("data_District_", "").isdigit()
    ])

    if max_districts:
        available_ids = available_ids[:max_districts]

    print(f"[CALIBRATION] Computing 35-year monthly baselines across {len(available_ids)} districts...")

    all_baseline_rows = []
    success_count = 0
    for idx, dist_id in enumerate(available_ids):
        try:
            meta = meta_map.get(dist_id, {"district": f"District_{dist_id}", "state": "Unknown"})
            bdf = compute_district_baseline(dist_id, meta)
            all_baseline_rows.append(bdf)
            success_count += 1
            if (idx + 1) % 100 == 0 or (idx + 1) == len(available_ids):
                print(f"  Processed {idx + 1}/{len(available_ids)} districts ({meta.get('district')})...")
        except Exception as e:
            print(f"  [ERROR] District {dist_id}: {e}")

    full_baselines_df = pd.concat(all_baseline_rows, ignore_index=True)
    BASELINES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_baselines_df.to_csv(BASELINES_CSV_PATH, index=False)

    print(f"\n[COMPLETE] Successfully compiled and exported baselines to {BASELINES_CSV_PATH}")
    print(f"  Total District-Month records: {len(full_baselines_df):,}")
    print(f"  Unique Districts calibrated : {success_count}")
    return BASELINES_CSV_PATH


_CALIBRATOR_INSTANCE: Optional[BaselineCalibrator] = None

def get_baseline_calibrator() -> BaselineCalibrator:
    """Singleton getter for cached BaselineCalibrator."""
    global _CALIBRATOR_INSTANCE
    if _CALIBRATOR_INSTANCE is None:
        _CALIBRATOR_INSTANCE = BaselineCalibrator()
    return _CALIBRATOR_INSTANCE


if __name__ == "__main__":
    compute_and_save_all_baselines()
